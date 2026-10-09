#![forbid(unsafe_code)]

mod factorized;

pub use factorized::reference as factorized_reference;
pub use factorized::{
    FactorizedPackTrainingStats, FactorizedTrainingConfig, FactorizedTrainingReport,
    train_factorized_bpe, train_factorized_bpe_with_provenance,
    train_factorized_bpe_with_provenance_report, train_factorized_bpe_with_report,
};

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use packtok_core::{
    BYTE_FALLBACK_TOKEN_COUNT, ByteFallback, DEFAULT_BYTE_FALLBACK_PACK_ID, PackDescriptor,
    PackRegistry,
};
use packtok_format::{
    Artifact, BpeModelError, FLAT_BPE_PACK_ID, FactorizedBpeModelError, FlatBpeMerge, FlatBpeModel,
    FormatError, MAX_BPE_TOKEN_BYTES,
};

/// M1's documented default training configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BpeTrainingConfig {
    /// Total vocabulary target including the initial 256 byte tokens.
    pub target_vocab_size: u32,
    /// Upper bound on added merge tokens.
    pub max_merges: u32,
    /// Minimum overlapping adjacent-pair count required to select a merge.
    pub min_pair_frequency: u64,
}

impl Default for BpeTrainingConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 512,
            max_merges: 256,
            min_pair_frequency: 2,
        }
    }
}

impl BpeTrainingConfig {
    /// Rejects targets that cannot contain byte symbols or merge limits beyond the target.
    pub fn validate(self) -> Result<Self, TrainingError> {
        if self.target_vocab_size < BYTE_FALLBACK_TOKEN_COUNT {
            return Err(TrainingError::InvalidConfig {
                field: "target_vocab_size",
                reason: "must be at least 256 to retain every raw-byte token",
            });
        }
        if self.max_merges > self.target_vocab_size - BYTE_FALLBACK_TOKEN_COUNT {
            return Err(TrainingError::InvalidConfig {
                field: "max_merges",
                reason: "cannot exceed target_vocab_size minus 256 byte tokens",
            });
        }
        if self.min_pair_frequency == 0 {
            return Err(TrainingError::InvalidConfig {
                field: "min_pair_frequency",
                reason: "must be greater than zero",
            });
        }
        Ok(self)
    }
}

/// Corpus metadata recorded descriptively in the trained artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorpusProvenance {
    /// User-supplied path or `memory` for an in-memory corpus.
    pub input: String,
    /// Files in the exact concatenation order; directory paths are root-relative.
    pub files: Vec<String>,
    /// Total raw bytes after concatenating files without separators.
    pub total_bytes: u64,
    /// FNV-1a 64-bit checksum over the concatenated raw bytes; not cryptographic.
    pub fnv1a64: u64,
}

/// Raw corpus bytes and descriptive provenance produced by [`load_corpus`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorpusData {
    bytes: Vec<u8>,
    provenance: CorpusProvenance,
}

impl CorpusData {
    /// Returns the exact concatenated bytes used for training.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns deterministic file order and raw-input metadata.
    #[must_use]
    pub fn provenance(&self) -> &CorpusProvenance {
        &self.provenance
    }
}

/// Loads one raw file or a directory tree in deterministic order.
///
/// Directories include regular files recursively, ordered by root-relative Unicode
/// path using `/` separators. Symlinks and non-Unicode paths are rejected. Files are
/// concatenated as raw bytes with no inserted separators or UTF-8 validation.
pub fn load_corpus(path: &Path) -> Result<CorpusData, CorpusError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| CorpusError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() {
        return Err(CorpusError::SymlinkNotSupported(path.to_path_buf()));
    }

    let input = path
        .to_str()
        .ok_or_else(|| CorpusError::NonUnicodePath(path.to_path_buf()))?
        .to_owned();
    let mut files = Vec::<(String, PathBuf)>::new();
    if metadata.is_file() {
        files.push((input.clone(), path.to_path_buf()));
    } else if metadata.is_dir() {
        collect_directory_files(path, path, &mut files)?;
        files.sort_by(|left, right| left.0.cmp(&right.0));
    } else {
        return Err(CorpusError::NotFileOrDirectory(path.to_path_buf()));
    }
    if files.len() > MAX_CORPUS_FILES {
        return Err(CorpusError::TooManyFiles {
            count: files.len(),
            maximum: MAX_CORPUS_FILES,
        });
    }

    let mut bytes = Vec::new();
    let mut names = Vec::new();
    names
        .try_reserve_exact(files.len())
        .map_err(|_| CorpusError::AllocationFailed)?;
    for (name, file_path) in files {
        let mut file = fs::File::open(&file_path).map_err(|source| CorpusError::Io {
            path: file_path.clone(),
            source,
        })?;
        // Avoid keeping a second whole-file buffer alongside the corpus.
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = match file.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(source) => {
                    return Err(CorpusError::Io {
                        path: file_path,
                        source,
                    });
                }
            };
            bytes
                .try_reserve(count)
                .map_err(|_| CorpusError::AllocationFailed)?;
            bytes.extend_from_slice(&buffer[..count]);
        }
        names.push(name);
    }
    let total_bytes = u64::try_from(bytes.len()).map_err(|_| CorpusError::LengthOverflow)?;
    let fnv1a64 = fnv1a64(&bytes);
    Ok(CorpusData {
        bytes,
        provenance: CorpusProvenance {
            input,
            files: names,
            total_bytes,
            fnv1a64,
        },
    })
}

/// Stable, non-cryptographic checksum for reproducible corpus identification.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Trains the simple production trainer and returns a self-contained v2 artifact.
pub fn train_bpe(corpus: &[u8], config: BpeTrainingConfig) -> Result<Artifact, TrainingError> {
    let total_bytes = u64::try_from(corpus.len()).map_err(|_| TrainingError::LengthOverflow)?;
    let provenance = CorpusProvenance {
        input: "memory".to_owned(),
        files: Vec::new(),
        total_bytes,
        fnv1a64: fnv1a64(corpus),
    };
    train_bpe_with_provenance(corpus, config, &provenance)
}

/// Trains BPE while recording explicit raw-corpus provenance.
pub fn train_bpe_with_provenance(
    corpus: &[u8],
    config: BpeTrainingConfig,
    provenance: &CorpusProvenance,
) -> Result<Artifact, TrainingError> {
    let config = config.validate()?;
    if provenance.files.len() > MAX_CORPUS_FILES {
        return Err(TrainingError::TooManyProvenanceFiles(
            provenance.files.len(),
        ));
    }
    let expected_bytes = u64::try_from(corpus.len()).map_err(|_| TrainingError::LengthOverflow)?;
    if provenance.total_bytes != expected_bytes || provenance.fnv1a64 != fnv1a64(corpus) {
        return Err(TrainingError::InvalidProvenance);
    }
    let model = train_model(corpus, config)?;
    let registry = PackRegistry::new(
        vec![
            PackDescriptor::new(FLAT_BPE_PACK_ID, "FLAT_BPE", model.vocabulary_size()),
            PackDescriptor::new(
                DEFAULT_BYTE_FALLBACK_PACK_ID,
                "BYTE_FALLBACK",
                BYTE_FALLBACK_TOKEN_COUNT,
            ),
        ],
        ByteFallback::new(DEFAULT_BYTE_FALLBACK_PACK_ID),
        Vec::new(),
    )?;
    let metadata = artifact_metadata(config, &model, provenance)?;
    Ok(Artifact::with_flat_bpe(registry, metadata, model)?)
}

/// Trains a flat BPE model with ordered pair counting and deterministic ties.
pub fn train_model(
    corpus: &[u8],
    config: BpeTrainingConfig,
) -> Result<FlatBpeModel, TrainingError> {
    let config = config.validate()?;
    if config.max_merges == 0 || corpus.len() < 2 {
        return Ok(FlatBpeModel::new(Vec::new())?);
    }
    let mut symbols = Vec::new();
    symbols
        .try_reserve_exact(corpus.len())
        .map_err(|_| TrainingError::AllocationFailed)?;
    symbols.extend(corpus.iter().map(|byte| u32::from(*byte)));
    let mut merges = Vec::new();
    let merge_limit = config
        .max_merges
        .min(config.target_vocab_size - BYTE_FALLBACK_TOKEN_COUNT);
    let maximum_corpus_merges = corpus.len().saturating_sub(1);
    let reserve_count = usize::try_from(merge_limit)
        .map_err(|_| TrainingError::LengthOverflow)?
        .min(maximum_corpus_merges);
    merges
        .try_reserve(reserve_count)
        .map_err(|_| TrainingError::AllocationFailed)?;
    let mut token_lengths = Vec::new();
    token_lengths
        .try_reserve_exact(BYTE_FALLBACK_TOKEN_COUNT as usize + reserve_count)
        .map_err(|_| TrainingError::AllocationFailed)?;
    token_lengths.extend(std::iter::repeat_n(
        1_usize,
        BYTE_FALLBACK_TOKEN_COUNT as usize,
    ));

    for _ in 0..merge_limit {
        let mut frequencies = BTreeMap::<(u32, u32), u64>::new();
        for pair in symbols.windows(2) {
            let count = frequencies.entry((pair[0], pair[1])).or_default();
            *count = count
                .checked_add(1)
                .ok_or(TrainingError::FrequencyOverflow)?;
        }
        let mut selected: Option<((u32, u32), u64)> = None;
        for (pair, frequency) in frequencies {
            let length = token_lengths[pair.0 as usize] + token_lengths[pair.1 as usize];
            if frequency < config.min_pair_frequency || length > MAX_BPE_TOKEN_BYTES {
                continue;
            }
            if selected.is_none_or(|(best_pair, best_frequency)| {
                frequency > best_frequency || (frequency == best_frequency && pair < best_pair)
            }) {
                selected = Some((pair, frequency));
            }
        }
        let Some(((left, right), _frequency)) = selected else {
            break;
        };
        let result = BYTE_FALLBACK_TOKEN_COUNT
            .checked_add(u32::try_from(merges.len()).map_err(|_| TrainingError::LengthOverflow)?)
            .ok_or(TrainingError::LengthOverflow)?;
        merges.push(FlatBpeMerge {
            left,
            right,
            result,
        });
        token_lengths.push(token_lengths[left as usize] + token_lengths[right as usize]);
        merge_pair(&mut symbols, (left, right), result);
    }
    Ok(FlatBpeModel::new(merges)?)
}

fn merge_pair(symbols: &mut Vec<u32>, pair: (u32, u32), result: u32) {
    let mut index = 0;
    let mut written = 0;
    while index < symbols.len() {
        if index + 1 < symbols.len() && (symbols[index], symbols[index + 1]) == pair {
            symbols[written] = result;
            index += 2;
        } else {
            symbols[written] = symbols[index];
            index += 1;
        }
        written += 1;
    }
    symbols.truncate(written);
}

fn artifact_metadata(
    config: BpeTrainingConfig,
    model: &FlatBpeModel,
    provenance: &CorpusProvenance,
) -> Result<BTreeMap<String, String>, TrainingError> {
    if provenance.files.len() > MAX_CORPUS_FILES {
        return Err(TrainingError::TooManyProvenanceFiles(
            provenance.files.len(),
        ));
    }
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "bpe.algorithm".to_owned(),
        "flat-byte-level-bpe-v1".to_owned(),
    );
    metadata.insert(
        "bpe.byte_token_count".to_owned(),
        BYTE_FALLBACK_TOKEN_COUNT.to_string(),
    );
    metadata.insert(
        "bpe.encoder_order".to_owned(),
        "ascending merge rank; replace all non-overlapping matches left-to-right".to_owned(),
    );
    metadata.insert(
        "bpe.merge_count".to_owned(),
        model.merges().len().to_string(),
    );
    metadata.insert(
        "bpe.merge_id_order".to_owned(),
        "result token ID = 256 + zero-based merge rank".to_owned(),
    );
    metadata.insert("bpe.max_merges".to_owned(), config.max_merges.to_string());
    metadata.insert(
        "bpe.min_pair_frequency".to_owned(),
        config.min_pair_frequency.to_string(),
    );
    metadata.insert("bpe.normalization".to_owned(), "none".to_owned());
    metadata.insert(
        "bpe.target_vocab_size".to_owned(),
        config.target_vocab_size.to_string(),
    );
    metadata.insert(
        "bpe.tie_break".to_owned(),
        "highest frequency, then ascending (left_id, right_id)".to_owned(),
    );
    metadata.insert(
        "bpe.vocabulary_size".to_owned(),
        model.vocabulary_size().to_string(),
    );
    metadata.insert(
        "corpus.file_count".to_owned(),
        provenance.files.len().to_string(),
    );
    metadata.insert(
        "corpus.fnv1a64".to_owned(),
        format!("{:016x}", provenance.fnv1a64),
    );
    metadata.insert("corpus.input".to_owned(), provenance.input.clone());
    metadata.insert("corpus.preprocessing".to_owned(), "raw bytes; no UTF-8 validation or normalization; file contents concatenated without separators".to_owned());
    metadata.insert(
        "corpus.total_bytes".to_owned(),
        provenance.total_bytes.to_string(),
    );
    for (index, file) in provenance.files.iter().enumerate() {
        metadata.insert(format!("corpus.file.{index:06}"), file.clone());
    }
    Ok(metadata)
}

const MAX_CORPUS_FILES: usize = 65_000;

fn collect_directory_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), CorpusError> {
    let mut pending = vec![directory.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).map_err(|source| CorpusError::Io {
            path: directory.to_path_buf(),
            source,
        })? {
            let entry = entry.map_err(|source| CorpusError::Io {
                path: directory.to_path_buf(),
                source,
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|source| CorpusError::Io {
                path: path.clone(),
                source,
            })?;
            if file_type.is_symlink() {
                return Err(CorpusError::SymlinkNotSupported(path));
            }
            if path.to_str().is_none() {
                return Err(CorpusError::NonUnicodePath(path));
            }
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file() {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|_| CorpusError::InvalidRelativePath(path.clone()))?;
                let relative = relative
                    .components()
                    .map(|component| {
                        component
                            .as_os_str()
                            .to_str()
                            .ok_or_else(|| CorpusError::NonUnicodePath(path.clone()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("/");
                if files.len() == MAX_CORPUS_FILES {
                    return Err(CorpusError::TooManyFiles {
                        count: files.len() + 1,
                        maximum: MAX_CORPUS_FILES,
                    });
                }
                files.push((relative, path));
            } else {
                return Err(CorpusError::NotFileOrDirectory(path));
            }
        }
    }
    Ok(())
}

/// Detailed corpus-loading failure.
#[derive(Debug)]
pub enum CorpusError {
    /// An operating-system read or traversal operation failed.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Input was neither a regular file nor a directory.
    NotFileOrDirectory(PathBuf),
    /// Symlinks are rejected so traversal cannot escape the named corpus tree.
    SymlinkNotSupported(PathBuf),
    /// A corpus path is not valid Unicode and cannot be recorded portably.
    NonUnicodePath(PathBuf),
    /// A discovered path could not be represented relative to its corpus root.
    InvalidRelativePath(PathBuf),
    /// Too many source files would exceed the artifact metadata entry limit.
    TooManyFiles { count: usize, maximum: usize },
    /// The input size exceeded the host's supported byte count.
    LengthOverflow,
    /// The allocator could not reserve corpus storage.
    AllocationFailed,
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "cannot read corpus path {}: {source}", path.display())
            }
            Self::NotFileOrDirectory(path) => write!(
                f,
                "corpus input {} is not a regular file or directory",
                path.display()
            ),
            Self::SymlinkNotSupported(path) => {
                write!(f, "corpus symlink {} is not supported", path.display())
            }
            Self::NonUnicodePath(path) => {
                write!(f, "corpus path {} is not Unicode", path.display())
            }
            Self::InvalidRelativePath(path) => write!(
                f,
                "cannot make corpus file {} relative to its root",
                path.display()
            ),
            Self::TooManyFiles { count, maximum } => {
                write!(f, "corpus contains {count} files; maximum is {maximum}")
            }
            Self::LengthOverflow => f.write_str("corpus byte length exceeds supported limits"),
            Self::AllocationFailed => f.write_str("could not allocate corpus storage"),
        }
    }
}

impl std::error::Error for CorpusError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Error from configuration validation, training, provenance, or artifact construction.
#[derive(Debug)]
pub enum TrainingError {
    /// A configuration option is outside the supported range.
    InvalidConfig {
        field: &'static str,
        reason: &'static str,
    },
    /// A pair occurrence count overflowed `u64`.
    FrequencyOverflow,
    /// An ID, byte count, or capacity calculation overflowed.
    LengthOverflow,
    /// The allocator could not reserve training storage.
    AllocationFailed,
    /// Provenance exceeds the artifact's bounded metadata capacity.
    TooManyProvenanceFiles(usize),
    /// Raw byte count or checksum does not match the supplied corpus.
    InvalidProvenance,
    /// The M2 training input is not valid UTF-8 text.
    InvalidUtf8Corpus { valid_up_to: usize },
    /// A router emitted a pack that is not part of the selected M2 policy.
    InvalidRouting,
    /// A merge model failed structural validation.
    InvalidModel(BpeModelError),
    /// The model or registry could not be serialized as a valid artifact.
    InvalidArtifact(FormatError),
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => write!(f, "invalid {field}: {reason}"),
            Self::FrequencyOverflow => f.write_str("pair frequency exceeds the supported range"),
            Self::LengthOverflow => {
                f.write_str("training size or token ID exceeds supported limits")
            }
            Self::AllocationFailed => f.write_str("could not allocate training storage"),
            Self::TooManyProvenanceFiles(count) => {
                write!(f, "cannot record {count} corpus files in the artifact")
            }
            Self::InvalidProvenance => f.write_str(
                "corpus provenance byte count or checksum does not match the supplied input",
            ),
            Self::InvalidUtf8Corpus { valid_up_to } => write!(
                f,
                "factorized training corpus is not UTF-8 (valid prefix: {valid_up_to} bytes)"
            ),
            Self::InvalidRouting => {
                f.write_str("router emitted a span outside the lexical-v1 packs")
            }
            Self::InvalidModel(error) => write!(f, "invalid trained BPE model: {error}"),
            Self::InvalidArtifact(error) => write!(f, "cannot create BPE artifact: {error}"),
        }
    }
}

impl std::error::Error for TrainingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidModel(error) => Some(error),
            Self::InvalidArtifact(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BpeModelError> for TrainingError {
    fn from(value: BpeModelError) -> Self {
        Self::InvalidModel(value)
    }
}

impl From<FactorizedBpeModelError> for TrainingError {
    fn from(value: FactorizedBpeModelError) -> Self {
        Self::InvalidArtifact(FormatError::InvalidFactorizedBpeModel(value))
    }
}

impl From<FormatError> for TrainingError {
    fn from(value: FormatError) -> Self {
        Self::InvalidArtifact(value)
    }
}

impl From<packtok_core::ValidationError> for TrainingError {
    fn from(value: packtok_core::ValidationError) -> Self {
        Self::InvalidArtifact(FormatError::InvalidRegistry(value))
    }
}

/// Deliberately simple independent oracle used by trainer/runtime differential tests.
pub mod reference {
    use super::*;

    /// Counts pairs by linear search and replaces them with a standalone scalar loop.
    /// Intended for small fixtures and tests; its counting path is deliberately slow.
    pub fn train_model(
        corpus: &[u8],
        config: BpeTrainingConfig,
    ) -> Result<FlatBpeModel, TrainingError> {
        let config = config.validate()?;
        let mut symbols: Vec<u32> = corpus.iter().map(|byte| u32::from(*byte)).collect();
        let mut merges = Vec::new();
        let mut lengths = vec![1_usize; BYTE_FALLBACK_TOKEN_COUNT as usize];
        let merge_limit = config
            .max_merges
            .min(config.target_vocab_size - BYTE_FALLBACK_TOKEN_COUNT);

        for _ in 0..merge_limit {
            let mut counts = Vec::<((u32, u32), u64)>::new();
            for window in symbols.windows(2) {
                let pair = (window[0], window[1]);
                if let Some((_, count)) = counts.iter_mut().find(|(known, _)| *known == pair) {
                    *count = count
                        .checked_add(1)
                        .ok_or(TrainingError::FrequencyOverflow)?;
                } else {
                    counts.push((pair, 1));
                }
            }
            let selected = counts
                .into_iter()
                .filter(|(_, count)| *count >= config.min_pair_frequency)
                .filter(|((left, right), _)| {
                    lengths[*left as usize] + lengths[*right as usize] <= MAX_BPE_TOKEN_BYTES
                })
                .min_by(|(pair_a, count_a), (pair_b, count_b)| {
                    count_b.cmp(count_a).then_with(|| pair_a.cmp(pair_b))
                });
            let Some(((left, right), _)) = selected else {
                break;
            };
            let result = BYTE_FALLBACK_TOKEN_COUNT
                .checked_add(
                    u32::try_from(merges.len()).map_err(|_| TrainingError::LengthOverflow)?,
                )
                .ok_or(TrainingError::LengthOverflow)?;
            merges.push(FlatBpeMerge {
                left,
                right,
                result,
            });
            lengths.push(lengths[left as usize] + lengths[right as usize]);

            let mut next = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    next.push(result);
                    index += 2;
                } else {
                    next.push(symbols[index]);
                    index += 1;
                }
            }
            symbols = next;
        }
        Ok(FlatBpeModel::new(merges)?)
    }

    /// Applies stored merges in rank order to produce local IDs for a byte sequence.
    pub fn encode(input: &[u8], model: &FlatBpeModel) -> Vec<u32> {
        let mut symbols: Vec<u32> = input.iter().map(|byte| u32::from(*byte)).collect();
        for merge in model.merges() {
            let mut next = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == merge.left
                    && symbols[index + 1] == merge.right
                {
                    next.push(merge.result);
                    index += 2;
                } else {
                    next.push(symbols[index]);
                    index += 1;
                }
            }
            symbols = next;
        }
        symbols
    }
}

/// Builds tokens with runtime code and checks them against the independent oracle.
pub fn reference_encode(input: &[u8], model: &FlatBpeModel) -> Vec<u32> {
    reference::encode(input, model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_tokenizer::{BpeTokenizer, Tokenizer};
    use std::sync::atomic::{AtomicU64, Ordering};

    const CONFIG: BpeTrainingConfig = BpeTrainingConfig {
        target_vocab_size: 260,
        max_merges: 4,
        min_pair_frequency: 2,
    };

    #[test]
    fn training_skips_merges_above_the_token_expansion_bound() {
        let corpus = vec![b'a'; 3 * packtok_format::MAX_BPE_TOKEN_BYTES];
        let config = BpeTrainingConfig::default();
        let model = train_model(&corpus, config).expect("valid corpus must still train");
        assert_eq!(model.merges().len(), 20);
        assert_eq!(
            model.byte_length(275),
            Some(packtok_format::MAX_BPE_TOKEN_BYTES)
        );
        assert_eq!(
            reference::train_model(&corpus, config).expect("oracle"),
            model
        );
        let runtime = BpeTokenizer::new(model);
        let tokens = runtime.encode_bytes(&corpus).expect("encode full corpus");
        assert_eq!(tokens.len(), 3);
        assert_eq!(runtime.decode_bytes(&tokens).expect("decode"), corpus);
    }

    #[test]
    fn training_considers_other_pairs_after_a_dominant_pair_reaches_the_bound() {
        let mut corpus = vec![b'a'; 4 * MAX_BPE_TOKEN_BYTES];
        corpus.extend_from_slice(b"xyzxyz");
        let config = BpeTrainingConfig::default();
        let model = train_model(&corpus, config).expect("skip oversized pair, keep training");
        assert_eq!(model.merges().len(), 22);
        assert_eq!(
            model.merges()[20],
            FlatBpeMerge {
                left: 120,
                right: 121,
                result: 276
            }
        );
        assert_eq!(
            reference::train_model(&corpus, config).expect("oracle"),
            model
        );
        let runtime = BpeTokenizer::new(model);
        let tokens = runtime.encode_bytes(&corpus).expect("encode corpus");
        assert_eq!(tokens.len(), 6);
        assert_eq!(runtime.decode_bytes(&tokens).expect("decode"), corpus);
    }

    #[test]
    fn dense_nonuniform_runtime_paths_match_reference_and_round_trip() {
        for pattern in [b"a".as_slice(), b"ab", b"abc", b"01234567"] {
            let mut merges = train_model(&pattern.repeat(512), BpeTrainingConfig::default())
                .expect("repetitive model")
                .merges()
                .to_vec();
            for right in 0..64 {
                merges.push(FlatBpeMerge {
                    left: 0,
                    right,
                    result: 256 + merges.len() as u32,
                });
            }
            let model = FlatBpeModel::new(merges).expect("extended model");
            let runtime = BpeTokenizer::new(model.clone());
            let mut input = pattern.repeat(16_384_usize.div_ceil(pattern.len()));
            input.truncate(16_384);
            for index in [0, 1, 31, 8191, 8192, 16383] {
                let mut mixed = input.clone();
                mixed[index] = 0xff;
                let (tokens, stats) = runtime.encode_bytes_with_stats(&mixed).expect("encode");
                assert_eq!(
                    tokens.iter().map(|token| token.local).collect::<Vec<_>>(),
                    reference::encode(&mixed, &model)
                );
                assert_eq!(runtime.decode_bytes(&tokens).expect("decode"), mixed);
                assert_eq!(
                    stats.allocation_requests, 2,
                    "dense input must avoid heap storage"
                );
            }
        }
    }

    #[test]
    fn default_configuration_and_ranges_are_explicit() {
        assert_eq!(
            BpeTrainingConfig::default(),
            BpeTrainingConfig {
                target_vocab_size: 512,
                max_merges: 256,
                min_pair_frequency: 2,
            }
        );
        assert!(
            BpeTrainingConfig {
                target_vocab_size: 255,
                ..CONFIG
            }
            .validate()
            .is_err()
        );
        assert!(
            BpeTrainingConfig {
                max_merges: 5,
                ..CONFIG
            }
            .validate()
            .is_err()
        );
        assert!(
            BpeTrainingConfig {
                min_pair_frequency: 0,
                ..CONFIG
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn known_tiny_corpus_has_expected_merge_sequence() {
        let model = train_model(b"abababab", CONFIG).expect("train tiny corpus");
        assert_eq!(
            model.merges(),
            &[
                FlatBpeMerge {
                    left: u32::from(b'a'),
                    right: u32::from(b'b'),
                    result: 256
                },
                FlatBpeMerge {
                    left: 256,
                    right: 256,
                    result: 257
                },
            ]
        );
    }

    #[test]
    fn pair_frequency_ties_choose_lexicographically_smallest_ids() {
        let model = train_model(b"ababcdcd", CONFIG).expect("train tie fixture");
        assert_eq!(
            model.merges()[0],
            FlatBpeMerge {
                left: u32::from(b'a'),
                right: u32::from(b'b'),
                result: 256,
            }
        );
    }

    #[test]
    fn repeated_pair_counts_overlap_but_replacement_is_non_overlapping() {
        let one_merge = BpeTrainingConfig {
            max_merges: 1,
            ..CONFIG
        };
        let model = train_model(b"aaaaaa", one_merge).expect("train repeated bytes");
        assert_eq!(
            model.merges()[0],
            FlatBpeMerge {
                left: u32::from(b'a'),
                right: u32::from(b'a'),
                result: 256,
            }
        );
        let runtime = BpeTokenizer::new(model);
        let tokens = runtime.encode_bytes(b"aaaaaa").expect("encode repeats");
        assert_eq!(tokens.len(), 3);
        assert_eq!(
            runtime.decode_bpe_bytes(&tokens).expect("decode repeats"),
            b"aaaaaa"
        );
    }

    #[test]
    fn production_trainer_matches_slow_reference_on_deterministic_inputs() {
        let config = BpeTrainingConfig {
            target_vocab_size: 270,
            max_merges: 14,
            min_pair_frequency: 1,
        };
        let mut state = 0x9e37_79b9_u32;
        for case in 0..128 {
            let length = (case * 17) % 73;
            let mut corpus = Vec::with_capacity(length);
            for _ in 0..length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                corpus.push(b'a' + u8::try_from(state % 5).expect("value is in 0..5"));
            }
            assert_eq!(
                train_model(&corpus, config).expect("production training"),
                reference::train_model(&corpus, config).expect("reference training"),
                "case {case}"
            );
        }
    }

    #[test]
    fn runtime_encoder_matches_reference_and_round_trips_bytes() {
        let corpus = "abababab Sämtliche Häuser äöüß 👩🏽‍💻 e\u{0301} 漢字\n".as_bytes();
        let model = train_model(corpus, CONFIG).expect("train model");
        let runtime = BpeTokenizer::new(model.clone());
        for input in [
            "PackTok ASCII",
            "Sämtliche Häuser; ä ö ü ß",
            "✨🦦🇩🇪👩🏽‍💻",
            "a\u{0308} e\u{0301}",
            "漢字かなカナ 中文",
            "line one\nline two\r\n\t  repeated   whitespace",
            "fn main() { println!(\"hello\"); }",
            "",
            "before\0after",
            "aaaaaaaa",
        ] {
            let tokens = runtime.encode(input).expect("encode UTF-8 string");
            assert_eq!(runtime.decode(&tokens).expect("decode UTF-8 string"), input);
            assert_eq!(
                runtime.decode_bytes(&tokens).expect("decode string bytes"),
                input.as_bytes()
            );
        }
        let samples = [
            b"PackTok ASCII".as_slice(),
            "Sämtliche Häuser; ä ö ü ß".as_bytes(),
            "✨🦦🇩🇪👩🏽‍💻".as_bytes(),
            "a\u{0308} e\u{0301}".as_bytes(),
            "漢字かなカナ 中文".as_bytes(),
            "line one\nline two\r\n\t  repeated   whitespace".as_bytes(),
            "fn main() { println!(\"hello\"); }".as_bytes(),
            b"abababab",
            b"aaaaaaaa",
            b"a",
            b"",
            &[0xff, 0, 0xc3, 0x28],
        ];
        for sample in samples {
            let tokens = runtime.encode_bytes(sample).expect("runtime encode");
            let ids: Vec<u32> = tokens.iter().map(|token| token.local).collect();
            assert_eq!(ids, reference::encode(sample, &model));
            assert_eq!(
                runtime.decode_bpe_bytes(&tokens).expect("runtime decode"),
                sample
            );
        }
        assert_ne!(
            runtime.encode("é").expect("precomposed"),
            runtime.encode("e\u{0301}").expect("combining sequence"),
            "M1 must not normalize Unicode"
        );
        assert!(
            runtime
                .decode_bpe_bytes(&[packtok_core::TokenId::new(0, model.vocabulary_size())])
                .is_err()
        );
        assert!(
            runtime
                .decode_bpe_bytes(&[packtok_core::TokenId::new(1, 0)])
                .is_err()
        );

        let mut state = 0xa341_316c_u32;
        for case in 0..128 {
            let length = (case * 13) % 97;
            let mut input = Vec::with_capacity(length);
            for _ in 0..length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                input.push(b'a' + u8::try_from(state % 7).expect("value is in 0..7"));
            }
            let tokens = runtime.encode_bytes(&input).expect("randomized encode");
            let ids: Vec<u32> = tokens.iter().map(|token| token.local).collect();
            assert_eq!(
                ids,
                reference::encode(&input, &model),
                "runtime case {case}"
            );
            assert_eq!(
                runtime
                    .decode_bpe_bytes(&tokens)
                    .expect("randomized decode"),
                input
            );
        }
    }

    #[test]
    fn training_twice_produces_identical_artifact_bytes() {
        let corpus = b"A small corpus. A small corpus.\n";
        let first = train_bpe(corpus, BpeTrainingConfig::default()).expect("first model");
        let second = train_bpe(corpus, BpeTrainingConfig::default()).expect("second model");
        assert_eq!(
            first.to_bytes().expect("serialize first"),
            second.to_bytes().expect("serialize second")
        );
    }

    #[test]
    fn directory_files_are_sorted_and_joined_without_separators() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "packtok-corpus-order-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(root.join("nested")).expect("create temp corpus");
        fs::write(root.join("z.txt"), b"Z").expect("write z");
        fs::write(root.join("nested/a.txt"), b"A").expect("write nested a");
        let corpus = load_corpus(&root).expect("load directory");
        assert_eq!(corpus.bytes(), b"AZ");
        assert_eq!(corpus.provenance().files, ["nested/a.txt", "z.txt"]);
        assert_eq!(corpus.provenance().total_bytes, 2);
        fs::remove_dir_all(root).expect("remove temp corpus");
    }

    #[test]
    fn artifact_preserves_training_provenance_and_encoding() {
        let artifact = train_bpe(b"the the the", CONFIG).expect("train artifact");
        let reloaded =
            Artifact::from_bytes(&artifact.to_bytes().expect("serialize")).expect("reload");
        assert_eq!(reloaded, artifact);
        assert_eq!(reloaded.metadata()["corpus.total_bytes"], "11");
        let original_runtime = BpeTokenizer::from_artifact(&artifact).expect("runtime original");
        let loaded_runtime = BpeTokenizer::from_artifact(&reloaded).expect("runtime loaded");
        assert_eq!(
            original_runtime.encode("the the").expect("encode"),
            loaded_runtime.encode("the the").expect("encode loaded")
        );
    }

    #[test]
    fn event_encoder_matches_reference_across_models_and_all_short_inputs() {
        for corpus in [
            b"aaaaababbcbcbabcabcabcabc".as_slice(),
            b"abababccccccccabccabcbaababa",
            b"aaabaaabaaabbbcccabc",
        ] {
            let model = train_model(
                corpus,
                BpeTrainingConfig {
                    target_vocab_size: 280,
                    max_merges: 24,
                    min_pair_frequency: 1,
                },
            )
            .expect("train event model");
            assert!(model.merges().len() > 8, "exercise event path");
            let runtime = BpeTokenizer::new(model.clone());
            for length in 0..=7_u32 {
                for mut value in 0..3_usize.pow(length) {
                    let input: Vec<u8> = (0..length)
                        .map(|_| {
                            let byte = b'a' + (value % 3) as u8;
                            value /= 3;
                            byte
                        })
                        .collect();
                    let tokens = runtime.encode_bytes(&input).expect("encode short input");
                    let ids: Vec<_> = tokens.iter().map(|token| token.local).collect();
                    assert_eq!(ids, reference::encode(&input, &model), "input {input:?}");
                    assert_eq!(
                        runtime.decode_bytes(&tokens).expect("decode short input"),
                        input
                    );
                }
            }
        }
    }

    #[test]
    fn event_encoder_matches_reference_on_fixture_prefixes_and_arbitrary_bytes() {
        let corpus = include_bytes!("../../../fixtures/m1_bpe_corpus.txt");
        let model = train_model(corpus, BpeTrainingConfig::default()).expect("train fixture");
        let runtime = BpeTokenizer::new(model.clone());
        for length in 0..=corpus.len() {
            let tokens = runtime
                .encode_bytes(&corpus[..length])
                .expect("encode prefix");
            assert_eq!(
                tokens.iter().map(|token| token.local).collect::<Vec<_>>(),
                reference::encode(&corpus[..length], &model)
            );
            assert_eq!(
                runtime.decode_bytes(&tokens).expect("decode prefix"),
                &corpus[..length]
            );
        }
        let all_bytes: Vec<_> = (0..=255_u8).collect();
        let input = all_bytes.repeat(16);
        let tokens = runtime
            .encode_bytes(&input)
            .expect("encode all byte values");
        assert_eq!(
            tokens.iter().map(|token| token.local).collect::<Vec<_>>(),
            reference::encode(&input, &model)
        );
        assert_eq!(
            runtime.decode_bytes(&tokens).expect("decode all bytes"),
            input
        );
        assert!(matches!(
            runtime.decode(&tokens),
            Err(packtok_tokenizer::DecodeError::InvalidUtf8 { .. })
        ));
    }

    #[test]
    fn zero_merge_training_and_empty_inputs_keep_all_byte_ids() {
        for corpus in [b"".as_slice(), b"a", b"aaaaaa", &[0xff, 0, 0xfe]] {
            let model = train_model(
                corpus,
                BpeTrainingConfig {
                    target_vocab_size: 256,
                    max_merges: 0,
                    min_pair_frequency: 1,
                },
            )
            .expect("byte model");
            assert!(model.merges().is_empty());
            let runtime = BpeTokenizer::new(model);
            let tokens = runtime.encode_bytes(corpus).expect("encode bytes");
            assert_eq!(tokens.len(), corpus.len());
            assert_eq!(runtime.decode_bytes(&tokens).expect("decode bytes"), corpus);
        }
    }

    #[test]
    fn corpus_reads_multiple_chunks_and_preserves_raw_bytes() {
        let root =
            std::env::temp_dir().join(format!("packtok-chunked-corpus-{}", std::process::id()));
        fs::create_dir(&root).expect("create corpus directory");
        let input: Vec<_> = (0..=255_u8).cycle().take(150_001).collect();
        fs::write(root.join("bytes.bin"), &input).expect("write input");
        fs::write(root.join("empty.bin"), b"").expect("write empty file");
        let corpus = load_corpus(&root).expect("read corpus");
        assert_eq!(corpus.bytes(), input);
        assert_eq!(corpus.provenance().fnv1a64, fnv1a64(&input));
        assert_eq!(corpus.provenance().files, ["bytes.bin", "empty.bin"]);
        fs::remove_dir_all(root).expect("remove corpus directory");
    }

    #[test]
    fn empty_directory_and_missing_corpus_are_explicit() {
        let root =
            std::env::temp_dir().join(format!("packtok-empty-corpus-{}", std::process::id()));
        fs::create_dir(&root).expect("create empty directory");
        let corpus = load_corpus(&root).expect("load empty directory");
        assert!(corpus.bytes().is_empty());
        assert!(corpus.provenance().files.is_empty());
        assert!(matches!(
            load_corpus(&root.join("missing")),
            Err(CorpusError::Io { .. })
        ));
        fs::remove_dir(root).expect("remove empty directory");
    }

    #[cfg(unix)]
    #[test]
    fn literal_backslash_and_nested_paths_remain_distinct_and_deterministic() {
        let root =
            std::env::temp_dir().join(format!("packtok-path-collision-{}", std::process::id()));
        fs::create_dir(&root).expect("create root");
        fs::create_dir(root.join("a")).expect("create nested directory");
        fs::write(root.join("a\\b"), b"literal").expect("write literal backslash name");
        fs::write(root.join("a/b"), b"nested").expect("write nested path");
        let first = load_corpus(&root).expect("first load");
        let second = load_corpus(&root).expect("second load");
        assert_eq!(first.provenance().files, ["a/b", "a\\b"]);
        assert_eq!(first.bytes(), b"nestedliteral");
        assert_eq!(first, second);
        fs::remove_dir_all(root).expect("remove test corpus");
    }
}
