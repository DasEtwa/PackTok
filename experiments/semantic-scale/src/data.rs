//! Bounded Rust preparation and seek-based batches for the scaling experiment.
//!
//! Source files are explicit pre-split `train.txt`, `validation.txt`, `test.txt`.
//! Lexical BPE trains only on the first 1 MiB of TRAIN, with target 512 IDs.
//! Each original line (including its newline) is encoded independently; lines
//! larger than 1 MiB are rejected before unbounded allocation. No bytes are
//! inserted, discarded or normalized. The on-disk stream has fixed 12-byte
//! records and can be read without holding the dataset in memory.
//!
//! Semantic input is deliberately causal: only context-free KnownVariant
//! resolutions emit an event, at the first lexical input token which has fully
//! observed the following delimiter scalar. Apostrophe delimiters are excluded.
//! An expression ending at EOF emits nothing. Context-resolved fr/ka never emit.
//! If a token completes several events, the latest completed expression wins.
//! Recognition stops at the first invalid/incomplete UTF-8 scalar in a line;
//! verified earlier events survive, and the lexical byte stream is unaffected.
//! Persisted event fields are provenance only. Batches recompute semantic events
//! from each exact INPUT window, so an event cannot see text before the window
//! or a delimiter appearing only in a future target token.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use packtok_core::TokenId;
use packtok_format::{Artifact, MAX_ARTIFACT_BYTES};
use packtok_packs::semantic::{SEMANTIC_POLICY_ID, SemanticReason, SemanticRegistry};
use packtok_tokenizer::{BpeTokenizer, Tokenizer};
use packtok_train::{BpeTrainingConfig, train_bpe};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Result;

pub const MAX_LINE_BYTES: usize = 1024 * 1024;
pub const TRAIN_SAMPLE_BYTES: usize = 1024 * 1024;
pub const CAUSAL_FEATURE_POLICY: &str = "causal-known-window-v1";
const MAX_EXPANSION_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_WINDOW_BYTES: usize = 16 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"PTKSCLD\0";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = 32;
const RECORD_BYTES: u32 = 12;
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const SPLITS: [&str; 3] = ["train", "validation", "test"];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SplitManifest {
    pub file: String,
    pub sha256: String,
    pub source_sha256: String,
    pub records: u64,
    pub raw_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub record_bytes: u32,
    pub max_line_bytes: usize,
    pub training_sample_bytes: usize,
    pub training_sample_sha256: String,
    pub tokenizer_sha256: String,
    pub vocabulary_size: u32,
    pub semantic_namespace: String,
    pub semantic_revision: u32,
    pub semantic_policy: String,
    pub semantic_fingerprint: u64,
    pub feature_policy: String,
    pub segmentation: String,
    pub splits: BTreeMap<String, SplitManifest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Record {
    token: u32,
    semantic: u32,
    bytes: u32,
}

/// Returns features aligned with the given cumulative lexical byte ends.
///
/// `token_ends` are for an existing lexical sequence, not a new segmentation.
/// Comparing a prefix must use the same consumed token IDs/byte ends: canonical
/// BPE itself may merge differently if text is independently re-tokenized.
pub fn semantic_features(raw: &[u8], token_ends: &[usize]) -> Result<Vec<u32>> {
    let mut previous = 0;
    for &end in token_ends {
        if end <= previous || end > raw.len() {
            return Err("semantic feature token byte ends are invalid".into());
        }
        previous = end;
    }
    if token_ends.last().copied().unwrap_or(0) != raw.len() {
        return Err("semantic feature byte ends do not cover input".into());
    }
    let mut features = vec![0; token_ends.len()];
    let mut line_start = 0;
    for line in raw.split_inclusive(|byte| *byte == b'\n') {
        let valid = match std::str::from_utf8(line) {
            Ok(text) => text,
            Err(error) => std::str::from_utf8(&line[..error.valid_up_to()])?,
        };
        for annotation in SemanticRegistry::builtin().annotate(valid) {
            if annotation.reason != SemanticReason::KnownVariant {
                continue;
            }
            let Some(id) = annotation.resolved_id() else {
                continue;
            };
            let Some(delimiter) = valid[annotation.end..].chars().next() else {
                continue;
            };
            if matches!(delimiter, '\'' | '\u{2019}') {
                continue;
            }
            let observed_end = line_start + annotation.end + delimiter.len_utf8();
            let token_index = token_ends.partition_point(|&end| end < observed_end);
            if let Some(feature) = features.get_mut(token_index) {
                *feature = id.0;
            }
        }
        line_start += line.len();
    }
    Ok(features)
}

/// Prepares a fresh immutable output directory. A partial failure retains its
/// output for diagnosis, but without manifest.json it cannot be loaded as ready.
pub fn prepare(input_dir: &Path, out: &Path) -> Result<()> {
    let train = File::open(input_dir.join("train.txt"))?;
    let mut sample = Vec::new();
    train
        .take(TRAIN_SAMPLE_BYTES as u64)
        .read_to_end(&mut sample)?;
    if sample.is_empty() {
        return Err("TRAIN must not be empty".into());
    }
    let artifact = train_bpe(&sample, BpeTrainingConfig::default())?;
    let tokenizer = BpeTokenizer::from_artifact(&artifact)?;
    let artifact_bytes = artifact.to_bytes()?;
    let lengths = token_byte_lengths(&tokenizer)?;
    // Refuse overwrite: a checkpoint's data identity must not silently change.
    fs::create_dir(out)?;
    write_new(&out.join("tokenizer.packtok"), &artifact_bytes)?;
    let mut splits = BTreeMap::new();
    for split in SPLITS {
        let manifest = prepare_split(
            &input_dir.join(format!("{split}.txt")),
            &out.join(format!("{split}.ptks")),
            &tokenizer,
            &lengths,
        )?;
        if splits
            .values()
            .any(|previous: &SplitManifest| previous.source_sha256 == manifest.source_sha256)
        {
            return Err(
                "identical source splits rejected; supply independent train/validation/test files"
                    .into(),
            );
        }
        splits.insert(split.to_owned(), manifest);
    }
    let registry = SemanticRegistry::builtin();
    let manifest = Manifest {
        version: VERSION,
        record_bytes: RECORD_BYTES,
        max_line_bytes: MAX_LINE_BYTES,
        training_sample_bytes: sample.len(),
        training_sample_sha256: digest(&sample),
        tokenizer_sha256: digest(&artifact_bytes),
        vocabulary_size: tokenizer.vocabulary_size(),
        semantic_namespace: registry.namespace().to_owned(),
        semantic_revision: registry.revision(),
        semantic_policy: SEMANTIC_POLICY_ID.to_owned(),
        semantic_fingerprint: registry.fingerprint(),
        feature_policy: CAUSAL_FEATURE_POLICY.to_owned(),
        segmentation: "independent original lines including LF; no normalization".to_owned(),
        splits,
    };
    write_new(
        &out.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

fn prepare_split(
    source: &Path,
    output: &Path,
    tokenizer: &BpeTokenizer,
    lengths: &[usize],
) -> Result<SplitManifest> {
    let mut input = BufReader::new(File::open(source)?);
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    writer.write_all(&header(0, 0))?;
    let mut source_hash = Sha256::new();
    let mut raw_bytes = 0_u64;
    let mut records = 0_u64;
    let mut line = Vec::new();
    while read_bounded_line(&mut input, &mut line)? {
        source_hash.update(&line);
        raw_bytes = raw_bytes
            .checked_add(line.len() as u64)
            .ok_or("source length overflow")?;
        let tokens = tokenizer.encode_bytes(&line)?;
        if tokenizer.decode_bytes(&tokens)? != line {
            return Err("preparation lexical roundtrip failed".into());
        }
        let mut ends = Vec::with_capacity(tokens.len());
        let mut end = 0_usize;
        for token in &tokens {
            let length = lengths
                .get(token.local as usize)
                .ok_or("unknown encoded token ID")?;
            end = end
                .checked_add(*length)
                .ok_or("token expansion length overflow")?;
            ends.push(end);
        }
        let features = semantic_features(&line, &ends)?;
        for (token, semantic) in tokens.iter().zip(features) {
            let bytes = u32::try_from(lengths[token.local as usize])?;
            writer.write_all(&record_bytes(Record {
                token: token.local,
                semantic,
                bytes,
            }))?;
            records = records.checked_add(1).ok_or("token count overflow")?;
        }
    }
    if raw_bytes == 0 || records == 0 {
        return Err(format!("{} must not be empty", source.display()).into());
    }
    writer.flush()?;
    writer.seek(SeekFrom::Start(0))?;
    writer.write_all(&header(records, raw_bytes))?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(SplitManifest {
        file: output
            .file_name()
            .ok_or("missing output name")?
            .to_string_lossy()
            .into_owned(),
        sha256: file_digest(output)?,
        source_sha256: format!("{:x}", source_hash.finalize()),
        records,
        raw_bytes,
    })
}

fn read_bounded_line<R: BufRead>(input: &mut R, line: &mut Vec<u8>) -> Result<bool> {
    line.clear();
    loop {
        let available = input.fill_buf()?;
        if available.is_empty() {
            return Ok(!line.is_empty());
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |i| i + 1);
        if line
            .len()
            .checked_add(take)
            .ok_or("source line length overflow")?
            > MAX_LINE_BYTES
        {
            return Err(format!("source line exceeds {MAX_LINE_BYTES} bytes; split long lines explicitly before preparation").into());
        }
        line.extend_from_slice(&available[..take]);
        let ended = available[take - 1] == b'\n';
        input.consume(take);
        if ended {
            return Ok(true);
        }
    }
}

fn header(records: u64, raw_bytes: u64) -> [u8; HEADER_BYTES as usize] {
    let mut bytes = [0; HEADER_BYTES as usize];
    bytes[..8].copy_from_slice(MAGIC);
    bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
    bytes[12..16].copy_from_slice(&RECORD_BYTES.to_le_bytes());
    bytes[16..24].copy_from_slice(&records.to_le_bytes());
    bytes[24..32].copy_from_slice(&raw_bytes.to_le_bytes());
    bytes
}

fn record_bytes(record: Record) -> [u8; RECORD_BYTES as usize] {
    let mut bytes = [0; RECORD_BYTES as usize];
    bytes[..4].copy_from_slice(&record.token.to_le_bytes());
    bytes[4..8].copy_from_slice(&record.semantic.to_le_bytes());
    bytes[8..12].copy_from_slice(&record.bytes.to_le_bytes());
    bytes
}

fn read_record(reader: &mut impl Read) -> Result<Record> {
    let mut bytes = [0; RECORD_BYTES as usize];
    reader.read_exact(&mut bytes)?;
    Ok(Record {
        token: u32::from_le_bytes(bytes[..4].try_into()?),
        semantic: u32::from_le_bytes(bytes[4..8].try_into()?),
        bytes: u32::from_le_bytes(bytes[8..12].try_into()?),
    })
}

fn token_byte_lengths(tokenizer: &BpeTokenizer) -> Result<Vec<usize>> {
    let mut lengths = vec![1_usize; 256];
    for merge in tokenizer.merges() {
        if merge.result as usize != lengths.len() {
            return Err("noncanonical BPE merge rank".into());
        }
        let left = *lengths
            .get(merge.left as usize)
            .ok_or("BPE left ID missing")?;
        let right = *lengths
            .get(merge.right as usize)
            .ok_or("BPE right ID missing")?;
        lengths.push(left.checked_add(right).ok_or("BPE expansion overflow")?);
    }
    Ok(lengths)
}

fn read_manifest(root: &Path) -> Result<Manifest> {
    let path = root.join("manifest.json");
    if fs::metadata(&path)?.len() > MAX_MANIFEST_BYTES {
        return Err("dataset manifest exceeds 64 KiB".into());
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
    let registry = SemanticRegistry::builtin();
    if manifest.version != VERSION
        || manifest.record_bytes != RECORD_BYTES
        || manifest.max_line_bytes != MAX_LINE_BYTES
        || manifest.training_sample_bytes == 0
        || manifest.training_sample_bytes > TRAIN_SAMPLE_BYTES
        || !(256..=512).contains(&manifest.vocabulary_size)
        || manifest.semantic_namespace != registry.namespace()
        || manifest.semantic_revision != registry.revision()
        || manifest.semantic_policy != SEMANTIC_POLICY_ID
        || manifest.semantic_fingerprint != registry.fingerprint()
        || manifest.feature_policy != CAUSAL_FEATURE_POLICY
        || manifest.segmentation != "independent original lines including LF; no normalization"
        || manifest.splits.len() != SPLITS.len()
    {
        return Err("unsupported or inconsistent prepared dataset manifest".into());
    }
    if !valid_digest(&manifest.tokenizer_sha256) || !valid_digest(&manifest.training_sample_sha256)
    {
        return Err("invalid dataset tokenizer/sample fingerprint".into());
    }
    for split in SPLITS {
        let entry = manifest.splits.get(split).ok_or("dataset split missing")?;
        if entry.file != format!("{split}.ptks")
            || entry.records == 0
            || entry.raw_bytes == 0
            || !valid_digest(&entry.sha256)
            || !valid_digest(&entry.source_sha256)
        {
            return Err("invalid prepared split metadata".into());
        }
    }
    Ok(manifest)
}

/// Identity used to bind checkpoints to exact prepared input and registry policy.
pub fn dataset_identity(root: &Path) -> Result<String> {
    read_manifest(root)?;
    file_digest(&root.join("manifest.json"))
}

/// Immutable file-backed dataset. Open validates a streaming SHA and every record;
/// batches thereafter read only their windows. Do not mutate prepared files while
/// a dataset is open. Separate file handles make concurrent batch reads independent.
pub struct Dataset {
    path: PathBuf,
    records: u64,
    raw_bytes: u64,
    token_expansions: Vec<Vec<u8>>,
}

impl Dataset {
    pub fn open(root: &Path, split: &str) -> Result<Self> {
        if !SPLITS.contains(&split) {
            return Err("split must be train, validation or test".into());
        }
        let manifest = read_manifest(root)?;
        let entry = manifest.splits.get(split).ok_or("split missing")?;
        let artifact_path = root.join("tokenizer.packtok");
        if fs::metadata(&artifact_path)?.len() > MAX_ARTIFACT_BYTES as u64 {
            return Err("tokenizer artifact is too large".into());
        }
        let artifact_bytes = fs::read(&artifact_path)?;
        if digest(&artifact_bytes) != manifest.tokenizer_sha256 {
            return Err("prepared tokenizer SHA256 mismatch".into());
        }
        let artifact = Artifact::from_bytes(&artifact_bytes)?;
        let tokenizer = BpeTokenizer::from_artifact(&artifact)?;
        if tokenizer.vocabulary_size() != manifest.vocabulary_size {
            return Err("prepared tokenizer vocabulary mismatch".into());
        }
        let lengths = token_byte_lengths(&tokenizer)?;
        let path = root.join(&entry.file);
        let expected_size = entry
            .records
            .checked_mul(u64::from(RECORD_BYTES))
            .and_then(|n| n.checked_add(HEADER_BYTES))
            .ok_or("prepared split size overflow")?;
        if fs::metadata(&path)?.len() != expected_size || file_digest(&path)? != entry.sha256 {
            return Err("prepared split length/SHA256 mismatch".into());
        }
        let mut reader = BufReader::new(File::open(&path)?);
        let mut actual_header = [0; HEADER_BYTES as usize];
        reader.read_exact(&mut actual_header)?;
        if actual_header != header(entry.records, entry.raw_bytes) {
            return Err("prepared split header mismatch".into());
        }
        let mut counted_bytes = 0_u64;
        for _ in 0..entry.records {
            let record = read_record(&mut reader)?;
            if lengths.get(record.token as usize).copied() != Some(record.bytes as usize)
                || record.semantic > 5
            {
                return Err("prepared split contains invalid token/semantic/byte record".into());
            }
            counted_bytes = counted_bytes
                .checked_add(u64::from(record.bytes))
                .ok_or("prepared bytes overflow")?;
        }
        if counted_bytes != entry.raw_bytes {
            return Err("prepared raw byte count mismatch".into());
        }
        let mut token_expansions = Vec::with_capacity(manifest.vocabulary_size as usize);
        let mut expansion_bytes = 0_usize;
        for local in 0..manifest.vocabulary_size {
            let expansion = tokenizer.decode_bytes(&[TokenId::new(0, local)])?;
            expansion_bytes = expansion_bytes
                .checked_add(expansion.len())
                .ok_or("tokenizer expansion cache size overflow")?;
            if expansion_bytes > MAX_EXPANSION_CACHE_BYTES {
                return Err("tokenizer expansion cache exceeds 64 MiB".into());
            }
            token_expansions.push(expansion);
        }
        Ok(Self {
            path,
            records: entry.records,
            raw_bytes: entry.raw_bytes,
            token_expansions,
        })
    }

    #[must_use]
    pub const fn len(&self) -> u64 {
        self.records
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.records == 0
    }
    #[must_use]
    pub const fn raw_bytes(&self) -> u64 {
        self.raw_bytes
    }

    /// Reads B contiguous input/next-token windows. A window never wraps. If a
    /// window would overrun this split, it restarts at token zero and discards
    /// the short tail; the first target token of each epoch is therefore token 1.
    /// `next_cursor` plus optimizer step fully determines deterministic resume.
    /// Semantic features are recomputed from this exact input window only;
    /// precomputed whole-line events never supply extra pre-window context.
    pub fn batch(
        &self,
        cursor: u64,
        batch: usize,
        context: usize,
        device: &Device,
        semantics: bool,
    ) -> Result<Batch> {
        if batch == 0 || batch > 1024 || context == 0 || context > 4096 {
            return Err("batch/context outside supported bounds".into());
        }
        if self.records <= context as u64 {
            return Err("dataset needs at least context+1 lexical tokens".into());
        }
        let positions = batch
            .checked_mul(context)
            .ok_or("batch position overflow")?;
        let mut inputs = Vec::with_capacity(positions);
        let mut semantic = Vec::with_capacity(positions);
        let mut targets = Vec::with_capacity(positions);
        let mut target_bytes = 0_u64;
        let mut cursor = cursor % self.records;
        let mut reader = BufReader::new(File::open(&self.path)?);
        for _ in 0..batch {
            if cursor
                .checked_add(context as u64)
                .ok_or("batch cursor overflow")?
                >= self.records
            {
                cursor = 0;
            }
            let offset = cursor
                .checked_mul(u64::from(RECORD_BYTES))
                .and_then(|n| n.checked_add(HEADER_BYTES))
                .ok_or("batch seek overflow")?;
            reader.seek(SeekFrom::Start(offset))?;
            let mut previous = read_record(&mut reader)?;
            let mut window_bytes = Vec::new();
            let mut window_ends = Vec::with_capacity(context);
            for _ in 0..context {
                let next = read_record(&mut reader)?;
                inputs.push(previous.token);
                if semantics {
                    let expansion = self
                        .token_expansions
                        .get(previous.token as usize)
                        .ok_or("input token outside cached vocabulary")?;
                    let length = window_bytes
                        .len()
                        .checked_add(expansion.len())
                        .ok_or("input window byte length overflow")?;
                    if length > MAX_WINDOW_BYTES {
                        return Err("semantic input window exceeds 16 MiB".into());
                    }
                    window_bytes.extend_from_slice(expansion);
                    window_ends.push(window_bytes.len());
                }
                targets.push(next.token);
                target_bytes = target_bytes
                    .checked_add(u64::from(next.bytes))
                    .ok_or("batch target byte overflow")?;
                previous = next;
            }
            if semantics {
                semantic.extend(semantic_features(&window_bytes, &window_ends)?);
            } else {
                semantic.resize(semantic.len() + context, 0);
            }
            cursor += context as u64;
        }
        Ok(Batch {
            inputs: Tensor::from_vec(inputs, (batch, context), device)?,
            semantic: Tensor::from_vec(semantic, (batch, context), device)?,
            targets: Tensor::from_vec(targets, (batch, context), device)?,
            target_bytes,
            next_cursor: cursor,
        })
    }
}

pub struct Batch {
    pub inputs: Tensor,
    pub semantic: Tensor,
    pub targets: Tensor,
    pub target_bytes: u64,
    pub next_cursor: u64,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "packtok-scale-data-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(root: &Path) {
        for (split, lead) in [
            ("train", "Training"),
            ("validation", "Validation"),
            ("test", "Testing"),
        ] {
            let text = format!("{lead} idk; ngl ich habe keine Ahnung! 🦦\n").repeat(8);
            fs::write(root.join(format!("{split}.txt")), text).unwrap();
        }
    }

    #[test]
    fn semantic_events_wait_for_observed_delimiter_and_ignore_future_context() {
        for (input, expected) in [
            ("idk", vec![0, 0, 0]),
            ("idk ", vec![0, 0, 0, 1]),
            ("idkXYZ ", vec![0; 7]),
            ("fr bro ", vec![0; 7]),
            ("ka warum ", vec![0; 9]),
            ("idk's ", vec![0; 6]),
            ("idk’ ", vec![0; 7]),
        ] {
            let ends = (1..=input.len()).collect::<Vec<_>>();
            assert_eq!(
                semantic_features(input.as_bytes(), &ends).unwrap(),
                expected,
                "{input}"
            );
        }
        let input = "I don't know 🦦";
        let ends = (1..=input.len()).collect::<Vec<_>>();
        let features = semantic_features(input.as_bytes(), &ends).unwrap();
        assert_eq!(features.iter().filter(|&&id| id != 0).count(), 1);
        assert_eq!(features[12], 1); // The space after the complete phrase.
        let input = "idk🦦";
        let ends = (1..=input.len()).collect::<Vec<_>>();
        let features = semantic_features(input.as_bytes(), &ends).unwrap();
        assert_eq!(features.last(), Some(&1));
        assert!(features[..features.len() - 1].iter().all(|&id| id == 0));
    }

    #[test]
    fn semantic_features_are_prefix_causal_for_fixed_lexical_boundaries() {
        for input in [
            "ong idk bro; ngl to be honest, keine Ahnung! I don't know?",
            "idkXYZ idk’s idk'word language: fr bro ka was 🙂 tbh\n",
            "idk 🦦 dunno✨ idk， idk\u{0301} 🧪",
            "I don't know keIne\tAhnung ngl\r\ntbh\0 ",
        ] {
            let bytes = input.as_bytes();
            let ends = (1..=bytes.len()).collect::<Vec<_>>();
            let full = semantic_features(bytes, &ends).unwrap();
            for end in 0..=bytes.len() {
                let prefix_ends = (1..=end).collect::<Vec<_>>();
                let prefix = semantic_features(&bytes[..end], &prefix_ends).unwrap();
                assert_eq!(prefix, full[..end], "{input:?} prefix={end}");
            }
        }
        let bytes = b"idk ok \xff ngl \n tbh ";
        let ends = (1..=bytes.len()).collect::<Vec<_>>();
        let full = semantic_features(bytes, &ends).unwrap();
        assert_eq!(full[3], 1);
        assert_eq!(full.iter().filter(|&&id| id != 0).count(), 2);
        for end in 0..=bytes.len() {
            assert_eq!(
                semantic_features(&bytes[..end], &(1..=end).collect::<Vec<_>>()).unwrap(),
                full[..end]
            );
        }
    }

    #[test]
    fn bounded_lines_preserve_all_bytes_and_reject_large_lines() {
        let input = b"one\r\ntwo\0\nlast";
        let mut reader = BufReader::with_capacity(3, Cursor::new(input));
        let mut line = Vec::new();
        let mut recovered = Vec::new();
        while read_bounded_line(&mut reader, &mut line).unwrap() {
            recovered.extend_from_slice(&line);
        }
        assert_eq!(recovered, input);
        let bytes = vec![b'x'; MAX_LINE_BYTES + 1];
        let mut reader = BufReader::with_capacity(1024, Cursor::new(bytes));
        assert!(read_bounded_line(&mut reader, &mut line).is_err());
        assert!(line.len() <= MAX_LINE_BYTES);
    }

    #[test]
    fn prepared_data_is_deterministic_seekable_and_resumable() {
        let temp = Temp::new();
        fixture(&temp.0);
        let first = temp.0.join("first");
        let second = temp.0.join("second");
        prepare(&temp.0, &first).unwrap();
        prepare(&temp.0, &second).unwrap();
        assert_eq!(
            dataset_identity(&first).unwrap(),
            dataset_identity(&second).unwrap()
        );
        for name in [
            "tokenizer.packtok",
            "manifest.json",
            "train.ptks",
            "validation.ptks",
            "test.ptks",
        ] {
            assert_eq!(
                fs::read(first.join(name)).unwrap(),
                fs::read(second.join(name)).unwrap()
            );
        }
        let ds = Dataset::open(&first, "train").unwrap();
        let a = ds.batch(0, 2, 4, &Device::Cpu, true).unwrap();
        let b = ds.batch(0, 2, 4, &Device::Cpu, false).unwrap();
        assert_eq!(
            a.inputs.to_vec2::<u32>().unwrap(),
            b.inputs.to_vec2::<u32>().unwrap()
        );
        assert_eq!(
            a.targets.to_vec2::<u32>().unwrap(),
            b.targets.to_vec2::<u32>().unwrap()
        );
        assert_eq!(a.target_bytes, b.target_bytes);
        assert!(
            b.semantic
                .to_vec2::<u32>()
                .unwrap()
                .iter()
                .flatten()
                .all(|&id| id == 0)
        );
        let resumed = ds.batch(a.next_cursor, 2, 4, &Device::Cpu, true).unwrap();
        let reopened = Dataset::open(&first, "train")
            .unwrap()
            .batch(a.next_cursor, 2, 4, &Device::Cpu, true)
            .unwrap();
        assert_eq!(
            resumed.inputs.to_vec2::<u32>().unwrap(),
            reopened.inputs.to_vec2::<u32>().unwrap()
        );
        assert_eq!(resumed.next_cursor, reopened.next_cursor);
        assert!(ds.batch(0, 0, 4, &Device::Cpu, true).is_err());
        assert!(ds.batch(0, 1, 4097, &Device::Cpu, true).is_err());
        assert!(prepare(&temp.0, &first).is_err());
    }

    #[test]
    fn batch_semantics_see_exact_input_window_and_never_the_target_delimiter() {
        let temp = Temp::new();
        let raw = b"to be honest xyz and more";
        let ends = (1..=raw.len()).collect::<Vec<_>>();
        let global = semantic_features(raw, &ends).unwrap();
        assert_eq!(global[12], 2);
        let path = temp.0.join("window.ptks");
        let mut bytes = header(raw.len() as u64, raw.len() as u64).to_vec();
        for (&token, &semantic) in raw.iter().zip(&global) {
            bytes.extend_from_slice(&record_bytes(Record {
                token: u32::from(token),
                semantic,
                bytes: 1,
            }));
        }
        fs::write(&path, bytes).unwrap();
        let ds = Dataset {
            path,
            records: raw.len() as u64,
            raw_bytes: raw.len() as u64,
            token_expansions: (0_u8..=255).map(|b| vec![b]).collect(),
        };
        // The stored global event sits inside this crop, but its originating
        // phrase begins before the model's visible input at byte/token three.
        let cropped = ds.batch(3, 1, 10, &Device::Cpu, true).unwrap();
        assert_eq!(
            cropped.inputs.to_vec2::<u32>().unwrap()[0],
            raw[3..13].iter().map(|&b| u32::from(b)).collect::<Vec<_>>()
        );
        assert!(
            cropped.semantic.to_vec2::<u32>().unwrap()[0]
                .iter()
                .all(|&id| id == 0)
        );
        let complete = ds.batch(0, 1, 13, &Device::Cpu, true).unwrap();
        assert_eq!(complete.semantic.to_vec2::<u32>().unwrap()[0][12], 2);
        // The phrase ends in the input, but its required delimiter exists only
        // in the target: it cannot activate an input-side semantic feature.
        let target_only = ds.batch(0, 1, 12, &Device::Cpu, true).unwrap();
        assert_eq!(
            *target_only.targets.to_vec2::<u32>().unwrap()[0]
                .last()
                .unwrap(),
            u32::from(b' ')
        );
        assert!(
            target_only.semantic.to_vec2::<u32>().unwrap()[0]
                .iter()
                .all(|&id| id == 0)
        );
    }

    #[test]
    fn malformed_prepared_data_is_rejected_even_with_updated_container_hash() {
        let temp = Temp::new();
        fixture(&temp.0);
        let root = temp.0.join("prepared");
        prepare(&temp.0, &root).unwrap();
        let path = root.join("train.ptks");
        let original = fs::read(&path).unwrap();
        let mut bytes = original.clone();
        bytes.pop();
        fs::write(&path, bytes).unwrap();
        assert!(Dataset::open(&root, "train").is_err());
        let mut bytes = original;
        bytes[HEADER_BYTES as usize + 4..HEADER_BYTES as usize + 8]
            .copy_from_slice(&999_u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        let mut manifest = read_manifest(&root).unwrap();
        manifest.splits.get_mut("train").unwrap().sha256 = digest(&bytes);
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        assert!(Dataset::open(&root, "train").is_err());
    }
}
