use std::collections::BTreeMap;

use packtok_core::{
    BYTE_FALLBACK_PACK_NAME, BYTE_FALLBACK_TOKEN_COUNT, ByteFallback,
    DEFAULT_BYTE_FALLBACK_PACK_ID, PackDescriptor, PackRegistry,
};
use packtok_format::{
    Artifact, FactorizedBpeModel, MAX_BPE_TOKEN_BYTES, PackBpeMerge, PackBpeModel, SymbolRef,
};
use packtok_packs::{
    LEXICAL_V1_ROUTER_ID, LexicalV1Router, NUMBER_PACK_ID, PackRouter, RoutedSpan,
    STRUCTURE_PACK_ID, TEXT_PACK_ID, lexical_pack_name,
};

use crate::{CorpusProvenance, TrainingError, fnv1a64};

/// Configuration for one global M2 learned-token budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FactorizedTrainingConfig {
    /// Total logical vocabulary slots, including the single shared 256-byte pack.
    pub target_vocab_size: u32,
    /// Upper bound on learned tokens shared by all specialized packs.
    pub max_learned_tokens: u32,
    /// Minimum overlapping adjacent-pair frequency within one pack.
    pub min_pair_frequency: u64,
}

impl Default for FactorizedTrainingConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 512,
            max_learned_tokens: 256,
            min_pair_frequency: 2,
        }
    }
}

impl FactorizedTrainingConfig {
    /// Validates the shared-byte base, learned-token budget, and positive frequency.
    pub fn validate(self) -> Result<Self, TrainingError> {
        if self.target_vocab_size < BYTE_FALLBACK_TOKEN_COUNT {
            return Err(TrainingError::InvalidConfig {
                field: "target_vocab_size",
                reason: "must include at least the 256 shared byte tokens",
            });
        }
        if self.max_learned_tokens > self.target_vocab_size - BYTE_FALLBACK_TOKEN_COUNT {
            return Err(TrainingError::InvalidConfig {
                field: "max_learned_tokens",
                reason: "cannot exceed target_vocab_size minus the 256 shared byte tokens",
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

/// Per-pack corpus and learned-vocabulary counts for a deterministic training run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FactorizedPackTrainingStats {
    /// Stable pack ID (shared fallback is `65535`).
    pub pack_id: u16,
    /// Number of routed spans in this pack before BPE training.
    pub spans: u64,
    /// Number of original corpus bytes routed to this pack.
    pub routed_bytes: u64,
    /// Number of learned local tokens assigned by the global budget.
    pub learned_merges: u32,
    /// Local learned vocabulary size; byte tokens are shared and not duplicated.
    pub local_vocab_size: u32,
}

/// Artifact plus explanatory pack-allocation counts from M2 training.
#[derive(Clone, Debug)]
pub struct FactorizedTrainingReport {
    /// Self-contained version-3 tokenizer artifact.
    pub artifact: Artifact,
    /// Counts in byte-fallback, TEXT, NUMBER, STRUCTURE order.
    pub packs: [FactorizedPackTrainingStats; 4],
    /// Realized total logical vocabulary size, shared bytes plus learned tokens.
    pub total_logical_vocab_size: u32,
}

#[derive(Debug)]
struct PackState {
    pack_id: u16,
    spans: Vec<Vec<SymbolRef>>,
    merges: Vec<PackBpeMerge>,
    local_lengths: Vec<usize>,
    routed_bytes: u64,
    span_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Candidate {
    left: SymbolRef,
    right: SymbolRef,
    frequency: u64,
}

/// Trains the default M2 policy on a UTF-8 corpus held in memory.
pub fn train_factorized_bpe(
    corpus: &[u8],
    config: FactorizedTrainingConfig,
) -> Result<Artifact, TrainingError> {
    Ok(train_factorized_bpe_with_report(corpus, config)?.artifact)
}

/// Trains M2 and records caller-supplied source provenance in descriptive metadata.
pub fn train_factorized_bpe_with_provenance(
    corpus: &[u8],
    config: FactorizedTrainingConfig,
    provenance: &CorpusProvenance,
) -> Result<Artifact, TrainingError> {
    Ok(train_factorized_bpe_with_provenance_report(corpus, config, provenance)?.artifact)
}

/// Trains M2 and returns its per-pack allocation report.
pub fn train_factorized_bpe_with_report(
    corpus: &[u8],
    config: FactorizedTrainingConfig,
) -> Result<FactorizedTrainingReport, TrainingError> {
    let total_bytes = u64::try_from(corpus.len()).map_err(|_| TrainingError::LengthOverflow)?;
    let provenance = CorpusProvenance {
        input: "memory".to_owned(),
        files: Vec::new(),
        total_bytes,
        fnv1a64: fnv1a64(corpus),
    };
    train_factorized_bpe_with_provenance_report(corpus, config, &provenance)
}

/// Trains M2 with validated provenance and returns stable per-pack counts.
pub fn train_factorized_bpe_with_provenance_report(
    corpus: &[u8],
    config: FactorizedTrainingConfig,
    provenance: &CorpusProvenance,
) -> Result<FactorizedTrainingReport, TrainingError> {
    let config = config.validate()?;
    if provenance.files.len() > 65_000 {
        return Err(TrainingError::TooManyProvenanceFiles(
            provenance.files.len(),
        ));
    }
    let total_bytes = u64::try_from(corpus.len()).map_err(|_| TrainingError::LengthOverflow)?;
    if provenance.total_bytes != total_bytes || provenance.fnv1a64 != fnv1a64(corpus) {
        return Err(TrainingError::InvalidProvenance);
    }
    let text = std::str::from_utf8(corpus).map_err(|error| TrainingError::InvalidUtf8Corpus {
        valid_up_to: error.valid_up_to(),
    })?;
    let (model, pack_stats, total_logical_vocab_size) =
        train_factorized_model_internal(text, config)?;
    let mut pack_descriptors = Vec::new();
    pack_descriptors
        .try_reserve_exact(model.packs().len() + 1)
        .map_err(|_| TrainingError::AllocationFailed)?;
    for pack in model.packs() {
        let name = lexical_pack_name(pack.pack_id()).ok_or(TrainingError::InvalidRouting)?;
        pack_descriptors.push(PackDescriptor::new(
            pack.pack_id(),
            name,
            pack.local_token_count(),
        ));
    }
    pack_descriptors.push(PackDescriptor::new(
        DEFAULT_BYTE_FALLBACK_PACK_ID,
        BYTE_FALLBACK_PACK_NAME,
        BYTE_FALLBACK_TOKEN_COUNT,
    ));
    let registry = PackRegistry::new(
        pack_descriptors,
        ByteFallback::new(DEFAULT_BYTE_FALLBACK_PACK_ID),
        Vec::new(),
    )?;
    let metadata = factorized_metadata(
        config,
        &model,
        &pack_stats,
        total_logical_vocab_size,
        provenance,
    )?;
    let artifact = Artifact::with_factorized_bpe(registry, metadata, model)?;
    Ok(FactorizedTrainingReport {
        artifact,
        packs: pack_stats,
        total_logical_vocab_size,
    })
}

impl PackState {
    fn new(pack_id: u16) -> Self {
        Self {
            pack_id,
            spans: Vec::new(),
            merges: Vec::new(),
            local_lengths: Vec::new(),
            routed_bytes: 0,
            span_count: 0,
        }
    }
}

fn state_index(pack_id: u16) -> Option<usize> {
    match pack_id {
        TEXT_PACK_ID => Some(0),
        NUMBER_PACK_ID => Some(1),
        STRUCTURE_PACK_ID => Some(2),
        _ => None,
    }
}

fn symbol_length(state: &PackState, symbol: SymbolRef) -> Result<usize, TrainingError> {
    match symbol {
        SymbolRef::Byte(_) => Ok(1),
        SymbolRef::Local(local) => state
            .local_lengths
            .get(usize::try_from(local).map_err(|_| TrainingError::LengthOverflow)?)
            .copied()
            .ok_or(TrainingError::InvalidRouting),
    }
}

fn best_candidate(
    state: &PackState,
    min_frequency: u64,
) -> Result<Option<Candidate>, TrainingError> {
    let mut counts = BTreeMap::<(SymbolRef, SymbolRef), u64>::new();
    for span in &state.spans {
        for pair in span.windows(2) {
            let count = counts.entry((pair[0], pair[1])).or_default();
            *count = count
                .checked_add(1)
                .ok_or(TrainingError::FrequencyOverflow)?;
        }
    }
    let mut best = None;
    for ((left, right), frequency) in counts {
        if frequency < min_frequency {
            continue;
        }
        let length = symbol_length(state, left)?
            .checked_add(symbol_length(state, right)?)
            .ok_or(TrainingError::LengthOverflow)?;
        if length > MAX_BPE_TOKEN_BYTES {
            continue;
        }
        if best.is_none_or(|current: Candidate| {
            frequency > current.frequency
                || (frequency == current.frequency && (left, right) < (current.left, current.right))
        }) {
            best = Some(Candidate {
                left,
                right,
                frequency,
            });
        }
    }
    Ok(best)
}

fn select_global_candidate(
    states: &[PackState; 3],
    candidates: &[Option<Candidate>; 3],
) -> Option<(usize, Candidate)> {
    let mut best = None;
    for (index, candidate) in candidates.iter().copied().enumerate() {
        let Some(candidate) = candidate else {
            continue;
        };
        let Some((best_index, best_candidate)) = best else {
            best = Some((index, candidate));
            continue;
        };
        let is_better = candidate.frequency > best_candidate.frequency
            || (candidate.frequency == best_candidate.frequency
                && (states[index].pack_id, candidate.left, candidate.right)
                    < (
                        states[best_index].pack_id,
                        best_candidate.left,
                        best_candidate.right,
                    ));
        if is_better {
            best = Some((index, candidate));
        }
    }
    best
}

fn replace_pair(
    symbols: &mut Vec<SymbolRef>,
    left: SymbolRef,
    right: SymbolRef,
    result: SymbolRef,
) {
    let mut read = 0;
    let mut written = 0;
    while read < symbols.len() {
        if read + 1 < symbols.len() && symbols[read] == left && symbols[read + 1] == right {
            symbols[written] = result;
            read += 2;
        } else {
            symbols[written] = symbols[read];
            read += 1;
        }
        written += 1;
    }
    symbols.truncate(written);
}

fn factorized_metadata(
    config: FactorizedTrainingConfig,
    model: &FactorizedBpeModel,
    stats: &[FactorizedPackTrainingStats; 4],
    total_vocab_size: u32,
    provenance: &CorpusProvenance,
) -> Result<BTreeMap<String, String>, TrainingError> {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "m2.algorithm".to_owned(),
        "factorized-byte-level-bpe-v1".to_owned(),
    );
    metadata.insert(
        "m2.byte_token_count".to_owned(),
        BYTE_FALLBACK_TOKEN_COUNT.to_string(),
    );
    metadata.insert(
        "m2.router_policy_id".to_owned(),
        model.router_policy_id().to_owned(),
    );
    metadata.insert(
        "m2.target_vocab_size".to_owned(),
        config.target_vocab_size.to_string(),
    );
    metadata.insert(
        "m2.max_learned_tokens".to_owned(),
        config.max_learned_tokens.to_string(),
    );
    metadata.insert(
        "m2.min_pair_frequency".to_owned(),
        config.min_pair_frequency.to_string(),
    );
    metadata.insert(
        "m2.total_learned_tokens".to_owned(),
        model.learned_token_count().to_string(),
    );
    metadata.insert(
        "m2.total_logical_vocab_size".to_owned(),
        total_vocab_size.to_string(),
    );
    metadata.insert("m2.tie_break".to_owned(), "higher frequency, then ascending pack_id, then ascending (left SymbolRef, right SymbolRef); SymbolRef order is Byte(u8) before Local(u32)".to_owned());
    metadata.insert("m2.merge_order".to_owned(), "global merge rank order; each selected pack assigns its next contiguous local ID; only the selected pack state is updated".to_owned());
    metadata.insert("m2.normalization".to_owned(), "none".to_owned());
    metadata.insert(
        "m2.routing_boundaries".to_owned(),
        "merge only within each contiguous lexical-v1 routed span".to_owned(),
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
    metadata.insert("corpus.preprocessing".to_owned(), "raw UTF-8 bytes; no normalization; files concatenated without inserted separators before lexical-v1 routing".to_owned());
    metadata.insert(
        "corpus.total_bytes".to_owned(),
        provenance.total_bytes.to_string(),
    );
    metadata.insert(
        "experiment.train_eval_boundary".to_owned(),
        "trainer reads only the supplied training input; held-out fixture files are not opened"
            .to_owned(),
    );
    for (index, file) in provenance.files.iter().enumerate() {
        metadata.insert(format!("corpus.file.{index:06}"), file.clone());
    }
    for pack_stats in stats {
        let pack_name = if pack_stats.pack_id == DEFAULT_BYTE_FALLBACK_PACK_ID {
            BYTE_FALLBACK_PACK_NAME
        } else {
            lexical_pack_name(pack_stats.pack_id).ok_or(TrainingError::InvalidRouting)?
        };
        let prefix = format!("m2.pack.{}", pack_stats.pack_id);
        metadata.insert(format!("{prefix}.name"), pack_name.to_owned());
        metadata.insert(format!("{prefix}.spans"), pack_stats.spans.to_string());
        metadata.insert(
            format!("{prefix}.routed_bytes"),
            pack_stats.routed_bytes.to_string(),
        );
        metadata.insert(
            format!("{prefix}.learned_merges"),
            pack_stats.learned_merges.to_string(),
        );
        metadata.insert(
            format!("{prefix}.local_vocab_size"),
            pack_stats.local_vocab_size.to_string(),
        );
    }
    Ok(metadata)
}

/// Independent, deliberately slow correctness oracle for M2 routing/training/runtime.
pub mod reference {
    use super::*;
    use packtok_core::TokenId;

    /// Routes by a separately written scalar classifier and coalesces equal classes.
    pub fn route_lexical_v1(input: &str) -> Vec<RoutedSpan> {
        let mut spans = Vec::new();
        let mut current: Option<(u16, usize)> = None;
        for (offset, ch) in input.char_indices() {
            let pack = if !ch.is_ascii() || ch.is_ascii_alphabetic() {
                TEXT_PACK_ID
            } else if ch.is_ascii_digit() {
                NUMBER_PACK_ID
            } else {
                STRUCTURE_PACK_ID
            };
            match current {
                Some((known, start)) if known == pack => current = Some((known, start)),
                Some((known, start)) => {
                    spans.push(RoutedSpan {
                        pack_id: known,
                        start,
                        end: offset,
                    });
                    current = Some((pack, offset));
                }
                None => current = Some((pack, offset)),
            }
        }
        if let Some((pack_id, start)) = current {
            spans.push(RoutedSpan {
                pack_id,
                start,
                end: input.len(),
            });
        }
        spans
    }

    /// Rebuilds each pack's best candidate with linear pair-list search at every rank.
    pub fn train_model(
        corpus: &str,
        config: FactorizedTrainingConfig,
    ) -> Result<FactorizedBpeModel, TrainingError> {
        let config = config.validate()?;
        let mut states = [
            PackState::new(TEXT_PACK_ID),
            PackState::new(NUMBER_PACK_ID),
            PackState::new(STRUCTURE_PACK_ID),
        ];
        for span in route_lexical_v1(corpus) {
            let state =
                &mut states[state_index(span.pack_id).ok_or(TrainingError::InvalidRouting)?];
            state.spans.push(
                corpus.as_bytes()[span.start..span.end]
                    .iter()
                    .copied()
                    .map(SymbolRef::Byte)
                    .collect(),
            );
            state.routed_bytes +=
                u64::try_from(span.end - span.start).map_err(|_| TrainingError::LengthOverflow)?;
            state.span_count += 1;
        }
        let budget = config
            .max_learned_tokens
            .min(config.target_vocab_size - BYTE_FALLBACK_TOKEN_COUNT);
        for _ in 0..budget {
            let mut selected: Option<(usize, Candidate)> = None;
            for (index, state) in states.iter().enumerate() {
                let Some(candidate) = reference_best_candidate(state, config.min_pair_frequency)?
                else {
                    continue;
                };
                if selected.is_none_or(|(best_index, best)| {
                    candidate.frequency > best.frequency
                        || (candidate.frequency == best.frequency
                            && (state.pack_id, candidate.left, candidate.right)
                                < (states[best_index].pack_id, best.left, best.right))
                }) {
                    selected = Some((index, candidate));
                }
            }
            let Some((index, candidate)) = selected else {
                break;
            };
            let state = &mut states[index];
            let local_id =
                u32::try_from(state.merges.len()).map_err(|_| TrainingError::LengthOverflow)?;
            let length = symbol_length(state, candidate.left)?
                .checked_add(symbol_length(state, candidate.right)?)
                .ok_or(TrainingError::LengthOverflow)?;
            state.merges.push(PackBpeMerge {
                left: candidate.left,
                right: candidate.right,
            });
            state.local_lengths.push(length);
            for span in &mut state.spans {
                let mut next = Vec::with_capacity(span.len());
                let mut index = 0;
                while index < span.len() {
                    if index + 1 < span.len()
                        && span[index] == candidate.left
                        && span[index + 1] == candidate.right
                    {
                        next.push(SymbolRef::Local(local_id));
                        index += 2;
                    } else {
                        next.push(span[index]);
                        index += 1;
                    }
                }
                *span = next;
            }
        }
        let packs = states
            .into_iter()
            .filter(|state| !state.merges.is_empty())
            .map(|state| PackBpeModel::new(state.pack_id, state.merges))
            .collect::<Result<Vec<_>, _>>()?;
        FactorizedBpeModel::new(LEXICAL_V1_ROUTER_ID, packs).map_err(TrainingError::from)
    }

    fn reference_best_candidate(
        state: &PackState,
        min_frequency: u64,
    ) -> Result<Option<Candidate>, TrainingError> {
        let mut counts = Vec::<((SymbolRef, SymbolRef), u64)>::new();
        for span in &state.spans {
            for pair in span.windows(2) {
                if let Some((_, count)) = counts
                    .iter_mut()
                    .find(|((left, right), _)| *left == pair[0] && *right == pair[1])
                {
                    *count = count
                        .checked_add(1)
                        .ok_or(TrainingError::FrequencyOverflow)?;
                } else {
                    counts.push(((pair[0], pair[1]), 1));
                }
            }
        }
        let mut best = None;
        for ((left, right), frequency) in counts {
            if frequency < min_frequency {
                continue;
            }
            let length = symbol_length(state, left)?
                .checked_add(symbol_length(state, right)?)
                .ok_or(TrainingError::LengthOverflow)?;
            if length > MAX_BPE_TOKEN_BYTES {
                continue;
            }
            if best.is_none_or(|current: Candidate| {
                frequency > current.frequency
                    || (frequency == current.frequency
                        && (left, right) < (current.left, current.right))
            }) {
                best = Some(Candidate {
                    left,
                    right,
                    frequency,
                });
            }
        }
        Ok(best)
    }

    /// Applies each local merge rank with a fresh output vector, independent of runtime.
    pub fn encode(input: &str, model: &FactorizedBpeModel) -> Vec<TokenId> {
        let mut output = Vec::new();
        for span in route_lexical_v1(input) {
            let pack_model = model.pack(span.pack_id);
            let mut symbols: Vec<SymbolRef> = input.as_bytes()[span.start..span.end]
                .iter()
                .copied()
                .map(SymbolRef::Byte)
                .collect();
            if let Some(pack_model) = pack_model {
                for (rank, merge) in pack_model.merges().iter().enumerate() {
                    let local = u32::try_from(rank).expect("validated merge rank fits u32");
                    let mut next = Vec::with_capacity(symbols.len());
                    let mut index = 0;
                    while index < symbols.len() {
                        if index + 1 < symbols.len()
                            && symbols[index] == merge.left
                            && symbols[index + 1] == merge.right
                        {
                            next.push(SymbolRef::Local(local));
                            index += 2;
                        } else {
                            next.push(symbols[index]);
                            index += 1;
                        }
                    }
                    symbols = next;
                }
            }
            output.extend(symbols.into_iter().map(|symbol| match symbol {
                SymbolRef::Byte(byte) => {
                    TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, u32::from(byte))
                }
                SymbolRef::Local(local) => TokenId::new(span.pack_id, local),
            }));
        }
        output
    }
}

fn train_factorized_model_internal(
    corpus: &str,
    config: FactorizedTrainingConfig,
) -> Result<(FactorizedBpeModel, [FactorizedPackTrainingStats; 4], u32), TrainingError> {
    let config = config.validate()?;
    let router = LexicalV1Router;
    let mut states = [
        PackState::new(TEXT_PACK_ID),
        PackState::new(NUMBER_PACK_ID),
        PackState::new(STRUCTURE_PACK_ID),
    ];
    for span in router.route(corpus) {
        let state = &mut states[state_index(span.pack_id).ok_or(TrainingError::InvalidRouting)?];
        state.spans.push(
            corpus.as_bytes()[span.start..span.end]
                .iter()
                .copied()
                .map(SymbolRef::Byte)
                .collect(),
        );
        state.routed_bytes = state
            .routed_bytes
            .checked_add(
                u64::try_from(span.end - span.start).map_err(|_| TrainingError::LengthOverflow)?,
            )
            .ok_or(TrainingError::LengthOverflow)?;
        state.span_count = state
            .span_count
            .checked_add(1)
            .ok_or(TrainingError::LengthOverflow)?;
    }
    let budget = config
        .max_learned_tokens
        .min(config.target_vocab_size - BYTE_FALLBACK_TOKEN_COUNT);
    let mut candidates = [None; 3];
    for (index, state) in states.iter().enumerate() {
        candidates[index] = best_candidate(state, config.min_pair_frequency)?;
    }
    for _ in 0..budget {
        let Some((index, candidate)) = select_global_candidate(&states, &candidates) else {
            break;
        };
        let state = &mut states[index];
        let local_id =
            u32::try_from(state.merges.len()).map_err(|_| TrainingError::LengthOverflow)?;
        let length = symbol_length(state, candidate.left)?
            .checked_add(symbol_length(state, candidate.right)?)
            .ok_or(TrainingError::LengthOverflow)?;
        state.merges.push(PackBpeMerge {
            left: candidate.left,
            right: candidate.right,
        });
        state.local_lengths.push(length);
        for span in &mut state.spans {
            replace_pair(
                span,
                candidate.left,
                candidate.right,
                SymbolRef::Local(local_id),
            );
        }
        candidates[index] = best_candidate(state, config.min_pair_frequency)?;
    }
    let mut pack_stats = [
        FactorizedPackTrainingStats {
            pack_id: DEFAULT_BYTE_FALLBACK_PACK_ID,
            spans: 0,
            routed_bytes: 0,
            learned_merges: 0,
            local_vocab_size: 0,
        },
        FactorizedPackTrainingStats {
            pack_id: TEXT_PACK_ID,
            spans: 0,
            routed_bytes: 0,
            learned_merges: 0,
            local_vocab_size: 0,
        },
        FactorizedPackTrainingStats {
            pack_id: NUMBER_PACK_ID,
            spans: 0,
            routed_bytes: 0,
            learned_merges: 0,
            local_vocab_size: 0,
        },
        FactorizedPackTrainingStats {
            pack_id: STRUCTURE_PACK_ID,
            spans: 0,
            routed_bytes: 0,
            learned_merges: 0,
            local_vocab_size: 0,
        },
    ];
    for (index, state) in states.iter().enumerate() {
        let merge_count =
            u32::try_from(state.merges.len()).map_err(|_| TrainingError::LengthOverflow)?;
        pack_stats[index + 1] = FactorizedPackTrainingStats {
            pack_id: state.pack_id,
            spans: state.span_count,
            routed_bytes: state.routed_bytes,
            learned_merges: merge_count,
            local_vocab_size: merge_count,
        };
    }
    let packs = states
        .into_iter()
        .filter(|state| !state.merges.is_empty())
        .map(|state| PackBpeModel::new(state.pack_id, state.merges))
        .collect::<Result<Vec<_>, _>>()?;
    let model = FactorizedBpeModel::new(LEXICAL_V1_ROUTER_ID, packs)?;
    let learned =
        u32::try_from(model.learned_token_count()).map_err(|_| TrainingError::LengthOverflow)?;
    let total_vocab = BYTE_FALLBACK_TOKEN_COUNT
        .checked_add(learned)
        .ok_or(TrainingError::LengthOverflow)?;
    Ok((model, pack_stats, total_vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_core::TokenId;
    use packtok_tokenizer::{FactorizedTokenizer, Tokenizer};

    fn assert_models_equal(corpus: &str, config: FactorizedTrainingConfig) {
        let production = train_factorized_model(corpus, config).expect("production training");
        let reference = reference::train_model(corpus, config).expect("reference training");
        assert_eq!(production, reference, "corpus {corpus:?}");
    }

    fn train_factorized_model(
        corpus: &str,
        config: FactorizedTrainingConfig,
    ) -> Result<FactorizedBpeModel, TrainingError> {
        train_factorized_model_internal(corpus, config).map(|(model, _, _)| model)
    }

    #[test]
    fn config_defaults_and_validation_preserve_shared_byte_budget() {
        let config = FactorizedTrainingConfig::default();
        assert_eq!(config.target_vocab_size, 512);
        assert_eq!(config.max_learned_tokens, 256);
        assert_eq!(config.validate().expect("valid config"), config);
        assert!(
            FactorizedTrainingConfig {
                target_vocab_size: 255,
                ..config
            }
            .validate()
            .is_err()
        );
        assert!(
            FactorizedTrainingConfig {
                target_vocab_size: 256,
                max_learned_tokens: 1,
                ..config
            }
            .validate()
            .is_err()
        );
        assert!(
            FactorizedTrainingConfig {
                min_pair_frequency: 0,
                ..config
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn global_candidate_ties_use_pack_id_then_symbol_refs() {
        let config = FactorizedTrainingConfig {
            target_vocab_size: 260,
            max_learned_tokens: 4,
            min_pair_frequency: 1,
        };
        let model = reference::train_model("ab12", config).expect("reference train");
        assert_eq!(
            model.pack(TEXT_PACK_ID).expect("text pack").merges()[0],
            PackBpeMerge {
                left: SymbolRef::Byte(b'a'),
                right: SymbolRef::Byte(b'b'),
            }
        );
        assert_eq!(
            model.pack(NUMBER_PACK_ID).expect("number pack").merges()[0],
            PackBpeMerge {
                left: SymbolRef::Byte(b'1'),
                right: SymbolRef::Byte(b'2'),
            }
        );
    }

    #[test]
    fn pack_local_ids_overlap_without_aliasing_and_byte_fallback_is_shared() {
        let config = FactorizedTrainingConfig {
            target_vocab_size: 258,
            max_learned_tokens: 2,
            min_pair_frequency: 1,
        };
        let artifact = train_factorized_bpe(b"abab 1212", config).expect("train");
        let tokenizer = FactorizedTokenizer::from_artifact(&artifact).expect("runtime");
        let tokens = tokenizer.encode("abab 1212!").expect("encode");
        let text_id = TokenId::new(TEXT_PACK_ID, 0);
        let number_id = TokenId::new(NUMBER_PACK_ID, 0);
        assert_ne!(text_id, number_id);
        assert!(tokens.contains(&text_id));
        assert!(tokens.contains(&number_id));
        assert!(
            tokens
                .iter()
                .any(|token| token.pack == DEFAULT_BYTE_FALLBACK_PACK_ID)
        );
        assert_eq!(
            tokenizer.decode_bytes(&tokens).expect("decode"),
            b"abab 1212!"
        );
    }

    #[test]
    fn routing_boundaries_prevent_cross_pack_merges() {
        let config = FactorizedTrainingConfig {
            target_vocab_size: 512,
            max_learned_tokens: 256,
            min_pair_frequency: 1,
        };
        let artifact = train_factorized_bpe(&b"hello123 hello123"[..], config).expect("train");
        let model = artifact.factorized_bpe().expect("model");
        let text = model.pack(TEXT_PACK_ID).expect("text merges");
        let number = model.pack(NUMBER_PACK_ID).expect("number merges");
        assert!(!text.merges().iter().chain(number.merges()).any(|merge| {
            matches!(merge.left, SymbolRef::Byte(b'o'))
                && matches!(merge.right, SymbolRef::Byte(b'1'))
        }));
        let tokenizer = FactorizedTokenizer::from_artifact(&artifact).expect("runtime");
        let tokens = tokenizer.encode("hello123").expect("encode");
        assert_eq!(
            tokenizer.decode_bytes(&tokens).expect("decode"),
            b"hello123"
        );
    }

    #[test]
    fn production_and_reference_match_on_deterministic_unicode_inputs() {
        let config = FactorizedTrainingConfig {
            target_vocab_size: 272,
            max_learned_tokens: 16,
            min_pair_frequency: 1,
        };
        for corpus in [
            "abab 1212!!",
            "Sämtliche Häuser 2026!",
            "漢字🙂漢字\n\t",
            "a\0a 11",
        ] {
            assert_models_equal(corpus, config);
        }
        let mut seed = 0x7a31_094d_u32;
        let alphabet = ['a', 'b', ' ', '3', '!', 'ä', '漢', '🙂'];
        for case in 0..96 {
            let length = (case * 11) % 61;
            let mut input = String::new();
            for _ in 0..length {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                input.push(alphabet[(seed as usize) % alphabet.len()]);
            }
            assert_models_equal(&input, config);
        }
    }

    #[test]
    fn runtime_ids_match_reference_and_round_trip_exactly() {
        let config = FactorizedTrainingConfig {
            target_vocab_size: 280,
            max_learned_tokens: 24,
            min_pair_frequency: 1,
        };
        let artifact = train_factorized_bpe(
            "abab 1212!! Sämtliche Häuser äöüß 👩🏽‍💻 e\u{301} 漢字\n".as_bytes(),
            config,
        )
        .expect("train artifact");
        let model = artifact.factorized_bpe().expect("factorized model");
        let tokenizer = FactorizedTokenizer::from_artifact(&artifact).expect("runtime");
        let mut seed = 0x94e2_01ab_u32;
        let alphabet = ['a', 'b', ' ', '2', '!', 'ö', '漢', '🙂', '\n'];
        for case in 0..128 {
            let length = (case * 7) % 53;
            let mut input = String::new();
            for _ in 0..length {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                input.push(alphabet[(seed as usize) % alphabet.len()]);
            }
            let runtime = tokenizer.encode(&input).expect("runtime encode");
            assert_eq!(runtime, reference::encode(&input, model));
            assert_eq!(
                tokenizer.decode_bytes(&runtime).expect("decode"),
                input.as_bytes()
            );
        }
    }

    #[test]
    fn artifact_training_twice_is_byte_identical_and_records_pack_stats() {
        let corpus = include_bytes!("../../../fixtures/benchmark/train.txt");
        let report_a =
            train_factorized_bpe_with_report(corpus, FactorizedTrainingConfig::default())
                .expect("first training");
        let report_b =
            train_factorized_bpe_with_report(corpus, FactorizedTrainingConfig::default())
                .expect("second training");
        assert_eq!(
            report_a.artifact.to_bytes().expect("first artifact"),
            report_b.artifact.to_bytes().expect("second artifact")
        );
        for stats in report_a.packs {
            let prefix = format!("m2.pack.{}", stats.pack_id);
            assert_eq!(
                report_a.artifact.metadata()[&format!("{prefix}.learned_merges")],
                stats.learned_merges.to_string()
            );
        }
    }

    #[test]
    fn reference_routing_agrees_with_shared_router_and_preserves_char_boundaries() {
        let input = "Aé 12! 漢🙂\0";
        assert_eq!(
            reference::route_lexical_v1(input),
            LexicalV1Router.route(input)
        );
    }

    #[test]
    fn malformed_non_utf8_corpus_is_rejected_without_normalization() {
        assert!(matches!(
            train_factorized_bpe(&[b'a', 0xff], FactorizedTrainingConfig::default()),
            Err(TrainingError::InvalidUtf8Corpus { valid_up_to: 1 })
        ));
        let artifact = train_factorized_bpe(
            "e\u{301}".as_bytes(),
            FactorizedTrainingConfig {
                target_vocab_size: 257,
                max_learned_tokens: 1,
                min_pair_frequency: 1,
            },
        )
        .expect("combining sequence remains raw UTF-8");
        assert_eq!(artifact.metadata()["m2.normalization"], "none");
    }
}
