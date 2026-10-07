use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

use packtok_core::{LocalTokenId, TokenId};
use packtok_format::{Artifact, BYTE_TOKEN_COUNT, FLAT_BPE_PACK_ID, FlatBpeModel};

use crate::{DecodeError, EncodeError, Tokenizer};

/// Runtime encoder and decoder for one immutable flat byte-level BPE model.
///
/// Encoding applies merges in ascending stored rank. It never retrains, normalizes,
/// consults corpus metadata, or depends on training-only code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BpeTokenizer {
    model: FlatBpeModel,
    // Rows indexed by left ID; each row is sorted by right ID.
    pair_offsets: Vec<usize>,
    pairs: Vec<(u32, u32)>,
}

struct Symbol {
    local: u32,
    previous: usize,
    next: usize,
}

const REMOVED: u32 = u32::MAX;
const END: usize = usize::MAX;

fn is_uniform(input: &[u8]) -> bool {
    let block = [input[0]; 32];
    let mut chunks = input.chunks_exact(block.len());
    chunks.all(|chunk| chunk == block) && chunks.remainder().iter().all(|byte| *byte == block[0])
}

// Dense short-period inputs are expensive heap workloads even when an occasional
// byte differs. Choose a repeated pattern from three spaced, aligned samples,
// then verify it over the complete input. This only selects an execution path;
// both paths apply exactly the same rank-ordered merges.
fn has_dense_repetitions(input: &[u8]) -> bool {
    if input.len() < 96 {
        return false;
    }
    'periods: for period in 1..=16 {
        let count = input.len() / period;
        let first = &input[..period];
        let middle = &input[(count / 2) * period..(count / 2 + 1) * period];
        let last = &input[(count - 1) * period..count * period];
        let pattern = if first == middle || first == last {
            first
        } else if middle == last {
            middle
        } else {
            continue;
        };
        // Compare whole batches aligned to the period, including periods such as
        // three bytes. A comparison per tiny pattern adds substantial overhead.
        let block_len = (32 / period) * period;
        let mut block = [0_u8; 32];
        for (index, byte) in block[..block_len].iter_mut().enumerate() {
            *byte = pattern[index % period];
        }
        let allowance = input.len().div_ceil(block_len).div_ceil(32);
        let mut mismatches = 0;
        for chunk in input.chunks(block_len) {
            if chunk != &block[..chunk.len()] {
                mismatches += 1;
                if mismatches > allowance {
                    continue 'periods;
                }
            }
        }
        return true;
    }
    false
}

/// Capacity-based buffer measurements for one successful runtime call.
/// Excludes the immutable model, caller input, allocator overhead and process RSS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BpeBufferStats {
    /// Successful allocation/reallocation requests that grew a vector's capacity.
    pub allocation_requests: usize,
    /// Largest simultaneous capacity in bytes of runtime-owned working/output vectors.
    pub peak_buffer_bytes: usize,
}

impl BpeBufferStats {
    fn observe<const TRACK: bool>(&mut self, before: usize, after: usize, bytes: usize) {
        if TRACK {
            self.allocation_requests += usize::from(after > before);
            self.peak_buffer_bytes = self.peak_buffer_bytes.max(bytes);
        }
    }
}

impl BpeTokenizer {
    /// Creates a runtime tokenizer from a structurally validated merge model.
    #[must_use]
    pub fn new(model: FlatBpeModel) -> Self {
        let mut ordered = model.merges().to_vec();
        ordered.sort_unstable_by_key(|merge| (merge.left, merge.right));
        let mut pair_offsets = vec![0; model.vocabulary_size() as usize + 1];
        for merge in &ordered {
            pair_offsets[merge.left as usize + 1] += 1;
        }
        for index in 1..pair_offsets.len() {
            pair_offsets[index] += pair_offsets[index - 1];
        }
        let pairs = ordered
            .iter()
            .map(|merge| (merge.right, merge.result))
            .collect();
        Self {
            model,
            pair_offsets,
            pairs,
        }
    }

    /// Loads the runtime model from a validated M1 artifact.
    pub fn from_artifact(artifact: &Artifact) -> Result<Self, TokenizerModelError> {
        artifact
            .flat_bpe()
            .cloned()
            .map(Self::new)
            .ok_or(TokenizerModelError::MissingFlatBpeModel)
    }

    /// Returns the model's flat vocabulary size, including the 256 byte tokens.
    #[must_use]
    pub fn vocabulary_size(&self) -> u32 {
        self.model.vocabulary_size()
    }

    /// Returns the validated model's rank-ordered merges.
    #[must_use]
    pub fn merges(&self) -> &[packtok_format::FlatBpeMerge] {
        self.model.merges()
    }

    /// Encodes arbitrary bytes. Initial byte IDs guarantee that no input is unknown.
    pub fn encode_bytes(&self, input: &[u8]) -> Result<Vec<TokenId>, EncodeError> {
        self.encode_measured::<false>(input, &mut BpeBufferStats::default())
    }

    /// Encodes with vector allocation and capacity counters outside the timed path.
    pub fn encode_bytes_with_stats(
        &self,
        input: &[u8],
    ) -> Result<(Vec<TokenId>, BpeBufferStats), EncodeError> {
        let mut stats = BpeBufferStats::default();
        let tokens = self.encode_measured::<true>(input, &mut stats)?;
        Ok((tokens, stats))
    }

    fn encode_measured<const TRACK: bool>(
        &self,
        input: &[u8],
        stats: &mut BpeBufferStats,
    ) -> Result<Vec<TokenId>, EncodeError> {
        // Without any initial byte-pair match, no learned token can be formed.
        if self.model.merges().is_empty()
            || !input.windows(2).any(|pair| {
                self.pair_result(u32::from(pair[0]), u32::from(pair[1]))
                    .is_some()
            })
        {
            return self.encode_base_bytes::<TRACK>(input, stats);
        }
        // For tiny models, contiguous passes cost less than event bookkeeping,
        // especially when an early merge compresses a dense run of repetitions.
        if self.model.merges().len() <= 8 || is_uniform(input) || has_dense_repetitions(input) {
            return self.encode_small_model::<TRACK>(input, stats);
        }
        let allocation_error = || EncodeError::AllocationFailed {
            requested_tokens: input.len(),
        };
        let mut symbols = Vec::new();
        symbols
            .try_reserve_exact(input.len())
            .map_err(|_| allocation_error())?;
        let symbol_bytes = symbols.capacity() * std::mem::size_of::<Symbol>();
        stats.observe::<TRACK>(0, symbols.capacity(), symbol_bytes);
        symbols.extend(input.iter().enumerate().map(|(index, byte)| Symbol {
            local: u32::from(*byte),
            previous: if index == 0 { END } else { index - 1 },
            next: if index + 1 == input.len() {
                END
            } else {
                index + 1
            },
        }));
        let mut pending = BinaryHeap::new();
        pending
            .try_reserve(input.len())
            .map_err(|_| allocation_error())?;
        stats.observe::<TRACK>(
            0,
            pending.capacity(),
            symbol_bytes + pending.capacity() * std::mem::size_of::<Reverse<(u32, usize)>>(),
        );
        for index in 0..input.len() - 1 {
            self.schedule_pair::<TRACK>(
                &symbols,
                index,
                &mut pending,
                input.len(),
                symbol_bytes,
                stats,
            )?;
        }

        let mut token_count = input.len();
        // Result IDs order events by rank, then original byte position. A new
        // adjacency includes the newly created ID, so it can only have a later
        // rank. This preserves whole-rank, left-to-right replacement semantics.
        while let Some(Reverse((result, left))) = pending.pop() {
            let right = symbols[left].next;
            let merge = &self.model.merges()[(result - BYTE_TOKEN_COUNT) as usize];
            if symbols[left].local != merge.left
                || right == END
                || symbols[right].local != merge.right
            {
                continue;
            }
            let previous = symbols[left].previous;
            let next = symbols[right].next;
            symbols[left].local = result;
            symbols[left].next = next;
            symbols[right].local = REMOVED;
            if next != END {
                symbols[next].previous = left;
            }
            token_count -= 1;
            if previous != END {
                self.schedule_pair::<TRACK>(
                    &symbols,
                    previous,
                    &mut pending,
                    input.len(),
                    symbol_bytes,
                    stats,
                )?;
            }
            self.schedule_pair::<TRACK>(
                &symbols,
                left,
                &mut pending,
                input.len(),
                symbol_bytes,
                stats,
            )?;
        }
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(token_count)
            .map_err(|_| allocation_error())?;
        stats.observe::<TRACK>(
            0,
            tokens.capacity(),
            symbol_bytes
                + pending.capacity() * std::mem::size_of::<Reverse<(u32, usize)>>()
                + tokens.capacity() * std::mem::size_of::<TokenId>(),
        );
        tokens.extend(
            symbols
                .into_iter()
                .filter(|symbol| symbol.local != REMOVED)
                .map(|symbol| TokenId::new(FLAT_BPE_PACK_ID, symbol.local)),
        );
        Ok(tokens)
    }

    fn encode_base_bytes<const TRACK: bool>(
        &self,
        input: &[u8],
        stats: &mut BpeBufferStats,
    ) -> Result<Vec<TokenId>, EncodeError> {
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(input.len())
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: input.len(),
            })?;
        stats.observe::<TRACK>(
            0,
            tokens.capacity(),
            tokens.capacity() * std::mem::size_of::<TokenId>(),
        );
        tokens.extend(
            input
                .iter()
                .map(|byte| TokenId::new(FLAT_BPE_PACK_ID, u32::from(*byte))),
        );
        Ok(tokens)
    }

    fn encode_small_model<const TRACK: bool>(
        &self,
        input: &[u8],
        stats: &mut BpeBufferStats,
    ) -> Result<Vec<TokenId>, EncodeError> {
        let mut symbols = Vec::new();
        let use_symbols = input.len() > 1 && !self.model.merges().is_empty();
        if use_symbols {
            symbols
                .try_reserve_exact(input.len())
                .map_err(|_| EncodeError::AllocationFailed {
                    requested_tokens: input.len(),
                })?;
            stats.observe::<TRACK>(
                0,
                symbols.capacity(),
                symbols.capacity() * std::mem::size_of::<u32>(),
            );
            symbols.extend(input.iter().map(|byte| u32::from(*byte)));
            for merge in self.model.merges() {
                if symbols.len() < 2 {
                    break;
                }
                let mut read = 0;
                let mut written = 0;
                while read < symbols.len() {
                    if read + 1 < symbols.len()
                        && symbols[read] == merge.left
                        && symbols[read + 1] == merge.right
                    {
                        symbols[written] = merge.result;
                        read += 2;
                    } else {
                        symbols[written] = symbols[read];
                        read += 1;
                    }
                    written += 1;
                }
                symbols.truncate(written);
            }
        }
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(if use_symbols {
                symbols.len()
            } else {
                input.len()
            })
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: input.len(),
            })?;
        stats.observe::<TRACK>(
            0,
            tokens.capacity(),
            tokens.capacity() * std::mem::size_of::<TokenId>()
                + symbols.capacity() * std::mem::size_of::<u32>(),
        );
        if use_symbols {
            tokens.extend(
                symbols
                    .into_iter()
                    .map(|local| TokenId::new(FLAT_BPE_PACK_ID, local)),
            );
        } else {
            tokens.extend(
                input
                    .iter()
                    .map(|byte| TokenId::new(FLAT_BPE_PACK_ID, u32::from(*byte))),
            );
        }
        Ok(tokens)
    }

    fn schedule_pair<const TRACK: bool>(
        &self,
        symbols: &[Symbol],
        left: usize,
        pending: &mut BinaryHeap<Reverse<(u32, usize)>>,
        input_len: usize,
        symbol_bytes: usize,
        stats: &mut BpeBufferStats,
    ) -> Result<(), EncodeError> {
        let right = symbols[left].next;
        if right == END {
            return Ok(());
        }
        if let Some(result) = self.pair_result(symbols[left].local, symbols[right].local) {
            let before = pending.capacity();
            pending
                .try_reserve(1)
                .map_err(|_| EncodeError::AllocationFailed {
                    requested_tokens: input_len,
                })?;
            stats.observe::<TRACK>(
                before,
                pending.capacity(),
                symbol_bytes + pending.capacity() * std::mem::size_of::<Reverse<(u32, usize)>>(),
            );
            pending.push(Reverse((result, left)));
        }
        Ok(())
    }

    fn pair_result(&self, left: u32, right: u32) -> Option<u32> {
        let local = left as usize;
        let row = &self.pairs[self.pair_offsets[local]..self.pair_offsets[local + 1]];
        let index = row.binary_search_by_key(&right, |&(right, _)| right).ok()?;
        Some(row[index].1)
    }

    /// Expands flat token IDs to exact bytes without requiring valid UTF-8.
    pub fn decode_bpe_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError> {
        self.decode_measured::<false>(tokens, &mut BpeBufferStats::default())
    }

    /// Decodes with vector allocation and capacity counters outside the timed path.
    pub fn decode_bytes_with_stats(
        &self,
        tokens: &[TokenId],
    ) -> Result<(Vec<u8>, BpeBufferStats), DecodeError> {
        let mut stats = BpeBufferStats::default();
        let bytes = self.decode_measured::<true>(tokens, &mut stats)?;
        Ok((bytes, stats))
    }

    fn decode_measured<const TRACK: bool>(
        &self,
        tokens: &[TokenId],
        stats: &mut BpeBufferStats,
    ) -> Result<Vec<u8>, DecodeError> {
        let mut output_len = 0_usize;
        for (index, token) in tokens.iter().enumerate() {
            if token.pack != FLAT_BPE_PACK_ID {
                return Err(DecodeError::UnexpectedPack {
                    index,
                    expected: FLAT_BPE_PACK_ID,
                    actual: token.pack,
                });
            }
            let length = self.model.byte_length(token.local).ok_or(
                DecodeError::LocalIdOutsideVocabulary {
                    index,
                    actual: token.local,
                    vocabulary_size: self.model.vocabulary_size(),
                },
            )?;
            output_len = output_len
                .checked_add(length)
                .ok_or(DecodeError::AllocationFailed {
                    requested_bytes: usize::MAX,
                })?;
        }

        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| DecodeError::AllocationFailed {
                requested_bytes: output_len,
            })?;
        stats.observe::<TRACK>(0, output.capacity(), output.capacity());
        let mut stack = Vec::<LocalTokenId>::new();

        for (token_index, token) in tokens.iter().enumerate() {
            if token.local < BYTE_TOKEN_COUNT {
                output.push(token.local as u8);
                continue;
            }
            stack.clear();
            let before = stack.capacity();
            stack
                .try_reserve(1)
                .map_err(|_| DecodeError::AllocationFailed { requested_bytes: 4 })?;
            stats.observe::<TRACK>(
                before,
                stack.capacity(),
                output.capacity() + stack.capacity() * std::mem::size_of::<LocalTokenId>(),
            );
            stack.push(token.local);
            while let Some(local) = stack.pop() {
                if local < BYTE_TOKEN_COUNT {
                    let byte =
                        u8::try_from(local).map_err(|_| DecodeError::LocalIdOutsideVocabulary {
                            index: token_index,
                            actual: local,
                            vocabulary_size: self.model.vocabulary_size(),
                        })?;
                    output.push(byte);
                    continue;
                }

                let rank = usize::try_from(local - BYTE_TOKEN_COUNT).map_err(|_| {
                    DecodeError::LocalIdOutsideVocabulary {
                        index: token_index,
                        actual: local,
                        vocabulary_size: self.model.vocabulary_size(),
                    }
                })?;
                let merge =
                    self.model
                        .merges()
                        .get(rank)
                        .ok_or(DecodeError::LocalIdOutsideVocabulary {
                            index: token_index,
                            actual: local,
                            vocabulary_size: self.model.vocabulary_size(),
                        })?;
                let before = stack.capacity();
                stack
                    .try_reserve(2)
                    .map_err(|_| DecodeError::AllocationFailed { requested_bytes: 8 })?;
                stats.observe::<TRACK>(
                    before,
                    stack.capacity(),
                    output.capacity() + stack.capacity() * std::mem::size_of::<LocalTokenId>(),
                );
                stack.push(merge.right);
                stack.push(merge.left);
            }
        }
        debug_assert_eq!(output.len(), output_len);
        Ok(output)
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, input: &str) -> Result<Vec<TokenId>, EncodeError> {
        self.encode_bytes(input.as_bytes())
    }

    fn decode_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError> {
        self.decode_bpe_bytes(tokens)
    }
}

/// Error while selecting a runtime tokenizer from an artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizerModelError {
    /// The artifact is a byte-fallback-only M0 artifact and has no BPE section.
    MissingFlatBpeModel,
}

impl fmt::Display for TokenizerModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFlatBpeModel => f.write_str("artifact does not contain a flat BPE model"),
        }
    }
}

impl std::error::Error for TokenizerModelError {}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_format::FlatBpeMerge;

    #[test]
    fn uniform_detection_checks_block_boundaries_and_remainder() {
        for length in [2, 31, 32, 33, 63, 64, 65, 1025] {
            let input = vec![0xff; length];
            assert!(is_uniform(&input));
            for index in [0, length / 2, length - 1] {
                let mut mixed = input.clone();
                mixed[index] = 0;
                assert!(!is_uniform(&mixed), "length={length}, index={index}");
            }
        }
    }

    #[test]
    fn dense_repetition_detection_handles_periods_and_exception_positions() {
        for period in 1..=16 {
            let pattern: Vec<_> = (0..period).map(|index| b'a' + index as u8).collect();
            let mut input = pattern.repeat(4096_usize.div_ceil(period));
            input.truncate(4096);
            assert!(has_dense_repetitions(&input), "period={period}");
            for index in [0, 1, 31, 2047, 2048, 4095] {
                let mut mixed = input.clone();
                mixed[index] = 0xff;
                assert!(
                    has_dense_repetitions(&mixed),
                    "period={period}, index={index}"
                );
            }
        }
        let sparse: Vec<_> = (0..4096).map(|index| (index % 251) as u8).collect();
        assert!(!has_dense_repetitions(&sparse));
        let mut prefix = vec![b'a'; 1024];
        prefix.extend_from_slice(&sparse);
        assert!(!has_dense_repetitions(&prefix));
    }

    #[test]
    fn measured_paths_preserve_results_and_count_owned_buffers() {
        let small = FlatBpeModel::new(vec![FlatBpeMerge {
            left: 97,
            right: 97,
            result: 256,
        }])
        .expect("small model");
        let mut merges = small.merges().to_vec();
        for rank in 1..12 {
            merges.push(FlatBpeMerge {
                left: rank,
                right: rank + 1,
                result: 256 + rank,
            });
        }
        let large = FlatBpeModel::new(merges).expect("event model");
        for (model, expected_allocations) in [(small, 2), (large, 3)] {
            let runtime = BpeTokenizer::new(model);
            let (tokens, stats) = runtime
                .encode_bytes_with_stats(b"aaaaab")
                .expect("measured encode");
            assert_eq!(runtime.encode_bytes(b"aaaaab").expect("encode"), tokens);
            assert_eq!(stats.allocation_requests, expected_allocations);
            assert!(stats.peak_buffer_bytes >= tokens.capacity() * std::mem::size_of::<TokenId>());
            let (bytes, decode_stats) = runtime
                .decode_bytes_with_stats(&tokens)
                .expect("measured decode");
            assert_eq!(runtime.decode_bpe_bytes(&tokens).expect("decode"), bytes);
            assert_eq!(bytes, b"aaaaab");
            assert_eq!(decode_stats.allocation_requests, 2);
            assert!(decode_stats.peak_buffer_bytes >= bytes.capacity());
            assert_eq!(
                runtime
                    .encode_bytes_with_stats(b"")
                    .expect("empty encode")
                    .1,
                BpeBufferStats::default()
            );
            assert_eq!(
                runtime
                    .decode_bytes_with_stats(&[])
                    .expect("empty decode")
                    .1,
                BpeBufferStats::default()
            );
            let (_, raw_stats) = runtime
                .decode_bytes_with_stats(&[TokenId::new(0, 255)])
                .expect("raw byte");
            assert_eq!(raw_stats.allocation_requests, 1);
            let (unknown, fallback_stats) = runtime
                .encode_bytes_with_stats(b"xyz")
                .expect("unmatched bytes");
            assert_eq!(
                unknown,
                [
                    TokenId::new(0, 120),
                    TokenId::new(0, 121),
                    TokenId::new(0, 122)
                ]
            );
            assert_eq!(fallback_stats.allocation_requests, 1);
        }
    }

    #[test]
    fn event_decoder_rejects_invalid_ids_and_handles_deep_merge_chains() {
        let mut merges = Vec::new();
        for rank in 0..1024 {
            merges.push(FlatBpeMerge {
                left: if rank == 0 { 97 } else { 255 + rank },
                right: 97,
                result: 256 + rank,
            });
        }
        let runtime = BpeTokenizer::new(FlatBpeModel::new(merges).expect("deep model"));
        let tokens = [TokenId::new(0, 1279)];
        assert_eq!(
            runtime.decode_bpe_bytes(&tokens).expect("iterative decode"),
            vec![b'a'; 1025]
        );
        assert!(matches!(
            runtime.decode_bpe_bytes(&[TokenId::new(0, u32::MAX)]),
            Err(DecodeError::LocalIdOutsideVocabulary { .. })
        ));
        assert!(matches!(
            runtime.decode_bpe_bytes(&[TokenId::new(1, 97)]),
            Err(DecodeError::UnexpectedPack { .. })
        ));
    }
}
