use std::fmt;
use std::mem::size_of;

use packtok_core::{LocalTokenId, PackId, TokenId};
use packtok_format::{Artifact, FactorizedBpeModel, SymbolRef};
use packtok_packs::{
    LEXICAL_V1_ROUTER_ID, LexicalV1Router, NUMBER_PACK_ID, PackRouter, RoutedSpan,
    STRUCTURE_PACK_ID, TEXT_PACK_ID, lexical_pack_name,
};

use crate::{DecodeError, EncodeError, Tokenizer};

/// Runtime observations for one output pack.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FactorizedPackStats {
    /// Stable pack ID represented by this row.
    pub pack_id: PackId,
    /// Input bytes assigned to this pack by routing.
    pub routed_bytes: u64,
    /// Number of contiguous routed spans assigned to this pack.
    pub spans: u64,
    /// Number of emitted output tokens from this pack.
    pub tokens: u64,
    /// Bytes represented by this pack's emitted tokens.
    pub token_bytes: u64,
}

/// Allocation observations for one encoding call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FactorizedEncodeStats {
    /// Byte fallback, TEXT, NUMBER, and STRUCTURE rows in that order.
    pub packs: [FactorizedPackStats; 4],
    /// Adjacent output-token pairs whose pack IDs differ.
    pub pack_transitions: u64,
    /// Capacity of the returned `Vec<TokenId>` in bytes.
    pub output_capacity_bytes: usize,
    /// Peak capacity of router-span and working-symbol vectors in bytes.
    pub temporary_peak_bytes: usize,
}

impl FactorizedEncodeStats {
    fn new(fallback_pack_id: PackId) -> Self {
        Self {
            packs: [
                FactorizedPackStats {
                    pack_id: fallback_pack_id,
                    ..FactorizedPackStats::default()
                },
                FactorizedPackStats {
                    pack_id: TEXT_PACK_ID,
                    ..FactorizedPackStats::default()
                },
                FactorizedPackStats {
                    pack_id: NUMBER_PACK_ID,
                    ..FactorizedPackStats::default()
                },
                FactorizedPackStats {
                    pack_id: STRUCTURE_PACK_ID,
                    ..FactorizedPackStats::default()
                },
            ],
            pack_transitions: 0,
            output_capacity_bytes: 0,
            temporary_peak_bytes: 0,
        }
    }

    /// Finds a statistic row by pack ID.
    #[must_use]
    pub fn for_pack(&self, pack_id: PackId) -> Option<&FactorizedPackStats> {
        self.packs.iter().find(|stats| stats.pack_id == pack_id)
    }
}

/// Immutable runtime for an artifact with a single shared byte fallback pack.
#[derive(Clone, Debug)]
pub struct FactorizedTokenizer<R = LexicalV1Router> {
    model: FactorizedBpeModel,
    fallback_pack_id: PackId,
    router: R,
}

impl<R: PackRouter> FactorizedTokenizer<R> {
    /// Creates a runtime after requiring the router ID to match the artifact.
    pub fn new(
        model: FactorizedBpeModel,
        fallback_pack_id: PackId,
        router: R,
    ) -> Result<Self, FactorizedTokenizerError> {
        if model.router_policy_id() != router.policy_id() {
            return Err(FactorizedTokenizerError::RouterPolicyMismatch {
                artifact: model.router_policy_id().to_owned(),
                runtime: router.policy_id().to_owned(),
            });
        }
        if fallback_pack_id == TEXT_PACK_ID
            || fallback_pack_id == NUMBER_PACK_ID
            || fallback_pack_id == STRUCTURE_PACK_ID
        {
            return Err(FactorizedTokenizerError::InvalidFallbackPackId {
                pack_id: fallback_pack_id,
            });
        }
        Ok(Self {
            model,
            fallback_pack_id,
            router,
        })
    }

    /// Returns the immutable normative merge model.
    #[must_use]
    pub fn model(&self) -> &FactorizedBpeModel {
        &self.model
    }

    /// Returns the shared byte fallback pack ID.
    #[must_use]
    pub const fn fallback_pack_id(&self) -> PackId {
        self.fallback_pack_id
    }

    /// Encodes arbitrary bytes; invalid UTF-8 uses only the shared byte fallback.
    pub fn encode_bytes(&self, input: &[u8]) -> Result<Vec<TokenId>, EncodeError> {
        match std::str::from_utf8(input) {
            Ok(text) => self.encode(text),
            Err(_) => {
                let mut tokens = Vec::new();
                tokens.try_reserve_exact(input.len()).map_err(|_| {
                    EncodeError::AllocationFailed {
                        requested_tokens: input.len(),
                    }
                })?;
                tokens.extend(
                    input
                        .iter()
                        .map(|byte| TokenId::new(self.fallback_pack_id, LocalTokenId::from(*byte))),
                );
                Ok(tokens)
            }
        }
    }

    /// Encodes and reports routed bytes, pack output counts, and vector capacities.
    pub fn encode_with_stats(
        &self,
        input: &str,
    ) -> Result<(Vec<TokenId>, FactorizedEncodeStats), EncodeError> {
        self.encode_impl::<true>(input)
    }

    fn encode_impl<const TRACK: bool>(
        &self,
        input: &str,
    ) -> Result<(Vec<TokenId>, FactorizedEncodeStats), EncodeError> {
        let spans = self.router.route(input);
        validate_spans(input, &spans)?;

        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(input.len())
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: input.len(),
            })?;
        let mut symbols = Vec::new();
        symbols
            .try_reserve_exact(input.len())
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: input.len(),
            })?;
        let mut stats = FactorizedEncodeStats::new(self.fallback_pack_id);

        for (span_index, span) in spans.iter().copied().enumerate() {
            let input_bytes = &input.as_bytes()[span.start..span.end];
            let pack_stats = stats_index(&mut stats, span.pack_id)
                .ok_or(EncodeError::InvalidRouterOutput { span_index })?;
            if TRACK {
                pack_stats.routed_bytes = pack_stats
                    .routed_bytes
                    .saturating_add(u64::try_from(input_bytes.len()).unwrap_or(u64::MAX));
                pack_stats.spans = pack_stats.spans.saturating_add(1);
            }

            symbols.clear();
            symbols.extend(input_bytes.iter().copied().map(SymbolRef::Byte));
            if let Some(pack_model) = self.model.pack(span.pack_id) {
                for (rank, merge) in pack_model.merges().iter().enumerate() {
                    let local = u32::try_from(rank).map_err(|_| EncodeError::AllocationFailed {
                        requested_tokens: input.len(),
                    })?;
                    let mut read = 0;
                    let mut written = 0;
                    while read < symbols.len() {
                        if read + 1 < symbols.len()
                            && symbols[read] == merge.left
                            && symbols[read + 1] == merge.right
                        {
                            symbols[written] = SymbolRef::Local(local);
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

            for symbol in symbols.iter().copied() {
                let (token, byte_length) = match symbol {
                    SymbolRef::Byte(byte) => (
                        TokenId::new(self.fallback_pack_id, LocalTokenId::from(byte)),
                        1_usize,
                    ),
                    SymbolRef::Local(local) => {
                        let pack_model = self
                            .model
                            .pack(span.pack_id)
                            .expect("only learned symbols come from a present pack model");
                        let length = pack_model
                            .byte_length(local)
                            .expect("validated merge creates every local symbol");
                        (TokenId::new(span.pack_id, local), length)
                    }
                };
                tokens.push(token);
                if TRACK {
                    if let Some(pack_stats) = stats_index(&mut stats, token.pack) {
                        pack_stats.tokens = pack_stats.tokens.saturating_add(1);
                        pack_stats.token_bytes = pack_stats
                            .token_bytes
                            .saturating_add(u64::try_from(byte_length).unwrap_or(u64::MAX));
                    }
                }
            }
        }

        if TRACK {
            stats.pack_transitions = u64::try_from(
                tokens
                    .windows(2)
                    .filter(|pair| pair[0].pack != pair[1].pack)
                    .count(),
            )
            .unwrap_or(u64::MAX);
            stats.output_capacity_bytes = tokens.capacity().saturating_mul(size_of::<TokenId>());
            stats.temporary_peak_bytes = spans
                .capacity()
                .saturating_mul(size_of::<RoutedSpan>())
                .saturating_add(symbols.capacity().saturating_mul(size_of::<SymbolRef>()));
        }
        Ok((tokens, stats))
    }

    /// Decodes tokens into the exact bytes represented by shared or learned IDs.
    pub fn decode_factorized_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError> {
        let mut decoded_length = 0_usize;
        for (index, token) in tokens.iter().enumerate() {
            let token_length = if token.pack == self.fallback_pack_id {
                u8::try_from(token.local).map_err(|_| DecodeError::LocalIdOutsideByteRange {
                    index,
                    actual: token.local,
                })?;
                1
            } else {
                let pack = self
                    .model
                    .pack(token.pack)
                    .ok_or(DecodeError::UnknownPack {
                        index,
                        actual: token.pack,
                    })?;
                pack.byte_length(token.local)
                    .ok_or(DecodeError::LocalIdOutsidePack {
                        index,
                        pack: token.pack,
                        actual: token.local,
                        local_token_count: pack.local_token_count(),
                    })?
            };
            decoded_length = decoded_length
                .checked_add(token_length)
                .ok_or(DecodeError::LengthOverflow)?;
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(decoded_length)
            .map_err(|_| DecodeError::AllocationFailed {
                requested_bytes: decoded_length,
            })?;
        let mut stack = Vec::<SymbolRef>::new();
        for (index, token) in tokens.iter().enumerate() {
            if token.pack == self.fallback_pack_id {
                bytes.push(u8::try_from(token.local).map_err(|_| {
                    DecodeError::LocalIdOutsideByteRange {
                        index,
                        actual: token.local,
                    }
                })?);
                continue;
            }
            let pack = self
                .model
                .pack(token.pack)
                .ok_or(DecodeError::UnknownPack {
                    index,
                    actual: token.pack,
                })?;
            let root = token.local;
            stack.clear();
            stack
                .try_reserve(1)
                .map_err(|_| DecodeError::AllocationFailed { requested_bytes: 1 })?;
            stack.push(SymbolRef::Local(root));
            while let Some(symbol) = stack.pop() {
                match symbol {
                    SymbolRef::Byte(byte) => bytes.push(byte),
                    SymbolRef::Local(local) => {
                        let merge = pack.merges()[usize::try_from(local).map_err(|_| {
                            DecodeError::LocalIdOutsidePack {
                                index,
                                pack: token.pack,
                                actual: LocalTokenId::from(local),
                                local_token_count: pack.local_token_count(),
                            }
                        })?];
                        stack
                            .try_reserve(2)
                            .map_err(|_| DecodeError::AllocationFailed {
                                requested_bytes: stack
                                    .capacity()
                                    .saturating_add(2)
                                    .saturating_mul(size_of::<SymbolRef>()),
                            })?;
                        stack.push(merge.right);
                        stack.push(merge.left);
                    }
                }
            }
        }
        Ok(bytes)
    }
}

impl FactorizedTokenizer<LexicalV1Router> {
    /// Loads the M2 tokenizer and verifies its policy against the built-in router.
    pub fn from_artifact(artifact: &Artifact) -> Result<Self, FactorizedTokenizerError> {
        let model = artifact
            .factorized_bpe()
            .cloned()
            .ok_or(FactorizedTokenizerError::MissingFactorizedModel)?;
        if model.router_policy_id() != LEXICAL_V1_ROUTER_ID {
            return Err(FactorizedTokenizerError::RouterPolicyMismatch {
                artifact: model.router_policy_id().to_owned(),
                runtime: LEXICAL_V1_ROUTER_ID.to_owned(),
            });
        }
        Self::new(
            model,
            artifact.registry().byte_fallback().pack_id(),
            LexicalV1Router,
        )
    }
}

impl<R: PackRouter> Tokenizer for FactorizedTokenizer<R> {
    fn encode(&self, input: &str) -> Result<Vec<TokenId>, EncodeError> {
        self.encode_impl::<false>(input).map(|(tokens, _)| tokens)
    }

    fn decode_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError> {
        self.decode_factorized_bytes(tokens)
    }
}

fn validate_spans(input: &str, spans: &[RoutedSpan]) -> Result<(), EncodeError> {
    let mut expected_start = 0;
    for (index, span) in spans.iter().enumerate() {
        if span.start != expected_start
            || span.start >= span.end
            || span.end > input.len()
            || !input.is_char_boundary(span.start)
            || !input.is_char_boundary(span.end)
            || lexical_pack_name(span.pack_id).is_none()
        {
            return Err(EncodeError::InvalidRouterOutput { span_index: index });
        }
        expected_start = span.end;
    }
    if expected_start != input.len() {
        return Err(EncodeError::InvalidRouterOutput {
            span_index: spans.len(),
        });
    }
    Ok(())
}

fn stats_index(
    stats: &mut FactorizedEncodeStats,
    pack_id: PackId,
) -> Option<&mut FactorizedPackStats> {
    stats
        .packs
        .iter_mut()
        .find(|entry| entry.pack_id == pack_id)
}

/// Failure while loading an M2 runtime tokenizer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FactorizedTokenizerError {
    /// The artifact has no version-3 factorized model section.
    MissingFactorizedModel,
    /// Artifact and runtime router identifiers differ.
    RouterPolicyMismatch { artifact: String, runtime: String },
    /// The selected fallback ID aliases a specialized lexical-v1 namespace.
    InvalidFallbackPackId { pack_id: PackId },
}

impl fmt::Display for FactorizedTokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingFactorizedModel => {
                formatter.write_str("artifact has no factorized BPE model")
            }
            Self::RouterPolicyMismatch { artifact, runtime } => write!(
                formatter,
                "artifact router {artifact:?} does not match runtime router {runtime:?}"
            ),
            Self::InvalidFallbackPackId { pack_id } => write!(
                formatter,
                "byte fallback pack ID {pack_id} aliases a lexical-v1 pack"
            ),
        }
    }
}

impl std::error::Error for FactorizedTokenizerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_core::DEFAULT_BYTE_FALLBACK_PACK_ID;
    use packtok_format::{PackBpeMerge, PackBpeModel, SymbolRef};

    fn model() -> FactorizedBpeModel {
        FactorizedBpeModel::new(
            LEXICAL_V1_ROUTER_ID,
            vec![
                PackBpeModel::new(
                    TEXT_PACK_ID,
                    vec![PackBpeMerge {
                        left: SymbolRef::Byte(b'h'),
                        right: SymbolRef::Byte(b'i'),
                    }],
                )
                .expect("text merge"),
                PackBpeModel::new(
                    NUMBER_PACK_ID,
                    vec![PackBpeMerge {
                        left: SymbolRef::Byte(b'1'),
                        right: SymbolRef::Byte(b'2'),
                    }],
                )
                .expect("number merge"),
            ],
        )
        .expect("factorized model")
    }

    fn tokenizer() -> FactorizedTokenizer {
        FactorizedTokenizer::new(model(), DEFAULT_BYTE_FALLBACK_PACK_ID, LexicalV1Router)
            .expect("factorized tokenizer")
    }

    #[test]
    fn shared_byte_and_overlapping_local_ids_are_distinct_tokens() {
        let tokenizer = tokenizer();
        let tokens = tokenizer.encode("hi 12").expect("encode");
        assert_eq!(tokens[0], TokenId::new(TEXT_PACK_ID, 0));
        assert_eq!(
            tokens[1],
            TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, u32::from(b' '))
        );
        assert_eq!(tokens[2], TokenId::new(NUMBER_PACK_ID, 0));
        assert_ne!(
            TokenId::new(TEXT_PACK_ID, 0),
            TokenId::new(NUMBER_PACK_ID, 0)
        );
    }

    #[test]
    fn mixed_pack_unicode_and_raw_byte_round_trips_are_exact() {
        let tokenizer = tokenizer();
        for input in [
            "",
            "hello 123!",
            "Sämtliche Häuser äöüß",
            "👩🏽‍💻",
            "e\u{301}",
            "漢字",
            "\0\t\n",
        ] {
            let encoded = tokenizer.encode(input).expect("encode text");
            assert_eq!(
                tokenizer.decode_bytes(&encoded).expect("decode"),
                input.as_bytes()
            );
        }
        let arbitrary = [0, 0xff, 0xc3, 0x28, b'a'];
        let encoded = tokenizer.encode_bytes(&arbitrary).expect("raw bytes");
        assert!(
            encoded
                .iter()
                .all(|token| token.pack == DEFAULT_BYTE_FALLBACK_PACK_ID)
        );
        assert_eq!(
            tokenizer.decode_bytes(&encoded).expect("raw decode"),
            arbitrary
        );
    }

    #[test]
    fn artifact_runtime_records_temporary_and_pack_metrics() {
        let tokenizer = tokenizer();
        let (tokens, stats) = tokenizer.encode_with_stats("hi 12!").expect("encode");
        assert_eq!(tokenizer.decode_bytes(&tokens).expect("decode"), b"hi 12!");
        assert_eq!(stats.for_pack(TEXT_PACK_ID).expect("text stats").tokens, 1);
        assert_eq!(
            stats
                .for_pack(NUMBER_PACK_ID)
                .expect("number stats")
                .routed_bytes,
            2
        );
        assert_eq!(
            stats
                .for_pack(STRUCTURE_PACK_ID)
                .expect("structure stats")
                .spans,
            2
        );
        assert_eq!(
            stats.output_capacity_bytes,
            tokens.capacity() * size_of::<TokenId>()
        );
        assert!(stats.temporary_peak_bytes > 0);
        assert_eq!(stats.pack_transitions, 3);
    }

    #[test]
    fn decoder_rejects_unknown_pack_and_local_id_gap() {
        let tokenizer = tokenizer();
        assert!(matches!(
            tokenizer.decode_bytes(&[TokenId::new(55, 0)]),
            Err(DecodeError::UnknownPack { actual: 55, .. })
        ));
        assert!(matches!(
            tokenizer.decode_bytes(&[TokenId::new(TEXT_PACK_ID, 1)]),
            Err(DecodeError::LocalIdOutsidePack { actual: 1, .. })
        ));
    }

    #[test]
    fn identical_bytes_in_distinct_pack_tokens_decode_without_ambiguity() {
        let shared_bytes = || {
            PackBpeModel::new(
                TEXT_PACK_ID,
                vec![PackBpeMerge {
                    left: SymbolRef::Byte(b'x'),
                    right: SymbolRef::Byte(b'y'),
                }],
            )
        };
        let model = FactorizedBpeModel::new(
            LEXICAL_V1_ROUTER_ID,
            vec![
                shared_bytes().expect("text graph"),
                PackBpeModel::new(
                    NUMBER_PACK_ID,
                    vec![PackBpeMerge {
                        left: SymbolRef::Byte(b'x'),
                        right: SymbolRef::Byte(b'y'),
                    }],
                )
                .expect("number graph"),
            ],
        )
        .expect("factorized model");
        let tokenizer =
            FactorizedTokenizer::new(model, DEFAULT_BYTE_FALLBACK_PACK_ID, LexicalV1Router)
                .expect("factorized tokenizer");
        let tokens = [
            TokenId::new(TEXT_PACK_ID, 0),
            TokenId::new(NUMBER_PACK_ID, 0),
        ];
        assert_ne!(tokens[0], tokens[1]);
        assert_eq!(tokenizer.decode_bytes(&tokens).expect("decode"), b"xyxy");
    }

    #[test]
    fn runtime_encoder_does_not_merge_across_pack_boundaries() {
        let tokenizer = tokenizer();
        let tokens = tokenizer.encode("hi1").expect("encode");
        assert_eq!(tokens[0], TokenId::new(TEXT_PACK_ID, 0));
        assert_eq!(
            tokens[1],
            TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, u32::from(b'1'))
        );
    }
}
