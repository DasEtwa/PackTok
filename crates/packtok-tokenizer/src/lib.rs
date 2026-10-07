#![forbid(unsafe_code)]

mod bpe;

use std::fmt;

pub use bpe::{BpeBufferStats, BpeTokenizer, TokenizerModelError};
pub use packtok_core::{
    ByteFallback, DEFAULT_BYTE_FALLBACK_PACK_ID, LocalTokenId, PackId, PackRegistry, TokenId,
};

/// Text-to-token and token-to-byte operations shared by runtime tokenizers.
pub trait Tokenizer {
    /// Encodes UTF-8 input without normalization.
    fn encode(&self, input: &str) -> Result<Vec<TokenId>, EncodeError>;

    /// Decodes token IDs to their exact byte sequence.
    fn decode_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError>;

    /// Decodes a token sequence as UTF-8 text.
    fn decode(&self, tokens: &[TokenId]) -> Result<String, DecodeError> {
        let bytes = self.decode_bytes(tokens)?;
        String::from_utf8(bytes).map_err(|error| {
            let utf8_error = error.utf8_error();
            DecodeError::InvalidUtf8 {
                valid_up_to: utf8_error.valid_up_to(),
                error_len: utf8_error.error_len(),
            }
        })
    }
}

/// A tokenizer whose only behavior is raw-byte fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteFallbackTokenizer {
    fallback: ByteFallback,
}

impl ByteFallbackTokenizer {
    /// Creates a tokenizer for the supplied reserved byte-fallback pack.
    #[must_use]
    pub const fn new(fallback: ByteFallback) -> Self {
        Self { fallback }
    }

    /// Creates a tokenizer using the fallback pack declared by a validated registry.
    #[must_use]
    pub const fn from_registry(registry: &PackRegistry) -> Self {
        Self::new(registry.byte_fallback())
    }

    /// Returns the pack ID used for byte tokens.
    #[must_use]
    pub const fn fallback_pack_id(self) -> PackId {
        self.fallback.pack_id()
    }
}

impl Default for ByteFallbackTokenizer {
    fn default() -> Self {
        Self::new(ByteFallback::new(DEFAULT_BYTE_FALLBACK_PACK_ID))
    }
}

impl Tokenizer for ByteFallbackTokenizer {
    fn encode(&self, input: &str) -> Result<Vec<TokenId>, EncodeError> {
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(input.len())
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: input.len(),
            })?;

        tokens.extend(
            input
                .as_bytes()
                .iter()
                .map(|byte| TokenId::new(self.fallback.pack_id(), LocalTokenId::from(*byte))),
        );
        Ok(tokens)
    }

    fn decode_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(tokens.len())
            .map_err(|_| DecodeError::AllocationFailed {
                requested_bytes: tokens.len(),
            })?;

        for (index, token) in tokens.iter().enumerate() {
            if token.pack != self.fallback.pack_id() {
                return Err(DecodeError::UnexpectedPack {
                    index,
                    expected: self.fallback.pack_id(),
                    actual: token.pack,
                });
            }

            let byte =
                u8::try_from(token.local).map_err(|_| DecodeError::LocalIdOutsideByteRange {
                    index,
                    actual: token.local,
                })?;
            bytes.push(byte);
        }

        Ok(bytes)
    }
}

/// Error returned when the encoder cannot reserve its output buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EncodeError {
    /// The requested token vector could not be reserved.
    AllocationFailed { requested_tokens: usize },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationFailed { requested_tokens } => write!(
                formatter,
                "could not reserve space for {requested_tokens} tokens"
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Error returned when token IDs cannot be decoded by the byte-fallback tokenizer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// The token points at a pack other than this tokenizer's byte-fallback pack.
    UnexpectedPack {
        index: usize,
        expected: PackId,
        actual: PackId,
    },
    /// The local ID does not identify a byte value in `0..=255`.
    LocalIdOutsideByteRange { index: usize, actual: LocalTokenId },
    /// The local ID is outside this BPE model's flat vocabulary.
    LocalIdOutsideVocabulary {
        index: usize,
        actual: LocalTokenId,
        vocabulary_size: u32,
    },
    /// Decoded bytes are not valid UTF-8.
    InvalidUtf8 {
        valid_up_to: usize,
        error_len: Option<usize>,
    },
    /// The decoder could not reserve its output buffer.
    AllocationFailed { requested_bytes: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedPack {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "token {index} references pack {actual}; expected pack {expected}"
            ),
            Self::LocalIdOutsideByteRange { index, actual } => write!(
                formatter,
                "token {index} has local ID {actual}; byte IDs must be in 0..=255"
            ),
            Self::LocalIdOutsideVocabulary {
                index,
                actual,
                vocabulary_size,
            } => write!(
                formatter,
                "token {index} has local ID {actual}; BPE vocabulary contains IDs 0..{vocabulary_size}"
            ),
            Self::InvalidUtf8 {
                valid_up_to,
                error_len,
            } => write!(
                formatter,
                "decoded bytes are not UTF-8 (valid prefix: {valid_up_to} bytes, error length: {error_len:?})"
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "could not reserve space for {requested_bytes} decoded bytes"
            ),
        }
    }
}

impl std::error::Error for DecodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_round_trip(input: &str) {
        let tokenizer = ByteFallbackTokenizer::default();
        let tokens = tokenizer.encode(input).expect("encode input");
        assert_eq!(tokenizer.decode(&tokens).expect("decode input"), input);
        assert_eq!(
            tokenizer.decode_bytes(&tokens).expect("decode bytes"),
            input.as_bytes()
        );
        assert_eq!(tokens.len(), input.len());
    }

    #[test]
    fn byte_fallback_round_trips_required_text_classes() {
        for input in [
            "PackTok byte fallback",
            "Sämtliche Häuser öffnen Türen; Grüße von der Straße.",
            "ä ö ü ß Ä Ö Ü",
            "✨🦦🇩🇪👩🏽‍💻",
            "a\u{0308} o\u{0308} u\u{0308} e\u{0301}",
            "漢字かなカナ中文",
            "first line\nsecond line\r\nthird line",
            "tabs\tand\tmore",
            "  repeated   spaces\t\tand newlines\n\n",
            "",
            "before\0after",
        ] {
            assert_round_trip(input);
        }
    }

    #[test]
    fn every_ascii_byte_and_nul_round_trip() {
        let input: String = (0_u8..=127).map(char::from).collect();
        assert_round_trip(&input);
    }

    #[test]
    fn deterministic_token_ids_are_byte_values_in_one_pack() {
        let tokenizer = ByteFallbackTokenizer::default();
        let expected = vec![
            TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, u32::from(b'a')),
            TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, u32::from(b'\n')),
            TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, 0xc3),
            TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, 0xa4),
        ];
        let first = tokenizer.encode("a\nä").expect("encode first time");
        let second = tokenizer.encode("a\nä").expect("encode second time");
        assert_eq!(first, expected);
        assert_eq!(second, expected);
    }

    #[test]
    fn deterministic_generated_unicode_round_trips() {
        let mut state = 0x4d59_5df4_u32;
        let mut input = String::new();
        for _ in 0..20_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let code_point = state % 0x11_0000;
            if let Some(character) = char::from_u32(code_point) {
                input.push(character);
            }
        }
        assert_round_trip(&input);
    }

    #[test]
    fn raw_byte_decode_preserves_invalid_utf8_while_text_decode_rejects_it() {
        let tokenizer = ByteFallbackTokenizer::default();
        let tokens = [0xff_u8, 0xc3, 0x28]
            .map(|byte| TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, u32::from(byte)));
        assert_eq!(
            tokenizer.decode_bytes(&tokens).expect("decode raw bytes"),
            [0xff, 0xc3, 0x28]
        );
        assert!(matches!(
            tokenizer.decode(&tokens),
            Err(DecodeError::InvalidUtf8 { .. })
        ));
    }

    #[test]
    fn decoder_rejects_unknown_packs_and_out_of_range_local_ids() {
        let tokenizer = ByteFallbackTokenizer::default();
        assert_eq!(
            tokenizer.decode_bytes(&[TokenId::new(4, 1)]),
            Err(DecodeError::UnexpectedPack {
                index: 0,
                expected: DEFAULT_BYTE_FALLBACK_PACK_ID,
                actual: 4,
            })
        );
        assert_eq!(
            tokenizer.decode_bytes(&[TokenId::new(DEFAULT_BYTE_FALLBACK_PACK_ID, 256)]),
            Err(DecodeError::LocalIdOutsideByteRange {
                index: 0,
                actual: 256,
            })
        );
    }

    #[test]
    fn tokenizer_uses_registry_fallback_id() {
        let registry = PackRegistry::new(
            vec![packtok_core::PackDescriptor::new(
                17,
                "raw",
                packtok_core::BYTE_FALLBACK_TOKEN_COUNT,
            )],
            ByteFallback::new(17),
            vec![],
        )
        .expect("valid registry");
        let tokenizer = ByteFallbackTokenizer::from_registry(&registry);
        let tokens = tokenizer.encode("x").expect("encode");
        assert_eq!(tokens, [TokenId::new(17, u32::from(b'x'))]);
    }
}
