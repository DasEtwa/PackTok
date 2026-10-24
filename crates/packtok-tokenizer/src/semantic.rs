use packtok_core::TokenId;
use packtok_packs::semantic::{SemanticAnnotation, SemanticRegistry, SemanticRegistryIdentity};

use crate::{EncodeError, Tokenizer};

/// Unmodified lexical tokens with an optional, independent semantic sidecar.
///
/// Offsets in semantic annotations refer to original input bytes, not token
/// positions: a BPE token can cross a semantic span boundary. Decode only
/// `lexical_tokens` with the original tokenizer to recover the exact input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticEncoding {
    /// The exact IDs returned by the underlying tokenizer.
    pub lexical_tokens: Vec<TokenId>,
    /// `None` when disabled; `Some([])` means enabled with no known matches.
    pub semantic_annotations: Option<Vec<SemanticAnnotation>>,
    /// Identity required to interpret semantic IDs, including custom registries.
    /// `None` when semantic annotation is disabled. Numeric IDs alone are not
    /// comparable across registry namespaces, revisions, or policy fingerprints.
    pub semantic_registry: Option<SemanticRegistryIdentity>,
}

/// Encodes without changing lexical segmentation or IDs, optionally annotating
/// the original text with deterministic semantic decisions.
///
/// Passing `None` performs no registry matching. Semantic IDs never enter the
/// lexical token namespace and never replace original text or decoded bytes.
/// This helper accepts UTF-8 text, like the existing `Tokenizer::encode` trait
/// method. Existing BPE `encode_bytes` methods remain available for arbitrary,
/// including invalid UTF-8, bytes and do not perform semantic annotation.
pub fn encode_with_semantics<T: Tokenizer + ?Sized>(
    tokenizer: &T,
    input: &str,
    registry: Option<&SemanticRegistry>,
) -> Result<SemanticEncoding, EncodeError> {
    let lexical_tokens = tokenizer.encode(input)?;
    let semantic_annotations = registry.map(|registry| registry.annotate(input));
    let semantic_registry = registry.map(SemanticRegistry::identity);
    Ok(SemanticEncoding {
        lexical_tokens,
        semantic_annotations,
        semantic_registry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ByteFallbackTokenizer;

    #[test]
    fn optional_annotation_preserves_lexical_tokens_and_exact_roundtrip() {
        let tokenizer = ByteFallbackTokenizer::default();
        let registry = SemanticRegistry::builtin();
        for input in [
            "",
            "ong idk bro",
            "ka, keine Ahnung! dunno; I don’t know.",
            "language: fr\nfr bro\nKA",
            "äöüß ÄÖÜ 👩🏽‍💻 e\u{301}",
            "\tngl\r\n tbh\u{a0}idk\0",
            "unknown_identifier 12345 漢字",
        ] {
            let baseline = tokenizer.encode(input).expect("baseline");
            let disabled = encode_with_semantics(&tokenizer, input, None).expect("disabled");
            assert_eq!(disabled.lexical_tokens, baseline);
            assert_eq!(disabled.semantic_annotations, None);
            assert_eq!(disabled.semantic_registry, None);
            let enabled =
                encode_with_semantics(&tokenizer, input, Some(registry)).expect("enabled");
            assert_eq!(enabled.lexical_tokens, baseline);
            assert_eq!(enabled.semantic_registry, Some(registry.identity()));
            assert_eq!(
                tokenizer.decode_bytes(&enabled.lexical_tokens).unwrap(),
                input.as_bytes()
            );
            assert_eq!(
                enabled,
                encode_with_semantics(&tokenizer, input, Some(registry)).unwrap()
            );
            for annotation in enabled.semantic_annotations.unwrap() {
                assert!(input.get(annotation.start..annotation.end).is_some());
            }
        }
    }

    #[test]
    fn unknown_text_distinguishes_disabled_from_enabled_with_no_matches() {
        let tokenizer: &dyn Tokenizer = &ByteFallbackTokenizer::default();
        let encoded = encode_with_semantics(tokenizer, "xyzzy", Some(SemanticRegistry::builtin()))
            .expect("unknown text is representable");
        assert_eq!(encoded.semantic_annotations, Some(vec![]));
        assert_eq!(
            encoded.semantic_registry,
            Some(SemanticRegistry::builtin().identity())
        );
        assert_eq!(tokenizer.decode(&encoded.lexical_tokens).unwrap(), "xyzzy");
    }

    #[test]
    fn encoding_preserves_registry_identity_when_numeric_ids_collide() {
        use packtok_packs::semantic::{
            ResolutionRule, SemanticConcept, SemanticVariant, UNCERTAINTY,
        };

        let builtin = SemanticRegistry::builtin();
        let custom = SemanticRegistry::new(
            "example.custom",
            1,
            vec![SemanticConcept {
                id: UNCERTAINTY,
                name: "UNCERTAINTY".to_owned(),
                description:
                    "A custom interpretation unrelated to the built-in uncertainty concept."
                        .to_owned(),
            }],
            vec![SemanticVariant {
                pattern: "idk".to_owned(),
                candidates: vec![UNCERTAINTY],
                rule: ResolutionRule::Direct,
            }],
        )
        .expect("valid custom registry with its own ID namespace");
        let tokenizer = ByteFallbackTokenizer::default();
        let first = encode_with_semantics(&tokenizer, "idk", Some(builtin)).unwrap();
        let second = encode_with_semantics(&tokenizer, "idk", Some(&custom)).unwrap();
        assert_eq!(first.lexical_tokens, second.lexical_tokens);
        assert_eq!(first.semantic_annotations, second.semantic_annotations);
        assert_eq!(first.semantic_registry, Some(builtin.identity()));
        assert_eq!(second.semantic_registry, Some(custom.identity()));
        assert_ne!(first.semantic_registry, second.semantic_registry);
        assert_ne!(first, second);
    }

    #[test]
    fn generated_unicode_is_never_normalized_by_semantic_annotation() {
        let tokenizer = ByteFallbackTokenizer::default();
        let mut input = String::from("idk 👩🏽‍💻 ");
        let mut state = 0xa511_e9b3_u32;
        for _ in 0..10_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if let Some(ch) = char::from_u32(state % 0x11_0000) {
                input.push(ch);
            }
        }
        let encoded =
            encode_with_semantics(&tokenizer, &input, Some(SemanticRegistry::builtin())).unwrap();
        assert_eq!(tokenizer.decode(&encoded.lexical_tokens).unwrap(), input);
        assert_eq!(encoded.lexical_tokens, tokenizer.encode(&input).unwrap());
    }
}
