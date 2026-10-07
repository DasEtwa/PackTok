#![forbid(unsafe_code)]

use packtok_core::PackId;

/// Stable identifier for M2's first experimental router.
pub const LEXICAL_V1_ROUTER_ID: &str = "lexical-v1";

/// Pack IDs used by `lexical-v1`; IDs are independent of local token IDs.
pub const TEXT_PACK_ID: PackId = 0;
pub const NUMBER_PACK_ID: PackId = 1;
pub const STRUCTURE_PACK_ID: PackId = 2;

/// Canonical metadata names for the packs exposed by `lexical-v1`.
pub const TEXT_PACK_NAME: &str = "TEXT";
pub const NUMBER_PACK_NAME: &str = "NUMBER";
pub const STRUCTURE_PACK_NAME: &str = "STRUCTURE";

/// A UTF-8-safe byte range routed to one specialized pack.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedSpan {
    /// Pack selected for every character in this span.
    pub pack_id: PackId,
    /// Inclusive byte offset in the original UTF-8 input.
    pub start: usize,
    /// Exclusive byte offset in the original UTF-8 input.
    pub end: usize,
}

/// Deterministic text-to-pack routing contract.
pub trait PackRouter {
    /// Stable policy identifier stored in artifacts.
    fn policy_id(&self) -> &'static str;

    /// Returns ordered, non-empty spans covering the complete input exactly.
    /// Every boundary must be a UTF-8 character boundary.
    fn route(&self, input: &str) -> Vec<RoutedSpan>;
}

/// Crude, context-free M2 routing policy. It does not use Unicode properties.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LexicalV1Router;

impl PackRouter for LexicalV1Router {
    fn policy_id(&self) -> &'static str {
        LEXICAL_V1_ROUTER_ID
    }

    fn route(&self, input: &str) -> Vec<RoutedSpan> {
        let mut spans = Vec::new();
        let mut chars = input.char_indices();
        let Some((first_start, first_char)) = chars.next() else {
            return spans;
        };

        let mut span_start = first_start;
        let mut current_pack = pack_for(first_char);
        for (start, ch) in chars {
            let pack_id = pack_for(ch);
            if pack_id != current_pack {
                spans.push(RoutedSpan {
                    pack_id: current_pack,
                    start: span_start,
                    end: start,
                });
                span_start = start;
                current_pack = pack_id;
            }
        }
        spans.push(RoutedSpan {
            pack_id: current_pack,
            start: span_start,
            end: input.len(),
        });
        spans
    }
}

fn pack_for(ch: char) -> PackId {
    if !ch.is_ascii() || ch.is_ascii_alphabetic() {
        TEXT_PACK_ID
    } else if ch.is_ascii_digit() {
        NUMBER_PACK_ID
    } else {
        STRUCTURE_PACK_ID
    }
}

/// Returns the canonical metadata name for a `lexical-v1` pack ID.
#[must_use]
pub const fn lexical_pack_name(pack_id: PackId) -> Option<&'static str> {
    match pack_id {
        TEXT_PACK_ID => Some(TEXT_PACK_NAME),
        NUMBER_PACK_ID => Some(NUMBER_PACK_NAME),
        STRUCTURE_PACK_ID => Some(STRUCTURE_PACK_NAME),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(input: &str) -> Vec<(PackId, &str)> {
        LexicalV1Router
            .route(input)
            .into_iter()
            .map(|span| (span.pack_id, &input[span.start..span.end]))
            .collect()
    }

    #[test]
    fn lexical_v1_coalesces_ascii_classes_and_keeps_unicode_scalars_whole() {
        assert_eq!(
            spans("Hello 123! äöüß漢🙂"),
            vec![
                (TEXT_PACK_ID, "Hello"),
                (STRUCTURE_PACK_ID, " "),
                (NUMBER_PACK_ID, "123"),
                (STRUCTURE_PACK_ID, "! "),
                (TEXT_PACK_ID, "äöüß漢🙂"),
            ]
        );
    }

    #[test]
    fn lexical_v1_routes_all_non_letters_ascii_to_structure() {
        assert_eq!(
            spans("\0\t\n <>_%"),
            vec![(STRUCTURE_PACK_ID, "\0\t\n <>_%")]
        );
        assert_eq!(spans(""), Vec::<(PackId, &str)>::new());
    }

    #[test]
    fn lexical_v1_is_deterministic_and_covers_only_character_boundaries() {
        let input = "a1é.漢\n";
        let first = LexicalV1Router.route(input);
        assert_eq!(first, LexicalV1Router.route(input));
        assert_eq!(first.first().map(|span| span.start), Some(0));
        assert_eq!(first.last().map(|span| span.end), Some(input.len()));
        for span in &first {
            assert!(input.is_char_boundary(span.start));
            assert!(input.is_char_boundary(span.end));
            assert!(span.start < span.end);
        }
        for pair in first.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
    }
}
