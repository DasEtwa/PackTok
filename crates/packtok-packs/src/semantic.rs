//! Experimental semantic-v1 registry and conservative annotation policy.
//!
//! IDs are semantic concepts in a separate namespace, never lexical token IDs.
//! Spans refer to the original UTF-8 bytes. Candidate lists are not exhaustive
//! descriptions of all possible meanings, and a resolved result is a rule-based
//! interpretation, not a calibrated probability or a claim of synonymy.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::OnceLock;

/// Resolver changes require a new policy ID and registry fingerprint.
pub const SEMANTIC_POLICY_ID: &str = "semantic-rules-v1.1";
pub const MAX_SEMANTIC_CONCEPTS: usize = 256;
pub const MAX_SEMANTIC_VARIANTS: usize = 1024;
const MAX_PATTERN_BYTES: usize = 128;

/// Stable concept ID, interpreted only within a registry namespace/revision.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SemanticId(pub u32);

/// Carries the interpretation domain alongside model-facing annotations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticRegistryIdentity {
    pub namespace: String,
    pub revision: u32,
    pub policy_id: &'static str,
    pub fingerprint: u64,
}

pub const UNCERTAINTY: SemanticId = SemanticId(1);
pub const HONESTY_MARKER: SemanticId = SemanticId(2);
pub const SWEARING_EMPHASIS: SemanticId = SemanticId(3);
pub const AGREEMENT_EMPHASIS: SemanticId = SemanticId(4);
pub const FRENCH_LANGUAGE: SemanticId = SemanticId(5);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticConcept {
    pub id: SemanticId,
    pub name: String,
    /// Nuance and intended scope, rather than a replacement for the surface text.
    pub description: String,
}

/// Small explicit rule set. No learned model or hidden language detector is used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolutionRule {
    Direct,
    FrContext,
    KaContext,
    LowercaseOnly,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticVariant {
    /// Lowercase ASCII phrase; spaces match horizontal Unicode whitespace and
    /// apostrophes also match U+2019. No punctuation/normalization is discarded.
    pub pattern: String,
    /// Sorted by ID by the validated registry. Multiple candidates never resolve
    /// without an explicit contextual rule.
    pub candidates: Vec<SemanticId>,
    pub rule: ResolutionRule,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticDecision {
    Resolved,
    Ambiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticReason {
    KnownVariant,
    SlangContext,
    LanguageContext,
    InsufficientContext,
    CaseAmbiguity,
    ConflictingContext,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticAnnotation {
    pub start: usize,
    pub end: usize,
    pub candidates: Vec<SemanticId>,
    pub decision: SemanticDecision,
    pub reason: SemanticReason,
}

impl SemanticAnnotation {
    /// An unambiguous rule-selected ID. Unresolved candidates are never returned.
    #[must_use]
    pub fn resolved_id(&self) -> Option<SemanticId> {
        if self.decision == SemanticDecision::Resolved && self.candidates.len() == 1 {
            self.candidates.first().copied()
        } else {
            None
        }
    }
}

/// Immutable, bounded registry. Built-in IDs/revision are frozen; custom
/// registries must use their own namespace and explicitly manage stable IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticRegistry {
    namespace: String,
    revision: u32,
    concepts: Vec<SemanticConcept>,
    variants: Vec<SemanticVariant>,
    fingerprint: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticRegistryError(pub &'static str);

impl fmt::Display for SemanticRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for SemanticRegistryError {}

impl SemanticRegistry {
    /// Validates bounded ASCII metadata, unique IDs/names/patterns, candidate
    /// references and the candidate contracts of context rules. Ordering of
    /// supplied entries does not affect recognition or fingerprints.
    pub fn new(
        namespace: &str,
        revision: u32,
        mut concepts: Vec<SemanticConcept>,
        mut variants: Vec<SemanticVariant>,
    ) -> Result<Self, SemanticRegistryError> {
        let invalid = |message| Err(SemanticRegistryError(message));
        if namespace.is_empty()
            || namespace.len() > 128
            || !namespace
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            || revision == 0
        {
            return invalid("invalid semantic namespace or revision");
        }
        if concepts.is_empty()
            || concepts.len() > MAX_SEMANTIC_CONCEPTS
            || variants.len() > MAX_SEMANTIC_VARIANTS
        {
            return invalid("semantic registry size exceeds bounds");
        }
        concepts.sort_by_key(|c| c.id);
        let mut names = BTreeSet::new();
        for (index, concept) in concepts.iter().enumerate() {
            if concept.id.0 == 0
                || (index > 0 && concepts[index - 1].id == concept.id)
                || concept.name.is_empty()
                || concept.name.len() > 64
                || !concept
                    .name
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b == b'_')
                || !names.insert(concept.name.as_str())
                || concept.description.len() > 1024
            {
                return invalid("invalid or duplicate semantic concept");
            }
        }
        variants.sort_by(|a, b| a.pattern.cmp(&b.pattern));
        for index in 0..variants.len() {
            let variant = &mut variants[index];
            let pattern = variant.pattern.as_bytes();
            if pattern.is_empty()
                || pattern.len() > MAX_PATTERN_BYTES
                || !pattern[0].is_ascii_lowercase()
                || !pattern[pattern.len() - 1].is_ascii_lowercase()
                || pattern.windows(2).any(|w| w == b"  ")
                || !pattern
                    .iter()
                    .all(|b| b.is_ascii_lowercase() || *b == b' ' || *b == b'\'')
                || variant.candidates.is_empty()
                || variant.candidates.len() > MAX_SEMANTIC_CONCEPTS
            {
                return invalid("invalid semantic variant pattern or candidates");
            }
            variant.candidates.sort_unstable();
            if variant.candidates.windows(2).any(|p| p[0] == p[1])
                || variant
                    .candidates
                    .iter()
                    .any(|id| concepts.binary_search_by_key(id, |c| c.id).is_err())
            {
                return invalid("duplicate or unknown semantic candidate ID");
            }
            match variant.rule {
                ResolutionRule::FrContext
                    if variant.pattern != "fr"
                        || variant.candidates != [AGREEMENT_EMPHASIS, FRENCH_LANGUAGE] =>
                {
                    return invalid("fr context rule requires fr and its two built-in IDs");
                }
                ResolutionRule::KaContext
                    if variant.pattern != "ka" || variant.candidates != [UNCERTAINTY] =>
                {
                    return invalid("ka context rule requires ka and uncertainty ID");
                }
                ResolutionRule::LowercaseOnly if variant.candidates.len() != 1 => {
                    return invalid("lowercase rule requires exactly one candidate");
                }
                _ => {}
            }
            if index > 0 && variants[index - 1].pattern == variants[index].pattern {
                return invalid("duplicate semantic variant pattern");
            }
        }
        let mut registry = Self {
            namespace: namespace.to_owned(),
            revision,
            concepts,
            variants,
            fingerprint: 0,
        };
        registry.fingerprint = registry.compute_fingerprint();
        Ok(registry)
    }

    /// Frozen small English/German inventory; adding/reinterpreting entries
    /// requires an explicit revision change. No corpus-driven dictionary growth.
    #[must_use]
    pub fn builtin() -> &'static Self {
        static REGISTRY: OnceLock<SemanticRegistry> = OnceLock::new();
        REGISTRY.get_or_init(|| {
            let concepts = [
                (UNCERTAINTY, "UNCERTAINTY", "Expression of not knowing; preserves strength, language and register in the surface span."),
                (HONESTY_MARKER, "HONESTY_MARKER", "Discourse marker introducing candor; ngl and tbh have different pragmatic nuance."),
                (SWEARING_EMPHASIS, "SWEARING_EMPHASIS", "Oath-like emphasis such as on God; not interchangeable with ordinary agreement."),
                (AGREEMENT_EMPHASIS, "AGREEMENT_EMPHASIS", "For-real emphasis or agreement; not a truth guarantee."),
                (FRENCH_LANGUAGE, "FRENCH_LANGUAGE", "French language code under an explicit language label."),
            ].into_iter().map(|(id, name, description)| SemanticConcept {
                id, name: name.to_owned(), description: description.to_owned(),
            }).collect();
            let mut variants = Vec::new();
            for (id, patterns) in [
                (UNCERTAINTY, &["idk", "dunno", "i don't know", "don't know", "keine ahnung"][..]),
                (HONESTY_MARKER, &["ngl", "tbh", "not gonna lie", "to be honest"][..]),
                (SWEARING_EMPHASIS, &["on god"][..]),
                (AGREEMENT_EMPHASIS, &["for real"][..]),
            ] {
                for pattern in patterns {
                    variants.push(SemanticVariant { pattern: (*pattern).to_owned(), candidates: vec![id], rule: ResolutionRule::Direct });
                }
            }
            variants.push(SemanticVariant { pattern: "ong".to_owned(), candidates: vec![SWEARING_EMPHASIS], rule: ResolutionRule::LowercaseOnly });
            variants.push(SemanticVariant { pattern: "fr".to_owned(), candidates: vec![AGREEMENT_EMPHASIS, FRENCH_LANGUAGE], rule: ResolutionRule::FrContext });
            variants.push(SemanticVariant { pattern: "ka".to_owned(), candidates: vec![UNCERTAINTY], rule: ResolutionRule::KaContext });
            Self::new("packtok.semantic", 1, concepts, variants).expect("valid frozen semantic registry")
        })
    }

    #[must_use]
    pub fn identity(&self) -> SemanticRegistryIdentity {
        SemanticRegistryIdentity {
            namespace: self.namespace.clone(),
            revision: self.revision,
            policy_id: SEMANTIC_POLICY_ID,
            fingerprint: self.fingerprint,
        }
    }
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
    #[must_use]
    pub const fn revision(&self) -> u32 {
        self.revision
    }
    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
    #[must_use]
    pub fn concepts(&self) -> &[SemanticConcept] {
        &self.concepts
    }
    #[must_use]
    pub fn variants(&self) -> &[SemanticVariant] {
        &self.variants
    }
    #[must_use]
    pub fn concept(&self, id: SemanticId) -> Option<&SemanticConcept> {
        self.concepts
            .binary_search_by_key(&id, |c| c.id)
            .ok()
            .map(|i| &self.concepts[i])
    }

    /// Left-to-right, longest non-overlapping whole-phrase matching. Matching is
    /// ASCII case-insensitive, with explicit case-sensitive resolution rules.
    /// Unknown spans receive no annotation. Context never crosses CR/LF.
    #[must_use]
    pub fn annotate(&self, input: &str) -> Vec<SemanticAnnotation> {
        let mut annotations = Vec::new();
        let mut pos = 0;
        while pos < input.len() {
            let ch = input[pos..]
                .chars()
                .next()
                .expect("nonempty UTF-8 remainder");
            if !ch.is_ascii_alphabetic() || preceding_word_character(input, pos) {
                pos += ch.len_utf8();
                continue;
            }
            let mut best: Option<(usize, &SemanticVariant)> = None;
            for variant in &self.variants {
                if let Some(end) = match_pattern(input, pos, &variant.pattern) {
                    if best.is_none_or(|(previous_end, _)| end > previous_end) {
                        best = Some((end, variant));
                    }
                }
            }
            if let Some((end, variant)) = best {
                annotations.push(resolve(input, pos, end, variant));
                pos = end;
            } else {
                // Skip an entire word after a failed start, avoiding repeated
                // variant scans in a large unknown identifier.
                pos += ch.len_utf8();
                while pos < input.len() {
                    let next = input[pos..].chars().next().expect("UTF-8 remainder");
                    if !word_character(next) {
                        break;
                    }
                    pos += next.len_utf8();
                }
            }
        }
        annotations
    }

    fn compute_fingerprint(&self) -> u64 {
        // Length-prefix every value; no delimiter ambiguities. This fingerprint
        // detects stale data; it is not a cryptographic authenticity mechanism.
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        let mut add = |bytes: &[u8]| {
            for byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
            }
        };
        add(SEMANTIC_POLICY_ID.as_bytes());
        add(self.namespace.as_bytes());
        add(&self.revision.to_le_bytes());
        add(&(self.concepts.len() as u64).to_le_bytes());
        for concept in &self.concepts {
            add(&concept.id.0.to_le_bytes());
            add(concept.name.as_bytes());
            add(concept.description.as_bytes());
        }
        add(&(self.variants.len() as u64).to_le_bytes());
        for variant in &self.variants {
            add(variant.pattern.as_bytes());
            add(&[match variant.rule {
                ResolutionRule::Direct => 1,
                ResolutionRule::FrContext => 2,
                ResolutionRule::KaContext => 3,
                ResolutionRule::LowercaseOnly => 4,
            }]);
            add(&(variant.candidates.len() as u64).to_le_bytes());
            for id in &variant.candidates {
                add(&id.0.to_le_bytes());
            }
        }
        hash
    }
}

fn horizontal_space(ch: char) -> bool {
    ch.is_whitespace()
        && !matches!(
            ch,
            '\u{000b}' | '\u{000c}' | '\r' | '\n' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
}

fn word_character(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_alphanumeric() || matches!(ch, '_' | '\'');
    }
    // Conservatively attach non-ASCII marks/format controls to adjacent words;
    // this prevents idk + combining mark / zero-width joiner matching as idk.
    // Selected punctuation/emoji ranges are boundaries; other symbols attach
    // conservatively. This is not a complete Unicode segmentation algorithm.
    ch == '\u{2019}'
        || ch.is_alphanumeric()
        || (!ch.is_whitespace()
            && !matches!(ch, '\u{2000}'..='\u{206f}' | '\u{2e00}'..='\u{2e7f}' | '\u{3000}'..='\u{303f}' | '\u{1f000}'..='\u{1faff}' | '\u{2600}'..='\u{27bf}')
            || matches!(ch, '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{206f}'))
}

fn apostrophe(ch: char) -> bool {
    matches!(ch, '\'' | '\u{2019}')
}

fn preceding_word_character(input: &str, pos: usize) -> bool {
    let mut chars = input[..pos].chars().rev();
    let Some(ch) = chars.next() else {
        return false;
    };
    if apostrophe(ch) {
        // An opening quote is a boundary; an intra-word apostrophe is not.
        chars.next().is_some_and(word_character)
    } else {
        word_character(ch)
    }
}

fn following_word_character(input: &str, pos: usize) -> bool {
    let mut chars = input[pos..].chars();
    let Some(ch) = chars.next() else {
        return false;
    };
    if apostrophe(ch) {
        chars.next().is_some_and(word_character)
    } else {
        word_character(ch)
    }
}

fn match_pattern(input: &str, start: usize, pattern: &str) -> Option<usize> {
    let mut pos = start;
    for expected in pattern.bytes() {
        let actual = input.get(pos..)?.chars().next()?;
        if expected == b' ' {
            if !horizontal_space(actual) {
                return None;
            }
            for ch in input[pos..].chars() {
                if !horizontal_space(ch) {
                    break;
                }
                pos += ch.len_utf8();
            }
        } else if (expected == b'\'' && matches!(actual, '\'' | '\u{2019}'))
            || (actual.is_ascii() && actual as u8 == expected.to_ascii_uppercase())
            || actual == char::from(expected)
        {
            pos += actual.len_utf8();
        } else {
            return None;
        }
    }
    if following_word_character(input, pos) {
        None
    } else {
        Some(pos)
    }
}

fn adjacent_word(input: &str, boundary: usize, before: bool) -> &str {
    // Bound context work even for an unbounded word or whitespace run.
    let mut edge = boundary;
    let mut remaining = 128;
    if before {
        while remaining > 0 {
            let Some(ch) = input[..edge].chars().next_back() else {
                break;
            };
            if !horizontal_space(ch) {
                break;
            }
            edge -= ch.len_utf8();
            remaining -= 1;
        }
        if edge == boundary {
            return "";
        }
        let end = edge;
        while remaining > 0 {
            let Some(ch) = input[..edge].chars().next_back() else {
                break;
            };
            if !ch.is_ascii_alphabetic() {
                break;
            }
            edge -= ch.len_utf8();
            remaining -= 1;
        }
        if edge > 0
            && input[..edge]
                .chars()
                .next_back()
                .is_some_and(word_character)
        {
            return "";
        }
        &input[edge..end]
    } else {
        while remaining > 0 {
            let Some(ch) = input[edge..].chars().next() else {
                break;
            };
            if !horizontal_space(ch) {
                break;
            }
            edge += ch.len_utf8();
            remaining -= 1;
        }
        if edge == boundary {
            return "";
        }
        let start = edge;
        while remaining > 0 {
            let Some(ch) = input[edge..].chars().next() else {
                break;
            };
            if !ch.is_ascii_alphabetic() {
                break;
            }
            edge += ch.len_utf8();
            remaining -= 1;
        }
        if input[edge..].chars().next().is_some_and(word_character) {
            return "";
        }
        &input[start..edge]
    }
}

fn language_label(input: &str, start: usize) -> bool {
    // Only an immediately adjacent ASCII metadata key with ':' or '='; quotes
    // are accepted for common JSON-like fields. Never infer from distant text.
    let context_start = input[..start]
        .char_indices()
        .rev()
        .nth(127)
        .map_or(0, |(offset, _)| offset);
    let mut chars = input[context_start..start].chars().rev().peekable();
    while chars
        .peek()
        .is_some_and(|c| horizontal_space(*c) || matches!(c, '"' | '\'' | '\u{2019}'))
    {
        chars.next();
    }
    if !chars.next().is_some_and(|c| matches!(c, ':' | '=')) {
        return false;
    }
    while chars
        .peek()
        .is_some_and(|c| horizontal_space(*c) || matches!(c, '"' | '\'' | '\u{2019}'))
    {
        chars.next();
    }
    let mut key = String::new();
    while chars.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
        key.push(chars.next().expect("peeked character").to_ascii_lowercase());
    }
    // Consume at most one opening key quote, then prove its left boundary.
    if chars
        .peek()
        .is_some_and(|c| matches!(c, '\'' | '\u{2019}' | '"'))
    {
        chars.next();
    }
    if (chars.peek().is_none() && context_start > 0)
        || chars.peek().is_some_and(|c| word_character(*c))
    {
        return false;
    }
    matches!(key.as_str(), "egaugnal" | "gnal" | "ehcarps")
}

fn resolve(input: &str, start: usize, end: usize, variant: &SemanticVariant) -> SemanticAnnotation {
    use SemanticDecision::{Ambiguous, Resolved};
    use SemanticReason::*;
    let surface = &input[start..end];
    let mut candidates = variant.candidates.clone();
    let (decision, reason) = match variant.rule {
        ResolutionRule::Direct if candidates.len() == 1 => (Resolved, KnownVariant),
        ResolutionRule::Direct => (Ambiguous, InsufficientContext),
        ResolutionRule::LowercaseOnly if !surface.bytes().any(|byte| byte.is_ascii_uppercase()) => {
            (Resolved, KnownVariant)
        }
        ResolutionRule::LowercaseOnly => (Ambiguous, CaseAmbiguity),
        ResolutionRule::FrContext => {
            let language = language_label(input, start);
            let slang = adjacent_word(input, start, true).eq_ignore_ascii_case("bro")
                || adjacent_word(input, end, false).eq_ignore_ascii_case("bro");
            if language && slang {
                (Ambiguous, ConflictingContext)
            } else if language {
                candidates = vec![FRENCH_LANGUAGE];
                (Resolved, LanguageContext)
            } else if surface != "fr" {
                (Ambiguous, CaseAmbiguity)
            } else if slang {
                candidates = vec![AGREEMENT_EMPHASIS];
                (Resolved, SlangContext)
            } else {
                (Ambiguous, InsufficientContext)
            }
        }
        ResolutionRule::KaContext => {
            let previous = adjacent_word(input, start, true);
            let next = adjacent_word(input, end, false);
            let german = previous.eq_ignore_ascii_case("ich")
                || ["was", "wie", "warum", "ob", "wer", "wo", "wann"]
                    .iter()
                    .any(|word| next.eq_ignore_ascii_case(word));
            if surface != "ka" {
                (Ambiguous, CaseAmbiguity)
            } else if german {
                (Resolved, SlangContext)
            } else {
                (Ambiguous, InsufficientContext)
            }
        }
    };
    SemanticAnnotation {
        start,
        end,
        candidates,
        decision,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ids_and_multilingual_variants_preserve_nuance() {
        let registry = SemanticRegistry::builtin();
        for text in [
            "idk",
            "IDK",
            "dunno",
            "I don't know",
            "I don’t know",
            "keine Ahnung",
            "ich ka",
            "ka warum",
        ] {
            let a = registry.annotate(text);
            assert_eq!(a.len(), 1, "{text}");
            assert_eq!(a[0].resolved_id(), Some(UNCERTAINTY), "{text}");
        }
        let a = registry.annotate("ong idk bro tbh ngl fr bro");
        assert_eq!(
            a.iter()
                .map(SemanticAnnotation::resolved_id)
                .collect::<Vec<_>>(),
            vec![
                Some(SWEARING_EMPHASIS),
                Some(UNCERTAINTY),
                Some(HONESTY_MARKER),
                Some(HONESTY_MARKER),
                Some(AGREEMENT_EMPHASIS)
            ]
        );
        assert_eq!(registry.concept(SemanticId(1)).unwrap().name, "UNCERTAINTY");
        assert!(registry.concept(SemanticId(999)).is_none());
        assert_eq!(registry.concepts().len(), 5);
        assert_eq!(registry.variants().len(), 14);
    }

    #[test]
    fn ambiguity_context_and_case_are_explicit() {
        let r = SemanticRegistry::builtin();
        for text in [
            "fr",
            "FR",
            "ka",
            "KA",
            "KA was",
            "ONG",
            "fr\nbro",
            "bro\rfr",
            "fr\u{2028}bro",
            "fr brother",
            "fr bro2",
            "fr bro_name",
        ] {
            let a = r.annotate(text);
            assert_eq!(a.len(), 1, "{text}");
            assert_eq!(a[0].resolved_id(), None, "{text}");
        }
        for text in [
            "language: fr",
            "lang=FR",
            "Sprache: fr",
            "\"language\": \"fr\"",
        ] {
            assert_eq!(
                r.annotate(text)[0].resolved_id(),
                Some(FRENCH_LANGUAGE),
                "{text}"
            );
        }
        assert_eq!(
            r.annotate("language: fr bro")[0].reason,
            SemanticReason::ConflictingContext
        );
        for text in [
            "mylanguage: fr",
            "language\n: fr",
            "language: \nfr",
            "language: xx fr",
        ] {
            assert_eq!(r.annotate(text)[0].resolved_id(), None, "{text}");
        }
    }

    #[test]
    fn longest_match_unicode_and_unknown_boundaries() {
        let r = SemanticRegistry::builtin();
        let text = "👩🏽‍💻 I\u{00a0}don’t\tknow! keine  Ahnung 🙂";
        let a = r.annotate(text);
        assert_eq!(a.len(), 2);
        assert_eq!(&text[a[0].start..a[0].end], "I\u{00a0}don’t\tknow");
        assert_eq!(&text[a[1].start..a[1].end], "keine  Ahnung");
        for text in [
            "",
            "xidk idk2 idk_name",
            "äidk idkä",
            "idk\u{0301}",
            "idk\u{200d}x",
            "idk’s",
            "i\ndon't\nknow",
            "漢idk字",
            "zzzz",
            "🦦",
        ] {
            assert!(r.annotate(text).is_empty(), "{text}");
        }
    }

    #[test]
    fn registry_rejects_invalid_data_and_is_order_independent() {
        let r = SemanticRegistry::builtin();
        let mut concepts = r.concepts().to_vec();
        concepts.reverse();
        let mut variants = r.variants().to_vec();
        variants.reverse();
        assert_eq!(
            &SemanticRegistry::new(
                r.namespace(),
                r.revision(),
                concepts.clone(),
                variants.clone()
            )
            .unwrap(),
            r
        );
        concepts.push(concepts[0].clone());
        assert!(SemanticRegistry::new("test", 1, concepts, variants.clone()).is_err());
        variants[0].candidates = vec![SemanticId(12345)];
        assert!(SemanticRegistry::new("test", 1, r.concepts().to_vec(), variants).is_err());
        for pattern in ["", " idk", "idk ", "IDK", "idk  bro", "ä", "foo2"] {
            let variants = vec![SemanticVariant {
                pattern: pattern.into(),
                candidates: vec![UNCERTAINTY],
                rule: ResolutionRule::Direct,
            }];
            assert!(
                SemanticRegistry::new("test", 1, r.concepts().to_vec(), variants).is_err(),
                "{pattern}"
            );
        }
        let mut variants = r.variants().to_vec();
        variants.push(variants[0].clone());
        assert!(SemanticRegistry::new("test", 1, r.concepts().to_vec(), variants).is_err());
    }

    #[test]
    fn regression_context_budget_never_fabricates_a_key_boundary() {
        let r = SemanticRegistry::builtin();
        for (key, spaces) in [("xlanguage", 119), ("xlang", 123), ("xsprache", 120)] {
            for extra in 0..=2 {
                let text = format!("{key}:{}fr", " ".repeat(spaces + extra));
                assert_eq!(r.annotate(&text)[0].resolved_id(), None, "{text:?}");
            }
        }
    }

    #[test]
    fn regression_vertical_separators_cannot_supply_context_or_phrase_spaces() {
        let r = SemanticRegistry::builtin();
        for separator in [
            '\u{000b}', '\u{000c}', '\r', '\n', '\u{0085}', '\u{2028}', '\u{2029}',
        ] {
            for text in [
                format!("fr{separator}bro"),
                format!("ka{separator}was"),
                format!("language:{separator}fr"),
            ] {
                assert_eq!(r.annotate(&text)[0].resolved_id(), None, "{text:?}");
            }
            assert!(r.annotate(&format!("keine{separator}ahnung")).is_empty());
        }
    }

    #[test]
    fn regression_quoted_metadata_keeps_apostrophe_word_protection() {
        let r = SemanticRegistry::builtin();
        for text in [
            "language: 'fr'",
            "'language': \"fr\"",
            "'language': 'fr'",
            "language: ’fr’",
        ] {
            let a = r.annotate(text);
            assert_eq!(a.len(), 1, "{text}");
            assert_eq!(a[0].resolved_id(), Some(FRENCH_LANGUAGE), "{text}");
        }
        for text in ["idk's", "idk’s", "someone'idk", "someone’idk"] {
            assert!(r.annotate(text).is_empty(), "{text}");
        }
    }

    #[test]
    fn regression_lowercase_rule_is_independent_of_spacing_and_apostrophe_style() {
        let r = SemanticRegistry::builtin();
        let variants = vec![SemanticVariant {
            pattern: "don't know".into(),
            candidates: vec![UNCERTAINTY],
            rule: ResolutionRule::LowercaseOnly,
        }];
        let custom =
            SemanticRegistry::new("test.custom", 1, r.concepts().to_vec(), variants).unwrap();
        for text in [
            "don't know",
            "don't  know",
            "don’t\tknow",
            "don't\u{00a0}know",
        ] {
            assert_eq!(
                custom.annotate(text)[0].resolved_id(),
                Some(UNCERTAINTY),
                "{text}"
            );
        }
        assert_eq!(
            custom.annotate("DON'T KNOW")[0].reason,
            SemanticReason::CaseAmbiguity
        );
    }

    #[test]
    fn deterministic_large_input_and_bounded_context() {
        let r = SemanticRegistry::builtin();
        let text = format!("{} fr{}bro idk", "x".repeat(500_000), " ".repeat(500_000));
        let a = r.annotate(&text);
        assert_eq!(a, r.annotate(&text));
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].resolved_id(), None);
        assert_eq!(a[1].resolved_id(), Some(UNCERTAINTY));
    }
}
