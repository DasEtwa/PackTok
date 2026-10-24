//! Semantic Packs v1 sidecar, independent of the lexical artifact versions.
//!
//! All integers are little endian. The 40-byte header contains `SPACKS\0\0`,
//! version (`u16`), reserved flags (`u16`, zero), registry fingerprint (`u64`),
//! original byte length (`u64`), original byte FNV-1a fingerprint (`u64`), and
//! annotation count (`u32`). Each record contains start/end byte offsets (`u64`),
//! decision/reason (`u8` each), candidate count (`u16`), then candidate IDs (`u32`).
//! Decisions are resolved=1 and ambiguous=2; reasons are documented in `reason_code`.
//! FNV is a reproducibility fingerprint, not an authenticity/security checksum.
//! Loading also regenerates annotations to reject stale or fabricated decisions.

use std::fmt;

use packtok_packs::semantic::{
    SemanticAnnotation, SemanticDecision, SemanticId, SemanticReason, SemanticRegistry,
};

/// This version applies only to semantic sidecars, never to tokenizer artifacts.
pub const SEMANTIC_SIDECAR_VERSION: u16 = 1;
/// Maximum accepted or emitted semantic sidecar size, including the header.
pub const MAX_SEMANTIC_SIDECAR_BYTES: usize = 16 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"SPACKS\0\0";
const HEADER_BYTES: usize = 40;
const RECORD_HEADER_BYTES: usize = 20;
const MIN_RECORD_BYTES: usize = RECORD_HEADER_BYTES + 4;

/// Canonical byte-span annotations bound to an original UTF-8 input and registry.
///
/// Decoding the original text uses lexical tokens alone. This object does not
/// replace lexical tokens, store normalized text, or consume lexical IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticSidecar {
    registry_fingerprint: u64,
    input_len: u64,
    input_fingerprint: u64,
    annotations: Vec<SemanticAnnotation>,
}

impl SemanticSidecar {
    /// Annotates the supplied input without changing any original byte.
    #[must_use]
    pub fn annotate(input: &str, registry: &SemanticRegistry) -> Self {
        Self {
            registry_fingerprint: registry.fingerprint(),
            input_len: input.len() as u64,
            input_fingerprint: fingerprint(input.as_bytes()),
            annotations: registry.annotate(input),
        }
    }

    /// Ordered, non-overlapping byte spans with their semantic decisions.
    #[must_use]
    pub fn annotations(&self) -> &[SemanticAnnotation] {
        &self.annotations
    }

    /// Fingerprint of semantic definitions and the context resolver policy.
    #[must_use]
    pub const fn registry_fingerprint(&self) -> u64 {
        self.registry_fingerprint
    }

    /// Original input byte length; offsets refer to these bytes.
    #[must_use]
    pub const fn input_len(&self) -> u64 {
        self.input_len
    }

    /// FNV-1a fingerprint of the exact original bytes, without normalization.
    #[must_use]
    pub const fn input_fingerprint(&self) -> u64 {
        self.input_fingerprint
    }

    /// Serializes this sidecar deterministically without touching lexical data.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SemanticSidecarError> {
        let mut size = HEADER_BYTES;
        for annotation in &self.annotations {
            let candidates = annotation.candidates.len();
            if candidates == 0 || candidates > usize::from(u16::MAX) {
                return Err(SemanticSidecarError::InvalidRecord);
            }
            size = size
                .checked_add(RECORD_HEADER_BYTES)
                .and_then(|value| value.checked_add(candidates.checked_mul(4)?))
                .ok_or(SemanticSidecarError::TooLarge)?;
            if size > MAX_SEMANTIC_SIDECAR_BYTES {
                return Err(SemanticSidecarError::TooLarge);
            }
        }
        let count =
            u32::try_from(self.annotations.len()).map_err(|_| SemanticSidecarError::TooLarge)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| SemanticSidecarError::AllocationFailed)?;
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&SEMANTIC_SIDECAR_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&self.registry_fingerprint.to_le_bytes());
        bytes.extend_from_slice(&self.input_len.to_le_bytes());
        bytes.extend_from_slice(&self.input_fingerprint.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        for annotation in &self.annotations {
            bytes.extend_from_slice(&(annotation.start as u64).to_le_bytes());
            bytes.extend_from_slice(&(annotation.end as u64).to_le_bytes());
            bytes.push(match annotation.decision {
                SemanticDecision::Resolved => 1,
                SemanticDecision::Ambiguous => 2,
            });
            bytes.push(reason_code(annotation.reason));
            bytes.extend_from_slice(&(annotation.candidates.len() as u16).to_le_bytes());
            for candidate in &annotation.candidates {
                bytes.extend_from_slice(&candidate.0.to_le_bytes());
            }
        }
        Ok(bytes)
    }

    /// Loads a canonical sidecar for the supplied original input and registry.
    ///
    /// Unknown versions/flags, trailing bytes, malformed spans, unknown IDs,
    /// incorrect decisions and mismatched input or registry are rejected. Count
    /// fields are checked against actual remaining bytes before reserving memory.
    pub fn from_bytes(
        bytes: &[u8],
        input: &str,
        registry: &SemanticRegistry,
    ) -> Result<Self, SemanticSidecarError> {
        if bytes.len() > MAX_SEMANTIC_SIDECAR_BYTES {
            return Err(SemanticSidecarError::TooLarge);
        }
        let mut reader = Reader { bytes, position: 0 };
        if reader.read::<8>()? != *MAGIC {
            return Err(SemanticSidecarError::InvalidHeader);
        }
        if u16::from_le_bytes(reader.read()?) != SEMANTIC_SIDECAR_VERSION {
            return Err(SemanticSidecarError::UnsupportedVersion);
        }
        if u16::from_le_bytes(reader.read()?) != 0 {
            return Err(SemanticSidecarError::InvalidHeader);
        }
        let registry_fingerprint = u64::from_le_bytes(reader.read()?);
        let input_len = u64::from_le_bytes(reader.read()?);
        let input_fingerprint = u64::from_le_bytes(reader.read()?);
        if registry_fingerprint != registry.fingerprint() {
            return Err(SemanticSidecarError::RegistryMismatch);
        }
        if input_len != input.len() as u64 || input_fingerprint != fingerprint(input.as_bytes()) {
            return Err(SemanticSidecarError::InputMismatch);
        }
        let count = u32::from_le_bytes(reader.read()?) as usize;
        if count > reader.remaining() / MIN_RECORD_BYTES {
            return Err(SemanticSidecarError::InvalidRecord);
        }
        let mut annotations = Vec::new();
        annotations
            .try_reserve_exact(count)
            .map_err(|_| SemanticSidecarError::AllocationFailed)?;
        let mut previous_end = 0;
        for _ in 0..count {
            let start = usize::try_from(u64::from_le_bytes(reader.read()?))
                .map_err(|_| SemanticSidecarError::InvalidRecord)?;
            let end = usize::try_from(u64::from_le_bytes(reader.read()?))
                .map_err(|_| SemanticSidecarError::InvalidRecord)?;
            if start < previous_end
                || start >= end
                || end > input.len()
                || !input.is_char_boundary(start)
                || !input.is_char_boundary(end)
            {
                return Err(SemanticSidecarError::InvalidRecord);
            }
            let decision = match reader.read::<1>()?[0] {
                1 => SemanticDecision::Resolved,
                2 => SemanticDecision::Ambiguous,
                _ => return Err(SemanticSidecarError::InvalidRecord),
            };
            let reason = read_reason(reader.read::<1>()?[0])?;
            let candidate_count = usize::from(u16::from_le_bytes(reader.read()?));
            if candidate_count == 0 || candidate_count > reader.remaining() / 4 {
                return Err(SemanticSidecarError::InvalidRecord);
            }
            if decision == SemanticDecision::Resolved && candidate_count != 1 {
                return Err(SemanticSidecarError::InvalidRecord);
            }
            let mut candidates = Vec::new();
            candidates
                .try_reserve_exact(candidate_count)
                .map_err(|_| SemanticSidecarError::AllocationFailed)?;
            for _ in 0..candidate_count {
                let id = SemanticId(u32::from_le_bytes(reader.read()?));
                if registry.concept(id).is_none()
                    || candidates.last().is_some_and(|previous| *previous >= id)
                {
                    return Err(SemanticSidecarError::InvalidRecord);
                }
                candidates.push(id);
            }
            annotations.push(SemanticAnnotation {
                start,
                end,
                candidates,
                decision,
                reason,
            });
            previous_end = end;
        }
        if reader.remaining() != 0 {
            return Err(SemanticSidecarError::TrailingBytes);
        }
        if annotations != registry.annotate(input) {
            return Err(SemanticSidecarError::NoncanonicalAnnotations);
        }
        Ok(Self {
            registry_fingerprint,
            input_len,
            input_fingerprint,
            annotations,
        })
    }
}

/// Failure while reading or writing a semantic sidecar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticSidecarError {
    /// File does not contain the full expected structure.
    Truncated,
    /// Magic or reserved header flags do not match this format.
    InvalidHeader,
    /// Sidecar version is unsupported.
    UnsupportedVersion,
    /// File or emitted data exceeds the documented bound.
    TooLarge,
    /// Registry definitions or resolver policy changed.
    RegistryMismatch,
    /// Original byte length or reproducibility fingerprint does not match.
    InputMismatch,
    /// Offset, ordering, candidate or decision fields are invalid.
    InvalidRecord,
    /// A valid record sequence is followed by unrecognized data.
    TrailingBytes,
    /// Annotations do not match the deterministic registry and context rules.
    NoncanonicalAnnotations,
    /// Memory reservation failed.
    AllocationFailed,
}

impl fmt::Display for SemanticSidecarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Truncated => "truncated semantic sidecar",
            Self::InvalidHeader => "invalid semantic sidecar header",
            Self::UnsupportedVersion => "unsupported semantic sidecar version",
            Self::TooLarge => "semantic sidecar exceeds 16 MiB limit",
            Self::RegistryMismatch => "semantic sidecar registry or resolver mismatch",
            Self::InputMismatch => "semantic sidecar original input mismatch",
            Self::InvalidRecord => "invalid semantic sidecar annotation record",
            Self::TrailingBytes => "unexpected bytes after semantic sidecar records",
            Self::NoncanonicalAnnotations => "semantic sidecar annotations are not canonical",
            Self::AllocationFailed => "could not reserve semantic sidecar memory",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for SemanticSidecarError {}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Reader<'_> {
    fn read<const N: usize>(&mut self) -> Result<[u8; N], SemanticSidecarError> {
        let end = self
            .position
            .checked_add(N)
            .ok_or(SemanticSidecarError::Truncated)?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(SemanticSidecarError::Truncated)?;
        let mut result = [0; N];
        result.copy_from_slice(slice);
        self.position = end;
        Ok(result)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }
}

fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

// Explicit discriminants freeze the wire contract independently of Rust enums.
fn reason_code(reason: SemanticReason) -> u8 {
    match reason {
        SemanticReason::KnownVariant => 1,
        SemanticReason::SlangContext => 2,
        SemanticReason::LanguageContext => 3,
        SemanticReason::InsufficientContext => 4,
        SemanticReason::CaseAmbiguity => 5,
        SemanticReason::ConflictingContext => 6,
    }
}

fn read_reason(code: u8) -> Result<SemanticReason, SemanticSidecarError> {
    match code {
        1 => Ok(SemanticReason::KnownVariant),
        2 => Ok(SemanticReason::SlangContext),
        3 => Ok(SemanticReason::LanguageContext),
        4 => Ok(SemanticReason::InsufficientContext),
        5 => Ok(SemanticReason::CaseAmbiguity),
        6 => Ok(SemanticReason::ConflictingContext),
        _ => Err(SemanticSidecarError::InvalidRecord),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sidecar(input: &str) -> SemanticSidecar {
        SemanticSidecar::annotate(input, SemanticRegistry::builtin())
    }

    fn load(bytes: &[u8], input: &str) -> Result<SemanticSidecar, SemanticSidecarError> {
        SemanticSidecar::from_bytes(bytes, input, SemanticRegistry::builtin())
    }

    #[test]
    fn canonical_serialization_preserves_annotations_and_original_byte_spans() {
        for input in [
            "",
            "plain German: sämtliche Grüße äöüß",
            "ong idk bro",
            "ka, keine Ahnung; fr bro; language: fr; KA; fr",
            "✨👩🏽‍💻 IDK\tngl\r\nI don't know\u{00a0}tbh\0",
        ] {
            let first = sidecar(input);
            let bytes = first.to_bytes().expect("serialize");
            assert_eq!(bytes, sidecar(input).to_bytes().expect("repeat serialize"));
            assert_eq!(load(&bytes, input).expect("reload"), first);
            assert_eq!(first.input_len(), input.len() as u64);
            assert_eq!(&bytes[..8], MAGIC);
            assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 1);
            for annotation in first.annotations() {
                assert!(input.get(annotation.start..annotation.end).is_some());
            }
        }
    }

    #[test]
    fn truncation_trailing_data_header_and_count_mutations_are_rejected() {
        let input = "ong idk bro";
        let original = sidecar(input).to_bytes().expect("serialize");
        for length in 0..original.len() {
            assert!(load(&original[..length], input).is_err(), "length {length}");
        }
        let mut trailing = original.clone();
        trailing.push(0);
        assert_eq!(
            load(&trailing, input),
            Err(SemanticSidecarError::TrailingBytes)
        );
        for offset in [0, 8, 10, 12, 20, 28, 36] {
            let mut changed = original.clone();
            changed[offset] ^= 0xff;
            assert!(load(&changed, input).is_err(), "header offset {offset}");
        }
        let mut count = original.clone();
        count[36..40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            load(&count, input),
            Err(SemanticSidecarError::InvalidRecord)
        );
    }

    #[test]
    fn input_binding_and_noncanonical_semantics_are_rejected() {
        let input = "idk";
        let original = sidecar(input).to_bytes().expect("serialize");
        assert!(original.len() > HEADER_BYTES);
        assert_eq!(
            load(&original, "tbh"),
            Err(SemanticSidecarError::InputMismatch)
        );
        assert_eq!(
            load(&original, "idk "),
            Err(SemanticSidecarError::InputMismatch)
        );
        // Another valid concept ID cannot forge a semantic decision.
        let mut forged = original.clone();
        forged[60..64].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            load(&forged, input),
            Err(SemanticSidecarError::NoncanonicalAnnotations)
        );
        // Likewise a valid but inappropriate reason is rejected.
        let mut reason = original.clone();
        reason[57] = 3;
        assert_eq!(
            load(&reason, input),
            Err(SemanticSidecarError::NoncanonicalAnnotations)
        );
    }

    #[test]
    fn invalid_spans_candidates_and_discriminants_are_rejected() {
        let input = "✨idk idk";
        let original = sidecar(input).to_bytes().expect("serialize");
        assert!(original.len() >= HEADER_BYTES + 2 * MIN_RECORD_BYTES);
        for (offset, value) in [(40, 1), (48, 255), (56, 0), (57, 255), (58, 0), (60, 255)] {
            let mut changed = original.clone();
            changed[offset] = value;
            assert!(load(&changed, input).is_err(), "record offset {offset}");
        }
        let mut overlap = original;
        overlap[64..72].copy_from_slice(&3_u64.to_le_bytes());
        assert!(load(&overlap, input).is_err());
    }

    #[test]
    fn hostile_byte_inputs_and_single_byte_mutations_never_panic() {
        let input = "fr bro; KA; idk ✨";
        let original = sidecar(input).to_bytes().expect("serialize");
        for offset in 0..original.len() {
            for value in [0, 1, 127, 255] {
                let mut mutated = original.clone();
                mutated[offset] = value;
                let _ = load(&mutated, input);
            }
        }
        let mut state = 0x6d42_4f27_u32;
        for length in 0..2048 {
            let mut bytes = vec![0; length];
            for byte in &mut bytes {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }
            assert!(load(&bytes, input).is_err());
        }
    }

    #[test]
    fn sidecar_limits_and_empty_input_wire_size_are_explicit() {
        assert_eq!(sidecar("").to_bytes().expect("empty").len(), HEADER_BYTES);
        assert_eq!(
            load(&vec![0; MAX_SEMANTIC_SIDECAR_BYTES + 1], ""),
            Err(SemanticSidecarError::TooLarge)
        );
    }
}
