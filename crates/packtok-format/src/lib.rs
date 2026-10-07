#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use packtok_core::{
    ByteFallback, MAX_PACK_COUNT, PackDescriptor, PackRegistry, SpecialToken, TokenId,
    ValidationError,
};

/// Historical M0 format version retained for source compatibility.
pub const FORMAT_VERSION: u16 = 1;

/// Version that adds the flat byte-level BPE model section.
pub const FLAT_BPE_FORMAT_VERSION: u16 = 2;

/// Single-pack address used for the flat M1 vocabulary.
pub const FLAT_BPE_PACK_ID: u16 = 0;

/// Maximum accepted or emitted artifact size.
pub const MAX_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

/// Practical expansion bound for a single BPE token, independent of pointer width.
pub const MAX_BPE_TOKEN_BYTES: usize = 1024 * 1024;

/// Number of fixed byte symbols at the start of every flat BPE vocabulary.
pub const BYTE_TOKEN_COUNT: u32 = 256;

const MAGIC: &[u8; 8] = b"PACKTOK\0";
const HEADER_BYTES: usize = 20;
const MAX_COLLECTION_ITEMS: usize = 65_536;

/// A minimal, validated tokenizer artifact.
///
/// Metadata values are descriptive. Runtime behavior is defined by the artifact's
/// validated registry and, for version 2, its flat BPE merge table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Artifact {
    registry: PackRegistry,
    metadata: BTreeMap<String, String>,
    format_version: u16,
    flat_bpe: Option<FlatBpeModel>,
}

/// One normative flat BPE merge. `result` is assigned sequentially from 256 in
/// merge-rank order; `left` and `right` must refer to earlier vocabulary IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlatBpeMerge {
    /// Earlier token placed on the left side of this merge.
    pub left: u32,
    /// Earlier token placed on the right side of this merge.
    pub right: u32,
    /// Token ID assigned by this merge rank.
    pub result: u32,
}

/// Ordered, validated BPE merge table stored in a version-2 artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlatBpeModel {
    merges: Vec<FlatBpeMerge>,
    vocabulary_size: u32,
    byte_lengths: Vec<usize>,
}

impl FlatBpeModel {
    /// Validates merge ranks, references, unique pairs, and expanded byte lengths.
    pub fn new(merges: Vec<FlatBpeMerge>) -> Result<Self, BpeModelError> {
        let merge_count = u32::try_from(merges.len()).map_err(|_| BpeModelError::TooManyMerges)?;
        let vocabulary_size = BYTE_TOKEN_COUNT
            .checked_add(merge_count)
            .ok_or(BpeModelError::TooManyMerges)?;
        let mut byte_lengths = Vec::new();
        byte_lengths
            .try_reserve_exact(
                usize::try_from(vocabulary_size).map_err(|_| BpeModelError::TooManyMerges)?,
            )
            .map_err(|_| BpeModelError::AllocationFailed)?;
        let byte_count =
            usize::try_from(BYTE_TOKEN_COUNT).map_err(|_| BpeModelError::TooManyMerges)?;
        byte_lengths.extend(std::iter::repeat_n(1_usize, byte_count));
        let mut seen_pairs = BTreeSet::new();

        for (rank, merge) in merges.iter().enumerate() {
            let rank = u32::try_from(rank).map_err(|_| BpeModelError::TooManyMerges)?;
            let expected = BYTE_TOKEN_COUNT
                .checked_add(rank)
                .ok_or(BpeModelError::TooManyMerges)?;
            if merge.result != expected {
                return Err(BpeModelError::InvalidResultId {
                    rank,
                    expected,
                    actual: merge.result,
                });
            }
            if merge.left >= expected || merge.right >= expected {
                return Err(BpeModelError::InvalidParent {
                    rank,
                    result: expected,
                    left: merge.left,
                    right: merge.right,
                });
            }
            if !seen_pairs.insert((merge.left, merge.right)) {
                return Err(BpeModelError::DuplicatePair {
                    left: merge.left,
                    right: merge.right,
                });
            }
            let left_index =
                usize::try_from(merge.left).map_err(|_| BpeModelError::TooManyMerges)?;
            let right_index =
                usize::try_from(merge.right).map_err(|_| BpeModelError::TooManyMerges)?;
            let length = byte_lengths[left_index]
                .checked_add(byte_lengths[right_index])
                .ok_or(BpeModelError::ExpandedTokenTooLarge { token_id: expected })?;
            if length > MAX_BPE_TOKEN_BYTES {
                return Err(BpeModelError::ExpandedTokenTooLarge { token_id: expected });
            }
            byte_lengths.push(length);
        }

        Ok(Self {
            merges,
            vocabulary_size,
            byte_lengths,
        })
    }

    /// Returns merges in rank order; each rank creates token `256 + rank`.
    #[must_use]
    pub fn merges(&self) -> &[FlatBpeMerge] {
        &self.merges
    }

    /// Returns the total vocabulary size including all 256 byte tokens.
    #[must_use]
    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    /// Returns the exact number of bytes represented by an in-range token.
    #[must_use]
    pub fn byte_length(&self, token_id: u32) -> Option<usize> {
        self.byte_lengths
            .get(usize::try_from(token_id).ok()?)
            .copied()
    }
}

/// Validation failure for a flat BPE merge table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BpeModelError {
    /// The merge count cannot fit the format's local-ID space.
    TooManyMerges,
    /// A merge result does not equal the ID assigned by its rank.
    InvalidResultId {
        rank: u32,
        expected: u32,
        actual: u32,
    },
    /// A merge refers to a token that does not exist before its result.
    InvalidParent {
        rank: u32,
        result: u32,
        left: u32,
        right: u32,
    },
    /// A previously selected pair occurs again in the merge table.
    DuplicatePair { left: u32, right: u32 },
    /// Expanding this token exceeds [`MAX_BPE_TOKEN_BYTES`] or the host byte-length type.
    ExpandedTokenTooLarge { token_id: u32 },
    /// Validation could not reserve its bounded lookup table.
    AllocationFailed,
}

impl fmt::Display for BpeModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyMerges => f.write_str("BPE merge count exceeds the local token ID space"),
            Self::InvalidResultId {
                rank,
                expected,
                actual,
            } => write!(
                f,
                "BPE merge rank {rank} must create token {expected}, got {actual}"
            ),
            Self::InvalidParent {
                rank,
                result,
                left,
                right,
            } => write!(
                f,
                "BPE merge rank {rank} creates {result} but references non-prior parents {left} and {right}"
            ),
            Self::DuplicatePair { left, right } => {
                write!(f, "BPE pair ({left}, {right}) is merged more than once")
            }
            Self::ExpandedTokenTooLarge { token_id } => write!(
                f,
                "expanded BPE token {token_id} exceeds the {MAX_BPE_TOKEN_BYTES}-byte token limit or host byte-length range"
            ),
            Self::AllocationFailed => f.write_str("could not allocate BPE validation storage"),
        }
    }
}

impl std::error::Error for BpeModelError {}

impl Artifact {
    /// Creates a version-1 artifact after validating metadata keys.
    pub fn new(
        registry: PackRegistry,
        metadata: BTreeMap<String, String>,
    ) -> Result<Self, FormatError> {
        validate_metadata(&metadata)?;
        let artifact = Self {
            registry,
            metadata,
            format_version: FORMAT_VERSION,
            flat_bpe: None,
        };
        artifact.serialized_size()?;
        Ok(artifact)
    }

    /// Creates a version-2 flat BPE artifact and checks that its registry matches
    /// the model's single flat vocabulary and independent raw-byte fallback pack.
    pub fn with_flat_bpe(
        registry: PackRegistry,
        metadata: BTreeMap<String, String>,
        flat_bpe: FlatBpeModel,
    ) -> Result<Self, FormatError> {
        validate_metadata(&metadata)?;
        validate_flat_bpe_registry(&registry, &flat_bpe)?;
        let artifact = Self {
            registry,
            metadata,
            format_version: FLAT_BPE_FORMAT_VERSION,
            flat_bpe: Some(flat_bpe),
        };
        artifact.serialized_size()?;
        Ok(artifact)
    }

    /// Returns the minimal artifact with the default raw-byte fallback pack.
    #[must_use]
    pub fn byte_fallback_only() -> Self {
        Self {
            registry: PackRegistry::byte_fallback_only(),
            metadata: BTreeMap::new(),
            format_version: FORMAT_VERSION,
            flat_bpe: None,
        }
    }

    /// Returns the artifact's validated pack registry.
    #[must_use]
    pub fn registry(&self) -> &PackRegistry {
        &self.registry
    }

    /// Returns opaque metadata in deterministic key order.
    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Returns the flat BPE model when this is a version-2 M1 artifact.
    #[must_use]
    pub fn flat_bpe(&self) -> Option<&FlatBpeModel> {
        self.flat_bpe.as_ref()
    }

    /// Returns the version used when this artifact is serialized.
    #[must_use]
    pub const fn format_version(&self) -> u16 {
        self.format_version
    }

    /// Serializes this artifact into its canonical versioned byte representation.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let size = self.serialized_size()?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(size)
            .map_err(|_| FormatError::AllocationFailed)?;

        output.extend_from_slice(MAGIC);
        write_u16(&mut output, self.format_version);
        write_u16(&mut output, 0); // flags reserved in versions 1 and 2
        write_u16(&mut output, self.registry.byte_fallback().pack_id());
        write_u16(&mut output, 0); // reserved
        write_count(&mut output, self.registry.packs().len())?;

        for pack in self.registry.packs() {
            write_u16(&mut output, pack.id());
            write_u32(&mut output, pack.local_token_count());
            write_string(&mut output, pack.name())?;
        }

        write_count(&mut output, self.registry.special_tokens().len())?;
        for special in self.registry.special_tokens() {
            write_u16(&mut output, special.id().pack);
            write_u32(&mut output, special.id().local);
            write_string(&mut output, special.name())?;
        }

        write_count(&mut output, self.metadata.len())?;
        for (key, value) in &self.metadata {
            write_string(&mut output, key)?;
            write_string(&mut output, value)?;
        }

        if let Some(model) = &self.flat_bpe {
            write_count(&mut output, model.merges().len())?;
            for merge in model.merges() {
                write_u32(&mut output, merge.left);
                write_u32(&mut output, merge.right);
                write_u32(&mut output, merge.result);
            }
        }

        debug_assert_eq!(output.len(), size);
        Ok(output)
    }

    /// Parses a canonical artifact and rejects unsupported or malformed definitions.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(FormatError::ArtifactTooLarge {
                size: bytes.len(),
                maximum: MAX_ARTIFACT_BYTES,
            });
        }

        let mut reader = Reader::new(bytes);
        if reader.read_bytes(MAGIC.len())? != MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        let version = reader.read_u16()?;
        if version != FORMAT_VERSION && version != FLAT_BPE_FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion { version });
        }

        let flags = reader.read_u16()?;
        if flags != 0 {
            return Err(FormatError::NonZeroFlags { flags });
        }

        let fallback_pack_id = reader.read_u16()?;
        let reserved = reader.read_u16()?;
        if reserved != 0 {
            return Err(FormatError::NonZeroReserved { value: reserved });
        }

        let pack_count = reader.read_count("packs", 10, MAX_PACK_COUNT)?;
        let mut packs = Vec::new();
        packs
            .try_reserve_exact(pack_count)
            .map_err(|_| FormatError::AllocationFailed)?;
        let mut previous_pack_id = None;
        for index in 0..pack_count {
            let id = reader.read_u16()?;
            if previous_pack_id.is_some_and(|previous| id <= previous) {
                return Err(FormatError::NonCanonicalOrder {
                    collection: "packs",
                    index,
                });
            }
            previous_pack_id = Some(id);

            let local_token_count = reader.read_u32()?;
            let name = reader.read_string("pack name", index)?;
            packs.push(PackDescriptor::new(id, name, local_token_count));
        }

        let special_count = reader.read_count("special tokens", 10, MAX_COLLECTION_ITEMS)?;
        let mut special_tokens = Vec::new();
        special_tokens
            .try_reserve_exact(special_count)
            .map_err(|_| FormatError::AllocationFailed)?;
        let mut previous_special_id = None;
        for index in 0..special_count {
            let id = TokenId::new(reader.read_u16()?, reader.read_u32()?);
            if previous_special_id.is_some_and(|previous| id <= previous) {
                return Err(FormatError::NonCanonicalOrder {
                    collection: "special tokens",
                    index,
                });
            }
            previous_special_id = Some(id);

            let name = reader.read_string("special-token name", index)?;
            special_tokens.push(SpecialToken::new(id, name));
        }

        let metadata_count = reader.read_count("metadata entries", 8, MAX_COLLECTION_ITEMS)?;
        let mut metadata = BTreeMap::new();
        let mut previous_key: Option<String> = None;
        for index in 0..metadata_count {
            let key = reader.read_string("metadata key", index)?;
            if key.is_empty() {
                return Err(FormatError::EmptyMetadataKey { index });
            }
            if previous_key
                .as_ref()
                .is_some_and(|previous| key <= *previous)
            {
                return Err(FormatError::NonCanonicalOrder {
                    collection: "metadata entries",
                    index,
                });
            }
            previous_key = Some(key.clone());

            let value = reader.read_string("metadata value", index)?;
            metadata.insert(key, value);
        }

        let registry =
            PackRegistry::new(packs, ByteFallback::new(fallback_pack_id), special_tokens)
                .map_err(FormatError::InvalidRegistry)?;
        let flat_bpe = if version == FLAT_BPE_FORMAT_VERSION {
            let max_merges = (MAX_ARTIFACT_BYTES - HEADER_BYTES) / 12;
            let count = reader.read_count("BPE merges", 12, max_merges)?;
            let mut merges = Vec::new();
            merges
                .try_reserve_exact(count)
                .map_err(|_| FormatError::AllocationFailed)?;
            for _ in 0..count {
                merges.push(FlatBpeMerge {
                    left: reader.read_u32()?,
                    right: reader.read_u32()?,
                    result: reader.read_u32()?,
                });
            }
            Some(FlatBpeModel::new(merges).map_err(FormatError::InvalidBpeModel)?)
        } else {
            None
        };

        if reader.remaining() != 0 {
            return Err(FormatError::TrailingBytes {
                count: reader.remaining(),
            });
        }

        match flat_bpe {
            Some(model) => Self::with_flat_bpe(registry, metadata, model),
            None => Self::new(registry, metadata),
        }
    }

    fn serialized_size(&self) -> Result<usize, FormatError> {
        validate_collection_size(
            "special tokens",
            self.registry.special_tokens().len(),
            MAX_COLLECTION_ITEMS,
        )?;
        validate_collection_size(
            "metadata entries",
            self.metadata.len(),
            MAX_COLLECTION_ITEMS,
        )?;
        let mut size = HEADER_BYTES
            .checked_add(4) // special-token count
            .and_then(|value| value.checked_add(4)) // metadata count
            .ok_or(FormatError::LengthOverflow)?;

        for pack in self.registry.packs() {
            size = checked_add(size, 10)?; // ID, token count, string length
            size = checked_add(size, checked_string_len(pack.name())?)?;
        }

        for special in self.registry.special_tokens() {
            size = checked_add(size, 10)?; // token ID and string length
            size = checked_add(size, checked_string_len(special.name())?)?;
        }

        for (key, value) in &self.metadata {
            size = checked_add(size, 8)?; // two string lengths
            size = checked_add(size, checked_string_len(key)?)?;
            size = checked_add(size, checked_string_len(value)?)?;
        }

        if let Some(model) = &self.flat_bpe {
            size = checked_add(size, 4)?; // BPE merge count
            size = checked_add(
                size,
                model
                    .merges()
                    .len()
                    .checked_mul(12)
                    .ok_or(FormatError::LengthOverflow)?,
            )?;
        }

        if size > MAX_ARTIFACT_BYTES {
            return Err(FormatError::ArtifactTooLarge {
                size,
                maximum: MAX_ARTIFACT_BYTES,
            });
        }
        Ok(size)
    }
}

fn validate_metadata(metadata: &BTreeMap<String, String>) -> Result<(), FormatError> {
    validate_collection_size("metadata entries", metadata.len(), MAX_COLLECTION_ITEMS)?;
    for (index, key) in metadata.keys().enumerate() {
        if key.is_empty() {
            return Err(FormatError::EmptyMetadataKey { index });
        }
    }
    Ok(())
}

fn validate_collection_size(
    collection: &'static str,
    count: usize,
    maximum: usize,
) -> Result<(), FormatError> {
    if count > maximum {
        return Err(FormatError::CollectionTooLarge {
            collection,
            count,
            maximum,
        });
    }
    Ok(())
}

fn validate_flat_bpe_registry(
    registry: &PackRegistry,
    model: &FlatBpeModel,
) -> Result<(), FormatError> {
    let fallback_id = registry.byte_fallback().pack_id();
    if fallback_id == FLAT_BPE_PACK_ID
        || registry.packs().len() != 2
        || !registry.special_tokens().is_empty()
    {
        return Err(FormatError::InvalidFlatBpeRegistry);
    }
    let flat_pack = registry
        .pack(FLAT_BPE_PACK_ID)
        .ok_or(FormatError::InvalidFlatBpeRegistry)?;
    if flat_pack.local_token_count() != model.vocabulary_size() {
        return Err(FormatError::FlatBpeVocabularySizeMismatch {
            declared: flat_pack.local_token_count(),
            actual: model.vocabulary_size(),
        });
    }
    Ok(())
}

fn checked_add(left: usize, right: usize) -> Result<usize, FormatError> {
    left.checked_add(right).ok_or(FormatError::LengthOverflow)
}

fn checked_string_len(value: &str) -> Result<usize, FormatError> {
    u32::try_from(value.len()).map_err(|_| FormatError::LengthOverflow)?;
    Ok(value.len())
}

fn write_count(output: &mut Vec<u8>, count: usize) -> Result<(), FormatError> {
    let count = u32::try_from(count).map_err(|_| FormatError::LengthOverflow)?;
    write_u32(output, count);
    Ok(())
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<(), FormatError> {
    let length = u32::try_from(value.len()).map_err(|_| FormatError::LengthOverflow)?;
    write_u32(output, length);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_bytes(&mut self, count: usize) -> Result<&'a [u8], FormatError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(FormatError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(FormatError::UnexpectedEof {
                offset: self.offset,
                needed: count,
                remaining: self.remaining(),
            });
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16, FormatError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, FormatError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_count(
        &mut self,
        collection: &'static str,
        minimum_record_bytes: usize,
        maximum: usize,
    ) -> Result<usize, FormatError> {
        let count = usize::try_from(self.read_u32()?).map_err(|_| FormatError::LengthOverflow)?;
        if count > maximum {
            return Err(FormatError::CollectionTooLarge {
                collection,
                count,
                maximum,
            });
        }

        let minimum_bytes = count
            .checked_mul(minimum_record_bytes)
            .ok_or(FormatError::LengthOverflow)?;
        if minimum_bytes > self.remaining() {
            return Err(FormatError::InsufficientBytesForCollection { collection, count });
        }
        Ok(count)
    }

    fn read_string(&mut self, field: &'static str, index: usize) -> Result<String, FormatError> {
        let length = usize::try_from(self.read_u32()?).map_err(|_| FormatError::LengthOverflow)?;
        let offset = self.offset;
        let bytes = self.read_bytes(length)?;
        let value = std::str::from_utf8(bytes).map_err(|_| FormatError::InvalidUtf8 {
            field,
            index,
            offset,
        })?;
        let mut owned = String::new();
        owned
            .try_reserve_exact(value.len())
            .map_err(|_| FormatError::AllocationFailed)?;
        owned.push_str(value);
        Ok(owned)
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

/// Error returned when an artifact is malformed, unsupported, or too large.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FormatError {
    /// The artifact exceeds the configured size limit.
    ArtifactTooLarge { size: usize, maximum: usize },
    /// The magic bytes do not identify a PackTok artifact.
    InvalidMagic,
    /// The artifact uses a version unsupported by this implementation.
    UnsupportedVersion { version: u16 },
    /// Reserved format flags are nonzero.
    NonZeroFlags { flags: u16 },
    /// A reserved format field is nonzero.
    NonZeroReserved { value: u16 },
    /// The input ended before a required field was available.
    UnexpectedEof {
        offset: usize,
        needed: usize,
        remaining: usize,
    },
    /// A collection count exceeds its defensive limit.
    CollectionTooLarge {
        collection: &'static str,
        count: usize,
        maximum: usize,
    },
    /// The remaining bytes cannot contain the declared number of records.
    InsufficientBytesForCollection {
        collection: &'static str,
        count: usize,
    },
    /// A serialized string is not valid UTF-8.
    InvalidUtf8 {
        field: &'static str,
        index: usize,
        offset: usize,
    },
    /// Records are not in the canonical order required by the artifact format.
    NonCanonicalOrder {
        collection: &'static str,
        index: usize,
    },
    /// A metadata key is empty.
    EmptyMetadataKey { index: usize },
    /// Extra bytes follow the final artifact record.
    TrailingBytes { count: usize },
    /// The artifact's pack declarations violate core invariants.
    InvalidRegistry(ValidationError),
    /// The flat BPE model violates a rank, reference, or expansion-size invariant.
    InvalidBpeModel(BpeModelError),
    /// A version-2 flat BPE artifact has an unsupported registry shape.
    InvalidFlatBpeRegistry,
    /// The declared flat vocabulary size differs from its merge table.
    FlatBpeVocabularySizeMismatch { declared: u32, actual: u32 },
    /// An integer or total-size calculation overflowed.
    LengthOverflow,
    /// The allocator could not reserve output or parse storage.
    AllocationFailed,
}

impl fmt::Display for FormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArtifactTooLarge { size, maximum } => {
                write!(
                    formatter,
                    "artifact size {size} exceeds limit {maximum} bytes"
                )
            }
            Self::InvalidMagic => formatter.write_str("invalid PackTok artifact magic"),
            Self::UnsupportedVersion { version } => {
                write!(formatter, "unsupported PackTok artifact version {version}")
            }
            Self::NonZeroFlags { flags } => {
                write!(
                    formatter,
                    "artifact has nonzero reserved flags {flags:#06x}"
                )
            }
            Self::NonZeroReserved { value } => {
                write!(
                    formatter,
                    "artifact has nonzero reserved field {value}"
                )
            }
            Self::UnexpectedEof {
                offset,
                needed,
                remaining,
            } => write!(
                formatter,
                "truncated artifact at byte {offset}: field needs {needed} bytes, only {remaining} remain"
            ),
            Self::CollectionTooLarge {
                collection,
                count,
                maximum,
            } => write!(
                formatter,
                "artifact declares {count} {collection}; maximum is {maximum}"
            ),
            Self::InsufficientBytesForCollection { collection, count } => write!(
                formatter,
                "artifact cannot contain the declared {count} {collection}"
            ),
            Self::InvalidUtf8 {
                field,
                index,
                offset,
            } => write!(
                formatter,
                "{field} {index} is not UTF-8 (starts at byte {offset})"
            ),
            Self::NonCanonicalOrder { collection, index } => write!(
                formatter,
                "{collection} are not in canonical order at entry {index}"
            ),
            Self::EmptyMetadataKey { index } => {
                write!(formatter, "metadata key {index} is empty")
            }
            Self::TrailingBytes { count } => {
                write!(formatter, "artifact has {count} trailing bytes")
            }
            Self::InvalidRegistry(error) => write!(formatter, "invalid pack registry: {error}"),
            Self::InvalidBpeModel(error) => write!(formatter, "invalid flat BPE model: {error}"),
            Self::InvalidFlatBpeRegistry => formatter.write_str("flat BPE artifact must declare one pack 0 vocabulary, one separate byte fallback pack, and no special tokens"),
            Self::FlatBpeVocabularySizeMismatch { declared, actual } => write!(formatter, "flat BPE pack declares {declared} tokens but its merge table defines {actual}"),
            Self::LengthOverflow => formatter.write_str("artifact length exceeds supported limits"),
            Self::AllocationFailed => formatter.write_str("could not allocate artifact storage"),
        }
    }
}

impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidRegistry(error) => Some(error),
            Self::InvalidBpeModel(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packtok_core::BYTE_FALLBACK_TOKEN_COUNT;
    use packtok_core::ValidationError;

    fn sample_artifact(reverse: bool) -> Artifact {
        let mut packs = vec![
            PackDescriptor::new(4, "segment-x", 16),
            PackDescriptor::new(65000, "raw-bytes", BYTE_FALLBACK_TOKEN_COUNT),
        ];
        let mut specials = vec![SpecialToken::new(TokenId::new(4, 15), "control-end")];
        if reverse {
            packs.reverse();
            specials.reverse();
        }

        let registry = PackRegistry::new(packs, ByteFallback::new(65000), specials)
            .expect("valid sample registry");
        let mut metadata = BTreeMap::new();
        if reverse {
            metadata.insert("z-key".to_owned(), "last".to_owned());
            metadata.insert("a-key".to_owned(), "first".to_owned());
        } else {
            metadata.insert("a-key".to_owned(), "first".to_owned());
            metadata.insert("z-key".to_owned(), "last".to_owned());
        }
        Artifact::new(registry, metadata).expect("valid sample artifact")
    }

    #[test]
    fn artifact_round_trip_preserves_semantics() {
        let artifact = sample_artifact(false);
        let bytes = artifact.to_bytes().expect("serialize artifact");
        let decoded = Artifact::from_bytes(&bytes).expect("deserialize artifact");
        assert_eq!(decoded, artifact);
    }

    #[test]
    fn serialization_is_deterministic_for_equivalent_input_order() {
        let first = sample_artifact(false).to_bytes().expect("serialize first");
        let second = sample_artifact(true).to_bytes().expect("serialize second");
        assert_eq!(first, second);
    }

    #[test]
    fn byte_fallback_artifact_has_stable_header() {
        let bytes = Artifact::byte_fallback_only()
            .to_bytes()
            .expect("serialize default artifact");
        assert_eq!(&bytes[..8], MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), FORMAT_VERSION);
        assert_eq!(
            bytes.len(),
            HEADER_BYTES + 10 + "BYTE_FALLBACK".len() + 4 + 4
        );
        assert_eq!(
            Artifact::from_bytes(&bytes),
            Ok(Artifact::byte_fallback_only())
        );
    }

    #[test]
    fn malformed_artifacts_fail_clearly() {
        assert_eq!(
            Artifact::from_bytes(&[]),
            Err(FormatError::UnexpectedEof {
                offset: 0,
                needed: MAGIC.len(),
                remaining: 0,
            })
        );

        let valid = Artifact::byte_fallback_only()
            .to_bytes()
            .expect("serialize default artifact");

        let mut bad_magic = valid.clone();
        bad_magic[0] ^= 0xff;
        assert_eq!(
            Artifact::from_bytes(&bad_magic),
            Err(FormatError::InvalidMagic)
        );

        let mut unknown_version = valid.clone();
        unknown_version[8..10].copy_from_slice(&3_u16.to_le_bytes());
        assert_eq!(
            Artifact::from_bytes(&unknown_version),
            Err(FormatError::UnsupportedVersion { version: 3 })
        );

        let mut nonzero_flags = valid.clone();
        nonzero_flags[10..12].copy_from_slice(&1_u16.to_le_bytes());
        assert_eq!(
            Artifact::from_bytes(&nonzero_flags),
            Err(FormatError::NonZeroFlags { flags: 1 })
        );

        let mut truncated = valid.clone();
        truncated.pop();
        assert!(matches!(
            Artifact::from_bytes(&truncated),
            Err(FormatError::UnexpectedEof { .. })
        ));

        let mut trailing = valid;
        trailing.push(0);
        assert_eq!(
            Artifact::from_bytes(&trailing),
            Err(FormatError::TrailingBytes { count: 1 })
        );
    }

    #[test]
    fn parser_rejects_noncanonical_metadata_order() {
        let mut bytes = sample_artifact(false)
            .to_bytes()
            .expect("serialize artifact");
        let key = b"a-key";
        let key_offset = bytes
            .windows(key.len())
            .position(|window| window == key)
            .expect("metadata key is in serialized artifact");
        bytes[key_offset..key_offset + key.len()].copy_from_slice(b"z-key");
        assert_eq!(
            Artifact::from_bytes(&bytes),
            Err(FormatError::NonCanonicalOrder {
                collection: "metadata entries",
                index: 1,
            })
        );
    }

    #[test]
    fn parser_rejects_malformed_pack_definitions() {
        let mut invalid_fallback_size = Artifact::byte_fallback_only()
            .to_bytes()
            .expect("serialize default artifact");
        invalid_fallback_size[22..26].copy_from_slice(&255_u32.to_le_bytes());
        assert_eq!(
            Artifact::from_bytes(&invalid_fallback_size),
            Err(FormatError::InvalidRegistry(
                ValidationError::InvalidByteFallbackSize {
                    id: packtok_core::DEFAULT_BYTE_FALLBACK_PACK_ID,
                    actual: 255,
                }
            ))
        );

        let mut invalid_pack_count = Artifact::byte_fallback_only()
            .to_bytes()
            .expect("serialize default artifact");
        invalid_pack_count[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Artifact::from_bytes(&invalid_pack_count),
            Err(FormatError::CollectionTooLarge {
                collection: "packs",
                count: usize::try_from(u32::MAX).expect("u32 fits usize"),
                maximum: MAX_PACK_COUNT,
            })
        );
    }

    #[test]
    fn parser_rejects_invalid_utf8_in_pack_metadata() {
        let mut bytes = Artifact::byte_fallback_only()
            .to_bytes()
            .expect("serialize default artifact");
        let pack_name_offset = HEADER_BYTES + 10;
        bytes[pack_name_offset] = 0xff;
        assert_eq!(
            Artifact::from_bytes(&bytes),
            Err(FormatError::InvalidUtf8 {
                field: "pack name",
                index: 0,
                offset: pack_name_offset,
            })
        );
    }

    #[test]
    fn parser_rejects_invalid_special_token_references() {
        let artifact = sample_artifact(false);
        let bytes = artifact.to_bytes().expect("serialize artifact");
        let packs_size = artifact
            .registry()
            .packs()
            .iter()
            .map(|pack| 10 + pack.name().len())
            .sum::<usize>();
        let special_offset = HEADER_BYTES + packs_size + 4;

        let mut unknown_pack = bytes.clone();
        unknown_pack[special_offset..special_offset + 2].copy_from_slice(&7_u16.to_le_bytes());
        assert!(matches!(
            Artifact::from_bytes(&unknown_pack),
            Err(FormatError::InvalidRegistry(
                ValidationError::SpecialTokenReferencesUnknownPack { .. }
            ))
        ));

        let mut out_of_range = bytes.clone();
        out_of_range[special_offset + 2..special_offset + 6].copy_from_slice(&16_u32.to_le_bytes());
        assert!(matches!(
            Artifact::from_bytes(&out_of_range),
            Err(FormatError::InvalidRegistry(
                ValidationError::SpecialTokenOutOfRange { .. }
            ))
        ));

        let mut fallback_collision = bytes;
        fallback_collision[special_offset..special_offset + 2]
            .copy_from_slice(&65000_u16.to_le_bytes());
        fallback_collision[special_offset + 2..special_offset + 6]
            .copy_from_slice(&0_u32.to_le_bytes());
        assert!(matches!(
            Artifact::from_bytes(&fallback_collision),
            Err(FormatError::InvalidRegistry(
                ValidationError::SpecialTokenInByteFallback { .. }
            ))
        ));
    }

    #[test]
    fn artifact_creation_rejects_empty_metadata_key() {
        let mut metadata = BTreeMap::new();
        metadata.insert(String::new(), "value".to_owned());
        assert!(matches!(
            Artifact::new(PackRegistry::default(), metadata),
            Err(FormatError::EmptyMetadataKey { index: 0 })
        ));
    }

    fn flat_bpe_artifact() -> Artifact {
        let model = FlatBpeModel::new(vec![FlatBpeMerge {
            left: u32::from(b'a'),
            right: u32::from(b'b'),
            result: 256,
        }])
        .expect("valid BPE model");
        let registry = PackRegistry::new(
            vec![
                PackDescriptor::new(FLAT_BPE_PACK_ID, "FLAT_BPE", 257),
                PackDescriptor::new(
                    packtok_core::DEFAULT_BYTE_FALLBACK_PACK_ID,
                    "BYTE_FALLBACK",
                    BYTE_FALLBACK_TOKEN_COUNT,
                ),
            ],
            ByteFallback::new(packtok_core::DEFAULT_BYTE_FALLBACK_PACK_ID),
            vec![],
        )
        .expect("valid flat BPE registry");
        Artifact::with_flat_bpe(registry, BTreeMap::new(), model).expect("valid v2 artifact")
    }

    #[test]
    fn version_two_round_trip_is_deterministic_and_version_one_is_preserved() {
        let legacy = Artifact::byte_fallback_only()
            .to_bytes()
            .expect("legacy bytes");
        assert_eq!(u16::from_le_bytes([legacy[8], legacy[9]]), FORMAT_VERSION);
        assert_eq!(
            Artifact::from_bytes(&legacy).expect("read legacy"),
            Artifact::byte_fallback_only()
        );

        let artifact = flat_bpe_artifact();
        let first = artifact.to_bytes().expect("serialize v2");
        let second = artifact.to_bytes().expect("serialize v2 again");
        assert_eq!(
            u16::from_le_bytes([first[8], first[9]]),
            FLAT_BPE_FORMAT_VERSION
        );
        assert_eq!(first, second);
        assert_eq!(Artifact::from_bytes(&first).expect("read v2"), artifact);
    }

    #[test]
    fn version_two_rejects_invalid_merge_references_and_truncation() {
        let valid = flat_bpe_artifact().to_bytes().expect("serialize v2");
        let merge_offset = valid.len() - 12;

        let mut invalid_parent = valid.clone();
        invalid_parent[merge_offset..merge_offset + 4].copy_from_slice(&256_u32.to_le_bytes());
        assert!(matches!(
            Artifact::from_bytes(&invalid_parent),
            Err(FormatError::InvalidBpeModel(
                BpeModelError::InvalidParent { .. }
            ))
        ));

        let mut invalid_result = valid.clone();
        invalid_result[merge_offset + 8..merge_offset + 12].copy_from_slice(&257_u32.to_le_bytes());
        assert!(matches!(
            Artifact::from_bytes(&invalid_result),
            Err(FormatError::InvalidBpeModel(
                BpeModelError::InvalidResultId { .. }
            ))
        ));

        let mut truncated = valid;
        truncated.pop();
        assert!(matches!(
            Artifact::from_bytes(&truncated),
            Err(FormatError::InsufficientBytesForCollection {
                collection: "BPE merges",
                count: 1,
            })
        ));
    }

    #[test]
    fn bpe_model_rejects_duplicate_pairs_and_expansion_length_overflow() {
        let duplicate = FlatBpeModel::new(vec![
            FlatBpeMerge {
                left: u32::from(b'a'),
                right: u32::from(b'b'),
                result: 256,
            },
            FlatBpeMerge {
                left: u32::from(b'a'),
                right: u32::from(b'b'),
                result: 257,
            },
        ]);
        assert!(matches!(
            duplicate,
            Err(BpeModelError::DuplicatePair {
                left: 97,
                right: 98
            })
        ));

        let mut exponential = Vec::new();
        for rank in 0..usize::BITS {
            let result = 256 + rank;
            exponential.push(FlatBpeMerge {
                left: result - 1,
                right: result - 1,
                result,
            });
        }
        assert!(matches!(
            FlatBpeModel::new(exponential),
            Err(BpeModelError::ExpandedTokenTooLarge { .. })
        ));
    }

    #[test]
    fn token_expansion_limit_accepts_boundary_and_rejects_compact_bombs() {
        let mut merges = Vec::new();
        let mut parent = u32::from(b'a');
        for rank in 0..20 {
            let result = BYTE_TOKEN_COUNT + rank;
            merges.push(FlatBpeMerge {
                left: parent,
                right: parent,
                result,
            });
            parent = result;
        }
        let boundary = FlatBpeModel::new(merges.clone()).expect("one MiB token allowed");
        assert_eq!(boundary.byte_length(parent), Some(MAX_BPE_TOKEN_BYTES));
        let bomb = FlatBpeMerge {
            left: parent,
            right: parent,
            result: parent + 1,
        };
        merges.push(bomb);
        assert_eq!(
            FlatBpeModel::new(merges),
            Err(BpeModelError::ExpandedTokenTooLarge {
                token_id: parent + 1
            })
        );

        // Forge an untrusted wire record rather than relying on constructors.
        let mut wire = flat_bpe_artifact().to_bytes().expect("small artifact");
        wire.truncate(wire.len() - 16); // previous merge count and single merge
        wire.extend_from_slice(&21_u32.to_le_bytes());
        for merge in boundary.merges().iter().chain(std::iter::once(&bomb)) {
            wire.extend_from_slice(&merge.left.to_le_bytes());
            wire.extend_from_slice(&merge.right.to_le_bytes());
            wire.extend_from_slice(&merge.result.to_le_bytes());
        }
        assert!(wire.len() < 512);
        assert_eq!(
            Artifact::from_bytes(&wire),
            Err(FormatError::InvalidBpeModel(
                BpeModelError::ExpandedTokenTooLarge {
                    token_id: parent + 1
                }
            ))
        );
    }

    #[test]
    fn constructors_reject_metadata_beyond_reader_limit_in_both_versions() {
        let metadata: BTreeMap<_, _> = (0..=MAX_COLLECTION_ITEMS)
            .map(|index| (format!("key-{index:06}"), String::new()))
            .collect();
        let expected = FormatError::CollectionTooLarge {
            collection: "metadata entries",
            count: MAX_COLLECTION_ITEMS + 1,
            maximum: MAX_COLLECTION_ITEMS,
        };
        assert_eq!(
            Artifact::new(PackRegistry::default(), metadata.clone()).map(|_| ()),
            Err(expected.clone())
        );
        let bpe = flat_bpe_artifact();
        assert_eq!(
            Artifact::with_flat_bpe(bpe.registry.clone(), metadata, bpe.flat_bpe.unwrap())
                .map(|_| ()),
            Err(expected)
        );
    }

    #[test]
    fn constructor_rejects_special_tokens_beyond_reader_limit() {
        let count = MAX_COLLECTION_ITEMS + 1;
        let registry = PackRegistry::new(
            vec![
                PackDescriptor::new(0, "specials", count as u32),
                PackDescriptor::new(1, "bytes", BYTE_TOKEN_COUNT),
            ],
            ByteFallback::new(1),
            (0..count)
                .map(|index| {
                    SpecialToken::new(TokenId::new(0, index as u32), format!("special-{index}"))
                })
                .collect(),
        )
        .expect("valid core address space");
        assert_eq!(
            Artifact::new(registry, BTreeMap::new()).map(|_| ()),
            Err(FormatError::CollectionTooLarge {
                collection: "special tokens",
                count,
                maximum: MAX_COLLECTION_ITEMS,
            })
        );
    }

    #[test]
    fn version_one_constructor_rejects_oversized_artifact() {
        let metadata = BTreeMap::from([("large".to_owned(), "x".repeat(MAX_ARTIFACT_BYTES))]);
        assert!(matches!(
            Artifact::new(PackRegistry::default(), metadata),
            Err(FormatError::ArtifactTooLarge { .. })
        ));
    }

    #[test]
    fn metadata_at_reader_limit_round_trips_in_both_versions() {
        let metadata: BTreeMap<_, _> = (0..MAX_COLLECTION_ITEMS)
            .map(|index| (format!("key-{index:06}"), String::new()))
            .collect();
        let legacy =
            Artifact::new(PackRegistry::default(), metadata.clone()).expect("limit allowed");
        let bpe = flat_bpe_artifact();
        let current = Artifact::with_flat_bpe(bpe.registry, metadata, bpe.flat_bpe.unwrap())
            .expect("limit allowed");
        for artifact in [legacy, current] {
            assert_eq!(
                Artifact::from_bytes(&artifact.to_bytes().expect("serialize")),
                Ok(artifact)
            );
        }
    }
}
