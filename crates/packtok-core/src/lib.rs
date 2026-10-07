#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

/// Globally unique identifier for a pack within one PackTok artifact.
pub type PackId = u16;

/// Identifier for a token local to one pack.
pub type LocalTokenId = u32;

/// Number of byte values represented by the raw-byte fallback pack.
pub const BYTE_FALLBACK_TOKEN_COUNT: u32 = 256;

/// Default ID reserved for the raw-byte fallback pack.
pub const DEFAULT_BYTE_FALLBACK_PACK_ID: PackId = PackId::MAX;

/// Default human-readable name for the raw-byte fallback pack.
pub const BYTE_FALLBACK_PACK_NAME: &str = "BYTE_FALLBACK";

/// Maximum number of distinct packs addressable by [`PackId`].
pub const MAX_PACK_COUNT: usize = PackId::MAX as usize + 1;

/// A token address: local IDs are meaningful only within their pack.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenId {
    /// The pack containing this token.
    pub pack: PackId,
    /// The token's local ID within `pack`.
    pub local: LocalTokenId,
}

impl TokenId {
    /// Creates a token address without checking it against a registry.
    #[must_use]
    pub const fn new(pack: PackId, local: LocalTokenId) -> Self {
        Self { pack, local }
    }
}

/// Describes the address space and stable metadata for one pack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackDescriptor {
    id: PackId,
    name: String,
    local_token_count: LocalTokenId,
}

impl PackDescriptor {
    /// Creates a descriptor. Registry-level invariants are checked by [`PackRegistry::new`].
    #[must_use]
    pub fn new(id: PackId, name: impl Into<String>, local_token_count: LocalTokenId) -> Self {
        Self {
            id,
            name: name.into(),
            local_token_count,
        }
    }

    /// Returns this pack's stable ID.
    #[must_use]
    pub const fn id(&self) -> PackId {
        self.id
    }

    /// Returns the pack's metadata name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of valid local IDs, which occupy `0..local_token_count`.
    #[must_use]
    pub const fn local_token_count(&self) -> LocalTokenId {
        self.local_token_count
    }
}

/// Identifies a token that has special meaning to a tokenizer or model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecialToken {
    id: TokenId,
    name: String,
}

impl SpecialToken {
    /// Creates a special-token declaration. Registry-level collisions are checked later.
    #[must_use]
    pub fn new(id: TokenId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
        }
    }

    /// Returns this special token's address.
    #[must_use]
    pub const fn id(&self) -> TokenId {
        self.id
    }

    /// Returns this special token's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Declares which pack is reserved for raw byte values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteFallback {
    pack_id: PackId,
}

impl ByteFallback {
    /// Creates a byte-fallback declaration. Its pack must contain exactly 256 IDs.
    #[must_use]
    pub const fn new(pack_id: PackId) -> Self {
        Self { pack_id }
    }

    /// Returns the ID reserved for the byte-fallback pack.
    #[must_use]
    pub const fn pack_id(self) -> PackId {
        self.pack_id
    }
}

/// Validated pack declarations with canonical ID ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackRegistry {
    packs: Vec<PackDescriptor>,
    byte_fallback: ByteFallback,
    special_tokens: Vec<SpecialToken>,
}

impl PackRegistry {
    /// Validates and canonically orders pack and special-token declarations.
    pub fn new(
        mut packs: Vec<PackDescriptor>,
        byte_fallback: ByteFallback,
        mut special_tokens: Vec<SpecialToken>,
    ) -> Result<Self, ValidationError> {
        if packs.len() > MAX_PACK_COUNT {
            return Err(ValidationError::TooManyPacks { count: packs.len() });
        }

        packs.sort_unstable_by_key(PackDescriptor::id);
        for pair in packs.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(ValidationError::DuplicatePackId { id: pair[0].id });
            }
        }

        let mut pack_names = BTreeSet::new();
        for pack in &packs {
            if pack.name.is_empty() {
                return Err(ValidationError::EmptyPackName { id: pack.id });
            }
            if pack.local_token_count == 0 {
                return Err(ValidationError::EmptyPack { id: pack.id });
            }
            if !pack_names.insert(pack.name.as_str()) {
                return Err(ValidationError::DuplicatePackName {
                    name: pack.name.clone(),
                });
            }
        }

        let fallback_index = packs
            .binary_search_by_key(&byte_fallback.pack_id, PackDescriptor::id)
            .map_err(|_| ValidationError::MissingByteFallbackPack {
                id: byte_fallback.pack_id,
            })?;
        let fallback_pack = &packs[fallback_index];
        if fallback_pack.local_token_count != BYTE_FALLBACK_TOKEN_COUNT {
            return Err(ValidationError::InvalidByteFallbackSize {
                id: fallback_pack.id,
                actual: fallback_pack.local_token_count,
            });
        }

        special_tokens.sort_unstable_by_key(SpecialToken::id);
        for pair in special_tokens.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(ValidationError::DuplicateSpecialTokenId { id: pair[0].id });
            }
        }

        let mut special_names = BTreeSet::new();
        for special in &special_tokens {
            if special.name.is_empty() {
                return Err(ValidationError::EmptySpecialTokenName { id: special.id });
            }
            if !special_names.insert(special.name.as_str()) {
                return Err(ValidationError::DuplicateSpecialTokenName {
                    name: special.name.clone(),
                });
            }
            if special.id.pack == byte_fallback.pack_id {
                return Err(ValidationError::SpecialTokenInByteFallback { id: special.id });
            }

            let pack_index = packs
                .binary_search_by_key(&special.id.pack, PackDescriptor::id)
                .map_err(|_| ValidationError::SpecialTokenReferencesUnknownPack {
                    id: special.id,
                })?;
            let pack = &packs[pack_index];
            if special.id.local >= pack.local_token_count {
                return Err(ValidationError::SpecialTokenOutOfRange {
                    id: special.id,
                    local_token_count: pack.local_token_count,
                });
            }
        }

        Ok(Self {
            packs,
            byte_fallback,
            special_tokens,
        })
    }

    /// Returns a valid registry containing only the default byte-fallback pack.
    #[must_use]
    pub fn byte_fallback_only() -> Self {
        Self {
            packs: vec![PackDescriptor::new(
                DEFAULT_BYTE_FALLBACK_PACK_ID,
                BYTE_FALLBACK_PACK_NAME,
                BYTE_FALLBACK_TOKEN_COUNT,
            )],
            byte_fallback: ByteFallback::new(DEFAULT_BYTE_FALLBACK_PACK_ID),
            special_tokens: Vec::new(),
        }
    }

    /// Returns descriptors in ascending pack-ID order.
    #[must_use]
    pub fn packs(&self) -> &[PackDescriptor] {
        &self.packs
    }

    /// Returns the byte-fallback declaration.
    #[must_use]
    pub const fn byte_fallback(&self) -> ByteFallback {
        self.byte_fallback
    }

    /// Returns special-token declarations in ascending token-ID order.
    #[must_use]
    pub fn special_tokens(&self) -> &[SpecialToken] {
        &self.special_tokens
    }

    /// Looks up one pack by ID.
    #[must_use]
    pub fn pack(&self, id: PackId) -> Option<&PackDescriptor> {
        self.packs
            .binary_search_by_key(&id, PackDescriptor::id)
            .ok()
            .map(|index| &self.packs[index])
    }
}

impl Default for PackRegistry {
    fn default() -> Self {
        Self::byte_fallback_only()
    }
}

/// A precise failure found while validating core pack contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    /// More descriptors were supplied than pack IDs can address.
    TooManyPacks { count: usize },
    /// Two pack descriptors use the same ID.
    DuplicatePackId { id: PackId },
    /// A pack name is empty.
    EmptyPackName { id: PackId },
    /// A pack declares no local token IDs.
    EmptyPack { id: PackId },
    /// Two packs use the same metadata name.
    DuplicatePackName { name: String },
    /// The declared byte-fallback pack is missing.
    MissingByteFallbackPack { id: PackId },
    /// The byte-fallback pack does not contain exactly 256 token IDs.
    InvalidByteFallbackSize { id: PackId, actual: LocalTokenId },
    /// Two special tokens use the same token address.
    DuplicateSpecialTokenId { id: TokenId },
    /// A special-token name is empty.
    EmptySpecialTokenName { id: TokenId },
    /// Two special tokens use the same name.
    DuplicateSpecialTokenName { name: String },
    /// A special token references an undeclared pack.
    SpecialTokenReferencesUnknownPack { id: TokenId },
    /// A special token's local ID is outside its pack's declared range.
    SpecialTokenOutOfRange {
        id: TokenId,
        local_token_count: LocalTokenId,
    },
    /// A special token attempted to reserve a byte-fallback ID.
    SpecialTokenInByteFallback { id: TokenId },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyPacks { count } => {
                write!(
                    formatter,
                    "{count} packs exceed the {MAX_PACK_COUNT} available IDs"
                )
            }
            Self::DuplicatePackId { id } => write!(formatter, "duplicate pack ID {id}"),
            Self::EmptyPackName { id } => write!(formatter, "pack {id} has an empty name"),
            Self::EmptyPack { id } => write!(formatter, "pack {id} has no local token IDs"),
            Self::DuplicatePackName { name } => {
                write!(formatter, "duplicate pack name {name:?}")
            }
            Self::MissingByteFallbackPack { id } => {
                write!(formatter, "byte-fallback pack {id} is not declared")
            }
            Self::InvalidByteFallbackSize { id, actual } => write!(
                formatter,
                "byte-fallback pack {id} declares {actual} IDs; exactly {BYTE_FALLBACK_TOKEN_COUNT} are required"
            ),
            Self::DuplicateSpecialTokenId { id } => write!(
                formatter,
                "multiple special tokens use pack {} local ID {}",
                id.pack, id.local
            ),
            Self::EmptySpecialTokenName { id } => write!(
                formatter,
                "special token at pack {} local ID {} has an empty name",
                id.pack, id.local
            ),
            Self::DuplicateSpecialTokenName { name } => {
                write!(formatter, "duplicate special-token name {name:?}")
            }
            Self::SpecialTokenReferencesUnknownPack { id } => write!(
                formatter,
                "special token references undeclared pack {}",
                id.pack
            ),
            Self::SpecialTokenOutOfRange {
                id,
                local_token_count,
            } => write!(
                formatter,
                "special token local ID {} is outside pack {} range 0..{}",
                id.local, id.pack, local_token_count
            ),
            Self::SpecialTokenInByteFallback { id } => write!(
                formatter,
                "special token at pack {} local ID {} collides with reserved byte fallback",
                id.pack, id.local
            ),
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn fallback_pack(id: PackId) -> PackDescriptor {
        PackDescriptor::new(id, BYTE_FALLBACK_PACK_NAME, BYTE_FALLBACK_TOKEN_COUNT)
    }

    #[test]
    fn token_id_order_is_pack_then_local_id() {
        let ids = [TokenId::new(2, 9), TokenId::new(1, 20), TokenId::new(1, 3)];
        let mut sorted = ids;
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            [TokenId::new(1, 3), TokenId::new(1, 20), TokenId::new(2, 9)]
        );
    }

    #[test]
    fn registry_sorts_arbitrary_packs_deterministically() {
        let registry = PackRegistry::new(
            vec![
                fallback_pack(65000),
                PackDescriptor::new(8, "domain-x", 12),
                PackDescriptor::new(2, "domain-y", 7),
            ],
            ByteFallback::new(65000),
            vec![SpecialToken::new(TokenId::new(8, 11), "control-x")],
        )
        .expect("valid arbitrary pack registry");

        assert_eq!(
            registry
                .packs()
                .iter()
                .map(PackDescriptor::id)
                .collect::<Vec<_>>(),
            [2, 8, 65000]
        );
        assert_eq!(registry.special_tokens()[0].id(), TokenId::new(8, 11));
    }

    #[test]
    fn registry_rejects_duplicate_pack_ids_and_names() {
        let duplicate_id = PackRegistry::new(
            vec![fallback_pack(10), PackDescriptor::new(10, "other", 1)],
            ByteFallback::new(10),
            vec![],
        );
        assert!(matches!(
            duplicate_id,
            Err(ValidationError::DuplicatePackId { id: 10 })
        ));

        let duplicate_name = PackRegistry::new(
            vec![
                fallback_pack(10),
                PackDescriptor::new(11, BYTE_FALLBACK_PACK_NAME, 1),
            ],
            ByteFallback::new(10),
            vec![],
        );
        assert!(matches!(
            duplicate_name,
            Err(ValidationError::DuplicatePackName { .. })
        ));
    }

    #[test]
    fn registry_requires_exact_byte_fallback_address_space() {
        let missing = PackRegistry::new(vec![], ByteFallback::new(9), vec![]);
        assert!(matches!(
            missing,
            Err(ValidationError::MissingByteFallbackPack { id: 9 })
        ));

        let wrong_size = PackRegistry::new(
            vec![PackDescriptor::new(9, "raw", 255)],
            ByteFallback::new(9),
            vec![],
        );
        assert!(matches!(
            wrong_size,
            Err(ValidationError::InvalidByteFallbackSize { id: 9, actual: 255 })
        ));
    }

    #[test]
    fn registry_rejects_special_token_collisions_and_invalid_references() {
        let fallback = fallback_pack(10);
        let other = PackDescriptor::new(3, "arbitrary", 4);

        let duplicate_id = PackRegistry::new(
            vec![fallback.clone(), other.clone()],
            ByteFallback::new(10),
            vec![
                SpecialToken::new(TokenId::new(3, 1), "first"),
                SpecialToken::new(TokenId::new(3, 1), "second"),
            ],
        );
        assert!(matches!(
            duplicate_id,
            Err(ValidationError::DuplicateSpecialTokenId { .. })
        ));

        let duplicate_name = PackRegistry::new(
            vec![fallback.clone(), other.clone()],
            ByteFallback::new(10),
            vec![
                SpecialToken::new(TokenId::new(3, 1), "same"),
                SpecialToken::new(TokenId::new(3, 2), "same"),
            ],
        );
        assert!(matches!(
            duplicate_name,
            Err(ValidationError::DuplicateSpecialTokenName { .. })
        ));

        let unknown_pack = PackRegistry::new(
            vec![fallback.clone(), other.clone()],
            ByteFallback::new(10),
            vec![SpecialToken::new(TokenId::new(7, 1), "unknown")],
        );
        assert!(matches!(
            unknown_pack,
            Err(ValidationError::SpecialTokenReferencesUnknownPack { .. })
        ));

        let out_of_range = PackRegistry::new(
            vec![fallback.clone(), other],
            ByteFallback::new(10),
            vec![SpecialToken::new(TokenId::new(3, 4), "outside")],
        );
        assert!(matches!(
            out_of_range,
            Err(ValidationError::SpecialTokenOutOfRange { .. })
        ));

        let byte_collision = PackRegistry::new(
            vec![fallback],
            ByteFallback::new(10),
            vec![SpecialToken::new(TokenId::new(10, 0), "byte-zero")],
        );
        assert!(matches!(
            byte_collision,
            Err(ValidationError::SpecialTokenInByteFallback { .. })
        ));
    }

    #[test]
    fn default_registry_has_exactly_one_byte_fallback_pack() {
        let registry = PackRegistry::default();
        assert_eq!(registry.packs().len(), 1);
        assert_eq!(
            registry.byte_fallback().pack_id(),
            DEFAULT_BYTE_FALLBACK_PACK_ID
        );
        assert_eq!(
            registry
                .pack(DEFAULT_BYTE_FALLBACK_PACK_ID)
                .expect("fallback pack exists")
                .local_token_count(),
            BYTE_FALLBACK_TOKEN_COUNT
        );
    }
}
