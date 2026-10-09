//! Artifact-derived bijections used by M4. No corpus observations enter them.
use crate::{ModelError, PackVocabulary};
use packtok_core::TokenId;
use std::collections::BTreeMap;

/// A complete bijection between global model IDs and pack/local IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdMapping {
    ids: Vec<TokenId>,
    inverse: BTreeMap<TokenId, u32>,
    packs: Vec<PackVocabulary>,
}

impl IdMapping {
    /// Concatenates nonempty pack vocabularies in numeric pack-ID/local-ID order.
    pub fn flatten(mut packs: Vec<PackVocabulary>) -> Result<Self, ModelError> {
        packs.sort_by_key(|p| p.pack_id);
        if packs.is_empty()
            || packs.len() > 1024
            || packs.windows(2).any(|p| p[0].pack_id == p[1].pack_id)
            || packs.iter().any(|p| p.token_count == 0)
            || packs.iter().map(|p| u64::from(p.token_count)).sum::<u64>() > 1_000_000
        {
            return Err(ModelError::InvalidVocabulary);
        }
        Self::new(
            packs
                .iter()
                .flat_map(|p| (0..p.token_count).map(|local| TokenId::new(p.pack_id, local)))
                .collect(),
        )
    }

    /// Sorts unsigned byte expansions lexicographically (ties: original ID),
    /// assigns rank modulo pack_count, and assigns locals in encounter order.
    pub fn balanced(expansions: &[Vec<u8>], pack_count: usize) -> Result<Self, ModelError> {
        if expansions.is_empty()
            || expansions.len() > 1_000_000
            || pack_count == 0
            || pack_count > 1024
            || pack_count > expansions.len()
            || expansions.iter().any(Vec::is_empty)
        {
            return Err(ModelError::InvalidVocabulary);
        }
        let mut sorted: Vec<usize> = (0..expansions.len()).collect();
        sorted.sort_by(|&a, &b| expansions[a].cmp(&expansions[b]).then(a.cmp(&b)));
        let mut counts = vec![0_u32; pack_count];
        let mut ids = vec![TokenId::new(0, 0); expansions.len()];
        for (rank, global) in sorted.into_iter().enumerate() {
            let pack = rank % pack_count;
            ids[global] = TokenId::new(pack as u16, counts[pack]);
            counts[pack] += 1;
        }
        Self::new(ids)
    }

    fn new(ids: Vec<TokenId>) -> Result<Self, ModelError> {
        if ids.is_empty() || ids.len() > 1_000_000 {
            return Err(ModelError::InvalidVocabulary);
        }
        let mut inverse = BTreeMap::new();
        let mut counts = BTreeMap::<u16, u32>::new();
        for (global, &token) in ids.iter().enumerate() {
            if inverse.insert(token, global as u32).is_some() {
                return Err(ModelError::InvalidVocabulary);
            }
            *counts.entry(token.pack).or_default() += 1;
        }
        if counts.len() > 1024 {
            return Err(ModelError::InvalidVocabulary);
        }
        for token in &ids {
            if token.local >= counts[&token.pack] {
                return Err(ModelError::InvalidVocabulary);
            }
        }
        let packs = counts
            .into_iter()
            .map(|(pack_id, token_count)| PackVocabulary {
                pack_id,
                token_count,
            })
            .collect();
        Ok(Self {
            ids,
            inverse,
            packs,
        })
    }

    /// Resolves a global ID, rejecting out-of-range IDs.
    pub fn to_packed(&self, global: u32) -> Result<TokenId, ModelError> {
        self.ids
            .get(global as usize)
            .copied()
            .ok_or(ModelError::InvalidVocabulary)
    }
    /// Resolves a pack/local ID, rejecting unknown packs or local IDs.
    pub fn to_global(&self, token: TokenId) -> Result<u32, ModelError> {
        self.inverse
            .get(&token)
            .copied()
            .ok_or(ModelError::InvalidVocabulary)
    }
    /// Nonempty pack declarations in canonical order.
    pub fn packs(&self) -> &[PackVocabulary] {
        &self.packs
    }
    /// Number of global IDs.
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    /// Whether the mapping has no rows (valid mappings always have rows).
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// Pack-concatenated embedding-row to original global-input-row permutation.
    pub fn embedding_permutation(&self) -> Vec<u32> {
        self.inverse.values().copied().collect()
    }
    /// Canonical model-side mapping bytes, independent of tokenizer wire formats.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = b"PTMAP4\0\x01".to_vec();
        bytes.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        for token in &self.ids {
            bytes.extend_from_slice(&token.pack.to_le_bytes());
            bytes.extend_from_slice(&token.local.to_le_bytes());
        }
        bytes
    }
    /// Validates size, complete contiguous local domains, and bijectivity.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < 12 || &bytes[..8] != b"PTMAP4\0\x01" {
            return Err(ModelError::InvalidVocabulary);
        }
        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        if count == 0 || count > 1_000_000 || bytes.len() != 12 + count * 6 {
            return Err(ModelError::InvalidVocabulary);
        }
        let ids = bytes[12..]
            .chunks_exact(6)
            .map(|b| {
                TokenId::new(
                    u16::from_le_bytes(b[..2].try_into().unwrap()),
                    u32::from_le_bytes(b[2..].try_into().unwrap()),
                )
            })
            .collect();
        Self::new(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_mapping_rejects_duplicate_gap_and_oversized_domains() {
        let map = IdMapping::flatten(vec![PackVocabulary {
            pack_id: 1,
            token_count: 2,
        }])
        .unwrap();
        let mut duplicate = map.to_bytes();
        duplicate[18..24].copy_from_slice(&map.to_bytes()[12..18]);
        assert!(IdMapping::from_bytes(&duplicate).is_err());
        let mut gap = map.to_bytes();
        gap[20..24].copy_from_slice(&2_u32.to_le_bytes());
        assert!(IdMapping::from_bytes(&gap).is_err());
        let mut large = map.to_bytes();
        large[8..12].copy_from_slice(&1_000_001_u32.to_le_bytes());
        assert!(IdMapping::from_bytes(&large).is_err());
        assert!(IdMapping::balanced(&[vec![0]], 2).is_err());
        assert!(IdMapping::balanced(&[vec![0]], 0).is_err());
    }
    #[test]
    fn balanced_artifact_only_deterministic_and_roundtrip() {
        let bytes: Vec<_> = (0..512_u32).map(|n| n.to_le_bytes().to_vec()).collect();
        let map = IdMapping::balanced(&bytes, 3).unwrap();
        assert_eq!(map, IdMapping::balanced(&bytes, 3).unwrap());
        assert_eq!(
            map.packs()
                .iter()
                .map(|p| p.token_count)
                .collect::<Vec<_>>(),
            vec![171, 171, 170]
        );
        for i in 0..512 {
            assert_eq!(map.to_global(map.to_packed(i).unwrap()).unwrap(), i);
        }
        assert_eq!(IdMapping::from_bytes(&map.to_bytes()).unwrap(), map);
        assert!(map.to_packed(512).is_err());
        assert!(map.to_global(TokenId::new(0, 171)).is_err());
        assert!(map.to_global(TokenId::new(999, 0)).is_err());
        assert!(IdMapping::from_bytes(&map.to_bytes()[..17]).is_err());
    }
    #[test]
    fn flatten_is_bijective_and_canonical() {
        let packs = vec![
            PackVocabulary {
                pack_id: 65535,
                token_count: 256,
            },
            PackVocabulary {
                pack_id: 2,
                token_count: 3,
            },
        ];
        let map = IdMapping::flatten(packs.clone()).unwrap();
        let mut reverse = packs;
        reverse.reverse();
        assert_eq!(map, IdMapping::flatten(reverse).unwrap());
        assert_eq!(map.to_packed(0).unwrap(), TokenId::new(2, 0));
        assert_eq!(map.to_packed(3).unwrap(), TokenId::new(65535, 0));
        assert!(map.to_packed(259).is_err());
        for i in 0..259 {
            assert_eq!(map.to_global(map.to_packed(i).unwrap()).unwrap(), i);
        }
    }
}
