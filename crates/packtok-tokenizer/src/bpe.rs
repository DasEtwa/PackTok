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
}

impl BpeTokenizer {
    /// Creates a runtime tokenizer from a structurally validated merge model.
    #[must_use]
    pub fn new(model: FlatBpeModel) -> Self {
        Self { model }
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
        let mut current = Vec::new();
        current
            .try_reserve_exact(input.len())
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: input.len(),
            })?;
        current.extend(input.iter().map(|byte| u32::from(*byte)));

        // Merge ranks are applied exactly in artifact order. Each pass replaces
        // all non-overlapping occurrences from left to right, matching training.
        for merge in self.model.merges() {
            if !current
                .windows(2)
                .any(|pair| pair[0] == merge.left && pair[1] == merge.right)
            {
                continue;
            }

            let mut next = Vec::new();
            next.try_reserve_exact(current.len())
                .map_err(|_| EncodeError::AllocationFailed {
                    requested_tokens: current.len(),
                })?;
            let mut index = 0;
            while index < current.len() {
                if index + 1 < current.len()
                    && current[index] == merge.left
                    && current[index + 1] == merge.right
                {
                    next.push(merge.result);
                    index += 2;
                } else {
                    next.push(current[index]);
                    index += 1;
                }
            }
            current = next;
        }

        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(current.len())
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: current.len(),
            })?;
        tokens.extend(
            current
                .into_iter()
                .map(|local| TokenId::new(FLAT_BPE_PACK_ID, local)),
        );
        Ok(tokens)
    }

    /// Expands flat token IDs to exact bytes without requiring valid UTF-8.
    pub fn decode_bpe_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError> {
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
        let mut stack = Vec::<LocalTokenId>::new();

        for (token_index, token) in tokens.iter().enumerate() {
            stack.clear();
            stack
                .try_reserve(1)
                .map_err(|_| DecodeError::AllocationFailed { requested_bytes: 4 })?;
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
                stack
                    .try_reserve(2)
                    .map_err(|_| DecodeError::AllocationFailed { requested_bytes: 8 })?;
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
