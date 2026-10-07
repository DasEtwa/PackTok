//! Pre-audit rank-scan encoder, retained for matched performance comparisons.

use packtok_format::{FLAT_BPE_PACK_ID, FlatBpeModel};
use packtok_tokenizer::{BpeTokenizer, DecodeError, EncodeError, TokenId, Tokenizer};

pub struct ScanTokenizer {
    pub model: FlatBpeModel,
    pub decoder: BpeTokenizer,
}

impl Tokenizer for ScanTokenizer {
    fn encode(&self, input: &str) -> Result<Vec<TokenId>, EncodeError> {
        let mut current = Vec::new();
        current
            .try_reserve_exact(input.len())
            .map_err(|_| EncodeError::AllocationFailed {
                requested_tokens: input.len(),
            })?;
        current.extend(input.as_bytes().iter().map(|byte| u32::from(*byte)));
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

    fn decode_bytes(&self, tokens: &[TokenId]) -> Result<Vec<u8>, DecodeError> {
        self.decoder.decode_bytes(tokens)
    }
}
