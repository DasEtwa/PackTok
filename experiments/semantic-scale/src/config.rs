//! Versioned, deterministic run plans. Planning never allocates a model.
use crate::{Result, model::ModelConfig, optim::AdamConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainPlan {
    pub version: u32,
    pub preset: String,
    pub model: ModelConfig,
    pub seed: u64,
    pub variant: String,
    pub dtype: String,
    pub micro_batch: usize,
    pub accumulation: usize,
    pub steps: usize,
    pub max_seconds: u64,
    pub checkpoint_every: usize,
    pub data_path: String,
    pub dataset_identity: String,
    pub activation_checkpointing: bool,
    pub optimizer: AdamConfig,
}

impl TrainPlan {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1
            || self.model != ModelConfig::preset(&self.preset, self.model.context)?
            || !["baseline", "semantic"].contains(&self.variant.as_str())
            || !["bf16", "f32"].contains(&self.dtype.as_str())
            || !(1..=32).contains(&self.micro_batch)
            || !(1..=1024).contains(&self.accumulation)
            || !(2..=1_000_000).contains(&self.steps)
            || !(1..=86400).contains(&self.max_seconds)
            || !(1..=10000).contains(&self.checkpoint_every)
            || self.dataset_identity.len() != 64
            || !self.dataset_identity.bytes().all(|b| b.is_ascii_hexdigit())
            || self.optimizer.total_steps != self.steps
            || !self.activation_checkpointing
        {
            return Err(
                "invalid versioned scale plan; expected explicit bounded preset/run values".into(),
            );
        }
        self.model.validate()?;
        self.optimizer.validate()?;
        Ok(())
    }

    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256(&serde_json::to_vec(self)?))
    }

    pub fn load(path: &Path) -> Result<Self> {
        if fs::metadata(path)?.len() > 64 * 1024 {
            return Err("plan exceeds 64 KiB".into());
        }
        let plan: Self = serde_json::from_slice(&fs::read(path)?)?;
        plan.validate()?;
        Ok(plan)
    }

    /// Rough admission estimate, not a measured VRAM guarantee. BF16 includes
    /// model2 + master4 + moments8 + accumulated F32grad4 bytes per parameter.
    /// FP32 reuses model as master: weights4 + moments8 + gradients4.
    pub fn estimated_training_bytes(&self) -> Result<u64> {
        let parameters = self.model.parameter_count()? as u128;
        let state = parameters * if self.dtype == "bf16" { 18 } else { 16 };
        let (b, t, d, f, l, h) = (
            self.micro_batch as u128,
            self.model.context as u128,
            self.model.hidden as u128,
            self.model.ffn as u128,
            self.model.layers as u128,
            self.model.heads as u128,
        );
        // Detached layer inputs + generously padded current-block activations,
        // attention/backprop workspace and a 2-GiB CUDA/transient reserve.
        let activations = b * t * d * l * 4 + b * t * (d * 32 + f * 12) * 4 + b * h * t * t * 24;
        Ok(u64::try_from(state + activations + 2 * 1024_u128.pow(3))?)
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn planning_one_billion_never_allocates_tensors_and_bounds_memory() {
        let model = ModelConfig::preset("1b", 128).unwrap();
        assert!((1_000_000_000..1_100_000_000).contains(&model.parameter_count().unwrap()));
        assert!(ModelConfig::preset("1b", usize::MAX).is_err());
    }
}
