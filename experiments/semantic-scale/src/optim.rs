//! FP32-master AdamW and bounded-host-memory, complete resumable checkpoints.
//!
//! Checkpoint format 1 stores one model variable and its three optimizer tensors
//! per Safetensors shard. Every shard has a SHA-256 entry in an immutable index.
//! Loading validates all shards before changing the model, then loads one shard
//! at a time. Thus checkpoint host memory scales with the largest variable,
//! rather than with all model/optimizer states. Device allocation failures during
//! the second pass are fatal to that in-memory run; disk checkpoints remain valid.
//!
//! A completed generation is published by renaming its temporary directory,
//! followed by atomically replacing latest.json on a local POSIX filesystem.
//! Network/Drive mounts may have weaker durability guarantees: copy completed
//! generations plus latest.json, and verify hashes after restoring them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device, Tensor};
use safetensors::tensor::serialize;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    Result,
    model::{ModelConfig, Transformer},
};

const FORMAT: u32 = 1;
const MAX_INDEX_BYTES: u64 = 4 * 1024 * 1024;
const MAX_SHARD_BYTES: u64 = 512 * 1024 * 1024;

/// All values, including schedule and clipping, are part of checkpoint identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdamConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub clip_norm: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            warmup_steps: 10,
            total_steps: 1000,
            clip_norm: 1.0,
        }
    }
}

impl AdamConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.beta1.is_finite()
            || !(0.0..1.0).contains(&self.beta1)
            || !self.beta2.is_finite()
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon < 1e-12
            || self.epsilon > 1.0
            || !self.weight_decay.is_finite()
            || !(0.0..=1.0).contains(&self.weight_decay)
            || self.total_steps == 0
            || self.total_steps > 1_000_000_000
            || self.warmup_steps >= self.total_steps
            || !self.clip_norm.is_finite()
            || self.clip_norm <= 0.0
        {
            return Err("invalid AdamW optimizer/schedule configuration".into());
        }
        Ok(())
    }

    /// Warmup followed by cosine decay; step numbers start at one.
    pub fn learning_rate_at(&self, step: usize) -> f64 {
        if step == 0 {
            return 0.0;
        }
        if self.warmup_steps > 0 && step <= self.warmup_steps {
            return self.learning_rate * step as f64 / self.warmup_steps as f64;
        }
        let fraction = step
            .saturating_sub(self.warmup_steps)
            .min(self.total_steps - self.warmup_steps) as f64
            / (self.total_steps - self.warmup_steps) as f64;
        self.learning_rate * 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos())
    }
}

/// Sampler progress needed for exact restart in this no-dropout experiment.
/// The run identity supplied to save/load additionally binds all training config,
/// dataset/tokenizer/registry identities, semantic variant and gradient policy.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointProgress {
    pub cursor: u64,
    pub tokens_seen: u64,
    pub target_bytes_seen: u64,
    pub seed: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shard {
    name: String,
    file: String,
    sha256: String,
    bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    format: u32,
    model: ModelConfig,
    dtype: String,
    identity: String,
    optimizer: AdamConfig,
    step: usize,
    progress: CheckpointProgress,
    shards: Vec<Shard>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Latest {
    format: u32,
    directory: String,
    index_sha256: String,
}

pub struct Optimizer {
    config: AdamConfig,
    step: usize,
    master: BTreeMap<String, Tensor>,
    first: BTreeMap<String, Tensor>,
    second: BTreeMap<String, Tensor>,
}

impl Optimizer {
    pub fn new(model: &Transformer, config: AdamConfig) -> Result<Self> {
        config.validate()?;
        let mut master = BTreeMap::new();
        let mut first = BTreeMap::new();
        let mut second = BTreeMap::new();
        for (name, var) in model.named_vars() {
            if !matches!(var.dtype(), DType::F32 | DType::BF16) {
                return Err("optimizer supports F32 or BF16 model weights".into());
            }
            // F32 to_dtype is a cheap clone. BF16 gets an independent F32 master.
            master.insert(name.clone(), var.as_tensor().to_dtype(DType::F32)?.detach());
            first.insert(
                name.clone(),
                Tensor::zeros(var.shape(), DType::F32, var.device())?,
            );
            second.insert(name, Tensor::zeros(var.shape(), DType::F32, var.device())?);
        }
        Ok(Self {
            config,
            step: 0,
            master,
            first,
            second,
        })
    }

    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.step
    }
    #[must_use]
    pub fn config(&self) -> &AdamConfig {
        &self.config
    }

    /// Applies one mean-gradient update, returning the actual learning rate.
    /// Rejects missing/extra/nonfinite gradients before changing any weights.
    /// Global L2 clipping is applied before moment updates, in FP32.
    pub fn step(&mut self, model: &Transformer, grads: &BTreeMap<String, Tensor>) -> Result<f64> {
        let next_step = self.step.checked_add(1).ok_or("optimizer step overflow")?;
        if next_step > self.config.total_steps {
            return Err("optimizer exceeds its declared schedule".into());
        }
        let variables = model.named_vars();
        if variables.keys().ne(grads.keys()) || variables.keys().ne(self.master.keys()) {
            return Err("optimizer gradient/model tensor set mismatch".into());
        }
        let mut norm_squared = 0.0_f64;
        for (name, var) in &variables {
            let grad = &grads[name];
            if grad.shape() != var.shape()
                || grad.dtype() != DType::F32
                || !grad.device().same_device(var.device())
            {
                return Err("optimizer gradient shape/dtype mismatch".into());
            }
            let square = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            if !square.is_finite() || square < 0.0 {
                return Err("nonfinite optimizer gradient/norm".into());
            }
            norm_squared += f64::from(square);
        }
        if !norm_squared.is_finite() {
            return Err("optimizer global norm overflow".into());
        }
        let clip =
            (self.config.clip_norm / norm_squared.sqrt().max(self.config.clip_norm)).min(1.0);
        let lr = self.config.learning_rate_at(next_step);
        let bc1 = 1.0 / (1.0 - self.config.beta1.powf(next_step as f64));
        let bc2 = 1.0 / (1.0 - self.config.beta2.powf(next_step as f64));
        // Tensor-by-tensor updates bound temporary device memory. Any resource or
        // arithmetic failure here aborts the run; resume the last disk generation.
        for (name, var) in variables {
            let grad = (&grads[&name] * clip)?.detach();
            let m = &self.first[&name];
            let v = &self.second[&name];
            let next_m =
                ((m * self.config.beta1)? + (&grad * (1.0 - self.config.beta1))?)?.detach();
            let next_v =
                ((v * self.config.beta2)? + (grad.sqr()? * (1.0 - self.config.beta2))?)?.detach();
            let numerator = (&next_m * bc1)?;
            let denominator = ((&next_v * bc2)?.sqrt()? + self.config.epsilon)?;
            let update = ((numerator / denominator)? * lr)?;
            let weight =
                ((&self.master[&name] * (1.0 - lr * self.config.weight_decay))? - update)?.detach();
            // Sum of squares detects NaN/Inf (including overflow) without an
            // expensive complete host copy. Failure must not publish a checkpoint.
            if !weight.sqr()?.sum_all()?.to_scalar::<f32>()?.is_finite() {
                return Err("nonfinite optimizer update; restart from checkpoint".into());
            }
            var.set(&weight.to_dtype(var.dtype())?)?;
            let master = if var.dtype() == DType::F32 {
                var.as_tensor().detach()
            } else {
                weight
            };
            self.master.insert(name.clone(), master);
            self.first.insert(name.clone(), next_m);
            self.second.insert(name, next_v);
        }
        self.step = next_step;
        Ok(lr)
    }

    /// Writes an immutable generation and advances latest only after every shard
    /// and index have been flushed. Existing generations are never overwritten.
    pub fn save(
        &self,
        model: &Transformer,
        root: &Path,
        identity: &str,
        progress: &CheckpointProgress,
    ) -> Result<PathBuf> {
        if identity.is_empty() || identity.len() > 4096 {
            return Err("invalid run identity".into());
        }
        self.config.validate()?;
        fs::create_dir_all(root)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let generation = format!("step-{:012}-{}-{nonce}", self.step, std::process::id());
        let temporary = root.join(format!("{generation}.tmp"));
        let completed = root.join(&generation);
        fs::create_dir(&temporary)?;
        let variables = model.named_vars();
        if variables.keys().ne(self.master.keys()) {
            return Err("checkpoint model set changed".into());
        }
        let mut shards = Vec::with_capacity(variables.len());
        let dtype = variables
            .values()
            .next()
            .ok_or("empty checkpoint model")?
            .dtype();
        for (ordinal, (name, var)) in variables.iter().enumerate() {
            if var.dtype() != dtype {
                return Err("mixed model dtypes unsupported".into());
            }
            let file = format!("tensor-{ordinal:06}.safetensors");
            let payload = (var.elem_count() as u64)
                .checked_mul(dtype.size_in_bytes() as u64 + 12)
                .ok_or("checkpoint shard size overflow")?;
            if payload > MAX_SHARD_BYTES - 4096 {
                return Err("checkpoint variable exceeds shard size bound".into());
            }
            let tensors = [
                ("model".to_owned(), var.as_tensor().clone()),
                ("master".to_owned(), self.master[name].clone()),
                ("m".to_owned(), self.first[name].clone()),
                ("v".to_owned(), self.second[name].clone()),
            ];
            // serialize copies only this variable's tensors to the host.
            let bytes = serialize(tensors, &None)?;
            if bytes.len() as u64 > MAX_SHARD_BYTES {
                return Err("checkpoint variable shard exceeds 512 MiB bound".into());
            }
            validate_shard(&bytes, var.shape(), dtype)?;
            write_new_sync(&temporary.join(&file), &bytes)?;
            shards.push(Shard {
                name: name.clone(),
                file,
                sha256: hash_bytes(&bytes),
                bytes: bytes.len() as u64,
            });
        }
        let index = Index {
            format: FORMAT,
            model: model.config.clone(),
            dtype: dtype_name(dtype)?.to_owned(),
            identity: identity.to_owned(),
            optimizer: self.config.clone(),
            step: self.step,
            progress: progress.clone(),
            shards,
        };
        let index_bytes = serde_json::to_vec_pretty(&index)?;
        if index_bytes.len() as u64 > MAX_INDEX_BYTES {
            return Err("checkpoint index too large".into());
        }
        write_new_sync(&temporary.join("index.json"), &index_bytes)?;
        sync_dir(&temporary)?;
        fs::rename(&temporary, &completed)?;
        sync_dir(root)?;
        let latest = Latest {
            format: FORMAT,
            directory: generation,
            index_sha256: hash_bytes(&index_bytes),
        };
        let pointer_tmp = root.join(format!("latest.json.tmp-{}-{nonce}", std::process::id()));
        write_new_sync(&pointer_tmp, &serde_json::to_vec_pretty(&latest)?)?;
        fs::rename(&pointer_tmp, root.join("latest.json"))?;
        sync_dir(root)?;
        Ok(completed)
    }

    /// Validates all metadata, hashes, tensor layouts and values on CPU before
    /// mutating model weights; then streams optimizer/model shards onto device.
    pub fn load(
        model: &Transformer,
        root: &Path,
        identity: &str,
        config: &AdamConfig,
    ) -> Result<(Self, CheckpointProgress)> {
        config.validate()?;
        let latest_bytes = read_bounded(&root.join("latest.json"), MAX_INDEX_BYTES)?;
        let latest: Latest = serde_json::from_slice(&latest_bytes)?;
        if latest.format != FORMAT
            || !safe_generation(&latest.directory)
            || !valid_hash(&latest.index_sha256)
        {
            return Err("invalid latest checkpoint pointer".into());
        }
        let generation = root.join(&latest.directory);
        let index_bytes = read_bounded(&generation.join("index.json"), MAX_INDEX_BYTES)?;
        if hash_bytes(&index_bytes) != latest.index_sha256 {
            return Err("checkpoint index hash mismatch".into());
        }
        let index: Index = serde_json::from_slice(&index_bytes)?;
        let variables = model.named_vars();
        let dtype = variables
            .values()
            .next()
            .ok_or("empty checkpoint model")?
            .dtype();
        if index.format != FORMAT
            || index.model != model.config
            || index.identity != identity
            || &index.optimizer != config
            || index.dtype != dtype_name(dtype)?
            || index.step > config.total_steps
            || index.shards.len() != variables.len()
        {
            return Err("checkpoint run/model/optimizer identity mismatch".into());
        }
        for (ordinal, ((name, var), shard)) in variables.iter().zip(&index.shards).enumerate() {
            if shard.name != *name
                || shard.file != format!("tensor-{ordinal:06}.safetensors")
                || !valid_hash(&shard.sha256)
                || shard.bytes > MAX_SHARD_BYTES
            {
                return Err("checkpoint shard manifest mismatch".into());
            }
            let bytes = read_bounded(&generation.join(&shard.file), MAX_SHARD_BYTES)?;
            if bytes.len() as u64 != shard.bytes || hash_bytes(&bytes) != shard.sha256 {
                return Err("checkpoint shard hash/length mismatch".into());
            }
            validate_shard(&bytes, var.shape(), dtype)?;
        }
        let mut optimizer = Self {
            config: config.clone(),
            step: index.step,
            master: BTreeMap::new(),
            first: BTreeMap::new(),
            second: BTreeMap::new(),
        };
        for ((name, var), shard) in variables.iter().zip(&index.shards) {
            // Re-check hashes to catch disk changes between validation/loading.
            let bytes = read_bounded(&generation.join(&shard.file), MAX_SHARD_BYTES)?;
            if hash_bytes(&bytes) != shard.sha256 {
                return Err("checkpoint changed while loading".into());
            }
            let mut tensors = candle_core::safetensors::load_buffer(&bytes, var.device())?;
            let master = tensors
                .remove("master")
                .ok_or("checkpoint master missing")?
                .detach();
            let weight = if dtype == DType::F32 {
                master.clone()
            } else {
                tensors
                    .remove("model")
                    .ok_or("checkpoint weight missing")?
                    .detach()
            };
            var.set(&weight)?;
            let master = if dtype == DType::F32 {
                var.as_tensor().detach()
            } else {
                master
            };
            optimizer.master.insert(name.clone(), master);
            optimizer.first.insert(
                name.clone(),
                tensors
                    .remove("m")
                    .ok_or("checkpoint moment missing")?
                    .detach(),
            );
            optimizer.second.insert(
                name.clone(),
                tensors
                    .remove("v")
                    .ok_or("checkpoint variance missing")?
                    .detach(),
            );
        }
        Ok((optimizer, index.progress))
    }
}

/// Explicitly retains at most `keep` complete generations for this run identity.
///
/// Keeps the currently published generation plus the newest `keep - 1` eligible
/// older generations, ordered by optimizer step and then generation name. Only
/// flat, complete, non-symlink generation directories in this format qualify.
/// Unknown identities, incomplete/orphan directories and unrelated files/logs
/// are preserved. Payload hashes are verified on load; retention checks declared
/// files and sizes without rereading every large shard. This function requires
/// one checkpoint writer per run directory; the notebook copier may read it.
pub fn prune_checkpoints(root: &Path, identity: &str, keep: usize) -> Result<()> {
    if !(1..=100).contains(&keep) || identity.is_empty() || identity.len() > 4096 {
        return Err("invalid checkpoint retention policy".into());
    }
    if !fs::symlink_metadata(root.join("latest.json"))?
        .file_type()
        .is_file()
    {
        return Err("checkpoint latest pointer must be a regular file".into());
    }
    let latest_bytes = read_bounded(&root.join("latest.json"), MAX_INDEX_BYTES)?;
    let latest: Latest = serde_json::from_slice(&latest_bytes)?;
    if latest.format != FORMAT
        || !retention_generation(&latest.directory)
        || !valid_hash(&latest.index_sha256)
    {
        return Err("invalid latest checkpoint pointer for retention".into());
    }
    let (current, index_bytes) = retention_index(root, &latest.directory)?;
    if current.identity != identity || hash_bytes(&index_bytes) != latest.index_sha256 {
        return Err("retention current checkpoint identity/hash mismatch".into());
    }
    let mut candidates = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name == latest.directory || !retention_generation(&name) || !entry.file_type()?.is_dir()
        {
            continue;
        }
        // An incomplete/unrecognized image is evidence, not a deletion target.
        let Ok((index, _)) = retention_index(root, &name) else {
            continue;
        };
        if index.identity == identity && index.step <= current.step {
            candidates.push((index.step, name));
        }
    }
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    let mut changed = false;
    for (_, directory) in candidates.into_iter().skip(keep - 1) {
        // Detect another writer publishing while retention was scanning. A
        // single-writer invariant avoids a race between this check and deletion.
        if read_bounded(&root.join("latest.json"), MAX_INDEX_BYTES)? != latest_bytes {
            return Err("latest checkpoint changed during retention".into());
        }
        // Recheck directory content immediately before removal; preserve newly
        // incomplete images or unrelated content rather than widening deletion scope.
        let Ok((index, _)) = retention_index(root, &directory) else {
            continue;
        };
        if index.identity != identity || directory == latest.directory {
            continue;
        }
        fs::remove_dir_all(root.join(directory))?;
        changed = true;
    }
    if changed {
        sync_dir(root)?;
    }
    Ok(())
}

fn retention_generation(name: &str) -> bool {
    let parts = name.split('-').collect::<Vec<_>>();
    parts.len() == 4
        && parts[0] == "step"
        && parts[1].len() == 12
        && parts[1..].iter().all(|part| {
            !part.is_empty() && part.len() <= 40 && part.bytes().all(|byte| byte.is_ascii_digit())
        })
}

/// Return only a completed directory with exactly its declared regular files.
/// Does not follow generation/index/shard symlinks or delete unrelated contents.
fn retention_index(root: &Path, generation: &str) -> Result<(Index, Vec<u8>)> {
    if !retention_generation(generation) {
        return Err("unrecognized generation name".into());
    }
    let path = root.join(generation);
    if !fs::symlink_metadata(&path)?.file_type().is_dir()
        || !fs::symlink_metadata(path.join("index.json"))?
            .file_type()
            .is_file()
    {
        return Err("checkpoint generation/index must not be symlinks".into());
    }
    let bytes = read_bounded(&path.join("index.json"), MAX_INDEX_BYTES)?;
    let index: Index = serde_json::from_slice(&bytes)?;
    let name_step: usize = generation
        .split('-')
        .nth(1)
        .ok_or("generation step missing")?
        .parse()?;
    if index.format != FORMAT
        || index.step != name_step
        || index.identity.is_empty()
        || index.identity.len() > 4096
        || !matches!(index.dtype.as_str(), "F32" | "BF16")
        || index.step > index.optimizer.total_steps
        || index.shards.is_empty()
        || index.shards.len() > 4096
        || index
            .shards
            .windows(2)
            .any(|pair| pair[0].name >= pair[1].name)
    {
        return Err("unrecognized checkpoint generation metadata".into());
    }
    index.optimizer.validate()?;
    index.model.parameter_count()?;
    let mut expected = BTreeSet::from(["index.json".to_owned()]);
    for (ordinal, shard) in index.shards.iter().enumerate() {
        if shard.file != format!("tensor-{ordinal:06}.safetensors")
            || !valid_hash(&shard.sha256)
            || shard.bytes == 0
            || shard.bytes > MAX_SHARD_BYTES
        {
            return Err("unrecognized checkpoint retention shard".into());
        }
        let metadata = fs::symlink_metadata(path.join(&shard.file))?;
        if !metadata.file_type().is_file() || metadata.len() != shard.bytes {
            return Err("checkpoint retention shard incomplete".into());
        }
        expected.insert(shard.file.clone());
    }
    let mut count = 0;
    for entry in fs::read_dir(&path)? {
        let entry = entry?;
        let name = entry.file_name();
        if !entry.file_type()?.is_file()
            || !name.to_str().is_some_and(|name| expected.contains(name))
        {
            return Err("checkpoint contains unrelated entries; preserving it".into());
        }
        count += 1;
    }
    if count != expected.len() {
        return Err("checkpoint generation is incomplete".into());
    }
    Ok((index, bytes))
}

fn dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("F32"),
        DType::BF16 => Ok("BF16"),
        _ => Err("unsupported checkpoint dtype".into()),
    }
}

fn safe_generation(name: &str) -> bool {
    name.starts_with("step-")
        && name.len() <= 100
        && name
            .bytes()
            .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase() || b == b'-')
}
fn valid_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}
fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    if size > maximum {
        return Err("checkpoint file exceeds size bound".into());
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(size)?)?;
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err("checkpoint grew beyond size bound".into());
    }
    Ok(bytes)
}
fn write_new_sync(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

/// Strict CPU validation, including the master/rounded-model consistency and
/// nonnegative second moments. No device allocation is needed in this pass.
fn validate_shard(bytes: &[u8], shape: &candle_core::Shape, dtype: DType) -> Result<()> {
    let tensors = candle_core::safetensors::load_buffer(bytes, &Device::Cpu)?;
    if tensors.len() != 4 {
        return Err("checkpoint shard tensor count mismatch".into());
    }
    for name in ["model", "master", "m", "v"] {
        let tensor = tensors.get(name).ok_or("checkpoint shard tensor missing")?;
        let expected_dtype = if name == "model" { dtype } else { DType::F32 };
        if tensor.shape() != shape || tensor.dtype() != expected_dtype {
            return Err("checkpoint tensor shape/dtype mismatch".into());
        }
        let values = tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if values
            .iter()
            .any(|v| !v.is_finite() || (name == "v" && *v < 0.0))
        {
            return Err("checkpoint tensor contains invalid values".into());
        }
    }
    let model = tensors["model"]
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let master = tensors["master"]
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if model != master {
        return Err("checkpoint model/master weights disagree".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(seed: u64) -> Transformer {
        Transformer::new(
            ModelConfig {
                layers: 1,
                hidden: 8,
                heads: 2,
                ffn: 16,
                context: 8,
                vocab: 512,
            },
            seed,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap()
    }
    fn grads(model: &Transformer) -> BTreeMap<String, Tensor> {
        model
            .named_vars()
            .into_iter()
            .map(|(name, var)| {
                (
                    name,
                    Tensor::ones(var.shape(), DType::F32, &Device::Cpu).unwrap(),
                )
            })
            .collect()
    }
    fn directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "packtok-scale-{label}-{}-{nonce}",
            std::process::id()
        ))
    }
    fn values(model: &Transformer) -> Vec<(String, Vec<f32>)> {
        model
            .named_vars()
            .into_iter()
            .map(|(name, var)| (name, var.flatten_all().unwrap().to_vec1::<f32>().unwrap()))
            .collect()
    }
    fn config() -> AdamConfig {
        AdamConfig {
            warmup_steps: 0,
            total_steps: 8,
            ..AdamConfig::default()
        }
    }
    fn progress() -> CheckpointProgress {
        CheckpointProgress {
            cursor: 7,
            tokens_seen: 8,
            target_bytes_seen: 9,
            seed: 11,
        }
    }

    #[test]
    fn first_adamw_step_matches_scalar_reference_with_global_clipping() -> Result<()> {
        let source = model(11);
        let old = values(&source);
        let settings = config();
        let mut optimizer = Optimizer::new(&source, settings.clone())?;
        let gradients = grads(&source);
        let parameters = source
            .named_vars()
            .values()
            .map(|v| v.elem_count())
            .sum::<usize>();
        let clipped = (settings.clip_norm / (parameters as f64).sqrt()) as f32;
        let lr = optimizer.step(&source, &gradients)?;
        for ((name, before), (_, after)) in old.iter().zip(values(&source)) {
            for (&weight, actual) in before.iter().zip(after) {
                let expected = f64::from(weight) * (1.0 - lr * settings.weight_decay)
                    - lr * f64::from(clipped) / (f64::from(clipped) + settings.epsilon);
                assert!((f64::from(actual) - expected).abs() < 2e-7, "{name}");
            }
        }
        Ok(())
    }

    #[test]
    fn interrupted_optimizer_matches_uninterrupted_weights() -> Result<()> {
        let original = model(11);
        let restarted = model(123);
        let mut optimizer = Optimizer::new(&original, config())?;
        optimizer.step(&original, &grads(&original))?;
        let root = directory("resume");
        optimizer.save(&original, &root, "identity-a", &progress())?;
        let (mut loaded, actual_progress) =
            Optimizer::load(&restarted, &root, "identity-a", &config())?;
        assert_eq!(actual_progress, progress());
        assert_eq!(loaded.step_count(), 1);
        assert_eq!(values(&original), values(&restarted));
        for _ in 0..3 {
            optimizer.step(&original, &grads(&original))?;
            loaded.step(&restarted, &grads(&restarted))?;
            assert_eq!(values(&original), values(&restarted));
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn corrupted_shard_and_wrong_identity_do_not_change_model() -> Result<()> {
        let source = model(11);
        let destination = model(12);
        let initial = values(&destination);
        let root = directory("corrupt");
        let optimizer = Optimizer::new(&source, config())?;
        let generation = optimizer.save(&source, &root, "identity-a", &progress())?;
        assert!(Optimizer::load(&destination, &root, "identity-b", &config()).is_err());
        let mut wrong = config();
        wrong.clip_norm = 2.0;
        assert!(Optimizer::load(&destination, &root, "identity-a", &wrong).is_err());
        fs::write(generation.join("tensor-000001.safetensors"), b"broken")?;
        assert!(Optimizer::load(&destination, &root, "identity-a", &config()).is_err());
        assert_eq!(values(&destination), initial);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn incomplete_generations_do_not_replace_latest_and_files_are_sharded() -> Result<()> {
        let source = model(11);
        let destination = model(12);
        let root = directory("atomic");
        let optimizer = Optimizer::new(&source, config())?;
        let first = optimizer.save(&source, &root, "identity", &progress())?;
        fs::create_dir(root.join("step-999999-incomplete.tmp"))?;
        fs::write(root.join("latest.json.tmp-interrupted"), b"incomplete")?;
        let (loaded, p) = Optimizer::load(&destination, &root, "identity", &config())?;
        assert_eq!(loaded.step_count(), 0);
        assert_eq!(p, progress());
        assert_eq!(values(&source), values(&destination));
        let second = optimizer.save(&source, &root, "identity", &progress())?;
        assert_ne!(first, second);
        assert!(first.join("index.json").exists());
        let shards = fs::read_dir(second)?
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|e| e == "safetensors"))
            .count();
        assert_eq!(shards, source.named_vars().len());
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn bf16_checkpoints_preserve_unrounded_master_weights() -> Result<()> {
        let shape = ModelConfig {
            layers: 1,
            hidden: 8,
            heads: 2,
            ffn: 16,
            context: 8,
            vocab: 512,
        };
        let source = Transformer::new(shape.clone(), 11, &Device::Cpu, DType::BF16)?;
        let destination = Transformer::new(shape, 22, &Device::Cpu, DType::BF16)?;
        let mut settings = config();
        settings.learning_rate = 1e-6;
        let mut optimizer = Optimizer::new(&source, settings.clone())?;
        optimizer.step(&source, &grads(&source))?;
        let root = directory("bf16");
        optimizer.save(&source, &root, "bf16-identity", &progress())?;
        let (mut loaded, _) = Optimizer::load(&destination, &root, "bf16-identity", &settings)?;
        for _ in 0..3 {
            optimizer.step(&source, &grads(&source))?;
            loaded.step(&destination, &grads(&destination))?;
            for name in optimizer.master.keys() {
                assert_eq!(
                    optimizer.master[name].flatten_all()?.to_vec1::<f32>()?,
                    loaded.master[name].flatten_all()?.to_vec1::<f32>()?
                );
            }
            for (name, var) in source.named_vars() {
                assert_eq!(
                    var.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    destination.named_vars()[&name]
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn explicit_retention_keeps_latest_and_one_older_without_unrelated_deletion() -> Result<()> {
        let source = model(11);
        let root = directory("retention");
        let mut optimizer = Optimizer::new(&source, config())?;
        let first = optimizer.save(&source, &root, "identity", &progress())?;
        optimizer.step(&source, &grads(&source))?;
        let second = optimizer.save(&source, &root, "identity", &progress())?;
        optimizer.step(&source, &grads(&source))?;
        let third = optimizer.save(&source, &root, "identity", &progress())?;
        // Separate-identity images, malformed generations, orphan temps, loose
        // logs, and generation directories with unrelated content all survive.
        let other_root = directory("retention-other");
        let other = optimizer.save(&source, &other_root, "other-identity", &progress())?;
        let other_name = other.file_name().unwrap().to_owned();
        fs::rename(&other, root.join(&other_name))?;
        fs::create_dir(root.join("step-000000000000-orphan.tmp"))?;
        fs::create_dir(root.join("step-000000000000-1-1"))?;
        fs::write(root.join("run.log"), "keep this evidence")?;
        #[cfg(unix)]
        std::os::unix::fs::symlink(&first, root.join("step-000000000000-2-2"))?;
        prune_checkpoints(&root, "identity", 2)?;
        assert!(!first.exists());
        assert!(second.exists());
        assert!(third.exists());
        assert!(root.join(other_name).exists());
        assert!(root.join("step-000000000000-orphan.tmp").exists());
        assert!(root.join("step-000000000000-1-1").exists());
        assert_eq!(
            fs::read_to_string(root.join("run.log"))?,
            "keep this evidence"
        );
        #[cfg(unix)]
        assert!(
            fs::symlink_metadata(root.join("step-000000000000-2-2"))?
                .file_type()
                .is_symlink()
        );
        assert!(prune_checkpoints(&root, "wrong-identity", 1).is_err());
        assert!(prune_checkpoints(&root, "identity", 0).is_err());
        prune_checkpoints(&root, "identity", 1)?;
        assert!(!second.exists());
        assert!(third.exists());
        let destination = model(12);
        assert_eq!(
            Optimizer::load(&destination, &root, "identity", &config())?
                .0
                .step_count(),
            2
        );
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(other_root)?;
        Ok(())
    }

    #[test]
    fn invalid_gradients_and_schedule_are_rejected_before_updates() -> Result<()> {
        let source = model(11);
        let before = values(&source);
        let mut optimizer = Optimizer::new(&source, config())?;
        let mut gradients = grads(&source);
        let first = gradients.keys().next().unwrap().clone();
        let invalid = (&gradients[&first] * f64::NAN)?;
        gradients.insert(first, invalid);
        assert!(optimizer.step(&source, &gradients).is_err());
        assert_eq!(values(&source), before);
        assert_eq!(optimizer.step_count(), 0);
        let mut cases = vec![config(); 5];
        cases[0].learning_rate = f64::NAN;
        cases[1].epsilon = 0.0;
        cases[2].beta1 = 1.0;
        cases[3].warmup_steps = cases[3].total_steps;
        cases[4].clip_norm = f64::INFINITY;
        for bad in cases {
            assert!(Optimizer::new(&source, bad).is_err());
        }
        Ok(())
    }
}
