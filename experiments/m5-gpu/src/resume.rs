use crate::{
    Result, hash,
    model::{ModelConfig, Transformer},
};
use candle_core::{DType, Device, Tensor, Var, backprop::GradStore};
use safetensors::tensor::serialize;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::Write,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub config_sha256: String,
    pub corpus_sha256: String,
    pub tokenizer_sha256: String,
    pub train_sequence_sha256: String,
    pub variant: String,
    pub regime: String,
    pub seed: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdamWConfig {
    pub learning_rate: f64,
    pub decay: String,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub total_steps: usize,
    pub warmup_steps: usize,
}
impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.0003,
            decay: "cosine".into(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            total_steps: 20_000,
            warmup_steps: 100,
        }
    }
}
impl AdamWConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !(0.0..1.0).contains(&self.beta1)
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
            || !["constant", "cosine"].contains(&self.decay.as_str())
            || self.total_steps == 0
            || self.warmup_steps >= self.total_steps
        {
            return Err("invalid resumable AdamW/schedule configuration".into());
        }
        Ok(())
    }
    pub fn learning_rate_at(&self, step: usize) -> f64 {
        if step == 0 {
            return 0.0;
        }
        if self.warmup_steps > 0 && step <= self.warmup_steps {
            return self.learning_rate * step as f64 / self.warmup_steps as f64;
        }
        if self.decay == "constant" {
            return self.learning_rate;
        }
        let decay_steps = self.total_steps - self.warmup_steps;
        let progress = (step - self.warmup_steps).min(decay_steps) as f64 / decay_steps as f64;
        self.learning_rate * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct CheckpointMeta {
    format: u32,
    model: ModelConfig,
    identity: Identity,
    optimizer_step: usize,
    training_step: usize,
    sample_cursor: u64,
    rng_state: u64,
    schedule: AdamWConfig,
    state_sha256: String,
}
pub struct ResumableAdamW {
    config: AdamWConfig,
    step: usize,
    first: BTreeMap<String, Tensor>,
    second: BTreeMap<String, Tensor>,
}
impl ResumableAdamW {
    pub fn new(model: &Transformer, config: AdamWConfig) -> Result<Self> {
        config.validate()?;
        let mut first = BTreeMap::new();
        let mut second = BTreeMap::new();
        for (name, var) in model.named_vars() {
            first.insert(
                name.clone(),
                Tensor::zeros(var.shape(), var.dtype(), var.device())?,
            );
            second.insert(name, Tensor::zeros(var.shape(), var.dtype(), var.device())?);
        }
        Ok(Self {
            config,
            step: 0,
            first,
            second,
        })
    }
    pub fn step_count(&self) -> usize {
        self.step
    }
    pub fn config(&self) -> &AdamWConfig {
        &self.config
    }
    pub fn apply(&mut self, model: &Transformer, grads: &GradStore) -> Result<()> {
        let next_step = self.step.checked_add(1).ok_or("optimizer step overflow")?;
        if next_step > self.config.total_steps {
            return Err("optimizer exceeds frozen schedule length".into());
        }
        let lr = self.config.learning_rate_at(next_step);
        let bc1 = 1.0 / (1.0 - self.config.beta1.powi(i32::try_from(next_step)?));
        let bc2 = 1.0 / (1.0 - self.config.beta2.powi(i32::try_from(next_step)?));
        let mut staged = Vec::new();
        let mut checks = Vec::new();
        for (name, var) in model.named_vars() {
            let grad = grads.get(&var).ok_or("resumable AdamW missing gradient")?;
            if grad.shape() != var.shape() || grad.dtype() != var.dtype() {
                return Err("resumable AdamW gradient layout mismatch".into());
            }
            let m = self.first.get(&name).ok_or("missing first moment")?;
            let v = self.second.get(&name).ok_or("missing second moment")?;
            let next_m = ((m * self.config.beta1)? + (grad * (1.0 - self.config.beta1))?)?;
            let next_v = ((v * self.config.beta2)? + (grad.sqr()? * (1.0 - self.config.beta2))?)?;
            let m_hat = &next_m * bc1;
            let v_hat = &next_v * bc2;
            let decayed = var.as_tensor() * (1.0 - lr * self.config.weight_decay);
            let update = (m_hat / (v_hat?.sqrt()? + self.config.epsilon)?)? * lr;
            let next_weight = (decayed? - update?)?;
            // Validate squared norms in one device-to-host transfer before publishing any state.
            // Zero gradients are valid; nonfinite gradients/results and FP32 norm overflow are not.
            for tensor in [grad, &next_m, &next_v, &next_weight] {
                checks.push(tensor.sqr()?.sum_all()?.reshape((1,))?);
            }
            staged.push((name, var, next_m, next_v, next_weight));
        }
        if Tensor::cat(&checks, 0)?
            .to_vec1::<f32>()?
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err("resumable AdamW nonfinite gradient or candidate state".into());
        }
        for (name, var, next_m, next_v, next_weight) in staged {
            var.set(&next_weight)?;
            self.first.insert(name.clone(), next_m);
            self.second.insert(name, next_v);
        }
        self.step = next_step;
        Ok(())
    }

    pub fn save_checkpoint(
        &self,
        model: &Transformer,
        identity: &Identity,
        rng_state: u64,
        training_step: usize,
        sample_cursor: u64,
        path: &Path,
    ) -> Result<()> {
        if training_step != self.step || rng_state == 0 {
            return Err("checkpoint cursor/optimizer state mismatch".into());
        }
        let model_vars = model.named_vars();
        let state_hash = tensor_hash(&model_vars, &self.first, &self.second)?;
        let meta = CheckpointMeta {
            format: 1,
            model: model.config.clone(),
            identity: identity.clone(),
            optimizer_step: self.step,
            training_step,
            sample_cursor,
            rng_state,
            schedule: self.config.clone(),
            state_sha256: state_hash,
        };
        let mut tensors: Vec<(String, Tensor)> = Vec::with_capacity(model_vars.len() * 3);
        for (name, var) in model_vars {
            tensors.push((format!("model.{name}"), var.as_tensor().clone()));
            tensors.push((format!("adamw.m.{name}"), self.first[&name].clone()));
            tensors.push((format!("adamw.v.{name}"), self.second[&name].clone()));
        }
        let mut metadata = HashMap::new();
        metadata.insert(
            "packtok_m5_checkpoint".to_string(),
            serde_json::to_string(&meta)?,
        );
        let bytes = serialize(tensors, &Some(metadata))?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let temp = path.with_extension(format!("safetensors.tmp-{}-{nonce}", std::process::id()));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        // Parse and validate the complete temporary image before atomically replacing the prior checkpoint.
        let verified = fs::read(&temp)?;
        verify_image(&verified, model, identity, &self.config)?;
        fs::rename(&temp, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    }
    pub fn load_checkpoint(
        model: &mut Transformer,
        identity: &Identity,
        expected_schedule: &AdamWConfig,
        path: &Path,
    ) -> Result<(Self, usize, u64, u64)> {
        let bytes = fs::read(path)?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
        let meta_json = header
            .metadata()
            .as_ref()
            .and_then(|m| m.get("packtok_m5_checkpoint"))
            .ok_or("checkpoint metadata missing")?;
        let meta: CheckpointMeta = serde_json::from_str(meta_json)?;
        if meta.format != 1
            || meta.model != model.config
            || &meta.identity != identity
            || &meta.schedule != expected_schedule
            || meta.training_step != meta.optimizer_step
            || meta.rng_state == 0
        {
            return Err("checkpoint identity or counters do not match".into());
        }
        meta.schedule.validate()?;
        if meta.optimizer_step > meta.schedule.total_steps {
            return Err("checkpoint optimizer beyond schedule".into());
        }
        let loaded = candle_core::safetensors::load_buffer(
            &bytes,
            model
                .named_vars()
                .values()
                .next()
                .ok_or("model has no variables")?
                .device(),
        )?;
        let current = model.named_vars();
        if loaded.len() != current.len() * 3 {
            return Err("checkpoint tensor set size".into());
        }
        let mut first = BTreeMap::new();
        let mut second = BTreeMap::new();
        for (name, var) in &current {
            for prefix in ["model.", "adamw.m.", "adamw.v."] {
                let tensor = loaded
                    .get(&format!("{prefix}{name}"))
                    .ok_or("checkpoint tensor missing")?;
                if tensor.shape() != var.shape()
                    || tensor.dtype() != DType::F32
                    || !tensor
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|x| x.is_finite())
                {
                    return Err("checkpoint tensor layout or value invalid".into());
                }
            }
            first.insert(name.clone(), loaded[&format!("adamw.m.{name}")].clone());
            second.insert(name.clone(), loaded[&format!("adamw.v.{name}")].clone());
        }
        let model_tensors: BTreeMap<String, Var> = current
            .iter()
            .map(|(n, v)| (n.clone(), v.clone()))
            .collect();
        let canonical_model: BTreeMap<String, Tensor> = current
            .keys()
            .map(|name| (name.clone(), loaded[&format!("model.{name}")].clone()))
            .collect();
        let checked_hash = tensor_hash_from_tensors(&canonical_model, &first, &second)?;
        if checked_hash != meta.state_sha256 {
            return Err("checkpoint tensor integrity hash mismatch".into());
        }
        // All tensors and identities are validated before mutating the model.
        for (name, var) in model_tensors {
            var.set(&loaded[&format!("model.{name}")])?;
        }
        let opt = Self {
            config: meta.schedule,
            step: meta.optimizer_step,
            first,
            second,
        };
        Ok((opt, meta.training_step, meta.sample_cursor, meta.rng_state))
    }
}
fn tensor_hash(
    model: &BTreeMap<String, Var>,
    first: &BTreeMap<String, Tensor>,
    second: &BTreeMap<String, Tensor>,
) -> Result<String> {
    let vars: BTreeMap<String, Tensor> = model
        .iter()
        .map(|(n, v)| (n.clone(), v.as_tensor().clone()))
        .collect();
    tensor_hash_from_tensors(&vars, first, second)
}
fn tensor_hash_from_tensors(
    model: &BTreeMap<String, Tensor>,
    first: &BTreeMap<String, Tensor>,
    second: &BTreeMap<String, Tensor>,
) -> Result<String> {
    let mut raw = Vec::new();
    for (prefix, values) in [("model.", model), ("adamw.m.", first), ("adamw.v.", second)] {
        for (name, tensor) in values {
            raw.extend_from_slice(prefix.as_bytes());
            raw.extend_from_slice(name.as_bytes());
            raw.push(0);
            for dim in tensor.dims() {
                raw.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            for value in tensor
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
            {
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(hash(&raw))
}
fn verify_image(
    bytes: &[u8],
    model: &Transformer,
    identity: &Identity,
    schedule: &AdamWConfig,
) -> Result<()> {
    let (_, header) = safetensors::SafeTensors::read_metadata(bytes)?;
    let meta = header
        .metadata()
        .as_ref()
        .and_then(|m| m.get("packtok_m5_checkpoint"))
        .ok_or("temporary checkpoint metadata missing")?;
    let meta: CheckpointMeta = serde_json::from_str(meta)?;
    if meta.format != 1
        || meta.model != model.config
        || &meta.identity != identity
        || &meta.schedule != schedule
        || meta.training_step != meta.optimizer_step
        || meta.rng_state == 0
        || meta.optimizer_step > meta.schedule.total_steps
        || meta.schedule.validate().is_err()
    {
        return Err("temporary checkpoint metadata mismatch".into());
    }
    let tensors = candle_core::safetensors::load_buffer(bytes, &Device::Cpu)?;
    let vars = model.named_vars();
    if tensors.len() != vars.len() * 3 {
        return Err("temporary checkpoint tensor count".into());
    }
    let mut model_tensors = BTreeMap::new();
    let mut first = BTreeMap::new();
    let mut second = BTreeMap::new();
    for (name, var) in vars {
        for prefix in ["model.", "adamw.m.", "adamw.v."] {
            let t = tensors
                .get(&format!("{prefix}{name}"))
                .ok_or("temporary tensor missing")?;
            if t.shape() != var.shape()
                || t.dtype() != DType::F32
                || !t
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite())
            {
                return Err("temporary checkpoint tensor invalid".into());
            }
        }
        model_tensors.insert(name.clone(), tensors[&format!("model.{name}")].clone());
        first.insert(name.clone(), tensors[&format!("adamw.m.{name}")].clone());
        second.insert(name.clone(), tensors[&format!("adamw.v.{name}")].clone());
    }
    if tensor_hash_from_tensors(&model_tensors, &first, &second)? != meta.state_sha256 {
        return Err("temporary checkpoint integrity hash mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::{Rng, loss},
        runner::tiny_config,
    };
    use candle_core::{Device, Tensor};

    fn fixture_identity() -> Identity {
        Identity {
            config_sha256: "a".repeat(64),
            corpus_sha256: "b".repeat(64),
            tokenizer_sha256: "c".repeat(64),
            train_sequence_sha256: "d".repeat(64),
            variant: "A".into(),
            regime: "T".into(),
            seed: 19,
        }
    }
    #[test]
    fn review_corrupt_and_truncated_checkpoint_preserve_destination() -> Result<()> {
        let source = Transformer::new(tiny_config(), 83, &Device::Cpu)?;
        let optimizer = ResumableAdamW::new(&source, AdamWConfig::default())?;
        let id = fixture_identity();
        let path = std::env::temp_dir().join(format!(
            "packtok-review-corrupt-{}.safetensors",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        optimizer.save_checkpoint(&source, &id, 55, 0, 0, &path)?;
        let intact = fs::read(&path)?;
        let mut changed = intact.clone();
        *changed.last_mut().unwrap() ^= 1;
        let mut destination = Transformer::new(tiny_config(), 84, &Device::Cpu)?;
        let before = crate::runner::init_hash(&destination)?;
        for image in [intact[..intact.len() - 1].to_vec(), changed, vec![0; 16]] {
            fs::write(&path, image)?;
            assert!(
                ResumableAdamW::load_checkpoint(&mut destination, &id, &optimizer.config, &path)
                    .is_err()
            );
            assert_eq!(crate::runner::init_hash(&destination)?, before);
        }
        fs::remove_file(path)?;
        Ok(())
    }
    #[test]
    fn review_missing_late_gradient_does_not_partially_update() -> Result<()> {
        let model = Transformer::new(tiny_config(), 71, &Device::Cpu)?;
        let mut optimizer = ResumableAdamW::new(&model, AdamWConfig::default())?;
        let before = tensor_hash(&model.named_vars(), &optimizer.first, &optimizer.second)?;
        let vars = model.named_vars();
        let mut terms = vars.values().take(vars.len() - 1).map(|v| v.sum_all());
        let mut objective = terms.next().unwrap()?;
        for term in terms {
            objective = (objective + term?)?;
        }
        assert!(optimizer.apply(&model, &objective.backward()?).is_err());
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(
            tensor_hash(&model.named_vars(), &optimizer.first, &optimizer.second)?,
            before
        );
        Ok(())
    }

    #[test]
    fn review_nonfinite_gradient_is_rejected_without_mutation() -> Result<()> {
        let model = Transformer::new(tiny_config(), 72, &Device::Cpu)?;
        let mut optimizer = ResumableAdamW::new(&model, AdamWConfig::default())?;
        let before = tensor_hash(&model.named_vars(), &optimizer.first, &optimizer.second)?;
        let vars = model.named_vars();
        let mut terms = vars.values().map(|v| v.sum_all());
        let mut objective = terms.next().unwrap()?;
        for term in terms {
            objective = (objective + term?)?;
        }
        let grads = (objective * f64::INFINITY)?.backward()?;
        assert!(optimizer.apply(&model, &grads).is_err());
        assert_eq!(optimizer.step_count(), 0);
        assert_eq!(
            tensor_hash(&model.named_vars(), &optimizer.first, &optimizer.second)?,
            before
        );
        Ok(())
    }

    #[test]
    fn review_checkpoint_schedule_mismatch_preserves_destination() -> Result<()> {
        let source = Transformer::new(tiny_config(), 73, &Device::Cpu)?;
        let optimizer = ResumableAdamW::new(&source, AdamWConfig::default())?;
        let path = std::env::temp_dir().join(format!(
            "packtok-review-schedule-{}.safetensors",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let id = fixture_identity();
        optimizer.save_checkpoint(&source, &id, 33, 0, 0, &path)?;
        let mut destination = Transformer::new(tiny_config(), 74, &Device::Cpu)?;
        let before = crate::runner::init_hash(&destination)?;
        for expected in [
            AdamWConfig {
                learning_rate: 0.001,
                ..optimizer.config.clone()
            },
            AdamWConfig {
                total_steps: 30_000,
                ..optimizer.config.clone()
            },
            AdamWConfig {
                warmup_steps: 200,
                ..optimizer.config.clone()
            },
            AdamWConfig {
                decay: "constant".into(),
                ..optimizer.config.clone()
            },
        ] {
            assert!(
                ResumableAdamW::load_checkpoint(&mut destination, &id, &expected, &path).is_err()
            );
            assert_eq!(crate::runner::init_hash(&destination)?, before);
        }
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn review_finite_gradient_overflow_is_atomic_and_zero_gradient_is_valid() -> Result<()> {
        let model = Transformer::new(tiny_config(), 75, &Device::Cpu)?;
        let mut optimizer = ResumableAdamW::new(&model, AdamWConfig::default())?;
        let before = tensor_hash(&model.named_vars(), &optimizer.first, &optimizer.second)?;
        let vars = model.named_vars();
        let mut terms = vars.values().map(|v| v.sum_all());
        let mut objective = terms.next().unwrap()?;
        for term in terms {
            objective = (objective + term?)?;
        }
        let mut grads = objective.backward()?;
        for var in vars.values() {
            grads.insert(var, (var.ones_like()? * 1e30)?);
        }
        assert!(optimizer.apply(&model, &grads).is_err());
        assert_eq!(
            tensor_hash(&model.named_vars(), &optimizer.first, &optimizer.second)?,
            before
        );
        for var in vars.values() {
            grads.insert(var, var.zeros_like()?);
        }
        optimizer.apply(&model, &grads)?;
        assert_eq!(optimizer.step_count(), 1);
        Ok(())
    }
    fn updates(
        model: &Transformer,
        optimizer: &mut ResumableAdamW,
        rng: &mut Rng,
        count: usize,
        cursor: &mut u64,
    ) -> Result<()> {
        let tokens: Vec<u32> = (0..64).map(|i| (i % 31 + 1) as u32).collect();
        for _ in 0..count {
            let mut input = Vec::new();
            let mut target = Vec::new();
            for _ in 0..4 {
                let start = (rng.next_u64() as usize) % (tokens.len() - 8);
                input.extend_from_slice(&tokens[start..start + 8]);
                target.extend_from_slice(&tokens[start + 1..start + 9]);
            }
            let x = Tensor::from_vec(input, (4, 8), &Device::Cpu)?;
            let y = Tensor::from_vec(target, (4, 8), &Device::Cpu)?;
            let gradients = loss(&model.forward(&x)?, &y)?.backward()?;
            optimizer.apply(model, &gradients)?;
            *cursor += 1;
        }
        Ok(())
    }
    #[test]
    fn checkpoint_resume_matches_uninterrupted_cpu_updates_exactly() -> Result<()> {
        let config = AdamWConfig {
            learning_rate: 0.001,
            total_steps: 4,
            warmup_steps: 1,
            ..AdamWConfig::default()
        };
        assert_eq!(config.learning_rate_at(1), 0.001);
        assert!(config.learning_rate_at(4) < 1e-12);
        let id = fixture_identity();
        let full = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
        let mut full_opt = ResumableAdamW::new(&full, config.clone())?;
        let mut full_rng = Rng::new(882);
        let mut full_cursor = 0;
        updates(&full, &mut full_opt, &mut full_rng, 4, &mut full_cursor)?;

        let first = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
        let mut first_opt = ResumableAdamW::new(&first, config)?;
        let mut first_rng = Rng::new(882);
        let mut cursor = 0;
        updates(&first, &mut first_opt, &mut first_rng, 2, &mut cursor)?;
        let path = std::env::temp_dir().join(format!(
            "packtok-m5-resume-{}.safetensors",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        first_opt.save_checkpoint(
            &first,
            &id,
            first_rng.state(),
            first_opt.step_count(),
            cursor,
            &path,
        )?;

        let mut resumed = Transformer::new(tiny_config(), 999, &Device::Cpu)?;
        let (mut resumed_opt, step, sample_cursor, rng_state) =
            ResumableAdamW::load_checkpoint(&mut resumed, &id, &first_opt.config, &path)?;
        assert_eq!((step, sample_cursor), (2, 2));
        let mut resumed_rng = Rng::from_state(rng_state)?;
        updates(&resumed, &mut resumed_opt, &mut resumed_rng, 2, &mut cursor)?;
        assert_eq!(resumed_opt.step_count(), 4);
        assert_eq!(full_rng.state(), resumed_rng.state());
        assert_eq!(full_cursor, cursor);
        assert_eq!(
            tensor_hash(&full.named_vars(), &full_opt.first, &full_opt.second)?,
            tensor_hash(
                &resumed.named_vars(),
                &resumed_opt.first,
                &resumed_opt.second
            )?
        );
        let _ = fs::remove_file(path);
        Ok(())
    }
    #[test]
    fn checkpoint_rejects_wrong_identity_without_mutating_model() -> Result<()> {
        let model = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
        let optimizer = ResumableAdamW::new(&model, AdamWConfig::default())?;
        let id = fixture_identity();
        let path = std::env::temp_dir().join(format!(
            "packtok-m5-identity-{}.safetensors",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        optimizer.save_checkpoint(&model, &id, 77, 0, 0, &path)?;
        let before = tensor_hash(&model.named_vars(), &optimizer.first, &optimizer.second)?;
        let mut other = Transformer::new(tiny_config(), 999, &Device::Cpu)?;
        let original = crate::runner::init_hash(&other)?;
        let mut bad = id.clone();
        bad.seed += 1;
        assert!(
            ResumableAdamW::load_checkpoint(&mut other, &bad, &optimizer.config, &path).is_err()
        );
        assert_eq!(crate::runner::init_hash(&other)?, original);
        assert_ne!(before, crate::runner::init_hash(&other)?);
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn resumable_adamw_matches_pinned_candle_adamw_update() -> Result<()> {
        use candle_nn::{AdamW, Optimizer, ParamsAdamW};
        let custom_model = Transformer::new(tiny_config(), 61, &Device::Cpu)?;
        let candle_model = Transformer::new(tiny_config(), 61, &Device::Cpu)?;
        let x = Tensor::new(&[[1_u32, 2, 3, 4, 5]], &Device::Cpu)?;
        let y = Tensor::new(&[[2_u32, 3, 4, 5, 6]], &Device::Cpu)?;
        let custom_grads = loss(&custom_model.forward(&x)?, &y)?.backward()?;
        let candle_grads = loss(&candle_model.forward(&x)?, &y)?.backward()?;
        let params = ParamsAdamW {
            lr: 0.0003,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let mut candle = AdamW::new(candle_model.optimizer_vars(), params)?;
        let mut custom = ResumableAdamW::new(
            &custom_model,
            AdamWConfig {
                learning_rate: 0.0003,
                decay: "constant".into(),
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.01,
                total_steps: 1,
                warmup_steps: 0,
            },
        )?;
        candle.step(&candle_grads)?;
        custom.apply(&custom_model, &custom_grads)?;
        assert_eq!(
            crate::runner::init_hash(&custom_model)?,
            crate::runner::init_hash(&candle_model)?
        );
        Ok(())
    }
}
