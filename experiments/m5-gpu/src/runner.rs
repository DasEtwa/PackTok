use crate::{
    Result,
    data::{Sequence, bits_per_byte},
    hash,
    model::{ModelConfig, Rng, Transformer, finite_gradients, loss},
    write_new,
};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

pub fn tiny_config() -> ModelConfig {
    ModelConfig {
        layers: 1,
        hidden: 8,
        heads: 2,
        ffn: 16,
        context: 8,
        vocab: 512,
    }
}
#[derive(Serialize, Deserialize)]
pub struct Reference {
    config: ModelConfig,
    seed: u64,
    logits: Vec<f32>,
    loss: f32,
    grads: BTreeMap<String, Vec<f32>>,
}
pub fn reference(path: &Path) -> Result<()> {
    let m = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
    let x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
    let y = Tensor::new(&[[2_u32, 3, 4, 1]], &Device::Cpu)?;
    let logits = m.forward(&x)?;
    let l = loss(&logits, &y)?;
    let g = l.backward()?;
    finite_gradients(&m, &g)?;
    let grads = m
        .named_vars()
        .into_iter()
        .map(|(n, v)| {
            Ok((
                n,
                g.get(&v)
                    .ok_or("missing gradient")?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            ))
        })
        .collect::<Result<_>>()?;
    let r = Reference {
        config: tiny_config(),
        seed: 19,
        logits: logits.flatten_all()?.to_vec1::<f32>()?,
        loss: l.to_scalar::<f32>()?,
        grads,
    };
    write_new(path, &serde_json::to_vec_pretty(&r)?)
}
pub fn cuda() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let d = Device::new_cuda(0)?;
        if !d.is_cuda() {
            return Err("CUDA device selection failed".into());
        }
        Ok(d)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("rebuild with explicit cuda feature; no CPU fallback".into())
    }
}
fn compare(actual: &[f32], expected: &[f32]) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err("reference shape mismatch".into());
    }
    let mut max = 0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        let delta = f64::from((a - b).abs());
        max = max.max(delta);
        if !a.is_finite() || delta > 1e-5 + 0.001 * f64::from(b.abs()) {
            return Err(format!("CPU/CUDA reference mismatch: {a} vs {b}").into());
        }
    }
    Ok(max)
}
fn gpu_snapshot() -> Result<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-gpu=name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !out.status.success() {
        return Err("nvidia-smi failed".into());
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}
fn emit(out: &mut fs::File, row: serde_json::Value) -> Result<()> {
    use std::io::Write;
    writeln!(out, "{}", serde_json::to_string(&row)?)?;
    out.flush()?;
    println!("{}", serde_json::to_string(&row)?);
    Ok(())
}
fn init_hash(m: &Transformer) -> Result<String> {
    let mut bytes = Vec::new();
    for (name, var) in m.named_vars() {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        for v in var.flatten_all()?.to_vec1::<f32>()? {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(hash(&bytes))
}
fn batch(
    seq: &Sequence,
    starts: &[usize],
    context: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, u64)> {
    let mut x = Vec::with_capacity(starts.len() * context);
    let mut y = Vec::with_capacity(x.capacity());
    let mut bytes = 0;
    for &start in starts {
        let end = start.checked_add(context).ok_or("window overflow")?;
        x.extend_from_slice(seq.tokens.get(start..end).ok_or("input window")?);
        y.extend_from_slice(seq.tokens.get(start + 1..end + 1).ok_or("target window")?);
        bytes += seq.target_bytes(start + 1, context)?;
    }
    Ok((
        Tensor::from_vec(x, (starts.len(), context), device)?,
        Tensor::from_vec(y, (starts.len(), context), device)?,
        bytes,
    ))
}
fn causal_gate(m: &Transformer, device: &Device) -> Result<()> {
    let x = Tensor::new(&[[1_u32, 2, 3, 4]], device)?;
    let y = Tensor::new(&[[1_u32, 2, 99, 98]], device)?;
    let a = m
        .forward(&x)?
        .narrow(1, 0, 2)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = m
        .forward(&y)?
        .narrow(1, 0, 2)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if a != b {
        return Err("future tokens alter causal prefix".into());
    }
    Ok(())
}
fn save_checkpoint(model: &Transformer, path: &Path) -> Result<()> {
    // Retain the last complete image if writing/transfer is interrupted.
    let temporary = path.with_extension("safetensors.tmp");
    model.vars.save(&temporary)?;
    fs::File::open(&temporary)?.sync_all()?;
    fs::rename(&temporary, path)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}
pub fn preflight(data: &Path, reference_path: &Path, out_dir: &Path) -> Result<()> {
    // This command genuinely requires CUDA; all dataset/tokenizer work was completed locally.
    fs::create_dir(out_dir)?;
    let mut log = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out_dir.join("preflight.jsonl"))?;
    let snapshot = gpu_snapshot()?;
    if snapshot.split(',').next().map(str::trim) != Some("NVIDIA L4") {
        return Err(format!("unauthorized GPU: {snapshot}").into());
    }
    let now = Instant::now();
    let device = cuda()?;
    device.synchronize()?;
    emit(
        &mut log,
        serde_json::json!({"stage":"cuda","seconds":now.elapsed().as_secs_f64(),"gpu":snapshot,"precision":"FP32"}),
    )?;
    let reference: Reference = serde_json::from_slice(&fs::read(reference_path)?)?;
    let tiny = Transformer::new(reference.config, reference.seed, &device)?;
    let x = Tensor::new(&[[1_u32, 2, 3, 4]], &device)?;
    let y = Tensor::new(&[[2_u32, 3, 4, 1]], &device)?;
    let logits = tiny.forward(&x)?;
    if !logits.device().is_cuda() {
        return Err("forward left CUDA".into());
    }
    let l = loss(&logits, &y)?;
    let grads = l.backward()?;
    let mut max = compare(&logits.flatten_all()?.to_vec1::<f32>()?, &reference.logits)?;
    compare(&[l.to_scalar::<f32>()?], &[reference.loss])?;
    for (name, v) in tiny.named_vars() {
        let g = grads.get(&v).ok_or("CUDA gradient missing")?;
        if !g.device().is_cuda() {
            return Err("gradient left CUDA".into());
        }
        max = max.max(compare(
            &g.flatten_all()?.to_vec1::<f32>()?,
            reference.grads.get(&name).ok_or("reference name")?,
        )?);
    }
    emit(
        &mut log,
        serde_json::json!({"stage":"cpu-cuda-reference","max_absolute_delta":max,"gradient_families":finite_gradients(&tiny,&grads)?}),
    )?;
    drop((tiny, grads, logits, l));
    let mut expected_init = None;
    for variant in ["A", "C"] {
        let seq = Sequence::load(&data.join(format!("{variant}-train.seq")))?;
        if seq.tokens.len() < 4097 {
            return Err("insufficient preflight tokens".into());
        }
        let now = Instant::now();
        let m = Transformer::new(ModelConfig::default(), 20261008, &device)?;
        device.synchronize()?;
        let seconds = now.elapsed().as_secs_f64();
        let initial = init_hash(&m)?;
        if expected_init.as_ref().is_some_and(|h| h != &initial) {
            return Err("A/C initial parameters differ".into());
        }
        expected_init = Some(initial.clone());
        causal_gate(&m, &device)?;
        emit(
            &mut log,
            serde_json::json!({"stage":"initialization","variant":variant,"seconds":seconds,
            "parameters":m.config.parameter_count()?,"initial_hash":initial,"gpu":gpu_snapshot()?}),
        )?;
        let mut optimizer = AdamW::new(m.optimizer_vars(), adam())?;
        let mut times = Vec::new();
        let mut total_bytes = 0_u64;
        let mut pipeline_start = None;
        let mut rng = Rng::new(20261008 ^ 0xa341316c9e3779b9);
        for step in 0..18 {
            if step == 2 {
                pipeline_start = Some(Instant::now());
            }
            let starts: Vec<_> = (0..8)
                .map(|_| (rng.next_u64() as usize) % (seq.tokens.len() - 256))
                .collect();
            device.synchronize()?;
            let now = Instant::now();
            let (x, y, bytes) = batch(&seq, &starts, 256, &device)?;
            if step == 0 {
                let returned = x.to_vec2::<u32>()?;
                for (row, &start) in returned.iter().zip(&starts) {
                    if row.as_slice() != &seq.tokens[start..start + 256] {
                        return Err("CUDA input ID transfer changed canonical IDs".into());
                    }
                }
            }
            let logits = m.forward(&x)?;
            let l = loss(&logits, &y)?;
            let value = l.to_scalar::<f32>()?;
            if !value.is_finite() {
                return Err("nonfinite preflight loss".into());
            }
            let gradients = l.backward()?;
            if step == 0 {
                for v in m.optimizer_vars() {
                    if !v.device().is_cuda() {
                        return Err("parameter left CUDA".into());
                    }
                }
                emit(
                    &mut log,
                    serde_json::json!({"stage":"full-gradient-gate","variant":variant,"families":finite_gradients(&m,&gradients)?}),
                )?;
            }
            optimizer.step(&gradients)?;
            device.synchronize()?;
            let elapsed = now.elapsed().as_secs_f64();
            if step >= 2 {
                times.push(elapsed);
                total_bytes += bytes;
            }
            emit(
                &mut log,
                serde_json::json!({"stage":"step","variant":variant,"step":step+1,"seconds":elapsed,"loss":value,
                "targets":2048,"raw_target_bytes":bytes,"gpu":if step == 0 || step == 17 {Some(gpu_snapshot()?)} else {None}}),
            )?;
        }
        if init_hash(&m)? == initial {
            return Err("AdamW did not change primary weights".into());
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        emit(
            &mut log,
            serde_json::json!({"stage":"throughput","variant":variant,"measured_steps":times.len(),
            "mean_step_seconds":mean,"pipeline_seconds":pipeline_start.ok_or("missing timing boundary")?.elapsed().as_secs_f64(),"tokens_per_second":2048.0/mean,"raw_bytes_per_second":total_bytes as f64/times.iter().sum::<f64>()}),
        )?;
        let sample = Sequence {
            tokens: seq.tokens[..4097].to_vec(),
            bytes: seq.bytes[..4097].to_vec(),
        };
        let scored = evaluate(&m, &sample, 8, &device)?;
        if scored.targets != 4096 || scored.raw_target_bytes != sample.target_bytes(1, 4096)? {
            return Err("GPU byte-normalized evaluation coverage failed".into());
        }
        emit(
            &mut log,
            serde_json::json!({"stage":"train-prefix-evaluation", "variant":variant,
            "metrics":scored,"purpose":"TRAIN-only timing/normalization gate, no held-out scoring"}),
        )?;
        // Fixed auxiliary overfit fixture; never touches validation/test or chooses primary hyperparameters.
        if variant == "A" {
            let x = Tensor::new(&[[1_u32, 2, 3, 4, 1, 2, 3, 4]], &device)?;
            let y = Tensor::new(&[[2_u32, 3, 4, 1, 2, 3, 4, 1]], &device)?;
            let before = loss(&m.forward(&x)?, &y)?.to_scalar::<f32>()?;
            let now = Instant::now();
            let mut overfit = AdamW::new(
                m.optimizer_vars(),
                ParamsAdamW {
                    lr: 0.005,
                    weight_decay: 0.0,
                    ..adam()
                },
            )?;
            for _ in 0..100 {
                let l = loss(&m.forward(&x)?, &y)?;
                if !l.to_scalar::<f32>()?.is_finite() {
                    return Err("overfit diverged".into());
                }
                overfit.backward_step(&l)?;
            }
            device.synchronize()?;
            let after = loss(&m.forward(&x)?, &y)?.to_scalar::<f32>()?;
            emit(
                &mut log,
                serde_json::json!({"stage":"tiny-overfit","initial_loss":before,"final_loss":after,"steps":100,"seconds":now.elapsed().as_secs_f64(),"gpu":gpu_snapshot()?}),
            )?;
            if !after.is_finite() || after >= 0.25 || after >= before * 0.1 {
                return Err("tiny GPU overfit gate failed".into());
            }
            let expected = m.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            let path = out_dir.join("preflight.safetensors");
            let now = Instant::now();
            save_checkpoint(&m, &path)?;
            device.synchronize()?;
            let seconds = now.elapsed().as_secs_f64();
            emit(
                &mut log,
                serde_json::json!({"stage":"checkpoint-write","seconds":seconds,"bytes":fs::metadata(&path)?.len(),"sha256":"computed locally after download/release"}),
            )?;
            let mut restored = Transformer::new(m.config.clone(), 20261009, &device)?;
            if restored.forward(&x)?.flatten_all()?.to_vec1::<f32>()? == expected {
                return Err("checkpoint fixture failed to distinguish initial weights".into());
            }
            restored.load(&path)?;
            compare(
                &restored.forward(&x)?.flatten_all()?.to_vec1::<f32>()?,
                &expected,
            )?;
            let metrics = bits_per_byte(f64::from(after) * 8.0, 8, 8)?;
            emit(
                &mut log,
                serde_json::json!({"stage":"checkpoint-reload-byte-normalization","bits_per_byte":metrics,"raw_target_bytes":8,"targets":8}),
            )?;
        }
    }
    emit(
        &mut log,
        serde_json::json!({"stage":"gate","status":"PASS","final_gpu":gpu_snapshot()?}),
    )?;
    Ok(())
}
pub fn adam() -> ParamsAdamW {
    ParamsAdamW {
        lr: 0.0003,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    }
}
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub model: ModelConfig,
    pub batch: usize,
    pub steps: usize,
    pub seeds: Vec<u64>,
    pub byte_budget: u64,
    pub precision: String,
    pub evaluation_every: usize,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            batch: 8,
            steps: 2000,
            seeds: vec![20261008, 20261009, 20261010],
            byte_budget: 8 * 1024 * 1024,
            precision: "FP32".into(),
            evaluation_every: 500,
        }
    }
}
#[derive(Serialize)]
struct Score {
    loss_per_token: f64,
    nll: f64,
    bits_per_byte: f64,
    targets: u64,
    raw_target_bytes: u64,
    seconds: f64,
}
fn evaluate(m: &Transformer, seq: &Sequence, batch_size: usize, device: &Device) -> Result<Score> {
    seq.validate()?;
    if seq.tokens.len() < 2 || batch_size == 0 {
        return Err("evaluation requires a target and a positive batch size".into());
    }
    device.synchronize()?;
    let now = Instant::now();
    let mut start = 0;
    let mut nll = 0.0;
    let mut targets = 0;
    let mut bytes = 0;
    while start + 1 < seq.tokens.len() {
        let context = m.config.context.min(seq.tokens.len() - 1 - start);
        let mut starts = vec![start];
        start += context;
        while starts.len() < batch_size && start + context < seq.tokens.len() {
            starts.push(start);
            start += context;
        }
        let (x, y, b) = batch(seq, &starts, context, device)?;
        let count = (starts.len() * context) as u64;
        let value = loss(&m.forward(&x)?, &y)?.to_scalar::<f32>()? as f64;
        if !value.is_finite() {
            return Err("nonfinite evaluation".into());
        }
        nll += value * count as f64;
        targets += count;
        bytes += b;
    }
    device.synchronize()?;
    if targets != seq.tokens.len().saturating_sub(1) as u64
        || bytes != seq.target_bytes(1, seq.tokens.len() - 1)?
    {
        return Err("evaluation coverage mismatch".into());
    }
    Ok(Score {
        loss_per_token: nll / targets as f64,
        nll,
        bits_per_byte: bits_per_byte(nll, targets, bytes)?,
        targets,
        raw_target_bytes: bytes,
        seconds: now.elapsed().as_secs_f64(),
    })
}
pub fn train(
    config: &TrainConfig,
    data: &Path,
    out: &Path,
    variant: &str,
    regime: &str,
    seed: u64,
) -> Result<()> {
    if config != &TrainConfig::default()
        || !["A", "C"].contains(&variant)
        || !["T", "B"].contains(&regime)
        || !config.seeds.contains(&seed)
    {
        return Err("final job violates frozen primary configuration".into());
    }
    fs::create_dir(out)?;
    let mut log = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("metrics.jsonl"))?;
    let device = cuda()?;
    if gpu_snapshot()?.split(',').next().map(str::trim) != Some("NVIDIA L4") {
        return Err("unauthorized GPU".into());
    }
    let train = Sequence::load(&data.join(format!("{variant}-train.seq")))?;
    let validation = Sequence::load(&data.join(format!("{variant}-validation.seq")))?;
    let test = Sequence::load(&data.join(format!("{variant}-test.seq")))?;
    if train.tokens.len() <= config.model.context || config.byte_budget == 0 {
        return Err("insufficient TRAIN window or empty byte budget".into());
    }
    let job_start = Instant::now();
    let m = Transformer::new(config.model.clone(), seed, &device)?;
    let mut optimizer = AdamW::new(m.optimizer_vars(), adam())?;
    let mut rng = Rng::new(seed ^ 0xa341316c9e3779b9);
    let mut step = 0;
    let mut bytes = 0;
    let mut positions = 0;
    let mut cursor = 0;
    let mut training_seconds = 0.0;
    let mut training_pipeline_seconds = 0.0;
    let mut checkpoint_seconds = 0.0;
    let mut last_validation = None;
    let mut last_checkpoint_step = None;
    while if regime == "T" {
        step < config.steps
    } else {
        bytes < config.byte_budget
    } {
        let pipeline_start = Instant::now();
        let mut starts = Vec::new();
        for _ in 0..config.batch {
            if regime == "T" {
                starts
                    .push((rng.next_u64() as usize) % (train.tokens.len() - config.model.context));
            } else {
                if cursor + config.model.context >= train.tokens.len() {
                    return Err("byte budget exceeds single immutable TRAIN pass".into());
                }
                starts.push(cursor);
                cursor += config.model.context;
            }
        }
        device.synchronize()?;
        let now = Instant::now();
        let (x, y, b) = batch(&train, &starts, config.model.context, &device)?;
        let l = loss(&m.forward(&x)?, &y)?;
        let value = l.to_scalar::<f32>()?;
        if !value.is_finite() {
            return Err("nonfinite training".into());
        }
        let gradients = l.backward()?;
        if step % 100 == 0 {
            finite_gradients(&m, &gradients)?;
        }
        optimizer.step(&gradients)?;
        device.synchronize()?;
        let seconds = now.elapsed().as_secs_f64();
        training_seconds += seconds;
        step += 1;
        bytes += b;
        positions += (config.batch * config.model.context) as u64;
        emit(
            &mut log,
            serde_json::json!({"stage":"train","variant":variant,"regime":regime,"seed":seed,"step":step,
            "loss_per_token":value,"step_seconds":seconds,"train_seconds":training_seconds,"targets":positions,"raw_target_bytes":bytes,
            "gpu":if step == 1 || step.is_multiple_of(100) {Some(gpu_snapshot()?)} else {None}}),
        )?;
        training_pipeline_seconds += pipeline_start.elapsed().as_secs_f64();
        if step == 1 || step % config.evaluation_every == 0 {
            let score = evaluate(&m, &validation, config.batch, &device)?;
            emit(
                &mut log,
                serde_json::json!({"stage":"validation","step":step,"metrics":score}),
            )?;
            let save_start = Instant::now();
            save_checkpoint(&m, &out.join("latest.safetensors"))?;
            checkpoint_seconds += save_start.elapsed().as_secs_f64();
            last_checkpoint_step = Some(step);
            last_validation = Some((step, score));
        }
    }
    let val = match last_validation {
        Some((scored_step, score)) if scored_step == step => score,
        _ => evaluate(&m, &validation, config.batch, &device)?,
    };
    let test = evaluate(&m, &test, config.batch, &device)?;
    let path = out.join("final.safetensors");
    let save_start = Instant::now();
    let latest = out.join("latest.safetensors");
    if last_checkpoint_step != Some(step) {
        save_checkpoint(&m, &latest)?;
    }
    fs::rename(&latest, &path)?;
    fs::File::open(out)?.sync_all()?;
    checkpoint_seconds += save_start.elapsed().as_secs_f64();
    let checkpoint_bytes = fs::metadata(&path)?.len();
    emit(
        &mut log,
        serde_json::json!({"stage":"final","variant":variant,"regime":regime,"seed":seed,"steps":step,"targets":positions,
        "raw_target_bytes":bytes,"train_seconds":training_seconds,"train_pipeline_seconds":training_pipeline_seconds,
        "checkpoint_seconds":checkpoint_seconds,"job_seconds":job_start.elapsed().as_secs_f64(),"tokens_per_second":positions as f64/training_seconds,
        "raw_bytes_per_second":bytes as f64/training_seconds,"parameters":m.config.parameter_count()?,"validation":val,"test":test,
        "checkpoint_sha256":"computed locally after download/release","checkpoint_bytes":checkpoint_bytes,"precision":"FP32","gpu":gpu_snapshot()?}),
    )?;
    Ok(())
}
pub fn plan(config: &TrainConfig, data: &Path, path: &Path) -> Result<()> {
    let mut variants = Vec::new();
    for variant in ["A", "C"] {
        let seq = Sequence::load(&data.join(format!("{variant}-train.seq")))?;
        let per = config.batch * config.model.context;
        let mut steps = 0;
        let mut bytes = 0;
        while bytes < config.byte_budget {
            bytes += seq.target_bytes(1 + steps * per, per)?;
            steps += 1;
        }
        variants.push(serde_json::json!({"variant":variant,"T_steps":config.steps,"T_targets":config.steps*per,
            "B_steps":steps,"B_raw_target_bytes":bytes,"B_targets":steps*per,"train_raw_bytes":seq.bytes.iter().map(|&b|u64::from(b)).sum::<u64>()}));
    }
    write_new(
        path,
        &serde_json::to_vec_pretty(
            &serde_json::json!({"config":config,"variants":variants,"parameters":config.model.parameter_count()?}),
        )?,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_job_rejects_undeclared_seed_before_allocating_or_writing() {
        let path =
            std::env::temp_dir().join(format!("packtok-m5-invalid-seed-{}", std::process::id()));
        assert!(!path.exists());
        assert!(
            train(
                &TrainConfig::default(),
                Path::new("absent-data"),
                &path,
                "A",
                "T",
                999
            )
            .is_err()
        );
        assert!(!path.exists());
    }
    #[test]
    fn evaluation_scores_every_target_and_no_unrepresented_bytes() -> Result<()> {
        let seq = Sequence {
            tokens: (0..23).map(|x| x % 5).collect(),
            bytes: vec![2; 23],
        };
        let m = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
        let s = evaluate(&m, &seq, 3, &Device::Cpu)?;
        assert_eq!(s.targets, 22);
        assert_eq!(s.raw_target_bytes, 44);
        assert!(s.bits_per_byte.is_finite());
        Ok(())
    }
    #[test]
    fn evaluation_rejects_empty_single_token_and_zero_batch() -> Result<()> {
        let m = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
        for tokens in [vec![], vec![1]] {
            let seq = Sequence {
                bytes: vec![1; tokens.len()],
                tokens,
            };
            assert!(evaluate(&m, &seq, 1, &Device::Cpu).is_err());
        }
        let seq = Sequence {
            tokens: vec![1, 2],
            bytes: vec![1, 1],
        };
        assert!(evaluate(&m, &seq, 0, &Device::Cpu).is_err());
        Ok(())
    }
    #[test]
    fn checkpoint_retains_complete_image_with_an_interrupted_temporary() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "packtok-m5-atomic-{}.safetensors",
            std::process::id()
        ));
        let temporary = path.with_extension("safetensors.tmp");
        let a = Transformer::new(tiny_config(), 19, &Device::Cpu)?;
        let b = Transformer::new(tiny_config(), 20, &Device::Cpu)?;
        save_checkpoint(&a, &path)?;
        fs::write(&temporary, b"interrupted write")?;
        let mut restored = Transformer::new(tiny_config(), 100, &Device::Cpu)?;
        restored.load(&path)?;
        assert_eq!(init_hash(&restored)?, init_hash(&a)?);
        save_checkpoint(&b, &path)?;
        restored.load(&path)?;
        assert_eq!(init_hash(&restored)?, init_hash(&b)?);
        assert!(!temporary.exists());
        fs::remove_file(path)?;
        Ok(())
    }
    #[test]
    fn paired_seed_training_is_reproducible() -> Result<()> {
        let x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
        let y = Tensor::new(&[[2_u32, 3, 4, 1]], &Device::Cpu)?;
        let a = Transformer::new(tiny_config(), 20261008, &Device::Cpu)?;
        let b = Transformer::new(tiny_config(), 20261008, &Device::Cpu)?;
        for m in [&a, &b] {
            let mut o = AdamW::new(m.optimizer_vars(), adam())?;
            for _ in 0..3 {
                o.backward_step(&loss(&m.forward(&x)?, &y)?)?;
            }
        }
        assert_eq!(init_hash(&a)?, init_hash(&b)?);
        Ok(())
    }
}
