//! Explicit bounded execution; default GPU path never falls back to CPU/presets.
use crate::{
    Result,
    config::TrainPlan,
    data::{self, Dataset},
    model::{self, Transformer},
    optim::{AdamConfig, CheckpointProgress, Optimizer, prune_checkpoints},
};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::Command,
    time::Instant,
};

const USAGE: &str = "packtok-semantic-scale prepare --input DIR --output DIR\n  plan --data DIR --out CONFIG --preset tiny|100m|1b [--context 128 --batch 1 --accum 4 --dtype bf16 --variant semantic --steps 100 --seed 42 --max-seconds 600 --checkpoint-every 25]\n  preflight --config CONFIG --output RUN [--cpu]\n  train --config CONFIG --output RUN [--resume] [--cpu]\n  evaluate --config CONFIG --output RUN [--cpu]";

pub fn cli(args: Vec<String>) -> Result<()> {
    let command = args.first().ok_or(USAGE)?;
    if ["help", "--help", "-h"].contains(&command.as_str()) {
        println!("{USAGE}");
        return Ok(());
    }
    let mut options = BTreeMap::new();
    let mut i = 1;
    while i < args.len() {
        let key = &args[i];
        if !key.starts_with("--") || options.contains_key(key) {
            return Err("expected unique --option arguments".into());
        }
        if ["--resume", "--cpu"].contains(&key.as_str()) {
            options.insert(key.clone(), "true".to_owned());
            i += 1;
        } else {
            let value = args.get(i + 1).ok_or("missing option value")?;
            if value.starts_with("--") {
                return Err("missing option value".into());
            }
            options.insert(key.clone(), value.clone());
            i += 2;
        }
    }
    let allowed: &[&str] = match command.as_str() {
        "prepare" => &["--input", "--output"],
        "plan" => &[
            "--data",
            "--out",
            "--preset",
            "--context",
            "--batch",
            "--accum",
            "--dtype",
            "--variant",
            "--steps",
            "--seed",
            "--max-seconds",
            "--checkpoint-every",
        ],
        "preflight" | "evaluate" => &["--config", "--output", "--cpu"],
        "train" => &["--config", "--output", "--cpu", "--resume"],
        _ => return Err(USAGE.into()),
    };
    if options.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err("unknown option for command".into());
    }
    let required = |key: &str| -> Result<&str> {
        options
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {key}").into())
    };
    if command == "prepare" {
        return data::prepare(
            Path::new(required("--input")?),
            Path::new(required("--output")?),
        );
    }
    if command == "plan" {
        let value = |key: &str, fallback: &str| {
            options
                .get(key)
                .cloned()
                .unwrap_or_else(|| fallback.to_owned())
        };
        let preset = value("--preset", "tiny");
        let context = value("--context", "128").parse()?;
        let steps = value("--steps", "100").parse()?;
        let data_path = fs::canonicalize(required("--data")?)?;
        for split in ["train", "validation", "test"] {
            if Dataset::open(&data_path, split)?.len() <= context as u64 {
                return Err(format!("{split} needs at least context+1 lexical tokens").into());
            }
        }
        let plan = TrainPlan {
            version: 1,
            model: model::ModelConfig::preset(&preset, context)?,
            preset,
            seed: value("--seed", "42").parse()?,
            variant: value("--variant", "semantic"),
            dtype: value("--dtype", "bf16"),
            micro_batch: value("--batch", "1").parse()?,
            accumulation: value("--accum", "4").parse()?,
            steps,
            max_seconds: value("--max-seconds", "600").parse()?,
            checkpoint_every: value("--checkpoint-every", "25").parse()?,
            dataset_identity: data::dataset_identity(&data_path)?,
            data_path: data_path
                .to_str()
                .ok_or("data path must be UTF-8")?
                .to_owned(),
            activation_checkpointing: true,
            optimizer: AdamConfig {
                total_steps: steps,
                warmup_steps: (steps / 10).min(100),
                ..AdamConfig::default()
            },
        };
        plan.validate()?;
        let bytes = serde_json::to_vec_pretty(&plan)?;
        let path = Path::new(required("--out")?);
        if path.exists() {
            if fs::read(path)? != bytes {
                return Err(
                    "existing run plan differs; use a new run name instead of overwriting".into(),
                );
            }
        } else {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, &bytes)?;
        }
        println!(
            "{}",
            json!({"config":path,"identity":plan.identity()?,"parameters":plan.model.parameter_count()?,"estimated_training_bytes":plan.estimated_training_bytes()?,"estimate_is_not_measured":true})
        );
        return Ok(());
    }
    let plan = TrainPlan::load(Path::new(required("--config")?))?;
    let output = Path::new(required("--output")?);
    let cpu = options.contains_key("--cpu");
    match command.as_str() {
        "preflight" => preflight(&plan, output, cpu),
        "train" => train(&plan, output, cpu, options.contains_key("--resume")),
        "evaluate" => evaluate(&plan, output, cpu),
        _ => unreachable!(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Hardware {
    name: String,
    uuid: String,
    total_mib: u64,
    free_mib: u64,
}
fn hardware() -> Result<Hardware> {
    let result = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !result.status.success() {
        return Err("nvidia-smi failed; select an NVIDIA L4 Colab GPU runtime".into());
    }
    let text = std::str::from_utf8(&result.stdout)?;
    parse_hardware(text)
}
fn parse_hardware(text: &str) -> Result<Hardware> {
    let lines: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if lines.len() != 1 {
        return Err("this single-device experiment requires exactly one visible GPU".into());
    }
    let values: Vec<_> = lines[0].split(',').map(str::trim).collect();
    if values.len() != 4 || !matches!(values[0], "NVIDIA L4" | "NVIDIA L4 GPU") {
        return Err(
            "this notebook targets NVIDIA L4; hardware is never silently substituted".into(),
        );
    }
    Ok(Hardware {
        name: values[0].into(),
        uuid: values[1].into(),
        total_mib: values[2].parse()?,
        free_mib: values[3].parse()?,
    })
}
fn device(plan: &TrainPlan, cpu: bool) -> Result<(Device, Option<Hardware>)> {
    plan.validate()?;
    if cpu {
        if plan.preset != "tiny" || plan.dtype != "f32" || plan.model.parameter_count()? > 5_000_000
        {
            return Err(
                "CPU diagnostics require tiny preset and f32; full model CPU training is disabled"
                    .into(),
            );
        }
        return Ok((Device::Cpu, None));
    }
    let gpu = hardware()?;
    if plan.estimated_training_bytes()? > gpu.free_mib * 1024 * 1024 {
        return Err(format!("estimated training memory {} MiB exceeds free GPU memory {} MiB; reduce context/microbatch explicitly or use a separate smaller-preset run", plan.estimated_training_bytes()? / 1024 / 1024, gpu.free_mib).into());
    }
    #[cfg(not(feature = "cuda"))]
    return Err("CUDA support is not compiled; build with --features cuda".into());
    #[cfg(feature = "cuda")]
    Ok((Device::new_cuda(0)?, Some(gpu)))
}
fn dtype(plan: &TrainPlan) -> DType {
    if plan.dtype == "bf16" {
        DType::BF16
    } else {
        DType::F32
    }
}
fn validate_data(plan: &TrainPlan) -> Result<()> {
    if data::dataset_identity(Path::new(&plan.data_path))? != plan.dataset_identity {
        return Err("prepared data manifest differs from run plan".into());
    }
    Ok(())
}
fn check_progress(
    plan: &TrainPlan,
    progress: &CheckpointProgress,
    step: usize,
    dataset: &Dataset,
) -> Result<()> {
    let expected = (step as u128)
        * plan.micro_batch as u128
        * plan.model.context as u128
        * plan.accumulation as u128;
    let windows_per_epoch = (dataset.len() - 1) / plan.model.context as u64;
    if windows_per_epoch == 0 {
        return Err("dataset too short for resume".into());
    }
    let windows = (step as u128) * plan.micro_batch as u128 * plan.accumulation as u128;
    let expected_cursor = if windows == 0 {
        0
    } else {
        (((windows - 1) % u128::from(windows_per_epoch) + 1) * plan.model.context as u128) as u64
    };
    if progress.seed != plan.seed
        || progress.cursor != expected_cursor
        || u128::from(progress.tokens_seen) != expected
        || progress.cursor >= dataset.len()
        || (step == 0 && (progress.cursor != 0 || progress.target_bytes_seen != 0))
    {
        return Err("checkpoint sampler seed/counters do not match run plan".into());
    }
    Ok(())
}
fn weights_hash(model: &Transformer) -> Result<String> {
    let mut hash = Sha256::new();
    for (name, var) in model.named_vars() {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name);
        let values = var.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let mut file = fs::File::create(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}
fn emit(log: &mut fs::File, value: serde_json::Value) -> Result<()> {
    let line = serde_json::to_string(&value)?;
    writeln!(log, "{line}")?;
    log.flush()?;
    println!("{line}");
    Ok(())
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Preflight {
    version: u32,
    config_identity: String,
    passed: bool,
    parameters: usize,
    hardware: Option<Hardware>,
    steps: usize,
    binary_sha256: String,
    seconds: f64,
    checkpoint_weights_equal: bool,
    cuda_runtime_tested: bool,
}

fn binary_hash() -> Result<String> {
    let mut input = fs::File::open(std::env::current_exe()?)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn preflight_deadline(plan: &TrainPlan, start: &Instant) -> Result<()> {
    if start.elapsed().as_secs() >= plan.max_seconds.min(1200) {
        return Err("preflight deadline reached; no PASS recorded".into());
    }
    Ok(())
}

fn lock_run(output: &Path) -> Result<fs::File> {
    fs::create_dir_all(output)?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(output.join(".run.lock"))?;
    file.try_lock()
        .map_err(|error| format!("run directory is already in use: {error}"))?;
    Ok(file) // OS releases advisory lock even after a killed process.
}

fn bind_run(plan: &TrainPlan, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let path = output.join("run-plan.json");
    if path.exists() {
        if TrainPlan::load(&path)?.identity()? != plan.identity()? {
            return Err(
                "run directory belongs to a different plan; choose a fresh run name".into(),
            );
        }
    } else {
        write_json(&path, plan)?;
    }
    Ok(())
}

/// Space for preflight + two retained generations + one new generation. This
/// estimate precedes model allocation. Drive quota is handled by notebook backup.
fn check_disk(plan: &TrainPlan, output: &Path) -> Result<()> {
    if plan.model.parameter_count()? < 5_000_000 {
        return Ok(());
    }
    let result = Command::new("df")
        .args(["-B1", "--output=avail"])
        .arg(output)
        .output()?;
    if !result.status.success() {
        return Err("cannot determine local checkpoint free space".into());
    }
    let text = std::str::from_utf8(&result.stdout)?;
    let available: u64 = text
        .lines()
        .last()
        .ok_or("missing disk free space")?
        .trim()
        .parse()?;
    let one = (plan.model.parameter_count()? as u64) * if plan.dtype == "bf16" { 14 } else { 16 }
        + 4 * 1024 * 1024;
    let mut existing = 0_u64;
    for directory in [
        output.join("checkpoints"),
        output.join("preflight-checkpoints"),
    ] {
        if !directory.exists() {
            continue;
        }
        for item in fs::read_dir(directory)? {
            let item = item?;
            if !item.file_type()?.is_dir() || item.file_name().to_string_lossy().ends_with(".tmp") {
                continue;
            }
            let index = item.path().join("index.json");
            if !index.exists() || fs::metadata(&index)?.len() > 4 * 1024 * 1024 {
                continue;
            }
            let record: serde_json::Value = serde_json::from_slice(&fs::read(index)?)?;
            if record["identity"].as_str() == Some(plan.identity()?.as_str()) {
                existing += 1;
            }
        }
    }
    let additional_images = 4_u64.saturating_sub(existing.min(3));
    let needed = one
        .checked_mul(additional_images)
        .and_then(|n| n.checked_add(2 * 1024_u64.pow(3)))
        .ok_or("checkpoint disk estimate overflow")?;
    if available < needed {
        return Err(format!("checkpoint storage estimate needs {} GiB free locally; available {} GiB (preflight, two retained generations, new write and reserve)", needed / 1024_u64.pow(3), available / 1024_u64.pow(3)).into());
    }
    Ok(())
}

fn reconcile_metrics(path: &Path, step: usize, identity: &str) -> Result<()> {
    if !path.exists() {
        if step > 0 {
            return Err(
                "checkpoint needs its metrics log to resume; restore the complete run directory"
                    .into(),
            );
        }
        return Ok(());
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let archive = path.with_file_name(format!("metrics-before-resume-{stamp}.jsonl"));
    fs::copy(path, &archive)?;
    let temporary = path.with_extension("reconciled.tmp");
    let mut output = fs::File::create(&temporary)?;
    let mut input = BufReader::new(fs::File::open(path)?);
    let mut canonical_train_step = 0_u64;
    loop {
        let mut line = Vec::new();
        let read = input
            .by_ref()
            .take(1024 * 1024 + 1)
            .read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        if line.len() > 1024 * 1024 {
            return Err("metrics line exceeds 1 MiB".into());
        }
        let row: serde_json::Value = match serde_json::from_slice(&line) {
            Ok(row) => row,
            Err(_) if !line.ends_with(b"\n") => break, // interrupted trailing log write; original archived
            Err(error) => return Err(error.into()),
        };
        if row
            .get("identity")
            .and_then(|v| v.as_str())
            .is_some_and(|id| id != identity)
        {
            return Err("metrics belong to another run identity".into());
        }
        let row_step = if row.get("stage").and_then(|v| v.as_str()) == Some("train") {
            let row_step = row
                .get("step")
                .and_then(|v| v.as_u64())
                .filter(|&n| n > 0)
                .ok_or("train metrics need a positive integer step")?;
            if row_step <= step as u64 {
                if row_step != canonical_train_step + 1 {
                    return Err("train metrics must contain each step through the checkpoint exactly once and in order".into());
                }
                canonical_train_step = row_step;
            }
            Some(row_step)
        } else {
            row.get("step")
                .or_else(|| row.get("completed_steps"))
                .and_then(|v| v.as_u64())
        };
        if row_step.is_none_or(|n| n <= step as u64) {
            output.write_all(&line)?;
            if !line.ends_with(b"\n") {
                output.write_all(b"\n")?;
            }
        }
    }
    if canonical_train_step != step as u64 {
        return Err("checkpoint needs complete train metrics through its saved step; restore the complete run directory".into());
    }
    writeln!(
        output,
        "{}",
        json!({"stage":"resume_boundary","checkpoint_step":step,"archived_previous_metrics":archive.file_name()})
    )?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn preflight(plan: &TrainPlan, output: &Path, cpu: bool) -> Result<()> {
    let _lock = lock_run(output)?;
    let started = Instant::now();
    bind_run(plan, output)?;
    let marker = output.join("preflight.json");
    // Remove an earlier PASS before attempting a new preflight. Failed runs can
    // never leave a stale success marker for a later train command.
    if marker.exists() {
        fs::remove_file(&marker)?;
    }
    validate_data(plan)?;
    check_disk(plan, output)?;
    let (device, gpu) = device(plan, cpu)?;
    let identity = plan.identity()?;
    let dataset = Dataset::open(Path::new(&plan.data_path), "train")?;
    let model = Transformer::new(plan.model.clone(), plan.seed, &device, dtype(plan))?;
    preflight_deadline(plan, &started)?;
    let mut optimizer = Optimizer::new(&model, plan.optimizer.clone())?;
    preflight_deadline(plan, &started)?;
    let batch = dataset.batch(
        0,
        plan.micro_batch,
        plan.model.context,
        &device,
        plan.variant == "semantic",
    )?;
    let mut gradients = BTreeMap::new();
    let before = weights_hash(&model)?;
    let loss = model.accumulate_gradients(
        &batch.inputs,
        &batch.semantic,
        &batch.targets,
        true,
        &mut gradients,
        1.0,
    )?;
    if !loss.is_finite() {
        return Err("nonfinite preflight loss".into());
    }
    optimizer.step(&model, &gradients)?;
    device.synchronize()?;
    drop(gradients);
    if before == weights_hash(&model)? {
        return Err("preflight optimizer failed to update model weights".into());
    }
    preflight_deadline(plan, &started)?;
    let checkpoint = output.join("preflight-checkpoints");
    let progress = CheckpointProgress {
        cursor: batch.next_cursor,
        tokens_seen: (plan.micro_batch * plan.model.context) as u64,
        target_bytes_seen: batch.target_bytes,
        seed: plan.seed,
    };
    optimizer.save(&model, &checkpoint, &identity, &progress)?;
    prune_checkpoints(&checkpoint, &identity, 1)?;
    let expected_weights = weights_hash(&model)?;
    drop(optimizer);
    preflight_deadline(plan, &started)?;
    let (mut optimizer, restored) =
        Optimizer::load(&model, &checkpoint, &identity, &plan.optimizer)?;
    if restored != progress
        || optimizer.step_count() != 1
        || weights_hash(&model)? != expected_weights
    {
        return Err("preflight checkpoint roundtrip failed".into());
    }
    preflight_deadline(plan, &started)?;
    let mut gradients = BTreeMap::new();
    model.accumulate_gradients(
        &batch.inputs,
        &batch.semantic,
        &batch.targets,
        true,
        &mut gradients,
        1.0,
    )?;
    optimizer.step(&model, &gradients)?;
    device.synchronize()?;
    if started.elapsed().as_secs() > plan.max_seconds.min(1200) {
        return Err(
            "preflight exceeded configured/20-minute soft wall limit; no PASS recorded".into(),
        );
    }
    let record = Preflight {
        version: 1,
        config_identity: identity,
        passed: true,
        parameters: plan.model.parameter_count()?,
        hardware: gpu,
        steps: 2,
        binary_sha256: binary_hash()?,
        seconds: started.elapsed().as_secs_f64(),
        checkpoint_weights_equal: true,
        cuda_runtime_tested: !cpu,
    };
    write_json(&marker, &record)?;
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

fn require_preflight(
    plan: &TrainPlan,
    output: &Path,
    cpu: bool,
    gpu: &Option<Hardware>,
) -> Result<()> {
    let path = output.join("preflight.json");
    if fs::metadata(&path)?.len() > 64 * 1024 {
        return Err("oversized preflight marker".into());
    }
    let record: Preflight = serde_json::from_slice(&fs::read(path)?)?;
    if record.version != 1
        || !record.passed
        || record.config_identity != plan.identity()?
        || record.binary_sha256 != binary_hash()?
        || record.cuda_runtime_tested == cpu
        || record.parameters != plan.model.parameter_count()?
        || record.hardware.as_ref().map(|h| &h.name) != gpu.as_ref().map(|h| &h.name)
    {
        return Err("matching successful preflight is required before training".into());
    }
    Ok(())
}
fn train(plan: &TrainPlan, output: &Path, cpu: bool, resume: bool) -> Result<()> {
    let _lock = lock_run(output)?;
    let started = Instant::now();
    bind_run(plan, output)?;
    check_disk(plan, output)?;
    validate_data(plan)?;
    let (device, gpu) = device(plan, cpu)?;
    require_preflight(plan, output, cpu, &gpu)?;
    let dataset = Dataset::open(Path::new(&plan.data_path), "train")?;
    let identity = plan.identity()?;
    let checkpoints = output.join("checkpoints");
    let model = Transformer::new(plan.model.clone(), plan.seed, &device, dtype(plan))?;
    let (mut optimizer, mut progress) = if checkpoints.join("latest.json").exists() {
        if !resume {
            return Err("checkpoint exists; pass --resume or choose a fresh run directory".into());
        }
        Optimizer::load(&model, &checkpoints, &identity, &plan.optimizer)?
    } else {
        (
            Optimizer::new(&model, plan.optimizer.clone())?,
            CheckpointProgress {
                cursor: 0,
                tokens_seen: 0,
                target_bytes_seen: 0,
                seed: plan.seed,
            },
        )
    };
    check_progress(plan, &progress, optimizer.step_count(), &dataset)?;
    if resume {
        reconcile_metrics(
            &output.join("metrics.jsonl"),
            optimizer.step_count(),
            &identity,
        )?;
    }
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("metrics.jsonl"))?;
    emit(
        &mut log,
        json!({"stage":"start","identity":identity,"step":optimizer.step_count(),"cursor":progress.cursor,"hardware":gpu,"precision":plan.dtype,"parameters":plan.model.parameter_count()?,"checkpointing":true,"accumulation":plan.accumulation,"source_commit":source_commit()}),
    )?;
    let mut saved_step = optimizer.step_count();
    'updates: while optimizer.step_count() < plan.steps {
        if started.elapsed().as_secs() >= plan.max_seconds {
            break;
        }
        let before = progress.clone();
        let update = Instant::now();
        let mut gradients = BTreeMap::<String, Tensor>::new();
        let mut mean_loss = 0.0;
        for _ in 0..plan.accumulation {
            if started.elapsed().as_secs() >= plan.max_seconds {
                progress = before;
                break 'updates;
            }
            let batch = dataset.batch(
                progress.cursor,
                plan.micro_batch,
                plan.model.context,
                &device,
                plan.variant == "semantic",
            )?;
            mean_loss += model.accumulate_gradients(
                &batch.inputs,
                &batch.semantic,
                &batch.targets,
                true,
                &mut gradients,
                1.0 / plan.accumulation as f64,
            )? / plan.accumulation as f64;
            progress.cursor = batch.next_cursor;
            progress.tokens_seen = progress
                .tokens_seen
                .checked_add((plan.micro_batch * plan.model.context) as u64)
                .ok_or("token counter overflow")?;
            progress.target_bytes_seen = progress
                .target_bytes_seen
                .checked_add(batch.target_bytes)
                .ok_or("byte counter overflow")?;
        }
        let lr = optimizer.step(&model, &gradients)?;
        device.synchronize()?;
        drop(gradients);
        let seconds = update.elapsed().as_secs_f64();
        emit(
            &mut log,
            json!({"stage":"train","step":optimizer.step_count(),"loss_per_token":mean_loss,"learning_rate":lr,"update_seconds":seconds,"tokens_per_second":(plan.micro_batch*plan.model.context*plan.accumulation) as f64/seconds,"tokens_seen":progress.tokens_seen,"target_bytes_seen":progress.target_bytes_seen,"cursor":progress.cursor,"active_wall_seconds":started.elapsed().as_secs_f64(),"sampled_gpu":if cpu {None}else{hardware().ok()}}),
        )?;
        if optimizer.step_count() % plan.checkpoint_every == 0 {
            optimizer.save(&model, &checkpoints, &identity, &progress)?;
            prune_checkpoints(&checkpoints, &identity, 2)?;
            saved_step = optimizer.step_count();
            emit(
                &mut log,
                json!({"stage":"checkpoint","step":saved_step,"active_wall_seconds":started.elapsed().as_secs_f64()}),
            )?;
        }
    }
    if saved_step != optimizer.step_count() || !checkpoints.join("latest.json").exists() {
        optimizer.save(&model, &checkpoints, &identity, &progress)?;
        prune_checkpoints(&checkpoints, &identity, 2)?;
    }
    let summary = json!({"version":1,"config_identity":identity,"completed_steps":optimizer.step_count(),"requested_steps":plan.steps,"tokens_seen":progress.tokens_seen,"target_bytes_seen":progress.target_bytes_seen,"active_wall_seconds":started.elapsed().as_secs_f64(),"wall_limit_soft_between_microbatches":true,"checkpoint_time_can_exceed_deadline":true,"complete":optimizer.step_count()==plan.steps});
    write_json(&output.join("summary.json"), &summary)?;
    emit(&mut log, summary)?;
    Ok(())
}
fn source_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn evaluate(plan: &TrainPlan, output: &Path, cpu: bool) -> Result<()> {
    let _lock = lock_run(output)?;
    validate_data(plan)?;
    let (device, gpu) = device(plan, cpu)?;
    let model = Transformer::new(plan.model.clone(), plan.seed, &device, dtype(plan))?;
    let (optimizer, progress) = Optimizer::load(
        &model,
        &output.join("checkpoints"),
        &plan.identity()?,
        &plan.optimizer,
    )?;
    let step = optimizer.step_count();
    drop(optimizer);
    let started = Instant::now();
    let mut reports = BTreeMap::new();
    for split in ["validation", "test"] {
        let dataset = Dataset::open(Path::new(&plan.data_path), split)?;
        let mut cursor = 0;
        let mut targets = 0_u64;
        let mut bytes = 0_u64;
        let mut nll = 0_f64;
        let mut windows = 0;
        // Bounded reporting window; never present this as a full-corpus score.
        for _ in 0..8 {
            if started.elapsed().as_secs() >= plan.max_seconds {
                break;
            }
            if cursor + (plan.model.context as u64) + 1 > dataset.len() {
                break;
            }
            let batch = dataset.batch(
                cursor,
                1,
                plan.model.context,
                &device,
                plan.variant == "semantic",
            )?;
            let logits = model.forward_detached(&batch.inputs, &batch.semantic)?;
            let loss = f64::from(model::loss(&logits, &batch.targets)?.to_scalar::<f32>()?);
            if !loss.is_finite() {
                return Err("nonfinite evaluation loss".into());
            }
            nll += loss * plan.model.context as f64;
            targets += plan.model.context as u64;
            bytes += batch.target_bytes;
            windows += 1;
            cursor = batch.next_cursor;
            if cursor == 0 {
                break;
            }
        }
        if targets == 0 || bytes == 0 {
            return Err("evaluation split is too short for selected context".into());
        }
        reports.insert(split,json!({"sampled_windows":windows,"target_tokens":targets,"target_raw_bytes":bytes,"nll_per_token":nll/targets as f64,"bits_per_raw_byte":nll/std::f64::consts::LN_2/bytes as f64,"full_corpus":false}));
    }
    let result = json!({"version":1,"config_identity":plan.identity()?,"step":step,"progress":progress,"hardware":gpu,"reports":reports,"same_split_required_for_comparison":true,"generation_or_generalization_improvement_unproven":true});
    write_json(&output.join("evaluation.json"), &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hardware_parser_accepts_only_the_two_l4_device_names() {
        for name in ["NVIDIA L4", "NVIDIA L4 GPU"] {
            let gpu = parse_hardware(&format!("{name}, GPU-test, 23034, 21000\n")).unwrap();
            assert_eq!(gpu.name, name);
            assert_eq!(gpu.total_mib, 23034);
            assert_eq!(gpu.free_mib, 21000);
        }
        for text in [
            "NVIDIA L40, GPU-test, 46068, 42000\n",
            "NVIDIA L40S, GPU-test, 46068, 42000\n",
            "Not NVIDIA L4, GPU-test, 23034, 21000\n",
            "NVIDIA L4, GPU-test, 23034, 21000\nNVIDIA L4, GPU-other, 23034, 21000\n",
            "NVIDIA L4, GPU-test, 23034\n",
            "NVIDIA L4, GPU-test, NaN, 21000\n",
            "NVIDIA L4, GPU-test, 23034, -1\n",
            "",
        ] {
            assert!(parse_hardware(text).is_err(), "accepted {text:?}");
        }
    }

    #[test]
    fn resume_metrics_separates_valid_unterminated_last_row_from_boundary() {
        let dir = std::env::temp_dir().join(format!(
            "scale-metrics-no-newline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");
        let original = b"{\"stage\":\"train\",\"step\":1}\n{\"stage\":\"train\",\"step\":2}";
        fs::write(&path, original).unwrap();
        reconcile_metrics(&path, 2, "test").unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["step"], 1);
        assert_eq!(rows[1]["step"], 2);
        assert_eq!(rows[2]["stage"], "resume_boundary");
        let archive = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|file| file != &path)
            .unwrap();
        assert_eq!(fs::read(archive).unwrap(), original);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resume_metrics_archive_abandoned_tail_and_keep_canonical_steps() {
        let dir = std::env::temp_dir().join(format!(
            "scale-metrics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");
        fs::write(&path,b"{\"stage\":\"train\",\"step\":1}\n{\"stage\":\"train\",\"step\":2}\n{\"stage\":\"train\",\"step\":3}\n{\"partial\":").unwrap();
        reconcile_metrics(&path, 2, "test").unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains("\"step\":3"));
        assert!(text.contains("resume_boundary"));
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resume_metrics_rejects_incomplete_or_invalid_training_history_without_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "scale-metrics-history-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for (case, (checkpoint_step, rows)) in [
            (2, vec![json!({"stage":"train", "step":1})]),
            (2, vec![json!({"stage":"train", "step":2})]),
            (
                3,
                vec![
                    json!({"stage":"train", "step":1}),
                    json!({"stage":"train", "step":3}),
                ],
            ),
            (
                2,
                vec![
                    json!({"stage":"train", "step":1}),
                    json!({"stage":"train", "step":1}),
                    json!({"stage":"train", "step":2}),
                ],
            ),
            (
                2,
                vec![
                    json!({"stage":"train", "step":2}),
                    json!({"stage":"train", "step":1}),
                ],
            ),
            (2, vec![json!({"stage":"train", "step":"1"})]),
            (2, vec![json!({"stage":"train", "step":-1})]),
            (2, vec![json!({"stage":"train", "step":1.5})]),
            (2, vec![json!({"stage":"train", "step":null})]),
            (2, vec![json!({"stage":"train"})]),
            (0, vec![json!({"stage":"train", "step":0})]),
            (2, vec![]),
        ]
        .into_iter()
        .enumerate()
        {
            let case_dir = dir.join(case.to_string());
            fs::create_dir_all(&case_dir).unwrap();
            let path = case_dir.join("metrics.jsonl");
            let original = rows
                .iter()
                .map(|row| format!("{row}\n"))
                .collect::<String>();
            fs::write(&path, &original).unwrap();
            assert!(
                reconcile_metrics(&path, checkpoint_step, "test").is_err(),
                "case {case} accepted invalid history {original:?}"
            );
            assert_eq!(fs::read_to_string(&path).unwrap(), original);
            let archive = fs::read_dir(&case_dir)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .find(|file| {
                    file.file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .starts_with("metrics-before-resume-")
                })
                .unwrap();
            assert_eq!(fs::read_to_string(archive).unwrap(), original);
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resume_metrics_allows_other_stages_and_filters_abandoned_train_tail_at_zero() {
        let dir = std::env::temp_dir().join(format!(
            "scale-metrics-zero-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");
        let other_row = json!({"stage":"custom", "step":"external-value"});
        fs::write(
            &path,
            format!(
                "{}\n{other_row}\n{}\n{}\n",
                json!({"stage":"start", "step":0}),
                json!({"stage":"train", "step":1}),
                json!({"stage":"train", "step":2}),
            ),
        )
        .unwrap();
        reconcile_metrics(&path, 0, "test").unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let rows: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1], other_row);
        assert_eq!(rows[2]["stage"], "resume_boundary");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_cli_options_fail_without_allocating_model() {
        for args in [
            vec!["plan", "--bad", "1"],
            vec!["prepare", "--input"],
            vec!["train", "--cpu", "--cpu"],
            vec!["unknown"],
        ] {
            assert!(cli(args.into_iter().map(String::from).collect()).is_err());
        }
    }
    #[test]
    fn concurrent_run_lock_is_released_on_handle_drop() {
        let dir = std::env::temp_dir().join(format!(
            "scale-lock-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first = lock_run(&dir).unwrap();
        assert!(lock_run(&dir).is_err());
        drop(first);
        assert!(lock_run(&dir).is_ok());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hashes_bind_ordered_bytes() {
        assert_ne!(crate::config::sha256(b"a"), crate::config::sha256(b"b"));
    }
}
