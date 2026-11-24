use candle_core::Device;
use packtok_m5::{
    Result,
    data::Sequence,
    extended::BatchData,
    hash,
    model::{Rng, Transformer},
    resume::{
        AdamWConfig, Identity, ResumableAdamW,
        v2::{Position, Timing, TrainingSpec, latest},
    },
    runner::tiny_config,
};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn directory() -> Result<PathBuf> {
    let dir = env::temp_dir().join(format!(
        "packtok-m6-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&dir)?;
    Ok(dir)
}

fn fixture(regime: &str) -> (Sequence, TrainingSpec) {
    let seq = Sequence {
        tokens: (0..53).map(|i| (i * 7 + 3) % 127).collect(),
        bytes: (0..53).map(|i| (i % 3) + 1).collect(),
    };
    let seq_digest = hash(&serde_json::to_vec(&(&seq.tokens, &seq.bytes)).unwrap());
    let spec = TrainingSpec {
        tokenizer_artifact_sha256: hash(b"fixture-token-ids"),
        tokenizer_mapping_sha256: hash(b"identity-mapping"),
        identity: Identity {
            config_sha256: hash(b"m6-process-test-v1"),
            corpus_sha256: seq_digest.clone(),
            tokenizer_sha256: hash(b"integer-fixture-v1"),
            train_sequence_sha256: seq_digest,
            variant: "fixture".into(),
            regime: regime.into(),
            seed: 19,
        },
        model: tiny_config(),
        schedule: AdamWConfig {
            total_steps: 4,
            warmup_steps: 1,
            ..Default::default()
        },
        batch_size: 2,
        train_tokens: seq.tokens.len(),
        first_target: if regime == "T" { 0 } else { 1 },
        target_limit: seq.tokens.len() as u64,
        backend: "cpu".into(),
    };
    (seq, spec)
}
fn update(
    model: &Transformer,
    opt: &mut ResumableAdamW,
    spec: &TrainingSpec,
    seq: &Sequence,
    pos: &mut Position,
    device: &Device,
) -> Result<String> {
    let mut rng = Rng::from_state(pos.rng_state)?;
    let batch = spec
        .next_batch(seq, &mut rng, &mut pos.cursor, device)?
        .ok_or("fixture exhausted")?;
    let samples = match &batch {
        BatchData::Full { x, y, .. } => (x.to_vec2::<u32>()?, y.to_vec2::<u32>()?),
        BatchData::Variable { x, targets, .. } => (x.to_vec2::<u32>()?, targets.clone()),
    };
    let loss = batch.loss(model)?;
    let loss_bits = loss.to_scalar::<f32>()?.to_bits();
    opt.apply(model, &loss.backward()?)?;
    pos.step = opt.step_count();
    pos.rng_state = rng.state();
    let lr_bits = opt.config().learning_rate_at(pos.step).to_bits();
    Ok(format!(
        "TRACE {} {} {} {loss_bits} {lr_bits} {samples:?}",
        pos.step, pos.cursor, pos.rng_state
    ))
}

// Invoked by the parent in a new address space for every segment.
#[test]
fn training_worker() -> Result<()> {
    let Some(dir) = env::var_os("PACKTOK_M6_FIXTURE_DIR") else {
        return Ok(());
    };
    let dir = Path::new(&dir);
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(60));
        std::process::exit(124);
    });
    let stop: usize = env::var("PACKTOK_M6_STOP")?.parse()?;
    let (seq, mut spec) = fixture(&env::var("PACKTOK_M6_REGIME")?);
    let device = match env::var("PACKTOK_M6_PROBE_DEVICE").as_deref() {
        Ok("L4") => {
            let gpu = packtok_m5::runner::require_l4()?;
            println!("GPU {gpu}");
            spec.backend = "cuda".into();
            packtok_m5::runner::cuda()?
        }
        Err(env::VarError::NotPresent) => Device::Cpu,
        _ => return Err("unsupported probe device".into()),
    };
    let image = latest(dir)?;
    let (model, mut opt, mut pos) = if let Some((step, path)) = image {
        let start = Instant::now();
        let restored = ResumableAdamW::restore(&path, &spec, &device, step)?;
        device.synchronize()?;
        println!("RESTORE_US {}", start.elapsed().as_micros());
        if let Some(audit) = env::var_os("PACKTOK_M6_RESTORE_AUDIT") {
            let copy = restored
                .1
                .publish(&restored.0, &spec, restored.2, Path::new(&audit))?;
            assert_eq!(
                fs::read(copy)?,
                fs::read(path)?,
                "immediate restore/re-save"
            );
            println!("RESTORE_BYTES_EXACT");
        }
        restored
    } else {
        let model = Transformer::new(spec.model.clone(), spec.identity.seed, &device)?;
        let opt = ResumableAdamW::new(&model, spec.schedule.clone())?;
        (
            model,
            opt,
            Position {
                step: 0,
                cursor: spec.first_target,
                rng_state: Rng::new(spec.identity.seed ^ 0xa341316c9e3779b9).state(),
                timing: Timing::default(),
                journal: None,
            },
        )
    };
    while pos.step < stop {
        println!(
            "{}",
            update(&model, &mut opt, &spec, &seq, &mut pos, &device)?
        );
    }
    device.synchronize()?;
    let start = Instant::now();
    let published = opt.publish(&model, &spec, pos, dir)?;
    device.synchronize()?;
    println!(
        "PUBLISH_US {} BYTES {}",
        start.elapsed().as_micros(),
        published.metadata()?.len()
    );
    if stop < 4 {
        // Lose a real, uncommitted update when the parent kills this process.
        update(&model, &mut opt, &spec, &seq, &mut pos, &device)?;
        device.synchronize()?;
        println!("KILL_HERE");
        std::io::stdout().flush()?;
        std::thread::park();
    }
    Ok(())
}

fn segment(dir: &Path, regime: &str, stop: usize) -> Result<Vec<String>> {
    let mut child = Command::new(env::current_exe()?)
        .args(["--exact", "training_worker", "--nocapture"])
        .env("PACKTOK_M6_FIXTURE_DIR", dir)
        .env("PACKTOK_M6_STOP", stop.to_string())
        .env("PACKTOK_M6_REGIME", regime)
        .env("RAYON_NUM_THREADS", "1")
        .stdout(Stdio::piped())
        .spawn()?;
    let mut observed = Vec::new();
    let mut killed = false;
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let line = line?;
        if let Some((_, row)) = line.split_once("TRACE ") {
            observed.push(row.to_owned());
        }
        if line.contains("KILL_HERE") {
            child.kill()?;
            killed = true;
            break;
        }
    }
    let status = child.wait()?;
    assert_eq!(killed, stop < 4);
    assert_eq!(status.success(), !killed);
    Ok(observed)
}

#[test]
fn fresh_process_replays_each_update_and_all_checkpoint_bytes() -> Result<()> {
    for regime in ["T", "B"] {
        let full = directory()?;
        let recovered = directory()?;
        let expected = segment(&full, regime, 4)?;
        let mut actual = Vec::new();
        for stop in [1, 2, 4] {
            actual.extend(segment(&recovered, regime, stop)?);
        }
        assert_eq!(expected.len(), 4);
        assert_eq!(
            expected, actual,
            "sample windows, loss/LR bits, RNG, cursor, step"
        );
        let (step, a) = latest(&full)?.unwrap();
        let (other, b) = latest(&recovered)?.unwrap();
        assert_eq!((step, other), (4, 4));
        let final_sha = hash(&fs::read(a)?);
        assert_eq!(
            final_sha,
            hash(&fs::read(b)?),
            "entire header, all weights and both moments"
        );
        eprintln!(
            "regime={regime} updates=4 trace_sha256={} checkpoint_sha256={final_sha}",
            hash(expected.join("\n").as_bytes())
        );
        fs::remove_dir_all(full)?;
        fs::remove_dir_all(recovered)?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn interrupted_writer_leaves_only_previous_generation_visible() -> Result<()> {
    let dir = directory()?;
    segment(&dir, "T", 1)?;
    let (_, path) = latest(&dir)?.unwrap();
    let before = hash(&fs::read(&path)?);
    assert!(path.metadata()?.len() > 16 << 10);
    // Linux enforces RLIMIT_FSIZE during the write, after the first nonempty flush.
    let status = Command::new("sh")
        .args(["-c", "ulimit -c 0; ulimit -f 16; exec \"$@\"", "fixture"])
        .arg(env::current_exe()?)
        .args(["--exact", "training_worker", "--nocapture"])
        .env("PACKTOK_M6_FIXTURE_DIR", &dir)
        .env("PACKTOK_M6_STOP", "2")
        .env("PACKTOK_M6_REGIME", "T")
        .env("RAYON_NUM_THREADS", "1")
        .stdout(Stdio::null())
        .status()?;
    assert!(!status.success());
    assert_eq!(latest(&dir)?, Some((1, path.clone())));
    assert_eq!(hash(&fs::read(path)?), before);
    assert!(
        fs::read_dir(&dir)?.filter_map(|e| e.ok()).any(|e| e
            .file_name()
            .to_string_lossy()
            .starts_with(".pending-")
            && e.metadata().is_ok_and(|m| m.len() > 0))
    );
    assert_eq!(segment(&dir, "T", 4)?.len(), 3);
    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn rejects_corruption_mismatch_and_interrupted_publication() -> Result<()> {
    let dir = directory()?;
    let (_, spec) = fixture("T");
    let model = Transformer::new(spec.model.clone(), 19, &Device::Cpu)?;
    let opt = ResumableAdamW::new(&model, spec.schedule.clone())?;
    let p = Position {
        step: 0,
        cursor: 0,
        rng_state: Rng::new(19 ^ 0xa341316c9e3779b9).state(),
        timing: Timing::default(),
        journal: None,
    };
    let good = opt.publish(&model, &spec, p, &dir)?;
    let original = fs::read(&good)?;
    assert!(opt.publish(&model, &spec, p, &dir).is_err());
    assert_eq!(dir.read_dir()?.count(), 1);
    fs::write(
        dir.join(".pending-crashed"),
        &original[..original.len() / 2],
    )?;
    assert_eq!(latest(&dir)?, Some((0, good.clone())));
    let load = |path: &Path, s: &TrainingSpec, step| -> Result<()> {
        ResumableAdamW::restore(path, s, &Device::Cpu, step)?;
        Ok(())
    };
    let bad = dir.join("step-00000000000000000001.ptc");
    for n in [0, 7, original.len() / 2, original.len() - 1] {
        fs::write(&bad, &original[..n])?;
        assert!(load(&bad, &spec, 0).is_err());
    }
    for index in [15, original.len() / 2, original.len() - 1] {
        let mut bytes = original.clone();
        bytes[index] ^= 1;
        fs::write(&bad, bytes)?;
        assert!(load(&bad, &spec, 0).is_err());
    }
    // Re-seal so these exercise structural checks independently of the outer SHA.
    let seal = |body: &[u8]| -> Result<()> {
        let mut bytes = body.to_vec();
        bytes.extend_from_slice(&Sha256::digest(body));
        fs::write(&bad, bytes)?;
        Ok(())
    };
    let mut body = original[..original.len() - 32].to_vec();
    let payload = 12 + u32::from_le_bytes(body[8..12].try_into()?) as usize;
    for value in [f32::NAN, f32::INFINITY] {
        let mut invalid = body.clone();
        invalid[payload..payload + 4].copy_from_slice(&value.to_le_bytes());
        seal(&invalid)?;
        assert!(load(&bad, &spec, 0).is_err());
    }
    let mut invalid = body.clone();
    let last = invalid.len() - 4; // final second-moment element
    invalid[last..].copy_from_slice(&(-1.0_f32).to_le_bytes());
    seal(&invalid)?;
    assert!(load(&bad, &spec, 0).is_err());
    body[7] = b'9';
    seal(&body)?;
    assert!(load(&bad, &spec, 0).is_err());
    body[7] = b'2';
    body.push(0);
    seal(&body)?;
    assert!(load(&bad, &spec, 0).is_err());
    let (step, newest) = latest(&dir)?.unwrap();
    assert!(load(&newest, &spec, step).is_err());
    for change in 0..9 {
        let mut mismatch = spec.clone();
        match change {
            0 => mismatch.schedule.epsilon *= 2.0,
            1 => mismatch.identity.tokenizer_sha256 = hash(b"wrong"),
            2 => mismatch.identity.train_sequence_sha256 = hash(b"wrong"),
            3 => mismatch.identity.corpus_sha256 = hash(b"wrong"),
            4 => mismatch.model.heads = 1,
            5 => mismatch.batch_size += 1,
            6 => mismatch.tokenizer_artifact_sha256 = hash(b"wrong"),
            7 => mismatch.tokenizer_mapping_sha256 = hash(b"wrong"),
            _ => mismatch.identity.config_sha256 = hash(b"wrong"),
        }
        assert!(load(&good, &mismatch, 0).is_err());
    }
    load(&good, &spec, 0)?;
    assert_eq!(fs::read(&good)?, original);
    let mut failed = opt;
    let unrelated = candle_core::Var::new(1.0_f32, &Device::Cpu)?;
    let gradients = unrelated.sqr()?.backward()?;
    assert!(failed.apply(&model, &gradients).is_err());
    let empty = directory()?;
    assert!(failed.publish(&model, &spec, p, &empty).is_err());
    assert!(latest(&empty)?.is_none());
    fs::remove_dir_all(empty)?;
    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
#[ignore = "requires the locally archived M5 pilot result directory"]
fn historical_pilot_images_restore_on_cpu() -> Result<()> {
    let root = PathBuf::from(env::var("PACKTOK_M5_PILOT_RESULTS")?);
    let config = include_bytes!("../configs/pilot-2k-v1.json");
    let plan: serde_json::Value = serde_json::from_slice(config)?;
    let inputs = &plan["scientific_inputs"];
    for (variant, digest) in [
        (
            "A",
            "da3b08bfda2bc991992a4487f4659c7b406b2af5f2abe62789bef8d80e50655a",
        ),
        (
            "C",
            "e7b45ad2bcaf52434aac6d35db06edab0ba92e86d27ca8c46930a2c1c43fc484",
        ),
    ] {
        let path = root.join(variant).join("latest.resume.safetensors");
        let mut sha = Sha256::new();
        std::io::copy(&mut fs::File::open(&path)?, &mut sha)?;
        assert_eq!(format!("{:x}", sha.finalize()), digest);
        let identity = Identity {
            config_sha256: hash(config),
            corpus_sha256: inputs["corpus_manifest_sha256"].as_str().unwrap().into(),
            tokenizer_sha256: inputs["tokenizer_sha256"][variant].as_str().unwrap().into(),
            train_sequence_sha256: inputs["prepared_sequence_sha256"]
                [format!("{variant}_train_sequence_sha256")]
            .as_str()
            .unwrap()
            .into(),
            variant: variant.into(),
            regime: "T".into(),
            seed: 20261008,
        };
        let mut model = Transformer::new(Default::default(), 19, &Device::Cpu)?;
        let (opt, step, cursor, rng) =
            ResumableAdamW::load_checkpoint(&mut model, &identity, &path)?;
        assert_eq!((opt.step_count(), step, cursor), (2000, 2000, 4096000));
        let mut expected = Rng::new(identity.seed ^ 0xa341316c9e3779b9);
        for _ in 0..2000 * 8 {
            expected.next_u64();
        }
        assert_eq!(rng, expected.state());
    }
    Ok(())
}
