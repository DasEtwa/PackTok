use crate::{
    corpus,
    source::{self, Split},
    tokens::{self, CONTEXT, Reader},
};
use candle_core::{Device, Tensor};
use packtok_m5::{
    Result,
    data::Adapter,
    hash,
    model::{Rng, Transformer, finite_gradients, loss},
    resume::{
        AdamWConfig, Identity, ResumableAdamW,
        v2::{self, JournalPrefix, Position, Timing, TrainingSpec},
    },
    runner,
};
use serde_json::json;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const STEPS: usize = 96;
const INTERVAL: usize = 32;
const SEED: u64 = 19;

fn probe(dataset: &Path, split: Split, adapter: &Adapter) -> Result<Vec<tokens::Frame>> {
    let data: corpus::Manifest = corpus::json(&dataset.join(corpus::MANIFEST))?;
    let mut result = Vec::new();
    let mut docs = 0;
    for shard in data.shards.iter().filter(|s| s.split == split) {
        corpus::walk(dataset, shard, |doc| {
            if docs < 8 {
                let mut n = doc.text.len().min(512);
                while !doc.text.is_char_boundary(n) {
                    n -= 1;
                }
                let seq = adapter.encode(&doc.text.as_bytes()[..n])?;
                result.push(tokens::Frame {
                    ids: seq.tokens,
                    lengths: seq.bytes,
                    chunk_start: true,
                });
                docs += 1;
            }
            Ok(())
        })?;
        if docs >= 8 {
            break;
        }
    }
    Ok(result)
}

fn score(
    model: &Transformer,
    docs: &[tokens::Frame],
    device: &Device,
) -> Result<serde_json::Value> {
    let mut nll = 0.0;
    let mut count = 0;
    let mut bytes = 0;
    for doc in docs {
        for start in (0..doc.ids.len().saturating_sub(1)).step_by(CONTEXT) {
            let n = CONTEXT.min(doc.ids.len() - start - 1);
            let x = Tensor::from_slice(&doc.ids[start..start + n], (1, n), device)?;
            let y = Tensor::from_slice(&doc.ids[start + 1..start + 1 + n], (1, n), device)?;
            nll += f64::from(loss(&model.forward(&x)?, &y)?.to_scalar::<f32>()?) * n as f64;
            bytes += doc.lengths[start + 1..start + 1 + n]
                .iter()
                .map(|&b| u64::from(b))
                .sum::<u64>();
            count += n;
        }
    }
    if !nll.is_finite() || count == 0 || bytes == 0 {
        return Err("invalid probe score".into());
    }
    Ok(
        json!({"nll":nll,"tokens":count,"bytes":bytes,"bits_per_byte":nll/bytes as f64/std::f64::consts::LN_2}),
    )
}

fn generate(model: &Transformer, adapter: &Adapter, device: &Device) -> Result<(String, String)> {
    let mut ids = adapter.encode(b" ")?.tokens;
    let initial = ids.len();
    for _ in 0..24 {
        let x = &ids[ids.len().saturating_sub(CONTEXT)..];
        let logits = model
            .forward(&Tensor::from_slice(x, (1, x.len()), device)?)?
            .to_vec3::<f32>()?;
        let row = &logits[0][x.len() - 1][..adapter.mapping.len()];
        if row.iter().any(|a| !a.is_finite()) {
            return Err("nonfinite generation logits".into());
        }
        let (id, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("empty vocabulary")?;
        ids.push(id as u32);
    }
    let bytes = adapter.decode_global(&ids[initial..])?;
    Ok((
        String::from_utf8_lossy(&bytes).into(),
        bytes.iter().map(|b| format!("{b:02x}")).collect(),
    ))
}

pub fn run(
    dataset: &Path,
    root: &Path,
    out: &Path,
    key: &str,
    stop: usize,
    device: &Device,
) -> Result<()> {
    if !(1..=STEPS).contains(&stop) {
        return Err("the frozen small probe permits 1..96 completed steps".into());
    }
    let (m, adapter) = tokens::load(root, key)?;
    if m.dataset_sha256 != source::digest(&dataset.join(corpus::MANIFEST))?.1 {
        return Err("dataset identity mismatch".into());
    }
    corpus::verify(dataset)?;
    let meta = &m.variants[key];
    let stream = meta.streams.get("train").ok_or("missing train metadata")?;
    if stream.full_frames == 0 || stream.full_frames > stream.frames {
        return Err("training requires full windows".into());
    }
    let mut input = Reader::open(root, key, Split::Train, meta)?;
    let backend = if device.is_cpu() { "cpu" } else { "cuda" };
    let spec = TrainingSpec {
        identity: Identity {
            config_sha256: hash(
                b"m7-fixture-train-v2/window02-t/lr008/96/seed19/probe-prefix512x8",
            ),
            corpus_sha256: m.dataset_sha256.clone(),
            train_sequence_sha256: stream.sha256.clone(),
            tokenizer_sha256: meta.artifact_and_mapping_sha256.clone(),
            variant: key.into(),
            regime: "T".into(),
            seed: SEED,
        },
        tokenizer_artifact_sha256: meta.artifact_sha256.clone(),
        tokenizer_mapping_sha256: meta.mapping_sha256.clone(),
        model: runner::tiny_config(),
        schedule: AdamWConfig {
            learning_rate: 0.008,
            total_steps: STEPS,
            warmup_steps: 4,
            ..Default::default()
        },
        batch_size: 1,
        train_tokens: usize::try_from(stream.scored_tokens)?,
        first_target: 0,
        target_limit: stream.scored_tokens,
        backend: backend.into(),
    };
    fs::create_dir_all(out)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(out.join(".training-writer"))?;
    lock.try_lock()
        .map_err(|e| format!("M7 training lock: {e}"))?;
    let ck = out.join("training-v2");
    fs::create_dir_all(&ck)?;
    File::open(out)?.sync_all()?;
    let log = out.join("metrics.jsonl");
    let initial = Transformer::new(spec.model.clone(), SEED, device)?;
    let train_probe = probe(dataset, Split::Train, &adapter)?;
    let val_probe = probe(dataset, Split::Validation, &adapter)?;
    let before_train = score(&initial, &train_probe, device)?;
    let before_val = score(&initial, &val_probe, device)?;
    let (model, mut opt, mut pos) = if let Some((step, path)) = v2::latest(&ck)? {
        if stop < step {
            return Err("requested step precedes committed checkpoint".into());
        }
        let restored = ResumableAdamW::restore(&path, &spec, device, step)?;
        let prefix = restored.2.journal;
        if step != 0 && prefix.is_none() {
            return Err("missing journal identity".into());
        }
        if log.try_exists()? || prefix.is_some() {
            let mut f = OpenOptions::new().read(true).write(true).open(&log)?;
            let bytes = prefix.map_or(0, |p| p.bytes);
            if let Some(prefix) = prefix
                && JournalPrefix::read(&mut f, bytes)? != prefix
            {
                return Err("training journal mismatch".into());
            }
            if f.metadata()?.len() > bytes {
                let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
                let archive = out.join(format!("metrics-before-recovery-{nonce}.jsonl"));
                let mut to = File::create_new(archive)?;
                io::copy(&mut File::open(&log)?, &mut to)?;
                to.sync_all()?;
                File::open(out)?.sync_all()?;
                f.set_len(bytes)?;
                f.sync_all()?;
            }
        }
        restored
    } else {
        let opt = ResumableAdamW::new(&initial, spec.schedule.clone())?;
        let pos = Position {
            step: 0,
            cursor: 0,
            rng_state: Rng::new(SEED ^ 0xa341316c9e3779b9).state(),
            timing: Timing::default(),
            journal: None,
        };
        if log.try_exists()? {
            return Err("journal exists without a published checkpoint".into());
        }
        opt.publish(&initial, &spec, pos, &ck)?;
        (initial, opt, pos)
    };
    let mut file = OpenOptions::new()
        .create(pos.step == 0)
        .append(true)
        .open(&log)?;
    File::open(out)?.sync_all()?;
    let initial_step = pos.step;
    let mut rng = Rng::from_state(pos.rng_state)?;
    let started = Instant::now();
    while pos.step < stop {
        let index = rng.next_u64() % input.frames;
        // Sequential fallback across short tails keeps one RNG draw per update and bounded reads.
        let mut next = index;
        let frame = loop {
            let frame = input.read(next)?;
            if frame.ids.len() == CONTEXT + 1 {
                break frame;
            }
            next = (next + 1) % input.frames;
            if next == index {
                return Err("no full training windows".into());
            }
        };
        let x = Tensor::from_slice(&frame.ids[..CONTEXT], (1, CONTEXT), device)?;
        let y = Tensor::from_slice(&frame.ids[1..], (1, CONTEXT), device)?;
        let value = loss(&model.forward(&x)?, &y)?;
        let scalar = value.to_scalar::<f32>()?;
        if !scalar.is_finite() {
            return Err("nonfinite training loss".into());
        }
        let grads = value.backward()?;
        finite_gradients(&model, &grads)?;
        opt.apply(&model, &grads)?;
        device.synchronize()?;
        pos.step = opt.step_count();
        pos.cursor += CONTEXT as u64;
        pos.rng_state = rng.state();
        let row = json!({"step":pos.step,"window":next,"rng":pos.rng_state,"lr":opt.config().learning_rate_at(pos.step),
            "loss":scalar,"loss_bits":scalar.to_bits(),"target_bytes":frame.lengths[1..].iter().copied().sum::<u32>()});
        serde_json::to_writer(&mut file, &row)?;
        file.write_all(b"\n")?;
        if pos.step.is_multiple_of(INTERVAL) || pos.step == stop {
            file.sync_all()?;
            let mut f = File::open(&log)?;
            pos.journal = Some(JournalPrefix::read(&mut f, file.metadata()?.len())?);
            opt.publish(&model, &spec, pos, &ck)?;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let after_train = score(&model, &train_probe, device)?;
    let after_val = score(&model, &val_probe, device)?;
    let (generated, generated_hex) = generate(&model, &adapter, device)?;
    let report = json!({"variant":key,"backend":backend,"step":pos.step,"parameters":spec.model.parameter_count()?,
        "dataset_sha256":m.dataset_sha256,"train_sequence_sha256":stream.sha256,"tokenizer_sha256":meta.artifact_sha256,
        "train_initial":before_train,"train_final":after_train,"validation_initial":before_val,"validation_final":after_val,
        "generated":generated,"generated_hex":generated_hex});
    packtok_m5::write_new(
        &out.join(format!("report-{:06}.json", pos.step)),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        json!({"report":report,"segment_seconds":seconds,
        "trained_targets":(stop-initial_step)*CONTEXT,"targets_per_second":(stop-initial_step) as f64*CONTEXT as f64/seconds})
    );
    Ok(())
}
