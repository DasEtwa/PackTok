use crate::{
    Result,
    data::Sequence,
    domains, hash,
    model::{ModelConfig, Rng, Transformer, finite_gradients, loss},
    resume::v2::{self, JournalPrefix, Position, Timing as MetricsProgress, TrainingSpec},
    resume::{AdamWConfig, Identity, ResumableAdamW},
    runner,
};
use candle_core::{Device, Tensor};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Seek, Write},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Deserialize)]
pub struct ExtendedPlan {
    pub revision: String,
    pub status: String,
    pub architecture: ModelConfig,
    pub parameters: usize,
    pub batch_size: usize,
    pub context: usize,
    pub target_positions_per_update: usize,
    pub optimizer: Value,
    pub schedule: Value,
    #[serde(rename = "B")]
    pub byte_regime: Option<Value>,
    pub variants: Vec<String>,
    pub regimes: Vec<String>,
    pub updates_per_model: usize,
    pub paired_seeds: Vec<u64>,
    pub scientific_inputs: Value,
    pub evaluation: Value,
}
impl ExtendedPlan {
    pub fn load(path: &Path, data: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let value: Value = serde_json::from_slice(&bytes)?;
        let plan: Self = serde_json::from_value(value)?;
        let parent = path.parent().ok_or("config parent")?;
        let (expected_name, expected_hash) = match plan.revision.as_str() {
            "m5-paired-pilot-2k-v1" => (
                "pilot-2k-v1.json",
                "6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8",
            ),
            "m5-extended-20k-v1" => (
                "extended-20k-v1.json",
                "2943cb456a82d76dc1c0f8371bd946897e151dd344b03de239cf98bb3534f621",
            ),
            "m5-extended-30k-v1" => (
                "extended-30k-extension-v1.json",
                "a121011c57269b82361403fd180fc4300470e8839b69303224926b4eb6bd86f0",
            ),
            _ => return Err("unknown frozen extended revision".into()),
        };
        if path.canonicalize()? != parent.join(expected_name).canonicalize()?
            || hash(&bytes) != expected_hash
        {
            return Err("extended config path or frozen bytes do not match revision".into());
        }
        if !plan.revision.starts_with("m5-")
            || !(plan.status.contains("approval") || plan.status.contains("approved"))
            || plan.parameters != 14_681_984
            || plan.architecture != ModelConfig::default()
            || plan.architecture.parameter_count()? != plan.parameters
            || plan.batch_size != 8
            || plan.context != 256
            || plan.architecture.context != 256
            || plan.target_positions_per_update != 2048
            || plan.variants != ["A", "C"]
            || plan.paired_seeds.is_empty()
            || plan.paired_seeds.iter().collect::<BTreeSet<_>>().len() != plan.paired_seeds.len()
            || plan.regimes.is_empty()
            || plan
                .regimes
                .iter()
                .any(|r| !["T", "B"].contains(&r.as_str()))
        {
            return Err("extended plan does not preserve the frozen M5 inputs".into());
        }
        let manifest = parent
            .parent()
            .ok_or("experiment root")?
            .join("provenance/corpus-v2-manifest.json");
        let primary = parent.join("primary.json");
        if hash(&fs::read(manifest)?)
            != plan.scientific_inputs["corpus_manifest_sha256"]
                .as_str()
                .ok_or("manifest identity")?
            || hash(&fs::read(primary)?)
                != plan.scientific_inputs["primary_config_sha256"]
                    .as_str()
                    .ok_or("primary config identity")?
        {
            return Err("extended plan corpus/primary config hash mismatch".into());
        }
        let experiment_root = parent.parent().ok_or("experiment root")?;
        for variant in ["A", "C"] {
            let mut identity =
                fs::read(experiment_root.join(format!("artifacts/corpus-v2/{variant}-0.packtok")))?;
            identity.extend(fs::read(
                experiment_root.join(format!("artifacts/corpus-v2/{variant}.mapping")),
            )?);
            if hash(&identity)
                != plan.scientific_inputs["tokenizer_sha256"][variant]
                    .as_str()
                    .ok_or("tokenizer identity")?
            {
                return Err("extended plan tokenizer artifact/mapping hash mismatch".into());
            }
        }
        for variant in ["A", "C"] {
            for split in ["train", "validation", "test"] {
                let path = data.join(format!("{variant}-{split}.seq"));
                let key = format!("{variant}_{split}_sequence_sha256");
                if sha_file(&path)?
                    != plan.scientific_inputs["prepared_sequence_sha256"][key]
                        .as_str()
                        .ok_or("prepared sequence identity")?
                {
                    return Err("extended plan prepared sequence hash mismatch".into());
                }
            }
        }
        if plan.optimizer["learning_rate"].as_f64() != Some(0.0003)
            || plan.optimizer["beta1"].as_f64() != Some(0.9)
            || plan.optimizer["beta2"].as_f64() != Some(0.999)
            || plan.optimizer["epsilon"].as_f64() != Some(1e-8)
            || plan.optimizer["weight_decay"].as_f64() != Some(0.01)
        {
            return Err(
                "extended optimizer does not match frozen M5/declared extended values".into(),
            );
        }
        if plan.schedule["total_updates"].as_u64() != Some(plan.updates_per_model as u64) {
            return Err("declared update count and scheduler total differ".into());
        }
        match plan.revision.as_str() {
            "m5-paired-pilot-2k-v1"
                if plan.updates_per_model == 2000
                    && plan.regimes == ["T"]
                    && plan.paired_seeds == [20261008]
                    && plan.schedule["warmup_updates"].as_u64() == Some(0)
                    && plan.schedule["decay"].as_str() == Some("constant") => {}
            "m5-extended-20k-v1"
                if plan.updates_per_model == 20000
                    && plan.regimes == ["T", "B"]
                    && plan.paired_seeds == [20261008, 20261009, 20261010, 20261011, 20261012]
                    && plan.schedule["warmup_updates"].as_u64() == Some(100)
                    && plan.schedule["decay"].as_str() == Some("cosine") => {}
            "m5-extended-30k-v1"
                if plan.updates_per_model == 30000
                    && plan.regimes == ["T", "B"]
                    && plan.paired_seeds == [20261008, 20261009, 20261010, 20261011, 20261012]
                    && plan.schedule["warmup_updates"].as_u64() == Some(100)
                    && plan.schedule["decay"].as_str() == Some("cosine") => {}
            _ => return Err("extended plan differs from its frozen revision".into()),
        }
        Ok(plan)
    }
    fn adamw(&self) -> Result<AdamWConfig> {
        let config = AdamWConfig {
            learning_rate: self.optimizer["learning_rate"]
                .as_f64()
                .ok_or("learning rate")?,
            beta1: self.optimizer["beta1"].as_f64().ok_or("beta1")?,
            beta2: self.optimizer["beta2"].as_f64().ok_or("beta2")?,
            epsilon: self.optimizer["epsilon"].as_f64().ok_or("epsilon")?,
            weight_decay: self.optimizer["weight_decay"]
                .as_f64()
                .ok_or("weight decay")?,
            total_steps: self.schedule["total_updates"]
                .as_u64()
                .ok_or("schedule updates")? as usize,
            warmup_steps: self.schedule["warmup_updates"]
                .as_u64()
                .ok_or("warmup updates")? as usize,
            decay: self.schedule["decay"]
                .as_str()
                .ok_or("decay schedule")?
                .to_owned(),
        };
        config.validate()?;
        Ok(config)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ByteMatch {
    pub raw_byte_start: u64,
    pub raw_target_bytes: u64,
    pub raw_byte_end: u64,
    pub a_start_token: usize,
    pub a_end_token: usize,
    pub c_start_token: usize,
    pub c_end_token: usize,
}
fn byte_boundaries(seq: &Sequence) -> BTreeMap<u64, usize> {
    let mut boundaries = BTreeMap::new();
    let mut offset = 0_u64;
    boundaries.insert(0, 0);
    for (index, &length) in seq.bytes.iter().enumerate() {
        offset += u64::from(length);
        boundaries.insert(offset, index + 1);
    }
    boundaries
}
pub fn matched_byte_interval(a: &Sequence, c: &Sequence, limit: u64) -> Result<ByteMatch> {
    let ap = byte_boundaries(a);
    let cp = byte_boundaries(c);
    let shared: Vec<_> = ap
        .iter()
        .filter_map(|(&offset, &ai)| cp.get(&offset).map(|&ci| (offset, ai, ci)))
        .collect();
    let mut left = 1_usize;
    let mut best: Option<(u64, u64, usize, usize, usize, usize)> = None;
    for right in 1..shared.len() {
        while left < right && shared[right].0 - shared[left].0 > limit {
            left += 1;
        }
        if left >= right {
            continue;
        }
        let (start, a_start, c_start) = shared[left];
        let (end, a_end, c_end) = shared[right];
        let bytes = end - start;
        if bytes > 0 && a_start > 0 && c_start > 0 && best.is_none_or(|old| bytes > old.0) {
            best = Some((bytes, start, a_start, a_end, c_start, c_end));
        }
    }
    let (raw_target_bytes, raw_byte_start, a_start_token, a_end_token, c_start_token, c_end_token) =
        best.ok_or("A/C have no positive shared raw-byte interval within the B budget")?;
    let raw_byte_end = raw_byte_start + raw_target_bytes;
    Ok(ByteMatch {
        raw_byte_start,
        raw_target_bytes,
        raw_byte_end,
        a_start_token,
        a_end_token,
        c_start_token,
        c_end_token,
    })
}
fn sha_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    std::io::copy(&mut fs::File::open(path)?, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}
fn nvidia_snapshot() -> Result<(String, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-gpu=name,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !out.status.success() {
        return Err("nvidia-smi failed".into());
    }
    let text = String::from_utf8(out.stdout)?;
    let mut fields = text.trim().split(',').map(str::trim);
    let name = fields.next().ok_or("GPU name")?.to_owned();
    let memory = fields.next().ok_or("GPU memory")?.parse()?;
    Ok((name, memory))
}

pub enum BatchData {
    Full {
        x: Tensor,
        y: Tensor,
        positions: u64,
        bytes: u64,
    },
    Variable {
        x: Tensor,
        targets: Vec<Vec<u32>>,
        lengths: Vec<usize>,
        positions: u64,
        bytes: u64,
    },
}
impl BatchData {
    fn positions(&self) -> u64 {
        match self {
            Self::Full { positions, .. } | Self::Variable { positions, .. } => *positions,
        }
    }
    fn bytes(&self) -> u64 {
        match self {
            Self::Full { bytes, .. } | Self::Variable { bytes, .. } => *bytes,
        }
    }
    pub fn loss(&self, model: &Transformer) -> Result<Tensor> {
        match self {
            Self::Full { x, y, .. } => Ok(loss(&model.forward(x)?, y)?),
            Self::Variable {
                x,
                targets,
                lengths,
                positions,
                ..
            } => {
                let logits = model.forward(x)?;
                let mut weighted: Option<Tensor> = None;
                for (row, (target, &len)) in targets.iter().zip(lengths).enumerate() {
                    let row_logits = logits.narrow(0, row, 1)?.narrow(1, 0, len)?;
                    let target = Tensor::from_vec(target.clone(), (1, len), logits.device())?;
                    let part = (loss(&row_logits, &target)? * len as f64)?;
                    weighted = Some(match weighted {
                        None => part,
                        Some(sum) => (sum + part)?,
                    });
                }
                Ok((weighted.ok_or("empty variable batch")? / (*positions as f64))?)
            }
        }
    }
}
#[derive(Clone, Copy)]
struct BatchSpec<'a> {
    regime: &'a str,
    target_limit: usize,
    batch_size: usize,
    context: usize,
}
impl TrainingSpec {
    pub fn next_batch(
        &self,
        seq: &Sequence,
        rng: &mut Rng,
        cursor: &mut u64,
        device: &Device,
    ) -> Result<Option<BatchData>> {
        if seq.tokens.len() != self.train_tokens
            || seq.bytes.len() != self.train_tokens
            || self.batch_size == 0
            || self.model.context == 0
            || self.model.context >= self.train_tokens
            || self.batch_size.checked_mul(self.model.context).is_none()
            || (self.identity.regime == "B"
                && (self.first_target == 0
                    || *cursor < self.first_target
                    || *cursor > self.target_limit
                    || self.target_limit > self.train_tokens as u64))
            || !["B", "T"].contains(&self.identity.regime.as_str())
        {
            return Err("invalid training batch configuration or cursor".into());
        }
        training_batch(
            seq,
            rng,
            cursor,
            device,
            BatchSpec {
                regime: &self.identity.regime,
                target_limit: usize::try_from(self.target_limit)?,
                batch_size: self.batch_size,
                context: self.model.context,
            },
        )
    }
}
fn training_batch(
    seq: &Sequence,
    rng: &mut Rng,
    cursor: &mut u64,
    device: &Device,
    spec: BatchSpec<'_>,
) -> Result<Option<BatchData>> {
    let BatchSpec {
        regime,
        target_limit,
        batch_size,
        context,
    } = spec;
    if regime == "T" {
        let starts: Vec<_> = (0..batch_size)
            .map(|_| (rng.next_u64() as usize) % (seq.tokens.len() - context))
            .collect();
        let (x, y, bytes) = runner::batch(seq, &starts, context, device)?;
        *cursor = cursor
            .checked_add((batch_size * context) as u64)
            .ok_or("sampler cursor overflow")?;
        return Ok(Some(BatchData::Full {
            x,
            y,
            positions: (batch_size * context) as u64,
            bytes,
        }));
    }
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut lengths = Vec::new();
    let mut bytes = 0_u64;
    for _ in 0..batch_size {
        if *cursor >= target_limit as u64 {
            break;
        }
        let target_start = usize::try_from(*cursor)?;
        let len = context.min(target_limit - target_start);
        if len == 0 {
            break;
        }
        let input_start = target_start - 1;
        inputs.push(seq.tokens[input_start..input_start + len].to_vec());
        targets.push(seq.tokens[target_start..target_start + len].to_vec());
        lengths.push(len);
        bytes += seq.target_bytes(target_start, len)?;
        *cursor += len as u64;
    }
    if inputs.is_empty() {
        return Ok(None);
    }
    let width = *lengths.iter().max().ok_or("empty B window lengths")?;
    let mut padded = Vec::with_capacity(inputs.len() * width);
    for row in &inputs {
        padded.extend_from_slice(row);
        padded.resize(
            padded.len() + (width - row.len()),
            *row.last().ok_or("empty B input")?,
        );
    }
    let x = Tensor::from_vec(padded, (inputs.len(), width), device)?;
    let positions = lengths.iter().sum::<usize>() as u64;
    Ok(Some(BatchData::Variable {
        x,
        targets,
        lengths,
        positions,
        bytes,
    }))
}
fn emit(log: &mut fs::File, row: Value) -> Result<()> {
    writeln!(log, "{}", serde_json::to_string(&row)?)?;
    log.flush()?;
    println!("{}", serde_json::to_string(&row)?);
    Ok(())
}
fn reconcile_metrics(path: &Path, checkpoint_step: usize) -> Result<MetricsProgress> {
    if !path.exists() {
        return if checkpoint_step == 0 {
            Ok(MetricsProgress::default())
        } else {
            Err("checkpoint has no metrics log needed to restore measured progress".into())
        };
    }
    let bytes = fs::read(path)?;
    let checkpoint_step = u64::try_from(checkpoint_step)?;
    let mut retained = Vec::with_capacity(bytes.len());
    let mut changed = false;
    let mut saved_progress = None;
    for line in bytes.split_inclusive(|&b| b == b'\n') {
        if !line.ends_with(b"\n") {
            changed = true;
            break;
        }
        let row: Value = serde_json::from_slice(&line[..line.len() - 1])?;
        let step = row
            .get("step")
            .and_then(Value::as_u64)
            .or_else(|| row.get("checkpoint_step").and_then(Value::as_u64))
            .or_else(|| {
                (row["stage"] == "final")
                    .then(|| row["updates"].as_u64())
                    .flatten()
            })
            .ok_or("metrics row has no step cursor")?;
        if step <= checkpoint_step && row["stage"] != "final" {
            retained.extend_from_slice(line);
            if row.get("stage").and_then(Value::as_str) == Some("checkpoint")
                && step == checkpoint_step
            {
                let number = |key: &str| -> Result<f64> {
                    let n = row[key].as_f64().ok_or("checkpoint metric missing")?;
                    if !n.is_finite() || n < 0.0 {
                        return Err("checkpoint metric invalid".into());
                    }
                    Ok(n)
                };
                saved_progress = Some(MetricsProgress {
                    training_seconds: number("training_seconds")?,
                    pipeline_seconds: number("pipeline_seconds")?,
                    wall_seconds: number("wall_seconds")?,
                    peak_vram_mib: row["peak_vram_mib"]
                        .as_u64()
                        .ok_or("checkpoint VRAM metric missing")?,
                });
            }
        } else {
            changed = true;
        }
    }
    if bytes.last().is_some_and(|b| *b != b'\n') {
        changed = true;
    }
    let progress = match saved_progress {
        Some(p) => p,
        None if checkpoint_step == 0 => MetricsProgress::default(),
        None => return Err("metrics log has no progress record for checkpoint".into()),
    };
    if changed {
        rewind_log(path, |f| Ok(f.write_all(&retained)?))?;
    }
    Ok(progress)
}

fn rewind_log(path: &Path, write: impl FnOnce(&mut fs::File) -> Result<()>) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary = path.with_extension(format!("jsonl.tmp-{}-{nonce}", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    write(&mut file)?;
    file.sync_all()?;
    drop(file);
    fs::hard_link(
        path,
        path.with_extension(format!("jsonl.before-recovery-{nonce}")),
    )?;
    fs::rename(&temporary, path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn recover_journal(path: &Path, pos: Position) -> Result<MetricsProgress> {
    let Some(prefix) = pos.journal else {
        if pos.step != 0 {
            return Err("training run checkpoint lacks metrics journal identity".into());
        }
        if path.try_exists()? {
            rewind_log(path, |_| Ok(()))?;
        }
        return Ok(pos.timing);
    };
    let mut journal = fs::File::open(path)?;
    if JournalPrefix::read(&mut journal, prefix.bytes)? != prefix {
        return Err("metrics journal differs from checkpoint".into());
    }
    if journal.metadata()?.len() > prefix.bytes {
        journal.rewind()?;
        rewind_log(path, |f| {
            if std::io::copy(&mut journal.take(prefix.bytes), f)? != prefix.bytes {
                return Err("journal prefix changed during recovery".into());
            }
            Ok(())
        })?;
    }
    Ok(pos.timing)
}
fn save_report(path: &Path, report: &domains::Report) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(report)?;
    if path.exists() {
        let mut saved: domains::Report = serde_json::from_slice(&fs::read(path)?)?;
        // A retry remeasures elapsed time. Preserve the original observation.
        saved.seconds = report.seconds;
        if &saved == report {
            return Ok(());
        }
        return Err(format!("refusing to replace existing report {}", path.display()).into());
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary = path.with_extension(format!("json.tmp-{}-{nonce}", std::process::id()));
    let mut f = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    f.write_all(&bytes)?;
    f.sync_all()?;
    drop(f);
    fs::rename(temporary, path)?;
    fs::File::open(path.parent().ok_or("report parent")?)?.sync_all()?;
    Ok(())
}
fn match_input(
    plan: &ExtendedPlan,
    config_path: &Path,
    variant: &str,
    regime: &str,
    seed: u64,
) -> Result<Identity> {
    // ExtendedPlan::load already verified these identities against the actual input files.
    let inputs = &plan.scientific_inputs;
    Ok(Identity {
        config_sha256: sha_file(config_path)?,
        corpus_sha256: inputs["corpus_manifest_sha256"]
            .as_str()
            .ok_or("manifest identity")?
            .into(),
        tokenizer_sha256: inputs["tokenizer_sha256"][variant]
            .as_str()
            .ok_or("tokenizer identity")?
            .into(),
        train_sequence_sha256: inputs["prepared_sequence_sha256"]
            [format!("{variant}_train_sequence_sha256")]
        .as_str()
        .ok_or("TRAIN identity")?
        .into(),
        variant: variant.into(),
        regime: regime.into(),
        seed,
    })
}
pub fn plan(config_path: &Path, data: &Path, output: &Path) -> Result<()> {
    let config = ExtendedPlan::load(config_path, data)?;
    let mut variants = Vec::new();
    let a = Sequence::load(&data.join("A-train.seq"))?;
    let c = Sequence::load(&data.join("C-train.seq"))?;
    let match_result = if config.regimes.iter().any(|r| r == "B") {
        Some(matched_byte_interval(
            &a,
            &c,
            config
                .byte_regime
                .as_ref()
                .and_then(|v| v["max_byte_budget_per_model"].as_u64())
                .ok_or("B byte budget")?,
        )?)
    } else {
        None
    };
    for variant in ["A", "C"] {
        let seq = if variant == "A" { &a } else { &c };
        let by_seed = config.paired_seeds.iter().map(|&seed| Ok(serde_json::json!({
            "seed":seed,
            "raw_target_bytes":deterministic_token_bytes(seq,config.updates_per_model,config.batch_size,config.context,seed)?
        }))).collect::<Result<Vec<_>>>()?;
        variants.push(
            serde_json::json!({"variant":variant,"T_updates":config.updates_per_model,
            "T_target_positions":config.updates_per_model*config.target_positions_per_update,
            "T_raw_target_bytes_by_seed":by_seed,"B_exact_pair_match":match_result}),
        );
    }
    crate::write_new(
        output,
        &serde_json::to_vec_pretty(&serde_json::json!({
            "config_revision":config.revision,"config_sha256":sha_file(config_path)?,
            "corpus_manifest_sha256":config.scientific_inputs["corpus_manifest_sha256"],
            "parameters":config.parameters,"schedule":config.schedule,"variants":variants,
            "claim_limit":"T is matched by target-token positions; B is matched by actual represented target bytes, not FLOPs"
        }))?,
    )?;
    Ok(())
}
pub fn train(
    paths: (&Path, &Path, &Path),
    variant: &str,
    regime: &str,
    seed: u64,
    approval: &str,
    recoverable: bool,
) -> Result<()> {
    let (config_path, data, _) = paths;
    if approval.is_empty()
        || std::env::var("PACKTOK_M5_GPU_APPROVAL").ok().as_deref() != Some(approval)
    {
        return Err(
            "extended GPU execution requires the matching PACKTOK_M5_GPU_APPROVAL reference".into(),
        );
    }
    let started = Instant::now();
    let plan = ExtendedPlan::load(config_path, data)?;
    run_job(
        paths,
        Job {
            variant,
            regime,
            seed,
            approval,
            recoverable,
            started,
        },
        plan,
        || Ok((runner::require_l4()?, runner::cuda()?)),
        nvidia_snapshot,
    )
}

struct Job<'a> {
    variant: &'a str,
    regime: &'a str,
    seed: u64,
    approval: &'a str,
    recoverable: bool,
    started: Instant,
}
fn run_job(
    paths: (&Path, &Path, &Path),
    job: Job<'_>,
    plan: ExtendedPlan,
    open_device: impl FnOnce() -> Result<(String, Device)>,
    snapshot: fn() -> Result<(String, u64)>,
) -> Result<()> {
    let (config_path, data, out) = paths;
    let Job {
        variant,
        regime,
        seed,
        approval,
        recoverable,
        started,
    } = job;
    if !plan.variants.contains(&variant.to_string())
        || !plan.regimes.contains(&regime.to_string())
        || !plan.paired_seeds.contains(&seed)
    {
        return Err("job is not declared in frozen extended config".into());
    }
    let root = config_path
        .parent()
        .and_then(Path::parent)
        .ok_or("experiment root")?;
    let train = Sequence::load(&data.join(format!("{variant}-train.seq")))?;
    let validation = Sequence::load(&data.join(format!("{variant}-validation.seq")))?;
    let test = Sequence::load(&data.join(format!("{variant}-test.seq")))?;
    if train.tokens.len() <= plan.context {
        return Err("insufficient immutable TRAIN sequence".into());
    }
    let byte_match = if regime == "B" {
        let a = Sequence::load(&data.join("A-train.seq"))?;
        let c = Sequence::load(&data.join("C-train.seq"))?;
        let budget = plan
            .byte_regime
            .as_ref()
            .and_then(|v| v["max_byte_budget_per_model"].as_u64())
            .ok_or("B byte budget")?;
        Some(matched_byte_interval(&a, &c, budget)?)
    } else {
        None
    };
    let target_start = byte_match
        .as_ref()
        .map(|m| {
            if variant == "A" {
                m.a_start_token
            } else {
                m.c_start_token
            }
        })
        .unwrap_or(0);
    let target_limit = byte_match
        .as_ref()
        .map(|m| {
            if variant == "A" {
                m.a_end_token
            } else {
                m.c_end_token
            }
        })
        .unwrap_or(usize::MAX);
    let total_scheduler = plan.adamw()?;
    let (gpu, device) = open_device()?;
    let mut model;
    let identity = match_input(&plan, config_path, variant, regime, seed)?;
    let checkpoint = out.join("latest.resume.safetensors");
    let metrics = out.join("metrics.jsonl");
    fs::create_dir_all(out)?;
    let writer = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(out.join(".training-writer"))?;
    writer
        .try_lock()
        .map_err(|e| format!("training output writer lock: {e}"))?;
    let store = out.join("training-v2");
    let generation = if recoverable {
        if checkpoint.exists() {
            return Err("legacy checkpoint in v2 output directory".into());
        }
        fs::create_dir_all(&store)?;
        fs::File::open(out)?.sync_all()?;
        v2::clear_abandoned_writes(&store)?;
        v2::latest(&store)?
    } else {
        if store.exists() {
            return Err("v2 run requires train-recoverable".into());
        }
        None
    };
    let had_checkpoint = checkpoint.exists() || generation.is_some();
    if metrics.exists() && !had_checkpoint {
        return Err(
            "metrics exist without a resumable checkpoint; refusing an ambiguous resume".into(),
        );
    }
    let spec = TrainingSpec {
        identity: identity.clone(),
        tokenizer_artifact_sha256: sha_file(
            &root.join(format!("artifacts/corpus-v2/{variant}-0.packtok")),
        )?,
        tokenizer_mapping_sha256: sha_file(
            &root.join(format!("artifacts/corpus-v2/{variant}.mapping")),
        )?,
        model: plan.architecture.clone(),
        schedule: total_scheduler.clone(),
        batch_size: plan.batch_size,
        train_tokens: train.tokens.len(),
        first_target: target_start as u64,
        target_limit: target_limit.min(train.tokens.len()) as u64,
        backend: if device.is_cpu() { "cpu" } else { "cuda" }.into(),
    };
    let mut embedded = None;
    let (mut optimizer, mut step, mut cursor, mut rng) = if let Some((step, path)) = generation {
        let (restored, opt, pos) = ResumableAdamW::restore(&path, &spec, &device, step)?;
        model = restored;
        embedded = Some(pos);
        (opt, step, pos.cursor, Rng::from_state(pos.rng_state)?)
    } else if had_checkpoint {
        model = Transformer::new(plan.architecture.clone(), seed, &device)?;
        let (optimizer, step, cursor, rng_state) =
            ResumableAdamW::load_checkpoint(&mut model, &identity, &checkpoint)?;
        if optimizer.config() != &total_scheduler {
            return Err("legacy schedule mismatch".into());
        }
        (optimizer, step, cursor, Rng::from_state(rng_state)?)
    } else {
        model = Transformer::new(plan.architecture.clone(), seed, &device)?;
        (
            ResumableAdamW::new(&model, total_scheduler)?,
            0,
            if regime == "B" {
                target_start as u64
            } else {
                0
            },
            Rng::new(seed ^ 0xa341316c9e3779b9),
        )
    };
    let save = |optimizer: &ResumableAdamW, mut position: Position| -> Result<()> {
        if recoverable {
            if position.step > 0 {
                let mut journal = fs::File::open(&metrics)?;
                let size = journal.metadata()?.len();
                position.journal = Some(JournalPrefix::read(&mut journal, size)?);
            }
            optimizer.publish(&model, &spec, position, &store)?;
        } else {
            optimizer.save_checkpoint(
                &model,
                &identity,
                position.rng_state,
                position.step,
                position.cursor,
                &checkpoint,
            )?;
        }
        Ok(())
    };
    let progress = match embedded {
        Some(pos) => recover_journal(&metrics, pos)?,
        None => reconcile_metrics(&metrics, step)?,
    };
    if !had_checkpoint {
        save(
            &optimizer,
            Position {
                step,
                cursor,
                rng_state: rng.state(),
                timing: progress,
                journal: None,
            },
        )?;
    }
    let wall_base = progress.wall_seconds;
    let mut train_seconds = progress.training_seconds;
    let mut pipeline_seconds = progress.pipeline_seconds;
    let mut peak_vram = progress.peak_vram_mib;
    let mut last_checkpoint_step = step;
    let (mut positions, mut bytes_seen) = if regime == "T" {
        let expected = step
            .checked_mul(plan.target_positions_per_update)
            .ok_or("target counter overflow")? as u64;
        if cursor != expected {
            return Err("T checkpoint sampler cursor mismatch".into());
        }
        let mut audit_rng = Rng::new(seed ^ 0xa341316c9e3779b9);
        let mut counted_bytes = 0_u64;
        for _ in 0..step {
            for _ in 0..plan.batch_size {
                let start = (audit_rng.next_u64() as usize) % (train.tokens.len() - plan.context);
                counted_bytes += train.target_bytes(start + 1, plan.context)?;
            }
        }
        if audit_rng.state() != rng.state() {
            return Err("T checkpoint RNG stream mismatch".into());
        }
        (cursor, counted_bytes)
    } else {
        let start = target_start;
        spec.validate(Position {
            step,
            cursor,
            rng_state: rng.state(),
            timing: progress,
            journal: None,
        })?;
        let done = usize::try_from(cursor)? - start;
        (done as u64, train.target_bytes(start, done)?)
    };
    let mut log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics)?;
    fs::File::open(out)?.sync_all()?;
    emit(
        &mut log,
        serde_json::json!({"stage":"resume","variant":variant,"regime":regime,"seed":seed,
        "checkpoint_step":step,"sample_cursor":cursor,"gpu":gpu,"approval_reference":approval}),
    )?;
    let updates = if regime == "T" {
        plan.updates_per_model
    } else {
        usize::MAX
    };
    while (regime == "T" && step < updates) || (regime == "B" && cursor < target_limit as u64) {
        let pipeline = Instant::now();
        device.synchronize()?;
        let now = Instant::now();
        let batch = spec
            .next_batch(&train, &mut rng, &mut cursor, &device)?
            .ok_or("empty extended training batch")?;
        let step_loss = batch.loss(&model)?;
        let value = step_loss.to_scalar::<f32>()?;
        if !value.is_finite() {
            return Err("nonfinite extended training loss".into());
        }
        let gradients = step_loss.backward()?;
        if step == 0 || step % 100 == 0 {
            finite_gradients(&model, &gradients)?;
        }
        optimizer.apply(&model, &gradients)?;
        device.synchronize()?;
        let seconds = now.elapsed().as_secs_f64();
        train_seconds += seconds;
        pipeline_seconds += pipeline.elapsed().as_secs_f64();
        step += 1;
        positions += batch.positions();
        bytes_seen += batch.bytes();

        if step == 1 || step % 100 == 0 {
            let (_, memory) = snapshot()?;
            peak_vram = peak_vram.max(memory);
        }
        emit(
            &mut log,
            serde_json::json!({"stage":"train","variant":variant,"regime":regime,"seed":seed,
            "step":step,"loss_per_token":value,"step_seconds":seconds,"training_seconds":train_seconds,
            "pipeline_seconds":pipeline_seconds,"active_wall_seconds":wall_base+started.elapsed().as_secs_f64(),
            "targets":positions,"raw_target_bytes":bytes_seen,"learning_rate":optimizer.config().learning_rate_at(step),
            "sample_cursor":cursor,"peak_vram_sampled_mib":peak_vram}),
        )?;
        if step == 1
            || step
                % (plan.evaluation["validation_every_updates"]
                    .as_u64()
                    .ok_or("validation interval")? as usize)
                == 0
        {
            device.synchronize()?;
            let score = runner::evaluate(&model, &validation, plan.batch_size, &device)?;
            emit(
                &mut log,
                serde_json::json!({"stage":"validation","step":step,"metrics":score}),
            )?;
            let wall_seconds = wall_base + started.elapsed().as_secs_f64();
            emit(
                &mut log,
                serde_json::json!({"stage":"checkpoint","step":step,"sample_cursor":cursor,
                    "training_seconds":train_seconds,"pipeline_seconds":pipeline_seconds,
                    "wall_seconds":wall_seconds,"peak_vram_mib":peak_vram}),
            )?;
            log.sync_all()?;
            save(
                &optimizer,
                Position {
                    step,
                    cursor,
                    rng_state: rng.state(),
                    journal: None,
                    timing: MetricsProgress {
                        training_seconds: train_seconds,
                        pipeline_seconds,
                        wall_seconds,
                        peak_vram_mib: peak_vram,
                    },
                },
            )?;
            last_checkpoint_step = step;
        }
    }
    if step > last_checkpoint_step {
        let wall_seconds = wall_base + started.elapsed().as_secs_f64();
        emit(
            &mut log,
            serde_json::json!({"stage":"checkpoint","step":step,"sample_cursor":cursor,
                "training_seconds":train_seconds,"pipeline_seconds":pipeline_seconds,
                "wall_seconds":wall_seconds,"peak_vram_mib":peak_vram}),
        )?;
        log.sync_all()?;
        save(
            &optimizer,
            Position {
                step,
                cursor,
                rng_state: rng.state(),
                journal: None,
                timing: MetricsProgress {
                    training_seconds: train_seconds,
                    pipeline_seconds,
                    wall_seconds,
                    peak_vram_mib: peak_vram,
                },
            },
        )?;
    }
    let val = runner::evaluate(&model, &validation, plan.batch_size, &device)?;
    let final_test = runner::evaluate(&model, &test, plan.batch_size, &device)?;
    let val_spans = domains::spans_from_manifest(
        &root.join("provenance/corpus-v2-manifest.json"),
        "validation",
    )?;
    let test_spans =
        domains::spans_from_manifest(&root.join("provenance/corpus-v2-manifest.json"), "test")?;
    let val_domains = domains::evaluate(&model, &validation, &val_spans, "validation", &device)?;
    let test_domains = domains::evaluate(&model, &test, &test_spans, "test", &device)?;
    let val_delta = (val_domains.global_bits_per_byte - val.bits_per_byte).abs();
    let test_delta = (test_domains.global_bits_per_byte - final_test.bits_per_byte).abs();
    if !val_delta.is_finite()
        || !test_delta.is_finite()
        || val_delta > 1e-4
        || test_delta > 1e-4
        || val_domains.global_tokens != val.targets
        || val_domains.global_target_bytes != val.raw_target_bytes
        || test_domains.global_tokens != final_test.targets
        || test_domains.global_target_bytes != final_test.raw_target_bytes
    {
        return Err(format!(
            "per-domain/global byte scoring mismatch val={val_delta} test={test_delta}"
        )
        .into());
    }
    save_report(&out.join("domains-validation.json"), &val_domains)?;
    save_report(&out.join("domains-test.json"), &test_domains)?;
    let (_, memory) = snapshot()?;
    peak_vram = peak_vram.max(memory);
    emit(
        &mut log,
        serde_json::json!({"stage":"final","revision":plan.revision,"variant":variant,"regime":regime,"seed":seed,
        "updates":step,"targets":positions,"raw_target_bytes":bytes_seen,"matched_byte_pair":byte_match,
        "training_seconds":train_seconds,"pipeline_seconds":pipeline_seconds,
        "wall_seconds":wall_base+started.elapsed().as_secs_f64(),
        "tokens_per_second":positions as f64/train_seconds,"raw_bytes_per_second":bytes_seen as f64/train_seconds,
        "sampled_peak_vram_mib":peak_vram,"validation":val,"test":final_test,"domains_validation":val_domains,
        "domains_test":test_domains,"gpu":gpu}),
    )?;
    Ok(())
}

fn deterministic_token_bytes(
    seq: &Sequence,
    updates: usize,
    batch: usize,
    context: usize,
    seed: u64,
) -> Result<u64> {
    let mut rng = Rng::new(seed ^ 0xa341316c9e3779b9);
    let mut bytes = 0_u64;
    for _ in 0..updates {
        for _ in 0..batch {
            let start = (rng.next_u64() as usize) % (seq.tokens.len() - context);
            bytes += seq.target_bytes(start + 1, context)?;
        }
    }
    Ok(bytes)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::tiny_config;

    #[test]
    fn tiny_runner_recovers_committed_journal_and_finishes() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "packtok-m6-runner-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        for sub in ["configs", "data", "provenance", "artifacts/corpus-v2"] {
            fs::create_dir_all(dir.join(sub))?;
        }
        let cfg = dir.join("configs/fixture.json");
        fs::write(dir.join("artifacts/corpus-v2/A-0.packtok"), b"integer-")?;
        fs::write(dir.join("artifacts/corpus-v2/A.mapping"), b"fixture")?;
        let data = dir.join("data");
        let mut plan: ExtendedPlan =
            serde_json::from_slice(include_bytes!("../configs/pilot-2k-v1.json"))?;
        plan.architecture = tiny_config();
        plan.batch_size = 2;
        plan.context = plan.architecture.context;
        plan.target_positions_per_update = plan.batch_size * plan.context;
        plan.updates_per_model = 4;
        plan.schedule["total_updates"] = 4.into();
        plan.evaluation["validation_every_updates"] = 2.into();
        let seq = Sequence {
            tokens: (1..54).collect(),
            bytes: vec![1; 53],
        };
        for split in ["train", "validation", "test"] {
            seq.save(&data.join(format!("A-{split}.seq")))?;
        }
        let manifest = serde_json::json!({"sources":[{"name":"fixture","category":"english-prose"}],
            "units":(["validation","test"].map(|s|serde_json::json!({"proposed":s,"kept":true,"source":"fixture","start":0,"end":53}))),
            "splits":(["validation","test"].map(|s|serde_json::json!({"split":s,"bytes":53})))});
        let manifest_path = dir.join("provenance/corpus-v2-manifest.json");
        fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        fs::write(&cfg, b"m6-tiny-runner-v1")?;
        plan.scientific_inputs["corpus_manifest_sha256"] = sha_file(&manifest_path)?.into();
        plan.scientific_inputs["prepared_sequence_sha256"]["A_train_sequence_sha256"] =
            sha_file(&data.join("A-train.seq"))?.into();
        plan.scientific_inputs["tokenizer_sha256"]["A"] = hash(b"integer-fixture").into();
        let run = |out: &Path, recoverable: bool| {
            run_job(
                (&cfg, &data, out),
                Job {
                    variant: "A",
                    regime: "T",
                    seed: 20261008,
                    approval: "fixture",
                    recoverable,
                    started: Instant::now(),
                },
                plan.clone(),
                || Ok(("CPU fixture".into(), Device::Cpu)),
                || Ok(("CPU fixture".into(), 0)),
            )
        };
        let full = dir.join("full");
        run(&full, true)?;
        assert!(run(&full, false).is_err());
        let recovery = dir.join("restart");
        let store = recovery.join("training-v2");
        fs::create_dir_all(&store)?;
        let name = "step-00000000000000000002.ptc";
        fs::copy(full.join("training-v2").join(name), store.join(name))?;
        let journal = recovery.join("metrics.jsonl");
        assert!(run(&recovery, true).is_err());
        let original = fs::read(full.join("metrics.jsonl"))?;
        fs::write(&journal, &original[..10])?;
        assert!(run(&recovery, true).is_err());
        let mut changed = original.clone();
        changed[0] ^= 1;
        fs::write(&journal, &changed)?;
        assert!(run(&recovery, true).is_err());
        assert_eq!(fs::read(&journal)?, changed);
        fs::write(&journal, &original)?;
        fs::write(store.join(".pending-12-123"), b"partial")?;
        let lock = fs::OpenOptions::new()
            .write(true)
            .open(recovery.join(".training-writer"))?;
        lock.try_lock()?;
        assert!(run(&recovery, true).is_err());
        drop(lock);
        run(&recovery, true)?;
        assert!(!store.join(".pending-12-123").exists());
        assert_eq!(v2::latest(&store)?.unwrap().0, 4);
        let rows = |path: &Path| -> Result<Vec<Value>> {
            fs::read_to_string(path)?
                .lines()
                .map(|line| Ok(serde_json::from_str(line)?))
                .collect()
        };
        let trajectory = |path: &Path| -> Result<Vec<Value>> {
            Ok(rows(path)?
                .iter()
                .filter(|r| r["stage"] == "train")
                .map(|r| {
                    serde_json::json!([
                        &r["step"],
                        &r["targets"],
                        &r["raw_target_bytes"],
                        &r["loss_per_token"],
                        &r["learning_rate"]
                    ])
                })
                .collect())
        };
        assert_eq!(
            trajectory(&journal)?,
            trajectory(&full.join("metrics.jsonl"))?
        );
        // Reentry after both final reports exist preserves their original timing and one final row.
        let report = fs::read(recovery.join("domains-test.json"))?;
        run(&recovery, true)?;
        assert_eq!(report, fs::read(recovery.join("domains-test.json"))?);
        assert_eq!(
            rows(&journal)?
                .iter()
                .filter(|r| r["stage"] == "final")
                .count(),
            1
        );
        let legacy = dir.join("legacy");
        fs::create_dir(&legacy)?;
        fs::write(legacy.join("latest.resume.safetensors"), b"fixture marker")?;
        assert!(run(&legacy, true).is_err());
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn resume_metrics_rewind_to_atomic_checkpoint_and_restore_timing() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("packtok-m5-metrics-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&path);
        let original = concat!(
            "{\"stage\":\"resume\",\"checkpoint_step\":0}\n",
            "{\"stage\":\"train\",\"step\":1,\"step_seconds\":0.1}\n",
            "{\"stage\":\"checkpoint\",\"step\":1,\"training_seconds\":0.1,\"pipeline_seconds\":0.2,\"wall_seconds\":0.4,\"peak_vram_mib\":96}\n",
            "{\"stage\":\"train\",\"step\":2,\"step_seconds\":0.3}\n",
            "{\"stage\":\"validation\",\"step\":2}\n",
            "{\"partial\""
        );
        fs::write(&path, original)?;
        let progress = reconcile_metrics(&path, 1)?;
        assert_eq!(
            progress,
            MetricsProgress {
                training_seconds: 0.1,
                pipeline_seconds: 0.2,
                wall_seconds: 0.4,
                peak_vram_mib: 96,
            }
        );
        assert_eq!(
            fs::read_to_string(&path)?,
            concat!(
                "{\"stage\":\"resume\",\"checkpoint_step\":0}\n",
                "{\"stage\":\"train\",\"step\":1,\"step_seconds\":0.1}\n",
                "{\"stage\":\"checkpoint\",\"step\":1,\"training_seconds\":0.1,\"pipeline_seconds\":0.2,\"wall_seconds\":0.4,\"peak_vram_mib\":96}\n"
            )
        );
        let _ = fs::remove_file(path);
        Ok(())
    }
    #[test]
    fn paired_byte_prefix_selects_equal_actual_exposure() {
        let a = Sequence {
            tokens: vec![1; 6],
            bytes: vec![1, 1, 2, 1, 2, 1],
        };
        let c = Sequence {
            tokens: vec![1; 6],
            bytes: vec![1, 2, 1, 1, 2, 1],
        };
        let matched = matched_byte_interval(&a, &c, 5).unwrap();
        assert_eq!(matched.raw_target_bytes, 4);
        assert_eq!((matched.raw_byte_start, matched.raw_byte_end), (1, 5));
        assert_eq!(
            a.bytes[matched.a_start_token..matched.a_end_token]
                .iter()
                .map(|&v| u64::from(v))
                .sum::<u64>(),
            4
        );
        assert_eq!(
            c.bytes[matched.c_start_token..matched.c_end_token]
                .iter()
                .map(|&v| u64::from(v))
                .sum::<u64>(),
            4
        );
    }

    #[test]
    fn report_retry_keeps_original_time_and_rejects_changed_science() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("packtok-report-retry-{}", std::process::id()));
        fs::create_dir(&dir)?;
        let path = dir.join("domains.json");
        let mut report = domains::Report {
            split: "test".into(),
            global_bits_per_byte: 1.25,
            global_target_bytes: 2,
            global_tokens: 1,
            seconds: 0.1,
            metrics: vec![],
            aggregation_matches_global: true,
        };
        save_report(&path, &report)?;
        let original = fs::read(&path)?;
        report.seconds = 1.0;
        save_report(&path, &report)?;
        report.global_tokens += 1;
        assert!(save_report(&path, &report).is_err());
        assert_eq!(fs::read(&path)?, original);
        assert!(reconcile_metrics(&dir.join("absent"), 10).is_err());
        fs::remove_dir_all(dir)?;
        Ok(())
    }
    #[test]
    fn variable_b_batch_masks_padding_and_preserves_all_real_targets() -> Result<()> {
        let model = Transformer::new(tiny_config(), 29, &Device::Cpu)?;
        let seq = Sequence {
            tokens: (1..=12).collect(),
            bytes: vec![1; 12],
        };
        let mut cursor = 1;
        let mut rng = Rng::new(55);
        let batch = training_batch(
            &seq,
            &mut rng,
            &mut cursor,
            &Device::Cpu,
            BatchSpec {
                regime: "B",
                target_limit: 6,
                batch_size: 2,
                context: 4,
            },
        )?
        .ok_or("expected B batch")?;
        assert_eq!(batch.positions(), 5);
        assert_eq!(batch.bytes(), 5);
        let measured = batch.loss(&model)?.to_scalar::<f32>()?;
        let a_x = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu)?;
        let a_y = Tensor::new(&[[2_u32, 3, 4, 5]], &Device::Cpu)?;
        let b_x = Tensor::new(&[[5_u32]], &Device::Cpu)?;
        let b_y = Tensor::new(&[[6_u32]], &Device::Cpu)?;
        let expected = (loss(&model.forward(&a_x)?, &a_y)?.to_scalar::<f32>()? * 4.0
            + loss(&model.forward(&b_x)?, &b_y)?.to_scalar::<f32>()?)
            / 5.0;
        assert!((measured - expected).abs() < 1e-6);
        Ok(())
    }
    #[test]
    fn frozen_configs_recheck_scientific_inputs_and_schedules() -> Result<()> {
        let data = Path::new("data/prepared-v2");
        let pilot = ExtendedPlan::load(Path::new("configs/pilot-2k-v1.json"), data)?;
        assert_eq!(pilot.updates_per_model, 2000);
        assert_eq!(pilot.paired_seeds, vec![20261008]);
        let long = ExtendedPlan::load(Path::new("configs/extended-20k-v1.json"), data)?;
        assert_eq!(long.updates_per_model, 20000);
        assert_eq!(long.paired_seeds.len(), 5);
        assert_eq!(long.schedule["warmup_updates"].as_u64(), Some(100));
        assert_eq!(long.schedule["decay"].as_str(), Some("cosine"));
        let extension: Value =
            serde_json::from_slice(&fs::read("configs/extended-30k-extension-v1.json")?)?;
        assert_eq!(extension["updates_per_model"].as_u64(), Some(30000));
        assert_eq!(extension["schedule"]["total_updates"].as_u64(), Some(30000));
        assert!(
            extension["status"]
                .as_str()
                .unwrap()
                .contains("not-approved")
        );
        Ok(())
    }
    #[test]
    fn extended_gpu_runner_refuses_execution_without_explicit_approval() {
        let out =
            std::env::temp_dir().join(format!("packtok-m5-no-approval-{}", std::process::id()));
        assert!(!out.exists());
        let result = train(
            (Path::new("does-not-exist"), Path::new("missing"), &out),
            "A",
            "T",
            20261008,
            "",
            true,
        );
        assert!(result.is_err());
        assert!(!out.exists());
    }
}
