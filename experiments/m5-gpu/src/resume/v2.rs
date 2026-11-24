//! Full training images. The version fixes FP32 tensor order, optimizer and sampler semantics.
use super::{AdamWConfig, Identity, ResumableAdamW};
use crate::{
    Result,
    model::{ModelConfig, Rng, Transformer},
};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const MAGIC: &[u8; 8] = b"PKTRN002";
const MAX_HEADER: usize = 16 << 20;
const CHUNK: usize = 65536;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingSpec {
    pub identity: Identity,
    pub tokenizer_artifact_sha256: String,
    pub tokenizer_mapping_sha256: String,
    pub model: ModelConfig,
    pub schedule: AdamWConfig,
    pub batch_size: usize,
    pub train_tokens: usize,
    pub first_target: u64,
    pub target_limit: u64,
    pub backend: String,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timing {
    pub training_seconds: f64,
    pub pipeline_seconds: f64,
    pub wall_seconds: f64,
    pub peak_vram_mib: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub step: usize,
    pub cursor: u64,
    pub rng_state: u64,
    pub timing: Timing,
    pub journal: Option<JournalPrefix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JournalPrefix {
    pub bytes: u64,
    pub sha256: [u8; 32],
}
impl JournalPrefix {
    pub fn read(file: &mut File, bytes: u64) -> Result<Self> {
        file.rewind()?;
        let mut sha = Sha256::new();
        if std::io::copy(&mut file.take(bytes), &mut sha)? != bytes {
            return Err("metrics journal truncated before checkpoint".into());
        }
        Ok(Self {
            bytes,
            sha256: sha.finalize().into(),
        })
    }
}

/// Publication errors release their own temporary. A killed process leaves a recognizable pending name.
struct Pending(PathBuf);
impl Drop for Pending {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

impl TrainingSpec {
    pub(crate) fn validate(&self, p: Position) -> Result<()> {
        self.model.validate()?;
        self.schedule.validate()?;
        if [
            p.timing.training_seconds,
            p.timing.pipeline_seconds,
            p.timing.wall_seconds,
        ]
        .iter()
        .any(|s| !s.is_finite() || *s < 0.0)
        {
            return Err("invalid checkpoint timing".into());
        }
        let id = &self.identity;
        for digest in [
            &id.config_sha256,
            &id.corpus_sha256,
            &id.tokenizer_sha256,
            &id.train_sequence_sha256,
            &self.tokenizer_artifact_sha256,
            &self.tokenizer_mapping_sha256,
        ] {
            if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err("invalid training identity SHA-256".into());
            }
        }
        if self.batch_size == 0
            || self.train_tokens <= self.model.context
            || !["cpu", "cuda"].contains(&self.backend.as_str())
            || p.rng_state == 0
            || p.step > self.schedule.total_steps
        {
            return Err("invalid training checkpoint configuration or position".into());
        }
        let width = self
            .batch_size
            .checked_mul(self.model.context)
            .ok_or("batch width overflow")?;
        let offset = p.step.checked_mul(width).ok_or("cursor overflow")? as u64;
        let cursor = match id.regime.as_str() {
            "T" if self.first_target == 0 && self.target_limit == self.train_tokens as u64 => {
                offset
            }
            "B" if self.first_target > 0
                && self.first_target < self.target_limit
                && self.target_limit <= self.train_tokens as u64 =>
            {
                let length = self.target_limit - self.first_target;
                let required_steps = length.div_ceil(width as u64);
                if required_steps > self.schedule.total_steps as u64
                    || p.step as u64 > required_steps
                    || p.rng_state != Rng::new(id.seed ^ 0xa341316c9e3779b9).state()
                {
                    return Err("invalid sequential sampler state".into());
                }
                self.first_target + length.min(offset)
            }
            _ => return Err("invalid training sampler bounds".into()),
        };
        if cursor != p.cursor {
            return Err("checkpoint step and sampling position disagree".into());
        }
        Ok(())
    }
}
#[derive(Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Header {
    spec: TrainingSpec,
    position: Position,
    precision: String,
    math: String,
    accumulation_steps: u32,
    sampler: String,
    layout: Vec<(String, Vec<usize>)>,
}
impl Header {
    fn new(spec: &TrainingSpec, model: &Transformer, position: Position) -> Result<Self> {
        spec.validate(position)?;
        if model.config != spec.model {
            return Err("checkpoint model configuration mismatch".into());
        }
        let mut layout = Vec::new();
        if spec.backend == "cuda" && std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            return Err(
                "CUDA FP32 recovery requires NVIDIA_TF32_OVERRIDE=0 at process start".into(),
            );
        }
        for (name, var) in model.named_vars() {
            let backend = if var.device().is_cpu() {
                "cpu"
            } else if var.device().is_cuda() {
                "cuda"
            } else {
                "unsupported"
            };
            if var.dtype() != DType::F32 || backend != spec.backend {
                return Err("checkpoint backend or precision mismatch".into());
            }
            layout.push((name, var.dims().to_vec()));
        }
        Ok(Self {
            spec: spec.clone(),
            position,
            precision: "fp32-le".into(),
            math: if spec.backend == "cuda" {
                "candle-0.9.1-cuda-tf32-off"
            } else {
                "candle-0.9.1-cpu"
            }
            .into(),
            accumulation_steps: 1,
            sampler: "xorshift64-13-7-17/m5-v1".into(),
            layout,
        })
    }
}

fn filename(step: usize) -> String {
    format!("step-{step:020}.ptc")
}

/// Caller holds the exclusive run writer lock. Only remove regular files named by this writer.
pub(crate) fn clear_abandoned_writes(dir: &Path) -> Result<()> {
    for entry in dir.read_dir()? {
        let entry = entry?;
        let name = entry.file_name();
        let Some((pid, nonce)) = name
            .to_str()
            .and_then(|s| s.strip_prefix(".pending-"))
            .and_then(|s| s.split_once('-'))
        else {
            continue;
        };
        if pid.parse::<u32>().is_ok()
            && nonce.parse::<u128>().is_ok()
            && entry.file_type()?.is_file()
        {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// Committed generations only. An unreadable newest generation is an error on load.
pub fn latest(dir: &Path) -> Result<Option<(usize, PathBuf)>> {
    let mut latest = None;
    for entry in dir.read_dir()? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.ends_with(".ptc") {
            continue;
        }
        let step: usize = name
            .strip_prefix("step-")
            .and_then(|s| s.strip_suffix(".ptc"))
            .ok_or("invalid published checkpoint filename")?
            .parse()?;
        if name != filename(step) || !entry.file_type()?.is_file() {
            return Err("invalid published checkpoint entry".into());
        }
        if latest.as_ref().is_none_or(|(prior, _)| *prior < step) {
            latest = Some((step, entry.path()));
        }
    }
    Ok(latest)
}

fn hash_body(file: &mut File) -> Result<[u8; 32]> {
    let size = file
        .metadata()?
        .len()
        .checked_sub(32)
        .ok_or("truncated checkpoint")?;
    file.rewind()?;
    let mut hasher = Sha256::new();
    if std::io::copy(&mut (&mut *file).take(size), &mut hasher)? != size {
        return Err("truncated checkpoint body".into());
    }
    let digest: [u8; 32] = hasher.finalize().into();
    let mut stored = [0; 32];
    file.read_exact(&mut stored)?;
    if digest != stored {
        return Err("training checkpoint SHA-256 mismatch".into());
    }
    file.rewind()?;
    Ok(digest)
}

fn valid_value(x: f32, second_moment: bool) -> bool {
    x.is_finite() && (!second_moment || x >= 0.0)
}

struct CheckedReader<R> {
    inner: R,
    sha: Sha256,
}
impl<R: Read> Read for CheckedReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(bytes)?;
        self.sha.update(&bytes[..n]);
        Ok(n)
    }
}

impl ResumableAdamW {
    /// Call only between successful complete optimizer updates.
    pub fn publish(
        &self,
        model: &Transformer,
        spec: &TrainingSpec,
        p: Position,
        dir: &Path,
    ) -> Result<PathBuf> {
        if !self.ready || p.step != self.step || self.config != spec.schedule {
            return Err("optimizer checkpoint mismatch".into());
        }
        let header = serde_json::to_vec(&Header::new(spec, model, p)?)?;
        if header.len() > MAX_HEADER {
            return Err("checkpoint layout too large".into());
        }
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let temp = dir.join(format!(".pending-{}-{nonce}", std::process::id()));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let pending = Pending(temp);
        let mut writer = BufWriter::new(&mut file);
        let mut digest = Sha256::new();
        let mut write = |bytes: &[u8]| -> Result<()> {
            writer.write_all(bytes)?;
            digest.update(bytes);
            Ok(())
        };
        write(MAGIC)?;
        write(&(header.len() as u32).to_le_bytes())?;
        write(&header)?;
        let mut bytes = Vec::with_capacity(CHUNK * 4);
        for (name, var) in model.named_vars() {
            for (i, t) in [
                var.as_tensor(),
                self.first.get(&name).ok_or("missing first moment")?,
                self.second.get(&name).ok_or("missing second moment")?,
            ]
            .into_iter()
            .enumerate()
            {
                if t.dims() != var.dims() || t.dtype() != DType::F32 {
                    return Err("checkpoint moment layout mismatch".into());
                }
                let flat = t.detach().flatten_all()?;
                for start in (0..flat.elem_count()).step_by(CHUNK) {
                    let n = CHUNK.min(flat.elem_count() - start);
                    let slice = flat.narrow(0, start, n)?;
                    // Candle copies the underlying storage on device transfer, even for narrow views.
                    let chunk = if slice.device().is_cpu() {
                        slice
                    } else {
                        slice.force_contiguous()?
                    };
                    let values = chunk.to_vec1::<f32>()?;
                    bytes.clear();
                    for x in values {
                        if !valid_value(x, i == 2) {
                            return Err("invalid FP32 optimizer/model tensor".into());
                        }
                        bytes.extend_from_slice(&x.to_le_bytes());
                    }
                    write(&bytes)?;
                }
            }
        }
        writer.write_all(&digest.finalize())?;
        writer.flush()?;
        drop(writer);
        file.sync_all()?;
        hash_body(&mut file)?;
        let destination = dir.join(filename(p.step));
        // link(2) atomically creates a name and refuses to replace an existing generation.
        fs::hard_link(&pending.0, &destination)?;
        File::open(dir)?.sync_all()?;
        Ok(destination)
    }

    /// Construct privately so an I/O/device error cannot expose a partially restored model.
    pub fn restore(
        path: &Path,
        spec: &TrainingSpec,
        device: &Device,
        expected_step: usize,
    ) -> Result<(Transformer, Self, Position)> {
        let mut reader = CheckedReader {
            inner: BufReader::new(File::open(path)?),
            sha: Sha256::new(),
        };
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err("unsupported training checkpoint format".into());
        }
        let mut len = [0; 4];
        reader.read_exact(&mut len)?;
        let len = u32::from_le_bytes(len) as usize;
        if len > MAX_HEADER {
            return Err("oversized training checkpoint header".into());
        }
        let mut bytes = vec![0; len];
        reader.read_exact(&mut bytes)?;
        let header: Header = serde_json::from_slice(&bytes)?;
        let p = header.position;
        if p.step != expected_step || header.spec != *spec {
            return Err("training checkpoint configuration or step mismatch".into());
        }
        spec.validate(p)?;
        let model = Transformer::new(spec.model.clone(), spec.identity.seed, device)?;
        if header != Header::new(spec, &model, p)? {
            return Err("training checkpoint configuration, layout or step mismatch".into());
        }
        let mut first = BTreeMap::new();
        let mut second = BTreeMap::new();
        bytes.resize(CHUNK * 4, 0);
        // Allocate at most one parameter-sized host tensor, then transfer directly into its live slot.
        for (name, var) in model.named_vars() {
            for slot in 0..3 {
                let mut values = Vec::with_capacity(var.elem_count());
                for start in (0..var.elem_count()).step_by(CHUNK) {
                    let n = CHUNK.min(var.elem_count() - start);
                    reader.read_exact(&mut bytes[..4 * n])?;
                    for raw in bytes[..4 * n].as_chunks::<4>().0 {
                        let x = f32::from_le_bytes(*raw);
                        if !valid_value(x, slot == 2) {
                            return Err("invalid checkpoint tensor".into());
                        }
                        values.push(x);
                    }
                }
                let tensor = Tensor::from_vec(values, var.shape(), var.device())?;
                match slot {
                    0 => var.set(&tensor)?,
                    1 => {
                        first.insert(name.clone(), tensor);
                    }
                    _ => {
                        second.insert(name.clone(), tensor);
                    }
                }
            }
        }
        let mut stored = [0; 32];
        reader.inner.read_exact(&mut stored)?;
        let digest: [u8; 32] = reader.sha.finalize().into();
        if digest != stored || reader.inner.read(&mut [0; 1])? != 0 {
            return Err("checkpoint integrity or payload length mismatch".into());
        }
        Ok((
            model,
            Self {
                ready: true,
                config: spec.schedule.clone(),
                step: p.step,
                first,
                second,
            },
            p,
        ))
    }
}
