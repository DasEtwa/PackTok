use crate::{
    corpus::{self, MANIFEST},
    source::{self, Split},
};
use packtok_format::Artifact;
use packtok_m5::{Result, data::Adapter};
use packtok_train::{BpeTrainingConfig, FactorizedTrainingConfig, train_bpe, train_factorized_bpe};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub const CONTEXT: usize = 8;
const WIDTH: usize = 1 + 2 * (CONTEXT + 1) * 4;
const MAGIC: &[u8; 8] = b"LIVWIN02";
const ENCODE_BYTES: usize = 65536;
const MAX_SAMPLE: usize = 8 << 20;
pub const VARIANTS: [&str; 2] = ["A", "C"];

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stream {
    pub bytes: u64,
    pub sha256: String,
    pub frames: u64,
    pub full_frames: u64,
    pub tokens: u64,
    pub text_bytes: u64,
    pub scored_tokens: u64,
    pub scored_bytes: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub artifact_sha256: String,
    pub mapping_sha256: String,
    pub artifact_and_mapping_sha256: String,
    pub vocabulary: u32,
    pub streams: BTreeMap<String, Stream>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub dataset_sha256: String,
    pub train_sample_sha256: String,
    pub sample_bytes: usize,
    pub sample_budget: usize,
    pub variants: BTreeMap<String, Variant>,
}

fn sample(root: &Path, m: &corpus::Manifest, limit: usize) -> Result<Vec<u8>> {
    let config = source::load(&root.join("sources.json"))?;
    let sources = config
        .sources
        .iter()
        .filter(|s| s.split == Split::Train)
        .count();
    if sources == 0 || !(1024..=MAX_SAMPLE).contains(&limit) || limit < sources * 2 {
        return Err("invalid bounded tokenizer sample budget".into());
    }
    let cap = limit / sources;
    let mut bytes = Vec::with_capacity(limit);
    let mut taken = BTreeMap::<String, usize>::new();
    for shard in m.shards.iter().filter(|s| s.split == Split::Train) {
        corpus::walk(root, shard, |doc| {
            let n = taken.entry(doc.source).or_default();
            let mut len = (cap.saturating_sub(*n).saturating_sub(1)).min(doc.text.len());
            while !doc.text.is_char_boundary(len) {
                len -= 1;
            }
            if len != 0 {
                bytes.extend_from_slice(&doc.text.as_bytes()[..len]);
                bytes.push(b'\n');
                *n += len + 1;
            }
            Ok(())
        })?;
    }
    if bytes.len() < 1024 {
        return Err("training sample is too small".into());
    }
    Ok(bytes)
}

fn mapping(adapter: &Adapter, artifact: &[u8], dir: &Path, key: &str) -> Result<Variant> {
    let mapping = adapter.mapping.to_bytes();
    packtok_m5::write_new(&dir.join(format!("{key}.packtok")), artifact)?;
    packtok_m5::write_new(&dir.join(format!("{key}.mapping")), &mapping)?;
    let combined = [artifact, mapping.as_slice()].concat();
    Ok(Variant {
        artifact_sha256: packtok_m5::hash(artifact),
        mapping_sha256: packtok_m5::hash(&mapping),
        artifact_and_mapping_sha256: packtok_m5::hash(&combined),
        vocabulary: adapter.mapping.len().try_into()?,
        streams: BTreeMap::new(),
    })
}

pub fn prepare(root: &Path, output: &Path, limit: usize) -> Result<()> {
    let corpus = corpus::verify(root)?;
    let sample = sample(root, &corpus, limit)?;
    let a = train_bpe(&sample, BpeTrainingConfig::default())?;
    let c = train_factorized_bpe(&sample, FactorizedTrainingConfig::default())?;
    let dir = corpus::stage(output)?;
    let mut m = Manifest {
        version: 1,
        dataset_sha256: source::digest(&root.join(MANIFEST))?.1,
        sample_bytes: sample.len(),
        sample_budget: limit,
        train_sample_sha256: packtok_m5::hash(&sample),
        variants: BTreeMap::new(),
    };
    for (key, artifact) in VARIANTS.into_iter().zip([a, c]) {
        let raw = artifact.to_bytes()?;
        let adapter = Adapter::new(artifact)?;
        let mut var = mapping(&adapter, &raw, &dir, key)?;
        for split in Split::ALL {
            let path = path(&dir, key, split);
            let mut out = BufWriter::new(File::create_new(&path)?);
            out.write_all(MAGIC)?;
            let mut stream = Stream {
                bytes: 0,
                sha256: String::new(),
                frames: 0,
                full_frames: 0,
                tokens: 0,
                text_bytes: 0,
                scored_tokens: 0,
                scored_bytes: 0,
            };
            for shard in corpus.shards.iter().filter(|s| s.split == split) {
                corpus::walk(root, shard, |doc| {
                    let mut rest = doc.text.as_str();
                    while !rest.is_empty() {
                        let mut take = rest.len().min(ENCODE_BYTES);
                        while !rest.is_char_boundary(take) {
                            take -= 1;
                        }
                        if take < rest.len() {
                            let cut = rest[..take]
                                .char_indices()
                                .rev()
                                .take(4096)
                                .find(|&(_, c)| c.is_whitespace())
                                .map(|(i, c)| i + c.len_utf8());
                            if let Some(cut) = cut.filter(|&cut| cut > 0) {
                                take = cut;
                            }
                        }
                        let seq = adapter.encode(&rest.as_bytes()[..take])?;
                        rest = &rest[take..];
                        stream.tokens += seq.tokens.len() as u64;
                        stream.text_bytes += take as u64;
                        // Preserve the first token in each chunk and overlap exactly one token between windows.
                        for start in (0..seq.tokens.len().saturating_sub(1).max(1)).step_by(CONTEXT)
                        {
                            let n = (seq.tokens.len() - start).min(CONTEXT + 1);
                            let mut frame = [0; WIDTH];
                            frame[0] = n as u8 | if start == 0 { 128 } else { 0 };
                            for i in 0..n {
                                let offset = 1 + 8 * i;
                                frame[offset..offset + 4]
                                    .copy_from_slice(&seq.tokens[start + i].to_le_bytes());
                                frame[offset + 4..offset + 8]
                                    .copy_from_slice(&seq.bytes[start + i].to_le_bytes());
                                if i > 0 {
                                    stream.scored_bytes += u64::from(seq.bytes[start + i]);
                                }
                            }
                            stream.scored_tokens += (n - 1) as u64;
                            stream.frames += 1;
                            stream.full_frames += u64::from(n == CONTEXT + 1);
                            out.write_all(&frame)?;
                        }
                    }
                    Ok(())
                })?;
            }
            out.flush()?;
            out.get_ref().sync_all()?;
            (stream.bytes, stream.sha256) = source::digest(&path)?;
            println!(
                "{key} {}: {} bytes {} tokens {:.4} bytes/token",
                split.name(),
                stream.text_bytes,
                stream.tokens,
                stream.text_bytes as f64 / stream.tokens as f64
            );
            var.streams.insert(split.name().into(), stream);
        }
        m.variants.insert(key.into(), var);
    }
    corpus::write_json(&dir.join("tokens.json"), &m)?;
    verify(root, &dir)?;
    corpus::publish(&dir, output)
}

fn path(root: &Path, key: &str, split: Split) -> std::path::PathBuf {
    root.join(format!("{key}-{}.windows", split.name()))
}

pub fn load(root: &Path, key: &str) -> Result<(Manifest, Adapter)> {
    if !VARIANTS.contains(&key) {
        return Err("unknown tokenizer variant".into());
    }
    let m: Manifest = corpus::json(&root.join("tokens.json"))?;
    let meta = m.variants.get(key).ok_or("missing tokenizer")?;
    if m.version != 1
        || m.variants.len() != VARIANTS.len()
        || meta.streams.len() != Split::ALL.len()
    {
        return Err("invalid token manifest".into());
    }
    let art = fs::read(root.join(format!("{key}.packtok")))?;
    let map = fs::read(root.join(format!("{key}.mapping")))?;
    let combined = [art.as_slice(), map.as_slice()].concat();
    let adapter = Adapter::new(Artifact::from_bytes(&art)?)?;
    if packtok_m5::hash(&art) != meta.artifact_sha256
        || packtok_m5::hash(&map) != meta.mapping_sha256
        || packtok_m5::hash(&combined) != meta.artifact_and_mapping_sha256
        || adapter.mapping.to_bytes() != map
        || adapter.mapping.len() != meta.vocabulary as usize
    {
        return Err("tokenizer or mapping identity mismatch".into());
    }
    Ok((m, adapter))
}

pub struct Frame {
    pub ids: Vec<u32>,
    pub lengths: Vec<u32>,
    pub chunk_start: bool,
}
pub struct Reader {
    file: BufReader<File>,
    pub frames: u64,
    vocab: u32,
}
impl Reader {
    pub fn open(root: &Path, key: &str, split: Split, meta: &Variant) -> Result<Self> {
        let stream = meta
            .streams
            .get(split.name())
            .ok_or("missing split stream")?;
        let path = path(root, key, split);
        let bytes = stream
            .frames
            .checked_mul(WIDTH as u64)
            .and_then(|n| n.checked_add(MAGIC.len() as u64))
            .ok_or("token count overflow")?;
        if stream.frames == 0
            || bytes != stream.bytes
            || source::digest(&path)? != (stream.bytes, stream.sha256.clone())
        {
            return Err("stream length or digest mismatch".into());
        }
        let mut file = BufReader::new(File::open(path)?);
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err("unknown window format".into());
        }
        Ok(Self {
            file,
            frames: stream.frames,
            vocab: meta.vocabulary,
        })
    }
    pub fn read(&mut self, i: u64) -> Result<Frame> {
        if i >= self.frames {
            return Err("window index out of range".into());
        }
        self.file.seek(SeekFrom::Start(8 + i * WIDTH as u64))?;
        let mut raw = [0; WIDTH];
        self.file.read_exact(&mut raw)?;
        let n = usize::from(raw[0] & 127);
        if !(1..=CONTEXT + 1).contains(&n) || raw[1 + n * 8..].iter().any(|&b| b != 0) {
            return Err("invalid window size or padding".into());
        }
        let mut ids = Vec::with_capacity(n);
        let mut lengths = Vec::with_capacity(n);
        for item in raw[1..1 + n * 8].as_chunks::<8>().0 {
            let id = u32::from_le_bytes(item[..4].try_into()?);
            let len = u32::from_le_bytes(item[4..].try_into()?);
            if id >= self.vocab || len == 0 {
                return Err("invalid window token".into());
            }
            ids.push(id);
            lengths.push(len);
        }
        Ok(Frame {
            ids,
            lengths,
            chunk_start: raw[0] & 128 != 0,
        })
    }
}

pub fn verify(dataset: &Path, root: &Path) -> Result<()> {
    let data: corpus::Manifest = corpus::json(&dataset.join(MANIFEST))?;
    let header: Manifest = corpus::json(&root.join("tokens.json"))?;
    let origin = sample(dataset, &data, header.sample_budget)?;
    if origin.len() != header.sample_bytes
        || packtok_m5::hash(&origin) != header.train_sample_sha256
    {
        return Err("tokenizer sample provenance mismatch".into());
    }
    for key in VARIANTS {
        let (m, _) = load(root, key)?;
        if m.dataset_sha256 != source::digest(&dataset.join(MANIFEST))?.1 {
            return Err("tokenized dataset identity mismatch".into());
        }
        for split in Split::ALL {
            let meta = &m.variants[key];
            let stream = &meta.streams[split.name()];
            let mut reader = Reader::open(root, key, split, meta)?;
            let (mut n, mut bytes, mut full, mut chunks, mut unscored) = (0, 0, 0, 0, 0);
            let mut previous = None;
            for i in 0..reader.frames {
                let frame = reader.read(i)?;
                if frame.chunk_start {
                    chunks += 1;
                    unscored += u64::from(frame.lengths[0]);
                } else if previous != Some((frame.ids[0], frame.lengths[0], true))
                    || frame.ids.len() < 2
                {
                    return Err("invalid continuation window".into());
                }
                previous = Some((
                    *frame.ids.last().unwrap(),
                    *frame.lengths.last().unwrap(),
                    frame.ids.len() == CONTEXT + 1,
                ));
                n += (frame.ids.len() - 1) as u64;
                bytes += frame.lengths[1..]
                    .iter()
                    .map(|&b| u64::from(b))
                    .sum::<u64>();
                full += u64::from(frame.ids.len() == CONTEXT + 1);
            }
            if (n, bytes, full)
                != (
                    stream.scored_tokens,
                    stream.scored_bytes,
                    stream.full_frames,
                )
                || n + chunks != stream.tokens
                || bytes + unscored != stream.text_bytes
                || stream.text_bytes
                    != data
                        .shards
                        .iter()
                        .filter(|s| s.split == split)
                        .map(|s| s.text_bytes)
                        .sum::<u64>()
            {
                return Err("token accounting mismatch".into());
            }
        }
    }
    Ok(())
}
