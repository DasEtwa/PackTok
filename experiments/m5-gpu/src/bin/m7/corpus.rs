use crate::{
    dedup::{self, Fingerprints, Index},
    source::{self, Format, Split},
};
use packtok_m5::Result;
use regex::Regex;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub const MAX_DOC: usize = 8 << 20;
const SHARD_BYTES: u64 = 64 << 20;
const RECORD_MAX: usize = 6 * MAX_DOC + 1024;
pub const MANIFEST: &str = "dataset.json";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub source: String,
    pub row: u64,
    pub text: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    pub split: Split,
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub rows: u64,
    pub text_bytes: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct SourceStats {
    documents: u64,
    processed_bytes: u64,
    processed_sha256: String,
    admitted_documents: u64,
    admitted_bytes: u64,
    rejected: BTreeMap<String, u64>,
    rejected_bytes: BTreeMap<String, u64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    version: u32,
    catalog_sha256: String,
    pub shards: Vec<Shard>,
    sources: BTreeMap<String, SourceStats>,
}

pub fn json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

pub fn write_json(path: &Path, v: &impl Serialize) -> Result<()> {
    let mut f = BufWriter::new(File::create_new(path)?);
    serde_json::to_writer_pretty(&mut f, v)?;
    f.write_all(b"\n")?;
    f.flush()?;
    f.get_ref().sync_all()?;
    Ok(())
}

pub fn stage(output: &Path) -> Result<PathBuf> {
    if output.try_exists()? {
        return Err("output exists".into());
    }
    work_directory(output)
}

fn work_directory(output: &Path) -> Result<PathBuf> {
    let name = output.file_name().ok_or("output directory name missing")?;
    let mut pending = name.to_os_string();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    pending.push(format!(".pending-{}-{nonce}", std::process::id()));
    let path = output.with_file_name(pending);
    fs::create_dir(&path)?;
    Ok(path)
}

pub fn publish(staging: &Path, output: &Path) -> Result<()> {
    File::open(staging)?.sync_all()?;
    if output.try_exists()? {
        return Err("output exists".into());
    }
    fs::rename(staging, output)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub fn record(
    reader: &mut impl BufRead,
    buf: &mut Vec<u8>,
    cap: usize,
) -> Result<Option<(bool, u64)>> {
    buf.clear();
    let n = reader.take(cap as u64 + 1).read_until(b'\n', buf)?;
    if n == 0 {
        return Ok(None);
    }
    if n <= cap {
        return Ok(Some((true, n as u64)));
    }
    let rest = if buf.ends_with(b"\n") {
        0
    } else {
        reader.skip_until(b'\n')?
    };
    Ok(Some((false, n as u64 + rest as u64)))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    text: String,
}

pub fn extract(
    path: &Path,
    format: Format,
    mut handle: impl FnMut(u64, u64, std::result::Result<&str, &str>) -> Result<()>,
) -> Result<()> {
    let mut file = BufReader::new(File::open(path)?);
    let mut buf = Vec::new();
    match format {
        Format::Gutenberg | Format::Text => {
            (&mut file).take(MAX_DOC as u64 + 1).read_to_end(&mut buf)?;
            let parsed = if buf.len() > MAX_DOC {
                Err("oversized-document")
            } else {
                std::str::from_utf8(&buf)
                    .map_err(|_| "invalid-utf8")
                    .and_then(|s| {
                        if matches!(format, Format::Gutenberg) {
                            gutenberg(s)
                        } else {
                            Ok(s)
                        }
                    })
            };
            let size = parsed
                .as_ref()
                .map_or(file.get_ref().metadata()?.len(), |s| s.len() as u64);
            handle(0, size, parsed)?;
        }
        Format::Jsonl => {
            let mut row = 0;
            while let Some((valid, raw_bytes)) = record(&mut file, &mut buf, MAX_DOC)? {
                let value = if !valid {
                    Err("oversized-record")
                } else if std::str::from_utf8(&buf).is_err() {
                    Err("invalid-utf8")
                } else {
                    serde_json::from_slice::<Input>(&buf).map_err(|_| "malformed-jsonl")
                };
                match &value {
                    Ok(v) => handle(row, v.text.len() as u64, Ok(&v.text))?,
                    Err(reason) => handle(row, raw_bytes, Err(reason))?,
                }
                row += 1;
            }
        }
    }
    Ok(())
}

fn gutenberg(text: &str) -> std::result::Result<&str, &'static str> {
    let mut offset = 0;
    let mut start = None;
    for line in text.split_inclusive('\n') {
        let marker = line.trim();
        if marker.starts_with("*** START OF ") {
            start = Some(offset + line.len());
        } else if marker.starts_with("*** END OF ")
            && let Some(start) = start
        {
            return Ok(&text[start..offset]);
        }
        offset += line.len();
    }
    Err("missing-gutenberg-boundary")
}

struct Filters {
    contact: Regex,
}

#[cfg(test)]
#[test]
fn filter_catches_spaced_contacts_and_labeled_phone_numbers() -> Result<()> {
    let f = Filters::new()?;
    for text in [
        "Tel: 030 1234567",
        "foo@example.invalid",
        "DE89 3704 0044 0532 0130 00",
    ] {
        assert_eq!(f.issue(text, "de"), Some("contact-pattern"));
    }
    Ok(())
}
impl Filters {
    fn new() -> Result<Self> {
        Ok(Self {
            contact: Regex::new(
                r"(?ix)
           [a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b
           | (?:tel(?:ephone|efon)?|phone|mobile|fax)\s*[:.]\s*\+?\d[\d()\x20.\-]{6,}\d
           | \bDE[0-9]{2}(?:\x20?[0-9]){18}\b",
            )?,
        })
    }

    fn issue(&self, text: &str, expected_language: &str) -> Option<&'static str> {
        if text
            .chars()
            .any(|c| c.is_control() && !matches!(c, '\r' | '\n' | '\t'))
        {
            return Some("control-character");
        }
        if self.contact.is_match(text) {
            return Some("contact-pattern");
        }
        let mut counts = [0usize; 2];
        let mut total = 0;
        for w in dedup::words(text).take(4096) {
            total += 1;
            if [
                "the", "and", "with", "are", "was", "were", "this", "that", "which", "his", "her",
                "of", "to", "in", "is", "for", "from",
            ]
            .contains(&w.as_str())
            {
                counts[0] += 1;
            }
            if [
                "der", "die", "das", "und", "den", "dem", "des", "ist", "sind", "nicht", "eine",
                "einer", "mit", "auch", "sie", "ich", "auf", "von", "zu", "sich",
            ]
            .contains(&w.as_str())
            {
                counts[1] += 1;
            }
        }
        if total < 24 {
            return Some("too-short");
        }
        let expected = usize::from(expected_language == "de");
        if counts[expected] < 3 || counts[expected] < counts[1 - expected] + 2 {
            return Some("language-uncertain-or-mismatch");
        }
        None
    }
}

struct Writer {
    out: BufWriter<File>,
    shard: Shard,
}

impl Writer {
    fn new(dir: &Path, split: Split, index: usize) -> Result<Self> {
        let name = format!("{}-{index:06}.jsonl", split.name());
        let out = BufWriter::new(File::create_new(dir.join(&name))?);
        Ok(Self {
            out,
            shard: Shard {
                split,
                name,
                bytes: 0,
                sha256: String::new(),
                rows: 0,
                text_bytes: 0,
            },
        })
    }
    fn finish(mut self, dir: &Path) -> Result<Shard> {
        self.out.flush()?;
        self.out.get_ref().sync_all()?;
        let (size, digest) = source::digest(&dir.join(&self.shard.name))?;
        if size != self.shard.bytes {
            return Err("shard length mismatch".into());
        }
        self.shard.sha256 = digest;
        Ok(self.shard)
    }
}

struct IndexFiles {
    database: PathBuf,
    scratch: Option<PathBuf>,
    removed: bool,
}

impl IndexFiles {
    fn remove(&mut self) -> Result<()> {
        fs::remove_file(&self.database)?;
        if let Some(dir) = &self.scratch {
            fs::remove_dir(dir)?;
        }
        self.removed = true;
        Ok(())
    }
}

impl Drop for IndexFiles {
    fn drop(&mut self) {
        if !self.removed {
            let _ = fs::remove_file(&self.database);
            if let Some(dir) = &self.scratch {
                let _ = fs::remove_dir(dir);
            }
        }
    }
}

pub fn build(catalog: &Path, cache: &Path, output: &Path, temp_root: Option<&Path>) -> Result<()> {
    let dir = stage(output)?;
    fs::copy(catalog, dir.join("sources.json"))?;
    File::open(dir.join("sources.json"))?.sync_all()?;
    let mut config = source::load(&dir.join("sources.json"))?;
    config
        .sources
        .sort_by(|a, b| (a.split, &a.id).cmp(&(b.split, &b.id)));
    let (_, catalog_sha256) = source::digest(&dir.join("sources.json"))?;
    let scratch = temp_root
        .map(|root| work_directory(&root.join("packtok-m7-dedup")))
        .transpose()?;
    let mut index_files = IndexFiles {
        database: scratch.as_deref().unwrap_or(&dir).join("dedup.redb"),
        scratch,
        removed: false,
    };
    let index = Index::open(&index_files.database)?;
    let filters = Filters::new()?;
    let mut manifest = Manifest {
        version: 1,
        catalog_sha256,
        shards: Vec::new(),
        sources: BTreeMap::new(),
    };
    let mut writer: Option<Writer> = None;
    for source in config.sources {
        let path = cache.join(&source.id);
        source::check(&source, &path)?;
        let mut stats = SourceStats::default();
        let mut processed = sha2::Sha256::default();
        use sha2::Digest;
        extract(&path, source.format, |row, size, text| {
            stats.documents += 1;
            let disposition = match text {
                Err(reason) => Err(reason),
                Ok(text) => {
                    stats.processed_bytes += size;
                    processed.update(size.to_le_bytes());
                    processed.update(text.as_bytes());
                    match filters.issue(text, &source.language) {
                        Some(reason) => Err(reason),
                        None => match index.admit(Fingerprints::new(text), source.split as u8)? {
                            Some(reason) => Err(reason),
                            None => Ok(text),
                        },
                    }
                }
            };
            let text = match disposition {
                Ok(text) => text,
                Err(reason) => {
                    *stats.rejected.entry(reason.into()).or_default() += 1;
                    *stats.rejected_bytes.entry(reason.into()).or_default() += size;
                    return Ok(());
                }
            };
            let mut line = serde_json::to_vec(&Document {
                source: source.id.clone(),
                row,
                text: text.into(),
            })?;
            line.push(b'\n');
            if writer.as_ref().is_some_and(|w| {
                w.shard.split != source.split || w.shard.bytes + line.len() as u64 > SHARD_BYTES
            }) {
                manifest.shards.push(writer.take().unwrap().finish(&dir)?);
            }
            let w = match writer.as_mut() {
                Some(w) => w,
                None => writer.insert(Writer::new(&dir, source.split, manifest.shards.len())?),
            };
            w.out.write_all(&line)?;
            w.shard.bytes += line.len() as u64;
            w.shard.rows += 1;
            w.shard.text_bytes += size;
            stats.admitted_documents += 1;
            stats.admitted_bytes += size;
            Ok(())
        })?;
        source::check(&source, &path)?;
        stats.processed_sha256 = format!("{:x}", processed.finalize());
        manifest.sources.insert(source.id, stats);
    }
    if let Some(w) = writer {
        manifest.shards.push(w.finish(&dir)?);
    }
    drop(index);
    for split in Split::ALL {
        if !manifest.shards.iter().any(|s| s.split == split) {
            return Err(format!("empty {} split after filtering", split.name()).into());
        }
    }
    write_json(&dir.join(MANIFEST), &manifest)?;
    verify(&dir)?;
    index_files.remove()?;
    publish(&dir, output)
}

pub fn walk(
    dir: &Path,
    shard: &Shard,
    mut visit: impl FnMut(Document) -> Result<()>,
) -> Result<()> {
    let mut file = BufReader::new(File::open(dir.join(&shard.name))?);
    let mut line = Vec::new();
    let mut rows = 0;
    let mut bytes = 0;
    while let Some((valid, _)) = record(&mut file, &mut line, RECORD_MAX)? {
        if !valid || !line.ends_with(b"\n") {
            return Err("oversized or truncated shard record".into());
        }
        let doc: Document = serde_json::from_slice(&line)?;
        if doc.text.len() > MAX_DOC {
            return Err("oversized shard document".into());
        }
        rows += 1;
        bytes += doc.text.len() as u64;
        visit(doc)?;
    }
    if (rows, bytes) != (shard.rows, shard.text_bytes) {
        return Err("shard document counts mismatch".into());
    }
    Ok(())
}

pub fn verify(dir: &Path) -> Result<Manifest> {
    let m: Manifest = json(&dir.join(MANIFEST))?;
    let sources = source::load(&dir.join("sources.json"))?;
    if m.version != 1 || m.catalog_sha256 != source::digest(&dir.join("sources.json"))?.1 {
        return Err("dataset schema or source catalog hash mismatch".into());
    }
    let by_id: BTreeMap<_, _> = sources
        .sources
        .iter()
        .map(|s| (s.id.as_str(), s.split))
        .collect();
    let mut last = None;
    let mut seen = BTreeMap::<String, (u64, u64)>::new();
    for (i, s) in m.shards.iter().enumerate() {
        if s.name != format!("{}-{i:06}.jsonl", s.split.name())
            || s.rows == 0
            || s.bytes > SHARD_BYTES
            || source::digest(&dir.join(&s.name))? != (s.bytes, s.sha256.clone())
        {
            return Err("dataset shard identity or size mismatch".into());
        }
        walk(dir, s, |doc| {
            if by_id.get(doc.source.as_str()) != Some(&s.split)
                || last
                    .as_ref()
                    .is_some_and(|p| p >= &(s.split, doc.source.clone(), doc.row))
            {
                return Err("source split, duplicate ID or document ordering violation".into());
            }
            let entry = seen.entry(doc.source.clone()).or_default();
            entry.0 += 1;
            entry.1 += doc.text.len() as u64;
            last = Some((s.split, doc.source, doc.row));
            Ok(())
        })?;
    }
    if m.sources.len() != by_id.len()
        || !by_id.keys().all(|id| m.sources.contains_key(*id))
        || m.sources.iter().any(|(id, s)| {
            seen.get(id).copied().unwrap_or_default() != (s.admitted_documents, s.admitted_bytes)
        })
        || Split::ALL
            .iter()
            .any(|v| !m.shards.iter().any(|s| s.split == *v))
    {
        return Err("dataset source counts or split coverage mismatch".into());
    }
    println!(
        "verified {} shards, {} documents, {} text bytes",
        m.shards.len(),
        m.shards.iter().map(|s| s.rows).sum::<u64>(),
        m.shards.iter().map(|s| s.text_bytes).sum::<u64>()
    );
    Ok(m)
}
