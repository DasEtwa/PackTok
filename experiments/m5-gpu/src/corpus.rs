use crate::{Result, hash, write_new};
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

const BLOCK: usize = 8192;
const LONG: usize = 128;
const SHINGLE: usize = 32;
const STRIDE: usize = 16;
#[derive(Serialize)]
struct Unit {
    source: String,
    start: usize,
    end: usize,
    proposed: &'static str,
    kept: bool,
    reason: &'static str,
    sha256: String,
}
#[derive(Serialize)]
struct Source {
    name: String,
    category: String,
    raw_bytes: usize,
    raw_sha256: String,
    body_start: usize,
    body_end: usize,
    contribution: [usize; 3],
}
struct BytesSource {
    name: String,
    category: String,
    raw: Vec<u8>,
    start: usize,
    end: usize,
}

pub fn rolling(bytes: &[u8], k: usize) -> Vec<u64> {
    if bytes.len() < k || k == 0 {
        return Vec::new();
    }
    let mut power = 1_u64;
    for _ in 1..k {
        power = power.wrapping_mul(257);
    }
    let mut h = 0_u64;
    for &b in &bytes[..k] {
        h = h.wrapping_mul(257).wrapping_add(u64::from(b) + 1);
    }
    let mut out = Vec::with_capacity(bytes.len() - k + 1);
    out.push(h);
    for i in k..bytes.len() {
        h = h
            .wrapping_sub((u64::from(bytes[i - k]) + 1).wrapping_mul(power))
            .wrapping_mul(257)
            .wrapping_add(u64::from(bytes[i]) + 1);
        out.push(h);
    }
    out
}
fn shingles(bytes: &[u8]) -> HashSet<u64> {
    rolling(bytes, SHINGLE)
        .into_iter()
        .step_by(STRIDE)
        .collect()
}
fn near(a: &HashSet<u64>, b: &HashSet<u64>) -> bool {
    let common = a.intersection(b).count();
    let union = a.len() + b.len() - common;
    union > 0 && common * 5 >= union * 4
}
fn body(raw: &[u8]) -> Result<(usize, usize)> {
    let t = std::str::from_utf8(raw)?;
    let start = t.find("*** START OF ").ok_or("missing PG start")?;
    let start = start + t[start..].find('\n').ok_or("missing PG newline")? + 1;
    let end = start + t[start..].find("*** END OF ").ok_or("missing PG end")?;
    Ok((start, end))
}
fn collect(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let p = entry?.path();
        let m = fs::symlink_metadata(&p)?;
        if m.file_type().is_symlink() {
            return Err("source symlink rejected".into());
        }
        if m.is_dir() {
            collect(&p, out)?;
        } else if ["rs", "toml", "json"]
            .contains(&p.extension().and_then(|s| s.to_str()).unwrap_or(""))
        {
            out.push(p);
        }
    }
    Ok(())
}
type Proposed = [Vec<(usize, usize, usize)>; 3];
fn partition(sources: &[BytesSource]) -> Proposed {
    let mut proposed = [Vec::<(usize, usize, usize)>::new(), Vec::new(), Vec::new()];
    let mut bi = 0;
    for (si, s) in sources.iter().enumerate() {
        let mut start = s.start;
        while start < s.end {
            let mut end = (start + BLOCK).min(s.end);
            if end < s.end {
                end += s.raw[end..s.end]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(s.end - end, |n| n + 1);
            }
            let split = match bi % 20 {
                18 => 1,
                19 => 2,
                _ => 0,
            };
            proposed[split].push((si, start, end));
            start = end;
            bi += 1;
        }
    }
    proposed
}

pub fn build(root: &Path) -> Result<()> {
    let data = root.join("data");
    let mut sources = Vec::new();
    let ids = [
        100, 145, 766, 883, 1023, 7469, 2403, 2404, 5323, 24782, 2335, 2336, 2337, 2338, 2339,
        2340, 2341, 2342,
    ];
    fs::create_dir_all(root.join("provenance/pg-notices"))?;
    for id in ids {
        let name = format!("pg{id}.txt");
        let raw = fs::read(data.join("sources").join(&name))?;
        let (start, end) = body(&raw)?;
        let mut notice = raw[..start].to_vec();
        notice.extend_from_slice(&raw[end..]);
        write_new(&root.join("provenance/pg-notices").join(&name), &notice)?;
        sources.push(BytesSource {
            name,
            category: if id == 100 {
                "english-drama"
            } else if id < 2000 || id == 7469 {
                "english-prose"
            } else {
                "german-prose"
            }
            .into(),
            raw,
            start,
            end,
        });
    }
    let rust = data.join("sources/rust");
    let mut paths = Vec::new();
    collect(&rust.join("library"), &mut paths)?;
    paths.push(rust.join("Cargo.toml"));
    paths.sort();
    let mut used = 0;
    for p in paths {
        let raw = fs::read(&p)?;
        std::str::from_utf8(&raw)?;
        if used + raw.len() > 20 * 1024 * 1024 {
            continue;
        } // predeclared whole-file code cap
        used += raw.len();
        let end = raw.len();
        let name = format!(
            "rust/{}",
            p.strip_prefix(&rust)?.to_str().ok_or("nonunicode source")?
        );
        sources.push(BytesSource {
            name,
            category: if p.extension().is_some_and(|s| s == "rs") {
                "code"
            } else {
                "structured"
            }
            .into(),
            raw,
            start: 0,
            end,
        });
    }
    for category in ["synthetic-json", "synthetic-unicode"] {
        let mut s = String::new();
        for i in 0..16_384_u64 {
            if category == "synthetic-json" {
                writeln!(
                    s,
                    "{{\"id\":{i},\"voltage\":{}.{} ,\"range\":[{},{}],\"enabled\":{},\"route\":\"/probe/{:016x}\",\"note\":\"Grüße 東京 🦀\"}}",
                    i % 241,
                    i % 100,
                    i * 31,
                    i * 31 + 17,
                    i % 2 == 0,
                    i.wrapping_mul(0x9e3779b97f4a7c15)
                )?;
            } else {
                writeln!(
                    s,
                    "Probe {i}: Straße, Häuser, Äpfel — 東京 漢字 🦀 🌍 👩🏽‍💻 e\u{0301}; {} ± {} = {}; checksum {:016x}.",
                    i * 31,
                    i % 19,
                    i * 31 + i % 19,
                    i.wrapping_mul(0xa341316c9e3779b9)
                )?;
            }
        }
        let raw = s.into_bytes();
        let end = raw.len();
        write_new(&data.join("sources").join(format!("{category}.txt")), &raw)?;
        sources.push(BytesSource {
            name: format!("{category}.txt"),
            category: category.into(),
            raw,
            start: 0,
            end,
        });
    }
    let names = ["train", "validation", "test"];
    let mut units = Vec::new();
    let mut buffers = [Vec::new(), Vec::new(), Vec::new()];
    let mut source_records: Vec<Source> = sources
        .iter()
        .map(|s| Source {
            name: s.name.clone(),
            category: s.category.clone(),
            raw_bytes: s.raw.len(),
            raw_sha256: hash(&s.raw),
            body_start: s.start,
            body_end: s.end,
            contribution: [0; 3],
        })
        .collect();
    let proposed = partition(&sources);
    let mut accepted_profiles = Vec::<HashSet<u64>>::new();
    let mut index = HashMap::<u64, Vec<usize>>::new();
    let mut long_set = HashSet::<u64>::new();
    for split in 0..3 {
        for &(si, start, end) in &proposed[split] {
            let s = &sources[si];
            let bytes = &s.raw[start..end];
            let profile = shingles(bytes);
            let mut candidates = HashSet::new();
            for h in &profile {
                if let Some(ids) = index.get(h) {
                    candidates.extend(ids.iter().copied());
                }
            }
            let reason = if split > 0 && rolling(bytes, LONG).iter().any(|h| long_set.contains(h)) {
                "shared-128-byte"
            } else if split > 0
                && candidates
                    .iter()
                    .any(|&id| near(&profile, &accepted_profiles[id]))
            {
                "near-duplicate-jaccard80"
            } else {
                "accepted"
            };
            let kept = reason == "accepted";
            if kept {
                buffers[split].extend_from_slice(bytes);
                source_records[si].contribution[split] += bytes.len();
                // Keep TRAIN and then VALIDATION comparison profiles. VALIDATION also rejects duplicates of earlier accepted VALIDATION blocks.
                if split < 2 {
                    let id = accepted_profiles.len();
                    for &h in &profile {
                        index.entry(h).or_default().push(id);
                    }
                    accepted_profiles.push(profile);
                    long_set.extend(rolling(bytes, LONG));
                }
            }
            units.push(Unit {
                source: s.name.clone(),
                start,
                end,
                proposed: names[split],
                kept,
                reason,
                sha256: hash(bytes),
            });
        }
        // Complete concatenations include newly formed boundaries, too.
        if split < 2 {
            long_set.extend(rolling(&buffers[split], LONG));
        }
    }
    guard(&buffers)?;
    let total: usize = buffers.iter().map(Vec::len).sum();
    if !(32 * 1024 * 1024..=64 * 1024 * 1024).contains(&total) {
        return Err(format!("M5 final corpus out of range: {total}").into());
    }
    let out = data.join("corpus-v2");
    fs::create_dir_all(&out)?;
    let mut splits = Vec::new();
    for (i, b) in buffers.iter().enumerate() {
        if b.len() < 100_000 {
            return Err("held-out split too small".into());
        }
        write_new(&out.join(format!("{}.txt", names[i])), b)?;
        splits.push(serde_json::json!({"split":names[i],"bytes":b.len(),"sha256":hash(b)}));
    }
    write_new(
        &root.join("provenance/corpus-v2-manifest.json"),
        &serde_json::to_vec_pretty(&serde_json::json!({
        "version":"m5-corpus-v2","block_bytes":BLOCK,"split_modulo":20,"validation_residue":18,"test_residue":19,
        "long_shared_guard":LONG,"near_shingle_bytes":SHINGLE,"near_stride":STRIDE,"near_jaccard":0.8,
        "raw_total_bytes":total,"sources":source_records,"splits":splits,"units":units}))?,
    )?;
    println!(
        "corpus complete: {total} bytes, split sizes {:?}",
        buffers.iter().map(Vec::len).collect::<Vec<_>>()
    );
    Ok(())
}
pub fn guard(splits: &[Vec<u8>]) -> Result<()> {
    for i in 0..splits.len() {
        let hashes: HashSet<_> = rolling(&splits[i], LONG).into_iter().collect();
        for other in &splits[i + 1..] {
            if rolling(other, LONG).iter().any(|h| hashes.contains(h)) {
                return Err("shared 128-byte passage across final splits".into());
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn short_source_files_share_one_split_counter() {
        let sources: Vec<_> = (0..20)
            .map(|i| BytesSource {
                name: format!("fixture-{i}"),
                category: "code".into(),
                raw: vec![b'a' + i as u8; 300],
                start: 0,
                end: 300,
            })
            .collect();
        let proposed = partition(&sources);
        assert_eq!(
            proposed.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![18, 1, 1]
        );
        assert_eq!(proposed[1], vec![(18, 0, 300)]);
        assert_eq!(proposed[2], vec![(19, 0, 300)]);
    }
    #[test]
    fn rolling_matches_direct_windows() {
        let x: Vec<_> = (0..255).cycle().take(700).collect();
        for k in [1, 32, 128] {
            let expected: Vec<_> = x
                .windows(k)
                .map(|w| {
                    w.iter().fold(0_u64, |h, &b| {
                        h.wrapping_mul(257).wrapping_add(u64::from(b) + 1)
                    })
                })
                .collect();
            assert_eq!(rolling(&x, k), expected);
        }
    }
    #[test]
    fn leakage_checks_exact_and_near() {
        let a: Vec<_> = (0..251).cycle().take(512).collect();
        assert!(guard(&[a.clone(), a.clone()]).is_err());
        let p = shingles(&a);
        assert!(near(&p, &p));
        let b = vec![0; 512];
        assert!(!near(&p, &shingles(&b)));
        assert!(guard(&[a, b]).is_ok());
    }
    #[test]
    fn gutenberg_extraction_keeps_body_bytes() {
        let x = b"notice\n*** START OF BOOK ***\r\nbody\r\n\x00\n*** END OF BOOK ***\nlicense";
        let (s, e) = body(x).unwrap();
        assert_eq!(&x[s..e], b"body\r\n\x00\n");
    }
}
