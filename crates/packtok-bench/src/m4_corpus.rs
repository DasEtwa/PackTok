//! Frozen raw-byte corpus preparation; no model/tokenizer outcome is consulted.
use std::{collections::HashSet, error::Error, fmt::Write, fs, path::Path};

pub(super) const ROOT: &str = "experiments/m4-ablation";
const PASSAGE: usize = 128;

pub(super) fn guard(splits: &[Vec<u8>]) -> Result<(), Box<dyn Error>> {
    for left in 0..splits.len() {
        let passages: HashSet<&[u8]> = splits[left].windows(PASSAGE).collect();
        for (right, other) in splits.iter().enumerate().skip(left + 1) {
            if other.windows(PASSAGE).any(|w| passages.contains(w)) {
                return Err(format!("shared {PASSAGE}-byte passage: splits {left}/{right}").into());
            }
        }
    }
    Ok(())
}

fn body(raw: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let text = std::str::from_utf8(raw)?;
    let start = text
        .find("*** START OF ")
        .ok_or("missing Gutenberg start")?;
    let start = start + text[start..].find('\n').ok_or("missing start newline")? + 1;
    let end = text[start..]
        .find("*** END OF ")
        .ok_or("missing Gutenberg end")?
        + start;
    Ok(raw[start..end].to_vec())
}

pub(super) fn run() -> Result<(), Box<dyn Error>> {
    let root = Path::new(ROOT);
    let mut json = String::new();
    let mut unicode = String::new();
    for i in 0..2000_u32 {
        writeln!(
            json,
            "{{\"record\":{i},\"seed\":{},\"voltage\":{}.{} ,\"enabled\":{},\"path\":\"/dataset/item/{i}\",\"limits\":[{},{}]}}",
            i.wrapping_mul(7919),
            i % 241,
            i % 100,
            i % 2 == 0,
            i + 17,
            i * 3
        )?;
        writeln!(
            unicode,
            "Datensatz {i}: Grüße aus Köln — Straße, Häuser, Äpfel; 東京 漢字 🦀 🌍 👩🏽‍💻 e\u{0301} / {} ± {} = {}.",
            i * 31,
            i % 19,
            i * 31 + i % 19
        )?;
    }
    for (name, raw) in [
        ("synthetic-json.txt", json.as_bytes()),
        ("synthetic-unicode.txt", unicode.as_bytes()),
    ] {
        let path = root.join("sources").join(name);
        if path.exists() {
            if fs::read(&path)? != raw {
                return Err("synthetic source changed".into());
            }
        } else {
            crate::m3::write_new(&path, raw)?;
        }
    }
    let mut sources = Vec::new();
    for name in [
        "austen.txt",
        "goethe.txt",
        "rust-vec.rs",
        "rust-map.rs",
        "rust-Cargo.toml",
        "synthetic-json.txt",
        "synthetic-unicode.txt",
    ] {
        let raw = fs::read(root.join("sources").join(name))?;
        std::str::from_utf8(&raw)?;
        let filtered = if name == "austen.txt" || name == "goethe.txt" {
            body(&raw)?
        } else {
            raw.clone()
        };
        sources.push((name, raw.len(), filtered));
    }
    let mut blocks: [Vec<(usize, &[u8])>; 3] = std::array::from_fn(|_| Vec::new());
    for (source, (_, _, bytes)) in sources.iter().enumerate() {
        let mut start = 0;
        let mut index = 0;
        while start < bytes.len() {
            let candidate = (start + 4096).min(bytes.len());
            let end = if candidate == bytes.len() {
                candidate
            } else {
                candidate
                    + bytes[candidate..]
                        .iter()
                        .position(|&b| b == b'\n')
                        .map_or(bytes.len() - candidate, |n| n + 1)
            };
            let split = match index % 20 {
                18 => 1,
                19 => 2,
                _ => 0,
            };
            blocks[split].push((source, &bytes[start..end]));
            start = end;
            index += 1;
        }
    }
    let train: Vec<u8> = blocks[0]
        .iter()
        .flat_map(|(_, b)| b.iter().copied())
        .collect();
    let train_passages: HashSet<&[u8]> = train.windows(PASSAGE).collect();
    let mut kept: [Vec<(usize, &[u8])>; 3] = [blocks[0].clone(), Vec::new(), Vec::new()];
    let mut removed = [0_usize; 3];
    for split in 1..3 {
        for &(source, bytes) in &blocks[split] {
            if bytes.windows(PASSAGE).any(|w| train_passages.contains(w)) {
                removed[split] += bytes.len();
            } else {
                kept[split].push((source, bytes));
            }
        }
    }
    let validation: Vec<u8> = kept[1]
        .iter()
        .flat_map(|(_, b)| b.iter().copied())
        .collect();
    let val_passages: HashSet<&[u8]> = validation.windows(PASSAGE).collect();
    kept[2].retain(|(_, bytes)| {
        if bytes.windows(PASSAGE).any(|w| val_passages.contains(w)) {
            removed[2] += bytes.len();
            false
        } else {
            true
        }
    });
    let splits: Vec<Vec<u8>> = kept
        .iter()
        .map(|blocks| blocks.iter().flat_map(|(_, b)| b.iter().copied()).collect())
        .collect();
    guard(&splits)?;
    if splits.iter().any(|s| s.len() < 10_000) {
        return Err("insufficient held-out bytes after leakage filtering".into());
    }
    let directory = root.join("corpus-v2");
    fs::create_dir_all(&directory)?;
    let mut report = String::from(
        "M4 corpus preparation v2; block=4096 extended through next LF; source block index modulo20:18 validation,19 test,others train. No inserted bytes or normalization. Drop whole held-out blocks with shared128bytes; train priority, then validation. Exact final guard includes concatenation boundaries.\nsource raw_bytes body_bytes train_bytes validation_bytes test_bytes\n",
    );
    for (index, (name, raw, bytes)) in sources.iter().enumerate() {
        let counts: Vec<usize> = kept
            .iter()
            .map(|b| {
                b.iter()
                    .filter(|(s, _)| *s == index)
                    .map(|(_, b)| b.len())
                    .sum()
            })
            .collect();
        writeln!(
            report,
            "{name} {raw} {} {} {} {}",
            bytes.len(),
            counts[0],
            counts[1],
            counts[2]
        )?;
    }
    for (index, name) in ["train", "validation", "test"].iter().enumerate() {
        let path = directory.join(format!("{name}.txt"));
        if path.exists() {
            if fs::read(&path)? != splits[index] {
                return Err("frozen split differs".into());
            }
        } else {
            crate::m3::write_new(&path, &splits[index])?;
        }
        writeln!(
            report,
            "split={name} bytes={} removed_leaking_bytes={}",
            splits[index].len(),
            removed[index]
        )?;
    }
    let path = directory.join("composition.txt");
    if path.exists() {
        if fs::read_to_string(&path)? != report {
            return Err("composition differs".into());
        }
    } else {
        crate::m3::write_new(&path, report.as_bytes())?;
    }
    print!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlap_guard_checks_long_partial_copy_not_file_equality() {
        let a: Vec<u8> = (0..200).collect();
        let mut b = vec![255; 30];
        b.extend_from_slice(&a[20..160]);
        assert!(guard(&[a, b]).is_err());
        assert!(guard(&[vec![1; 200], vec![2; 200]]).is_ok());
    }
    #[test]
    fn gutenberg_extraction_preserves_body_bytes() {
        assert_eq!(body(b"metadata\n*** START OF TITLE ***\r\n\r\nGr\xc3\xbc\xc3\x9fe\r\n*** END OF TITLE ***\nlicense").unwrap(),b"\r\nGr\xc3\xbc\xc3\x9fe\r\n");
    }
}
