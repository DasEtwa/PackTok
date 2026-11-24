use crate::{
    corpus,
    dedup::{Fingerprints, Index},
    source::{self, Catalog, Format, RightsDecision, Source, Split},
    tokens,
};
use packtok_m5::{Result, hash};
use std::{
    fs,
    io::Cursor,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
#[ignore = "opt-in 100 MB synthetic streaming scale rehearsal"]
fn scale_100_mb() -> Result<()> {
    use std::{
        io::{BufWriter, Write},
        path::Path,
    };
    let root = std::env::var("PACKTOK_M7_SCALE_OUT")?;
    let dir = Path::new(&root);
    fs::create_dir(dir)?;
    let mut rng = packtok_m5::model::Rng::new(20261124);
    let mut sources = Vec::new();
    for (split, id, docs) in [
        (Split::Train, "scale-train", 800),
        (Split::Validation, "scale-validation", 100),
        (Split::Test, "scale-test", 100),
    ] {
        let path = dir.join(id);
        let mut file = BufWriter::new(fs::File::create_new(&path)?);
        for _ in 0..docs {
            let mut text = String::with_capacity(99_988);
            let (mut en, mut de) = (0, 0);
            use std::fmt::Write as _;
            while text.len() + 21 <= 99_988 {
                write!(
                    text,
                    "{} {:016x} ",
                    if id == "scale-test" { "und" } else { "the" },
                    rng.next_u64()
                )?;
                en += usize::from(id != "scale-test");
                de += usize::from(id == "scale-test");
            }
            assert!(en > 3 || de > 3);
            text.extend(std::iter::repeat_n(' ', 99_988 - text.len()));
            serde_json::to_writer(&mut file, &serde_json::json!({"text":text}))?;
            file.write_all(b"\n")?;
        }
        file.flush()?;
        let mut s = entry(
            id,
            split,
            if id == "scale-test" { "de" } else { "en" },
            b"fixture",
        );
        s.format = Format::Jsonl;
        (s.bytes, s.sha256) = source::digest(&path)?;
        sources.push(s);
    }
    let catalog = dir.join("catalog.json");
    corpus::write_json(
        &catalog,
        &Catalog {
            version: 1,
            sources,
        },
    )?;
    let start = std::time::Instant::now();
    corpus::build(&catalog, dir, &dir.join("dataset"))?;
    println!("synthetic build seconds={}", start.elapsed().as_secs_f64());
    let start = std::time::Instant::now();
    tokens::prepare(&dir.join("dataset"), &dir.join("tokenized"), 262144)?;
    println!(
        "synthetic tokenize seconds={}",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

pub fn temp() -> Result<PathBuf> {
    let d = std::env::temp_dir().join(format!(
        "m7-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&d)?;
    Ok(d)
}
pub fn entry(id: &str, split: Split, lang: &str, text: &[u8]) -> Source {
    Source {
        id: id.into(),
        group: id.into(),
        split,
        language: lang.into(),
        category: "fixtures".into(),
        format: Format::Text,
        author: "PackTok".into(),
        publisher: "PackTok".into(),
        version: "fixture-1".into(),
        url: "https://example.invalid/origin".into(),
        download_url: Some("https://example.invalid/file".into()),
        acquired: "synthetic".into(),
        rights: "Apache-2.0".into(),
        rights_decision: RightsDecision::RedistributionCleared,
        evidence: "repository fixture".into(),
        attribution: "PackTok".into(),
        jurisdiction: "fixture".into(),
        bytes: text.len() as u64,
        sha256: hash(text),
    }
}

#[test]
fn bounded_records_skip_oversize_and_reject_invalid_utf8_without_lossy_conversion() -> Result<()> {
    let mut r = Cursor::new(b"123456789\nabc\n".as_slice());
    let mut buffer = Vec::new();
    assert_eq!(corpus::record(&mut r, &mut buffer, 5)?, Some((false, 10)));
    assert_eq!(corpus::record(&mut r, &mut buffer, 5)?, Some((true, 4)));
    assert_eq!(buffer, b"abc\n");
    let d = temp()?;
    let f = d.join("input");
    fs::write(&f, b"{\"text\":\"\xff\"}\n{\"text\":\"ok\"}\n")?;
    let mut got = Vec::new();
    corpus::extract(&f, Format::Jsonl, |_, _, item| {
        got.push(item.map(String::from).map_err(String::from));
        Ok(())
    })?;
    assert_eq!(got, [Err("invalid-utf8".into()), Ok("ok".into())]);
    fs::remove_dir_all(d)?;
    Ok(())
}

#[test]
fn exact_near_and_partial_cross_split_contamination_are_distinguished() -> Result<()> {
    let d = temp()?;
    let index = Index::open(&d.join("dedup"))?;
    let core = (0..130).map(|i| format!("literal{i} ")).collect::<String>();
    assert_eq!(index.admit(Fingerprints::new(&core), 0)?, None);
    assert_eq!(index.admit(Fingerprints::new(&core), 1)?, Some("exact"));
    assert_eq!(
        index.admit(Fingerprints::new(&core.to_uppercase()), 0)?,
        Some("normalized-exact")
    );
    let padded = (0..1300)
        .map(|i| format!("outside{i} "))
        .collect::<String>()
        + &core;
    assert_eq!(
        index.admit(Fingerprints::new(&padded), 1)?,
        Some("cross-split-passage")
    );
    drop(index);
    fs::remove_dir_all(d)?;
    Ok(())
}

#[test]
fn train_artifacts_ignore_held_out_text_and_corruption_is_rejected() -> Result<()> {
    let d = temp()?;
    let train =
        "the chair and the door are in the house and the cat is under the window and in the hall. "
            .repeat(20);
    let val="der Hund ist auf dem alten Dach und die Katze ist unter dem Wagen und sie sitzt auch mit ihr am Rand. ".repeat(12);
    let alt="sie und ich sind auf der anderen Seite und die Erde ist rund und die Wege sind lang und schön. ".repeat(12);
    let test="with both hands he was holding the rope and it was still above the quiet water as far as they could see. ".repeat(12);
    fs::write(d.join("a"), &train)?;
    fs::write(d.join("c"), &test)?;
    let cat = d.join("catalog.json");
    for (i, heldout) in [&val, &alt].into_iter().enumerate() {
        fs::write(d.join("b"), heldout)?;
        if cat.exists() {
            fs::remove_file(&cat)?;
        }
        corpus::write_json(
            &cat,
            &Catalog {
                version: 1,
                sources: vec![
                    entry("c", Split::Test, "en", test.as_bytes()),
                    entry("b", Split::Validation, "de", heldout.as_bytes()),
                    entry("a", Split::Train, "en", train.as_bytes()),
                ],
            },
        )?;
        let out = d.join(format!("d{i}"));
        corpus::build(&cat, &d, &out)?;
        tokens::prepare(&out, &d.join(format!("t{i}")), 1024)?;
    }
    for key in ["A", "C"] {
        for ext in ["packtok", "mapping"] {
            assert_eq!(
                fs::read(d.join(format!("t0/{key}.{ext}")))?,
                fs::read(d.join(format!("t1/{key}.{ext}")))?
            );
        }
    }
    let manifest = d.join("t1/tokens.json");
    let mut edited: serde_json::Value = corpus::json(&manifest)?;
    edited["variants"]["A"]["streams"]["train"]["tokens"] = 0.into();
    fs::write(&manifest, serde_json::to_vec(&edited)?)?;
    assert!(tokens::verify(&d.join("d1"), &d.join("t1")).is_err());
    let win = d.join("t0/A-train.windows");
    let mut raw = fs::read(&win)?;
    raw[10] ^= 1;
    fs::write(&win, raw)?;
    assert!(tokens::verify(&d.join("d0"), &d.join("t0")).is_err());
    let mut bad = source::load(&cat)?;
    bad.sources[0].rights_decision = RightsDecision::Quarantined;
    let quarantine = d.join("quarantine.json");
    corpus::write_json(&quarantine, &bad)?;
    assert!(source::load(&quarantine).is_err());
    bad.sources[0].rights_decision = RightsDecision::RedistributionCleared;
    bad.sources[0].group = bad.sources[2].group.clone();
    let conflict = d.join("conflict.json");
    corpus::write_json(&conflict, &bad)?;
    assert!(source::load(&conflict).is_err());
    fs::remove_dir_all(d)?;
    Ok(())
}
