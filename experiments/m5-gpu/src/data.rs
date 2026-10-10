use crate::{Result, hash, write_new};
use packtok_core::TokenId;
use packtok_format::Artifact;
use packtok_model::{IdMapping, PackVocabulary};
use packtok_tokenizer::{BpeTokenizer, FactorizedTokenizer, Tokenizer};
use packtok_train::{
    BpeTrainingConfig, FactorizedTrainingConfig, load_corpus, train_bpe_with_provenance,
    train_factorized_bpe_with_provenance_report,
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, time::Instant};

pub struct Adapter {
    artifact: Artifact,
    flat: Option<BpeTokenizer>,
    packed: Option<FactorizedTokenizer>,
    pub mapping: IdMapping,
}
impl Adapter {
    pub fn new(artifact: Artifact) -> Result<Self> {
        let (flat, packed, packs) = if let Some(m) = artifact.flat_bpe() {
            (
                Some(BpeTokenizer::from_artifact(&artifact)?),
                None,
                vec![PackVocabulary {
                    pack_id: 0,
                    token_count: m.vocabulary_size(),
                }],
            )
        } else if let Some(m) = artifact.factorized_bpe() {
            let mut packs: Vec<_> = m
                .packs()
                .iter()
                .map(|p| PackVocabulary {
                    pack_id: p.pack_id(),
                    token_count: p.merges().len() as u32,
                })
                .collect();
            packs.push(PackVocabulary {
                pack_id: artifact.registry().byte_fallback().pack_id(),
                token_count: 256,
            });
            (
                None,
                Some(FactorizedTokenizer::from_artifact(&artifact)?),
                packs,
            )
        } else {
            return Err("unsupported M5 tokenizer".into());
        };
        Ok(Self {
            artifact,
            flat,
            packed,
            mapping: IdMapping::flatten(packs)?,
        })
    }
    pub fn encode(&self, raw: &[u8]) -> Result<Sequence> {
        let ids = if let Some(t) = &self.flat {
            t.encode_bytes(raw)?
        } else {
            self.packed.as_ref().unwrap().encode_bytes(raw)?
        };
        if self.decode(&ids)? != raw {
            return Err("tokenizer roundtrip".into());
        }
        let mut tokens = Vec::with_capacity(ids.len());
        let mut bytes = Vec::with_capacity(ids.len());
        for &id in &ids {
            let global = self.mapping.to_global(id)?;
            if self.mapping.to_packed(global)? != id {
                return Err("mapping sequence identity".into());
            }
            let n = if let Some(m) = self.artifact.flat_bpe() {
                m.byte_length(id.local).ok_or("M1 length")?
            } else if id.pack == self.artifact.registry().byte_fallback().pack_id() {
                1
            } else {
                self.artifact
                    .factorized_bpe()
                    .unwrap()
                    .pack(id.pack)
                    .and_then(|p| p.byte_length(id.local))
                    .ok_or("M2 length")?
            };
            tokens.push(global);
            bytes.push(n as u32);
        }
        let s = Sequence { tokens, bytes };
        s.validate()?;
        if s.bytes.iter().map(|&b| u64::from(b)).sum::<u64>() != raw.len() as u64 {
            return Err("byte coverage".into());
        }
        Ok(s)
    }
    fn decode(&self, ids: &[TokenId]) -> Result<Vec<u8>> {
        Ok(if let Some(t) = &self.flat {
            t.decode_bytes(ids)?
        } else {
            self.packed.as_ref().unwrap().decode_bytes(ids)?
        })
    }
    pub fn decode_global(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let packed = ids
            .iter()
            .map(|&id| self.mapping.to_packed(id))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        self.decode(&packed)
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Sequence {
    pub tokens: Vec<u32>,
    pub bytes: Vec<u32>,
}
impl Sequence {
    pub fn validate(&self) -> Result<()> {
        if self.tokens.len() != self.bytes.len()
            || self.tokens.iter().any(|&t| t >= 512)
            || self.bytes.contains(&0)
        {
            return Err("invalid encoded sequence".into());
        }
        Ok(())
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let mut raw = b"PTM5SEQ\x01".to_vec();
        raw.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        for (&id, &bytes) in self.tokens.iter().zip(&self.bytes) {
            raw.extend_from_slice(&id.to_le_bytes());
            raw.extend_from_slice(&bytes.to_le_bytes());
        }
        write_new(path, &raw)
    }
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read(path)?;
        if raw.len() < 16 || &raw[..8] != b"PTM5SEQ\x01" {
            return Err("sequence header".into());
        }
        let n = usize::try_from(u64::from_le_bytes(raw[8..16].try_into()?))?;
        if n.checked_mul(8).and_then(|n| n.checked_add(16)) != Some(raw.len()) {
            return Err("sequence body size".into());
        }
        let mut tokens = Vec::with_capacity(n);
        let mut bytes = Vec::with_capacity(n);
        for w in raw[16..].chunks(8) {
            tokens.push(u32::from_le_bytes(w[..4].try_into()?));
            bytes.push(u32::from_le_bytes(w[4..].try_into()?));
        }
        let s = Self { tokens, bytes };
        s.validate()?;
        Ok(s)
    }
    pub fn target_bytes(&self, start: usize, n: usize) -> Result<u64> {
        let end = start.checked_add(n).ok_or("target range overflow")?;
        Ok(self
            .bytes
            .get(start..end)
            .ok_or("target range")?
            .iter()
            .map(|&b| u64::from(b))
            .sum())
    }
}
pub fn bits_per_byte(nll: f64, targets: u64, bytes: u64) -> Result<f64> {
    if targets == 0 || bytes == 0 || !nll.is_finite() || nll < 0.0 {
        return Err("invalid metric".into());
    }
    Ok(nll / bytes as f64 / std::f64::consts::LN_2)
}
pub fn prepare(root: &Path) -> Result<()> {
    let train_path = root.join("data/corpus-v2/train.txt");
    let corpus = load_corpus(&train_path)?;
    let artifacts = root.join("artifacts/corpus-v2");
    fs::create_dir_all(&artifacts)?;
    let prepared = root.join("data/prepared-v2");
    fs::create_dir_all(&prepared)?;
    let mut stats = Vec::new();
    for variant in ["A", "C"] {
        let mut previous = None;
        for repetition in 0..2 {
            let now = Instant::now();
            let artifact = if variant == "A" {
                train_bpe_with_provenance(
                    corpus.bytes(),
                    BpeTrainingConfig::default(),
                    corpus.provenance(),
                )?
            } else {
                let report = train_factorized_bpe_with_provenance_report(
                    corpus.bytes(),
                    FactorizedTrainingConfig::default(),
                    corpus.provenance(),
                )?;
                println!("M2 allocation: {:?}", report.packs);
                report.artifact
            };
            let elapsed = now.elapsed().as_secs_f64();
            let raw = artifact.to_bytes()?;
            if previous.as_ref().is_some_and(|b| b != &raw) {
                return Err("tokenizer nondeterminism".into());
            }
            previous = Some(raw.clone());
            write_new(
                &artifacts.join(format!("{variant}-{repetition}.packtok")),
                &raw,
            )?;
            println!(
                "tokenizer {variant} repetition {repetition}: {:.3}s, {} bytes, SHA256 {}",
                elapsed,
                raw.len(),
                hash(&raw)
            );
            stats.push(serde_json::json!({"stage":"tokenizer","variant":variant,"repetition":repetition,"seconds":elapsed,"bytes":raw.len(),"sha256":hash(&raw)}));
        }
        let artifact = Artifact::from_bytes(previous.as_ref().unwrap())?;
        let adapter = Adapter::new(artifact)?;
        if adapter.mapping.len() != 512 {
            return Err("must realize exactly 512 logical IDs".into());
        }
        write_new(
            &artifacts.join(format!("{variant}.mapping")),
            &adapter.mapping.to_bytes(),
        )?;
        for split in ["train", "validation", "test"] {
            let raw = fs::read(root.join(format!("data/corpus-v2/{split}.txt")))?;
            let now = Instant::now();
            let seq = adapter.encode(&raw)?;
            let elapsed = now.elapsed().as_secs_f64();
            let path = prepared.join(format!("{variant}-{split}.seq"));
            seq.save(&path)?;
            stats.push(serde_json::json!({"stage":"encode-roundtrip-map","variant":variant,"split":split,"seconds":elapsed,
                "raw_bytes":raw.len(),"tokens":seq.tokens.len(),"sequence_sha256":hash(&fs::read(path)?),
                "raw_sha256":hash(&raw),"first_token_bytes":seq.bytes.first()}));
            println!(
                "encoded {variant}/{split}: {} bytes -> {} tokens, {:.3}s",
                raw.len(),
                seq.tokens.len(),
                elapsed
            );
        }
    }
    write_new(
        &root.join("provenance/preparation-v2-metrics.json"),
        &serde_json::to_vec_pretty(&stats)?,
    )?;
    Ok(())
}
pub fn verify_prepared(root: &Path) -> Result<()> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("provenance/corpus-v2-manifest.json"))?)?;
    for split in manifest["splits"].as_array().ok_or("missing splits")? {
        let name = split["split"].as_str().ok_or("split name")?;
        if !["train", "validation", "test"].contains(&name) {
            return Err("unknown split".into());
        }
        let bytes = fs::read(root.join(format!("data/corpus-v2/{name}.txt")))?;
        if Some(hash(&bytes).as_str()) != split["sha256"].as_str()
            || Some(bytes.len() as u64) != split["bytes"].as_u64()
        {
            return Err("immutable raw split mismatch".into());
        }
    }
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(
        root.join("provenance/preparation-v2-metrics.json"),
    )?)?;
    let mut seen = std::collections::BTreeSet::new();
    for row in rows {
        let variant = row["variant"].as_str().ok_or("variant")?;
        if !["A", "C"].contains(&variant) {
            return Err("unknown variant".into());
        }
        match row["stage"].as_str() {
            Some("tokenizer") => {
                let repetition = row["repetition"].as_u64().ok_or("repetition")?;
                if repetition > 1 || !seen.insert(format!("tokenizer-{variant}-{repetition}")) {
                    return Err("duplicate repetition".into());
                }
                let raw = fs::read(root.join(format!(
                    "artifacts/corpus-v2/{variant}-{repetition}.packtok"
                )))?;
                if Some(hash(&raw).as_str()) != row["sha256"].as_str() {
                    return Err("tokenizer checksum mismatch".into());
                }
                let artifact = Artifact::from_bytes(&raw)?;
                if artifact.to_bytes()? != raw {
                    return Err("tokenizer encoding mismatch".into());
                }
                let adapter = Adapter::new(artifact)?;
                if adapter.mapping.len() != 512
                    || adapter.mapping.to_bytes()
                        != fs::read(root.join(format!("artifacts/corpus-v2/{variant}.mapping")))?
                {
                    return Err("mapping checksum/width mismatch".into());
                }
                for global in 0..512 {
                    if adapter
                        .mapping
                        .to_global(adapter.mapping.to_packed(global)?)?
                        != global
                    {
                        return Err("mapping bijection".into());
                    }
                }
            }
            Some("encode-roundtrip-map") => {
                let split = row["split"].as_str().ok_or("encoded split")?;
                if !["train", "validation", "test"].contains(&split)
                    || !seen.insert(format!("encoded-{variant}-{split}"))
                {
                    return Err("duplicate encoded split".into());
                }
                let path = root.join(format!("data/prepared-v2/{variant}-{split}.seq"));
                let raw = fs::read(&path)?;
                if Some(hash(&raw).as_str()) != row["sequence_sha256"].as_str() {
                    return Err("sequence checksum mismatch".into());
                }
                let seq = Sequence::load(&path)?;
                if Some(seq.tokens.len() as u64) != row["tokens"].as_u64()
                    || Some(seq.target_bytes(0, seq.tokens.len())?) != row["raw_bytes"].as_u64()
                {
                    return Err("sequence coverage metadata mismatch".into());
                }
            }
            _ => return Err("unknown preparation stage".into()),
        }
    }
    if seen.len() != 10 {
        return Err("incomplete preparation evidence".into());
    }
    for variant in ["A", "C"] {
        if fs::read(root.join(format!("artifacts/corpus-v2/{variant}-0.packtok")))?
            != fs::read(root.join(format!("artifacts/corpus-v2/{variant}-1.packtok")))?
        {
            return Err("tokenizer repetitions differ".into());
        }
    }
    println!(
        "CPU verification: immutable splits, six encoded sequences, both tokenizer repetitions and exhaustive mappings match"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn review_adapter_roundtrips_empty_all_bytes_and_repeated_boundaries() -> Result<()> {
        let samples = [
            Vec::new(),
            (0_u8..=255).collect(),
            " äöüß🙂12\r\n{}\t東京 ".repeat(33).into_bytes(),
            vec![0xff; 257],
            b"a1 a12_\r\nb123\t{} ".repeat(41),
        ];
        for name in ["m1-flat-v2", "m2-factorized-v3"] {
            let artifact = Artifact::from_bytes(&fs::read(format!(
                "../../experiments/m3-model/artifacts/{name}.packtok"
            ))?)?;
            let adapter = Adapter::new(artifact)?;
            for raw in &samples {
                let first = adapter.encode(raw)?;
                let second = adapter.encode(raw)?;
                assert_eq!(first.tokens, second.tokens);
                assert_eq!(first.bytes, second.bytes);
                assert_eq!(adapter.decode_global(&first.tokens)?, *raw);
                assert_eq!(first.target_bytes(0, first.tokens.len())?, raw.len() as u64);
            }
        }
        Ok(())
    }
    #[test]
    fn adapters_exact_mapping_and_prompt_preservation() -> Result<()> {
        for name in ["m1-flat-v2", "m2-factorized-v3"] {
            let a = Artifact::from_bytes(&fs::read(format!(
                "../../experiments/m3-model/artifacts/{name}.packtok"
            ))?)?;
            let ad = Adapter::new(a)?;
            for raw in [
                b"At the harbor".as_slice(),
                "Grüße äöüß 🦀 東京\n".as_bytes(),
                b"\x00\xff\r\n",
            ] {
                let s = ad.encode(raw)?;
                assert_eq!(ad.decode_global(&s.tokens)?, raw);
            }
            assert!(ad.decode_global(&[512]).is_err());
        }
        Ok(())
    }
    #[test]
    fn normalization_excludes_unscored_bytes() -> Result<()> {
        assert!((bits_per_byte(8.0 * std::f64::consts::LN_2, 2, 4)? - 2.0).abs() < 1e-12);
        assert!(bits_per_byte(f64::NAN, 2, 4).is_err());
        assert!(bits_per_byte(1.0, 0, 4).is_err());
        let s = Sequence {
            tokens: vec![1, 2, 3],
            bytes: vec![5, 2, 4],
        };
        assert_eq!(s.target_bytes(1, 2)?, 6);
        assert!(s.target_bytes(usize::MAX, 2).is_err());
        Ok(())
    }
    #[test]
    fn sequence_format_roundtrip_rejects_invalid() -> Result<()> {
        let p = std::env::temp_dir().join(format!("packtok-m5-seq-{}.bin", std::process::id()));
        let s = Sequence {
            tokens: vec![0, 511],
            bytes: vec![1, 4],
        };
        s.save(&p)?;
        let t = Sequence::load(&p)?;
        assert_eq!(s.tokens, t.tokens);
        assert_eq!(s.bytes, t.bytes);
        fs::write(&p, b"PTM5SEQ\x01\xff\xff\xff\xff\xff\xff\xff\xff")?;
        assert!(Sequence::load(&p).is_err());
        fs::remove_file(&p)?;
        assert!(
            Sequence {
                tokens: vec![512],
                bytes: vec![1]
            }
            .validate()
            .is_err()
        );
        Ok(())
    }
}
