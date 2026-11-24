use packtok_m5::{Result, hash};
use redb::{Database, ReadableDatabase, TableDefinition};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, VecDeque},
    path::Path,
};

const EXACT: TableDefinition<&str, u8> = TableDefinition::new("exact");
const NORMALIZED: TableDefinition<&str, u8> = TableDefinition::new("normalized");
const BANDS: TableDefinition<(u32, u64), u8> = TableDefinition::new("bands");
const PASSAGES: TableDefinition<u64, u8> = TableDefinition::new("passages");
const WINDOW: usize = 16;

pub fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
}

pub struct Fingerprints {
    exact: String,
    normalized: String,
    near: Option<u64>,
    passages: BTreeSet<u64>,
}

impl Fingerprints {
    pub fn new(text: &str) -> Self {
        let mut scores = [0i32; 64];
        let mut shingles = VecDeque::with_capacity(12);
        let mut minima: VecDeque<(usize, u64)> = VecDeque::with_capacity(WINDOW);
        let mut passages = BTreeSet::new();
        let mut count = 0;
        let mut normalized = Sha256::new();
        for (i, word) in words(text).enumerate() {
            normalized.update(word.as_bytes());
            normalized.update(b"\0");
            let code = packtok_train::fnv1a64(word.as_bytes());
            if shingles.len() == 12 {
                shingles.pop_front();
            }
            shingles.push_back(code);
            if i >= 2 {
                let h = shingle(shingles.iter().rev().take(3).copied());
                for (bit, score) in scores.iter_mut().enumerate() {
                    *score += if h & (1u64 << bit) != 0 { 1 } else { -1 };
                }
            }
            if shingles.len() == 12 {
                let h = shingle(shingles.iter().copied());
                while minima.back().is_some_and(|&(_, min)| min >= h) {
                    minima.pop_back();
                }
                minima.push_back((i, h));
                if minima.front().is_some_and(|&(pos, _)| pos + WINDOW <= i) {
                    minima.pop_front();
                }
                if i + 1 >= 12 + WINDOW - 1 {
                    passages.insert(minima.front().unwrap().1);
                }
            }
            count += 1;
        }
        // Short documents have too few shingles for a useful near-duplicate score.
        let near = (count >= 32).then(|| {
            scores
                .iter()
                .enumerate()
                .fold(0, |h, (i, &s)| h | (u64::from(s > 0) << i))
        });
        Self {
            exact: hash(text.as_bytes()),
            normalized: format!("{:x}", normalized.finalize()),
            near,
            passages,
        }
    }
}

fn shingle(words: impl Iterator<Item = u64>) -> u64 {
    let mut x = words.fold(0xcbf29ce484222325, |x, w| {
        (x ^ w).wrapping_mul(0x100000001b3)
    });
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

fn bands(h: u64) -> impl Iterator<Item = u32> {
    (0..4).map(move |i| (i << 16) | ((h >> (16 * i)) & 0xffff) as u32)
}

pub struct Index(Database);
impl Index {
    pub fn open(path: &Path) -> Result<Self> {
        let mut builder = Database::builder();
        builder.set_cache_size(32 * 1024 * 1024);
        let db = builder.create(path)?;
        let txn = db.begin_write()?;
        txn.open_table(EXACT)?;
        txn.open_table(NORMALIZED)?;
        txn.open_table(BANDS)?;
        txn.open_table(PASSAGES)?;
        txn.commit()?;
        Ok(Self(db))
    }
    pub fn admit(&self, fp: Fingerprints, split: u8) -> Result<Option<&'static str>> {
        let txn = self.0.begin_read()?;
        if txn.open_table(EXACT)?.get(fp.exact.as_str())?.is_some() {
            return Ok(Some("exact"));
        }
        if txn
            .open_table(NORMALIZED)?
            .get(fp.normalized.as_str())?
            .is_some()
        {
            return Ok(Some("normalized-exact"));
        }
        if let Some(h) = fp.near {
            let table = txn.open_table(BANDS)?;
            for band in bands(h) {
                for item in table.range((band, 0)..=(band, u64::MAX))? {
                    let (key, _) = item?;
                    if (key.value().1 ^ h).count_ones() <= 3 {
                        return Ok(Some("near"));
                    }
                }
            }
        }
        let p = txn.open_table(PASSAGES)?;
        for &h in &fp.passages {
            if p.get(h)?.is_some_and(|entry| entry.value() != split) {
                return Ok(Some("cross-split-passage"));
            }
        }
        drop(p);
        txn.close()?;
        let mut txn = self.0.begin_write()?;
        // This index is disposable. A failed build publishes no dataset.
        txn.set_durability(redb::Durability::None)?;
        txn.open_table(EXACT)?.insert(fp.exact.as_str(), split)?;
        txn.open_table(NORMALIZED)?
            .insert(fp.normalized.as_str(), split)?;
        if let Some(h) = fp.near {
            let mut table = txn.open_table(BANDS)?;
            for band in bands(h) {
                table.insert((band, h), split)?;
            }
        }
        let mut p = txn.open_table(PASSAGES)?;
        for h in fp.passages {
            p.insert(h, split)?;
        }
        drop(p);
        txn.commit()?;
        Ok(None)
    }
}

#[cfg(test)]
#[test]
fn near_threshold_reaches_candidate_through_shared_band() -> Result<()> {
    let root = crate::tests::temp()?;
    let db = Index::open(&root.join("near"))?;
    let signature = |label: &str, near| Fingerprints {
        exact: label.into(),
        normalized: label.into(),
        near: Some(near),
        passages: BTreeSet::new(),
    };
    assert_eq!(db.admit(signature("reference", 0), 0)?, None);
    assert_eq!(
        db.admit(signature("three", 1 | (1 << 16) | (1 << 32)), 1)?,
        Some("near")
    );
    assert_eq!(
        db.admit(signature("four", 1 | (1 << 16) | (1 << 32) | (1 << 48)), 1)?,
        None
    );
    drop(db);
    std::fs::remove_dir_all(root)?;
    Ok(())
}
