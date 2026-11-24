use packtok_m5::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{self, Read},
    path::Path,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Ord, PartialOrd)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::Test];
    pub fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Gutenberg,
    Text,
    Jsonl,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RightsDecision {
    LocalResearch,
    RedistributionCleared,
    Quarantined,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub group: String,
    pub split: Split,
    pub language: String,
    pub category: String,
    pub format: Format,
    pub author: String,
    pub publisher: String,
    pub version: String,
    pub url: String,
    pub download_url: Option<String>,
    pub acquired: String,
    pub rights: String,
    pub rights_decision: RightsDecision,
    pub evidence: String,
    pub attribution: String,
    pub jurisdiction: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub version: u32,
    pub sources: Vec<Source>,
}

pub fn digest(path: &Path) -> Result<(u64, String)> {
    let mut sha = Sha256::new();
    let bytes = io::copy(&mut File::open(path)?, &mut sha)?;
    Ok((bytes, format!("{:x}", sha.finalize())))
}

pub fn load(path: &Path) -> Result<Catalog> {
    let catalog: Catalog = serde_json::from_reader(File::open(path)?)?;
    if catalog.version != 1 || catalog.sources.is_empty() {
        return Err("unsupported or empty source catalog".into());
    }
    let mut ids = BTreeSet::new();
    let mut groups = BTreeMap::new();
    for s in &catalog.sources {
        let required = [
            &s.group,
            &s.category,
            &s.author,
            &s.publisher,
            &s.acquired,
            &s.version,
            &s.rights,
            &s.evidence,
            &s.attribution,
            &s.jurisdiction,
        ];
        if s.id.is_empty()
            || !s.id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
            || !ids.insert(&s.id)
            || required.iter().any(|v| v.trim().is_empty())
            || s.rights_decision == RightsDecision::Quarantined
            || !["de", "en"].contains(&s.language.as_str())
            || !https(&s.url)
            || s.download_url.as_ref().is_some_and(|url| !https(url))
            || s.sha256.len() != 64
            || !s
                .sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            || s.bytes == 0
            || groups
                .insert(&s.group, s.split)
                .is_some_and(|prev| prev != s.split)
        {
            return Err(format!(
                "invalid catalog entry or cross-split source group: {}",
                s.id
            )
            .into());
        }
    }
    Ok(catalog)
}

fn https(url: &str) -> bool {
    url.starts_with("https://") && !url.contains(char::is_whitespace)
}

pub fn fetch(catalog: &Path, cache: &Path, id: &str, budget: u64) -> Result<()> {
    fetch_with(catalog, cache, id, budget, &mut Command::new("curl"))
}

fn fetch_with(
    catalog: &Path,
    cache: &Path,
    id: &str,
    budget: u64,
    cmd: &mut Command,
) -> Result<()> {
    let catalog = load(catalog)?;
    let s = catalog
        .sources
        .iter()
        .find(|s| s.id == id)
        .ok_or("source outside catalog")?;
    let url = s
        .download_url
        .as_deref()
        .ok_or("this source is local-cache only")?;
    if s.bytes > budget {
        return Err("source exceeds authorized byte budget".into());
    }
    fs::create_dir_all(cache)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(cache.join(format!(".{id}.lock")))?;
    lock.try_lock()
        .map_err(|e| format!("source cache lock: {e}"))?;
    let path = cache.join(id);
    let part = cache.join(format!(".{id}.part"));
    if path.try_exists()? {
        check(s, &path)?;
        if part.try_exists()? {
            fs::remove_file(&part)?;
        }
        return Ok(());
    }
    let have = if part.try_exists()? {
        part.metadata()?.len()
    } else {
        0
    };
    let have = if have > s.bytes || (have == s.bytes && check(s, &part).is_err()) {
        quarantine(&part)?;
        0
    } else {
        have
    };
    let left = s.bytes - have;
    if left != 0 {
        let mut piece = OpenOptions::new().create(true).append(true).open(&part)?;
        cmd.args([
            "-q",
            "--silent",
            "--show-error",
            "--fail",
            "--globoff",
            "--proto",
            "=https",
            "--connect-timeout",
            "20",
            "--max-time",
            "900",
            "--continue-at",
        ])
        .arg(have.to_string())
        .args(["--max-filesize", &s.bytes.to_string(), "--", url])
        .stdout(Stdio::piped());
        let mut child = cmd.spawn()?;
        let result = (|| -> Result<()> {
            let mut stdout = child.stdout.take().ok_or("curl stdout missing")?;
            io::copy(&mut (&mut stdout).take(left), &mut piece)?;
            if stdout.read(&mut [0])? != 0 {
                return Err("response exceeded the pinned source length".into());
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = child.kill();
        }
        let status = child.wait()?;
        piece.sync_all()?;
        result?;
        if !status.success() {
            return Err(format!(
                "download interrupted ({status}); partial retained for final hash verification"
            )
            .into());
        }
    }
    if let Err(error) = check(s, &part) {
        if part.metadata()?.len() >= s.bytes {
            quarantine(&part)?;
        }
        return Err(error);
    }
    fs::hard_link(&part, &path)?;
    File::open(cache)?.sync_all()?;
    fs::remove_file(part)?;
    Ok(())
}

fn quarantine(part: &Path) -> Result<()> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let mut name = part.as_os_str().to_os_string();
    name.push(format!(".bad-{nonce}"));
    fs::rename(part, name)?;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    #[test]
    fn interrupted_fetch_resumes_at_persisted_offset_and_publishes_only_verified_bytes()
    -> Result<()> {
        let dir = crate::tests::temp()?;
        let catalog = dir.join("catalog.json");
        let src = crate::tests::entry("candidate", Split::Train, "en", b"abcdef");
        crate::corpus::write_json(
            &catalog,
            &Catalog {
                version: 1,
                sources: vec![src],
            },
        )?;
        let mut first = Command::new("sh");
        first.args(["-c", "printf abc; exit 18", "transport"]);
        assert!(fetch_with(&catalog, &dir, "candidate", 5, &mut first).is_err());
        assert!(!dir.join(".candidate.part").exists());
        assert!(fetch_with(&catalog, &dir, "candidate", 6, &mut first).is_err());
        assert_eq!(fs::read(dir.join(".candidate.part"))?, b"abc");
        assert!(!dir.join("candidate").exists());
        let mut second = Command::new("sh");
        second.args(["-c","while [ $# -gt 0 ]; do if [ \"$1\" = --continue-at ]; then test \"$2\" = 3 || exit 17; fi; shift; done; printf def","transport"]);
        fetch_with(&catalog, &dir, "candidate", 6, &mut second)?;
        assert_eq!(fs::read(dir.join("candidate"))?, b"abcdef");
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}

pub fn check(s: &Source, path: &Path) -> Result<()> {
    if digest(path)? != (s.bytes, s.sha256.clone()) {
        return Err(format!("unexpected length or SHA-256 for {}", s.id).into());
    }
    Ok(())
}
