#![forbid(unsafe_code)]
pub mod corpus;
pub mod data;
pub mod domains;
pub mod extended;
pub mod model;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub fn hash(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Flush a parent directory after publishing a file, propagating any failure.
pub(crate) fn sync_directory(path: &std::path::Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // CreateFile requires BACKUP_SEMANTICS for directory handles; flushing
        // requires write access. No backup privilege is requested or bypassed.
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)?
            .sync_all()
    }
    #[cfg(not(windows))]
    {
        std::fs::File::open(path)?.sync_all()
    }
}

#[cfg(test)]
mod directory_sync_tests {
    use super::sync_directory;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn sync_directory_succeeds_for_existing_and_rejects_missing_directory() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "packtok-m5-dir-sync-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        assert!(sync_directory(&dir).is_ok());
        assert!(sync_directory(&dir.join("missing")).is_err());
        fs::remove_dir(dir).unwrap();
    }
}
pub fn write_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    // Immutable deterministic artifacts: accept an existing byte-identical copy, reject changes.
    if path.exists() {
        if std::fs::read(path)? == bytes {
            return Ok(());
        }
        return Err(format!("refusing to replace {}", path.display()).into());
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
pub mod overfit;
pub mod resume;
pub mod runner;
