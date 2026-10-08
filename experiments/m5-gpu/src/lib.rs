#![forbid(unsafe_code)]
pub mod corpus;
pub mod data;
pub mod model;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub fn hash(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
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
pub mod runner;
