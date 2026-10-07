use std::error::Error;

#[derive(Debug, PartialEq)]
pub struct SequenceLengths {
    pub count: usize,
    pub min: usize,
    pub median: usize,
    pub p95: usize,
    pub max: usize,
    pub mean: f64,
}

impl SequenceLengths {
    pub fn new(mut lengths: Vec<usize>) -> Option<Self> {
        if lengths.is_empty() {
            return None;
        }
        lengths.sort_unstable();
        let count = lengths.len();
        let sum: u128 = lengths.iter().map(|length| *length as u128).sum();
        Some(Self {
            count,
            min: lengths[0],
            median: lengths[count.div_ceil(2) - 1],
            p95: lengths[(count * 95).div_ceil(100) - 1],
            max: lengths[count - 1],
            mean: sum as f64 / count as f64,
        })
    }
}

/// OS high-water resident memory for the complete benchmark process.
/// This includes training, model loading, all tokenizers and timed operations.
pub fn process_peak_memory_bytes() -> Result<u64, Box<dyn Error>> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let output = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command"])
            .arg(format!(
                "[System.Diagnostics.Process]::GetProcessById({}).PeakWorkingSet64",
                std::process::id()
            ))
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW; no visible helper window
            .output()?;
        if !output.status.success() {
            return Err(format!(
                "process memory query failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(std::str::from_utf8(&output.stdout)?.trim().parse()?)
    }
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status")?;
        let kib = status
            .lines()
            .find_map(|line| line.strip_prefix("VmHWM:"))
            .ok_or("VmHWM unavailable")?
            .split_whitespace()
            .next()
            .ok_or("VmHWM empty")?
            .parse::<u64>()?;
        Ok(kib
            .checked_mul(1024)
            .ok_or("process memory counter overflow")?)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Err("process peak memory currently supports Windows and Linux".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_quantiles_use_nearest_rank_and_include_empty_records() {
        assert_eq!(SequenceLengths::new(vec![]), None);
        assert_eq!(
            SequenceLengths::new(vec![17, 1, 0, 5]),
            Some(SequenceLengths {
                count: 4,
                min: 0,
                median: 1,
                p95: 17,
                max: 17,
                mean: 5.75,
            })
        );
        let summary = SequenceLengths::new((1..=100).rev().collect()).expect("summary");
        assert_eq!((summary.median, summary.p95, summary.mean), (50, 95, 50.5));
    }

    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn process_peak_memory_is_available_and_nonzero() {
        assert!(process_peak_memory_bytes().expect("OS high-water memory") > 0);
    }
}
