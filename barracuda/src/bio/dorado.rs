// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dorado basecaller subprocess delegation for nanopore data.
//!
//! Discovers the `dorado` binary at runtime via `$WETSPRING_DORADO_BIN`,
//! `$PATH`, or standard install locations. Invokes it as a subprocess and
//! parses the resulting FASTQ output. Falls back to the built-in threshold
//! basecaller when Dorado is unavailable.
//!
//! This follows the primal pattern: no hardcoded paths, runtime capability
//! discovery, and graceful degradation.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Dorado model selection for basecalling quality/speed tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoradoModel {
    /// Fast model (~4kbp/s per GPU) — lowest accuracy.
    Fast,
    /// High accuracy model (~1kbp/s per GPU).
    Hac,
    /// Super accuracy model (~0.3kbp/s per GPU) — highest accuracy.
    Sup,
}

impl DoradoModel {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Hac => "hac",
            Self::Sup => "sup",
        }
    }
}

/// Configuration for a Dorado basecalling run.
#[derive(Debug, Clone)]
pub struct DoradoConfig {
    /// Path to the Dorado binary (auto-discovered if `None`).
    pub binary: Option<PathBuf>,
    /// Basecalling model to use.
    pub model: DoradoModel,
    /// Device specification (e.g. "cuda:0", "cpu").
    pub device: Option<String>,
    /// Emit moves table for alignment.
    pub emit_moves: bool,
    /// Minimum quality score filter (Q-score).
    pub min_qscore: Option<u32>,
}

impl Default for DoradoConfig {
    fn default() -> Self {
        Self {
            binary: None,
            model: DoradoModel::Hac,
            device: None,
            emit_moves: false,
            min_qscore: None,
        }
    }
}

/// Result of a Dorado basecalling run.
#[derive(Debug, Clone)]
pub struct DoradoResult {
    /// Number of reads basecalled.
    pub n_reads: usize,
    /// Total bases called.
    pub total_bases: usize,
    /// Path to the output FASTQ file.
    pub output_path: PathBuf,
    /// Dorado version string (if captured).
    pub dorado_version: Option<String>,
}

/// A basecalled read from Dorado FASTQ output.
#[derive(Debug, Clone)]
pub struct BasecalledRead {
    /// Read identifier from the FASTQ header.
    pub read_id: String,
    /// Basecalled nucleotide sequence.
    pub sequence: Vec<u8>,
    /// Quality scores (Phred+33 encoded).
    pub quality: Vec<u8>,
}

/// Environment variable override for the Dorado binary path.
const DORADO_ENV_VAR: &str = "WETSPRING_DORADO_BIN";

/// Binary name to search for on `$PATH`.
const DORADO_BIN_NAME: &str = "dorado";

/// Standard ONT install locations (checked after `$PATH`).
const STANDARD_DORADO_DIRS: &[&str] = &["/opt/ont/dorado/bin"];

/// Discover the Dorado binary using capability-based resolution.
///
/// # Discovery cascade
///
/// 1. **`$WETSPRING_DORADO_BIN`** — explicit override (gate operators, CI).
/// 2. **`$PATH` search** — pure Rust directory scan, no subprocess.
/// 3. **Standard install locations** — `/opt/ont/dorado/bin/dorado`,
///    `$HOME/.local/bin/dorado`.
///
/// Returns `None` if Dorado is not found. No subprocess is spawned
/// during discovery (the previous `which` call is replaced by a
/// direct `$PATH` scan for cross-platform correctness).
#[must_use]
pub fn discover_dorado() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var(DORADO_ENV_VAR) {
        let p = PathBuf::from(&explicit);
        if p.is_file() {
            return Some(p);
        }
    }

    if let Some(found) = find_on_path(DORADO_BIN_NAME) {
        return Some(found);
    }

    STANDARD_DORADO_DIRS
        .iter()
        .map(|d| Path::new(d).join(DORADO_BIN_NAME))
        .chain(dirs_home().map(|h| h.join(".local/bin").join(DORADO_BIN_NAME)))
        .find(|p| p.is_file())
}

/// Search `$PATH` for a binary by name (pure Rust, no subprocess).
fn find_on_path(binary: &str) -> Option<PathBuf> {
    let path_var = std::env::var("PATH").ok()?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Check whether Dorado is available on this system.
#[must_use]
pub fn is_dorado_available() -> bool {
    discover_dorado().is_some()
}

/// Query the installed Dorado version.
///
/// # Errors
///
/// Returns [`Error::Nanopore`] if Dorado is not found or version query fails.
pub fn dorado_version() -> Result<String> {
    let bin = discover_dorado().ok_or_else(|| Error::Nanopore("dorado binary not found".into()))?;

    let output = Command::new(&bin)
        .arg("--version")
        .output()
        .map_err(|e| Error::Nanopore(format!("failed to run dorado: {e}")))?;

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(combined.trim().to_string())
}

/// Run Dorado basecalling on an input file (POD5 or FAST5).
///
/// Writes FASTQ output to `output_dir` and returns a summary.
///
/// # Errors
///
/// Returns [`Error::Nanopore`] if Dorado is not found, the input file
/// doesn't exist, or the subprocess fails.
pub fn basecall(input: &Path, output_dir: &Path, config: &DoradoConfig) -> Result<DoradoResult> {
    if !input.exists() {
        return Err(Error::Nanopore(format!(
            "input file not found: {}",
            input.display()
        )));
    }

    let bin = config
        .binary
        .clone()
        .or_else(discover_dorado)
        .ok_or_else(|| Error::Nanopore("dorado binary not found".into()))?;

    std::fs::create_dir_all(output_dir).map_err(|e| {
        Error::Nanopore(format!(
            "cannot create output directory {}: {e}",
            output_dir.display()
        ))
    })?;

    let output_path = output_dir.join("basecalled.fastq");

    let mut cmd = Command::new(&bin);
    cmd.arg("basecaller").arg(config.model.as_str()).arg(input);

    if let Some(device) = &config.device {
        cmd.arg("--device").arg(device);
    }

    if config.emit_moves {
        cmd.arg("--emit-moves");
    }

    if let Some(min_q) = config.min_qscore {
        cmd.arg("--min-qscore").arg(min_q.to_string());
    }

    let output = cmd
        .output()
        .map_err(|e| Error::Nanopore(format!("dorado invocation failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Nanopore(format!(
            "dorado exited with {}: {stderr}",
            output.status
        )));
    }

    std::fs::write(&output_path, &output.stdout)
        .map_err(|e| Error::Nanopore(format!("failed to write output: {e}")))?;

    let (n_reads, total_bases) = count_fastq_reads(&output.stdout);

    Ok(DoradoResult {
        n_reads,
        total_bases,
        output_path,
        dorado_version: dorado_version().ok(),
    })
}

/// Parse basecalled reads from Dorado FASTQ output bytes.
#[must_use]
pub fn parse_basecalled_reads(fastq_bytes: &[u8]) -> Vec<BasecalledRead> {
    let text = String::from_utf8_lossy(fastq_bytes);
    let lines: Vec<&str> = text.lines().collect();
    let mut reads = Vec::new();

    let mut i = 0;
    while i + 3 < lines.len() {
        if let Some(header) = lines[i].strip_prefix('@') {
            let read_id = header
                .split_whitespace()
                .next()
                .unwrap_or(header)
                .to_string();
            let sequence = lines[i + 1].as_bytes().to_vec();
            let quality = lines[i + 3].as_bytes().to_vec();
            reads.push(BasecalledRead {
                read_id,
                sequence,
                quality,
            });
            i += 4;
        } else {
            i += 1;
        }
    }

    reads
}

fn count_fastq_reads(data: &[u8]) -> (usize, usize) {
    let text = String::from_utf8_lossy(data);
    let lines: Vec<&str> = text.lines().collect();
    let mut n_reads = 0;
    let mut total_bases = 0;

    let mut i = 0;
    while i + 1 < lines.len() {
        if lines[i].starts_with('@') {
            n_reads += 1;
            total_bases += lines[i + 1].len();
            i += 4;
        } else {
            i += 1;
        }
    }

    (n_reads, total_bases)
}

/// User home for `~/.local/bin` discovery: `$HOME`, `$USERPROFILE`, Linux
/// proc environ `HOME=`, else [`std::env::current_dir`]. [`None`] if unset.
fn dirs_home() -> Option<PathBuf> {
    if let Ok(h) = std::env::var("HOME") {
        if !h.is_empty() {
            return Some(PathBuf::from(h));
        }
    }
    if let Ok(h) = std::env::var("USERPROFILE") {
        if !h.is_empty() {
            return Some(PathBuf::from(h));
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(p) = home_from_proc_self_environ() {
            return Some(p);
        }
    }
    std::env::current_dir().ok()
}

#[cfg(target_os = "linux")]
fn home_from_proc_self_environ() -> Option<PathBuf> {
    let bytes = std::fs::read("/proc/self/environ").ok()?;
    for entry in bytes.split(|b| *b == 0) {
        let line = std::str::from_utf8(entry).ok()?;
        if let Some(rest) = line.strip_prefix("HOME=") {
            if !rest.is_empty() {
                return Some(PathBuf::from(rest));
            }
        }
    }
    None
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn discover_dorado_returns_option() {
        // Discovery is environment-dependent; verify it returns a valid
        // type without panicking (env mutation is unsafe in modern Rust).
        let _ = discover_dorado();
    }

    #[test]
    fn dorado_model_strings() {
        assert_eq!(DoradoModel::Fast.as_str(), "fast");
        assert_eq!(DoradoModel::Hac.as_str(), "hac");
        assert_eq!(DoradoModel::Sup.as_str(), "sup");
    }

    #[test]
    fn default_config() {
        let cfg = DoradoConfig::default();
        assert_eq!(cfg.model, DoradoModel::Hac);
        assert!(cfg.binary.is_none());
        assert!(cfg.device.is_none());
        assert!(!cfg.emit_moves);
        assert!(cfg.min_qscore.is_none());
    }

    #[test]
    fn parse_fastq_output() {
        let fastq = b"@read_001 runid=abc\nACGTACGT\n+\n!!!!!!!!!\n@read_002\nGGCC\n+\n####\n";
        let reads = parse_basecalled_reads(fastq);

        assert_eq!(reads.len(), 2);
        assert_eq!(reads[0].read_id, "read_001");
        assert_eq!(reads[0].sequence, b"ACGTACGT");
        assert_eq!(reads[1].read_id, "read_002");
        assert_eq!(reads[1].sequence, b"GGCC");
    }

    #[test]
    fn parse_empty_fastq() {
        let reads = parse_basecalled_reads(b"");
        assert!(reads.is_empty());
    }

    #[test]
    fn count_fastq() {
        let fastq = b"@r1\nACGT\n+\n!!!!\n@r2\nGGCC\n+\n####\n";
        let (n, bases) = count_fastq_reads(fastq);
        assert_eq!(n, 2);
        assert_eq!(bases, 8);
    }

    #[test]
    fn basecall_missing_input() {
        let cfg = DoradoConfig::default();
        let result = basecall(
            Path::new("/nonexistent/input_9z.pod5"),
            Path::new("/tmp"),
            &cfg,
        );
        assert!(result.is_err());
    }

    #[test]
    fn basecall_no_dorado_binary() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("fake.pod5");
        std::fs::write(&input, b"fake pod5 data").unwrap();

        let cfg = DoradoConfig {
            binary: Some(PathBuf::from("/nonexistent/dorado_9z8x7y")),
            ..DoradoConfig::default()
        };
        let result = basecall(&input, dir.path(), &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn is_available_check() {
        // Just verify it doesn't panic
        let _ = is_dorado_available();
    }
}
