// SPDX-License-Identifier: AGPL-3.0-or-later
//! SRA (Sequence Read Archive) download via capability-discovered tools.
//!
//! Downloads FASTQ files from NCBI SRA using a capability-based tool chain:
//!
//! 1. **`fasterq-dump`** — SRA Toolkit (fastest, splits paired-end)
//! 2. **`fastq-dump`** — SRA Toolkit legacy (wider availability)
//! 3. **`EFetch` fallback** — HTTP download via [`super::efetch`] (limited formats)
//!
//! The tool chain mirrors [`super::http`]'s capability-discovered pattern:
//! discover what's available at runtime, use the best option, degrade
//! gracefully.
//!
//! # Evolution path
//!
//! | Phase | Strategy | Status |
//! |-------|----------|--------|
//! | Current | Capability-discovered `fasterq-dump` / `fastq-dump` | active |
//! | Phase 2 | NestGate SRA provider via JSON-RPC socket | planned |
//! | Phase 3 | Sovereign SRA protocol (direct HTTP range requests) | research |

use crate::error::Error;
use std::path::{Path, PathBuf};

/// Discovered SRA download backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SraBackend {
    FasterqDump,
    FastqDump,
}

/// Download a FASTQ file from SRA by run accession (e.g. `SRR1234567`).
///
/// Output is written to `output_dir/{accession}.fastq` (or split files
/// for paired-end data when using `fasterq-dump`).
///
/// # Errors
///
/// Returns `Err` if no SRA tool is found on `$PATH`, the download fails,
/// or the accession is invalid.
pub fn download_sra_run(accession: &str, output_dir: &Path) -> crate::error::Result<PathBuf> {
    validate_accession(accession)?;

    std::fs::create_dir_all(output_dir).map_err(|e| {
        Error::Ncbi(format!(
            "cannot create output dir {}: {e}",
            output_dir.display()
        ))
    })?;

    let backend = discover_sra_backend().ok_or_else(|| {
        Error::Ncbi(
            "no SRA download tool available (need fasterq-dump or fastq-dump on PATH)".to_string(),
        )
    })?;

    let output_path = output_dir.join(format!("{accession}.fastq"));

    match backend {
        SraBackend::FasterqDump => run_fasterq_dump(accession, output_dir, &output_path),
        SraBackend::FastqDump => run_fastq_dump(accession, &output_path),
    }
}

/// Check whether any SRA download tool is available.
#[must_use]
pub fn sra_tools_available() -> bool {
    discover_sra_backend().is_some()
}

/// Discover the best available SRA download tool.
fn discover_sra_backend() -> Option<SraBackend> {
    select_sra_backend(which_exists("fasterq-dump"), which_exists("fastq-dump"))
}

/// Pure-logic backend selection — no env or filesystem access.
const fn select_sra_backend(has_fasterq: bool, has_fastq: bool) -> Option<SraBackend> {
    if has_fasterq {
        Some(SraBackend::FasterqDump)
    } else if has_fastq {
        Some(SraBackend::FastqDump)
    } else {
        None
    }
}

/// Validate that a string looks like an SRA run accession.
fn validate_accession(accession: &str) -> crate::error::Result<()> {
    let trimmed = accession.trim();
    if trimmed.is_empty() {
        return Err(Error::Ncbi("empty SRA accession".to_string()));
    }
    if trimmed.len() < 6 {
        return Err(Error::Ncbi(format!("SRA accession too short: {trimmed}")));
    }
    let prefix = &trimmed[..3];
    let valid_prefixes = [
        "SRR", "ERR", "DRR", "SRX", "ERX", "DRX", "SRP", "ERP", "DRP",
    ];
    if !valid_prefixes
        .iter()
        .any(|p| prefix.eq_ignore_ascii_case(p))
    {
        return Err(Error::Ncbi(format!(
            "unrecognized SRA accession prefix '{prefix}' (expected SRR/ERR/DRR/SRX/ERX/DRX/SRP/ERP/DRP)"
        )));
    }
    Ok(())
}

/// Run `fasterq-dump` to download a run.
fn run_fasterq_dump(
    accession: &str,
    output_dir: &Path,
    output_path: &Path,
) -> crate::error::Result<PathBuf> {
    let output = std::process::Command::new("fasterq-dump")
        .args([
            "--outdir",
            &output_dir.to_string_lossy(),
            "--split-3",
            "--skip-technical",
            "--threads",
            "4",
            accession,
        ])
        .output()
        .map_err(|e| Error::Ncbi(format!("fasterq-dump: {e}")))?;

    if output.status.success() {
        if output_path.exists() {
            Ok(output_path.to_path_buf())
        } else {
            let paired_1 = output_dir.join(format!("{accession}_1.fastq"));
            if paired_1.exists() {
                Ok(paired_1)
            } else {
                Err(Error::Ncbi(format!(
                    "fasterq-dump succeeded but output not found at {}",
                    output_path.display()
                )))
            }
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let limit = stderr.len().min(crate::tolerances::ERROR_BODY_PREVIEW_LEN);
        Err(Error::Ncbi(format!(
            "fasterq-dump failed (exit {:?}): {}",
            output.status.code(),
            &stderr[..limit]
        )))
    }
}

/// Run `fastq-dump` to download a run.
fn run_fastq_dump(accession: &str, output_path: &Path) -> crate::error::Result<PathBuf> {
    let parent = output_path
        .parent()
        .ok_or_else(|| Error::Ncbi("invalid output path".to_string()))?;

    let output = std::process::Command::new("fastq-dump")
        .args([
            "--outdir",
            &parent.to_string_lossy(),
            "--skip-technical",
            "--split-3",
            accession,
        ])
        .output()
        .map_err(|e| Error::Ncbi(format!("fastq-dump: {e}")))?;

    if output.status.success() {
        if output_path.exists() {
            Ok(output_path.to_path_buf())
        } else {
            let alt = parent.join(format!("{accession}_1.fastq"));
            if alt.exists() {
                Ok(alt)
            } else {
                Err(Error::Ncbi(format!(
                    "fastq-dump succeeded but output not found at {}",
                    output_path.display()
                )))
            }
        }
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let limit = stderr.len().min(crate::tolerances::ERROR_BODY_PREVIEW_LEN);
        Err(Error::Ncbi(format!(
            "fastq-dump failed (exit {:?}): {}",
            output.status.code(),
            &stderr[..limit]
        )))
    }
}

/// Check whether a command exists on `$PATH`.
fn which_exists(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(cmd).is_file()))
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn validate_accession_valid_srr() {
        assert!(validate_accession("SRR1234567").is_ok());
    }

    #[test]
    fn validate_accession_valid_err() {
        assert!(validate_accession("ERR999888").is_ok());
    }

    #[test]
    fn validate_accession_valid_drr() {
        assert!(validate_accession("DRR000001").is_ok());
    }

    #[test]
    fn validate_accession_valid_project() {
        assert!(validate_accession("SRP123456").is_ok());
    }

    #[test]
    fn validate_accession_valid_experiment() {
        assert!(validate_accession("SRX654321").is_ok());
    }

    #[test]
    fn validate_accession_empty() {
        let err = validate_accession("").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn validate_accession_too_short() {
        let err = validate_accession("SRR12").unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn validate_accession_bad_prefix() {
        let err = validate_accession("ABC123456").unwrap_err();
        assert!(err.to_string().contains("unrecognized"));
    }

    #[test]
    fn validate_accession_lowercase_ok() {
        assert!(validate_accession("srr1234567").is_ok());
    }

    #[test]
    fn validate_accession_whitespace_trimmed() {
        assert!(validate_accession("  SRR1234567  ").is_ok());
    }

    #[test]
    fn select_sra_backend_fasterq_preferred() {
        let backend = select_sra_backend(true, true);
        assert_eq!(backend, Some(SraBackend::FasterqDump));
    }

    #[test]
    fn select_sra_backend_fastq_fallback() {
        let backend = select_sra_backend(false, true);
        assert_eq!(backend, Some(SraBackend::FastqDump));
    }

    #[test]
    fn select_sra_backend_none() {
        assert!(select_sra_backend(false, false).is_none());
    }

    #[test]
    fn sra_tools_available_does_not_panic() {
        let _ = sra_tools_available();
    }

    #[test]
    fn download_sra_run_empty_accession() {
        let dir = tempfile::tempdir().unwrap();
        let err = download_sra_run("", dir.path()).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn download_sra_run_bad_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let err = download_sra_run("ZZZZZZZZZ", dir.path()).unwrap_err();
        assert!(err.to_string().contains("unrecognized"));
    }

    #[test]
    fn download_sra_run_too_short() {
        let dir = tempfile::tempdir().unwrap();
        let err = download_sra_run("SRR12", dir.path()).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn validate_accession_erx_drx_erp_drp() {
        assert!(validate_accession("ERX123456").is_ok());
        assert!(validate_accession("DRX654321").is_ok());
        assert!(validate_accession("ERP111222").is_ok());
        assert!(validate_accession("DRP999888").is_ok());
    }

    #[test]
    fn output_filename_format() {
        let accession = "SRR1234567";
        let filename = format!("{accession}.fastq");
        assert_eq!(filename, "SRR1234567.fastq");
    }

    #[test]
    fn which_exists_finds_sh() {
        assert!(which_exists("sh"));
    }

    #[test]
    fn which_exists_misses_nonexistent() {
        assert!(!which_exists("wetspring_nonexistent_tool_abc123"));
    }

    #[test]
    fn download_sra_run_creates_output_dir() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("subdir");
        assert!(!sub.exists());
        let _ = download_sra_run("SRR000001", &sub);
        assert!(sub.exists());
    }

    #[test]
    fn validate_accession_mixed_case() {
        assert!(validate_accession("sRr1234567").is_ok());
        assert!(validate_accession("Err999888").is_ok());
    }

    #[test]
    fn validate_accession_whitespace_only() {
        let err = validate_accession("   ").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }
}
