// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation framework for Python-baseline comparison.
//!
//! Used by validation binaries (`validate_fastq`, `validate_mzml`, etc.)
//! to compare Rust results against documented Python baselines.
//! Each check prints a formatted pass/fail line with the actual value,
//! the expected baseline, and the tolerance applied.
//!
//! # hotSpring pattern
//!
//! Every validation binary follows the same contract:
//! - Hardcoded expected values sourced from documented Python runs
//! - Explicit pass/fail per check with human-readable output
//! - Exit code 0 = all passed, 1 = at least one failed, 2 = skipped
//!
//! # Usage
//!
//! Prefer the [`Validator`] struct over bare [`check`] calls — it
//! tracks pass/fail counts automatically and avoids manual bookkeeping.
//! Route output with [`ValidationSink`] ([`StdoutSink`] by default, [`SilentSink`],
//! or [`CollectingSink`]) when you need programmatic consumption instead of stdout.

mod data_dir;
mod domain;
mod harness;
mod or_exit;
pub mod sink;
pub mod test_data;
mod timing;

pub use data_dir::{data_dir, discover_bench_dir, resolve_data_dir};
pub use domain::{DomainResult, print_domain_summary};
pub use harness::Validator;
pub use or_exit::OrExit;
pub use sink::{
    CheckAbsOrRelResult, CheckResult, CollectingSink, SilentSink, StdoutSink, ValidationSink,
};
pub use timing::{bench, bench_n_us, print_timing_table, timed_us};

// ── Standalone helpers (for one-off use) ──────────────────────

/// Compare `actual` against `expected` within absolute `tolerance`.
///
/// Prints a formatted `[OK]` or `[FAIL]` line and returns whether
/// the check passed. Tolerance of `0.0` requires exact match.
///
/// ```
/// use wetspring_barracuda::validation::check;
///
/// assert!(check("Shannon(uniform,4)", 4.0_f64.ln(), 4.0_f64.ln(), 1e-12));
/// assert!(!check("deliberate fail", 2.0, 1.0, 0.5));
/// ```
#[must_use]
pub fn check(label: &str, actual: f64, expected: f64, tolerance: f64) -> bool {
    let pass = (actual - expected).abs() <= tolerance;
    let tag = if pass { "OK" } else { "FAIL" };
    println!("  [{tag}]  {label}: {actual:.6} (expected {expected:.6}, tol {tolerance:.6})");
    pass
}

/// Compare an exact count — no floating-point conversion needed.
///
/// # Examples
///
/// ```
/// use wetspring_barracuda::validation::check_count;
///
/// assert!(check_count("record count", 42, 42));
/// assert!(!check_count("mismatched", 10, 20));
/// ```
#[must_use]
pub fn check_count(label: &str, actual: usize, expected: usize) -> bool {
    let pass = actual == expected;
    let tag = if pass { "OK" } else { "FAIL" };
    println!("  [{tag}]  {label}: {actual} (expected {expected})");
    pass
}

/// Print summary and return whether all checks passed.
///
/// Returns `false` when `total == 0` — a validator that runs no checks
/// is a failure (prevents silent pass-through when data is missing).
#[must_use]
pub fn print_result(name: &str, passed: u32, total: u32) -> bool {
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  {name}: {passed}/{total} checks passed");
    if total == 0 {
        println!("  RESULT: FAIL (no checks executed)");
    } else if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");
    total > 0 && passed == total
}

/// Print summary banner and exit with appropriate code.
///
/// Exit code 0 if all checks passed, 1 otherwise.
///
/// Prefer [`print_result`] + `ExitCode` for new binaries.
pub fn exit_with_result(name: &str, passed: u32, total: u32) {
    let ok = print_result(name, passed, total);
    std::process::exit(i32::from(!ok));
}

/// Exit code for skipped validations (data unavailable, no GPU, etc.).
///
/// Use `return SKIP.into()` from `fn main() -> ExitCode` instead of
/// the diverging `exit_skipped()`.
pub const SKIP_CODE: u8 = 2;

/// Print skip reason and exit with code 2 (skipped — not a failure).
///
/// For new binaries using `fn main() -> ExitCode`, prefer
/// [`skip_with_code`] which returns `ExitCode` without calling
/// `process::exit`.
pub fn exit_skipped(reason: &str) -> ! {
    println!("  SKIP: {reason}");
    println!("  (exit 2 = skipped, not a failure)");
    std::process::exit(2)
}

/// Print skip reason and return `ExitCode` without calling `process::exit`.
///
/// Composable alternative to [`exit_skipped`] for the `fn main() -> ExitCode`
/// pattern.
#[must_use]
pub fn skip_with_code(reason: &str) -> std::process::ExitCode {
    println!("  SKIP: {reason}");
    println!("  (exit 2 = skipped, not a failure)");
    std::process::ExitCode::from(SKIP_CODE)
}

/// Initialize GPU and exit with code 2 if unavailable or lacking f64.
///
/// Centralizes the GPU init + skip pattern used by 50+ validation binaries.
/// Prints device info on success.
///
/// # Panics
///
/// Never — exits with code 2 on any failure.
#[cfg(feature = "gpu")]
pub async fn gpu_or_skip() -> crate::gpu::GpuF64 {
    let gpu = match crate::gpu::GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        exit_skipped("No SHADER_F64 support on this GPU");
    }
    gpu
}

#[cfg(test)]
mod tests;
