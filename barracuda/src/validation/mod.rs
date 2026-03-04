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
/// Prints a formatted `[OK]` or `[FAIL]` line and returns whether
/// the check passed.
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
/// Separates logic from exit behavior for testability.
#[must_use]
pub fn print_result(name: &str, passed: u32, total: u32) -> bool {
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  {name}: {passed}/{total} checks passed");
    if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");
    passed == total
}

/// Print summary banner and exit with appropriate code.
///
/// Exit code 0 if all checks passed, 1 otherwise.
pub fn exit_with_result(name: &str, passed: u32, total: u32) {
    let ok = print_result(name, passed, total);
    std::process::exit(i32::from(!ok));
}

/// Exit with code 2 indicating the test was skipped (data unavailable).
pub fn exit_skipped(reason: &str) -> ! {
    println!("  SKIP: {reason}");
    println!("  (exit 2 = skipped, not a failure)");
    std::process::exit(2)
}

/// Discover the benchmark results directory via capability-based discovery.
///
/// Discovery order:
/// 1. `WETSPRING_BENCH_DIR` env var (explicit override)
/// 2. `{CARGO_MANIFEST_DIR}/../benchmarks/results` for development
/// 3. `benchmarks/results` relative to cwd for deployment
#[must_use]
pub fn discover_bench_dir() -> std::path::PathBuf {
    data_dir("WETSPRING_BENCH_DIR", "benchmarks/results")
}

/// Resolve a data directory using a cascading discovery strategy.
///
/// Implements capability-based discovery: primal code has no hardcoded paths and
/// discovers data at runtime via explicit configuration or environment.
///
/// # Discovery cascade (in order)
///
/// 1. **Explicit env var** — If `env_var` is set, use it. Overrides everything.
///    Use case: per-dataset override (e.g. `WETSPRING_FASTQ_DIR`).
///
/// 2. **General data root** — If `WETSPRING_DATA_ROOT` is set and
///    `WETSPRING_DATA_ROOT/{default_subpath}` exists, use it.
///    Use case: CI, shared data mounts.
///
/// 3. **Development fallback** — If `CARGO_MANIFEST_DIR/../{default_subpath}` exists,
///    use it. Use case: local `cargo run` from repo.
///
/// 4. **Deployment fallback** — Otherwise return `default_subpath` (relative to cwd).
///    Use case: standalone binaries run from a directory with data alongside.
///
/// # Example
///
/// ```text
/// let dir = data_dir("WETSPRING_FASTQ_DIR", "data/validation/MiSeq_SOP");
/// ```
#[must_use]
pub fn data_dir(env_var: &str, default_subpath: &str) -> std::path::PathBuf {
    let specific = std::env::var(env_var).ok();
    let data_root = std::env::var("WETSPRING_DATA_ROOT").ok();
    resolve_data_dir(specific.as_deref(), data_root.as_deref(), default_subpath)
}

/// Pure logic for data directory resolution — no global state access.
///
/// Takes pre-read environment values so it can be tested without mutating
/// process-wide environment variables (which is `unsafe` in edition 2024).
#[must_use]
pub fn resolve_data_dir(
    specific_override: Option<&str>,
    data_root: Option<&str>,
    default_subpath: &str,
) -> std::path::PathBuf {
    // 1. Specific override
    if let Some(dir) = specific_override {
        return std::path::PathBuf::from(dir);
    }
    // 2. General data root
    if let Some(root) = data_root {
        let p = std::path::Path::new(root).join(default_subpath);
        if p.exists() {
            return p;
        }
    }
    // 3. Development: relative to crate manifest
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(default_subpath);
    if manifest.exists() {
        return manifest;
    }
    // 4. Deployment: relative to cwd
    std::path::PathBuf::from(default_subpath)
}

// ── Benchmark helper ──────────────────────────────────────────

/// Run a closure and return `(result, elapsed_ms)`.
///
/// Shared helper for validation binaries that record per-domain timing.
#[must_use]
pub fn bench<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let result = f();
    let ms = start.elapsed().as_secs_f64() * 1e3;
    (result, ms)
}

/// Run a closure and return `(result, elapsed_microseconds)`.
///
/// Companion to [`bench()`] (which returns milliseconds). This gives µs
/// precision for per-domain timing tables in cross-spring validators.
#[must_use]
pub fn timed_us<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let result = f();
    #[allow(clippy::cast_precision_loss)]
    let us = start.elapsed().as_micros() as f64;
    (result, us)
}

/// Print a timing summary table with columns: Name, CPU µs, GPU µs, Status.
///
/// Accepts rows as `(name, cpu_us, gpu_us, status)` tuples.
pub fn print_timing_table(rows: &[(&str, f64, f64, &str)]) {
    println!(
        "\n  {:40} {:>12} {:>12} {:>8}",
        "Domain", "CPU (µs)", "GPU (µs)", "Status"
    );
    println!("  {}", "─".repeat(76));
    for &(name, cpu, gpu, status) in rows {
        println!("  {name:40} {cpu:12.0} {gpu:12.0} {status:>8}");
    }
}

// ── Validator: structured check accumulator ───────────────────

/// Accumulated validation state, removing manual pass/fail bookkeeping.
///
/// # Examples
///
/// ```
/// use wetspring_barracuda::validation::Validator;
///
/// let mut v = Validator::new("doc-test");
/// v.check("pi", std::f64::consts::PI, 3.14159, 1e-4);
/// v.check_count("records", 10, 10);
/// let (passed, total) = v.counts();
/// assert_eq!((passed, total), (2, 2));
/// ```
pub struct Validator {
    name: String,
    passed: u32,
    total: u32,
}

impl Validator {
    /// Create a new validator for the given binary name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        println!("═══════════════════════════════════════════════════════════");
        println!("  {name}");
        println!("═══════════════════════════════════════════════════════════\n");
        Self {
            name,
            passed: 0,
            total: 0,
        }
    }

    /// Print a section header (no check counted).
    pub fn section(&self, label: &str) {
        println!("\n{label}");
    }

    /// Check an f64 value against expected within tolerance.
    pub fn check(&mut self, label: &str, actual: f64, expected: f64, tolerance: f64) {
        self.total += 1;
        if check(label, actual, expected, tolerance) {
            self.passed += 1;
        }
    }

    /// Check a boolean condition (pass = 1.0, fail = 0.0).
    pub fn check_pass(&mut self, label: &str, pass: bool) {
        let actual = if pass { 1.0 } else { 0.0 };
        self.check(label, actual, 1.0, 0.0);
    }

    /// Check an exact count (`usize`) — no floating-point conversion.
    pub fn check_count(&mut self, label: &str, actual: usize, expected: usize) {
        self.total += 1;
        if check_count(label, actual, expected) {
            self.passed += 1;
        }
    }

    /// Check an exact count (`u64`) — no floating-point conversion.
    pub fn check_count_u64(&mut self, label: &str, actual: u64, expected: u64) {
        self.total += 1;
        let pass = actual == expected;
        let tag = if pass { "OK" } else { "FAIL" };
        println!("  [{tag}]  {label}: {actual} (expected {expected})");
        if pass {
            self.passed += 1;
        }
    }

    /// Retrieve current (passed, total) for external logic.
    #[must_use]
    pub const fn counts(&self) -> (u32, u32) {
        (self.passed, self.total)
    }

    /// Print summary and exit with 0 (pass) or 1 (fail).
    pub fn finish(self) -> ! {
        let ok = print_result(&self.name, self.passed, self.total);
        std::process::exit(i32::from(!ok))
    }
}

#[cfg(test)]
mod tests;
