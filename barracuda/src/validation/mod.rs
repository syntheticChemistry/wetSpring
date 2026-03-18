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
    #[expect(
        clippy::cast_precision_loss,
        reason = "precision: microsecond timing fits f64"
    )]
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

    /// Print summary and return `ExitCode` without calling `process::exit`.
    ///
    /// Prefer this over [`finish`](Self::finish) in binaries that use the
    /// `fn main() -> ExitCode` + `fn run()` zero-panic pattern.
    #[must_use]
    pub fn finish_with_code(self) -> std::process::ExitCode {
        let ok = print_result(&self.name, self.passed, self.total);
        if ok {
            std::process::ExitCode::SUCCESS
        } else {
            std::process::ExitCode::FAILURE
        }
    }

    /// Returns `true` if every check so far has passed.
    #[must_use]
    pub const fn all_passed(&self) -> bool {
        self.passed == self.total
    }
}

/// Zero-panic error handling for validation infrastructure.
///
/// Replaces `.expect("msg")` and `.unwrap()` in validation binaries with
/// clean stderr + `process::exit(1)` — no panic, deterministic exit code.
/// Follows the groundSpring V109 zero-panic validation pattern.
///
/// # Examples
///
/// ```
/// use wetspring_barracuda::validation::OrExit;
///
/// let val: Result<i32, &str> = Ok(42);
/// assert_eq!(val.or_exit("should not fail"), 42);
///
/// let opt: Option<i32> = Some(7);
/// assert_eq!(opt.or_exit("should not be None"), 7);
/// ```
pub trait OrExit<T> {
    /// Unwrap or print to stderr and `process::exit(1)`.
    fn or_exit(self, context: &str) -> T;
}

impl<T, E: std::fmt::Display> OrExit<T> for Result<T, E> {
    fn or_exit(self, context: &str) -> T {
        match self {
            Ok(v) => v,
            Err(e) => {
                eprintln!("FATAL: {context}: {e}");
                std::process::exit(1)
            }
        }
    }
}

impl<T> OrExit<T> for Option<T> {
    #[expect(
        clippy::option_if_let_else,
        reason = "explicit if-let is clearer for a fatal-exit path"
    )]
    fn or_exit(self, context: &str) -> T {
        if let Some(v) = self {
            v
        } else {
            eprintln!("FATAL: {context}");
            std::process::exit(1)
        }
    }
}

pub mod test_data;

/// Domain timing result for multi-domain validators.
///
/// Shared struct replacing duplicated `DomainResult` definitions across
/// S79+ validation binaries (`validate_barrier_disruption_s79`,
/// `validate_skin_anderson_s79`, etc.).
#[derive(Debug)]
pub struct DomainResult {
    /// Domain or section name.
    pub name: &'static str,
    /// Optional originating spring (cross-spring validators).
    pub spring: Option<&'static str>,
    /// Elapsed time in milliseconds.
    pub ms: f64,
    /// Number of validation checks in this domain.
    pub checks: u32,
}

/// Print a formatted domain summary table.
///
/// Renders a box-drawing table with optional Spring column for cross-spring
/// validators. Totals are computed from the supplied slice.
pub fn print_domain_summary(title: &str, domains: &[DomainResult]) {
    let has_spring = domains.iter().any(|d| d.spring.is_some());
    let mut total_checks: u32 = 0;
    let mut total_ms: f64 = 0.0;

    println!("\n╔════════════════════════════════════════════════════════════════════╗");
    println!("║  {title:<64} ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");

    if has_spring {
        println!(
            "║ {:<22} │ {:<18} │ {:>7} │ {:>3} ║",
            "Domain", "Spring", "Time", "✓"
        );
    } else {
        println!("║ {:<22} │ {:>7} │ {:>3} ║", "Domain", "Time", "✓");
    }

    println!("╠════════════════════════════════════════════════════════════════════╣");

    for d in domains {
        total_checks += d.checks;
        total_ms += d.ms;
        if has_spring {
            let spring = d.spring.unwrap_or("—");
            println!(
                "║ {:<22} │ {:<18} │ {:>5.1}ms │ {:>3} ║",
                d.name, spring, d.ms, d.checks
            );
        } else {
            println!("║ {:<22} │ {:>5.1}ms │ {:>3} ║", d.name, d.ms, d.checks);
        }
    }

    println!("╠════════════════════════════════════════════════════════════════════╣");
    if has_spring {
        println!(
            "║ {:<22} │ {:<18} │ {:>5.1}ms │ {:>3} ║",
            "TOTAL", "", total_ms, total_checks
        );
    } else {
        println!(
            "║ {:<22} │ {:>5.1}ms │ {:>3} ║",
            "TOTAL", total_ms, total_checks
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests;
