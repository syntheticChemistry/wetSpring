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
/// Returns `false` when `total == 0` — a validator that runs no checks
/// is a failure (prevents silent pass-through when data is missing).
/// Separates logic from exit behavior for testability.
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

// ── ValidationSink: programmatic output routing ─────────────

/// One float [`Validator::check`] outcome for inspection (e.g. [`CollectingSink`]).
#[derive(Clone, Debug, PartialEq)]
pub struct CheckResult {
    /// Human-readable check label.
    pub label: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Observed value.
    pub actual: f64,
    /// Baseline or expected value.
    pub expected: f64,
    /// Absolute tol. ([`Validator::check`]) or relative fraction ([`Validator::check_relative`]).
    pub tolerance: f64,
}

/// Row from [`Validator::check_abs_or_rel`] (absolute and relative tolerances).
#[derive(Clone, Debug, PartialEq)]
pub struct CheckAbsOrRelResult {
    /// Human-readable check label.
    pub label: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Observed value.
    pub actual: f64,
    /// Baseline or expected value.
    pub expected: f64,
    /// Absolute tolerance bound.
    pub abs_tolerance: f64,
    /// Relative tolerance (fraction of `expected`).
    pub rel_tolerance: f64,
}

/// Receives validation events so results can be routed to stdout, a buffer, or nowhere.
///
/// Used by [`Validator`] for CI, biomeOS integration, and tests without coupling to `println!`.
pub trait ValidationSink: Send {
    /// A single floating-point tolerance check completed.
    fn on_check(&mut self, label: &str, passed: bool, actual: f64, expected: f64, tolerance: f64);

    /// A single exact count check completed.
    fn on_check_count(&mut self, label: &str, passed: bool, actual: usize, expected: usize);

    /// A single relative-tolerance float check completed ([`Validator::check_relative`]).
    fn on_check_relative(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        tolerance: f64,
    );

    /// A single absolute-or-relative float check completed ([`Validator::check_abs_or_rel`]).
    fn on_check_abs_or_rel(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        abs_tol: f64,
        rel_tol: f64,
    );

    /// A section header (not counted as a check).
    fn on_section(&mut self, label: &str);

    /// Validation run finished; `success` matches [`print_result`] semantics (`total > 0` and all passed).
    fn on_finish(&mut self, name: &str, passed: u32, total: u32, success: bool);
}

/// [`ValidationSink`] that prints the same lines as the standalone [`check`], [`check_count`], and [`print_result`] helpers.
#[derive(Clone, Copy, Debug, Default)]
pub struct StdoutSink;

impl ValidationSink for StdoutSink {
    fn on_check(&mut self, label: &str, passed: bool, actual: f64, expected: f64, tolerance: f64) {
        let tag = if passed { "OK" } else { "FAIL" };
        println!("  [{tag}]  {label}: {actual:.6} (expected {expected:.6}, tol {tolerance:.6})");
    }

    fn on_check_count(&mut self, label: &str, passed: bool, actual: usize, expected: usize) {
        let tag = if passed { "OK" } else { "FAIL" };
        println!("  [{tag}]  {label}: {actual} (expected {expected})");
    }

    fn on_check_relative(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        tolerance: f64,
    ) {
        let tag = if passed { "OK" } else { "FAIL" };
        println!(
            "  [{tag}]  {label}: {actual:.6} (expected {expected:.6}, rel_tol {tolerance:.6})"
        );
    }

    fn on_check_abs_or_rel(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) {
        let tag = if passed { "OK" } else { "FAIL" };
        println!(
            "  [{tag}]  {label}: {actual:.6} (expected {expected:.6}, abs_tol {abs_tol:.6}, rel_tol {rel_tol:.6})"
        );
    }

    fn on_section(&mut self, label: &str) {
        println!("\n{label}");
    }

    fn on_finish(&mut self, name: &str, passed: u32, total: u32, success: bool) {
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
        let _ = success;
    }
}

/// [`ValidationSink`] that discards all events (programmatic runs with no console noise).
#[derive(Clone, Copy, Debug, Default)]
pub struct SilentSink;

impl ValidationSink for SilentSink {
    fn on_check(&mut self, _: &str, _: bool, _: f64, _: f64, _: f64) {}

    fn on_check_count(&mut self, _: &str, _: bool, _: usize, _: usize) {}

    fn on_check_relative(&mut self, _: &str, _: bool, _: f64, _: f64, _: f64) {}

    fn on_check_abs_or_rel(&mut self, _: &str, _: bool, _: f64, _: f64, _: f64, _: f64) {}

    fn on_section(&mut self, _: &str) {}

    fn on_finish(&mut self, _: &str, _: u32, _: u32, _: bool) {}
}

/// Collects [`Validator`] float checks into shared [`Vec`]s ([`std::sync::Arc`] / [`std::sync::Mutex`]).
#[derive(Clone, Debug, Default)]
pub struct CollectingSink {
    /// Captured float checks ([`ValidationSink::on_check`] only).
    pub results: std::sync::Arc<std::sync::Mutex<Vec<CheckResult>>>,
    /// Relative checks ([`ValidationSink::on_check_relative`]); [`CheckResult::tolerance`] is the relative fraction.
    pub relative_results: std::sync::Arc<std::sync::Mutex<Vec<CheckResult>>>,
    /// Captured abs-or-rel checks ([`ValidationSink::on_check_abs_or_rel`]).
    pub abs_or_rel_results: std::sync::Arc<std::sync::Mutex<Vec<CheckAbsOrRelResult>>>,
}

impl ValidationSink for CollectingSink {
    fn on_check(&mut self, label: &str, passed: bool, actual: f64, expected: f64, tolerance: f64) {
        if let Ok(mut guard) = self.results.lock() {
            guard.push(CheckResult {
                label: label.to_string(),
                passed,
                actual,
                expected,
                tolerance,
            });
        }
    }

    fn on_check_count(&mut self, _label: &str, _passed: bool, _actual: usize, _expected: usize) {}

    fn on_check_relative(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        tolerance: f64,
    ) {
        if let Ok(mut guard) = self.relative_results.lock() {
            guard.push(CheckResult {
                label: label.to_string(),
                passed,
                actual,
                expected,
                tolerance,
            });
        }
    }

    fn on_check_abs_or_rel(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) {
        if let Ok(mut guard) = self.abs_or_rel_results.lock() {
            guard.push(CheckAbsOrRelResult {
                label: label.to_string(),
                passed,
                actual,
                expected,
                abs_tolerance: abs_tol,
                rel_tolerance: rel_tol,
            });
        }
    }

    fn on_section(&mut self, _label: &str) {}

    fn on_finish(&mut self, _name: &str, _passed: u32, _total: u32, _success: bool) {}
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
    sink: Box<dyn ValidationSink>,
}

impl Validator {
    /// Create a new validator for the given binary name.
    ///
    /// Uses [`StdoutSink`] and prints the opening banner (same behavior as before [`ValidationSink`]).
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
            sink: Box::new(StdoutSink),
        }
    }

    /// Create a validator with a custom [`ValidationSink`] (no opening banner).
    #[must_use]
    pub fn with_sink(name: impl Into<String>, sink: Box<dyn ValidationSink>) -> Self {
        let name = name.into();
        Self {
            name,
            passed: 0,
            total: 0,
            sink,
        }
    }

    /// Shorthand for [`Self::with_sink`] with [`SilentSink`].
    #[must_use]
    pub fn silent(name: impl Into<String>) -> Self {
        Self::with_sink(name, Box::new(SilentSink))
    }

    /// Print a section header (no check counted).
    pub fn section(&mut self, label: &str) {
        self.sink.on_section(label);
    }

    /// Check an f64 value against expected within tolerance.
    pub fn check(&mut self, label: &str, actual: f64, expected: f64, tolerance: f64) {
        self.total += 1;
        let pass = (actual - expected).abs() <= tolerance;
        self.sink.on_check(label, pass, actual, expected, tolerance);
        if pass {
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
        let pass = actual == expected;
        self.sink.on_check_count(label, pass, actual, expected);
        if pass {
            self.passed += 1;
        }
    }

    /// Check an exact count (`u64`) — no floating-point conversion.
    pub fn check_count_u64(&mut self, label: &str, actual: u64, expected: u64) {
        self.total += 1;
        let pass = actual == expected;
        let a = usize::try_from(actual).unwrap_or(usize::MAX);
        let e = usize::try_from(expected).unwrap_or(usize::MAX);
        self.sink.on_check_count(label, pass, a, e);
        if pass {
            self.passed += 1;
        }
    }

    /// Check `actual` against `expected` with relative tolerance:
    /// `|(actual - expected) / expected| <= tolerance`.
    ///
    /// When `expected == 0.0`, uses `|actual - expected| <= tolerance` instead (absolute bound;
    /// avoids division by zero).
    pub fn check_relative(&mut self, label: &str, actual: f64, expected: f64, tolerance: f64) {
        self.total += 1;
        let pass = if expected == 0.0 {
            (actual - expected).abs() <= tolerance
        } else {
            ((actual - expected) / expected).abs() <= tolerance
        };
        self.sink
            .on_check_relative(label, pass, actual, expected, tolerance);
        if pass {
            self.passed += 1;
        }
    }

    /// Passes if **either** `|actual - expected| <= abs_tol` **or**
    /// `|(actual - expected) / expected| <= rel_tol` (when `expected != 0.0`).
    ///
    /// When `expected == 0.0`, only the absolute branch applies.
    pub fn check_abs_or_rel(
        &mut self,
        label: &str,
        actual: f64,
        expected: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) {
        self.total += 1;
        let pass_abs = (actual - expected).abs() <= abs_tol;
        let pass_rel = if expected == 0.0 {
            false
        } else {
            ((actual - expected) / expected).abs() <= rel_tol
        };
        let pass = pass_abs || pass_rel;
        self.sink
            .on_check_abs_or_rel(label, pass, actual, expected, abs_tol, rel_tol);
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
    pub fn finish(mut self) -> ! {
        let success = self.total > 0 && self.passed == self.total;
        self.sink
            .on_finish(&self.name, self.passed, self.total, success);
        std::process::exit(i32::from(!success))
    }

    /// Print summary and return `ExitCode` without calling `process::exit`.
    ///
    /// Prefer this over [`finish`](Self::finish) in binaries that use the
    /// `fn main() -> ExitCode` + `fn run()` zero-panic pattern.
    #[must_use]
    pub fn finish_with_code(mut self) -> std::process::ExitCode {
        let success = self.total > 0 && self.passed == self.total;
        self.sink
            .on_finish(&self.name, self.passed, self.total, success);
        if success {
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

/// Zero-panic error handling: [`Result::unwrap`]/[`Option::expect`] replacement via stderr + exit 1.
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

/// Per-domain timing row for [`print_domain_summary`].
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

/// Box-drawing domain summary table; optional Spring column when any row sets [`DomainResult::spring`].
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
