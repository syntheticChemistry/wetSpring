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
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn check_exact_match() {
        assert!(check("exact", 42.0, 42.0, 0.0));
    }

    #[test]
    fn check_within_tolerance() {
        assert!(check("close", 42.001, 42.0, 0.01));
    }

    #[test]
    fn check_outside_tolerance() {
        assert!(!check("far", 50.0, 42.0, 1.0));
    }

    #[test]
    fn check_count_exact() {
        assert!(check_count("exact", 42, 42));
    }

    #[test]
    fn check_count_mismatch() {
        assert!(!check_count("diff", 42, 43));
    }

    #[test]
    fn print_result_pass() {
        assert!(print_result("test", 3, 3));
    }

    #[test]
    fn print_result_fail() {
        assert!(!print_result("test", 2, 3));
    }

    #[test]
    fn validator_accumulates() {
        let mut v = Validator {
            name: String::from("test"),
            passed: 0,
            total: 0,
        };
        v.check("ok", 1.0, 1.0, 0.0);
        v.check("fail", 2.0, 1.0, 0.0);
        v.check_count("count_ok", 5, 5);
        v.check_count("count_fail", 4, 5);
        assert_eq!(v.counts(), (2, 4));
    }

    #[test]
    fn validator_section_does_not_count() {
        let v = Validator {
            name: String::from("test"),
            passed: 0,
            total: 0,
        };
        v.section("── some section ──");
        assert_eq!(v.counts(), (0, 0));
    }

    #[test]
    fn validator_new_prints_banner() {
        let v = Validator::new("My Test Suite");
        assert_eq!(v.counts(), (0, 0));
    }

    #[test]
    fn validator_check_count_u64_pass_and_fail() {
        let mut v = Validator::new("u64 test");
        v.check_count_u64("exact", 42, 42);
        v.check_count_u64("off", 41, 42);
        assert_eq!(v.counts(), (1, 2));
    }

    #[test]
    fn validator_check_pass() {
        let mut v = Validator::new("check_pass test");
        v.check_pass("pass", true);
        v.check_pass("fail", false);
        assert_eq!(v.counts(), (1, 2));
    }

    #[test]
    fn validator_full_workflow() {
        let mut v = Validator::new("integration");
        v.section("── section A ──");
        v.check("float ok", 1.0, 1.0, 0.0);
        v.check("float fail", 2.0, 1.0, 0.0);
        v.section("── section B ──");
        v.check_count("count ok", 5, 5);
        v.check_count("count fail", 4, 5);
        v.check_count_u64("u64 ok", 100, 100);
        assert_eq!(v.counts(), (3, 5));
    }

    // ── resolve_data_dir: pure function tests (zero unsafe) ──────

    #[test]
    fn resolve_specific_override_wins() {
        let dir = resolve_data_dir(Some("/explicit/path"), Some("/root"), "data/default");
        assert_eq!(dir.to_string_lossy(), "/explicit/path");
    }

    #[test]
    fn resolve_data_root_with_existing_subpath() {
        let tmp = tempfile::tempdir().unwrap();
        let subpath = "resolve_test/data";
        let full = tmp.path().join(subpath);
        std::fs::create_dir_all(&full).unwrap();

        let root = tmp.path().to_string_lossy().to_string();
        let dir = resolve_data_dir(None, Some(&root), subpath);
        assert_eq!(dir, full);
    }

    #[test]
    fn resolve_data_root_nonexistent_falls_through() {
        let dir = resolve_data_dir(None, Some("/nonexistent_root"), "sub/path");
        let s = dir.to_string_lossy();
        assert!(s.contains("sub/path"));
    }

    #[test]
    fn resolve_no_env_uses_cwd_fallback() {
        let subpath = "___nonexistent_resolve_test/data";
        let dir = resolve_data_dir(None, None, subpath);
        assert_eq!(dir.to_string_lossy(), subpath);
    }

    #[test]
    fn resolve_nested_subpath() {
        let dir = resolve_data_dir(None, None, "a/b/c/d/e");
        let s = dir.to_string_lossy();
        assert!(s.contains("a/b/c/d/e"));
    }

    // ── data_dir: integration (reads real env, no mutation) ──────

    #[test]
    fn data_dir_fallback_uses_manifest() {
        let dir = data_dir("WETSPRING_NONEXISTENT_12345", "data/test");
        let s = dir.to_string_lossy();
        assert!(s.contains("data/test"), "path should contain subpath");
    }

    #[test]
    fn data_dir_env_override() {
        let key = "WETSPRING_TEST_DATA_DIR_UNIT";
        let dir = data_dir(key, "data/default");
        let s = dir.to_string_lossy();
        assert!(
            s.contains("data/default"),
            "fallback path should contain subpath"
        );
    }

    // ── check() edge cases ──────────────────────────────────────

    #[test]
    fn check_nan_always_fails() {
        assert!(!check("NaN test", f64::NAN, 0.0, 1.0));
        assert!(!check("NaN expected", 0.0, f64::NAN, 1.0));
    }

    #[test]
    fn check_infinity_values() {
        assert!(!check("inf-inf is NaN", f64::INFINITY, f64::INFINITY, 0.0));
        assert!(!check("inf vs finite", f64::INFINITY, 0.0, 1e100));
    }

    #[test]
    fn check_negative_zero() {
        assert!(check("neg zero", -0.0, 0.0, 0.0));
    }

    #[test]
    fn check_boundary_tolerance() {
        assert!(!check("at exact boundary (fp rounding)", 1.01, 1.0, 0.01));
        assert!(check("within tolerance", 1.009, 1.0, 0.01));
        assert!(!check("past boundary", 1.02, 1.0, 0.01));
    }

    #[test]
    fn check_count_zero() {
        assert!(check_count("both zero", 0, 0));
    }

    #[test]
    fn check_count_large() {
        assert!(check_count("large", usize::MAX, usize::MAX));
        assert!(!check_count("large diff", usize::MAX, usize::MAX - 1));
    }

    #[test]
    fn print_result_zero_total() {
        assert!(print_result("empty", 0, 0));
    }

    #[test]
    fn validator_all_pass() {
        let mut v = Validator {
            name: String::from("all-pass"),
            passed: 0,
            total: 0,
        };
        for i in 0..10 {
            v.check(&format!("check {i}"), 1.0, 1.0, 0.0);
        }
        assert_eq!(v.counts(), (10, 10));
    }

    #[test]
    fn validator_all_fail() {
        let mut v = Validator {
            name: String::from("all-fail"),
            passed: 0,
            total: 0,
        };
        for i in 0..5 {
            v.check(&format!("fail {i}"), 999.0, 0.0, 0.0);
        }
        assert_eq!(v.counts(), (0, 5));
    }

    // ── Determinism (rerun-identical) tests ────────────────────────

    #[test]
    #[allow(clippy::float_cmp)]
    fn determinism_diversity() {
        use crate::bio::diversity::{bray_curtis, chao1, shannon, simpson};

        let counts = [10.0, 20.0, 30.0, 40.0];
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 3.0, 4.0];

        let sh1 = shannon(&counts);
        let sh2 = shannon(&counts);
        assert_eq!(sh1, sh2, "shannon must be bitwise identical");

        let si1 = simpson(&counts);
        let si2 = simpson(&counts);
        assert_eq!(si1, si2, "simpson must be bitwise identical");

        let c1 = chao1(&counts);
        let c2 = chao1(&counts);
        assert_eq!(c1, c2, "chao1 must be bitwise identical");

        let bc1 = bray_curtis(&a, &b);
        let bc2 = bray_curtis(&a, &b);
        assert_eq!(bc1, bc2, "bray_curtis must be bitwise identical");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn determinism_ode() {
        use crate::bio::qs_biofilm::{QsBiofilmParams, run_scenario};

        let y0 = [0.01, 0.0, 0.0, 2.0, 0.5];
        let t_end = 10.0;
        let dt = 0.1;
        let params = QsBiofilmParams::default();

        let r1 = run_scenario(&y0, t_end, dt, &params);
        let r2 = run_scenario(&y0, t_end, dt, &params);

        assert_eq!(r1.t.len(), r2.t.len(), "ODE trajectory length");
        for (i, (&t1, &t2)) in r1.t.iter().zip(r2.t.iter()).enumerate() {
            assert_eq!(t1, t2, "t[{i}] must be bitwise identical");
        }
        for (i, (&y1, &y2)) in r1.y.iter().zip(r2.y.iter()).enumerate() {
            assert_eq!(y1, y2, "y[{i}] must be bitwise identical");
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn determinism_special_functions() {
        let x = 1.5;
        let erf1 = barracuda::special::erf(x);
        let erf2 = barracuda::special::erf(x);
        assert_eq!(erf1, erf2, "erf must be bitwise identical");

        let ncdf1 = barracuda::stats::norm_cdf(x);
        let ncdf2 = barracuda::stats::norm_cdf(x);
        assert_eq!(ncdf1, ncdf2, "norm_cdf must be bitwise identical");

        let lg1 = barracuda::special::ln_gamma(x).unwrap_or(f64::INFINITY);
        let lg2 = barracuda::special::ln_gamma(x).unwrap_or(f64::INFINITY);
        assert_eq!(lg1, lg2, "ln_gamma must be bitwise identical");
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[allow(clippy::float_cmp)]
    fn determinism_anderson_spectral() {
        use barracuda::spectral::{anderson_2d, lanczos, lanczos_eigenvalues, level_spacing_ratio};

        const SEED: u64 = 42;
        let l = 8_usize;
        let w = 2.0_f64;
        let n_iter = 30_usize;

        let mat1 = anderson_2d(l, l, w, SEED);
        let mat2 = anderson_2d(l, l, w, SEED);
        let tri1 = lanczos(&mat1, n_iter, SEED);
        let tri2 = lanczos(&mat2, n_iter, SEED);
        let eigs1 = lanczos_eigenvalues(&tri1);
        let eigs2 = lanczos_eigenvalues(&tri2);
        let r1 = level_spacing_ratio(&eigs1);
        let r2 = level_spacing_ratio(&eigs2);

        assert_eq!(eigs1.len(), eigs2.len(), "eigenvalue count");
        for (i, (&e1, &e2)) in eigs1.iter().zip(eigs2.iter()).enumerate() {
            assert_eq!(e1, e2, "eigenvalue[{i}] must be bitwise identical");
        }
        assert_eq!(r1, r2, "level_spacing_ratio must be bitwise identical");
    }

    #[test]
    fn determinism_encoding_roundtrip() {
        use crate::encoding::{base64_decode, base64_encode};

        let data = b"Hello, deterministic world!";
        let enc1 = base64_encode(data);
        let enc2 = base64_encode(data);
        assert_eq!(enc1, enc2, "base64 encode must be identical");

        let dec1 = base64_decode(&enc1).expect("decode");
        let dec2 = base64_decode(&enc2).expect("decode");
        assert_eq!(dec1, dec2, "base64 decode must be identical");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn determinism_fastq_parsing() {
        use crate::io::fastq::{compute_stats, parse_fastq};
        use std::fs::File;
        use std::io::Write;

        let synthetic = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nGGCCAATT\n+\n!!!!!!!!\n";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("determinism.fastq");
        let mut f = File::create(&path).expect("create");
        f.write_all(synthetic.as_bytes()).expect("write");
        drop(f);

        let records1 = parse_fastq(&path).expect("parse 1");
        let records2 = parse_fastq(&path).expect("parse 2");
        let stats1 = compute_stats(&records1);
        let stats2 = compute_stats(&records2);

        assert_eq!(stats1.num_sequences, stats2.num_sequences);
        assert_eq!(stats1.total_bases, stats2.total_bases);
        assert_eq!(stats1.min_length, stats2.min_length);
        assert_eq!(stats1.max_length, stats2.max_length);
        assert_eq!(
            stats1.mean_length, stats2.mean_length,
            "mean_length bitwise"
        );
        assert_eq!(
            stats1.mean_quality, stats2.mean_quality,
            "mean_quality bitwise"
        );
        assert_eq!(stats1.gc_content, stats2.gc_content, "gc_content bitwise");
        assert_eq!(stats1.q30_count, stats2.q30_count);
        assert_eq!(stats1.length_distribution, stats2.length_distribution);
    }
}
