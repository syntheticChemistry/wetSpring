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

// ── Validator: structured check accumulator ───────────────────

/// Accumulated validation state, removing manual pass/fail bookkeeping.
///
/// ```text
/// let mut v = Validator::new("FASTQ Validation");
/// v.section("── F3D0_R1 ──");
/// v.check("Shannon", 1.386, 4.0_f64.ln(), 1e-12);
/// v.check_count("Sequences", 7793, 7793);
/// v.finish();
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
    pub fn counts(&self) -> (u32, u32) {
        (self.passed, self.total)
    }

    /// Print summary and exit with 0 (pass) or 1 (fail).
    pub fn finish(self) -> ! {
        let ok = print_result(&self.name, self.passed, self.total);
        std::process::exit(i32::from(!ok))
    }
}

#[cfg(test)]
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
}
