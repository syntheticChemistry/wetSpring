// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation output sinks — route check results to stdout, a buffer, or `/dev/null`.

/// One float [`super::Validator`] check outcome for inspection.
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
    /// Absolute tol. or relative fraction.
    pub tolerance: f64,
}

/// Row from [`super::Validator::check_abs_or_rel`] (absolute and relative tolerances).
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
/// Used by [`super::Validator`] for CI, biomeOS integration, and tests without coupling to `println!`.
pub trait ValidationSink: Send {
    /// A single floating-point tolerance check completed.
    fn on_check(&mut self, label: &str, passed: bool, actual: f64, expected: f64, tolerance: f64);

    /// A single exact count check completed.
    fn on_check_count(&mut self, label: &str, passed: bool, actual: usize, expected: usize);

    /// A single relative-tolerance float check completed.
    fn on_check_relative(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        tolerance: f64,
    );

    /// A single absolute-or-relative float check completed.
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

    /// Validation run finished; `success` matches [`super::print_result`] semantics.
    fn on_finish(&mut self, name: &str, passed: u32, total: u32, success: bool);
}

/// [`ValidationSink`] that prints the same lines as the standalone helpers.
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

/// [`ValidationSink`] that discards all events.
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

/// Collects [`super::Validator`] float checks into shared [`Vec`]s.
#[derive(Clone, Debug, Default)]
pub struct CollectingSink {
    /// Captured float checks.
    pub results: std::sync::Arc<std::sync::Mutex<Vec<CheckResult>>>,
    /// Relative checks; [`CheckResult::tolerance`] is the relative fraction.
    pub relative_results: std::sync::Arc<std::sync::Mutex<Vec<CheckResult>>>,
    /// Captured abs-or-rel checks.
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
