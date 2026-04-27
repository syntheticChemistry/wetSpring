// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation output sinks — route check results to stdout, a buffer, or `/dev/null`.

use std::io::Write as _;

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

/// Enum-dispatched validation sink — routes check results to stdout, a buffer,
/// or `/dev/null`. Three finite variants; zero trait-object indirection.
pub enum ValidationSink {
    /// Prints formatted `[OK]`/`[FAIL]` lines to stdout.
    Stdout(StdoutSink),
    /// Discards all events.
    Silent(SilentSink),
    /// Collects float checks into shared `Vec`s for programmatic inspection.
    Collecting(CollectingSink),
}

impl ValidationSink {
    /// A single floating-point tolerance check completed.
    pub fn on_check(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        tolerance: f64,
    ) {
        match self {
            Self::Stdout(_) => {
                let tag = if passed { "OK" } else { "FAIL" };
                let _ = writeln!(std::io::stdout().lock(), "  [{tag}]  {label}: {actual:.6} (expected {expected:.6}, tol {tolerance:.6})");
            }
            Self::Silent(_) => {}
            Self::Collecting(c) => {
                if let Ok(mut guard) = c.results.lock() {
                    guard.push(CheckResult {
                        label: label.to_string(),
                        passed,
                        actual,
                        expected,
                        tolerance,
                    });
                }
            }
        }
    }

    /// A single exact count check completed.
    pub fn on_check_count(
        &mut self,
        label: &str,
        passed: bool,
        actual: usize,
        expected: usize,
    ) {
        match self {
            Self::Stdout(_) => {
                let tag = if passed { "OK" } else { "FAIL" };
                let _ = writeln!(std::io::stdout().lock(), "  [{tag}]  {label}: {actual} (expected {expected})");
            }
            Self::Silent(_) | Self::Collecting(_) => {}
        }
    }

    /// A single relative-tolerance float check completed.
    pub fn on_check_relative(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        tolerance: f64,
    ) {
        match self {
            Self::Stdout(_) => {
                let tag = if passed { "OK" } else { "FAIL" };
                let _ = writeln!(
                    std::io::stdout().lock(),
                    "  [{tag}]  {label}: {actual:.6} (expected {expected:.6}, rel_tol {tolerance:.6})"
                );
            }
            Self::Silent(_) => {}
            Self::Collecting(c) => {
                if let Ok(mut guard) = c.relative_results.lock() {
                    guard.push(CheckResult {
                        label: label.to_string(),
                        passed,
                        actual,
                        expected,
                        tolerance,
                    });
                }
            }
        }
    }

    /// A single absolute-or-relative float check completed.
    pub fn on_check_abs_or_rel(
        &mut self,
        label: &str,
        passed: bool,
        actual: f64,
        expected: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) {
        match self {
            Self::Stdout(_) => {
                let tag = if passed { "OK" } else { "FAIL" };
                let _ = writeln!(
                    std::io::stdout().lock(),
                    "  [{tag}]  {label}: {actual:.6} (expected {expected:.6}, abs_tol {abs_tol:.6}, rel_tol {rel_tol:.6})"
                );
            }
            Self::Silent(_) => {}
            Self::Collecting(c) => {
                if let Ok(mut guard) = c.abs_or_rel_results.lock() {
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
        }
    }

    /// A section header (not counted as a check).
    pub fn on_section(&mut self, label: &str) {
        if let Self::Stdout(_) = self {
            let _ = writeln!(std::io::stdout().lock(), "\n{label}");
        }
    }

    /// Validation run finished; prints summary for [`Stdout`](Self::Stdout).
    pub fn on_finish(&mut self, name: &str, passed: u32, total: u32) {
        if let Self::Stdout(_) = self {
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "\n═══════════════════════════════════════════════════════════");
            let _ = writeln!(out, "  {name}: {passed}/{total} checks passed");
            if total == 0 {
                let _ = writeln!(out, "  RESULT: FAIL (no checks executed)");
            } else if passed == total {
                let _ = writeln!(out, "  RESULT: PASS");
            } else {
                let _ = writeln!(out, "  RESULT: FAIL ({} checks failed)", total - passed);
            }
            let _ = writeln!(out, "═══════════════════════════════════════════════════════════");
        }
    }
}

/// Marker for stdout output (prints formatted `[OK]`/`[FAIL]` lines).
#[derive(Clone, Copy, Debug, Default)]
pub struct StdoutSink;

/// Marker for silent output (discards all events).
#[derive(Clone, Copy, Debug, Default)]
pub struct SilentSink;

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
