// SPDX-License-Identifier: AGPL-3.0-or-later
//! [`Validator`] — structured check accumulator removing manual pass/fail bookkeeping.

use super::sink::{CollectingSink, SilentSink, StdoutSink, ValidationSink};

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
    sink: ValidationSink,
}

impl Validator {
    /// Create a new validator for the given binary name.
    ///
    /// Uses [`StdoutSink`] and prints the opening banner.
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
            sink: ValidationSink::Stdout(StdoutSink),
        }
    }

    /// Create a validator with a [`ValidationSink`] variant (no opening banner).
    #[must_use]
    pub fn with_sink(name: impl Into<String>, sink: ValidationSink) -> Self {
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
        Self::with_sink(name, ValidationSink::Silent(SilentSink))
    }

    /// Create a validator with [`CollectingSink`] for programmatic inspection.
    #[must_use]
    pub fn collecting(name: impl Into<String>, sink: CollectingSink) -> Self {
        Self::with_sink(name, ValidationSink::Collecting(sink))
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
    /// When `expected == 0.0`, falls back to absolute comparison.
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
        self.sink.on_finish(&self.name, self.passed, self.total);
        std::process::exit(i32::from(!success))
    }

    /// Print summary and return `ExitCode` without calling `process::exit`.
    ///
    /// Prefer this over [`finish`](Self::finish) in binaries that use the
    /// `fn main() -> ExitCode` + `fn run()` zero-panic pattern.
    #[must_use]
    pub fn finish_with_code(mut self) -> std::process::ExitCode {
        let success = self.total > 0 && self.passed == self.total;
        self.sink.on_finish(&self.name, self.passed, self.total);
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
