// SPDX-License-Identifier: AGPL-3.0-or-later
//! Numerically stable helpers for biological computation.
//!
//! Wraps `f64::ln_1p()`, `f64::exp_m1()`, and related functions with
//! domain-specific context. These avoid catastrophic cancellation when
//! computing log(1+x) or exp(x)-1 for small x — common in diversity
//! indices, ODE models near equilibrium, and drug dose-response curves.
//!
//! # Why these matter for biology
//!
//! - Shannon H' = -Σ pᵢ ln(pᵢ): when pᵢ ≈ 0, `ln(pᵢ)` is fine, but
//!   perturbation analysis needs `ln(1 + δ)` where δ is small.
//! - ODE near equilibrium: `exp(rate × dt) - 1` for small `rate × dt`
//!   loses precision with naive `exp(x) - 1.0`.
//! - Dose-response: Hill function `f(x) = x^n / (K^n + x^n)` at low x
//!   benefits from `expm1`/`ln_1p` rearrangements.

/// Numerically stable `ln(1 + x)` — avoids cancellation for |x| << 1.
///
/// Delegates to `f64::ln_1p()` (IEEE 754 recommended operation).
#[inline]
#[must_use]
pub fn stable_ln1p(x: f64) -> f64 {
    x.ln_1p()
}

/// Numerically stable `exp(x) - 1` — avoids cancellation for |x| << 1.
///
/// Delegates to `f64::exp_m1()` (IEEE 754 recommended operation).
#[inline]
#[must_use]
pub fn stable_expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// Stable log-sum-exp: `ln(exp(a) + exp(b))` without overflow.
///
/// Uses the identity `max(a,b) + ln(1 + exp(-|a-b|))`.
/// Essential for combining log-probabilities (taxonomy classifier,
/// HMM forward algorithm).
///
/// Intentionally local: barraCuda provides `LogsumexpWgsl` (GPU shader)
/// but not a CPU scalar equivalent. This is the CPU-side implementation
/// used by HMM forward, taxonomy classification, and other per-element
/// log-probability combiners that run on CPU.
#[inline]
#[must_use]
pub fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max.is_infinite() && max.is_sign_negative() {
        return f64::NEG_INFINITY;
    }
    max + (a - max).exp().ln_1p().max((b - max).exp().ln_1p())
}

/// Stable log-sum-exp over a slice of log-probabilities.
///
/// Reduces pairwise to avoid overflow. Returns `NEG_INFINITY` for empty input.
#[must_use]
pub fn log_sum_exp_slice(values: &[f64]) -> f64 {
    match values {
        [] => f64::NEG_INFINITY,
        [single] => *single,
        _ => {
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max.is_infinite() && max.is_sign_negative() {
                return f64::NEG_INFINITY;
            }
            let sum_exp: f64 = values.iter().map(|&v| (v - max).exp()).sum();
            max + sum_exp.ln()
        }
    }
}

/// Stable relative difference: `|a - b| / max(|a|, |b|, ε)`.
///
/// Avoids division by zero for values near zero. Used for tolerance
/// comparison in validation binaries.
#[inline]
#[must_use]
pub fn relative_diff(a: f64, b: f64) -> f64 {
    let denom = a.abs().max(b.abs()).max(f64::EPSILON);
    (a - b).abs() / denom
}

/// Kahan compensated summation for improved accuracy over large arrays.
///
/// Delegates to `barracuda::shaders::precision::cpu::kahan_sum` — the
/// canonical implementation. Standard `Iterator::sum()` accumulates O(n)
/// floating-point error; Kahan summation reduces this to O(1).
#[inline]
#[must_use]
pub fn kahan_sum(values: &[f64]) -> f64 {
    barracuda::shaders::precision::cpu::kahan_sum(values)
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "test module: exact float comparisons for special values"
)]
#[expect(
    clippy::imprecise_flops,
    reason = "test module: naive impls intentional for comparison"
)]
mod tests {
    use super::*;

    #[test]
    fn ln1p_small_x() {
        let x: f64 = 1e-15;
        let naive = (1.0 + x).ln();
        let stable = stable_ln1p(x);
        assert!(
            (stable - x).abs() < 1e-25,
            "ln(1+x) ≈ x for small x: stable={stable}"
        );
        assert!(
            (stable - x).abs() <= (naive - x).abs(),
            "stable should be at least as accurate as naive"
        );
    }

    #[test]
    fn expm1_small_x() {
        let x: f64 = 1e-15;
        let naive = x.exp() - 1.0;
        let stable = stable_expm1(x);
        assert!(
            (stable - x).abs() < 1e-25,
            "exp(x)-1 ≈ x for small x: stable={stable}"
        );
        assert!(
            (stable - x).abs() <= (naive - x).abs(),
            "stable should be at least as accurate as naive"
        );
    }

    #[test]
    fn log_sum_exp_basic() {
        let result = log_sum_exp(0.0, 0.0);
        assert!((result - 2.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn log_sum_exp_large_values() {
        let result = log_sum_exp(1000.0, 1000.0);
        assert!((result - (1000.0 + 2.0_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn log_sum_exp_neg_infinity() {
        assert_eq!(
            log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn log_sum_exp_slice_empty() {
        assert_eq!(log_sum_exp_slice(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_slice_single() {
        assert!((log_sum_exp_slice(&[3.0]) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn log_sum_exp_slice_multiple() {
        let result = log_sum_exp_slice(&[0.0, 0.0, 0.0]);
        assert!((result - 3.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn relative_diff_identical() {
        assert!(relative_diff(1.0, 1.0) < f64::EPSILON);
    }

    #[test]
    fn relative_diff_near_zero() {
        assert!(relative_diff(0.0, 0.0) < f64::EPSILON);
        assert!(relative_diff(1e-300, 0.0) < 1.0);
    }

    #[test]
    fn kahan_sum_accuracy() {
        let values = vec![1.0; 1_000_000];
        let naive: f64 = values.iter().sum();
        let kahan = kahan_sum(&values);
        assert!((kahan - 1_000_000.0).abs() < 1e-10);
        assert!((kahan - 1_000_000.0).abs() <= (naive - 1_000_000.0).abs());
    }

    #[test]
    fn kahan_sum_cancellation_resilient() {
        let values: Vec<f64> = (0..10_000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let result = kahan_sum(&values);
        assert!((result - 0.0).abs() < 1e-10);
    }
}
