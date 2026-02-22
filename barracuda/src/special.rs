// SPDX-License-Identifier: AGPL-3.0-or-later
//! Special mathematical functions for life-science computation.
//!
//! Sovereign implementations of `erf`, `ln_gamma`, and the regularized
//! lower incomplete gamma function.  Promoted from `bio::special` to a
//! top-level module as the first step toward `barracuda::math`.
//!
//! # Consumers
//!
//! - [`crate::bio::pangenome`] — `normal_cdf` for hypergeometric enrichment
//! - [`crate::bio::dada2`] — `regularized_gamma_lower` for Poisson p-values
//!
//! # References
//!
//! - Abramowitz & Stegun 7.1.26 (error function approximation)
//! - Lanczos 1964 (gamma function via reflection)
//! - DLMF §8.2 (regularized incomplete gamma series)

/// Error function approximation (Abramowitz & Stegun 7.1.26).
///
/// Maximum relative error < 1.5 × 10⁻⁷ for all real `x`.
#[must_use]
pub fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / 0.327_591_1_f64.mul_add(x, 1.0);
    let poly = 1.061_405_429_f64
        .mul_add(t, -1.453_152_027)
        .mul_add(t, 1.421_413_741)
        .mul_add(t, -0.284_496_736)
        .mul_add(t, 0.254_829_592);
    let y = (poly * t).mul_add(-(-x * x).exp(), 1.0);
    sign * y
}

/// Normal CDF via the error function: Φ(x) = 0.5 × (1 + erf(x / √2)).
#[must_use]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Lanczos approximation for ln(Γ(x)), g = 5, n = 6 coefficients.
///
/// Returns `f64::INFINITY` for non-positive `x` (poles of the gamma function).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -5.395_239_384_953_e-6,
    ];

    if x <= 0.0 {
        return f64::INFINITY;
    }

    let g = 5.0;
    let z = x - 1.0;
    let mut sum = 0.999_999_999_999_997_1_f64;
    for (i, &c) in COEFFS.iter().enumerate() {
        sum += c / (z + 1.0 + i as f64);
    }

    let t = z + g + 0.5;
    0.5f64.mul_add((2.0 * std::f64::consts::PI).ln(), (z + 0.5) * t.ln()) - t + sum.ln()
}

/// Regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// Uses the series expansion with early termination at 1e-15 relative
/// tolerance.  Returns 0.0 for non-positive `x`, 1.0 when `x` is far
/// in the right tail.
#[must_use]
pub fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x > a + 200.0 {
        return 1.0;
    }

    let log_gamma_a = ln_gamma(a);

    let mut sum = 0.0_f64;
    let mut term = 1.0 / a;
    sum += term;
    for n in 1..crate::tolerances::GAMMA_SERIES_MAX_ITER {
        term *= x / (a + f64::from(n as u32));
        sum += term;
        if term.abs() < crate::tolerances::GAMMA_SERIES_CONVERGENCE * sum.abs() {
            break;
        }
    }

    let log_result = a.mul_add(x.ln(), -x) - log_gamma_a + sum.ln();
    if log_result > 0.0 {
        1.0
    } else {
        log_result.exp().clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        // A&S 7.1.26 max absolute error < 5e-4 near zero (polynomial residual),
        // tighter at non-zero arguments where exp(-x²) dominates.
        assert!(erf(0.0).abs() < 5e-7, "erf(0) ≈ 0, got {}", erf(0.0));
        assert!((erf(1.0) - 0.842_700_792_949_715).abs() < 5e-7);
        assert!((erf(-1.0) + 0.842_700_792_949_715).abs() < 5e-7);
        assert!(erf(3.0) > 0.999_9, "erf(3) near 1, got {}", erf(3.0));
        assert!((erf(-3.0) + 1.0).abs() < 1e-4);
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!(normal_cdf(-4.0) < 1e-4);
        assert!(normal_cdf(4.0) > 0.9999);
    }

    #[test]
    fn ln_gamma_known_values() {
        assert!(ln_gamma(1.0).abs() < 1e-10, "Γ(1) = 1, ln(1) = 0");
        assert!(
            (ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10,
            "Γ(5) = 4! = 24"
        );
        assert!(
            (ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10,
            "Γ(0.5) = √π"
        );
    }

    #[test]
    fn ln_gamma_non_positive() {
        assert!(ln_gamma(0.0).is_infinite());
        assert!(ln_gamma(-1.0).is_infinite());
    }

    #[test]
    fn regularized_gamma_bounds() {
        assert!(regularized_gamma_lower(1.0, 0.0).abs() < f64::EPSILON);
        let val = regularized_gamma_lower(1.0, 10.0);
        assert!(val > 0.99, "P(1, 10) should be near 1.0, got {val}");
    }

    #[test]
    fn regularized_gamma_right_tail() {
        let val = regularized_gamma_lower(1.0, 300.0);
        assert!((val - 1.0).abs() < 1e-10);
    }
}
