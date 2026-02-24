// SPDX-License-Identifier: AGPL-3.0-or-later
//! Special mathematical functions for life-science computation.
//!
//! When the `gpu` feature is active, `erf`, `ln_gamma`, and
//! `regularized_gamma_lower` delegate to `ToadStool`'s
//! `barracuda::special` — one implementation, no duplicate math.
//! Without `gpu`, sovereign local implementations are used.
//!
//! `normal_cdf` is always local (trivial wrapper over `erf`).
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

// ── GPU path: delegate to ToadStool barracuda::special ──────────

/// Error function approximation.
///
/// With `gpu`: delegates to `barracuda::special::erf` (A&S 7.1.26).
/// Without `gpu`: sovereign A&S 7.1.26 (max relative error < 1.5 × 10⁻⁷).
#[cfg(feature = "gpu")]
#[must_use]
pub fn erf(x: f64) -> f64 {
    barracuda::special::erf(x)
}

/// Error function — sovereign A&S 7.1.26 (max relative error < 1.5 × 10⁻⁷).
#[cfg(not(feature = "gpu"))]
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

/// Lanczos approximation for ln(Γ(x)).
///
/// With `gpu`: delegates to `barracuda::special::ln_gamma`.
/// Without `gpu`: sovereign Lanczos g = 5, n = 6 coefficients.
/// Returns `f64::INFINITY` for non-positive `x` (poles of the gamma function).
#[cfg(feature = "gpu")]
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    barracuda::special::ln_gamma(x).unwrap_or(f64::INFINITY)
}

/// Lanczos approximation for ln(Γ(x)) — sovereign Lanczos g = 5, n = 6.
///
/// Returns `f64::INFINITY` for non-positive `x`.
#[cfg(not(feature = "gpu"))]
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
/// With `gpu`: delegates to `barracuda::special::regularized_gamma_p`.
/// Without `gpu`: sovereign series expansion with early termination.
/// Returns 0.0 for non-positive `x`, 1.0 when `x` is far in the right tail.
#[cfg(feature = "gpu")]
#[must_use]
pub fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    barracuda::special::regularized_gamma_p(a, x).unwrap_or(0.0)
}

/// Regularized lower incomplete gamma P(a, x) — sovereign series expansion.
///
/// Returns 0.0 for non-positive `x`, 1.0 in the far right tail.
#[cfg(not(feature = "gpu"))]
#[must_use]
pub fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x > a + crate::tolerances::GAMMA_RIGHT_TAIL_OFFSET {
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

/// L2 (Euclidean) norm of a slice: `sqrt(Σ x²)`.
///
/// Used in spectral comparison, signal normalization, and validation
/// binaries. Prefer this over inline `.map(|x| x * x).sum().sqrt()`.
#[must_use]
pub fn l2_norm(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product of two equal-length slices.
///
/// Panics in debug if lengths differ. Returns 0 for empty slices.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

    #[test]
    fn regularized_gamma_negative_x() {
        assert!(regularized_gamma_lower(1.0, -1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn regularized_gamma_exact_p1_1() {
        let val = regularized_gamma_lower(1.0, 1.0);
        let expected = 1.0 - (-1.0_f64).exp(); // P(1,1) = 1 - e^{-1}
        assert!(
            (val - expected).abs() < 1e-10,
            "P(1,1) = {val}, expected {expected}"
        );
    }

    #[test]
    fn regularized_gamma_small_a() {
        let val = regularized_gamma_lower(0.5, 1.0);
        assert!(val > 0.68 && val < 0.85, "P(0.5, 1.0) ≈ 0.843, got {val}");
    }

    #[test]
    fn erf_large_argument() {
        assert!((erf(6.0) - 1.0).abs() < 1e-10, "erf(6) ≈ 1");
        assert!((erf(-6.0) + 1.0).abs() < 1e-10, "erf(-6) ≈ -1");
    }

    #[test]
    fn erf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.5] {
            assert!((erf(x) + erf(-x)).abs() < 1e-12, "erf is odd");
        }
    }

    #[test]
    fn normal_cdf_symmetry() {
        assert!((normal_cdf(1.0) + normal_cdf(-1.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn l2_norm_empty() {
        assert!(l2_norm(&[]).abs() < f64::EPSILON);
    }

    #[test]
    fn l2_norm_single() {
        assert!((l2_norm(&[3.0]) - 3.0).abs() < f64::EPSILON);
        assert!((l2_norm(&[-5.0]) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn l2_norm_3_4_5() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn l2_norm_unit_vectors() {
        assert!((l2_norm(&[1.0, 0.0, 0.0]) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn dot_empty() {
        assert!(dot(&[], &[]).abs() < f64::EPSILON);
    }

    #[test]
    fn dot_orthogonal() {
        assert!(dot(&[1.0, 0.0], &[0.0, 1.0]).abs() < f64::EPSILON);
    }

    #[test]
    fn dot_parallel() {
        assert!((dot(&[2.0, 3.0], &[2.0, 3.0]) - 13.0).abs() < f64::EPSILON);
    }

    #[test]
    fn dot_known_value() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ln_gamma_large_argument() {
        let val = ln_gamma(10.0);
        let expected = (362_880.0_f64).ln(); // Γ(10) = 9! = 362880
        assert!(
            (val - expected).abs() < 1e-8,
            "ln(Γ(10)) = {val}, expected {expected}"
        );
    }
}
