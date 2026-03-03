// SPDX-License-Identifier: AGPL-3.0-or-later
//! Special mathematical functions for life-science computation.
//!
//! `erf`, `ln_gamma`, `regularized_gamma_lower`, and `normal_cdf` delegate
//! to `ToadStool`'s `barracuda` crate — one implementation, no duplicate
//! math. `barracuda` is always available (default-features = false for
//! CPU-only builds, full GPU when `gpu` feature is active).
//!
//! `dot` and `l2_norm` delegate to `barracuda::stats` (S64 absorption).
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

/// Error function approximation.
///
/// Delegates to `barracuda::special::erf` (A&S 7.1.26, max relative error < 1.5 × 10⁻⁷).
#[must_use]
pub fn erf(x: f64) -> f64 {
    barracuda::special::erf(x)
}

/// Normal CDF via the error function: Φ(x) = 0.5 × (1 + erf(x / √2)).
///
/// Delegates to `barracuda::stats::norm_cdf` (same formula, single implementation).
#[must_use]
pub fn normal_cdf(x: f64) -> f64 {
    barracuda::stats::norm_cdf(x)
}

/// Lanczos approximation for ln(Γ(x)).
///
/// Delegates to `barracuda::special::ln_gamma`.
/// Returns `f64::INFINITY` for non-positive `x` (poles of the gamma function).
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    barracuda::special::ln_gamma(x).unwrap_or(f64::INFINITY)
}

/// Regularized lower incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// Delegates to `barracuda::special::regularized_gamma_p`.
/// Returns 0.0 for non-positive `x`, 1.0 when `x` is far in the right tail.
#[must_use]
pub fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    barracuda::special::regularized_gamma_p(a, x).unwrap_or(0.0)
}

/// L2 (Euclidean) norm of a slice: `sqrt(Σ x²)`.
///
/// Delegates to `barracuda::stats::l2_norm` (S64).
#[must_use]
pub fn l2_norm(xs: &[f64]) -> f64 {
    barracuda::stats::l2_norm(xs)
}

/// Dot product of two equal-length slices.
///
/// Delegates to `barracuda::stats::dot` (S64).
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    barracuda::stats::dot(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        assert!(
            erf(0.0).abs() < crate::tolerances::ERF_PARITY,
            "erf(0) ≈ 0, got {}",
            erf(0.0)
        );
        assert!((erf(1.0) - 0.842_700_792_949_715).abs() < crate::tolerances::ERF_PARITY);
        assert!((erf(-1.0) + 0.842_700_792_949_715).abs() < crate::tolerances::ERF_PARITY);
        assert!(erf(3.0) > 0.999_9, "erf(3) near 1, got {}", erf(3.0));
        assert!((erf(-3.0) + 1.0).abs() < crate::tolerances::NORM_CDF_TAIL);
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.96) - 0.975).abs() < crate::tolerances::NORM_CDF_PARITY);
        assert!(normal_cdf(-4.0) < crate::tolerances::NORM_CDF_TAIL);
        assert!(normal_cdf(4.0) > 0.9999);
    }

    #[test]
    fn ln_gamma_known_values() {
        assert!(
            ln_gamma(1.0).abs() < crate::tolerances::ANALYTICAL_LOOSE,
            "Γ(1) = 1, ln(1) = 0"
        );
        assert!(
            (ln_gamma(5.0) - 24.0_f64.ln()).abs() < crate::tolerances::ANALYTICAL_LOOSE,
            "Γ(5) = 4! = 24"
        );
        assert!(
            (ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs()
                < crate::tolerances::ANALYTICAL_LOOSE,
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
        assert!((val - 1.0).abs() < crate::tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn regularized_gamma_negative_x() {
        assert!(regularized_gamma_lower(1.0, -1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn regularized_gamma_exact_p1_1() {
        let val = regularized_gamma_lower(1.0, 1.0);
        let expected = 1.0 - (-1.0_f64).exp();
        assert!(
            (val - expected).abs() < crate::tolerances::ANALYTICAL_LOOSE,
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
        assert!(
            (erf(6.0) - 1.0).abs() < crate::tolerances::ANALYTICAL_LOOSE,
            "erf(6) ≈ 1"
        );
        assert!(
            (erf(-6.0) + 1.0).abs() < crate::tolerances::ANALYTICAL_LOOSE,
            "erf(-6) ≈ -1"
        );
    }

    #[test]
    fn erf_symmetry() {
        for &x in &[0.5, 1.0, 2.0, 3.5] {
            assert!(
                (erf(x) + erf(-x)).abs() < crate::tolerances::ANALYTICAL_F64,
                "erf is odd"
            );
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
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < crate::tolerances::ANALYTICAL_F64);
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
        let expected = (362_880.0_f64).ln();
        assert!(
            (val - expected).abs() < 1e-8,
            "ln(Γ(10)) = {val}, expected {expected}"
        );
    }
}
