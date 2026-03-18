// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batch Anderson spectral analysis with eigensolve dispatch.
//!
//! Wraps `barracuda::spectral` primitives into domain-specific helpers
//! for wetSpring's Anderson disorder / ecology mapping. Provides a
//! batch sweep interface that parallelizes across disorder strengths.

use barracuda::spectral::{
    GOE_R, POISSON_R, anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio,
    spectral_bandwidth,
};

/// Result of a single Anderson disorder analysis at a given W.
#[derive(Debug, Clone)]
pub struct AndersonPoint {
    /// Disorder strength W.
    pub w: f64,
    /// Level spacing ratio `<r>`.
    pub r: f64,
    /// Spectral bandwidth.
    pub bandwidth: f64,
    /// Whether the system is in the extended (GOE-like) regime.
    pub is_extended: bool,
}

/// Run Anderson analysis for a single disorder strength.
///
/// - `lattice_l`: linear dimension (e.g. 8 for an 8×8×8 lattice)
/// - `w`: disorder strength
/// - `n_lanczos`: number of Lanczos iterations
/// - `seed`: RNG seed for reproducibility
#[must_use]
pub fn analyze_single(lattice_l: usize, w: f64, n_lanczos: usize, seed: u64) -> AndersonPoint {
    let csr = anderson_3d(lattice_l, lattice_l, lattice_l, w, seed);
    let tri = lanczos(&csr, n_lanczos, seed);
    let eigs = lanczos_eigenvalues(&tri);
    let r = level_spacing_ratio(&eigs);
    let bandwidth = spectral_bandwidth(&eigs);
    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    AndersonPoint {
        w,
        r,
        bandwidth,
        is_extended: r > midpoint,
    }
}

/// Sweep disorder strength W across a range, returning one `AndersonPoint` per value.
///
/// Useful for mapping the Anderson transition in ecology: low W → extended
/// (healthy community), high W → localized (stressed community).
///
/// - `lattice_l`: linear dimension
/// - `w_values`: disorder strengths to probe
/// - `n_lanczos`: Lanczos iterations per point
/// - `seed`: base seed (incremented per point for statistical independence)
#[must_use]
pub fn sweep(
    lattice_l: usize,
    w_values: &[f64],
    n_lanczos: usize,
    seed: u64,
) -> Vec<AndersonPoint> {
    w_values
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let point_seed = seed.wrapping_add(crate::cast::usize_u64(i));
            analyze_single(lattice_l, w, n_lanczos, point_seed)
        })
        .collect()
}

/// Estimate the critical disorder strength `W_c` from a sweep.
///
/// Finds the W value where `<r>` crosses the midpoint between GOE and
/// Poisson. Returns `None` if the transition is not observed in the data.
#[must_use]
pub fn estimate_w_c(sweep_points: &[AndersonPoint]) -> Option<f64> {
    let midpoint = f64::midpoint(GOE_R, POISSON_R);
    sweep_points.windows(2).find_map(|pair| {
        let (a, b) = (&pair[0], &pair[1]);
        if (a.r >= midpoint) == (b.r >= midpoint) {
            None
        } else {
            let frac = (midpoint - a.r) / (b.r - a.r);
            Some(a.w.mul_add(1.0 - frac, b.w * frac))
        }
    })
}

/// Map a Pielou evenness J (0–1) to Anderson disorder W.
///
/// High evenness → low disorder (extended), low evenness → high disorder
/// (localized). Linear mapping: W = `w_max` × (1 − J).
#[must_use]
pub fn pielou_to_disorder(pielou_j: f64, w_max: f64) -> f64 {
    w_max * (1.0 - pielou_j.clamp(0.0, 1.0))
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn single_weak_disorder_is_extended() {
        let point = analyze_single(6, 2.0, 50, 42);
        assert!(
            point.r > POISSON_R,
            "weak disorder should give r > Poisson: r={}",
            point.r
        );
    }

    #[test]
    fn strong_disorder_produces_valid_result() {
        let point = analyze_single(6, 30.0, 50, 42);
        assert!(
            point.r > 0.0 && point.r < 1.0,
            "r must be in (0,1): r={}",
            point.r
        );
        assert!(point.bandwidth > 0.0, "bandwidth must be positive");
    }

    #[test]
    fn sweep_produces_correct_count() {
        let w_values: Vec<f64> = (0..10).map(|i| f64::from(i).mul_add(3.0, 1.0)).collect();
        let points = sweep(6, &w_values, 50, 42);
        assert_eq!(points.len(), 10);
        assert!(points[0].bandwidth > 0.0, "bandwidth should be positive");
    }

    #[test]
    fn pielou_mapping_boundary_values() {
        assert!((pielou_to_disorder(1.0, 20.0) - 0.0).abs() < tolerances::PYTHON_PARITY);
        assert!((pielou_to_disorder(0.0, 20.0) - 20.0).abs() < tolerances::PYTHON_PARITY);
        assert!((pielou_to_disorder(0.5, 20.0) - 10.0).abs() < tolerances::PYTHON_PARITY);
    }

    #[test]
    fn estimate_w_c_finds_crossing() {
        let points = vec![
            AndersonPoint {
                w: 1.0,
                r: GOE_R,
                bandwidth: 1.0,
                is_extended: true,
            },
            AndersonPoint {
                w: 5.0,
                r: f64::midpoint(GOE_R, POISSON_R) + 0.01,
                bandwidth: 1.0,
                is_extended: true,
            },
            AndersonPoint {
                w: 10.0,
                r: f64::midpoint(GOE_R, POISSON_R) - 0.01,
                bandwidth: 0.5,
                is_extended: false,
            },
            AndersonPoint {
                w: 20.0,
                r: POISSON_R,
                bandwidth: 0.3,
                is_extended: false,
            },
        ];
        let w_c = estimate_w_c(&points);
        assert!(w_c.is_some());
        let w_c = w_c.unwrap();
        assert!(
            w_c > 5.0 && w_c < 10.0,
            "W_c should be between 5 and 10: {w_c}"
        );
    }
}
