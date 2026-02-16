// SPDX-License-Identifier: AGPL-3.0-or-later
//! Principal Coordinates Analysis (`PCoA`) — classical multidimensional scaling.
//!
//! Converts a condensed distance matrix (from [`super::diversity::bray_curtis_condensed`])
//! to ordination coordinates via Gower's double-centering + eigendecomposition.
//!
//! This is the CPU reference implementation. For GPU-accelerated `PCoA` on
//! large sample counts, see [`super::pcoa_gpu`] (requires `gpu` feature).
//!
//! # Algorithm
//!
//! 1. Expand condensed distance vector to full N×N matrix D
//! 2. Square element-wise: A = D²
//! 3. Double-center: B = -0.5 × (A - row\_means - col\_means + grand\_mean)
//! 4. Symmetric eigendecomposition of B (Jacobi rotations)
//! 5. Sort eigenvalues descending, take top k
//! 6. Coordinates: X\[i,j\] = eigenvector\[i,j\] × √(eigenvalue\[j\])
//!
//! # References
//!
//! - Gower, J.C. (1966). "Some distance properties of latent root and vector
//!   methods used in multivariate analysis." Biometrika 53(3-4): 325–338.
//! - Legendre, P. & Legendre, L. (2012). Numerical Ecology, 3rd ed. §9.2.

use crate::error::{Error, Result};

/// `PCoA` ordination result.
#[derive(Debug, Clone)]
pub struct PcoaResult {
    /// Ordination coordinates \[n\_samples\]\[n\_axes\], each inner vec has `n_axes` elements.
    pub coordinates: Vec<Vec<f64>>,
    /// Eigenvalues for the retained axes (descending order).
    pub eigenvalues: Vec<f64>,
    /// Proportion of variance explained by each axis.
    pub proportion_explained: Vec<f64>,
}

/// Run `PCoA` on a condensed distance matrix.
///
/// # Arguments
///
/// * `condensed` — Pairwise distances in condensed form (N*(N-1)/2 values,
///   same order as [`super::diversity::bray_curtis_condensed`]).
/// * `n_samples` — Number of samples (N). Must satisfy `condensed.len() == N*(N-1)/2`.
/// * `n_axes` — Number of ordination axes to return (typically 2 or 3).
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if dimensions are inconsistent.
pub fn pcoa(condensed: &[f64], n_samples: usize, n_axes: usize) -> Result<PcoaResult> {
    let expected_pairs = n_samples * (n_samples - 1) / 2;
    if condensed.len() != expected_pairs {
        return Err(Error::InvalidInput(format!(
            "condensed length {} != expected {} for {} samples",
            condensed.len(),
            expected_pairs,
            n_samples
        )));
    }
    if n_samples < 2 {
        return Err(Error::InvalidInput(
            "PCoA requires at least 2 samples".into(),
        ));
    }

    let n = n_samples;
    let k = n_axes.min(n - 1);

    // 1. Expand condensed to full NxN and square
    let mut d_sq = vec![0.0; n * n];
    for i in 1..n {
        for j in 0..i {
            let idx = condensed_index(i, j);
            let val = condensed[idx];
            let sq = val * val;
            d_sq[i * n + j] = sq;
            d_sq[j * n + i] = sq;
        }
    }

    // 2. Double-center: B = -0.5 * (D² - row_means - col_means + grand_mean)
    #[allow(clippy::cast_precision_loss)] // N < 2^53 for any real dataset
    let n_f = n as f64;
    let mut row_means = vec![0.0; n];
    for i in 0..n {
        let sum: f64 = d_sq[i * n..(i + 1) * n].iter().sum();
        row_means[i] = sum / n_f;
    }

    let grand_mean: f64 = row_means.iter().sum::<f64>() / n_f;

    let mut centered = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            centered[i * n + j] =
                -0.5 * (d_sq[i * n + j] - row_means[i] - row_means[j] + grand_mean);
        }
    }

    // 3. Eigendecomposition via Jacobi rotations
    let (eigenvalues, eigenvectors) = jacobi_eigen(&centered, n);

    // 4. Sort descending by eigenvalue
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_vals: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();

    // 5. Proportion explained (only positive eigenvalues contribute)
    let positive_sum: f64 = sorted_vals.iter().filter(|&&v| v > 0.0).sum();
    let proportion: Vec<f64> = sorted_vals[..k]
        .iter()
        .map(|&v| {
            if positive_sum > 0.0 {
                v.max(0.0) / positive_sum
            } else {
                0.0
            }
        })
        .collect();

    // 6. Coordinates: X[i,j] = eigvec[i,j] * sqrt(max(eigenval[j], 0))
    let mut coordinates = vec![vec![0.0; k]; n];
    for axis in 0..k {
        let col_idx = order[axis];
        let scale = sorted_vals[axis].max(0.0).sqrt();
        for sample in 0..n {
            coordinates[sample][axis] = eigenvectors[sample * n + col_idx] * scale;
        }
    }

    Ok(PcoaResult {
        coordinates,
        eigenvalues: sorted_vals[..k].to_vec(),
        proportion_explained: proportion,
    })
}

/// Map (i, j) where i > j to condensed index.
///
/// Condensed order: (1,0), (2,0), (2,1), (3,0), ...
/// Index = i*(i-1)/2 + j
#[inline]
fn condensed_index(i: usize, j: usize) -> usize {
    debug_assert!(i > j);
    i * (i - 1) / 2 + j
}

/// Jacobi eigendecomposition for a symmetric matrix.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors is row-major N×N.
/// Convergence: off-diagonal norm < 1e-24 absolute, max 100×N sweeps.
#[allow(clippy::many_single_char_names)] // standard notation: a=matrix, v=eigvecs, t/c/s=Givens
fn jacobi_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];

    // Initialize eigenvectors to identity
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 100 * n;
    for _sweep in 0..max_sweeps {
        // Check convergence: sum of squared off-diagonal elements
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag += a[i * n + j] * a[i * n + j];
            }
        }
        if off_diag < 1e-24 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }

                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau.abs() > 1e15 {
                    1.0 / (2.0 * tau)
                } else {
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = c * arp - s * arq;
                        a[p * n + r] = a[r * n + p];
                        a[r * n + q] = s * arp + c * arq;
                        a[q * n + r] = a[r * n + q];
                    }
                }

                // Update eigenvectors
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcoa_identical_samples_gives_zero_coordinates() {
        // All distances = 0 → all coordinates = 0
        let condensed = vec![0.0; 3]; // 3 samples, 3 pairs
        let result = pcoa(&condensed, 3, 2).unwrap();
        assert_eq!(result.coordinates.len(), 3);
        for coord in &result.coordinates {
            for &v in coord {
                assert!(v.abs() < 1e-10, "expected ~0, got {v}");
            }
        }
    }

    #[test]
    fn pcoa_two_samples_single_axis() {
        // Two samples at distance 1.0 → single axis, coordinates ±0.5
        let condensed = vec![1.0];
        let result = pcoa(&condensed, 2, 1).unwrap();
        assert_eq!(result.coordinates.len(), 2);
        assert_eq!(result.eigenvalues.len(), 1);
        // Points should be equidistant from origin
        let d = (result.coordinates[0][0] - result.coordinates[1][0]).abs();
        assert!((d - 1.0).abs() < 1e-10, "expected distance 1.0, got {d}");
    }

    #[test]
    fn pcoa_equilateral_triangle() {
        // Three samples, all pairwise distances = 1.0
        // Should produce 2D coordinates forming an equilateral triangle
        let condensed = vec![1.0, 1.0, 1.0]; // (1,0), (2,0), (2,1)
        let result = pcoa(&condensed, 3, 2).unwrap();
        assert_eq!(result.coordinates.len(), 3);

        // Verify reconstructed distances ≈ 1.0
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dx = result.coordinates[i][0] - result.coordinates[j][0];
                let dy = result.coordinates[i][1] - result.coordinates[j][1];
                let dist = (dx * dx + dy * dy).sqrt();
                assert!(
                    (dist - 1.0).abs() < 1e-8,
                    "pair ({i},{j}): expected dist 1.0, got {dist}"
                );
            }
        }
    }

    #[test]
    fn pcoa_proportion_explained_sums_to_one() {
        let condensed = vec![0.5, 1.0, 0.7]; // 3 samples
        let result = pcoa(&condensed, 3, 2).unwrap();
        let sum: f64 = result.proportion_explained.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "proportions should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn pcoa_dimension_mismatch_returns_error() {
        let condensed = vec![1.0, 2.0]; // wrong length for 3 samples
        let result = pcoa(&condensed, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn pcoa_axes_capped_at_n_minus_one() {
        // Request 10 axes but only 3 samples → capped at 2
        let condensed = vec![1.0, 0.5, 0.8];
        let result = pcoa(&condensed, 3, 10).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.coordinates[0].len(), 2);
    }

    #[test]
    fn pcoa_with_bray_curtis_integration() {
        use crate::bio::diversity;

        let samples = vec![
            vec![10.0, 20.0, 30.0, 40.0],
            vec![15.0, 10.0, 25.0, 50.0],
            vec![0.0, 50.0, 0.0, 10.0],
            vec![5.0, 5.0, 45.0, 5.0],
        ];
        let condensed = diversity::bray_curtis_condensed(&samples);
        let result = pcoa(&condensed, 4, 2).unwrap();

        assert_eq!(result.coordinates.len(), 4);
        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0] >= result.eigenvalues[1]);

        // First axis should explain more variance
        assert!(result.proportion_explained[0] >= result.proportion_explained[1]);
    }
}
