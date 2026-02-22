// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated `PCoA` via `ToadStool`'s `BatchedEighGpu`.
//!
//! Workflow:
//! 1. Double-center the squared distance matrix (CPU — O(N²))
//! 2. Eigendecompose via `BatchedEighGpu` (GPU — O(N³) Jacobi rotations)
//! 3. Extract top-k coordinates (CPU)
//!
//! For small N (< 32), uses single-dispatch Jacobi for maximum GPU efficiency.
//! Falls back to multi-dispatch for larger matrices.
//!
//! # Usage
//!
//! ```ignore
//! let condensed = bray_curtis_condensed_gpu(&gpu, &samples)?;
//! let result = pcoa_gpu(&gpu, &condensed, n_samples, 2)?;
//! // result.sample_coords(i) = &[axis1, axis2] for sample i
//! ```

use crate::bio::pcoa::PcoaResult;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::linalg::batched_eigh_gpu::BatchedEighGpu;

/// `PCoA` with GPU-accelerated eigendecomposition.
///
/// Uses `ToadStool`'s `BatchedEighGpu` for the eigensolve step.
/// CPU handles double-centering (O(N²)) and coordinate extraction.
///
/// # Arguments
///
/// * `gpu` — GPU device context (must have `SHADER_F64`).
/// * `condensed` — Pairwise distances in condensed form (N*(N-1)/2).
/// * `n_samples` — Number of samples (N).
/// * `n_axes` — Number of ordination axes to return.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if eigendecomposition fails, or
/// [`Error::InvalidInput`] if dimensions are inconsistent.
pub fn pcoa_gpu(
    gpu: &GpuF64,
    condensed: &[f64],
    n_samples: usize,
    n_axes: usize,
) -> Result<PcoaResult> {
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
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 not supported on this GPU".into()));
    }

    let n = n_samples;
    let k = n_axes.min(n - 1);

    // 1. Build double-centered matrix (CPU — fast for typical N < 1000)
    let centered = double_center(condensed, n);

    // 2. GPU eigendecomposition via ToadStool
    let device = gpu.to_wgpu_device();
    #[allow(clippy::cast_possible_truncation)] // capped at 5000, fits u32
    let max_sweeps = (100 * n).min(5000) as u32;

    let (raw_eigenvalues, raw_eigenvectors) = if n <= 32 {
        BatchedEighGpu::execute_single_dispatch(
            device, &centered, n, 1, // batch_size = 1 (single matrix)
            max_sweeps, 1e-12,
        )
    } else {
        BatchedEighGpu::execute_f64(device, &centered, n, 1, max_sweeps)
    }
    .map_err(|e| Error::Gpu(format!("eigendecomposition: {e}")))?;

    // 3. Sort eigenvalues descending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        raw_eigenvalues[b]
            .partial_cmp(&raw_eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_vals: Vec<f64> = order.iter().map(|&i| raw_eigenvalues[i]).collect();

    // 4. Proportion explained
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

    // 5. Coordinates: X[i,j] = eigvec[i,j] * sqrt(max(eigenval[j], 0))
    let mut coordinates = vec![0.0; n * k];
    for axis in 0..k {
        let col_idx = order[axis];
        let scale = sorted_vals[axis].max(0.0).sqrt();
        for sample in 0..n {
            coordinates[sample * k + axis] = raw_eigenvectors[sample * n + col_idx] * scale;
        }
    }

    Ok(PcoaResult {
        n_samples: n,
        n_axes: k,
        coordinates,
        eigenvalues: sorted_vals[..k].to_vec(),
        proportion_explained: proportion,
    })
}

/// Double-center a condensed distance matrix.
///
/// Returns row-major N×N matrix B where:
/// B\_ij = -0.5 × (D²\_ij - row\_mean\_i - col\_mean\_j + grand\_mean)
fn double_center(condensed: &[f64], n: usize) -> Vec<f64> {
    // Expand condensed → full NxN squared distances
    let mut d_sq = vec![0.0; n * n];
    for i in 1..n {
        for j in 0..i {
            let idx = i * (i - 1) / 2 + j;
            let sq = condensed[idx] * condensed[idx];
            d_sq[i * n + j] = sq;
            d_sq[j * n + i] = sq;
        }
    }

    // Row means (= column means since D² is symmetric)
    #[allow(clippy::cast_precision_loss)] // N < 2^53 for any real dataset
    let n_f = n as f64;
    let mut row_means = vec![0.0; n];
    for i in 0..n {
        let sum: f64 = d_sq[i * n..(i + 1) * n].iter().sum();
        row_means[i] = sum / n_f;
    }

    let grand_mean: f64 = row_means.iter().sum::<f64>() / n_f;

    // Center
    let mut centered = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            centered[i * n + j] =
                -0.5 * (d_sq[i * n + j] - row_means[i] - row_means[j] + grand_mean);
        }
    }
    centered
}
