// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated MS2 spectral similarity via `ToadStool` `GemmF64`.
//!
//! Computes pairwise cosine similarity for pre-aligned spectral vectors
//! using GPU matrix multiplication for dot products and
//! `FusedMapReduceF64` for vector norms.
//!
//! For N spectra of D bins each:
//! - Dot products: single GEMM dispatch (N×D) × (D×N) → N×N
//! - Norms: N independent `FusedMapReduceF64` dispatches
//!
//! # GPU promotion path
//!
//! CPU `spectral_match::pairwise_cosine` → GPU `pairwise_cosine_gpu`
//! Result should match within [`crate::tolerances::GPU_VS_CPU_F64`].

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::linalg::gemm_f64::GemmF64;

/// Compute pairwise cosine similarity matrix on GPU for pre-aligned spectra.
///
/// Each spectrum is a vector of D intensity values (same bins across all spectra).
/// Returns a condensed similarity matrix: N*(N-1)/2 values.
///
/// For sparse/unaligned spectra, align to common bins first on CPU,
/// then call this for the batch computation.
///
/// # Arguments
///
/// * `gpu` — GPU context.
/// * `spectra` — N spectra, each of length D (all same length).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if GEMM or reduce fails, or spectra have
/// inconsistent dimensions.
pub fn pairwise_cosine_gpu(gpu: &GpuF64, spectra: &[Vec<f64>]) -> Result<Vec<f64>> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 not supported on this GPU".into()));
    }

    let n_spectra = spectra.len();
    if n_spectra < 2 {
        return Ok(vec![]);
    }

    let dim = spectra[0].len();
    for s in spectra {
        if s.len() != dim {
            return Err(Error::Gpu("spectra must have equal dimensions".into()));
        }
    }

    let device = gpu.to_wgpu_device();

    // 1. Compute dot products via GEMM: A × A^T
    //    A is [N, D], A^T is [D, N], result is [N, N]
    let flat_a: Vec<f64> = spectra.iter().flat_map(|s| s.iter().copied()).collect();

    // Transpose: A^T [D, N]
    let mut flat_at = vec![0.0_f64; dim * n_spectra];
    for i in 0..n_spectra {
        for j in 0..dim {
            flat_at[j * n_spectra + i] = flat_a[i * dim + j];
        }
    }

    let dot_matrix = GemmF64::execute(
        device.clone(),
        &flat_a,
        &flat_at,
        n_spectra,
        dim,
        n_spectra,
        1,
    )
    .map_err(|e| Error::Gpu(format!("GEMM dot products: {e}")))?;

    // 2. Compute norms using FusedMapReduceF64 (sum of squares)
    let fmr = FusedMapReduceF64::new(device)
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    let mut norms = Vec::with_capacity(n_spectra);
    for spectrum in spectra {
        let sum_sq = fmr
            .sum_of_squares(spectrum)
            .map_err(|e| Error::Gpu(format!("norm reduce: {e}")))?;
        norms.push(sum_sq.sqrt());
    }

    // 3. Extract condensed cosine similarity: score = dot / (norm_i * norm_j)
    let mut condensed = Vec::with_capacity(n_spectra * (n_spectra - 1) / 2);
    for i in 1..n_spectra {
        for j in 0..i {
            let dot = dot_matrix[i * n_spectra + j];
            let denom = norms[i] * norms[j];
            let score = if denom > 0.0 { dot / denom } else { 0.0 };
            condensed.push(score.clamp(0.0, 1.0));
        }
    }

    Ok(condensed)
}

/// Compute cosine similarity between a single query and a library of references.
///
/// Returns similarity scores for each reference, in the same order.
///
/// # Arguments
///
/// * `gpu` — GPU context.
/// * `query` — Query spectrum vector (length D).
/// * `references` — Library spectra (each length D).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if GEMM or reduce fails.
pub fn cosine_vs_library_gpu(
    gpu: &GpuF64,
    query: &[f64],
    references: &[Vec<f64>],
) -> Result<Vec<f64>> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 not supported on this GPU".into()));
    }

    if references.is_empty() {
        return Ok(vec![]);
    }

    let dim = query.len();
    for r in references {
        if r.len() != dim {
            return Err(Error::Gpu(
                "query and references must have equal dimensions".into(),
            ));
        }
    }

    let device = gpu.to_wgpu_device();
    let n_ref = references.len();

    // GEMM: query [1, D] × references^T [D, N] → dots [1, N]
    let flat_ref: Vec<f64> = references.iter().flat_map(|r| r.iter().copied()).collect();
    let mut flat_ref_t = vec![0.0_f64; dim * n_ref];
    for i in 0..n_ref {
        for j in 0..dim {
            flat_ref_t[j * n_ref + i] = flat_ref[i * dim + j];
        }
    }

    let dots = GemmF64::execute(device.clone(), query, &flat_ref_t, 1, dim, n_ref, 1)
        .map_err(|e| Error::Gpu(format!("GEMM query vs library: {e}")))?;

    // Norms
    let fmr = FusedMapReduceF64::new(device)
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    let q_norm = fmr
        .sum_of_squares(query)
        .map_err(|e| Error::Gpu(format!("query norm: {e}")))?
        .sqrt();

    let mut scores = Vec::with_capacity(n_ref);
    for (i, reference) in references.iter().enumerate() {
        let r_norm = fmr
            .sum_of_squares(reference)
            .map_err(|e| Error::Gpu(format!("ref norm: {e}")))?
            .sqrt();
        let denom = q_norm * r_norm;
        let score = if denom > 0.0 { dots[i] / denom } else { 0.0 };
        scores.push(score.clamp(0.0, 1.0));
    }

    Ok(scores)
}
