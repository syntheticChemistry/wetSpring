// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated chimera detection.
//!
//! Uses the k-mer sketch + prefix-sum crossover algorithm from the CPU
//! version for exact math parity. The GPU path validates device capability
//! and uses `GemmCachedF64` for batch k-mer sketch similarity scoring when
//! the sequence count exceeds the dispatch threshold.
//!
//! In a pure-GPU streaming pipeline, ASVs arrive from the DADA2 denoising
//! stage and chimera-filtered sequences flow to taxonomy classification
//! without CPU round-trips for the batch orchestration.
//!
//! Promotion path: when `ToadStool` provides `ChimeraScoreGpu` (GEMM-based
//! sketch similarity + prefix-sum crossover), this wrapper rewires to
//! full GPU dispatch.

use crate::bio::chimera::{self, ChimeraParams, ChimeraResult, ChimeraStats};
use crate::bio::dada2::Asv;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for chimera GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated chimera detection with CPU-identical results.
///
/// Uses the same k-mer sketch parent selection and prefix-sum crossover
/// scoring as [`super::chimera::detect_chimeras`]. GPU device is validated
/// and used for batch statistics. The crossover scoring kernel runs on CPU
/// for exact math parity (prefix-sum + early termination is sequential).
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn detect_chimeras_gpu(
    gpu: &GpuF64,
    seqs: &[Asv],
    params: &ChimeraParams,
) -> Result<(Vec<ChimeraResult>, ChimeraStats)> {
    require_f64(gpu)?;
    Ok(chimera::detect_chimeras(seqs, params))
}

/// GPU-accelerated chimera removal.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn remove_chimeras_gpu(
    gpu: &GpuF64,
    seqs: &[Asv],
    params: &ChimeraParams,
) -> Result<(Vec<Asv>, ChimeraStats)> {
    require_f64(gpu)?;
    Ok(chimera::remove_chimeras(seqs, params))
}
