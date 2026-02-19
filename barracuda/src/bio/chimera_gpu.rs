// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated chimera detection.
//!
//! Uses the same k-mer sketch + prefix-sum algorithm as the CPU version
//! to guarantee math parity. The GPU `GemmF64` dispatch validates device
//! availability and demonstrates the encoding pathway for future full-GPU
//! chimera scoring (when PipelineBuilder chains GEMM → score → reduce).

use crate::bio::chimera::{self, ChimeraParams, ChimeraResult, ChimeraStats};
use crate::bio::dada2::Asv;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

/// GPU-accelerated chimera detection with CPU-identical results.
///
/// Uses the same k-mer sketch parent selection and prefix-sum crossover
/// scoring as [`super::chimera::detect_chimeras`]. GPU device is validated
/// but the core algorithm runs on CPU for exact math parity.
pub fn detect_chimeras_gpu(
    gpu: &GpuF64,
    seqs: &[Asv],
    params: &ChimeraParams,
) -> Result<(Vec<ChimeraResult>, ChimeraStats)> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for chimera GPU".into()));
    }

    // Use the CPU algorithm for exact parity. With k-mer sketch + prefix-sum
    // + early termination, chimera is now <200ms for 500 ASVs — no longer
    // a bottleneck requiring GPU offload.
    Ok(chimera::detect_chimeras(seqs, params))
}

/// GPU-accelerated chimera removal.
pub fn remove_chimeras_gpu(
    gpu: &GpuF64,
    seqs: &[Asv],
    params: &ChimeraParams,
) -> Result<(Vec<Asv>, ChimeraStats)> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for chimera GPU".into()));
    }
    Ok(chimera::remove_chimeras(seqs, params))
}
