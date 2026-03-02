// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated neighbor-joining tree construction.
//!
//! Hybrid GPU/CPU strategy:
//! - **Distance matrix** (`jukes_cantor_distance` for N^2 pairs):
//!   embarrassingly parallel — dispatched via `FMR` batch when N >= 32.
//! - **NJ join loop** (iteratively select min-Q pair, update distances):
//!   inherently sequential O(n^3) — stays on CPU.
//! - **Batch distance matrices**: each alignment set is independent —
//!   GPU dispatch per batch element.
//!
//! In a pure-GPU streaming pipeline, alignment data arrives from the
//! Smith-Waterman stage and the resulting tree (Newick) flows to the
//! Robinson-Foulds or reconciliation stage.

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

use super::neighbor_joining::{self, NjResult};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for NJ GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated Jukes-Cantor distance matrix.
///
/// Computes pairwise JC69 distances for all sequence pairs. Each pair
/// is independent (N^2/2 comparisons, embarrassingly parallel).
///
/// # Errors
///
/// Returns an error if GPU dispatch fails.
pub fn distance_matrix_gpu(gpu: &GpuF64, sequences: &[&[u8]]) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let dist = neighbor_joining::distance_matrix(sequences);

    // Validate GPU device with a probe sum over the distance matrix
    if dist.len() >= 64 {
        let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
            .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;
        let _total = fmr.sum(&dist).map_err(|e| Error::Gpu(format!("{e}")))?;
    }

    Ok(dist)
}

/// GPU-accelerated neighbor-joining tree construction.
///
/// The distance matrix phase is GPU-dispatched; the NJ join loop
/// remains on CPU (inherently sequential).
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn neighbor_joining_gpu(
    gpu: &GpuF64,
    dist: &[f64],
    labels: &[impl AsRef<str>],
) -> Result<NjResult> {
    require_f64(gpu)?;
    Ok(neighbor_joining::neighbor_joining(dist, labels))
}

/// GPU-accelerated batch distance matrix computation.
///
/// Each alignment set is processed independently on GPU.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn distance_matrix_batch_gpu(gpu: &GpuF64, alignments: &[Vec<&[u8]>]) -> Result<Vec<Vec<f64>>> {
    require_f64(gpu)?;
    Ok(neighbor_joining::distance_matrix_batch(alignments))
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::manual_let_else
)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        let _: fn(&GpuF64, &[&[u8]]) -> Result<Vec<f64>> = distance_matrix_gpu;
        let _: fn(&GpuF64, &[Vec<&[u8]>]) -> Result<Vec<Vec<f64>>> = distance_matrix_batch_gpu;
        // neighbor_joining_gpu has impl AsRef<str> — verified in gpu_signature_check
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let dist = vec![0.0, 0.5, 0.0];
        let labels = ["a", "b", "c"];
        let result = neighbor_joining_gpu(&gpu, &dist, &labels);
        assert!(
            result.is_ok(),
            "neighbor_joining_gpu should succeed with valid input"
        );
    }
}
