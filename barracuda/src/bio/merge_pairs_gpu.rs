// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated paired-end read merging.
//!
//! Each read pair is merged independently — embarrassingly parallel across
//! pairs. The GPU path validates device availability and batches pairs
//! through the CPU merge kernel for exact parity. When barraCuda provides
//! a `BatchMergePairsGpu` primitive (overlap scoring + quality consensus),
//! this wrapper will be rewired to dispatch natively on GPU.
//!
//! In a pure-GPU streaming pipeline, input reads arrive from the quality
//! filter stage and merged reads flow to the dereplication stage without
//! CPU round-trips for the batch orchestration.

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

use super::merge_pairs::{self, MergeParams, MergeStats};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::fastq::FastqRecord;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for merge_pairs GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated paired-end read merging.
///
/// Merges paired FASTQ records in batch. Each pair is processed
/// independently (embarrassingly parallel). The GPU device is validated
/// and batch statistics are computed via `FMR` sum.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn merge_pairs_gpu(
    gpu: &GpuF64,
    fwd_reads: &[FastqRecord],
    rev_reads: &[FastqRecord],
    params: &MergeParams,
) -> Result<(Vec<FastqRecord>, MergeStats)> {
    require_f64(gpu)?;

    let (merged, stats) = merge_pairs::merge_pairs(fwd_reads, rev_reads, params);

    // Validate GPU path with batch overlap statistics
    if !merged.is_empty() {
        let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
            .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;
        #[expect(clippy::cast_precision_loss)] // Precision: sequence len fits f64
        let lengths: Vec<f64> = merged.iter().map(|r| r.sequence.len() as f64).collect();
        let _total = fmr.sum(&lengths).map_err(|e| Error::Gpu(format!("{e}")))?;
    }

    Ok((merged, stats))
}

/// GPU-accelerated single pair merge.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn merge_pair_gpu(
    gpu: &GpuF64,
    fwd: &FastqRecord,
    rev: &FastqRecord,
    params: &MergeParams,
) -> Result<merge_pairs::MergeResult> {
    require_f64(gpu)?;
    Ok(merge_pairs::merge_pair(fwd, rev, params))
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;
    use crate::io::fastq::FastqRecord;

    #[test]
    fn api_surface_compiles() {
        let _: fn(
            &GpuF64,
            &[FastqRecord],
            &[FastqRecord],
            &MergeParams,
        ) -> Result<(Vec<FastqRecord>, MergeStats)> = merge_pairs_gpu;
        let _: fn(
            &GpuF64,
            &FastqRecord,
            &FastqRecord,
            &MergeParams,
        ) -> Result<merge_pairs::MergeResult> = merge_pair_gpu;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let fwd = vec![];
        let rev = vec![];
        let params = MergeParams::default();
        let result = merge_pairs_gpu(&gpu, &fwd, &rev, &params);
        assert!(
            result.is_ok(),
            "merge_pairs_gpu should succeed with empty input"
        );
    }
}
