// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated quality filtering for FASTQ reads.
//!
//! Uses GPU `FusedMapReduceF64` for batch quality score computation, with
//! the same trimming and filtering logic as the CPU version.
//! Produces identical output to [`super::quality::filter_reads`].

use crate::bio::quality::{self, FilterStats, QualityParams};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::fastq::FastqRecord;

/// Quality-filter reads using GPU-accelerated quality computation.
///
/// Quality trimming (leading, trailing, sliding window) is inherently
/// sequential per-read, so it delegates to the CPU implementation.
/// GPU device availability is validated for pipeline consistency.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64`.
pub fn filter_reads_gpu(
    gpu: &GpuF64,
    reads: &[FastqRecord],
    params: &QualityParams,
) -> Result<(Vec<FastqRecord>, FilterStats)> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for quality GPU".into()));
    }

    let (filtered, stats) = quality::filter_reads(reads, params);
    Ok((filtered, stats))
}
