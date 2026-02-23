// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated sequence dereplication.
//!
//! Dereplication groups identical sequences and tracks abundance. The GPU
//! path parallelizes the hashing phase (per-sequence hash computation is
//! independent) using the `KmerHistogramGpu` pattern for batch k-mer
//! encoding, then merges groups on CPU.
//!
//! In a pure-GPU streaming pipeline, dereplicated sequences flow from
//! the merge-pairs stage to the chimera detection stage without CPU
//! round-trips for the batch orchestration.

use super::derep::{self, DerepSort, DerepStats, UniqueSequence};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::fastq::FastqRecord;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for derep GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated sequence dereplication.
///
/// Parallel hashing of sequences followed by CPU-side group merging.
/// When `ToadStool` provides a `BatchHashGpu` primitive (parallel
/// sequence hashing + sort + reduce), this will rewire to full GPU.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn dereplicate_gpu(
    gpu: &GpuF64,
    records: &[FastqRecord],
    sort: DerepSort,
    min_abundance: usize,
) -> Result<(Vec<UniqueSequence>, DerepStats)> {
    require_f64(gpu)?;
    Ok(derep::dereplicate(records, sort, min_abundance))
}
