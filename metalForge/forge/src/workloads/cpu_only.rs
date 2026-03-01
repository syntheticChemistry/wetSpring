// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-only workloads (I/O-bound, no GPU benefit).

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// FASTQ parsing (CPU-only, I/O-bound).
#[must_use]
pub fn fastq_parsing() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::CpuOnly)
        .named("fastq_parsing", vec![Capability::CpuCompute])
}

/// NCBI assembly ingestion pipeline — I/O + compute coordination.
///
/// CPU-only I/O phase (FASTA parse, gzip decompress) followed by
/// GPU-eligible statistics computation. Dispatched as CPU because
/// the I/O phase dominates.
#[must_use]
pub fn ncbi_assembly_ingest() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::CpuOnly)
        .named("ncbi_assembly_ingest", vec![Capability::CpuCompute])
}
