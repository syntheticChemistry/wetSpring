// SPDX-License-Identifier: AGPL-3.0-or-later

//! Genomics workloads (K-mer, alignment, assembly, dereplication).

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// K-mer histogram counting.
#[must_use]
pub fn kmer_histogram() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "kmer_histogram",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("KmerHistogramGpu")
}

/// Smith-Waterman alignment.
#[must_use]
pub fn smith_waterman() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "smith_waterman",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("SmithWatermanGpu")
}

/// Dereplication — parallel hashing via `KmerHistogramGpu` pattern.
#[must_use]
pub fn dereplication() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "dereplication",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("KmerHistogramGpu")
}

/// Chimera detection — GEMM-based sketch scoring.
#[must_use]
pub fn chimera() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "chimera",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("GemmCachedF64")
}

/// Assembly statistics (N50, GC, genome size) — CPU f64 compute.
///
/// Processes NCBI genome assemblies resolved via the Nest data chain.
/// GPU promotion via `FusedMapReduceF64` for large collections.
#[must_use]
pub fn assembly_statistics() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "assembly_statistics",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Vibrio landscape analysis — cross-assembly comparative genomics.
///
/// K-mer profiling + diversity across Vibrio assembly collections.
#[must_use]
pub fn vibrio_landscape() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "vibrio_landscape",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + KmerHistogramGpu")
}

/// Campylobacterota comparative genomics — pan-genome statistics.
#[must_use]
pub fn campylobacterota_comparative() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "campylobacterota_comparative",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}
