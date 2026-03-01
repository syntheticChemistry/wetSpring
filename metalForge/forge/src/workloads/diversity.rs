// SPDX-License-Identifier: AGPL-3.0-or-later

//! Diversity metrics workloads (Shannon, Simpson, Bray-Curtis, fusion).

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// Diversity metrics (Shannon, Simpson, Bray-Curtis).
#[must_use]
pub fn diversity() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "diversity",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BrayCurtisF64")
}

/// Fused diversity metrics (Shannon + Simpson + evenness in single dispatch).
///
/// Local WGSL extension following hotSpring's absorption pattern.
/// Computes all three diversity indices in one kernel pass, avoiding
/// three separate `FusedMapReduceF64` dispatches.
/// Absorbed by `ToadStool` S63 as `ops::bio::diversity_fusion`.
#[must_use]
pub fn diversity_fusion() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "diversity_fusion",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("DiversityFusionGpu")
}

/// GC content analysis across assembly collections.
///
/// Computes per-assembly GC fractions and collection-level diversity
/// (Shannon entropy of GC distribution).
#[must_use]
pub fn gc_analysis() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "gc_analysis",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Genome diversity metrics across assembly collections.
///
/// Shannon/Simpson entropy on genome size distributions and GC profiles.
#[must_use]
pub fn genome_diversity() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "genome_diversity",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("DiversityFusionGpu")
}
