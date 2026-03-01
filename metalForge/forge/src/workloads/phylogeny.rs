// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phylogenetics workloads (`PCoA`, `UniFrac`, Felsenstein, Robinson-Foulds, etc.).

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// `PCoA` eigendecomposition.
#[must_use]
pub fn pcoa() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "pcoa",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedEighGpu")
}

/// `UniFrac` tree propagation.
#[must_use]
pub fn unifrac_propagate() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "unifrac_propagate",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("UniFracPropagateGpu")
}

/// Felsenstein phylogenetic pruning.
#[must_use]
pub fn felsenstein() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "felsenstein",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FelsensteinGpu")
}

/// Robinson-Foulds tree distance — `PairwiseHammingGpu`.
#[must_use]
pub fn robinson_foulds() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "robinson_foulds",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("PairwiseHammingGpu")
}

/// DADA2 denoising — GPU E-step via `Dada2EStepGpu`.
#[must_use]
pub fn dada2() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "dada2",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("Dada2EStepGpu")
}

/// Phylogenetic bootstrap — column resampling + `FelsensteinGpu` per replicate.
#[must_use]
pub fn bootstrap() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "bootstrap",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FelsensteinGpu")
}

/// Metagenomic placement — edge-parallel `FelsensteinGpu` for reads.
#[must_use]
pub fn placement() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "placement",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FelsensteinGpu")
}

/// Neighbor joining — GPU distance matrix + CPU NJ loop.
#[must_use]
pub fn neighbor_joining() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "neighbor_joining",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// DTL reconciliation — batch workgroup-per-family.
#[must_use]
pub fn reconciliation() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "reconciliation",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchReconcileGpu")
}

/// Molecular clock — element-wise relaxed rates via FMR.
#[must_use]
pub fn molecular_clock() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "molecular_clock",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}
