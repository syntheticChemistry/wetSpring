// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extension papers (Exp144-156) — Anderson-QS three-tier workloads.

use super::provenance::{BioWorkload, ShaderOrigin};
use crate::substrate::Capability;

/// Cold seep QS catalog (Exp144): diversity + Anderson + ODE.
#[must_use]
pub fn cold_seep_catalog() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "cold_seep_catalog",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BatchedOdeRK4<QsBiofilm>")
}

/// Cold seep QS geometry (Exp145): diversity + Anderson localization.
#[must_use]
pub fn cold_seep_geometry() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "cold_seep_geometry",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BrayCurtisF64")
}

/// `LuxR` phylogeny geometry (Exp146): diversity + phylogenetics + Anderson.
#[must_use]
pub fn luxr_phylogeny() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "luxr_phylogeny",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("FusedMapReduceF64 + RobinsonFouldsF64")
}

/// Mechanical wave Anderson (Exp147): Anderson localization + wave model.
#[must_use]
pub fn mechanical_wave() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "mechanical_wave_anderson",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4 + FusedMapReduceF64")
}

/// QS wave localization (Exp148): Anderson + QS ODE.
#[must_use]
pub fn qs_wave_localization() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "qs_wave_localization",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<QsBiofilm> + FusedMapReduceF64")
        .with_ode(4, 17)
}

/// Burst statistics Anderson (Exp149): stochastic + Anderson.
#[must_use]
pub fn burst_statistics() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "burst_statistics_anderson",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64")
}

/// Physical communication Anderson (Exp152): comm + Anderson disorder.
#[must_use]
pub fn physical_comm() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "physical_comm_anderson",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BatchedOdeRK4")
}

/// Nitrifying QS (Exp153): QS biofilm + diversity + Anderson.
#[must_use]
pub fn nitrifying_qs() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "nitrifying_qs",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BatchedOdeRK4<QsBiofilm>")
        .with_ode(4, 17)
}

/// Marine interkingdom QS (Exp154): cross-domain QS + diversity.
#[must_use]
pub fn marine_interkingdom() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "marine_interkingdom_qs",
            vec![Capability::F64Compute, Capability::ScalarReduce],
        )
        .with_primitive("FusedMapReduceF64 + BrayCurtisF64")
}

/// Myxococcus critical density (Exp155): cooperation ODE + Anderson.
#[must_use]
pub fn myxococcus_critical_density() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "myxococcus_critical_density",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<CooperationOde>")
        .with_ode(4, 13)
}

/// Dictyostelium relay (Exp156): signal relay ODE + Anderson.
#[must_use]
pub fn dictyostelium_relay() -> BioWorkload {
    BioWorkload::new_static(ShaderOrigin::Absorbed)
        .named(
            "dictyostelium_relay",
            vec![Capability::F64Compute, Capability::ShaderDispatch],
        )
        .with_primitive("BatchedOdeRK4<MultiSignalOde>")
        .with_ode(7, 24)
}
