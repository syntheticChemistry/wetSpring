// SPDX-License-Identifier: AGPL-3.0-or-later

//! Preset workloads for life science and analytical chemistry domains.
//!
//! Each workload declares its required capabilities and shader origin (local
//! WGSL or absorbed `ToadStool` primitive). The origin tracking enables:
//!
//! 1. Dispatch decisions — local shaders need `compile_shader_f64`; absorbed
//!    primitives use `ToadStool`'s pre-built pipelines.
//! 2. Absorption planning — `ToadStool` can see which domains still use local
//!    shaders and prioritize absorption accordingly.
//! 3. Validation routing — local shaders need CPU ↔ GPU parity checks;
//!    absorbed primitives are ToadStool-validated upstream.
//!
//! # Write → Absorb → Lean
//!
//! When `ToadStool` absorbs a local shader, we update the origin from
//! [`ShaderOrigin::Local`] to [`ShaderOrigin::Absorbed`] and rewire the
//! dispatch to use the upstream primitive. This is the Lean step.

mod cpu_only;
mod diversity;
mod extension_papers;
mod genomics;
mod inventory;
mod ode;
mod phylogeny;
mod provenance;
mod s86_science;
mod spectral;
mod taxonomy;

#[cfg(test)]
mod tests;

pub use cpu_only::{fastq_parsing, ncbi_assembly_ingest};
pub use diversity::{diversity, diversity_fusion, gc_analysis, genome_diversity};
pub use extension_papers::{
    burst_statistics, cold_seep_catalog, cold_seep_geometry, dictyostelium_relay, luxr_phylogeny,
    marine_interkingdom, mechanical_wave, myxococcus_critical_density, nitrifying_qs,
    physical_comm, qs_wave_localization,
};
pub use genomics::{
    assembly_statistics, campylobacterota_comparative, chimera, dereplication, kmer_histogram,
    smith_waterman, vibrio_landscape,
};
pub use ode::{
    bistable_ode, capacitor_ode, cooperation_ode, multi_signal_ode, phage_defense_ode,
    qs_biofilm_ode,
};
pub use phylogeny::{
    bootstrap, dada2, felsenstein, molecular_clock, neighbor_joining, pcoa, placement,
    reconciliation, robinson_foulds, unifrac_propagate,
};
pub use provenance::*;
pub use s86_science::{
    anderson_spectral, belief_propagation, boltzmann_sampling, graph_laplacian,
    hofstadter_butterfly, hydrology_et0, space_filling_sampling,
};
pub use spectral::{
    feature_table, gbm_inference, kmd, merge_pairs, pfas_spectral_match, signal_processing,
};
pub use taxonomy::taxonomy;

/// All known bio domain workloads.
///
/// Returns the full catalog for dispatch planning and absorption tracking.
#[must_use]
pub fn all_workloads() -> Vec<BioWorkload> {
    inventory::all_workloads()
}

/// Count workloads by shader origin.
#[must_use]
pub fn origin_summary() -> (usize, usize, usize) {
    let all = all_workloads();
    let absorbed = all.iter().filter(|w| w.is_absorbed()).count();
    let local = all.iter().filter(|w| w.is_local()).count();
    let cpu_only = all
        .iter()
        .filter(|w| matches!(w.origin, ShaderOrigin::CpuOnly))
        .count();
    (absorbed, local, cpu_only)
}
