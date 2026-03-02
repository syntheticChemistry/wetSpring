// SPDX-License-Identifier: AGPL-3.0-or-later

//! Workload catalog aggregation.

use super::cpu_only;
use super::diversity;
use super::extension_papers;
use super::genomics;
use super::ode;
use super::phylogeny;
use super::provenance::BioWorkload;
use super::s86_science;
use super::spectral;
use super::taxonomy;

/// All known bio domain workloads.
///
/// Returns the full catalog for dispatch planning and absorption tracking.
#[must_use]
pub fn all_workloads() -> Vec<BioWorkload> {
    vec![
        diversity::diversity(),
        phylogeny::pcoa(),
        genomics::kmer_histogram(),
        phylogeny::unifrac_propagate(),
        ode::qs_biofilm_ode(),
        genomics::smith_waterman(),
        phylogeny::felsenstein(),
        taxonomy::taxonomy(),
        spectral::kmd(),
        spectral::gbm_inference(),
        spectral::merge_pairs(),
        spectral::signal_processing(),
        spectral::feature_table(),
        phylogeny::robinson_foulds(),
        genomics::dereplication(),
        genomics::chimera(),
        phylogeny::neighbor_joining(),
        phylogeny::reconciliation(),
        phylogeny::molecular_clock(),
        ode::phage_defense_ode(),
        ode::bistable_ode(),
        ode::multi_signal_ode(),
        ode::cooperation_ode(),
        ode::capacitor_ode(),
        phylogeny::dada2(),
        phylogeny::bootstrap(),
        phylogeny::placement(),
        diversity::diversity_fusion(),
        extension_papers::cold_seep_catalog(),
        extension_papers::cold_seep_geometry(),
        extension_papers::luxr_phylogeny(),
        extension_papers::mechanical_wave(),
        extension_papers::qs_wave_localization(),
        extension_papers::burst_statistics(),
        extension_papers::physical_comm(),
        extension_papers::nitrifying_qs(),
        extension_papers::marine_interkingdom(),
        extension_papers::myxococcus_critical_density(),
        extension_papers::dictyostelium_relay(),
        genomics::assembly_statistics(),
        diversity::gc_analysis(),
        diversity::genome_diversity(),
        spectral::pfas_spectral_match(),
        genomics::vibrio_landscape(),
        genomics::campylobacterota_comparative(),
        cpu_only::ncbi_assembly_ingest(),
        cpu_only::fastq_parsing(),
        s86_science::anderson_spectral(),
        s86_science::hofstadter_butterfly(),
        s86_science::graph_laplacian(),
        s86_science::belief_propagation(),
        s86_science::boltzmann_sampling(),
        s86_science::space_filling_sampling(),
        s86_science::hydrology_et0(),
    ]
}
