// SPDX-License-Identifier: AGPL-3.0-or-later

//! Node types — assembly and collection statistics.

use crate::data::DataSource;
use crate::dispatch::Reason;
use crate::substrate::SubstrateKind;

/// Assembly statistics computed from a genome FASTA.
#[derive(Debug, Clone)]
pub struct AssemblyStats {
    /// Assembly accession.
    pub accession: String,
    /// Number of contigs/scaffolds.
    pub num_contigs: usize,
    /// Total assembly length in base pairs.
    pub total_length: u64,
    /// N50 value (50% of assembly in contigs this size or larger).
    pub n50: u64,
    /// GC content as a fraction (0.0 to 1.0).
    pub gc_content: f64,
    /// Largest contig length.
    pub largest_contig: u64,
}

/// Collection-level statistics across multiple assemblies.
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Dataset name.
    pub dataset: String,
    /// Number of assemblies analyzed.
    pub assembly_count: usize,
    /// Mean genome size across assemblies.
    pub mean_genome_size: f64,
    /// Standard deviation of genome sizes.
    pub genome_size_std: f64,
    /// Mean GC content across assemblies.
    pub mean_gc: f64,
    /// Standard deviation of GC content.
    pub gc_std: f64,
    /// Mean N50 across assemblies.
    pub mean_n50: f64,
    /// Shannon entropy of the GC distribution (binned).
    pub gc_shannon_entropy: f64,
    /// Individual assembly stats.
    pub assemblies: Vec<AssemblyStats>,
}

/// Result of a Node compute pipeline run.
#[derive(Debug)]
pub struct PipelineResult {
    /// Data source used.
    pub data_source: DataSource,
    /// Compute substrate chosen.
    pub substrate_kind: SubstrateKind,
    /// Dispatch reason.
    pub dispatch_reason: Reason,
    /// Computed collection statistics.
    pub stats: CollectionStats,
}
