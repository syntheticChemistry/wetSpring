// SPDX-License-Identifier: AGPL-3.0-or-later
//! Population genomics scenario builders: SNP calling, Robinson-Foulds /
//! neighbor-joining distances, and k-mer frequency spectrum.

use crate::bio::kmer;
use crate::bio::neighbor_joining;
use crate::bio::robinson_foulds;
use crate::bio::snp;
use crate::bio::unifrac::PhyloTree;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, distribution, heatmap, node, scaffold, spectrum};

/// SNP calling scenario.
///
/// Calls SNPs from a synthetic alignment and visualises the SNP position
/// matrix and allele frequency distribution.
#[must_use]
pub fn snp_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "SNP Calling",
        "Variant calling from multiple sequence alignment",
    );

    let seq1 = b"ATCGATCGATCGATCG";
    let seq2 = b"ATCAATCGATCGATCG";
    let seq3 = b"ATCGATCGATCAATCG";
    let seq4 = b"GCCGATCGATCGATCG";
    let sequences: Vec<&[u8]> = vec![seq1, seq2, seq3, seq4];

    let result = snp::call_snps(&sequences);

    let mut snp_node = node("snp", "SNP Calling", "compute", &["science.snp"]);

    let n_seqs = sequences.len();
    let aln_len = seq1.len();
    let seq_labels: Vec<String> = (0..n_seqs).map(|i| format!("seq_{i}")).collect();
    let pos_labels: Vec<String> = (0..aln_len).map(|i| format!("{i}")).collect();

    let mut matrix_vals = Vec::with_capacity(n_seqs * aln_len);
    for seq in &sequences {
        for &base in *seq {
            matrix_vals.push(f64::from(base));
        }
    }

    snp_node.data_channels.push(heatmap(
        "snp_matrix",
        "Alignment Matrix",
        &pos_labels,
        &seq_labels,
        &matrix_vals,
        "base",
    ));

    let alt_freqs: Vec<f64> = result.variants.iter().map(|v| v.alt_frequency()).collect();
    if !alt_freqs.is_empty() {
        let mean_af = alt_freqs.iter().sum::<f64>() / alt_freqs.len() as f64;
        let std_af = (alt_freqs.iter().map(|v| (v - mean_af).powi(2)).sum::<f64>()
            / alt_freqs.len() as f64)
            .sqrt();
        snp_node.data_channels.push(distribution(
            "allele_freq",
            "Alternate Allele Frequencies",
            "frequency",
            &alt_freqs,
            mean_af,
            std_af,
        ));
    }
    s.nodes.push(snp_node);
    (s, vec![])
}

/// Population genomics distances scenario.
///
/// Computes Robinson-Foulds distances between tree topologies and
/// neighbor-joining distance matrices for population structure.
#[must_use]
pub fn population_genomics_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Population Genomics",
        "Robinson-Foulds distances and neighbor-joining tree estimation",
    );

    let tree_a = PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.35);");
    let tree_b = PhyloTree::from_newick("((A:0.12,C:0.18):0.28,(B:0.22,D:0.3):0.32);");
    let rf = robinson_foulds::rf_distance(&tree_a, &tree_b);
    let rf_norm = robinson_foulds::rf_distance_normalized(&tree_a, &tree_b);

    let seq_a: &[u8] = b"ATCGATCGATCGATCG";
    let seq_b: &[u8] = b"ATCAATCGATCGATCG";
    let seq_c: &[u8] = b"GCCGATCGATCGATCG";
    let seq_d: &[u8] = b"ATCGATCGATCAATCG";
    let seqs: Vec<&[u8]> = vec![seq_a, seq_b, seq_c, seq_d];
    let labels = ["Pop_A", "Pop_B", "Pop_C", "Pop_D"];

    let dist = neighbor_joining::distance_matrix(&seqs);
    let _nj_result = neighbor_joining::neighbor_joining(&dist, &labels);

    let mut popgen_node = node(
        "population_genomics",
        "Population Genomics",
        "compute",
        &["science.population_genomics"],
    );

    popgen_node.data_channels.push(bar(
        "rf_distances",
        "Robinson-Foulds Distance",
        &["RF (raw)", "RF (normalized)"],
        &[rf as f64, rf_norm],
        "splits",
    ));

    let label_strs: Vec<String> = labels.iter().map(|s| (*s).into()).collect();
    popgen_node.data_channels.push(heatmap(
        "nj_distance",
        "NJ Distance Matrix",
        &label_strs,
        &label_strs,
        &dist,
        "JC distance",
    ));

    s.nodes.push(popgen_node);
    (s, vec![])
}

/// K-mer frequency spectrum scenario.
///
/// Counts k-mers in a synthetic metagenome and renders the frequency
/// spectrum using the `Spectrum` channel type (first real user).
#[must_use]
pub fn kmer_spectrum_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "K-mer Frequency Spectrum",
        "K-mer counting and frequency spectrum for genome profiling",
    );

    let sequence = b"ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\
                      GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\
                      ATCGATCGATCGATCGATCGATCGATCGATCGATCG";

    let counts = kmer::count_kmers(sequence, 5);
    let histogram = counts.to_histogram();

    let mut kmer_node = node(
        "kmer_spectrum",
        "K-mer Spectrum",
        "compute",
        &["science.kmer"],
    );

    let frequencies: Vec<f64> = (0..histogram.len()).map(|i| (i + 1) as f64).collect();
    let amplitudes: Vec<f64> = histogram.iter().map(|&c| f64::from(c)).collect();

    kmer_node.data_channels.push(spectrum(
        "kmer_freq_spectrum",
        "K-mer Frequency Spectrum (k=5)",
        "count",
        &frequencies,
        &amplitudes,
    ));

    let top = counts.top_n(10);
    let top_labels: Vec<String> = top.iter().map(|(k, _)| kmer::decode_kmer(*k, 5)).collect();
    let top_counts: Vec<f64> = top.iter().map(|(_, c)| f64::from(*c)).collect();

    kmer_node.data_channels.push(bar(
        "top_kmers",
        "Top 10 K-mers",
        &top_labels,
        &top_counts,
        "count",
    ));
    s.nodes.push(kmer_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snp_scenario_builds() {
        let (s, _) = snp_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
    }

    #[test]
    fn population_genomics_builds() {
        let (s, _) = population_genomics_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 2);
    }

    #[test]
    fn kmer_spectrum_builds() {
        let (s, _) = kmer_spectrum_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 2);
    }
}
