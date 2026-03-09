// SPDX-License-Identifier: AGPL-3.0-or-later
//! 16S pipeline scenario builders: quality filtering, DADA2 denoising,
//! taxonomy classification, and end-to-end pipeline overview.

use crate::bio::dada2::{self, Dada2Params};
use crate::bio::quality::QualityParams;
use crate::visualization::types::{EcologyScenario, ScenarioEdge};

use super::{bar, distribution, edge, gauge, heatmap, node, scaffold, timeseries};

/// Quality filtering scenario.
///
/// Demonstrates Phred score distributions and per-position quality profiles
/// using synthetic reads.
#[must_use]
pub fn quality_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Quality Filtering",
        "Phred score distribution and per-position quality profile",
    );

    let params = QualityParams::default();
    let phred_scores: Vec<f64> = (0..150)
        .map(|i| {
            let base = 35.0 - (i as f64 / 150.0) * 15.0;
            base + ((i as f64) * 0.3).sin() * 3.0
        })
        .collect();
    let positions: Vec<f64> = (0..150).map(|i| i as f64).collect();

    let mut qual_node = node(
        "quality",
        "Quality Filtering",
        "compute",
        &["science.quality"],
    );

    qual_node.data_channels.push(timeseries(
        "per_position_quality",
        "Per-Position Phred Score",
        "Position (bp)",
        "Phred Score",
        "Q",
        &positions,
        &phred_scores,
    ));

    let mean_q = phred_scores.iter().sum::<f64>() / phred_scores.len() as f64;
    let std_q = (phred_scores
        .iter()
        .map(|v| (v - mean_q).powi(2))
        .sum::<f64>()
        / phred_scores.len() as f64)
        .sqrt();
    qual_node.data_channels.push(distribution(
        "phred_dist",
        "Phred Score Distribution",
        "Q",
        &phred_scores,
        mean_q,
        std_q,
    ));

    qual_node.data_channels.push(gauge(
        "min_quality",
        "Minimum Quality Threshold",
        f64::from(params.window_min_quality),
        0.0,
        40.0,
        "Q",
        [20.0, 40.0],
        [10.0, 20.0],
    ));
    s.nodes.push(qual_node);
    (s, vec![])
}

/// DADA2 denoising scenario.
///
/// Runs DADA2 on synthetic unique sequences and visualises the error model
/// and ASV output counts.
#[must_use]
pub fn dada2_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    use crate::bio::derep::UniqueSequence;

    let mut s = scaffold(
        "DADA2 Denoising",
        "Amplicon sequence variant denoising with error model",
    );

    let make_unique = |seq: &[u8], abundance: usize| UniqueSequence {
        sequence: seq.to_vec(),
        abundance,
        best_quality: 35.0,
        representative_id: String::new(),
        representative_quality: vec![35; seq.len()],
    };
    let seqs = vec![
        make_unique(b"ATCGATCGATCGATCG", 100),
        make_unique(b"ATCGATCGATCGATCA", 80),
        make_unique(b"ATCGATCGATCGATCG", 5),
        make_unique(b"GGGGATCGATCGATCG", 60),
        make_unique(b"GGGGATCGATCGATCA", 3),
    ];
    let params = Dada2Params::default();
    let (asvs, stats) = dada2::denoise(&seqs, &params);

    let mut dada2_node = node("dada2", "DADA2 Denoising", "compute", &["science.dada2"]);

    let asv_labels: Vec<String> = asvs
        .iter()
        .enumerate()
        .map(|(i, _)| format!("ASV_{i}"))
        .collect();
    let asv_abundances: Vec<f64> = asvs.iter().map(|a| a.abundance as f64).collect();

    dada2_node.data_channels.push(bar(
        "asv_counts",
        "ASV Abundances",
        &asv_labels,
        &asv_abundances,
        "reads",
    ));

    dada2_node.data_channels.push(gauge(
        "input_uniques",
        "Input Unique Sequences",
        stats.input_uniques as f64,
        0.0,
        (stats.input_uniques * 2) as f64,
        "sequences",
        [0.0, stats.input_uniques as f64],
        [0.0, 0.0],
    ));

    dada2_node.data_channels.push(gauge(
        "output_asvs",
        "Output ASVs",
        stats.output_asvs as f64,
        0.0,
        stats.input_uniques as f64,
        "ASVs",
        [1.0, stats.input_uniques as f64],
        [0.0, 1.0],
    ));

    let error_rates: Vec<f64> = (0..16)
        .map(|i| 0.01 * (1.0 + (i as f64 * 0.5).sin().abs()))
        .collect();
    let _nuc_pairs: Vec<String> = [
        "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG",
        "TT",
    ]
    .iter()
    .map(|s| (*s).into())
    .collect();
    dada2_node.data_channels.push(heatmap(
        "error_model",
        "DADA2 Error Model",
        &["A", "C", "G", "T"]
            .iter()
            .map(|s| (*s).into())
            .collect::<Vec<String>>(),
        &["A", "C", "G", "T"]
            .iter()
            .map(|s| (*s).into())
            .collect::<Vec<String>>(),
        &error_rates,
        "error rate",
    ));
    s.nodes.push(dada2_node);
    (s, vec![])
}

/// Taxonomy classification scenario.
///
/// Shows genus-level abundance bar chart and sample-by-taxon heatmap
/// from synthetic classification results.
#[must_use]
pub fn taxonomy_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Taxonomy Classification",
        "Naive Bayes genus-level classification with confidence",
    );

    let genera = vec![
        "Bacteroides",
        "Prevotella",
        "Faecalibacterium",
        "Ruminococcus",
        "Bifidobacterium",
        "Akkermansia",
        "Escherichia",
        "Lactobacillus",
    ];

    let sample1_abundances: Vec<f64> = vec![25.0, 15.0, 20.0, 10.0, 8.0, 5.0, 12.0, 5.0];
    let sample2_abundances: Vec<f64> = vec![10.0, 30.0, 5.0, 15.0, 12.0, 3.0, 20.0, 5.0];
    let sample3_abundances: Vec<f64> = vec![20.0, 10.0, 25.0, 8.0, 15.0, 10.0, 7.0, 5.0];

    let mut tax_node = node(
        "taxonomy",
        "Taxonomy Classification",
        "compute",
        &["science.taxonomy"],
    );

    tax_node.data_channels.push(bar(
        "genus_abundances",
        "Genus Abundances (Sample 1)",
        &genera,
        &sample1_abundances,
        "%",
    ));

    let sample_labels: Vec<String> = vec!["Sample1".into(), "Sample2".into(), "Sample3".into()];
    let genus_labels: Vec<String> = genera.iter().map(|s| (*s).into()).collect();
    let mut heatmap_vals = Vec::new();
    heatmap_vals.extend_from_slice(&sample1_abundances);
    heatmap_vals.extend_from_slice(&sample2_abundances);
    heatmap_vals.extend_from_slice(&sample3_abundances);

    tax_node.data_channels.push(heatmap(
        "sample_taxon",
        "Sample × Taxon Abundance",
        &genus_labels,
        &sample_labels,
        &heatmap_vals,
        "%",
    ));
    s.nodes.push(tax_node);
    (s, vec![])
}

/// 16S pipeline overview scenario.
///
/// Composite gauge showing reads retained at each stage and per-stage pass rates.
#[must_use]
pub fn pipeline_overview_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "16S Pipeline Overview",
        "End-to-end 16S rRNA analysis pipeline progress",
    );

    let stages = ["Raw", "Quality", "DADA2", "Taxonomy", "Diversity"];
    let reads_retained: Vec<f64> = vec![10000.0, 8500.0, 7200.0, 7200.0, 7200.0];
    let pass_rates: Vec<f64> = vec![100.0, 85.0, 84.7, 100.0, 100.0];

    let mut pipeline_node = node(
        "pipeline_overview",
        "Pipeline Overview",
        "compute",
        &["science.pipeline"],
    );

    for (i, stage) in stages.iter().enumerate() {
        pipeline_node.data_channels.push(gauge(
            &format!("stage_{}", stage.to_lowercase()),
            &format!("{stage} Reads Retained"),
            reads_retained[i],
            0.0,
            reads_retained[0],
            "reads",
            [reads_retained[0] * 0.7, reads_retained[0]],
            [reads_retained[0] * 0.3, reads_retained[0] * 0.7],
        ));
    }

    pipeline_node.data_channels.push(bar(
        "pass_rates",
        "Stage Pass Rates",
        &stages,
        &pass_rates,
        "%",
    ));
    s.nodes.push(pipeline_node);

    let edges = vec![
        edge("quality", "dada2", "filtered reads"),
        edge("dada2", "taxonomy", "ASVs"),
        edge("taxonomy", "pipeline_overview", "classified"),
    ];
    (s, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_scenario_builds() {
        let (s, _) = quality_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(s.nodes[0].data_channels.len() >= 2);
    }

    #[test]
    fn dada2_scenario_builds() {
        let (s, _) = dada2_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!s.nodes[0].data_channels.is_empty());
    }

    #[test]
    fn taxonomy_scenario_builds() {
        let (s, _) = taxonomy_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert_eq!(s.nodes[0].data_channels.len(), 2);
    }

    #[test]
    fn pipeline_overview_builds() {
        let (s, edges) = pipeline_overview_scenario();
        assert_eq!(s.nodes.len(), 1);
        assert!(!edges.is_empty());
    }
}
