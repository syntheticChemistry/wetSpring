// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pangenome scenario: presence/absence heatmap, core/accessory bars, Heap's alpha.

use crate::bio::pangenome::{self, GeneCluster, PangenomeResult};
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, gauge, heatmap, node, scaffold};

/// Build a pangenome visualization scenario from gene clusters.
///
/// Produces:
/// - **Heatmap**: gene presence/absence matrix
/// - **Bar**: core / accessory / unique gene counts
/// - **Gauge**: Heap's alpha (pangenome openness)
#[must_use]
pub fn pangenome_scenario(
    clusters: &[GeneCluster],
    n_genomes: usize,
    genome_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Pangenome",
        "Gene presence/absence, core/accessory/unique, Heap's alpha",
    );

    let result = pangenome::analyze(clusters, n_genomes);

    let mut pan_node = node(
        "pangenome",
        "Pangenome Analysis",
        "compute",
        &["science.pangenome"],
    );

    build_presence_heatmap(&mut pan_node, clusters, n_genomes, genome_labels);
    build_gene_counts_bar(&mut pan_node, &result);
    build_heaps_gauge(&mut pan_node, &result);

    pan_node.scientific_ranges.push(ScientificRange {
        label: "Open pangenome".into(),
        min: 0.0,
        max: 0.6,
        status: "normal".into(),
    });
    pan_node.scientific_ranges.push(ScientificRange {
        label: "Closed pangenome".into(),
        min: 0.6,
        max: 1.0,
        status: "warning".into(),
    });

    s.nodes.push(pan_node);
    (s, vec![])
}

fn build_presence_heatmap(
    pan_node: &mut crate::visualization::ScenarioNode,
    clusters: &[GeneCluster],
    n_genomes: usize,
    genome_labels: &[String],
) {
    let flat = pangenome::presence_matrix_flat(clusters, n_genomes);
    let values: Vec<f64> = flat.iter().map(|&b| f64::from(b)).collect();

    let gene_labels: Vec<String> = clusters
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if c.id.is_empty() {
                format!("gene_{i}")
            } else {
                c.id.clone()
            }
        })
        .collect();

    pan_node.data_channels.push(heatmap(
        "presence_absence",
        "Gene Presence/Absence",
        genome_labels,
        &gene_labels,
        &values,
        "present",
    ));
}

#[expect(clippy::cast_precision_loss, reason = "gene counts ≤ millions")]
fn build_gene_counts_bar(
    pan_node: &mut crate::visualization::ScenarioNode,
    result: &PangenomeResult,
) {
    pan_node.data_channels.push(bar(
        "gene_counts",
        "Gene Category Counts",
        &["Core", "Accessory", "Unique"],
        &[
            result.core_size as f64,
            result.accessory_size as f64,
            result.unique_size as f64,
        ],
        "genes",
    ));
}

fn build_heaps_gauge(pan_node: &mut crate::visualization::ScenarioNode, result: &PangenomeResult) {
    let alpha = result.heaps_alpha.unwrap_or(0.0);
    pan_node.data_channels.push(gauge(
        "heaps_alpha",
        "Heap's Alpha (Openness)",
        alpha,
        0.0,
        1.0,
        "α",
        [0.0, 0.6],
        [0.6, 1.0],
    ));
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use unwrap/expect for clarity")]
mod tests {
    use super::*;

    fn sample_clusters() -> Vec<GeneCluster> {
        vec![
            GeneCluster {
                id: "geneA".into(),
                presence: vec![true, true, true],
            },
            GeneCluster {
                id: "geneB".into(),
                presence: vec![true, false, true],
            },
            GeneCluster {
                id: "geneC".into(),
                presence: vec![false, false, true],
            },
        ]
    }

    #[test]
    fn pangenome_produces_channels() {
        let clusters = sample_clusters();
        let labels = vec!["G1".into(), "G2".into(), "G3".into()];
        let (scenario, edges) = pangenome_scenario(&clusters, 3, &labels);
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 3);
        assert!(edges.is_empty());
    }

    #[test]
    fn pangenome_serializes() {
        let clusters = sample_clusters();
        let labels = vec!["G1".into(), "G2".into(), "G3".into()];
        let (scenario, _) = pangenome_scenario(&clusters, 3, &labels);
        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("presence_absence"));
        assert!(json.contains("gene_counts"));
        assert!(json.contains("heaps_alpha"));
    }
}
