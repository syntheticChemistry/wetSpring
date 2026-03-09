// SPDX-License-Identifier: AGPL-3.0-or-later
//! Similarity scenario: ANI pairwise heatmap and distribution.

use crate::bio::ani;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{distribution, heatmap, node, scaffold};

/// Build a pairwise ANI similarity scenario.
///
/// Produces:
/// - **Heatmap**: ANI pairwise distance matrix
/// - **Distribution**: ANI value distribution across all pairs
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "pair counts ≤ millions")]
pub fn similarity_scenario(
    sequences: &[&[u8]],
    genome_labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring ANI Similarity",
        "Average Nucleotide Identity pairwise matrix",
    );

    let mut sim_node = node(
        "similarity",
        "ANI Pairwise Similarity",
        "compute",
        &["science.similarity"],
    );

    let condensed = ani::ani_matrix(sequences);
    let n = sequences.len();

    let mut full_matrix = vec![1.0_f64; n * n];
    let mut idx = 0;
    for i in 1..n {
        for j in 0..i {
            full_matrix[i * n + j] = condensed[idx];
            full_matrix[j * n + i] = condensed[idx];
            idx += 1;
        }
    }

    sim_node.data_channels.push(heatmap(
        "ani_matrix",
        "ANI Pairwise Matrix",
        genome_labels,
        genome_labels,
        &full_matrix,
        "ANI",
    ));

    let off_diag = &condensed;

    if !off_diag.is_empty() {
        let sum: f64 = off_diag.iter().sum();
        let mean = sum / off_diag.len() as f64;
        let var = off_diag.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / off_diag.len() as f64;
        let std = var.sqrt();

        sim_node.data_channels.push(distribution(
            "ani_distribution",
            "ANI Value Distribution",
            "ANI",
            off_diag,
            mean,
            std,
        ));
    }

    sim_node.scientific_ranges.push(ScientificRange {
        label: "Same species (ANI ≥ 95%)".into(),
        min: 0.95,
        max: 1.0,
        status: "normal".into(),
    });
    sim_node.scientific_ranges.push(ScientificRange {
        label: "Same genus (ANI 80-95%)".into(),
        min: 0.80,
        max: 0.95,
        status: "warning".into(),
    });

    s.nodes.push(sim_node);
    (s, vec![])
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests use unwrap/expect for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn similarity_produces_channels() {
        let s1 = b"ATCGATCGATCG";
        let s2 = b"ATCGATCGATCC";
        let s3 = b"ATCGATCGATCA";
        let seqs: Vec<&[u8]> = vec![s1, s2, s3];
        let labels = vec!["G1".into(), "G2".into(), "G3".into()];
        let (scenario, edges) = similarity_scenario(&seqs, &labels);
        assert_eq!(scenario.nodes.len(), 1);
        assert_eq!(scenario.nodes[0].data_channels.len(), 2);
        assert!(edges.is_empty());
    }

    #[test]
    fn similarity_serializes() {
        let s1 = b"ATCGATCG";
        let s2 = b"ATCGATCC";
        let seqs: Vec<&[u8]> = vec![s1, s2];
        let labels = vec!["A".into(), "B".into()];
        let (scenario, _) = similarity_scenario(&seqs, &labels);
        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("ani_matrix"));
    }
}
