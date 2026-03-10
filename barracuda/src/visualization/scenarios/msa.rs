// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multiple Sequence Alignment scenario: conservation, identity heatmap.
//!
//! Wraps [`crate::bio::msa`] output into a petalTongue-ready scenario with
//! per-position conservation, pairwise identity heatmap, and mean identity gauge.

use crate::bio::msa;
use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{bar, gauge, heatmap, node, scaffold};

/// Build an MSA scenario from a set of sequences.
///
/// Performs progressive alignment, then extracts per-column conservation
/// scores and a pairwise identity matrix.
///
/// # Panics
///
/// Panics if fewer than 2 sequences are provided.
#[must_use]
#[expect(clippy::cast_precision_loss, clippy::naive_bytecount)]
pub fn msa_scenario(
    sequences: &[&[u8]],
    labels: &[String],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let result = msa::align_multiple(sequences, labels, &msa::MsaParams::default());
    let n = result.aligned.len();
    let cols = result.alignment_length;

    let mut s = scaffold(
        "wetSpring MSA Alignment",
        &format!("{n} sequences, {cols} columns aligned"),
    );

    let mut msa_node = node("msa", "Multiple Sequence Alignment", "compute", &["msa"]);

    // Per-column conservation score
    let positions: Vec<String> = (1..=cols).map(|i| format!("{i}")).collect();
    let conservation: Vec<f64> = (0..cols)
        .map(|col| {
            let residues: Vec<u8> = result
                .aligned
                .iter()
                .map(|seq| seq[col])
                .filter(|&b| b != b'-')
                .collect();
            if residues.is_empty() {
                return 0.0;
            }
            let most_common = *residues
                .iter()
                .max_by_key(|&&b| residues.iter().filter(|&&r| r == b).count())
                .unwrap_or(&0);
            let count = residues.iter().filter(|&&r| r == most_common).count();
            count as f64 / n as f64
        })
        .collect();
    msa_node.data_channels.push(bar(
        "conservation",
        "Per-Position Conservation",
        &positions,
        &conservation,
        "fraction",
    ));

    // Pairwise identity heatmap
    let mut identities = Vec::with_capacity(n * n);
    for a in &result.aligned {
        for b in &result.aligned {
            let matches = a.iter().zip(b.iter()).filter(|&(&x, &y)| x == y && x != b'-').count();
            let compared = a
                .iter()
                .zip(b.iter())
                .filter(|&(&x, &y)| x != b'-' || y != b'-')
                .count();
            let identity = if compared > 0 {
                matches as f64 / compared as f64
            } else {
                0.0
            };
            identities.push(identity);
        }
    }
    msa_node.data_channels.push(heatmap(
        "pairwise_identity",
        "Pairwise Sequence Identity",
        labels,
        labels,
        &identities,
        "fraction",
    ));

    // Mean identity gauge
    let off_diag: Vec<f64> = identities
        .iter()
        .enumerate()
        .filter(|&(idx, _)| idx / n != idx % n)
        .map(|(_, &v)| v)
        .collect();
    let mean_identity = if off_diag.is_empty() {
        0.0
    } else {
        off_diag.iter().sum::<f64>() / off_diag.len() as f64
    };
    msa_node.data_channels.push(gauge(
        "mean_identity",
        "Mean Pairwise Identity",
        mean_identity,
        0.0,
        1.0,
        "fraction",
        [0.7, 1.0],
        [0.0, 0.7],
    ));

    msa_node.scientific_ranges.push(ScientificRange {
        label: "High conservation".into(),
        min: 0.7,
        max: 1.0,
        status: "normal".into(),
    });
    msa_node.scientific_ranges.push(ScientificRange {
        label: "Low conservation".into(),
        min: 0.0,
        max: 0.7,
        status: "warning".into(),
    });

    s.nodes.push(msa_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msa_scenario_basic() {
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACCTACGT", b"ACGTACCT"];
        let labels: Vec<String> = vec!["Seq1".into(), "Seq2".into(), "Seq3".into()];
        let (scenario, _) = msa_scenario(&seqs, &labels);
        assert_eq!(scenario.nodes.len(), 1);
        let n = &scenario.nodes[0];
        assert_eq!(n.data_channels.len(), 3);
        assert!(!n.scientific_ranges.is_empty());
    }

    #[test]
    fn msa_scenario_two_sequences() {
        let seqs: Vec<&[u8]> = vec![b"ACGT", b"ACGT"];
        let labels = vec!["A".into(), "B".into()];
        let (scenario, _) = msa_scenario(&seqs, &labels);
        assert_eq!(scenario.nodes[0].data_channels.len(), 3);
    }
}
