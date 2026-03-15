// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dorado basecalling scenario: QC metrics, read length distribution, quality.
//!
//! Wraps [`crate::bio::dorado`] output into a petalTongue-ready scenario
//! for monitoring nanopore basecalling results.

use crate::visualization::types::{EcologyScenario, ScenarioEdge, ScientificRange};

use super::{distribution, gauge, node, scaffold};

/// Build a basecalling QC scenario from dorado output metrics.
///
/// Takes summary statistics from a basecalling run and produces:
/// - **Gauge**: total reads, mean quality, pass rate
/// - **Distribution**: read length distribution
#[must_use]
#[expect(clippy::cast_precision_loss)] // Precision: total_reads, passed_reads fit f64
pub fn basecalling_scenario(
    total_reads: usize,
    passed_reads: usize,
    read_lengths: &[f64],
    mean_quality: f64,
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "Nanopore Basecalling QC",
        &format!("{total_reads} total reads, {passed_reads} passed (Q >= 10)"),
    );

    let mut qc_node = node("basecall_qc", "Basecalling QC", "compute", &["basecalling"]);

    let pass_rate = if total_reads > 0 {
        passed_reads as f64 / total_reads as f64
    } else {
        0.0
    };
    qc_node.data_channels.push(gauge(
        "pass_rate",
        "Pass Rate (Q >= 10)",
        pass_rate,
        0.0,
        1.0,
        "fraction",
        [0.8, 1.0],
        [0.0, 0.8],
    ));
    qc_node.data_channels.push(gauge(
        "mean_quality",
        "Mean Quality Score",
        mean_quality,
        0.0,
        40.0,
        "Q",
        [10.0, 40.0],
        [0.0, 10.0],
    ));

    if !read_lengths.is_empty() {
        let mean_len = read_lengths.iter().sum::<f64>() / read_lengths.len() as f64;
        let variance = read_lengths
            .iter()
            .map(|&l| (l - mean_len).powi(2))
            .sum::<f64>()
            / read_lengths.len() as f64;
        let std_dev = variance.sqrt();

        qc_node.data_channels.push(distribution(
            "read_lengths",
            "Read Length Distribution",
            "bp",
            read_lengths,
            mean_len,
            std_dev,
        ));
    }

    qc_node.scientific_ranges.push(ScientificRange {
        label: "Good quality".into(),
        min: 10.0,
        max: 40.0,
        status: "normal".into(),
    });
    qc_node.scientific_ranges.push(ScientificRange {
        label: "Low quality".into(),
        min: 0.0,
        max: 10.0,
        status: "warning".into(),
    });

    s.nodes.push(qc_node);
    (s, vec![])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basecalling_basic() {
        let lengths = vec![500.0, 1200.0, 800.0, 3000.0, 1500.0];
        let (scenario, _) = basecalling_scenario(1000, 850, &lengths, 12.5);
        assert_eq!(scenario.nodes.len(), 1);
        let n = &scenario.nodes[0];
        assert!(n.data_channels.len() >= 3);
        assert!(!n.scientific_ranges.is_empty());
    }

    #[test]
    fn basecalling_empty_reads() {
        let (scenario, _) = basecalling_scenario(0, 0, &[], 0.0);
        assert_eq!(scenario.nodes.len(), 1);
    }

    #[test]
    fn basecalling_pass_rate_gauge() {
        let (scenario, _) = basecalling_scenario(100, 90, &[], 15.0);
        let n = &scenario.nodes[0];
        assert_eq!(n.data_channels.len(), 2);
    }
}
