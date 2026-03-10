// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming pipeline scenario: multi-stage GPU pipeline visualization.
//!
//! Maps pipeline stages (QF → DADA2 → taxonomy → diversity → Bray-Curtis)
//! into a multi-node scenario graph suitable for progressive streaming
//! via [`StreamSession`](crate::visualization::stream::StreamSession).

use crate::visualization::types::{EcologyScenario, ScenarioEdge};
use crate::visualization::ScientificRange;

use super::{bar, edge, gauge, node, scaffold};

/// Pipeline stage metadata for visualization.
pub struct PipelineStage {
    /// Stage identifier.
    pub id: String,
    /// Human-readable stage name.
    pub name: String,
    /// Processing time in milliseconds.
    pub latency_ms: f64,
    /// Items processed (reads, ASVs, etc.).
    pub items_processed: usize,
    /// Items output (may differ from input, e.g., after filtering).
    pub items_output: usize,
    /// Quality pass rate (0.0-1.0), if applicable.
    pub pass_rate: Option<f64>,
}

/// Build a streaming pipeline visualization scenario.
///
/// Produces a multi-node graph with per-stage metrics:
/// - **`TimeSeries`**: per-stage latency over batches
/// - **Bar**: items processed vs output per stage
/// - **Gauge**: quality pass rate, pipeline progress
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "stage counts ≤ dozens")]
pub fn streaming_pipeline_scenario(
    stages: &[PipelineStage],
) -> (EcologyScenario, Vec<ScenarioEdge>) {
    let mut s = scaffold(
        "wetSpring Streaming Pipeline",
        "Multi-stage GPU pipeline: QF → DADA2 → taxonomy → diversity → β-diversity",
    );

    let total_stages = stages.len();
    let mut edges = Vec::new();

    let stage_names: Vec<String> = stages.iter().map(|st| st.name.clone()).collect();
    let latencies: Vec<f64> = stages.iter().map(|st| st.latency_ms).collect();

    let mut overview = node(
        "pipeline_overview",
        "Pipeline Overview",
        "pipeline",
        &["science.streaming_pipeline"],
    );
    overview.data_channels.push(bar(
        "stage_latencies",
        "Stage Latencies",
        &stage_names,
        &latencies,
        "ms",
    ));

    let total_latency: f64 = latencies.iter().sum();
    overview.data_channels.push(gauge(
        "total_latency",
        "Total Pipeline Latency",
        total_latency,
        0.0,
        total_latency * 2.0,
        "ms",
        [0.0, total_latency * 1.5],
        [total_latency * 1.5, total_latency * 2.0],
    ));

    s.nodes.push(overview);

    let mut prev_id: Option<String> = Some("pipeline_overview".into());

    for (i, stage) in stages.iter().enumerate() {
        let mut stage_node = node(&stage.id, &stage.name, "compute", &[]);

        let progress = (i + 1) as f64 / total_stages as f64 * 100.0;
        stage_node.data_channels.push(gauge(
            &format!("{}_progress", stage.id),
            &format!("{} Progress", stage.name),
            progress,
            0.0,
            100.0,
            "%",
            [0.0, 100.0],
            [0.0, 0.0],
        ));

        stage_node.data_channels.push(bar(
            &format!("{}_throughput", stage.id),
            &format!("{} Throughput", stage.name),
            &["Input", "Output"],
            &[stage.items_processed as f64, stage.items_output as f64],
            "items",
        ));

        if let Some(rate) = stage.pass_rate {
            stage_node.data_channels.push(gauge(
                &format!("{}_pass_rate", stage.id),
                &format!("{} Pass Rate", stage.name),
                rate * 100.0,
                0.0,
                100.0,
                "%",
                [80.0, 100.0],
                [50.0, 80.0],
            ));
            stage_node.scientific_ranges.push(ScientificRange {
                label: "Quality pass rate 80–100%".into(),
                min: 80.0,
                max: 100.0,
                status: "normal".into(),
            });
            stage_node.scientific_ranges.push(ScientificRange {
                label: "Quality pass rate 0–80%".into(),
                min: 0.0,
                max: 80.0,
                status: "warning".into(),
            });
        }

        if let Some(ref prev) = prev_id {
            edges.push(edge(prev, &stage.id, &format!("{} → {}", prev, stage.name)));
        }
        prev_id = Some(stage.id.clone());

        s.nodes.push(stage_node);
    }

    (s, edges)
}

/// Build a demo streaming pipeline scenario with typical metagenomics stages.
#[must_use]
pub fn demo_streaming_pipeline_scenario() -> (EcologyScenario, Vec<ScenarioEdge>) {
    let stages = vec![
        PipelineStage {
            id: "qf".into(),
            name: "Quality Filter".into(),
            latency_ms: 12.5,
            items_processed: 10_000,
            items_output: 8_500,
            pass_rate: Some(0.85),
        },
        PipelineStage {
            id: "dada2".into(),
            name: "DADA2 Denoising".into(),
            latency_ms: 45.2,
            items_processed: 8_500,
            items_output: 250,
            pass_rate: None,
        },
        PipelineStage {
            id: "taxonomy".into(),
            name: "Taxonomy Classification".into(),
            latency_ms: 28.7,
            items_processed: 250,
            items_output: 250,
            pass_rate: Some(0.92),
        },
        PipelineStage {
            id: "diversity".into(),
            name: "Alpha Diversity".into(),
            latency_ms: 3.1,
            items_processed: 250,
            items_output: 4,
            pass_rate: None,
        },
        PipelineStage {
            id: "beta".into(),
            name: "Bray-Curtis β-diversity".into(),
            latency_ms: 8.4,
            items_processed: 4,
            items_output: 1,
            pass_rate: None,
        },
    ];
    streaming_pipeline_scenario(&stages)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use unwrap/expect for clarity")]
mod tests {
    use super::*;

    #[test]
    fn demo_pipeline_produces_nodes() {
        let (scenario, edges) = demo_streaming_pipeline_scenario();
        assert!(scenario.nodes.len() >= 6);
        assert!(!edges.is_empty());
    }

    #[test]
    fn pipeline_serializes() {
        let (scenario, _) = demo_streaming_pipeline_scenario();
        let json = serde_json::to_string(&scenario).expect("serialize");
        assert!(json.contains("Quality Filter"));
        assert!(json.contains("DADA2"));
        assert!(json.contains("stage_latencies"));
    }

    #[test]
    fn custom_stages_work() {
        let stages = vec![
            PipelineStage {
                id: "step1".into(),
                name: "Step 1".into(),
                latency_ms: 5.0,
                items_processed: 100,
                items_output: 90,
                pass_rate: Some(0.9),
            },
            PipelineStage {
                id: "step2".into(),
                name: "Step 2".into(),
                latency_ms: 10.0,
                items_processed: 90,
                items_output: 80,
                pass_rate: None,
            },
        ];
        let (scenario, edges) = streaming_pipeline_scenario(&stages);
        assert_eq!(scenario.nodes.len(), 3);
        assert_eq!(edges.len(), 2);
    }
}
