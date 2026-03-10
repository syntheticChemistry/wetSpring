// SPDX-License-Identifier: AGPL-3.0-or-later
//! `petalTongue`-compatible scenario export for wetSpring pipelines.
//!
//! Produces two output modes:
//! 1. **JSON file** — `petalTongue --scenario <path>` loads scenarios directly
//! 2. **IPC push** — JSON-RPC over Unix socket (`visualization.render`)
//!
//! No `petalTongue` crate dependency — integration is via JSON schema only.
//! All scenarios are built from live `barraCuda` math, never mock data.

pub mod capabilities;
pub mod ipc_push;
pub mod live_pipeline;
pub mod scenarios;
pub mod stream;
mod types;

pub use types::*;

/// Serialize a scenario to pretty-printed JSON.
///
/// # Errors
///
/// Returns `serde_json::Error` if serialization fails (should not happen
/// for well-formed scenarios).
pub fn scenario_to_json(scenario: &EcologyScenario) -> serde_json::Result<String> {
    serde_json::to_string_pretty(scenario)
}

/// Serialize a scenario with edges merged into the JSON output.
///
/// # Errors
///
/// Returns `serde_json::Error` if serialization fails.
pub fn scenario_with_edges_json(
    scenario: &EcologyScenario,
    edges: &[ScenarioEdge],
) -> serde_json::Result<String> {
    let mut combined = scenario.clone();
    combined.edges = edges.to_vec();
    scenario_to_json(&combined)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests use unwrap/expect for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_scenario() {
        let s = EcologyScenario {
            name: "test".into(),
            description: "empty".into(),
            version: "1.0.0".into(),
            mode: "static".into(),
            domain: "ecology".into(),
            nodes: vec![],
            edges: vec![],
        };
        let json = scenario_to_json(&s).unwrap();
        assert!(json.contains("\"name\": \"test\""));
        assert!(json.contains("\"domain\": \"ecology\""));
    }

    #[test]
    fn scenario_with_edges_includes_edges() {
        let s = EcologyScenario {
            name: "t".into(),
            description: "d".into(),
            version: "1.0.0".into(),
            mode: "static".into(),
            domain: "ecology".into(),
            nodes: vec![],
            edges: vec![],
        };
        let edges = vec![ScenarioEdge {
            from: "a".into(),
            to: "b".into(),
            edge_type: "data_flow".into(),
            label: "test edge".into(),
        }];
        let json = scenario_with_edges_json(&s, &edges).unwrap();
        assert!(json.contains("\"from\": \"a\""));
    }

    #[test]
    fn data_channel_timeseries_serializes() {
        let ch = DataChannel::TimeSeries {
            id: "ts1".into(),
            label: "Test".into(),
            x_label: "X".into(),
            y_label: "Y".into(),
            unit: "AU".into(),
            x_values: vec![1.0, 2.0],
            y_values: vec![3.0, 4.0],
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"timeseries\""));
    }

    #[test]
    fn data_channel_bar_serializes() {
        let ch = DataChannel::Bar {
            id: "b1".into(),
            label: "Test".into(),
            categories: vec!["A".into(), "B".into()],
            values: vec![1.0, 2.0],
            unit: "count".into(),
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"bar\""));
    }

    #[test]
    fn data_channel_gauge_serializes() {
        let ch = DataChannel::Gauge {
            id: "g1".into(),
            label: "Test".into(),
            value: 50.0,
            min: 0.0,
            max: 100.0,
            unit: "%".into(),
            normal_range: [20.0, 80.0],
            warning_range: [10.0, 20.0],
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"gauge\""));
    }

    #[test]
    fn data_channel_heatmap_serializes() {
        let ch = DataChannel::Heatmap {
            id: "h1".into(),
            label: "Test".into(),
            x_labels: vec!["A".into()],
            y_labels: vec!["B".into()],
            values: vec![0.5],
            unit: "index".into(),
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"heatmap\""));
    }

    #[test]
    fn data_channel_scatter_serializes() {
        let ch = DataChannel::Scatter {
            id: "s1".into(),
            label: "Test".into(),
            x: vec![1.0],
            y: vec![2.0],
            point_labels: vec![],
            x_label: "X".into(),
            y_label: "Y".into(),
            unit: "AU".into(),
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"scatter\""));
        assert!(!json.contains("point_labels"));
    }

    #[test]
    fn data_channel_distribution_serializes() {
        let ch = DataChannel::Distribution {
            id: "d1".into(),
            label: "Test".into(),
            unit: "AU".into(),
            values: vec![1.0, 2.0, 3.0],
            mean: 2.0,
            std: 0.816,
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"distribution\""));
    }

    #[test]
    fn data_channel_spectrum_serializes() {
        let ch = DataChannel::Spectrum {
            id: "sp1".into(),
            label: "Test Spectrum".into(),
            unit: "dB".into(),
            frequencies: vec![100.0, 200.0, 300.0],
            amplitudes: vec![0.5, 0.8, 0.3],
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"spectrum\""));
        assert!(json.contains("\"frequencies\""));
        assert!(json.contains("\"amplitudes\""));
    }

    #[test]
    fn data_channel_scatter3d_serializes() {
        let ch = DataChannel::Scatter3D {
            id: "s3d".into(),
            label: "3D PCoA".into(),
            x: vec![1.0, 2.0],
            y: vec![3.0, 4.0],
            z: vec![5.0, 6.0],
            point_labels: vec!["A".into(), "B".into()],
            x_label: "PC1".into(),
            y_label: "PC2".into(),
            z_label: "PC3".into(),
            unit: "proportion".into(),
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"scatter3d\""));
        assert!(json.contains("\"z\""));
        assert!(json.contains("\"z_label\""));
    }

    #[test]
    fn data_channel_fieldmap_serializes() {
        let ch = DataChannel::FieldMap {
            id: "fm1".into(),
            label: "Test Field".into(),
            grid_x: vec![0.0, 1.0],
            grid_y: vec![0.0, 1.0],
            values: vec![1.0, 2.0, 3.0, 4.0],
            unit: "mg/L".into(),
        };
        let json = serde_json::to_string(&ch).expect("serialize");
        assert!(json.contains("\"channel_type\":\"fieldmap\""));
        assert!(json.contains("\"grid_x\""));
    }

    #[test]
    fn scientific_range_serializes() {
        let r = ScientificRange {
            label: "Optimal".into(),
            min: 3.0,
            max: 6.0,
            status: "normal".into(),
        };
        let json = serde_json::to_string(&r).expect("serialize");
        assert!(json.contains("\"label\":\"Optimal\""));
    }

    #[test]
    fn ui_config_default() {
        let cfg = UiConfig::default();
        assert_eq!(cfg.theme, "ecology-dark");
        assert!(cfg.show_panels.left_sidebar);
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(json.contains("\"theme\":\"ecology-dark\""));
    }
}
