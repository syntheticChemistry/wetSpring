// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! Exp327: `petalTongue` visualization module — schema compliance and control
//! validation.
//!
//! Validates that every `DataChannel` variant serializes correctly, every
//! scenario builder produces structurally valid output, and IPC params
//! conform to the `petalTongue` wire protocol.
//!
//! | Domain | Checks |
//! |--------|--------|
//! | V1 Schema     | `DataChannel` round-trip, required fields, `channel_type` tags |
//! | V2 Scenarios  | ecology, ordination, dynamics, chemistry, anderson, benchmarks |
//! | V3 IPC params | `visualization.render` / `visualization.render.stream` shapes  |
//! | V4 Full chain | `full_pipeline_scenario`, `scenario_with_edges_json`            |
//!
//! Reference: `dump_wetspring_scenarios.rs` (Exp327 companion).
//! Follows `hotSpring` validation pattern: hardcoded expected, explicit
//! pass/fail, exit code 0/1.

use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::visualization::ipc_push::PetalTonguePushClient;
use wetspring_barracuda::visualization::scenarios::{self, benchmarks::TierResult};
use wetspring_barracuda::visualization::{
    DataChannel, EcologyScenario, ScenarioNode, ScientificRange, scenario_with_edges_json,
};

fn make_test_timeseries() -> DataChannel {
    DataChannel::TimeSeries {
        id: "t1".into(),
        label: "Test TS".into(),
        x_label: "X".into(),
        y_label: "Y".into(),
        unit: "AU".into(),
        x_values: vec![1.0, 2.0, 3.0],
        y_values: vec![4.0, 5.0, 6.0],
    }
}

fn validate_schema(v: &mut Validator) -> DataChannel {
    v.section("V1 — DataChannel Schema");

    let ts = make_test_timeseries();
    let ts_json = serde_json::to_string(&ts).or_exit("unexpected error");
    v.check_pass(
        "TimeSeries has channel_type tag",
        ts_json.contains("\"channel_type\":\"timeseries\""),
    );
    v.check_pass("TimeSeries has x_values", ts_json.contains("\"x_values\""));
    v.check_pass("TimeSeries has y_values", ts_json.contains("\"y_values\""));

    let bar = DataChannel::Bar {
        id: "b1".into(),
        label: "Test Bar".into(),
        categories: vec!["A".into(), "B".into()],
        values: vec![1.0, 2.0],
        unit: "count".into(),
    };
    let bar_json = serde_json::to_string(&bar).or_exit("unexpected error");
    v.check_pass(
        "Bar has channel_type tag",
        bar_json.contains("\"channel_type\":\"bar\""),
    );
    v.check_pass("Bar has categories", bar_json.contains("\"categories\""));

    let gauge = DataChannel::Gauge {
        id: "g1".into(),
        label: "Test Gauge".into(),
        value: 50.0,
        min: 0.0,
        max: 100.0,
        unit: "%".into(),
        normal_range: [20.0, 80.0],
        warning_range: [10.0, 20.0],
    };
    let gauge_json = serde_json::to_string(&gauge).or_exit("unexpected error");
    v.check_pass(
        "Gauge has channel_type tag",
        gauge_json.contains("\"channel_type\":\"gauge\""),
    );
    v.check_pass(
        "Gauge has normal_range",
        gauge_json.contains("\"normal_range\""),
    );

    let heatmap = DataChannel::Heatmap {
        id: "h1".into(),
        label: "Test Heatmap".into(),
        x_labels: vec!["X1".into()],
        y_labels: vec!["Y1".into()],
        values: vec![0.5],
        unit: "index".into(),
    };
    let hm_json = serde_json::to_string(&heatmap).or_exit("unexpected error");
    v.check_pass(
        "Heatmap has channel_type tag",
        hm_json.contains("\"channel_type\":\"heatmap\""),
    );

    let scatter_ch = DataChannel::Scatter {
        id: "s1".into(),
        label: "Test Scatter".into(),
        x: vec![1.0],
        y: vec![2.0],
        point_labels: vec![],
        x_label: "PC1".into(),
        y_label: "PC2".into(),
        unit: "AU".into(),
    };
    let sc_json = serde_json::to_string(&scatter_ch).or_exit("unexpected error");
    v.check_pass(
        "Scatter has channel_type tag",
        sc_json.contains("\"channel_type\":\"scatter\""),
    );
    v.check_pass(
        "Scatter omits empty point_labels",
        !sc_json.contains("point_labels"),
    );

    let dist = DataChannel::Distribution {
        id: "d1".into(),
        label: "Test Dist".into(),
        unit: "AU".into(),
        values: vec![1.0, 2.0, 3.0],
        mean: 2.0,
        std: 0.816,
    };
    let dist_json = serde_json::to_string(&dist).or_exit("unexpected error");
    v.check_pass(
        "Distribution has channel_type tag",
        dist_json.contains("\"channel_type\":\"distribution\""),
    );

    ts
}

fn validate_scenario_builders(v: &mut Validator, samples: &[Vec<f64>], labels: &[String]) {
    v.section("V2 — Scenario Builders");

    let (eco, eco_edges) = scenarios::ecology_scenario(samples, labels);
    v.check_pass("ecology: domain = ecology", eco.domain == "ecology");
    v.check_count("ecology: ≥2 nodes (alpha + beta)", eco.nodes.len(), 2);
    v.check_pass(
        "ecology: alpha node has channels",
        !eco.nodes[0].data_channels.is_empty(),
    );
    v.check_pass("ecology: has edges for multi-sample", !eco_edges.is_empty());
    v.check_pass(
        "ecology: alpha_metrics bar present",
        eco.nodes[0]
            .data_channels
            .iter()
            .any(|ch| matches!(ch, DataChannel::Bar { id, .. } if id == "alpha_metrics")),
    );
    v.check_pass(
        "ecology: rarefaction timeseries present",
        eco.nodes[0]
            .data_channels
            .iter()
            .any(|ch| matches!(ch, DataChannel::TimeSeries { id, .. } if id == "rarefaction")),
    );
    v.check_pass(
        "ecology: beta node has heatmap",
        eco.nodes.get(1).is_some_and(|n| {
            n.data_channels
                .iter()
                .any(|ch| matches!(ch, DataChannel::Heatmap { .. }))
        }),
    );

    let dm = vec![0.5, 0.8, 0.6];
    let ord_labels: Vec<String> = (1..=3).map(|i| format!("S{i}")).collect();
    let ord_result = scenarios::ordination_scenario(&dm, 3, &ord_labels);
    v.check_pass("ordination: succeeds", ord_result.is_ok());
    if let Ok((ord, _)) = &ord_result {
        v.check_pass(
            "ordination: has scatter channel",
            ord.nodes[0]
                .data_channels
                .iter()
                .any(|ch| matches!(ch, DataChannel::Scatter { id, .. } if id == "pcoa_scatter")),
        );
        v.check_pass(
            "ordination: has scree bar",
            ord.nodes[0]
                .data_channels
                .iter()
                .any(|ch| matches!(ch, DataChannel::Bar { id, .. } if id == "scree")),
        );
    }

    let (dyn_s, dyn_edges) = scenarios::dynamics_scenario();
    v.check_pass("dynamics: ≥2 nodes (QS + bistable)", dyn_s.nodes.len() >= 2);
    v.check_count(
        "dynamics: QS has 5 timeseries",
        dyn_s.nodes[0].data_channels.len(),
        5,
    );
    v.check_pass(
        "dynamics: all timeseries channels",
        dyn_s.nodes[0]
            .data_channels
            .iter()
            .all(|ch| matches!(ch, DataChannel::TimeSeries { .. })),
    );
    v.check_pass("dynamics: has edges", !dyn_edges.is_empty());

    let (and, _) = scenarios::anderson_scenario(0.53, &[10.0, 12.0, 14.0], &[0.0, 1.0, 2.0]);
    v.check_pass(
        "anderson: has gauge channel",
        and.nodes[0]
            .data_channels
            .iter()
            .any(|ch| matches!(ch, DataChannel::Gauge { id, .. } if id == "level_spacing")),
    );
    v.check_pass(
        "anderson: has W(t) timeseries",
        and.nodes[0]
            .data_channels
            .iter()
            .any(|ch| matches!(ch, DataChannel::TimeSeries { id, .. } if id == "w_t_curve")),
    );
    v.check_pass(
        "anderson: has scientific range",
        !and.nodes[0].scientific_ranges.is_empty(),
    );

    let tiers = vec![
        TierResult {
            stage: "DADA2".into(),
            galaxy_s: 6.8,
            cpu_s: 0.33,
            gpu_s: 0.013,
        },
        TierResult {
            stage: "Total".into(),
            galaxy_s: 31.9,
            cpu_s: 1.2,
            gpu_s: 0.05,
        },
    ];
    let (bench, _) = scenarios::benchmark_scenario(&tiers);
    v.check_pass("benchmark: domain = default", bench.domain == "default");
    v.check_count(
        "benchmark: has 4 bar channels",
        bench.nodes[0].data_channels.len(),
        4,
    );
}

fn validate_ipc(v: &mut Validator, ts: &DataChannel) {
    v.section("V3 — IPC Param Shapes");

    let test_scenario = EcologyScenario {
        name: "ipc_test".into(),
        description: "test".into(),
        version: "1.0.0".into(),
        mode: "static".into(),
        domain: "ecology".into(),
        nodes: vec![ScenarioNode {
            id: "n1".into(),
            name: "N1".into(),
            node_type: "compute".into(),
            family: wetspring_barracuda::PRIMAL_NAME.into(),
            status: "healthy".into(),
            health: 100,
            confidence: 100,
            capabilities: vec!["science.test".into()],
            data_channels: vec![ts.clone()],
            scientific_ranges: vec![],
        }],
        edges: vec![],
    };

    let render_json = serde_json::to_string(&test_scenario).or_exit("unexpected error");
    v.check_pass("IPC render: scenario serializes", !render_json.is_empty());
    v.check_pass(
        "IPC render: contains nodes",
        render_json.contains("\"nodes\""),
    );
    v.check_pass(
        "IPC render: contains domain",
        render_json.contains("\"domain\":\"ecology\""),
    );

    let explicit_client =
        PetalTonguePushClient::new(std::env::temp_dir().join("nonexistent-petaltongue.sock"));
    v.check_pass(
        "IPC push_render: fails on missing socket",
        explicit_client
            .push_render("s1", "t", &test_scenario)
            .is_err(),
    );
}

fn validate_full_chain(v: &mut Validator, samples: &[Vec<f64>], labels: &[String]) {
    v.section("V4 — Full Chain");

    let (full, full_edges) = scenarios::full_pipeline_scenario(samples, labels);
    v.check_pass("full_pipeline: ≥3 nodes", full.nodes.len() >= 3);
    v.check_pass("full_pipeline: merges ecology + dynamics", {
        let has_diversity = full.nodes.iter().any(|n| n.id == "diversity");
        let has_qs = full.nodes.iter().any(|n| n.id == "qs_ode");
        has_diversity && has_qs
    });
    v.check_pass(
        "full_pipeline: cross-domain edges",
        full_edges
            .iter()
            .any(|e| e.from == "diversity" && e.to == "qs_ode"),
    );

    let json_result = scenario_with_edges_json(&full, &full_edges);
    v.check_pass("scenario_with_edges_json: succeeds", json_result.is_ok());
    if let Ok(ref json) = json_result {
        v.check_pass("JSON: contains edges", json.contains("\"edges\""));
        v.check_pass("JSON: contains nodes", json.contains("\"nodes\""));
        v.check_pass(
            "JSON: valid parse",
            serde_json::from_str::<serde_json::Value>(json).is_ok(),
        );
    }

    v.section("V5 — Metadata Compliance");
    v.check_pass("ScientificRange serializes", {
        let r = ScientificRange {
            label: "Optimal".into(),
            min: 3.0,
            max: 6.0,
            status: "normal".into(),
        };
        serde_json::to_string(&r).is_ok()
    });
    v.check_pass(
        "node family is wetspring",
        full.nodes
            .iter()
            .all(|n| n.family == wetspring_barracuda::PRIMAL_NAME),
    );
    v.check_pass("version is semver", full.version.split('.').count() == 3);
    v.check_pass("mode is live-ecosystem", full.mode == "live-ecosystem");
}

fn main() {
    let mut v = Validator::new("Exp327: petalTongue Visualization V1");

    let samples = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0, 25.0],
        vec![15.0, 25.0, 5.0, 35.0, 45.0, 10.0, 20.0, 30.0],
        vec![8.0, 12.0, 40.0, 20.0, 60.0, 3.0, 7.0, 50.0],
    ];
    let labels: Vec<String> = (1..=3).map(|i| format!("Sample{i}")).collect();

    let ts = validate_schema(&mut v);
    validate_scenario_builders(&mut v, &samples, &labels);
    validate_ipc(&mut v, &ts);
    validate_full_chain(&mut v, &samples, &labels);

    v.finish();
}
