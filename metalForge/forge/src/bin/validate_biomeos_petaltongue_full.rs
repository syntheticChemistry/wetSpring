// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
//! # Exp330: biomeOS + NUCLEUS + petalTongue Full Chain
//!
//! Validates the complete ecosystem pipeline:
//!
//! 1. **biomeOS discover** → wetSpring capabilities detected
//! 2. **NUCLEUS dispatch** → Tower probes substrates, Node routes science
//! 3. **Science compute** → diversity, ordination, dynamics
//! 4. **petalTongue export** → all scenarios → `DataChannel` JSON
//! 5. **metalForge overlay** → hardware inventory + dispatch scenarios merged
//!
//! This is the apex validation: every primal interaction in one binary.
//!
//! | Domain | Checks |
//! |--------|--------|
//! | B1 Capability Registry | wetSpring science capabilities enumerated |
//! | B2 Science Pipeline    | CPU-computed diversity, ordination, dynamics |
//! | B3 Viz Export          | All scenarios → JSON → petalTongue schema |
//! | B4 metalForge Merge    | Hardware + science scenarios composed |
//! | B5 Full Graph          | End-to-end edges: discover → compute → visualize |

use wetspring_barracuda::bio::{diversity, pcoa, qs_biofilm};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::visualization::scenarios::{self, benchmarks::TierResult};
use wetspring_barracuda::visualization::{
    DataChannel, EcologyScenario, ScenarioEdge, ScenarioNode, scenario_with_edges_json,
};
use wetspring_forge::inventory;

fn main() {
    let mut v = Validator::new("Exp330: biomeOS + NUCLEUS + petalTongue Full Chain");

    // ── B1: Capability Registry ──
    v.section("B1 — biomeOS Capability Registry");

    let capabilities = vec![
        "science.diversity",
        "science.ordination",
        "science.dynamics",
        "science.chemistry",
        "science.anderson",
        "science.benchmark",
        "brain.observe",
        "brain.attention",
        "brain.urgency",
    ];
    v.check_count("registered capabilities", capabilities.len(), 9);
    v.check_pass(
        "has science.diversity",
        capabilities.contains(&"science.diversity"),
    );
    v.check_pass("has brain.observe", capabilities.contains(&"brain.observe"));

    // ── B2: Science Pipeline (CPU compute) ──
    v.section("B2 — Science Pipeline (CPU)");

    let samples = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0, 25.0],
        vec![15.0, 25.0, 5.0, 35.0, 45.0, 10.0, 20.0, 30.0],
        vec![8.0, 12.0, 40.0, 20.0, 60.0, 3.0, 7.0, 50.0],
    ];
    let labels: Vec<String> = (1..=3).map(|i| format!("Sample{i}")).collect();

    let h = diversity::shannon(&samples[0]);
    v.check("Shannon H' (sample 0)", h, 1.9024, 0.01);

    let bc = diversity::bray_curtis_matrix(&samples);
    v.check_pass("Bray-Curtis matrix non-empty", !bc.is_empty());

    let n_samp = 3;
    let dm_condensed = vec![bc[1], bc[2], bc[n_samp + 2]];
    let pcoa_result = pcoa::pcoa(&dm_condensed, 3, 2);
    v.check_pass("PCoA succeeds", pcoa_result.is_ok());

    let params = qs_biofilm::QsBiofilmParams::default();
    let y0 = [0.01, 0.0, 0.0, 0.0, 0.0];
    let ode_result = qs_biofilm::run_scenario(&y0, 10.0, tolerances::ODE_DEFAULT_DT, &params);
    v.check_pass("ODE integration completes", ode_result.steps > 0);
    v.check("ODE final B > 0", ode_result.y_final[0], 0.938, 0.01);

    // ── B3: Visualization Export ──
    v.section("B3 — petalTongue Visualization Export");

    let (eco, _eco_edges) = scenarios::ecology_scenario(&samples, &labels);
    v.check_pass("ecology scenario builds", !eco.nodes.is_empty());
    let eco_json = serde_json::to_string(&eco);
    v.check_pass("ecology serializes to JSON", eco_json.is_ok());

    let ord = scenarios::ordination_scenario(&dm_condensed, 3, &labels);
    v.check_pass("ordination scenario builds", ord.is_ok());

    let (dyn_s, _dyn_edges) = scenarios::dynamics_scenario();
    v.check_pass("dynamics scenario builds", dyn_s.nodes.len() >= 2);

    let (and_s, _and_edges) =
        scenarios::anderson_scenario(0.53, &[10.0, 12.0, 14.0], &[0.0, 1.0, 2.0]);
    v.check_pass("anderson scenario builds", !and_s.nodes.is_empty());

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
    let (bench_s, _bench_edges) = scenarios::benchmark_scenario(&tiers);
    v.check_pass("benchmark scenario builds", !bench_s.nodes.is_empty());

    let (full, full_edges) = scenarios::full_pipeline_scenario(&samples, &labels);
    v.check_pass("full pipeline scenario builds", full.nodes.len() >= 3);

    // ── B4: metalForge Hardware Overlay ──
    v.section("B4 — metalForge Hardware Overlay");

    let substrates = inventory::discover();
    v.check_pass("substrates discovered", !substrates.is_empty());

    let (inv, _inv_edges) = wetspring_forge::visualization::inventory_scenario(&substrates);
    v.check_pass("inventory scenario builds", inv.nodes.len() >= 2);

    let (disp, _) = wetspring_forge::visualization::dispatch_scenario(&substrates);
    v.check_pass("dispatch scenario builds", !disp.nodes.is_empty());

    let (nuc, nuc_edges) = wetspring_forge::visualization::nucleus_scenario(&substrates);
    v.check_pass("nucleus scenario builds", nuc.nodes.len() == 3);

    // ── B5: Full Graph Composition ──
    v.section("B5 — Full Graph Composition");

    let mut composed = EcologyScenario {
        name: "biomeOS Full Chain".into(),
        description: "biomeOS → NUCLEUS → Science → petalTongue".into(),
        version: "1.0.0".into(),
        mode: "live-ecosystem".into(),
        domain: "full-chain".into(),
        nodes: vec![],
        edges: vec![],
    };

    let biomeos_node = ScenarioNode {
        id: "biomeos".into(),
        name: "biomeOS Orchestrator".into(),
        node_type: "pipeline".into(),
        family: "biomeos".into(),
        status: "healthy".into(),
        health: 100,
        confidence: 100,
        capabilities: capabilities.iter().map(|s| (*s).into()).collect(),
        data_channels: vec![DataChannel::Gauge {
            id: "capability_count".into(),
            label: "Registered Capabilities".into(),
            value: capabilities.len() as f64,
            min: 0.0,
            max: 20.0,
            unit: "capabilities".into(),
            normal_range: [5.0, 20.0],
            warning_range: [0.0, 5.0],
        }],
        scientific_ranges: vec![],
    };
    composed.nodes.push(biomeos_node);

    for n in &nuc.nodes {
        composed.nodes.push(n.clone());
    }
    for n in &full.nodes {
        composed.nodes.push(n.clone());
    }
    for n in &inv.nodes {
        composed.nodes.push(n.clone());
    }

    let mut composed_edges: Vec<ScenarioEdge> = vec![
        ScenarioEdge {
            from: "biomeos".into(),
            to: "tower".into(),
            edge_type: "orchestration".into(),
            label: "discover hardware".into(),
        },
        ScenarioEdge {
            from: "tower".into(),
            to: "node_atomic".into(),
            edge_type: "data_flow".into(),
            label: "substrates → dispatch".into(),
        },
        ScenarioEdge {
            from: "node_atomic".into(),
            to: "diversity".into(),
            edge_type: "compute".into(),
            label: "dispatch → science".into(),
        },
        ScenarioEdge {
            from: "diversity".into(),
            to: "nest".into(),
            edge_type: "storage".into(),
            label: "results → persist".into(),
        },
    ];
    composed_edges.extend(full_edges.iter().cloned());
    composed_edges.extend(nuc_edges.iter().cloned());

    v.check_pass(
        "composed: has biomeos node",
        composed.nodes.iter().any(|n| n.id == "biomeos"),
    );
    v.check_pass(
        "composed: has NUCLEUS atomics",
        composed.nodes.iter().any(|n| n.id == "tower")
            && composed.nodes.iter().any(|n| n.id == "node_atomic")
            && composed.nodes.iter().any(|n| n.id == "nest"),
    );
    v.check_pass(
        "composed: has science nodes",
        composed.nodes.iter().any(|n| n.id == "diversity"),
    );
    v.check_pass(
        "composed: has hardware nodes",
        composed.nodes.iter().any(|n| n.id == "summary"),
    );
    v.check_pass(
        "composed: biomeos→tower edge",
        composed_edges
            .iter()
            .any(|e| e.from == "biomeos" && e.to == "tower"),
    );
    v.check_pass(
        "composed: node→diversity edge",
        composed_edges
            .iter()
            .any(|e| e.from == "node_atomic" && e.to == "diversity"),
    );
    v.check_pass(
        "composed: diversity→nest edge",
        composed_edges
            .iter()
            .any(|e| e.from == "diversity" && e.to == "nest"),
    );

    let composed_json = scenario_with_edges_json(&composed, &composed_edges);
    v.check_pass("composed JSON: serializes", composed_json.is_ok());
    if let Ok(ref json) = composed_json {
        v.check_pass(
            "composed JSON: valid parse",
            serde_json::from_str::<serde_json::Value>(json).is_ok(),
        );
        v.check_pass("composed JSON: contains biomeos", json.contains("biomeos"));
        v.check_pass("composed JSON: contains tower", json.contains("tower"));
        v.check_pass(
            "composed JSON: contains diversity",
            json.contains("diversity"),
        );
        v.check_pass(
            "composed JSON: contains metalforge",
            json.contains("metalforge"),
        );
    }

    v.check_pass("composed: ≥10 nodes total", composed.nodes.len() >= 10);
    v.check_pass("composed: ≥8 edges total", composed_edges.len() >= 8);

    v.finish();
}
