// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(clippy::too_many_lines)]
//! # Exp334: Science-to-Viz Pipeline
//!
//! End-to-end validation of the science → visualization pipeline:
//!
//! | Domain | Checks |
//! |--------|--------|
//! | P1 Ecology     | diversity compute → ecology_scenario → JSON |
//! | P2 Full Pipeline | full_pipeline_scenario → combined graph |
//! | P3 IPC Wire    | handle_diversity with visualization=true |
//! | P4 Pangenome   | analyze → pangenome_scenario → JSON |
//! | P5 HMM         | forward + viterbi → hmm_scenario → JSON |
//! | P6 Stochastic  | birth_death_ensemble → stochastic_scenario → JSON |
//! | P7 NMF         | nmf → nmf_scenario → JSON |
//! | P8 Pipeline Viz | demo_streaming_pipeline → JSON roundtrip |
//! | P9 Regression  | Existing scenarios still produce valid output |

use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::visualization;

fn main() {
    let mut v = Validator::new("Exp334: Science-to-Viz Pipeline");

    // ── P1: Ecology end-to-end ──
    v.section("P1 — Ecology Science→Viz");

    let counts = vec![10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 8.0, 12.0];
    let labels = vec!["SampleA".into()];
    let (eco_s, eco_edges) =
        visualization::scenarios::ecology_scenario(std::slice::from_ref(&counts), &labels);
    v.check_count("ecology nodes", eco_s.nodes.len(), 1);
    v.check_pass(
        "ecology has diversity channels",
        !eco_s.nodes[0].data_channels.is_empty(),
    );
    v.check_pass("ecology edges empty (single sample)", eco_edges.is_empty());
    let eco_json = visualization::scenario_to_json(&eco_s).or_exit("eco json");
    v.check_pass("ecology JSON valid", !eco_json.is_empty());
    v.check_pass(
        "ecology JSON has rarefaction",
        eco_json.contains("rarefaction"),
    );

    // ── P2: Full Pipeline ──
    v.section("P2 — Full Pipeline Scenario");

    let (full_s, full_edges) =
        visualization::scenarios::full_pipeline_scenario(std::slice::from_ref(&counts), &labels);
    v.check_pass("full pipeline ≥ 2 nodes", full_s.nodes.len() >= 2);
    v.check_pass("full pipeline has edges", !full_edges.is_empty());
    let full_json = visualization::scenario_with_edges_json(&full_s, &full_edges).or_exit("json");
    v.check_pass("full pipeline JSON valid", !full_json.is_empty());

    // ── P3: IPC Wire ──
    v.section("P3 — IPC Diversity with Visualization Flag");

    // Simulate the IPC params with visualization=true
    let ipc_params = serde_json::json!({
        "counts": counts,
        "visualization": true,
    });
    let diversity_result =
        wetspring_barracuda::ipc::handlers::handle_diversity(&ipc_params).or_exit("diversity");
    v.check_pass(
        "diversity result has visualization field",
        diversity_result.get("visualization").is_some(),
    );
    let viz_json_str = diversity_result["visualization"].as_str().unwrap_or("");
    v.check_pass("visualization JSON non-empty", !viz_json_str.is_empty());
    v.check_pass(
        "visualization JSON has ecology",
        viz_json_str.contains("ecology"),
    );

    // ── P4: Pangenome end-to-end ──
    v.section("P4 — Pangenome Science→Viz");

    let clusters = vec![
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "g1".into(),
            presence: vec![true, true, true, true],
        },
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "g2".into(),
            presence: vec![true, true, false, false],
        },
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "g3".into(),
            presence: vec![false, false, false, true],
        },
    ];
    let pan_result = wetspring_barracuda::bio::pangenome::analyze(&clusters, 4);
    v.check_pass("pangenome core_size > 0", pan_result.core_size > 0);
    let genome_labels = vec!["A".into(), "B".into(), "C".into(), "D".into()];
    let (pan_s, _) = visualization::scenarios::pangenome_scenario(&clusters, 4, &genome_labels);
    let pan_json = serde_json::to_string(&pan_s).or_exit("pan json");
    v.check_pass("pangenome → JSON valid", !pan_json.is_empty());
    v.check_pass(
        "pangenome JSON has heatmap",
        pan_json.contains("\"channel_type\":\"heatmap\""),
    );

    // ── P5: HMM end-to-end ──
    v.section("P5 — HMM Science→Viz");

    let model = wetspring_barracuda::bio::hmm::HmmModel {
        n_states: 3,
        log_pi: vec![
            (1.0 / 3.0_f64).ln(),
            (1.0 / 3.0_f64).ln(),
            (1.0 / 3.0_f64).ln(),
        ],
        log_trans: vec![
            0.7_f64.ln(),
            0.2_f64.ln(),
            0.1_f64.ln(),
            0.1_f64.ln(),
            0.6_f64.ln(),
            0.3_f64.ln(),
            0.2_f64.ln(),
            0.2_f64.ln(),
            0.6_f64.ln(),
        ],
        n_symbols: 2,
        log_emit: vec![
            0.9_f64.ln(),
            0.1_f64.ln(),
            0.4_f64.ln(),
            0.6_f64.ln(),
            0.5_f64.ln(),
            0.5_f64.ln(),
        ],
    };
    let obs = vec![0, 1, 0, 0, 1, 1, 0];
    let fwd = wetspring_barracuda::bio::hmm::forward(&model, &obs);
    v.check_pass("HMM forward LL finite", fwd.log_likelihood.is_finite());
    let vit = wetspring_barracuda::bio::hmm::viterbi(&model, &obs);
    v.check_count("viterbi path length", vit.path.len(), obs.len());

    let state_labels = vec!["Low".into(), "Med".into(), "High".into()];
    let (hmm_s, _) = visualization::scenarios::hmm_scenario(&model, &obs, &state_labels);
    let hmm_json = serde_json::to_string(&hmm_s).or_exit("hmm json");
    v.check_pass("hmm → JSON valid", !hmm_json.is_empty());
    v.check_pass("hmm JSON has viterbi", hmm_json.contains("viterbi_path"));
    v.check_pass("hmm JSON has posterior", hmm_json.contains("posterior"));

    // ── P6: Stochastic end-to-end ──
    v.section("P6 — Stochastic Science→Viz");

    let stats = wetspring_barracuda::bio::gillespie::birth_death_ensemble(1.0, 0.3, 20.0, 10, 42);
    v.check_pass("ensemble mean > 0", stats.mean > 0.0);
    v.check_pass("ensemble n_runs = 10", stats.n_runs == 10);

    let (stoch_s, _) =
        visualization::scenarios::stochastic::birth_death_scenario(1.0, 0.3, 20.0, 10, 42);
    let stoch_json = serde_json::to_string(&stoch_s).or_exit("stoch json");
    v.check_pass("stochastic → JSON valid", !stoch_json.is_empty());
    v.check_pass(
        "stochastic JSON has trajectory",
        stoch_json.contains("trajectory_"),
    );

    // ── P7: NMF end-to-end ──
    v.section("P7 — NMF Science→Viz");

    let nmf_data = vec![5.0, 0.1, 3.0, 0.2, 4.0, 0.3, 0.1, 6.0, 0.2, 5.0, 0.1, 7.0];
    let config = wetspring_barracuda::bio::nmf::NmfConfig {
        rank: 2,
        max_iter: 100,
        tol: 1e-4,
        objective: wetspring_barracuda::bio::nmf::NmfObjective::KlDivergence,
        seed: 123,
    };
    let nmf_result = wetspring_barracuda::bio::nmf::nmf(&nmf_data, 2, 6, &config);
    v.check_pass("NMF compute succeeds", nmf_result.is_ok());
    if let Ok(ref nmf) = nmf_result {
        let sample_labels = vec!["S1".into(), "S2".into()];
        let feature_labels = vec![
            "F1".into(),
            "F2".into(),
            "F3".into(),
            "F4".into(),
            "F5".into(),
            "F6".into(),
        ];
        let (nmf_s, _) =
            visualization::scenarios::nmf::nmf_scenario(nmf, &sample_labels, &feature_labels);
        let nmf_json = serde_json::to_string(&nmf_s).or_exit("nmf json");
        v.check_pass("nmf → JSON valid", !nmf_json.is_empty());
        v.check_pass("nmf JSON has w_matrix", nmf_json.contains("w_matrix"));
        v.check_pass("nmf JSON has h_matrix", nmf_json.contains("h_matrix"));
    }

    // ── P8: Streaming Pipeline ──
    v.section("P8 — Streaming Pipeline Roundtrip");

    let (pipe_s, pipe_edges) =
        visualization::scenarios::streaming_pipeline::demo_streaming_pipeline_scenario();
    let pipe_json =
        visualization::scenario_with_edges_json(&pipe_s, &pipe_edges).or_exit("pipe json");
    v.check_pass("pipeline JSON valid", !pipe_json.is_empty());
    let parsed: serde_json::Value = serde_json::from_str(&pipe_json).or_exit("parse");
    v.check_pass("parsed JSON is object", parsed.is_object());
    v.check_pass("parsed has nodes array", parsed["nodes"].is_array());
    v.check_pass("parsed has edges array", parsed["edges"].is_array());

    // ── P9: Regression ──
    v.section("P9 — Existing Scenario Regression");

    let (dyn_s, _) = visualization::scenarios::dynamics_scenario();
    v.check_pass("dynamics scenario still works", !dyn_s.nodes.is_empty());

    let anderson = visualization::scenarios::anderson_scenario(0.53, &[1.0, 2.0], &[0.0, 1.0]);
    v.check_pass(
        "anderson scenario still works",
        !anderson.0.nodes.is_empty(),
    );

    let bench_tiers = vec![visualization::scenarios::benchmarks::TierResult {
        stage: "DADA2".into(),
        galaxy_s: 120.0,
        cpu_s: 8.0,
        gpu_s: 0.4,
    }];
    let bench = visualization::scenarios::benchmark_scenario(&bench_tiers);
    v.check_pass("benchmark scenario still works", !bench.0.nodes.is_empty());

    v.finish();
}
