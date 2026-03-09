// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines, clippy::unwrap_used, clippy::expect_used)]
//! # Exp333: Visualization Evolution
//!
//! Validates all new visualization capabilities:
//!
//! | Domain | Checks |
//! |--------|--------|
//! | V1 Spectrum     | New `DataChannel::Spectrum` variant serializes correctly |
//! | V2 StreamSession | Session lifecycle: open, state, close, reject |
//! | V3 Capabilities | Songbird announcement has all fields |
//! | V4 Pangenome    | Scenario builder: heatmap, bar, gauge channels |
//! | V5 HMM          | Scenario builder: timeseries, bar, heatmap channels |
//! | V6 Stochastic   | Scenario builder: timeseries, distribution, gauge channels |
//! | V7 Similarity   | Scenario builder: heatmap, distribution channels |
//! | V8 Rarefaction  | Scenario builder: timeseries, gauge channels |
//! | V9 NMF          | Scenario builder: heatmap W/H, bar top features |
//! | V10 Pipeline    | Streaming pipeline scenario: multi-node graph |

use std::path::PathBuf;

use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::visualization;

fn main() {
    let mut v = Validator::new("Exp333: Visualization Evolution");

    // ── V1: Spectrum DataChannel ──
    v.section("V1 — Spectrum DataChannel");

    let spectrum = visualization::DataChannel::Spectrum {
        id: "sp1".into(),
        label: "Test Spectrum".into(),
        unit: "dB".into(),
        frequencies: vec![100.0, 200.0, 300.0],
        amplitudes: vec![0.5, 0.8, 0.3],
    };
    let json = serde_json::to_string(&spectrum).expect("serialize spectrum");
    v.check_pass("Spectrum serializes", !json.is_empty());
    v.check_pass(
        "channel_type = spectrum",
        json.contains("\"channel_type\":\"spectrum\""),
    );
    v.check_pass("has frequencies", json.contains("\"frequencies\""));
    v.check_pass("has amplitudes", json.contains("\"amplitudes\""));

    // ── V2: StreamSession ──
    v.section("V2 — StreamSession Lifecycle");

    let client = visualization::ipc_push::PetalTonguePushClient::new(PathBuf::from(
        "/tmp/nonexistent-exp333.sock",
    ));
    let mut session = visualization::stream::StreamSession::open(client, "exp333-session");
    v.check_pass("session is open", session.is_open());
    v.check_count(
        "initial frame count",
        usize::try_from(session.frame_count()).unwrap_or(usize::MAX),
        0,
    );
    v.check_pass(
        "session_id correct",
        session.session_id() == "exp333-session",
    );

    let state = session.state();
    v.check_pass("state snapshot is_open", state.is_open);
    v.check_pass(
        "state snapshot session_id",
        state.session_id == "exp333-session",
    );

    session.close();
    v.check_pass("session closed", !session.is_open());

    let scenario = visualization::EcologyScenario {
        name: "t".into(),
        description: "d".into(),
        version: "1.0.0".into(),
        mode: "static".into(),
        domain: "ecology".into(),
        nodes: vec![],
        edges: vec![],
    };
    let push_result = session.push_initial_render("test", &scenario);
    v.check_pass("closed session rejects push", push_result.is_err());

    // ── V3: Capabilities ──
    v.section("V3 — Songbird Capabilities");

    let ann = visualization::capabilities::announcement();
    v.check_pass("primal = wetspring", ann.primal == "wetspring");
    v.check_pass("domain = ecology", ann.domain == "ecology");
    v.check_pass("supports_streaming = true", ann.supports_streaming);
    v.check_pass("≥10 capabilities", ann.capabilities.len() >= 10);
    v.check_pass(
        "has spectrum channel type",
        ann.channel_types.contains(&"spectrum".into()),
    );
    v.check_pass(
        "has diversity cap",
        ann.capabilities
            .contains(&"visualization.ecology.diversity".into()),
    );
    v.check_pass(
        "has pangenome cap",
        ann.capabilities
            .contains(&"visualization.ecology.pangenome".into()),
    );

    let ann_json = visualization::capabilities::announcement_json().expect("serialize");
    v.check_pass("announcement JSON non-empty", !ann_json.is_empty());

    // ── V4: Pangenome Scenario ──
    v.section("V4 — Pangenome Scenario");

    let clusters = vec![
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "gA".into(),
            presence: vec![true, true, true],
        },
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "gB".into(),
            presence: vec![true, false, true],
        },
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "gC".into(),
            presence: vec![false, false, true],
        },
    ];
    let labels = vec!["G1".into(), "G2".into(), "G3".into()];
    let (pan_s, pan_edges) = visualization::scenarios::pangenome_scenario(&clusters, 3, &labels);
    v.check_count("pangenome nodes", pan_s.nodes.len(), 1);
    v.check_count("pangenome channels", pan_s.nodes[0].data_channels.len(), 3);
    v.check_pass("pangenome edges empty", pan_edges.is_empty());
    let pan_json = serde_json::to_string(&pan_s).expect("serialize pangenome");
    v.check_pass(
        "pangenome JSON has presence_absence",
        pan_json.contains("presence_absence"),
    );

    // ── V5: HMM Scenario ──
    v.section("V5 — HMM Scenario");

    let model = wetspring_barracuda::bio::hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.5_f64.ln(), 0.5_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.9_f64.ln(), 0.1_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()],
    };
    let obs = vec![0, 1, 0, 1, 0];
    let state_labels = vec!["State0".into(), "State1".into()];
    let (hmm_s, hmm_edges) = visualization::scenarios::hmm_scenario(&model, &obs, &state_labels);
    v.check_count("hmm nodes", hmm_s.nodes.len(), 1);
    v.check_pass("hmm channels ≥ 4", hmm_s.nodes[0].data_channels.len() >= 4);
    v.check_pass("hmm edges empty", hmm_edges.is_empty());

    // ── V6: Stochastic Scenario ──
    v.section("V6 — Stochastic Scenario");

    let (stoch_s, stoch_edges) =
        visualization::scenarios::stochastic::birth_death_scenario(1.0, 0.5, 10.0, 5, 42);
    v.check_count("stochastic nodes", stoch_s.nodes.len(), 1);
    v.check_pass(
        "stochastic channels ≥ 3",
        stoch_s.nodes[0].data_channels.len() >= 3,
    );
    v.check_pass("stochastic edges empty", stoch_edges.is_empty());

    // ── V7: Similarity Scenario ──
    v.section("V7 — Similarity Scenario");

    let s1 = b"ATCGATCGATCG";
    let s2 = b"ATCGATCGATCC";
    let s3 = b"ATCGATCGATCA";
    let seqs: Vec<&[u8]> = vec![s1, s2, s3];
    let glabels = vec!["G1".into(), "G2".into(), "G3".into()];
    let (sim_s, sim_edges) = visualization::scenarios::similarity_scenario(&seqs, &glabels);
    v.check_count("similarity nodes", sim_s.nodes.len(), 1);
    v.check_count("similarity channels", sim_s.nodes[0].data_channels.len(), 2);
    v.check_pass("similarity edges empty", sim_edges.is_empty());
    v.check_pass(
        "similarity has scientific ranges",
        !sim_s.nodes[0].scientific_ranges.is_empty(),
    );

    // ── V8: Rarefaction Scenario ──
    v.section("V8 — Rarefaction Scenario");

    let counts = vec![vec![10.0, 20.0, 5.0, 15.0, 30.0]];
    let rlabels = vec!["SampleA".into()];
    let (rare_s, rare_edges) = visualization::scenarios::rarefaction_scenario(&counts, &rlabels);
    v.check_count("rarefaction nodes", rare_s.nodes.len(), 1);
    v.check_count(
        "rarefaction channels (curve + gauge)",
        rare_s.nodes[0].data_channels.len(),
        2,
    );
    v.check_pass("rarefaction edges empty", rare_edges.is_empty());

    // ── V9: NMF Scenario ──
    v.section("V9 — NMF Scenario");

    let nmf_data = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.3, 0.5, 0.3, 1.0, 0.8, 0.2, 0.1];
    let nmf_config = wetspring_barracuda::bio::nmf::NmfConfig {
        rank: 2,
        max_iter: 50,
        tol: 1e-4,
        objective: wetspring_barracuda::bio::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let nmf_samples = vec!["S1".into(), "S2".into(), "S3".into(), "S4".into()];
    let nmf_features = vec!["F1".into(), "F2".into(), "F3".into()];
    let nmf_result = visualization::scenarios::nmf::nmf_scenario_from_data(
        &nmf_data,
        4,
        3,
        &nmf_config,
        &nmf_samples,
        &nmf_features,
    );
    v.check_pass("NMF scenario succeeds", nmf_result.is_ok());
    if let Ok((nmf_s, nmf_edges)) = nmf_result {
        v.check_count("nmf nodes", nmf_s.nodes.len(), 1);
        v.check_pass(
            "nmf channels ≥ 4 (W, H, 2 top_features)",
            nmf_s.nodes[0].data_channels.len() >= 4,
        );
        v.check_pass("nmf edges empty", nmf_edges.is_empty());
    }

    // ── V10: Streaming Pipeline ──
    v.section("V10 — Streaming Pipeline Scenario");

    let (pipe_s, pipe_edges) =
        visualization::scenarios::streaming_pipeline::demo_streaming_pipeline_scenario();
    v.check_pass("pipeline ≥ 6 nodes", pipe_s.nodes.len() >= 6);
    v.check_pass("pipeline has edges", !pipe_edges.is_empty());
    let pipe_json = serde_json::to_string(&pipe_s).expect("serialize pipeline");
    v.check_pass(
        "pipeline JSON has Quality Filter",
        pipe_json.contains("Quality Filter"),
    );
    v.check_pass("pipeline JSON has DADA2", pipe_json.contains("DADA2"));

    v.finish();
}
