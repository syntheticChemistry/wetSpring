// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! Validate all visualization scenario builders and streaming infrastructure.
//!
//! Exercises every scenario builder introduced in the `petalTongue` V2 integration,
//! validates JSON serialization round-trips, checks node/edge counts, and confirms
//! the streaming session API. Exit code 0 if all checks pass.
//!
//! # Provenance
//!
//! | Field             | Value                                     |
//! |-------------------|-------------------------------------------|
//! | Validation class  | Visualization V2 (`petalTongue` integration)|
//! | Covers            | 28 scenario builders + streaming + types  |
//! | Date              | 2026-03-09                                |
//!
//! Provenance: Visualization scenario generation validation (V2)

use std::path::PathBuf;
use std::process;

use wetspring_barracuda::visualization::ipc_push::PetalTonguePushClient;
use wetspring_barracuda::visualization::scenarios;
use wetspring_barracuda::visualization::stream::{BackpressureConfig, StreamSession};
use wetspring_barracuda::visualization::{
    DataChannel, EcologyScenario, ScenarioEdge, UiConfig, scenario_with_edges_json,
};

struct CheckTracker {
    checks: u32,
    passed: u32,
    failed_labels: Vec<String>,
}

impl CheckTracker {
    const fn new() -> Self {
        Self {
            checks: 0,
            passed: 0,
            failed_labels: Vec::new(),
        }
    }

    fn check(&mut self, label: &str, ok: bool) {
        self.checks += 1;
        if ok {
            self.passed += 1;
        } else {
            eprintln!("  FAIL  {label}");
            self.failed_labels.push(label.into());
        }
    }

    fn check_scenario(
        &mut self,
        name: &str,
        scenario: &EcologyScenario,
        edges: &[ScenarioEdge],
        min_nodes: usize,
        min_channels: usize,
    ) {
        self.check(
            &format!("{name}: has >= {min_nodes} node(s)"),
            scenario.nodes.len() >= min_nodes,
        );
        let total_channels: usize = scenario.nodes.iter().map(|n| n.data_channels.len()).sum();
        self.check(
            &format!("{name}: has >= {min_channels} channel(s)"),
            total_channels >= min_channels,
        );
        self.check(
            &format!("{name}: domain is non-empty"),
            !scenario.domain.is_empty(),
        );
        let json_ok = scenario_with_edges_json(scenario, edges).is_ok();
        self.check(&format!("{name}: JSON serialization"), json_ok);
    }
}

fn validate_types(c: &mut CheckTracker) {
    eprintln!("\n=== Phase 1: Types ===");

    let fm = DataChannel::FieldMap {
        id: "fm1".into(),
        label: "test".into(),
        grid_x: vec![0.0, 1.0],
        grid_y: vec![0.0, 1.0],
        values: vec![1.0, 2.0, 3.0, 4.0],
        unit: "test".into(),
    };
    let fm_json = serde_json::to_string(&fm);
    c.check("FieldMap serializes", fm_json.is_ok());
    if let Ok(ref j) = fm_json {
        c.check("FieldMap has grid_x", j.contains("grid_x"));
        c.check("FieldMap has fieldmap type", j.contains("fieldmap"));
    }

    let cfg = UiConfig::default();
    let cfg_json = serde_json::to_string(&cfg);
    c.check("UiConfig serializes", cfg_json.is_ok());
    c.check("UiConfig default theme", cfg.theme == "ecology-dark");
    c.check("UiConfig panels left_sidebar", cfg.show_panels.left_sidebar);
}

fn validate_scenarios(c: &mut CheckTracker) {
    let samples = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![15.0, 25.0, 5.0, 35.0, 45.0],
    ];
    let labels: Vec<String> = vec!["S1".into(), "S2".into()];

    eprintln!("=== Existing Scenarios ===");
    let (s, e) = scenarios::ecology_scenario(&samples, &labels);
    c.check_scenario("ecology", &s, &e, 1, 1);
    let (s, e) = scenarios::dynamics_scenario();
    c.check_scenario("dynamics", &s, &e, 2, 1);

    eprintln!("\n=== Phase 2: Phylogenetics ===");
    for (name, builder) in [
        ("felsenstein", scenarios::felsenstein_scenario as fn() -> _),
        ("placement", scenarios::placement_scenario),
        ("unifrac", scenarios::unifrac_scenario),
        ("dnds", scenarios::dnds_scenario),
        ("molecular_clock", scenarios::molecular_clock_scenario),
        ("reconciliation", scenarios::reconciliation_scenario),
    ] {
        let (s, e) = builder();
        let min_ch = if name == "molecular_clock" { 1 } else { 2 };
        c.check_scenario(name, &s, &e, 1, min_ch);
    }

    eprintln!("\n=== Phase 2: ODE Systems ===");
    for (name, builder, min_ch) in [
        (
            "phage_defense",
            scenarios::phage_defense_scenario as fn() -> _,
            4,
        ),
        ("bistable", scenarios::bistable_scenario, 5),
        ("cooperation", scenarios::cooperation_scenario, 4),
        ("multi_signal", scenarios::multi_signal_scenario, 7),
        ("capacitor", scenarios::capacitor_scenario, 6),
    ] {
        let (s, e) = builder();
        c.check_scenario(name, &s, &e, 1, min_ch);
    }

    eprintln!("\n=== Phase 2: 16S Pipeline ===");
    for (name, builder, min_ch) in [
        ("quality", scenarios::quality_scenario as fn() -> _, 2),
        ("dada2", scenarios::dada2_scenario, 3),
        ("taxonomy", scenarios::taxonomy_scenario, 2),
        (
            "pipeline_overview",
            scenarios::pipeline_overview_scenario,
            2,
        ),
    ] {
        let (s, e) = builder();
        c.check_scenario(name, &s, &e, 1, min_ch);
    }

    eprintln!("\n=== Phase 2: Population Genomics ===");
    let (s, e) = scenarios::snp_scenario();
    c.check_scenario("snp", &s, &e, 1, 1);
    let (s, e) = scenarios::population_genomics_scenario();
    c.check_scenario("population_genomics", &s, &e, 1, 2);
    let (s, e) = scenarios::kmer_spectrum_scenario();
    c.check_scenario("kmer_spectrum", &s, &e, 1, 2);
    let has_spectrum = s.nodes.iter().any(|n| {
        n.data_channels
            .iter()
            .any(|ch| matches!(ch, DataChannel::Spectrum { .. }))
    });
    c.check("kmer_spectrum uses Spectrum channel", has_spectrum);

    eprintln!("\n=== Phase 2: LC-MS / PFAS ===");
    for (name, builder) in [
        (
            "spectral_match",
            scenarios::spectral_match_scenario as fn() -> _,
        ),
        ("tolerance_search", scenarios::tolerance_search_scenario),
    ] {
        let (s, e) = builder();
        c.check_scenario(name, &s, &e, 1, 2);
    }
    let (s, e) = scenarios::pfas_overview_scenario();
    c.check_scenario("pfas_overview", &s, &e, 3, 4);

    eprintln!("\n=== Phase 2: ML Models ===");
    for (name, builder, min_ch) in [
        (
            "decision_tree",
            scenarios::decision_tree_scenario as fn() -> _,
            2,
        ),
        ("random_forest", scenarios::random_forest_scenario, 2),
        ("esn", scenarios::esn_scenario, 3),
    ] {
        let (s, e) = builder();
        c.check_scenario(name, &s, &e, 1, min_ch);
    }

    eprintln!("\n=== Phase 3: Composite ===");
    let (s, e) = scenarios::full_16s_scenario(&samples, &labels);
    c.check_scenario("full_16s", &s, &e, 4, 5);
    let (s, e) = scenarios::full_pfas_scenario();
    c.check_scenario("full_pfas", &s, &e, 3, 4);
    let (s, e) = scenarios::full_qs_scenario();
    c.check_scenario("full_qs", &s, &e, 5, 20);
    let (s, e) = scenarios::full_ecology_scenario(&samples, &labels);
    c.check_scenario("full_ecology", &s, &e, 10, 20);
}

fn validate_streaming(c: &mut CheckTracker) {
    eprintln!("\n=== Phase 4: Streaming ===");

    let sock_path = std::env::temp_dir().join("wetspring-viz-v2-nonexistent.sock");
    let client = PetalTonguePushClient::new(PathBuf::from(&sock_path));
    let mut session = StreamSession::open(client, "validate-v2");
    c.check("stream session opens", session.is_open());
    c.check(
        "stream session frame_count starts at 0",
        session.frame_count() == 0,
    );
    c.check("stream session not in cooldown", !session.in_cooldown());

    session.close();
    c.check("stream session closes", !session.is_open());
    c.check(
        "closed session rejects gauge",
        session.push_gauge_update("g1", 1.0).is_err(),
    );
    c.check(
        "closed session rejects diversity",
        session.push_diversity_update("div", 1.0, 0.5, 0.8).is_err(),
    );
    c.check(
        "closed session rejects ode_step",
        session.push_ode_step("ode", 1.0, &[0.1]).is_err(),
    );
    c.check(
        "closed session rejects pipeline_progress",
        session
            .push_pipeline_progress("stage", 100.0, 0.95)
            .is_err(),
    );

    let bp = BackpressureConfig::default();
    c.check(
        "backpressure timeout is 500ms",
        bp.timeout.as_millis() == 500,
    );
    c.check(
        "backpressure cooldown is 200ms",
        bp.cooldown.as_millis() == 200,
    );
    c.check("backpressure max_slow is 3", bp.max_slow_pushes == 3);

    let bp_custom = BackpressureConfig {
        timeout: std::time::Duration::from_millis(100),
        cooldown: std::time::Duration::from_millis(50),
        max_slow_pushes: 2,
    };
    let client2 = PetalTonguePushClient::new(PathBuf::from(&sock_path));
    let session2 = StreamSession::open_with_backpressure(client2, "bp-test", bp_custom);
    c.check("backpressure session opens", session2.is_open());
    c.check(
        "backpressure config timeout",
        session2.backpressure().timeout.as_millis() == 100,
    );
}

fn main() {
    eprintln!("validate_visualization_v2: petalTongue V2 integration checks\n");
    let mut c = CheckTracker::new();

    validate_types(&mut c);
    validate_scenarios(&mut c);
    validate_streaming(&mut c);

    eprintln!("\n════════════════════════════════════════════════════════════");
    eprintln!("  {}/{} checks passed", c.passed, c.checks);

    if c.passed == c.checks {
        eprintln!("  ALL PASSED");
        eprintln!("════════════════════════════════════════════════════════════");
    } else {
        eprintln!("  FAILURES:");
        for label in &c.failed_labels {
            eprintln!("    - {label}");
        }
        eprintln!("════════════════════════════════════════════════════════════");
        process::exit(1);
    }
}
