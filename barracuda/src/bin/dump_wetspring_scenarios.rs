// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! Dump wetSpring `petalTongue` scenarios to JSON or push via IPC.
//!
//! Constructs all scenario types from synthetic/known-value data,
//! validates schema structure, and outputs to `petalTongue` or files.
//!
//! # Usage
//!
//! ```bash
//! # Write scenarios to sandbox/scenarios/
//! cargo run --features json --bin dump_wetspring_scenarios
//!
//! # Push to petalTongue via IPC (if socket is available)
//! PETALTONGUE_SOCKET=/tmp/petaltongue.sock cargo run --features json --bin dump_wetspring_scenarios
//! ```
//!
//! Provenance: Scenario dump utility for petalTongue visualization

use wetspring_barracuda::tolerances;
use wetspring_barracuda::visualization::ipc_push::PetalTonguePushClient;
use wetspring_barracuda::visualization::scenarios;
use wetspring_barracuda::visualization::{EcologyScenario, ScenarioEdge, scenario_with_edges_json};

use std::fs;
use std::path::Path;
use std::process;

type ScenarioEntry<'a> = (&'a str, EcologyScenario, Vec<ScenarioEdge>);

struct CheckTracker {
    checks: u32,
    passed: u32,
}

impl CheckTracker {
    const fn new() -> Self {
        Self {
            checks: 0,
            passed: 0,
        }
    }

    fn check(&mut self, label: &str, ok: bool) {
        self.checks += 1;
        if ok {
            eprintln!("  PASS  {label}");
            self.passed += 1;
        } else {
            eprintln!("  FAIL  {label}");
        }
    }
}

fn build_core_scenarios<'a>(
    t: &mut CheckTracker,
    samples: &[Vec<f64>],
    labels: &[String],
) -> Vec<ScenarioEntry<'a>> {
    let mut all: Vec<ScenarioEntry<'a>> = Vec::new();

    let (eco, eco_edges) = scenarios::ecology_scenario(samples, labels);
    t.check("ecology has nodes", !eco.nodes.is_empty());
    t.check(
        "ecology has diversity channels",
        !eco.nodes[0].data_channels.is_empty(),
    );
    t.check("ecology has beta node", eco.nodes.len() >= 2);
    all.push(("ecology", eco, eco_edges));

    let dm = vec![0.5, 0.8, 0.6];
    let ord_labels: Vec<String> = (1..=3).map(|i| format!("S{i}")).collect();
    if let Ok((ord, ord_edges)) = scenarios::ordination_scenario(&dm, 3, &ord_labels) {
        t.check(
            "ordination has scatter + scree",
            ord.nodes[0].data_channels.len() >= 2,
        );
        all.push(("ordination", ord, ord_edges));
    }

    let (dyn_s, dyn_edges) = scenarios::dynamics_scenario();
    t.check("dynamics has QS + bistable", dyn_s.nodes.len() >= 2);
    t.check(
        "QS ODE has 5 timeseries",
        dyn_s.nodes[0].data_channels.len() == 5,
    );
    all.push(("dynamics", dyn_s, dyn_edges));

    let t_vals: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let w_vals: Vec<f64> = t_vals.iter().map(|tv| 5.0 + tv * 0.5).collect();
    let (and, and_edges) = scenarios::anderson_scenario(0.53, &w_vals, &t_vals);
    t.check(
        "anderson has gauge + timeseries",
        and.nodes[0].data_channels.len() == 2,
    );
    all.push(("anderson", and, and_edges));

    // Benchmark timings from Exp069 (three-tier benchmark, 2026-02-21).
    // Galaxy: Docker QIIME2 on Eastgate hardware (i9-12900K, 64 GB).
    // CPU: BarraCuda v0.1.0, Rust 1.87, --release.
    // GPU: ToadStool S41, RTX 4070 12 GB, wgpu 28.
    let tiers = vec![
        scenarios::benchmarks::TierResult {
            stage: "DADA2".into(),
            galaxy_s: 6.8,
            cpu_s: 0.33,
            gpu_s: 0.013,
        },
        scenarios::benchmarks::TierResult {
            stage: "Taxonomy".into(),
            galaxy_s: 3.0,
            cpu_s: 0.115,
            gpu_s: 0.011,
        },
        scenarios::benchmarks::TierResult {
            stage: "Total Pipeline".into(),
            galaxy_s: 31.9,
            cpu_s: 1.2,
            gpu_s: 0.05,
        },
    ];
    let (bench, bench_edges) = scenarios::benchmark_scenario(&tiers);
    t.check(
        "benchmark has 4 bar channels",
        bench.nodes[0].data_channels.len() == 4,
    );
    all.push(("benchmarks", bench, bench_edges));

    all
}

fn build_analysis_scenarios<'a>(
    t: &mut CheckTracker,
    samples: &[Vec<f64>],
    labels: &[String],
) -> Vec<ScenarioEntry<'a>> {
    let mut all: Vec<ScenarioEntry<'a>> = Vec::new();

    let clusters = vec![
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "geneA".into(),
            presence: vec![true, true, true],
        },
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "geneB".into(),
            presence: vec![true, false, true],
        },
        wetspring_barracuda::bio::pangenome::GeneCluster {
            id: "geneC".into(),
            presence: vec![false, false, true],
        },
    ];
    let genome_labels = vec!["G1".into(), "G2".into(), "G3".into()];
    let (pan, pan_edges) = scenarios::pangenome_scenario(&clusters, 3, &genome_labels);
    t.check(
        "pangenome has 3 channels",
        pan.nodes[0].data_channels.len() == 3,
    );
    all.push(("pangenome", pan, pan_edges));

    let model = wetspring_barracuda::bio::hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.5_f64.ln(), 0.5_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.9_f64.ln(), 0.1_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()],
    };
    let obs = vec![0, 1, 0, 1, 0];
    let state_labels = vec!["State0".into(), "State1".into()];
    let (hmm, hmm_edges) = scenarios::hmm_scenario(&model, &obs, &state_labels);
    t.check(
        "hmm has ≥ 4 channels",
        hmm.nodes[0].data_channels.len() >= 4,
    );
    all.push(("hmm", hmm, hmm_edges));

    let (stoch, stoch_edges) = scenarios::stochastic::birth_death_scenario(1.0, 0.5, 10.0, 5, 42);
    t.check(
        "stochastic has ≥ 3 channels",
        stoch.nodes[0].data_channels.len() >= 3,
    );
    all.push(("stochastic", stoch, stoch_edges));

    let s1 = b"ATCGATCGATCG";
    let s2 = b"ATCGATCGATCC";
    let s3 = b"ATCGATCGATCA";
    let seqs: Vec<&[u8]> = vec![s1, s2, s3];
    let sim_labels = vec!["G1".into(), "G2".into(), "G3".into()];
    let (sim, sim_edges) = scenarios::similarity_scenario(&seqs, &sim_labels);
    t.check(
        "similarity has 2 channels",
        sim.nodes[0].data_channels.len() == 2,
    );
    all.push(("similarity", sim, sim_edges));

    let rare_samples = vec![
        vec![10.0, 20.0, 5.0, 15.0, 30.0],
        vec![5.0, 10.0, 25.0, 8.0, 12.0],
    ];
    let rare_labels = vec!["SampleA".into(), "SampleB".into()];
    let (rare, rare_edges) = scenarios::rarefaction_scenario(&rare_samples, &rare_labels);
    t.check(
        "rarefaction has 4 channels",
        rare.nodes[0].data_channels.len() == 4,
    );
    all.push(("rarefaction", rare, rare_edges));

    let nmf_data = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.3, 0.5, 0.3, 1.0, 0.8, 0.2, 0.1];
    let nmf_config = wetspring_barracuda::bio::nmf::NmfConfig {
        rank: 2,
        max_iter: 50,
        tol: tolerances::NMF_CONVERGENCE_LOOSE,
        objective: wetspring_barracuda::bio::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let nmf_samples = vec!["S1".into(), "S2".into(), "S3".into(), "S4".into()];
    let nmf_features = vec!["F1".into(), "F2".into(), "F3".into()];
    if let Ok((nmf, nmf_edges)) = scenarios::nmf::nmf_scenario_from_data(
        &nmf_data,
        4,
        3,
        &nmf_config,
        &nmf_samples,
        &nmf_features,
    ) {
        t.check(
            "nmf has ≥ 4 channels",
            nmf.nodes[0].data_channels.len() >= 4,
        );
        all.push(("nmf", nmf, nmf_edges));
    }

    let (pipe, pipe_edges) = scenarios::streaming_pipeline::demo_streaming_pipeline_scenario();
    t.check("pipeline has ≥ 6 nodes", pipe.nodes.len() >= 6);
    all.push(("streaming_pipeline", pipe, pipe_edges));

    let (full, full_edges) = scenarios::full_pipeline_scenario(samples, labels);
    t.check(
        "full pipeline merges ecology + dynamics",
        full.nodes.len() >= 3,
    );
    all.push(("full_pipeline", full, full_edges));

    all
}

fn output_scenarios(all_scenarios: &[ScenarioEntry<'_>], out_dir: &Path) {
    let ipc_client = PetalTonguePushClient::discover();

    match &ipc_client {
        Ok(_) => eprintln!("petalTongue discovered — pushing scenarios via IPC"),
        Err(e) => eprintln!("{e} — writing JSON files to {}", out_dir.display()),
    }

    for (name, scenario, edges) in all_scenarios {
        if let Ok(ref client) = ipc_client {
            if let Err(e) = client.push_render(name, name, scenario) {
                eprintln!("  IPC push {name}: {e}");
            } else {
                eprintln!("  pushed {name}");
            }
        } else {
            match scenario_with_edges_json(scenario, edges) {
                Ok(json) => {
                    if fs::create_dir_all(out_dir).is_ok() {
                        let path = out_dir.join(format!("{name}.json"));
                        match fs::write(&path, &json) {
                            Ok(()) => eprintln!("  wrote {}", path.display()),
                            Err(e) => eprintln!("  write {name}: {e}"),
                        }
                    }
                }
                Err(e) => eprintln!("  serialize {name}: {e}"),
            }
        }
    }
}

fn run_stream_demo(all_scenarios: &[ScenarioEntry<'_>]) {
    eprintln!("\n  --stream mode: demonstrating StreamSession lifecycle");
    let stream_client = PetalTonguePushClient::discover();
    match stream_client {
        Ok(client) => {
            let mut session = wetspring_barracuda::visualization::stream::StreamSession::open(
                client,
                "dump-stream-demo",
            );
            eprintln!("  StreamSession opened: {}", session.session_id());

            if let Some((_, scenario, _)) = all_scenarios.first() {
                if let Err(e) = session.push_initial_render("Stream Demo", scenario) {
                    eprintln!("  initial render push failed: {e}");
                } else {
                    eprintln!("  initial render pushed (frame {})", session.frame_count());
                }
            }

            session.close();
            eprintln!(
                "  StreamSession closed (total frames: {})",
                session.frame_count()
            );
        }
        Err(e) => {
            eprintln!("  --stream: {e} — session lifecycle demo only (no IPC)");
            let client = PetalTonguePushClient::new(
                std::env::temp_dir().join("nonexistent-stream-demo.sock"),
            );
            let mut session = wetspring_barracuda::visualization::stream::StreamSession::open(
                client,
                "dump-stream-demo",
            );
            eprintln!(
                "  session: id={} open={} frames={}",
                session.session_id(),
                session.is_open(),
                session.frame_count()
            );
            session.close();
            eprintln!(
                "  session closed: open={} frames={}",
                session.is_open(),
                session.frame_count()
            );
        }
    }
}

fn main() {
    let stream_mode = std::env::args().any(|a| a == "--stream");
    let out_dir = Path::new("sandbox/scenarios");

    let mut tracker = CheckTracker::new();
    let samples = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0, 25.0],
        vec![15.0, 25.0, 5.0, 35.0, 45.0, 10.0, 20.0, 30.0],
        vec![8.0, 12.0, 40.0, 20.0, 60.0, 3.0, 7.0, 50.0],
    ];
    let labels: Vec<String> = (1..=3).map(|i| format!("Sample{i}")).collect();
    let mut all_scenarios = build_core_scenarios(&mut tracker, &samples, &labels);
    all_scenarios.extend(build_analysis_scenarios(&mut tracker, &samples, &labels));

    output_scenarios(&all_scenarios, out_dir);

    if stream_mode {
        run_stream_demo(&all_scenarios);
    }

    eprintln!("\n  {}/{} checks passed", tracker.passed, tracker.checks);
    if tracker.passed < tracker.checks {
        eprintln!("FAIL");
        process::exit(1);
    }
    eprintln!("OK");
}
