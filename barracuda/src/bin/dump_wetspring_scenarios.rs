// SPDX-License-Identifier: AGPL-3.0-or-later
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

use wetspring_barracuda::visualization::ipc_push::PetalTonguePushClient;
use wetspring_barracuda::visualization::scenarios;
use wetspring_barracuda::visualization::{EcologyScenario, ScenarioEdge, scenario_with_edges_json};

use std::fs;
use std::path::Path;
use std::process;

fn main() {
    let out_dir = Path::new("sandbox/scenarios");

    let mut all_scenarios: Vec<(&str, EcologyScenario, Vec<ScenarioEdge>)> = Vec::new();
    let mut checks = 0_u32;
    let mut passed = 0_u32;

    // ── Ecology ──
    let samples = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0, 25.0],
        vec![15.0, 25.0, 5.0, 35.0, 45.0, 10.0, 20.0, 30.0],
        vec![8.0, 12.0, 40.0, 20.0, 60.0, 3.0, 7.0, 50.0],
    ];
    let labels: Vec<String> = (1..=3).map(|i| format!("Sample{i}")).collect();
    let (eco, eco_edges) = scenarios::ecology_scenario(&samples, &labels);
    passed += check("ecology has nodes", !eco.nodes.is_empty(), &mut checks);
    passed += check(
        "ecology has diversity channels",
        !eco.nodes[0].data_channels.is_empty(),
        &mut checks,
    );
    passed += check("ecology has beta node", eco.nodes.len() >= 2, &mut checks);
    all_scenarios.push(("ecology", eco, eco_edges));

    // ── Ordination (condensed upper-triangle: n*(n-1)/2 = 3 for 3 samples) ──
    let dm = vec![0.5, 0.8, 0.6];
    let ord_labels: Vec<String> = (1..=3).map(|i| format!("S{i}")).collect();
    match scenarios::ordination_scenario(&dm, 3, &ord_labels) {
        Ok((ord, ord_edges)) => {
            passed += check(
                "ordination has scatter + scree",
                ord.nodes[0].data_channels.len() >= 2,
                &mut checks,
            );
            all_scenarios.push(("ordination", ord, ord_edges));
        }
        Err(e) => {
            eprintln!("SKIP ordination: {e}");
        }
    }

    // ── Dynamics ──
    let (dyn_s, dyn_edges) = scenarios::dynamics_scenario();
    passed += check(
        "dynamics has QS + bistable",
        dyn_s.nodes.len() >= 2,
        &mut checks,
    );
    passed += check(
        "QS ODE has 5 timeseries",
        dyn_s.nodes[0].data_channels.len() == 5,
        &mut checks,
    );
    all_scenarios.push(("dynamics", dyn_s, dyn_edges));

    // ── Anderson ──
    let t_vals: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let w_vals: Vec<f64> = t_vals.iter().map(|t| 5.0 + t * 0.5).collect();
    let (and, and_edges) = scenarios::anderson_scenario(0.53, &w_vals, &t_vals);
    passed += check(
        "anderson has gauge + timeseries",
        and.nodes[0].data_channels.len() == 2,
        &mut checks,
    );
    all_scenarios.push(("anderson", and, and_edges));

    // ── Benchmarks ──
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
    passed += check(
        "benchmark has 4 bar channels",
        bench.nodes[0].data_channels.len() == 4,
        &mut checks,
    );
    all_scenarios.push(("benchmarks", bench, bench_edges));

    // ── Full pipeline ──
    let (full, full_edges) = scenarios::full_pipeline_scenario(&samples, &labels);
    passed += check(
        "full pipeline merges ecology + dynamics",
        full.nodes.len() >= 3,
        &mut checks,
    );
    all_scenarios.push(("full_pipeline", full, full_edges));

    // ── Attempt IPC push, fall back to file output ──
    let ipc_client = PetalTonguePushClient::discover();

    match &ipc_client {
        Ok(_) => eprintln!("petalTongue discovered — pushing scenarios via IPC"),
        Err(e) => eprintln!("{e} — writing JSON files to {}", out_dir.display()),
    }

    for (name, scenario, edges) in &all_scenarios {
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

    // ── Summary ──
    eprintln!("\n  {passed}/{checks} checks passed");
    if passed < checks {
        eprintln!("FAIL");
        process::exit(1);
    }
    eprintln!("OK");
}

fn check(label: &str, ok: bool, total: &mut u32) -> u32 {
    *total += 1;
    if ok {
        eprintln!("  PASS  {label}");
        1
    } else {
        eprintln!("  FAIL  {label}");
        0
    }
}
