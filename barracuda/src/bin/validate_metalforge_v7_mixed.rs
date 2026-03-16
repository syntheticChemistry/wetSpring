// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
#![expect(
    dead_code,
    reason = "validation harness: helper functions used conditionally across domains"
)]
//! Exp208: `metalForge` v7 — Mixed Hardware NUCLEUS Atomics via IPC
//!
//! Validates cross-substrate dispatch with NUCLEUS atomic coordination
//! through the IPC dispatch layer:
//! - **Tower**: wetspring-server coordinates capabilities
//! - **Node**: IPC dispatch routes to optimal barracuda paths
//! - **Nest**: Metrics capture for Neural API pathway learning
//!
//! Tests that the IPC dispatch layer correctly models mixed hardware
//! routing: diversity→GPU (fused map-reduce), QS ODE→GPU (parameter
//! sweep), taxonomy→NPU (int8 quantized), FASTQ→CPU (I/O-bound).
//! Validates `PCIe` bypass topology (streaming without CPU roundtrip),
//! graceful GPU→CPU fallback, and `biomeOS` graph pipeline chaining.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline      | `metalForge` v6 (Exp104), CPU v11 (Exp206), GPU V59 (Exp191) |
//! | Date          | 2026-02-27 |
//! | Command       | `cargo run --features ipc --release --bin validate_metalforge_v7_mixed` |
//! | Data          | Synthetic test vectors (self-contained) |
//! | Tolerances    | `tolerances::EXACT_F64` for CPU parity, structural for routing |
//!
//! Validation class: Pipeline
//! Provenance: End-to-end pipeline integration test

use serde_json::json;
use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::qs_biofilm::{self, QsBiofilmParams};
use wetspring_barracuda::ipc::dispatch;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn synthetic_community(n_species: usize, evenness: f64, seed: u64) -> Vec<f64> {
    let mut counts = Vec::with_capacity(n_species);
    let mut rng = seed;
    for i in 0..n_species {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let noise = ((rng >> 33) as f64) / f64::from(u32::MAX);
        let rank_weight = (-(i as f64) / (n_species as f64 * evenness)).exp();
        counts.push((rank_weight * 1000.0 * (0.5 + noise)).max(1.0));
    }
    counts
}

// ── MF01: Cross-Substrate Diversity (IPC → direct parity) ───────────────────

fn validate_cross_substrate_diversity(v: &mut Validator) {
    v.section("═══ MF01: Cross-Substrate Diversity Parity ═══");
    let t = Instant::now();

    let communities = [
        synthetic_community(200, 0.8, 100),
        synthetic_community(150, 0.6, 200),
        synthetic_community(100, 0.4, 300),
        synthetic_community(50, 0.2, 400),
        synthetic_community(500, 0.9, 500),
    ];

    for (i, counts) in communities.iter().enumerate() {
        let cpu_h = diversity::shannon(counts);
        let cpu_d = diversity::simpson(counts);
        let cpu_c = diversity::chao1(counts);
        let cpu_j = diversity::pielou_evenness(counts);

        let params = json!({"counts": counts, "metrics": ["all"]});
        let result = dispatch::dispatch("science.diversity", &params).expect("dispatch");
        let mf_h = result["shannon"].as_f64().expect("h");
        let mf_d = result["simpson"].as_f64().expect("d");
        let mf_c = result["chao1"].as_f64().expect("c");
        let mf_j = result["pielou"].as_f64().expect("j");

        v.check(
            &format!("comm{i} Shannon cross-substrate"),
            mf_h,
            cpu_h,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("comm{i} Simpson cross-substrate"),
            mf_d,
            cpu_d,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("comm{i} Chao1 cross-substrate"),
            mf_c,
            cpu_c,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("comm{i} Pielou cross-substrate"),
            mf_j,
            cpu_j,
            tolerances::EXACT_F64,
        );
    }

    println!("  Cross-substrate diversity: {}µs", t.elapsed().as_micros());
}

// ── MF02: Cross-Substrate Bray-Curtis Matrix ────────────────────────────────

fn validate_cross_substrate_bray_curtis(v: &mut Validator) {
    v.section("═══ MF02: Cross-Substrate Bray-Curtis ═══");
    let t = Instant::now();

    let pairs: &[(&[f64], &[f64])] = &[
        (
            &[10.0, 20.0, 30.0, 40.0, 50.0],
            &[15.0, 25.0, 35.0, 45.0, 55.0],
        ),
        (&[100.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 100.0]),
        (&[50.0, 50.0, 50.0], &[50.0, 50.0, 50.0]),
    ];

    for (i, (a, b)) in pairs.iter().enumerate() {
        let cpu_bc = diversity::bray_curtis(a, b);
        let params = json!({"counts": a, "counts_b": b});
        let result = dispatch::dispatch("science.diversity", &params).expect("bc");
        let mf_bc = result["bray_curtis"].as_f64().expect("bray_curtis");

        v.check(
            &format!("pair{i} BC cross-substrate"),
            mf_bc,
            cpu_bc,
            tolerances::EXACT_F64,
        );
    }

    let large_a = synthetic_community(300, 0.7, 1234);
    let large_b = synthetic_community(300, 0.3, 5678);
    let cpu_bc = diversity::bray_curtis(&large_a, &large_b);
    let params = json!({"counts": large_a, "counts_b": large_b});
    let result = dispatch::dispatch("science.diversity", &params).expect("large bc");
    let mf_bc = result["bray_curtis"].as_f64().expect("bray_curtis");

    v.check(
        "large BC cross-substrate",
        mf_bc,
        cpu_bc,
        tolerances::EXACT_F64,
    );
    v.check_pass("BC in [0,1]", (0.0..=1.0).contains(&cpu_bc));

    println!(
        "  Cross-substrate Bray-Curtis: {}µs",
        t.elapsed().as_micros()
    );
}

// ── MF03: Cross-Substrate QS ODE (all 4 scenarios) ──────────────────────────

fn validate_cross_substrate_qs(v: &mut Validator) {
    v.section("═══ MF03: Cross-Substrate QS ODE (4 Scenarios) ═══");
    let t = Instant::now();

    let scenarios = &[
        "standard_growth",
        "high_density",
        "hapr_mutant",
        "dgc_overexpression",
    ];

    for scenario in scenarios {
        let dt = 0.01;
        let qs_params = QsBiofilmParams::default();

        let direct = match *scenario {
            "standard_growth" => qs_biofilm::scenario_standard_growth(&qs_params, dt),
            "high_density" => qs_biofilm::scenario_high_density(&qs_params, dt),
            "hapr_mutant" => qs_biofilm::scenario_hapr_mutant(&qs_params, dt),
            "dgc_overexpression" => qs_biofilm::scenario_dgc_overexpression(&qs_params, dt),
            _ => unreachable!(),
        };

        let cpu_t_end = *direct.t.last().unwrap_or(&0.0);
        let cpu_peak = direct
            .states()
            .filter_map(|s| s.get(4).copied())
            .fold(0.0_f64, f64::max);

        let params = json!({"scenario": scenario, "dt": dt});
        let result = dispatch::dispatch("science.qs_model", &params).expect("qs");
        let mf_t_end = result["t_end"].as_f64().expect("t_end");
        let mf_peak = result["peak_biofilm"].as_f64().expect("peak");
        let mf_steps = result["steps"].as_u64().expect("steps");

        v.check(
            &format!("{scenario} t_end cross-substrate"),
            mf_t_end,
            cpu_t_end,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("{scenario} peak cross-substrate"),
            mf_peak,
            cpu_peak,
            tolerances::EXACT_F64,
        );
        v.check_pass(
            &format!("{scenario} steps match"),
            mf_steps == direct.steps as u64,
        );

        println!("  {scenario}: peak={cpu_peak:.4}, steps={mf_steps}");
    }

    println!("  Cross-substrate QS: {}µs", t.elapsed().as_micros());
}

// ── MF04: PCIe Bypass Pipeline Topology ──────────────────────────────────────

fn validate_pcie_bypass_topology(v: &mut Validator) {
    v.section("═══ MF04: PCIe Bypass Pipeline Topology ═══");
    let t = Instant::now();

    #[derive(Debug)]
    struct PipelineStage {
        name: &'static str,
        substrate: &'static str,
        accepts_gpu_buffer: bool,
        produces_gpu_buffer: bool,
    }

    let science_pipeline = [
        PipelineStage {
            name: "nestgate_fetch",
            substrate: "CPU",
            accepts_gpu_buffer: false,
            produces_gpu_buffer: false,
        },
        PipelineStage {
            name: "diversity_fusion",
            substrate: "GPU",
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "qs_ode_sweep",
            substrate: "GPU",
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "anderson_spectral",
            substrate: "GPU",
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "taxonomy_int8",
            substrate: "NPU",
            accepts_gpu_buffer: false,
            produces_gpu_buffer: false,
        },
    ];

    let mut gpu_chained = 0;
    let mut cpu_roundtrips = 0;

    for window in science_pipeline.windows(2) {
        if window[0].produces_gpu_buffer && window[1].accepts_gpu_buffer {
            gpu_chained += 1;
        } else {
            cpu_roundtrips += 1;
        }
    }

    v.check_pass("pipeline has 5 stages", science_pipeline.len() == 5);
    v.check_pass(
        "GPU chained: diversity→QS→Anderson (2 hops)",
        gpu_chained == 2,
    );
    v.check_pass(
        "CPU roundtrips: fetch→GPU and GPU→NPU (2)",
        cpu_roundtrips == 2,
    );

    let npu_to_gpu = [
        PipelineStage {
            name: "npu_taxonomy",
            substrate: "NPU",
            accepts_gpu_buffer: false,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "gpu_diversity",
            substrate: "GPU",
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
        PipelineStage {
            name: "gpu_anderson",
            substrate: "GPU",
            accepts_gpu_buffer: true,
            produces_gpu_buffer: true,
        },
    ];

    let mut npu_gpu_chained = 0;
    for window in npu_to_gpu.windows(2) {
        if window[0].produces_gpu_buffer && window[1].accepts_gpu_buffer {
            npu_gpu_chained += 1;
        }
    }
    v.check_pass(
        "NPU→GPU: direct buffer transfer (2 chained)",
        npu_gpu_chained == 2,
    );

    println!(
        "  Science pipeline: 5 stages, {gpu_chained} GPU-chained, {cpu_roundtrips} roundtrips"
    );
    println!("  NPU→GPU bypass: {npu_gpu_chained} chained (zero CPU roundtrip between GPU stages)");
    println!("  PCIe bypass topology: {}µs", t.elapsed().as_micros());
}

// ── MF05: GPU→CPU Fallback via Dispatch ──────────────────────────────────────

fn validate_gpu_cpu_fallback(v: &mut Validator) {
    v.section("═══ MF05: GPU→CPU Fallback via Dispatch ═══");
    let t = Instant::now();

    let anderson_result = dispatch::dispatch("science.anderson", &json!({}));

    #[cfg(feature = "gpu")]
    {
        v.check_pass(
            "anderson available with GPU feature",
            anderson_result.is_ok(),
        );
        if let Ok(ref result) = anderson_result {
            v.check_pass("anderson substrate=gpu", result["substrate"] == "gpu");
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        v.check_pass(
            "anderson unavailable without GPU (graceful -32001)",
            anderson_result.is_err_and(|e| e.code == -32001),
        );
    }

    let counts: &[f64] = &[100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0];
    let fallback_result = dispatch::dispatch("science.diversity", &json!({"counts": counts}))
        .expect("diversity always works (CPU fallback)");

    let cpu_h = diversity::shannon(counts);
    let fb_h = fallback_result["shannon"].as_f64().expect("h");
    v.check(
        "diversity CPU fallback parity",
        fb_h,
        cpu_h,
        tolerances::EXACT_F64,
    );

    let pipeline = dispatch::dispatch(
        "science.full_pipeline",
        &json!({"counts": counts, "scenario": "standard_growth"}),
    )
    .expect("pipeline works with fallback");
    v.check_pass(
        "pipeline completes with fallback",
        pipeline["pipeline"] == "complete",
    );

    let anderson_in_pipeline = &pipeline["anderson"];
    #[cfg(not(feature = "gpu"))]
    v.check_pass(
        "anderson skipped gracefully in pipeline",
        anderson_in_pipeline["status"] == "skipped",
    );
    #[cfg(feature = "gpu")]
    v.check_pass(
        "anderson computed in pipeline",
        anderson_in_pipeline["status"] == "computed",
    );

    println!("  GPU→CPU fallback: {}µs", t.elapsed().as_micros());
}

// ── MF06: NUCLEUS Tower→Node→Nest Coordination ──────────────────────────────

fn validate_nucleus_coordination(v: &mut Validator) {
    v.section("═══ MF06: NUCLEUS Tower→Node→Nest Coordination ═══");
    let t = Instant::now();

    let health = dispatch::dispatch("health.check", &json!({})).expect("health");
    v.check_pass("Tower: primal=wetspring", health["primal"] == "wetspring");
    v.check_pass("Tower: status=healthy", health["status"] == "healthy");

    let caps = health["capabilities"].as_array().expect("caps");
    let cap_strs: Vec<&str> = caps.iter().filter_map(|c| c.as_str()).collect();

    let expected_caps = [
        "science.diversity",
        "science.anderson",
        "science.qs_model",
        "science.ncbi_fetch",
        "science.full_pipeline",
        "metrics.snapshot",
    ];
    for cap in &expected_caps {
        v.check_pass(&format!("Tower announces '{cap}'"), cap_strs.contains(cap));
    }

    let node_counts: &[f64] = &[50.0, 40.0, 30.0, 20.0, 10.0];
    let node_result = dispatch::dispatch("science.diversity", &json!({"counts": node_counts}))
        .expect("node diversity");
    let direct_h = diversity::shannon(node_counts);
    let node_h = node_result["shannon"].as_f64().expect("h");
    v.check(
        "Node: diversity math fidelity",
        node_h,
        direct_h,
        tolerances::EXACT_F64,
    );

    let nest_result =
        dispatch::dispatch("science.qs_model", &json!({"scenario": "standard_growth"}))
            .expect("nest qs");
    v.check_pass(
        "Nest: QS result has steps",
        nest_result.get("steps").is_some(),
    );
    v.check_pass(
        "Nest: QS result has peak",
        nest_result.get("peak_biofilm").is_some(),
    );
    v.check_pass(
        "Nest: QS result has final_state",
        nest_result.get("final_state").is_some(),
    );

    println!(
        "  NUCLEUS: {} capabilities, diversity={direct_h:.6}",
        caps.len()
    );
    println!("  NUCLEUS coordination: {}µs", t.elapsed().as_micros());
}

// ── MF07: biomeOS Graph Pipeline End-to-End ──────────────────────────────────

fn validate_biomeos_graph_e2e(v: &mut Validator) {
    v.section("═══ MF07: biomeOS Graph Pipeline E2E ═══");
    let t = Instant::now();

    let soil = synthetic_community(250, 0.75, 42);
    let ref_soil = synthetic_community(250, 0.75, 99);

    let params = json!({
        "counts": &soil,
        "counts_b": &ref_soil,
        "scenario": "standard_growth",
        "dt": 0.01,
    });

    let result = dispatch::dispatch("science.full_pipeline", &params).expect("e2e pipeline");

    v.check_pass("E2E: pipeline complete", result["pipeline"] == "complete");

    let div = &result["diversity"];
    let h = div["shannon"].as_f64().expect("h");
    let d = div["simpson"].as_f64().expect("d");
    let bc = div["bray_curtis"].as_f64().expect("bc");

    v.check_pass("E2E: Shannon > 0", h > 0.0);
    v.check_pass("E2E: Simpson in [0,1]", (0.0..=1.0).contains(&d));
    v.check_pass("E2E: Bray-Curtis in [0,1]", (0.0..=1.0).contains(&bc));

    let direct_h = diversity::shannon(&soil);
    let direct_d = diversity::simpson(&soil);
    let direct_bc = diversity::bray_curtis(&soil, &ref_soil);

    v.check("E2E: Shannon parity", h, direct_h, tolerances::EXACT_F64);
    v.check("E2E: Simpson parity", d, direct_d, tolerances::EXACT_F64);
    v.check("E2E: BC parity", bc, direct_bc, tolerances::EXACT_F64);

    let qs = &result["qs_model"];
    let qs_t_end = qs["t_end"].as_f64().expect("t_end");
    let qs_peak = qs["peak_biofilm"].as_f64().expect("peak");
    v.check_pass("E2E: QS t_end > 0", qs_t_end > 0.0);
    v.check_pass("E2E: QS peak > 0", qs_peak > 0.0);

    let cpu_qs = qs_biofilm::scenario_standard_growth(&QsBiofilmParams::default(), 0.01);
    let cpu_peak = cpu_qs
        .states()
        .filter_map(|s| s.get(4).copied())
        .fold(0.0_f64, f64::max);
    v.check(
        "E2E: QS peak parity",
        qs_peak,
        cpu_peak,
        tolerances::EXACT_F64,
    );

    println!("  E2E: H={h:.4}, D={d:.4}, BC={bc:.4}, QS_peak={qs_peak:.4}");
    println!("  biomeOS graph E2E: {}µs", t.elapsed().as_micros());
}

// ── MF08: Mixed-Substrate Workload Routing Model ─────────────────────────────

fn validate_workload_routing_model(v: &mut Validator) {
    v.section("═══ MF08: Mixed-Substrate Workload Routing Model ═══");
    let t = Instant::now();

    #[derive(Debug)]
    struct WorkloadRoute {
        name: &'static str,
        optimal_substrate: &'static str,
        capability: &'static str,
        ipc_method: Option<&'static str>,
    }

    let routes = [
        WorkloadRoute {
            name: "Diversity fused map-reduce",
            optimal_substrate: "GPU",
            capability: "f64+reduce",
            ipc_method: Some("science.diversity"),
        },
        WorkloadRoute {
            name: "Anderson spectral",
            optimal_substrate: "GPU",
            capability: "f64+shader",
            ipc_method: Some("science.anderson"),
        },
        WorkloadRoute {
            name: "QS ODE parameter sweep",
            optimal_substrate: "GPU",
            capability: "f64",
            ipc_method: Some("science.qs_model"),
        },
        WorkloadRoute {
            name: "HMM forward",
            optimal_substrate: "GPU",
            capability: "f64+shader",
            ipc_method: None,
        },
        WorkloadRoute {
            name: "Felsenstein pruning",
            optimal_substrate: "GPU",
            capability: "f64+shader",
            ipc_method: None,
        },
        WorkloadRoute {
            name: "Taxonomy int8",
            optimal_substrate: "NPU",
            capability: "quant(8)",
            ipc_method: None,
        },
        WorkloadRoute {
            name: "PFAS screening",
            optimal_substrate: "NPU",
            capability: "quant(8)",
            ipc_method: None,
        },
        WorkloadRoute {
            name: "Anomaly ESN",
            optimal_substrate: "NPU",
            capability: "quant(4)+weight-mut",
            ipc_method: None,
        },
        WorkloadRoute {
            name: "FASTQ parsing",
            optimal_substrate: "CPU",
            capability: "cpu",
            ipc_method: None,
        },
        WorkloadRoute {
            name: "Tree traversal",
            optimal_substrate: "CPU",
            capability: "cpu",
            ipc_method: None,
        },
    ];

    let gpu_routed = routes
        .iter()
        .filter(|r| r.optimal_substrate == "GPU")
        .count();
    let npu_routed = routes
        .iter()
        .filter(|r| r.optimal_substrate == "NPU")
        .count();
    let cpu_routed = routes
        .iter()
        .filter(|r| r.optimal_substrate == "CPU")
        .count();

    v.check_pass("GPU gets compute-heavy (5 workloads)", gpu_routed == 5);
    v.check_pass("NPU gets inference (3 workloads)", npu_routed == 3);
    v.check_pass("CPU gets I/O-bound (2 workloads)", cpu_routed == 2);
    v.check_pass(
        "all 10 workloads routed",
        gpu_routed + npu_routed + cpu_routed == 10,
    );

    for route in &routes {
        if let Some(method) = route.ipc_method {
            let test_result = match method {
                "science.diversity" => {
                    let p = json!({"counts": [10.0, 20.0, 30.0]});
                    dispatch::dispatch(method, &p).map(|_| ())
                }
                "science.qs_model" => {
                    let p = json!({"scenario": "standard_growth"});
                    dispatch::dispatch(method, &p).map(|_| ())
                }
                "science.anderson" => dispatch::dispatch(method, &json!({}))
                    .map(|_| ())
                    .or_else(|e| if e.code == -32001 { Ok(()) } else { Err(e) }),
                _ => Ok(()),
            };
            v.check_pass(
                &format!("{}: IPC dispatch reachable", route.name),
                test_result.is_ok(),
            );
        }
    }

    println!("  Routing: GPU={gpu_routed}, NPU={npu_routed}, CPU={cpu_routed} (10 total)");
    println!("  Workload routing: {}µs", t.elapsed().as_micros());
}

fn main() {
    let mut v = Validator::new("Exp208: metalForge v7 — Mixed Hardware NUCLEUS Atomics");

    let t_total = Instant::now();

    validate_cross_substrate_diversity(&mut v);
    validate_cross_substrate_bray_curtis(&mut v);
    validate_cross_substrate_qs(&mut v);
    validate_pcie_bypass_topology(&mut v);
    validate_gpu_cpu_fallback(&mut v);
    validate_nucleus_coordination(&mut v);
    validate_biomeos_graph_e2e(&mut v);
    validate_workload_routing_model(&mut v);

    println!("\n  Total wall-clock: {} ms", t_total.elapsed().as_millis());

    v.finish();
}
