// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    dead_code
)]
//! Exp207: `BarraCuda` GPU Parity v4 — IPC Science Capabilities on GPU
//!
//! Validates that every science capability dispatched through the IPC layer
//! produces GPU↔CPU parity when `ToadStool` compute dispatch is available.
//! Exercises diversity (fused map-reduce), QS ODE (parameter sweep),
//! Anderson spectral analysis, and full pipeline chaining on GPU.
//!
//! This proves the math is truly portable: CPU call == GPU call == IPC dispatch
//! for the same input data. `ToadStool`'s unidirectional streaming eliminates
//! round-trips; the math is identical.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline      | CPU dispatch parity (Exp206), GPU V59 (Exp191) |
//! | Date          | 2026-02-27 |
//! | Command       | `cargo run --features gpu,ipc --release --bin validate_barracuda_gpu_v4` |
//! | Data          | Synthetic test vectors (self-contained) |
//! | Tolerances    | `tolerances::GPU_VS_CPU_F64` for GPU↔CPU, `EXACT_F64` for structural |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in barracuda::bio

use barracuda::spectral::{GOE_R, POISSON_R};
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

fn prewarm_gpu() {
    println!("  Pre-warming GPU device...");
    let t = Instant::now();
    let _ = dispatch::dispatch("health.check", &json!({}));
    println!("  GPU device ready: {}ms", t.elapsed().as_millis());
}

// ── G01: GPU Diversity via Dispatch ──────────────────────────────────────────

fn validate_gpu_diversity(v: &mut Validator) {
    v.section("═══ G01: GPU Diversity via Dispatch ═══");
    let t = Instant::now();

    let communities = [
        synthetic_community(15_000, 0.8, 100),
        synthetic_community(20_000, 0.6, 200),
        synthetic_community(25_000, 0.4, 300),
        synthetic_community(10_000, 0.2, 400),
        synthetic_community(50_000, 0.9, 500),
    ];

    println!(
        "  {:>12} {:>10} {:>10} {:>8} {:>8}",
        "Community", "Shannon", "Simpson", "S_obs", "Pielou"
    );

    for (i, counts) in communities.iter().enumerate() {
        let cpu_h = diversity::shannon(counts);
        let cpu_d = diversity::simpson(counts);
        let cpu_s = diversity::observed_features(counts);
        let cpu_j = diversity::pielou_evenness(counts);

        let params = json!({"counts": counts, "metrics": ["all"]});
        let result = dispatch::dispatch("science.diversity", &params).expect("dispatch");
        let disp_h = result["shannon"].as_f64().expect("shannon");
        let disp_d = result["simpson"].as_f64().expect("simpson");
        let disp_s = result["observed"].as_f64().expect("observed");
        let disp_j = result["pielou"].as_f64().expect("pielou");

        println!("  community_{i:>2} {cpu_h:>10.6} {cpu_d:>10.6} {cpu_s:>8.0} {cpu_j:>8.6}");

        v.check(
            &format!("gpu_comm{i} Shannon GPU==CPU"),
            disp_h,
            cpu_h,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            &format!("gpu_comm{i} Simpson GPU==CPU"),
            disp_d,
            cpu_d,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            &format!("gpu_comm{i} S_obs GPU==CPU"),
            disp_s,
            cpu_s,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            &format!("gpu_comm{i} Pielou GPU==CPU"),
            disp_j,
            cpu_j,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    println!("  GPU diversity dispatch: {}µs", t.elapsed().as_micros());
}

// ── G02: GPU Bray-Curtis via Dispatch ────────────────────────────────────────

fn validate_gpu_bray_curtis(v: &mut Validator) {
    v.section("═══ G02: GPU Bray-Curtis via Dispatch ═══");
    let t = Instant::now();

    let comm_a = synthetic_community(15_000, 0.7, 1234);
    let comm_b = synthetic_community(15_000, 0.3, 5678);

    let cpu_bc = diversity::bray_curtis(&comm_a, &comm_b);

    let params = json!({"counts": comm_a, "counts_b": comm_b});
    let result = dispatch::dispatch("science.diversity", &params).expect("dispatch bc");
    let gpu_bc = result["bray_curtis"].as_f64().expect("bray_curtis");

    v.check(
        "Bray-Curtis GPU==CPU (dispatch)",
        gpu_bc,
        cpu_bc,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check_pass("Bray-Curtis in [0,1]", (0.0..=1.0).contains(&cpu_bc));

    let identical = synthetic_community(15_000, 0.5, 999);
    let bc_self = diversity::bray_curtis(&identical, &identical);
    v.check(
        "identical communities BC=0",
        bc_self,
        0.0,
        tolerances::EXACT_F64,
    );

    println!(
        "  GPU Bray-Curtis: {cpu_bc:.6}, {}µs",
        t.elapsed().as_micros()
    );
}

// ── G03: GPU QS ODE Model via Dispatch ───────────────────────────────────────

fn validate_gpu_qs_model(v: &mut Validator) {
    v.section("═══ G03: GPU QS ODE Model via Dispatch ═══");
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
        let result = dispatch::dispatch("science.qs_model", &params).expect("dispatch qs");
        let gpu_t_end = result["t_end"].as_f64().expect("t_end");
        let gpu_peak = result["peak_biofilm"].as_f64().expect("peak_biofilm");

        v.check(
            &format!("{scenario} t_end GPU==CPU"),
            gpu_t_end,
            cpu_t_end,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            &format!("{scenario} peak GPU==CPU"),
            gpu_peak,
            cpu_peak,
            tolerances::GPU_VS_CPU_F64,
        );

        println!("  {scenario}: peak={cpu_peak:.4}");
    }

    println!("  GPU QS ODE dispatch: {}µs", t.elapsed().as_micros());
}

// ── G04: GPU Anderson Spectral Dispatch ──────────────────────────────────────

fn validate_gpu_anderson(v: &mut Validator) {
    v.section("═══ G04: GPU Anderson Spectral via Dispatch ═══");
    let t = Instant::now();

    let disorder_values: &[f64] = &[5.0, 16.5, 30.0];

    for &w in disorder_values {
        let params = json!({"lattice_size": 8, "disorder": w, "seed": 42});
        let result = dispatch::dispatch("science.anderson", &params).expect("anderson dispatch");

        v.check_pass(
            &format!("W={w} status=computed"),
            result["status"] == "computed",
        );
        v.check_pass(
            &format!("W={w} substrate=gpu"),
            result["substrate"] == "gpu",
        );

        let r = result["level_spacing_ratio"].as_f64().expect("r");
        let regime = result["regime"].as_str().unwrap_or("unknown");
        println!("    W={w:5.1} → r={r:.4} ({regime})");

        v.check_pass(
            &format!("W={w} r in valid range"),
            (POISSON_R - 0.1..=GOE_R + 0.1).contains(&r),
        );
    }

    println!("  GPU Anderson spectral: {}µs", t.elapsed().as_micros());
}

// ── G05: Full Pipeline GPU Streaming ─────────────────────────────────────────

fn validate_gpu_full_pipeline(v: &mut Validator) {
    v.section("═══ G05: Full Pipeline GPU Streaming ═══");
    let t = Instant::now();

    let counts = synthetic_community(20_000, 0.7, 42);

    let params = json!({
        "counts": &counts,
        "scenario": "standard_growth",
        "dt": 0.01,
    });

    let result = dispatch::dispatch("science.full_pipeline", &params).expect("full pipeline");

    v.check_pass("pipeline complete", result["pipeline"] == "complete");
    v.check_pass("pipeline has diversity", result.get("diversity").is_some());
    v.check_pass("pipeline has qs_model", result.get("qs_model").is_some());
    v.check_pass("pipeline has anderson", result.get("anderson").is_some());

    let pipeline_h = result["diversity"]["shannon"].as_f64().expect("h");
    let direct_h = diversity::shannon(&counts);
    v.check(
        "pipeline Shannon==direct (GPU path)",
        pipeline_h,
        direct_h,
        tolerances::GPU_VS_CPU_F64,
    );

    let anderson_status = result["anderson"]["status"].as_str().unwrap_or("unknown");
    v.check_pass(
        "anderson stage computed in pipeline",
        anderson_status == "computed",
    );

    println!(
        "  Full pipeline GPU streaming: {}µs",
        t.elapsed().as_micros()
    );
}

// ── G06: ToadStool Compute Dispatch Model ────────────────────────────────────

fn validate_toadstool_dispatch(v: &mut Validator) {
    v.section("═══ G06: ToadStool Compute Dispatch Model ═══");
    let t = Instant::now();

    let health = dispatch::dispatch("health.check", &json!({})).expect("health");
    let version = health["version"].as_str().unwrap_or("unknown");
    let caps = health["capabilities"].as_array().expect("capabilities");

    println!("  Primal: wetspring v{version}");
    println!("  Capabilities: {}", caps.len());

    v.check_pass("dispatch reports 6+ capabilities", caps.len() >= 6);
    v.check_pass("version is populated", !version.is_empty());

    let cap_strs: Vec<&str> = caps.iter().filter_map(|c| c.as_str()).collect();
    let expected = [
        "science.diversity",
        "science.anderson",
        "science.qs_model",
        "science.ncbi_fetch",
        "science.full_pipeline",
        "metrics.snapshot",
    ];
    for exp in &expected {
        v.check_pass(
            &format!("capability '{exp}' registered"),
            cap_strs.contains(exp),
        );
    }

    println!("  ToadStool dispatch model: {}µs", t.elapsed().as_micros());
}

fn main() {
    let mut v = Validator::new("Exp207: BarraCuda GPU v4 — IPC Science on GPU");

    prewarm_gpu();
    let t_total = Instant::now();

    validate_gpu_diversity(&mut v);
    validate_gpu_bray_curtis(&mut v);
    validate_gpu_qs_model(&mut v);
    validate_gpu_anderson(&mut v);
    validate_gpu_full_pipeline(&mut v);
    validate_toadstool_dispatch(&mut v);

    println!("\n  Total wall-clock: {} ms", t_total.elapsed().as_millis());

    v.finish();
}
