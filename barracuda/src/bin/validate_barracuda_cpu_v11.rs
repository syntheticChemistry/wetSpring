// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names
)]
//! Exp206: `BarraCuda` CPU Parity v11 — IPC Dispatch Layer Math Fidelity
//!
//! Proves that calling barracuda science functions through the IPC dispatch
//! layer produces bit-identical results to calling them directly. The dispatch
//! layer is purely structural — no math duplication, no numeric drift.
//!
//! Also validates the three-tier `NestGate` routing logic structurally and
//! the NUCLEUS atomic coordination model (Tower→Node→Nest).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline      | Direct barracuda function calls (Exp070, Exp163, Exp190) |
//! | Date          | 2026-02-27 |
//! | Command       | `cargo run --features ipc --release --bin validate_barracuda_cpu_v11` |
//! | Data          | Synthetic test vectors (self-contained) |
//! | Tolerances    | `tolerances::EXACT_F64` for dispatch↔direct parity |

use serde_json::json;
use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::qs_biofilm::{self, QsBiofilmParams};
use wetspring_barracuda::ipc::dispatch;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

// ── D01: Dispatch↔Direct Diversity Parity ────────────────────────────────────

fn validate_diversity_dispatch_parity(v: &mut Validator) {
    v.section("═══ D01: Dispatch↔Direct Diversity Parity ═══");
    let t = Instant::now();

    let test_communities: &[&[f64]] = &[
        &[100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
        &[55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0, 55.0],
        &[500.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[1.0],
        &[
            200.0, 180.0, 160.0, 140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0,
        ],
    ];

    for (i, counts) in test_communities.iter().enumerate() {
        let direct_h = diversity::shannon(counts);
        let direct_d = diversity::simpson(counts);
        let direct_c = diversity::chao1(counts);
        let direct_s = diversity::observed_features(counts);
        let direct_j = diversity::pielou_evenness(counts);

        let params = json!({"counts": counts, "metrics": ["all"]});
        let result = dispatch::dispatch("science.diversity", &params).expect("dispatch diversity");

        let disp_h = result["shannon"].as_f64().expect("shannon");
        let disp_d = result["simpson"].as_f64().expect("simpson");
        let disp_c = result["chao1"].as_f64().expect("chao1");
        let disp_s = result["observed"].as_f64().expect("observed");
        let disp_j = result["pielou"].as_f64().expect("pielou");

        v.check(
            &format!("comm{i} Shannon dispatch==direct"),
            disp_h,
            direct_h,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("comm{i} Simpson dispatch==direct"),
            disp_d,
            direct_d,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("comm{i} Chao1 dispatch==direct"),
            disp_c,
            direct_c,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("comm{i} S_obs dispatch==direct"),
            disp_s,
            direct_s,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("comm{i} Pielou dispatch==direct"),
            disp_j,
            direct_j,
            tolerances::EXACT_F64,
        );
    }

    println!("  Diversity dispatch parity: {}µs", t.elapsed().as_micros());
}

// ── D02: Dispatch↔Direct Bray-Curtis Parity ─────────────────────────────────

fn validate_bray_curtis_dispatch_parity(v: &mut Validator) {
    v.section("═══ D02: Dispatch↔Direct Bray-Curtis Parity ═══");
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
        let direct = diversity::bray_curtis(a, b);

        let params = json!({"counts": a, "counts_b": b});
        let result = dispatch::dispatch("science.diversity", &params).expect("dispatch bc");
        let dispatched = result["bray_curtis"].as_f64().expect("bray_curtis");

        v.check(
            &format!("pair{i} Bray-Curtis dispatch==direct"),
            dispatched,
            direct,
            tolerances::EXACT_F64,
        );
    }

    let identical_a = &[50.0, 50.0, 50.0][..];
    let identical_b = &[50.0, 50.0, 50.0][..];
    let bc_identical = diversity::bray_curtis(identical_a, identical_b);
    v.check(
        "identical communities BC=0",
        bc_identical,
        0.0,
        tolerances::EXACT_F64,
    );

    println!(
        "  Bray-Curtis dispatch parity: {}µs",
        t.elapsed().as_micros()
    );
}

// ── D03: Dispatch↔Direct QS Model Parity ────────────────────────────────────

fn validate_qs_dispatch_parity(v: &mut Validator) {
    v.section("═══ D03: Dispatch↔Direct QS Model Parity ═══");
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

        let direct_t_end = *direct.t.last().unwrap_or(&0.0);
        let direct_peak = direct
            .states()
            .filter_map(|s| s.get(4).copied())
            .fold(0.0_f64, f64::max);

        let params = json!({"scenario": scenario, "dt": dt});
        let result = dispatch::dispatch("science.qs_model", &params).expect("dispatch qs");
        let disp_t_end = result["t_end"].as_f64().expect("t_end");
        let disp_peak = result["peak_biofilm"].as_f64().expect("peak_biofilm");
        let disp_steps = result["steps"].as_u64().expect("steps");

        v.check(
            &format!("{scenario} t_end dispatch==direct"),
            disp_t_end,
            direct_t_end,
            tolerances::EXACT_F64,
        );
        v.check(
            &format!("{scenario} peak_biofilm dispatch==direct"),
            disp_peak,
            direct_peak,
            tolerances::EXACT_F64,
        );
        v.check_pass(
            &format!("{scenario} steps match"),
            disp_steps == direct.steps as u64,
        );

        println!(
            "  {scenario}: t_end={direct_t_end:.2}, peak={direct_peak:.4}, steps={disp_steps}"
        );
    }

    println!("  QS model dispatch parity: {}µs", t.elapsed().as_micros());
}

// ── D04: Full Pipeline Dispatch Chaining ─────────────────────────────────────

fn validate_full_pipeline_dispatch(v: &mut Validator) {
    v.section("═══ D04: Full Pipeline Dispatch Chaining ═══");
    let t = Instant::now();

    let counts: &[f64] = &[100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 3.0];

    let params = json!({
        "counts": counts,
        "scenario": "standard_growth",
        "dt": 0.01,
    });
    let result = dispatch::dispatch("science.full_pipeline", &params).expect("dispatch pipeline");

    v.check_pass(
        "pipeline contains diversity stage",
        result.get("diversity").is_some(),
    );
    v.check_pass(
        "pipeline contains qs_model stage",
        result.get("qs_model").is_some(),
    );
    v.check_pass("pipeline marked complete", result["pipeline"] == "complete");

    let div = &result["diversity"];
    let direct_h = diversity::shannon(counts);
    let pipeline_h = div["shannon"].as_f64().expect("pipeline shannon");
    v.check(
        "pipeline diversity Shannon==direct",
        pipeline_h,
        direct_h,
        tolerances::EXACT_F64,
    );

    let qs = &result["qs_model"];
    let pipeline_t_end = qs["t_end"].as_f64().expect("pipeline t_end");
    v.check_pass("pipeline QS t_end > 0", pipeline_t_end > 0.0);

    println!(
        "  Full pipeline dispatch chaining: {}µs",
        t.elapsed().as_micros()
    );
}

// ── D05: Three-Tier NestGate Routing (Structural) ────────────────────────────

fn validate_nestgate_three_tier(v: &mut Validator) {
    v.section("═══ D05: Three-Tier NestGate Routing (Structural) ═══");
    let t = Instant::now();

    use wetspring_barracuda::ncbi::nestgate;

    let biomeos_socket = nestgate::discover_biomeos_socket();
    let nestgate_enabled = nestgate::is_enabled();
    let nestgate_socket = nestgate::discover_socket();

    println!("  biomeOS socket: {biomeos_socket:?}");
    println!("  NestGate enabled: {nestgate_enabled}");
    println!("  NestGate socket: {nestgate_socket:?}");

    v.check_pass("discover_biomeos_socket returns Option (no panic)", true);
    v.check_pass("is_enabled returns bool (no panic)", true);
    v.check_pass("discover_socket returns Option (no panic)", true);

    let err = dispatch::dispatch("science.ncbi_fetch", &json!({}));
    v.check_pass(
        "ncbi_fetch without id returns -32602",
        err.is_err_and(|e| e.code == -32602),
    );

    println!(
        "  Three-tier routing structural: {}µs",
        t.elapsed().as_micros()
    );
}

// ── D06: NUCLEUS Atomic Coordination Model ───────────────────────────────────

fn validate_nucleus_atomics(v: &mut Validator) {
    v.section("═══ D06: NUCLEUS Atomic Coordination (Tower→Node→Nest) ═══");
    let t = Instant::now();

    let health = dispatch::dispatch("health.check", &json!({})).expect("health");
    let caps = health["capabilities"].as_array().expect("capabilities");
    let cap_names: Vec<&str> = caps.iter().filter_map(|c| c.as_str()).collect();

    v.check_pass(
        "Tower: health.check reports healthy",
        health["status"] == "healthy",
    );
    v.check_pass(
        "Tower: primal identity = wetspring",
        health["primal"] == "wetspring",
    );

    v.check_pass(
        "Node: science.diversity registered",
        cap_names.contains(&"science.diversity"),
    );
    v.check_pass(
        "Node: science.qs_model registered",
        cap_names.contains(&"science.qs_model"),
    );
    v.check_pass(
        "Node: science.full_pipeline registered",
        cap_names.contains(&"science.full_pipeline"),
    );
    v.check_pass(
        "Node: science.anderson registered",
        cap_names.contains(&"science.anderson"),
    );
    v.check_pass(
        "Node: science.ncbi_fetch registered",
        cap_names.contains(&"science.ncbi_fetch"),
    );
    v.check_pass(
        "Nest: metrics.snapshot registered",
        cap_names.contains(&"metrics.snapshot"),
    );

    let anderson_err = dispatch::dispatch("science.anderson", &json!({}));
    v.check_pass(
        "Node: anderson reports gpu-required (CPU-only build)",
        anderson_err.is_err_and(|e| e.code == -32001),
    );

    let unknown = dispatch::dispatch("nonexistent.method", &json!({}));
    v.check_pass(
        "Tower: unknown method returns -32601",
        unknown.is_err_and(|e| e.code == -32601),
    );

    println!("  NUCLEUS atomics: {}µs", t.elapsed().as_micros());
}

// ── D07: Dispatch Error Handling ─────────────────────────────────────────────

fn validate_dispatch_errors(v: &mut Validator) {
    v.section("═══ D07: Dispatch Error Handling ═══");
    let t = Instant::now();

    let empty_counts = dispatch::dispatch("science.diversity", &json!({"counts": []}));
    v.check_pass(
        "empty counts → -32602",
        empty_counts.is_err_and(|e| e.code == -32602),
    );

    let missing_counts = dispatch::dispatch("science.diversity", &json!({}));
    v.check_pass(
        "missing counts → -32602",
        missing_counts.is_err_and(|e| e.code == -32602),
    );

    let bad_scenario = dispatch::dispatch("science.qs_model", &json!({"scenario": "bogus"}));
    v.check_pass(
        "unknown QS scenario → -32602",
        bad_scenario.is_err_and(|e| e.code == -32602),
    );

    let missing_id = dispatch::dispatch("science.ncbi_fetch", &json!({}));
    v.check_pass(
        "ncbi_fetch missing id → -32602",
        missing_id.is_err_and(|e| e.code == -32602),
    );

    println!("  Error handling: {}µs", t.elapsed().as_micros());
}

fn main() {
    let mut v = Validator::new("Exp206: BarraCuda CPU v11 — IPC Dispatch Math Fidelity");

    let t_total = Instant::now();

    validate_diversity_dispatch_parity(&mut v);
    validate_bray_curtis_dispatch_parity(&mut v);
    validate_qs_dispatch_parity(&mut v);
    validate_full_pipeline_dispatch(&mut v);
    validate_nestgate_three_tier(&mut v);
    validate_nucleus_atomics(&mut v);
    validate_dispatch_errors(&mut v);

    println!("\n  Total wall-clock: {} ms", t_total.elapsed().as_millis());

    v.finish();
}
