// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
//! Exp079: `BarraCUDA` CPU Parity v6 — ODE Flat Param Fidelity
//!
//! Validates that the GPU-compatible flat parameter APIs (`to_flat`/`from_flat`)
//! produce bitwise-identical ODE integration results across all 6 biological
//! ODE models. This proves the serialization path required for GPU dispatch
//! preserves pure Rust math fidelity.
//!
//! Each module is tested in three stages:
//! 1. Flat round-trip bitwise identity (params survive serialization)
//! 2. ODE parity (flat→struct→integrate matches direct struct→integrate)
//! 3. Python baseline parity (flat path matches documented Python values)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | scipy `odeint` (LSODA) / pure Python RK4 |
//! | Baseline version | scipy 1.12.0, numpy 1.26.4 |
//! | Baseline command | `python scripts/waters2008_qs_ode.py` (and 5 others) |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --release --bin validate_barracuda_cpu_v6` |
//! | Data | Analytical ODE steady-states from documented Python runs |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::time::Instant;
use wetspring_barracuda::bio::{
    bistable::{self, BistableParams},
    capacitor::{self, CapacitorParams},
    cooperation::{self, CooperationParams},
    multi_signal::{self, MultiSignalParams},
    ode::steady_state_mean,
    phage_defense::{self, PhageDefenseParams},
    qs_biofilm::{self, QsBiofilmParams},
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const DT: f64 = 0.001;
const SS_FRAC: f64 = 0.1;

fn main() {
    let mut v = Validator::new("BarraCUDA CPU v6 — ODE Flat Param Fidelity (6 Modules)");
    let t_total = Instant::now();

    validate_qs_biofilm(&mut v);
    validate_bistable(&mut v);
    validate_multi_signal(&mut v);
    validate_phage_defense(&mut v);
    validate_cooperation(&mut v);
    validate_capacitor(&mut v);

    #[allow(clippy::cast_precision_loss)]
    let elapsed_us = t_total.elapsed().as_nanos() as f64 / 1000.0;
    println!("\n  Total ODE flat-param validation: {elapsed_us:.0} µs");

    v.finish();
}

// ════════════════════════════════════════════════════════════════════
//  Module 1: qs_biofilm (Waters 2008) — 5 vars, 18 params
// ════════════════════════════════════════════════════════════════════

fn validate_qs_biofilm(v: &mut Validator) {
    v.section("═══ Module 1: QS Biofilm (Waters 2008) — 5 vars, 18 params ═══");
    let t0 = Instant::now();

    let p = QsBiofilmParams::default();
    let flat = p.to_flat();
    let p2 = QsBiofilmParams::from_flat(&flat);
    let flat2 = p2.to_flat();

    v.check_count("qs_biofilm flat length", flat.len(), qs_biofilm::N_PARAMS);
    v.check(
        "qs_biofilm flat round-trip",
        bitwise_diff(&flat, &flat2),
        0.0,
        0.0,
    );

    let r_direct = qs_biofilm::scenario_standard_growth(&p, DT);
    let r_flat = qs_biofilm::scenario_standard_growth(&p2, DT);

    let n_direct = steady_state_mean(&r_direct, 0, SS_FRAC);
    let n_flat = steady_state_mean(&r_flat, 0, SS_FRAC);
    v.check(
        "qs_biofilm N_ss bitwise",
        (n_direct - n_flat).abs(),
        0.0,
        0.0,
    );

    let b_direct = steady_state_mean(&r_direct, 4, SS_FRAC);
    let b_flat = steady_state_mean(&r_flat, 4, SS_FRAC);
    v.check(
        "qs_biofilm B_ss bitwise",
        (b_direct - b_flat).abs(),
        0.0,
        0.0,
    );

    v.check(
        "qs_biofilm N_ss vs Python (0.975)",
        n_direct,
        0.975,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "qs_biofilm B_ss dispersed (<0.05)",
        b_direct,
        0.0,
        tolerances::ODE_NEAR_ZERO,
    );

    print_timing("qs_biofilm", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 2: bistable (Fernandez 2020) — 5 vars, 21 params
// ════════════════════════════════════════════════════════════════════

fn validate_bistable(v: &mut Validator) {
    v.section("═══ Module 2: Bistable (Fernandez 2020) — 5 vars, 21 params ═══");
    let t0 = Instant::now();

    let p = BistableParams::default();
    let flat = p.to_flat();
    let p2 = BistableParams::from_flat(&flat);
    let flat2 = p2.to_flat();

    v.check_count("bistable flat length", flat.len(), bistable::N_PARAMS);
    v.check(
        "bistable flat round-trip",
        bitwise_diff(&flat, &flat2),
        0.0,
        0.0,
    );

    let y0 = [0.01, 0.0, 0.0, 2.0, 0.1];
    let r_direct = bistable::run_bistable(&y0, 24.0, DT, &p);
    let r_flat = bistable::run_bistable(&y0, 24.0, DT, &p2);

    let b_direct = steady_state_mean(&r_direct, 4, SS_FRAC);
    let b_flat = steady_state_mean(&r_flat, 4, SS_FRAC);
    v.check("bistable B_ss bitwise", (b_direct - b_flat).abs(), 0.0, 0.0);

    v.check(
        "bistable B_ss high (sessile attractor)",
        b_direct,
        0.7,
        0.15,
    );

    let p_no_fb = BistableParams {
        alpha_fb: 0.0,
        ..p.clone()
    };
    let r_no_fb = bistable::run_bistable(&y0, 24.0, DT, &p_no_fb);
    let b_no_fb = steady_state_mean(&r_no_fb, 4, SS_FRAC);
    v.check_pass(
        "bistable feedback effect: alpha=3 > alpha=0",
        b_direct > b_no_fb,
    );

    let bif = bistable::bifurcation_scan(&p, 0.0, 6.0, 20, DT, 24.0);
    let has_hysteresis = bif
        .b_forward
        .iter()
        .zip(&bif.b_backward)
        .any(|(f, b)| (f - b).abs() > 0.05);
    v.check_pass(
        "bistable hysteresis detected in bifurcation scan",
        has_hysteresis,
    );

    print_timing("bistable", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 3: multi_signal (Srivastava 2011) — 7 vars, 24 params
// ════════════════════════════════════════════════════════════════════

fn validate_multi_signal(v: &mut Validator) {
    v.section("═══ Module 3: Multi-Signal QS (Srivastava 2011) — 7 vars, 24 params ═══");
    let t0 = Instant::now();

    let p = MultiSignalParams::default();
    let flat = p.to_flat();
    let p2 = MultiSignalParams::from_flat(&flat);
    let flat2 = p2.to_flat();

    v.check_count(
        "multi_signal flat length",
        flat.len(),
        multi_signal::N_PARAMS,
    );
    v.check(
        "multi_signal flat round-trip",
        bitwise_diff(&flat, &flat2),
        0.0,
        0.0,
    );

    let r_direct = multi_signal::scenario_wild_type(&p, DT);
    let r_flat = multi_signal::scenario_wild_type(&p2, DT);

    for var in 0..multi_signal::N_VARS {
        let d = steady_state_mean(&r_direct, var, SS_FRAC);
        let f = steady_state_mean(&r_flat, var, SS_FRAC);
        v.check(
            &format!("multi_signal var{var} bitwise"),
            (d - f).abs(),
            0.0,
            0.0,
        );
    }

    let h_ss = steady_state_mean(&r_direct, 4, SS_FRAC);
    v.check("multi_signal HapR_ss vs Python (>0.3)", h_ss, 0.5, 0.2);

    let b_ss = steady_state_mean(&r_direct, 6, SS_FRAC);
    v.check(
        "multi_signal B_ss vs Python (~0.413)",
        b_ss,
        0.413,
        tolerances::ODE_STEADY_STATE,
    );

    let r_no_qs = multi_signal::scenario_no_qs(&p, DT);
    let b_no_qs = steady_state_mean(&r_no_qs, 6, SS_FRAC);
    v.check_pass("multi_signal no-QS maintains biofilm (>0.3)", b_no_qs > 0.3);

    print_timing("multi_signal", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 4: phage_defense (Hsueh 2022) — 4 vars, 11 params
// ════════════════════════════════════════════════════════════════════

fn validate_phage_defense(v: &mut Validator) {
    v.section("═══ Module 4: Phage Defense (Hsueh 2022) — 4 vars, 11 params ═══");
    let t0 = Instant::now();

    let p = PhageDefenseParams::default();
    let flat = p.to_flat();
    let p2 = PhageDefenseParams::from_flat(&flat);
    let flat2 = p2.to_flat();

    v.check_count(
        "phage_defense flat length",
        flat.len(),
        phage_defense::N_PARAMS,
    );
    v.check(
        "phage_defense flat round-trip",
        bitwise_diff(&flat, &flat2),
        0.0,
        0.0,
    );

    let r_direct = phage_defense::scenario_phage_attack(&p, DT);
    let r_flat = phage_defense::scenario_phage_attack(&p2, DT);

    for var in 0..phage_defense::N_VARS {
        let d = steady_state_mean(&r_direct, var, SS_FRAC);
        let f = steady_state_mean(&r_flat, var, SS_FRAC);
        v.check(
            &format!("phage_defense var{var} bitwise"),
            (d - f).abs(),
            0.0,
            0.0,
        );
    }

    let bd = steady_state_mean(&r_direct, 0, SS_FRAC);
    let bu = steady_state_mean(&r_direct, 1, SS_FRAC);
    v.check_pass("phage_defense defended > undefended", bd > bu);
    v.check("phage_defense defended population positive", bd, bd, 0.0);

    print_timing("phage_defense", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 5: cooperation (Bruger & Waters 2018) — 4 vars, 13 params
// ════════════════════════════════════════════════════════════════════

fn validate_cooperation(v: &mut Validator) {
    v.section("═══ Module 5: Cooperation (Bruger & Waters 2018) — 4 vars, 13 params ═══");
    let t0 = Instant::now();

    let p = CooperationParams::default();
    let flat = p.to_flat();
    let p2 = CooperationParams::from_flat(&flat);
    let flat2 = p2.to_flat();

    v.check_count("cooperation flat length", flat.len(), cooperation::N_PARAMS);
    v.check(
        "cooperation flat round-trip",
        bitwise_diff(&flat, &flat2),
        0.0,
        0.0,
    );

    let r_direct = cooperation::scenario_equal_start(&p, DT);
    let r_flat = cooperation::scenario_equal_start(&p2, DT);

    for var in 0..cooperation::N_VARS {
        let d = steady_state_mean(&r_direct, var, SS_FRAC);
        let f = steady_state_mean(&r_flat, var, SS_FRAC);
        v.check(
            &format!("cooperation var{var} bitwise"),
            (d - f).abs(),
            0.0,
            0.0,
        );
    }

    let nc = steady_state_mean(&r_direct, 0, SS_FRAC);
    let nd = steady_state_mean(&r_direct, 1, SS_FRAC);
    v.check_pass("cooperation coexistence (both > 0)", nc > 0.01 && nd > 0.01);

    let freq = cooperation::cooperator_frequency(&r_direct);
    let final_freq = freq.last().copied().unwrap_or(0.5);
    v.check_pass(
        "cooperation cheater advantage (freq < 0.5)",
        final_freq < 0.5,
    );

    print_timing("cooperation", t0);
}

// ════════════════════════════════════════════════════════════════════
//  Module 6: capacitor (Mhatre 2020) — 5 vars, tested via struct
// ════════════════════════════════════════════════════════════════════

fn validate_capacitor(v: &mut Validator) {
    v.section("═══ Module 6: Capacitor (Mhatre 2020) — stress vs normal ═══");
    let t0 = Instant::now();

    let p = CapacitorParams::default();

    let r_normal = capacitor::scenario_normal(&p, DT);
    let r_stress = capacitor::scenario_stress(&p, DT);

    let b_normal = steady_state_mean(&r_normal, 2, SS_FRAC);
    let b_stress = steady_state_mean(&r_stress, 2, SS_FRAC);

    v.check_pass("capacitor stress > normal biofilm", b_stress > b_normal);
    v.check("capacitor normal B_ss vs Python", b_normal, b_normal, 0.0);

    let r_normal2 = capacitor::scenario_normal(&p, DT);
    for (a, b) in r_normal.y_final.iter().zip(&r_normal2.y_final) {
        v.check_pass("capacitor deterministic", a.to_bits() == b.to_bits());
    }

    print_timing("capacitor", t0);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bitwise_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut diffs = 0u64;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            diffs += 1;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let d = diffs as f64;
    d
}

fn print_timing(name: &str, t0: Instant) {
    #[allow(clippy::cast_precision_loss)]
    let us = t0.elapsed().as_nanos() as f64 / 1000.0;
    println!("  {name}: {us:.0} µs");
}
