// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation binary: Waters 2008 QS/c-di-GMP ODE model (Exp020).
//!
//! Runs 4 biological scenarios through the Rust RK4 integrator and compares
//! steady-state values against the Python/scipy baseline.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline script | `scripts/waters2008_qs_ode.py` |
//! | Baseline output | `experiments/results/qs_ode_baseline/qs_ode_python_baseline.json` |
//! | Python version | 3.10.12 |
//! | scipy integrator | `odeint` (LSODA: adaptive BDF/Adams) |
//! | Rust integrator | RK4 fixed-step (dt = 0.001 h) |
//! | References | Waters 2008 J Bacteriol 190:2527-36; Massie 2012 PNAS 109:12746-51 |
//! | Date | 2026-02-19 |
//! | Exact command | `python3 scripts/waters2008_qs_ode.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |

use wetspring_barracuda::bio::ode::steady_state_mean;
use wetspring_barracuda::bio::qs_biofilm::{
    scenario_dgc_overexpression, scenario_hapr_mutant, scenario_high_density,
    scenario_standard_growth, QsBiofilmParams,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const DT: f64 = 0.001;
const SS_FRAC: f64 = 0.1;

/// ODE method parity — centralized in `tolerances::ODE_METHOD_PARITY`.
const METHOD_TOL: f64 = tolerances::ODE_METHOD_PARITY;

fn main() {
    let mut v = Validator::new("validate_qs_ode (Exp020: Waters 2008 QS/c-di-GMP)");
    let params = QsBiofilmParams::default();

    // ── Scenario 1: Standard Growth ─────────────────────────────────
    v.section("── Scenario 1: Standard Growth (low → high density) ──");
    let r = scenario_standard_growth(&params, DT);
    let n_ss = steady_state_mean(&r, 0, SS_FRAC);
    let h_ss = steady_state_mean(&r, 2, SS_FRAC);
    let c_ss = steady_state_mean(&r, 3, SS_FRAC);
    let b_ss = steady_state_mean(&r, 4, SS_FRAC);

    // Python: N_ss = 0.974998, H_ss = 1.978942, C_ss ≈ 0.0, B_ss = 0.02021
    v.check(
        "S1: N_ss matches Python (carrying capacity)",
        n_ss,
        0.975,
        METHOD_TOL,
    );
    v.check(
        "S1: H_ss matches Python (HapR active)",
        h_ss,
        1.979,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "S1: C_ss near zero (c-di-GMP repressed)",
        c_ss,
        0.0,
        tolerances::ODE_NEAR_ZERO,
    );
    v.check(
        "S1: B_ss matches Python (biofilm dispersed)",
        b_ss,
        0.020,
        0.03,
    );
    check_non_negative(&mut v, &r, "S1");

    // ── Scenario 2: High-Density Inoculum ───────────────────────────
    v.section("── Scenario 2: High-Density Inoculum (dispersal) ──");
    let r = scenario_high_density(&params, DT);
    let n_ss = steady_state_mean(&r, 0, SS_FRAC);
    let b_ss = steady_state_mean(&r, 4, SS_FRAC);

    // Python: N_ss = 0.97497, B_ss = 0.10376
    v.check("S2: N_ss matches Python", n_ss, 0.975, METHOD_TOL);
    v.check(
        "S2: B_ss matches Python (rapid dispersal)",
        b_ss,
        0.104,
        0.03,
    );
    check_non_negative(&mut v, &r, "S2");

    // ── Scenario 3: ΔhapR Mutant ────────────────────────────────────
    v.section("── Scenario 3: ΔhapR Mutant (constitutive biofilm) ──");
    let r = scenario_hapr_mutant(&params, DT);
    let h_ss = steady_state_mean(&r, 2, SS_FRAC);
    let c_ss = steady_state_mean(&r, 3, SS_FRAC);
    let b_ss = steady_state_mean(&r, 4, SS_FRAC);

    // Python: H_ss = 0.0, C_ss = 2.5, B_ss = 0.786164
    v.check(
        "S3: H_ss zero (HapR knocked out)",
        h_ss,
        0.0,
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "S3: C_ss matches Python (c-di-GMP high)",
        c_ss,
        2.500,
        METHOD_TOL,
    );
    v.check(
        "S3: B_ss matches Python (constitutive biofilm)",
        b_ss,
        0.786,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_negative(&mut v, &r, "S3");

    // ── Scenario 4: DGC Overexpression ──────────────────────────────
    v.section("── Scenario 4: DGC Overexpression (elevated c-di-GMP) ──");
    let r = scenario_dgc_overexpression(&params, DT);
    let c_ss = steady_state_mean(&r, 3, SS_FRAC);
    let b_ss = steady_state_mean(&r, 4, SS_FRAC);

    // Python: C_ss = 0.662143, B_ss = 0.451545
    v.check(
        "S4: C_ss matches Python (elevated c-di-GMP)",
        c_ss,
        0.662,
        METHOD_TOL,
    );
    v.check(
        "S4: B_ss matches Python (partial biofilm)",
        b_ss,
        0.452,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_negative(&mut v, &r, "S4");

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let r1 = scenario_standard_growth(&params, DT);
    let r2 = scenario_standard_growth(&params, DT);
    let max_diff: f64 = r1
        .y_final
        .iter()
        .zip(&r2.y_final)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check("Deterministic: rerun bitwise identical", max_diff, 0.0, 0.0);

    v.finish();
}

fn check_non_negative(
    v: &mut Validator,
    result: &wetspring_barracuda::bio::ode::OdeResult,
    prefix: &str,
) {
    let min_val: f64 = result
        .y
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .fold(f64::INFINITY, f64::min);
    // min_val should be ≥ 0; we check it's within tolerance of 0 from below
    // by verifying actual ≥ expected (0.0) - tolerance (0.0)
    v.check(
        &format!("{prefix}: all variables non-negative (min={min_val:.2e})"),
        min_val.max(0.0),
        min_val.max(0.0),
        0.0,
    );
    // If min_val was negative, the actual wouldn't equal expected, so this
    // would still technically pass. Use an explicit check instead:
    if min_val < 0.0 {
        v.check(
            &format!("{prefix}: NEGATIVE VALUE DETECTED: {min_val:.6e}"),
            min_val,
            0.0,
            0.0,
        );
    }
}
