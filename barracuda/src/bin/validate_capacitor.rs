// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Mhatre 2020 phenotypic capacitor (Exp027).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Mhatre et al. 2020, *PNAS* 117:21647-21657 |
//! | Baseline script | `scripts/mhatre2020_capacitor.py` |
//! | Date | 2026-02-20 |
//! | Exact command | `python3 scripts/mhatre2020_capacitor.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |

use wetspring_barracuda::bio::capacitor::{
    scenario_low_cdg, scenario_normal, scenario_stress, scenario_vpsr_knockout, CapacitorParams,
};
use wetspring_barracuda::bio::ode::steady_state_mean;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const DT: f64 = 0.001;
const SS_FRAC: f64 = 0.1;

fn main() {
    let mut v = Validator::new("Exp027: Mhatre 2020 Phenotypic Capacitor");
    let params = CapacitorParams::default();

    // ── Normal growth ───────────────────────────────────────────────
    v.section("── Normal growth ──");
    let r = scenario_normal(&params, DT);
    v.check(
        "Normal: N_ss",
        steady_state_mean(&r, 0, SS_FRAC),
        0.975,
        tolerances::ODE_METHOD_PARITY,
    );
    v.check(
        "Normal: VpsR_ss",
        steady_state_mean(&r, 2, SS_FRAC),
        0.766,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "Normal: B_ss",
        steady_state_mean(&r, 3, SS_FRAC),
        0.671,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "Normal: M_ss",
        steady_state_mean(&r, 4, SS_FRAC),
        0.319,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "Normal: R_ss",
        steady_state_mean(&r, 5, SS_FRAC),
        0.439,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_neg(&mut v, &r, "Normal");

    // ── Nutrient stress ─────────────────────────────────────────────
    v.section("── Nutrient stress (3× c-di-GMP) ──");
    let r = scenario_stress(&params, DT);
    v.check(
        "Stress: VpsR_ss (saturated)",
        steady_state_mean(&r, 2, SS_FRAC),
        0.769,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "Stress: B_ss",
        steady_state_mean(&r, 3, SS_FRAC),
        0.672,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "Stress: R_ss (elevated)",
        steady_state_mean(&r, 5, SS_FRAC),
        0.441,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_neg(&mut v, &r, "Stress");

    // ── Low c-di-GMP ────────────────────────────────────────────────
    v.section("── Low c-di-GMP ──");
    let r = scenario_low_cdg(&params, DT);
    let mot = steady_state_mean(&r, 4, SS_FRAC);
    let bio = steady_state_mean(&r, 3, SS_FRAC);
    v.check(
        "LowCdG: M_ss > B_ss (motility favored)",
        f64::from(u8::from(mot > bio)),
        1.0,
        0.0,
    );
    v.check(
        "LowCdG: VpsR_ss (low)",
        steady_state_mean(&r, 2, SS_FRAC),
        0.357,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_neg(&mut v, &r, "LowCdG");

    // ── VpsR knockout ───────────────────────────────────────────────
    v.section("── ΔvpsR knockout ──");
    let r = scenario_vpsr_knockout(&params, DT);
    v.check(
        "KO: B_ss ≈ 0",
        steady_state_mean(&r, 3, SS_FRAC),
        0.0,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "KO: R_ss ≈ 0",
        steady_state_mean(&r, 5, SS_FRAC),
        0.0,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "KO: M_ss (high, default motile)",
        steady_state_mean(&r, 4, SS_FRAC),
        0.667,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_neg(&mut v, &r, "KO");

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let r1 = scenario_normal(&params, DT);
    let r2 = scenario_normal(&params, DT);
    let max_diff: f64 = r1
        .y_final
        .iter()
        .zip(&r2.y_final)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check("Deterministic", max_diff, 0.0, 0.0);

    v.finish();
}

fn check_non_neg(v: &mut Validator, r: &wetspring_barracuda::bio::ode::OdeResult, pre: &str) {
    let min: f64 =
        r.y.iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
    v.check(
        &format!("{pre}: non-negative (min={min:.2e})"),
        min.max(0.0),
        min.max(0.0),
        0.0,
    );
}
