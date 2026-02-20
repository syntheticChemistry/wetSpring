// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Srivastava 2011 multi-input QS network (Exp024).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Srivastava et al. 2011, *J Bacteriology* 193:6331-41 |
//! | Baseline script | `scripts/srivastava2011_multi_signal.py` |
//! | Baseline output | `experiments/results/024_multi_signal/srivastava2011_python_baseline.json` |
//! | Python version | 3.10.12 |
//! | scipy integrator | `odeint` (LSODA) |
//! | Rust integrator | RK4 fixed-step (dt = 0.001 h) |
//! | Date | 2026-02-20 |

use wetspring_barracuda::bio::multi_signal::{
    scenario_ai2_only, scenario_cai1_only, scenario_exogenous_cai1, scenario_no_qs,
    scenario_wild_type, MultiSignalParams,
};
use wetspring_barracuda::bio::ode::steady_state_mean;
use wetspring_barracuda::validation::Validator;

const DT: f64 = 0.001;
const SS_FRAC: f64 = 0.1;
const METHOD_TOL: f64 = 1e-3;

fn main() {
    let mut v = Validator::new("Exp024: Srivastava 2011 Multi-Input QS Network");
    let params = MultiSignalParams::default();

    // ── Wild type ────────────────────────────────────────────────────
    v.section("── Wild type (both signals) ──");
    let r = scenario_wild_type(&params, DT);
    let n_ss = steady_state_mean(&r, 0, SS_FRAC);
    let h_ss = steady_state_mean(&r, 4, SS_FRAC);
    let c_ss = steady_state_mean(&r, 5, SS_FRAC);
    let b_ss = steady_state_mean(&r, 6, SS_FRAC);

    v.check("WT: N_ss", n_ss, 0.975, METHOD_TOL);
    v.check("WT: HapR_ss", h_ss, 0.543, 0.01);
    v.check("WT: CdG_ss", c_ss, 0.600, 0.01);
    v.check("WT: B_ss", b_ss, 0.413, 0.01);
    check_non_negative(&mut v, &r, "WT");

    // ── CAI-1 only (ΔluxS) ─────────────────────────────────────────
    v.section("── CAI-1 only (ΔluxS) ──");
    let r = scenario_cai1_only(&params, DT);
    let h_ss = steady_state_mean(&r, 4, SS_FRAC);
    let b_ss = steady_state_mean(&r, 6, SS_FRAC);

    v.check("CAI1: HapR_ss", h_ss, 0.238, 0.01);
    v.check("CAI1: B_ss (more biofilm)", b_ss, 0.676, 0.01);
    check_non_negative(&mut v, &r, "CAI1");

    // ── AI-2 only (ΔcqsA) ──────────────────────────────────────────
    v.section("── AI-2 only (ΔcqsA) ──");
    let r = scenario_ai2_only(&params, DT);
    let h_ss = steady_state_mean(&r, 4, SS_FRAC);
    let b_ss = steady_state_mean(&r, 6, SS_FRAC);

    v.check("AI2: HapR_ss (symmetric with CAI1)", h_ss, 0.238, 0.01);
    v.check("AI2: B_ss (symmetric with CAI1)", b_ss, 0.676, 0.01);
    check_non_negative(&mut v, &r, "AI2");

    // ── No QS (ΔluxS ΔcqsA) ────────────────────────────────────────
    v.section("── No QS (ΔluxS ΔcqsA) ──");
    let r = scenario_no_qs(&params, DT);
    let h_ss = steady_state_mean(&r, 4, SS_FRAC);
    let b_ss = steady_state_mean(&r, 6, SS_FRAC);

    v.check("NoQS: HapR_ss (very low)", h_ss, 0.031, 0.005);
    v.check("NoQS: B_ss (constitutive biofilm)", b_ss, 0.777, 0.01);
    check_non_negative(&mut v, &r, "NoQS");

    // ── Exogenous CAI-1 ─────────────────────────────────────────────
    v.section("── Exogenous CAI-1 (low density + signal) ──");
    let r = scenario_exogenous_cai1(&params, DT);
    let h_ss = steady_state_mean(&r, 4, SS_FRAC);

    v.check(
        "ExoCAI: HapR activated by exogenous signal",
        h_ss,
        0.543,
        0.01,
    );
    check_non_negative(&mut v, &r, "ExoCAI");

    // ── Signal hierarchy ────────────────────────────────────────────
    v.section("── Signal hierarchy ──");
    let h_wt = steady_state_mean(&scenario_wild_type(&params, DT), 4, SS_FRAC);
    let h_cai1 = steady_state_mean(&scenario_cai1_only(&params, DT), 4, SS_FRAC);
    let h_noqs = steady_state_mean(&scenario_no_qs(&params, DT), 4, SS_FRAC);

    v.check(
        "Hierarchy: H_wt > H_cai1",
        f64::from(u8::from(h_wt > h_cai1)),
        1.0,
        0.0,
    );
    v.check(
        "Hierarchy: H_cai1 > H_noqs",
        f64::from(u8::from(h_cai1 > h_noqs)),
        1.0,
        0.0,
    );

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let r1 = scenario_wild_type(&params, DT);
    let r2 = scenario_wild_type(&params, DT);
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
    v.check(
        &format!("{prefix}: all variables non-negative (min={min_val:.2e})"),
        min_val.max(0.0),
        min_val.max(0.0),
        0.0,
    );
}
