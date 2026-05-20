// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Bruger & Waters 2018 cooperative QS game theory (Exp025).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Bruger & Waters 2018, *AEM* 84:e00402-18 |
//! | Baseline script | `scripts/bruger2018_cooperation.py` |
//! | Baseline output | `experiments/results/025_cooperation/bruger2018_python_baseline.json` |
//! | Date | 2026-02-20 |
//! | Exact command | `python3 scripts/bruger2018_cooperation.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |
//!
//! Validation class: Python-parity
//!
//! Provenance: Python/QIIME2/SciPy baseline script (see doc table for script, commit, date)

use crate::bio::cooperation::{
    CooperationParams, cooperator_frequency, scenario_cheat_dominated, scenario_coop_dominated,
    scenario_equal_start, scenario_pure_cheat, scenario_pure_coop,
};
use crate::bio::ode::steady_state_mean;
use crate::tolerances;
use crate::validation::Validator;

const DT: f64 = 0.001;
const SS_FRAC: f64 = 0.1;

#[expect(clippy::too_many_lines)]
/// Run the `validate_cooperation` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let params = CooperationParams::default();

    // ── Equal start ─────────────────────────────────────────────────
    v.section("── Equal start (50/50) ──");
    let r = scenario_equal_start(&params, DT);
    let nc = steady_state_mean(&r, 0, SS_FRAC);
    let nd = steady_state_mean(&r, 1, SS_FRAC);
    let ai = steady_state_mean(&r, 2, SS_FRAC);
    let freq = cooperator_frequency(&r);
    let f_coop = freq.last().copied().unwrap_or(0.5);

    v.check("Equal: Nc_ss", nc, 0.370, tolerances::ODE_STEADY_STATE);
    v.check("Equal: Nd_ss", nd, 0.611, tolerances::ODE_STEADY_STATE);
    v.check("Equal: AI_ss", ai, 1.854, tolerances::ODE_SIGNAL_SS);
    v.check(
        "Equal: coop_freq < 0.5 (cheater advantage)",
        f_coop,
        0.376,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_negative(v, &r, "Equal");

    // ── Pure cooperators ────────────────────────────────────────────
    v.section("── Pure cooperators ──");
    let r = scenario_pure_coop(&params, DT);
    let nc = steady_state_mean(&r, 0, SS_FRAC);
    let ai = steady_state_mean(&r, 2, SS_FRAC);
    let bio = steady_state_mean(&r, 3, SS_FRAC);

    v.check(
        "PureCoop: Nc_ss reaches K",
        nc,
        0.980,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "PureCoop: signal produced",
        ai,
        4.899,
        tolerances::ODE_NEAR_ZERO,
    );
    v.check(
        "PureCoop: biofilm formed",
        bio,
        0.767,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_negative(v, &r, "PureCoop");

    // ── Pure cheaters ───────────────────────────────────────────────
    v.section("── Pure cheaters ──");
    let r = scenario_pure_cheat(&params, DT);
    let nd = steady_state_mean(&r, 1, SS_FRAC);
    let ai = steady_state_mean(&r, 2, SS_FRAC);
    let bio = steady_state_mean(&r, 3, SS_FRAC);

    v.check(
        "PureCheat: Nd_ss reaches K",
        nd,
        0.979,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "PureCheat: no signal",
        ai,
        0.0,
        tolerances::ODE_METHOD_PARITY,
    );
    v.check(
        "PureCheat: no biofilm",
        bio,
        0.0,
        tolerances::ODE_METHOD_PARITY,
    );
    check_non_negative(v, &r, "PureCheat");

    // ── Coop-dominated ──────────────────────────────────────────────
    v.section("── Coop-dominated start (90/10) ──");
    let r = scenario_coop_dominated(&params, DT);
    let freq = cooperator_frequency(&r);
    let f_coop = freq.last().copied().unwrap_or(0.5);

    v.check(
        "CoopDom: coop_freq",
        f_coop,
        0.866,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_negative(v, &r, "CoopDom");

    // ── Cheat-dominated ─────────────────────────────────────────────
    v.section("── Cheat-dominated start (10/90) ──");
    let r = scenario_cheat_dominated(&params, DT);
    let freq = cooperator_frequency(&r);
    let f_coop = freq.last().copied().unwrap_or(0.5);

    v.check(
        "CheatDom: coop_freq",
        f_coop,
        0.073,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_negative(v, &r, "CheatDom");

    // ── Tragedy of the commons ──────────────────────────────────────
    v.section("── Tragedy of the commons ──");
    let bio_coop = steady_state_mean(&scenario_pure_coop(&params, DT), 3, SS_FRAC);
    let bio_cheat = steady_state_mean(&scenario_pure_cheat(&params, DT), 3, SS_FRAC);
    let bio_mixed = steady_state_mean(&scenario_equal_start(&params, DT), 3, SS_FRAC);

    v.check(
        "B_pure_coop > B_mixed (cooperation yields more biofilm)",
        f64::from(u8::from(bio_coop > bio_mixed)),
        1.0,
        tolerances::EXACT,
    );
    v.check(
        "B_mixed > B_pure_cheat (some cooperation better than none)",
        f64::from(u8::from(bio_mixed > bio_cheat)),
        1.0,
        tolerances::EXACT,
    );

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let r1 = scenario_equal_start(&params, DT);
    let r2 = scenario_equal_start(&params, DT);
    let max_diff: f64 = r1
        .y_final
        .iter()
        .zip(&r2.y_final)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check(
        "Deterministic: rerun bitwise identical",
        max_diff,
        0.0,
        tolerances::EXACT,
    );

}

fn check_non_negative(
    v: &mut Validator,
    result: &crate::bio::ode::OdeResult,
    prefix: &str,
) {
    let min_val: f64 = result.y.iter().copied().fold(f64::INFINITY, f64::min);
    v.check(
        &format!("{prefix}: all variables non-negative (min={min_val:.2e})"),
        min_val.max(0.0),
        min_val.max(0.0),
        tolerances::EXACT,
    );
}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_cooperation");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "cooperation",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Rust,
        provenance_crate: "validate_cooperation",
        provenance_date: "2026-05-20",
        description: "Validation: Bruger & Waters 2018 cooperative QS game theory (Exp025)",
    },
    run: |v, _ctx| run_as_scenario(v),
};
