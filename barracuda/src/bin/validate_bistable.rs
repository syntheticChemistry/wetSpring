// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Fernandez 2020 bistable phenotypic switching (Exp023).
//!
//! Validates the Rust bistable ODE model against the Python/scipy baseline.
//! Tests steady-state values for zero-feedback, default, and strong feedback
//! scenarios, plus bifurcation hysteresis detection.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Fernandez et al. 2020, *PNAS* 117:29046-29054 |
//! | Baseline script | `scripts/fernandez2020_bistable.py` |
//! | Baseline output | `experiments/results/023_bistable/fernandez2020_python_baseline.json` |
//! | Python version | 3.10.12 |
//! | scipy integrator | `odeint` (LSODA: adaptive BDF/Adams) |
//! | Rust integrator | RK4 fixed-step (dt = 0.001 h) |
//! | Date | 2026-02-20 |
//! | Exact command | `python3 scripts/fernandez2020_bistable.py` |
//! | Hardware | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04 |

use wetspring_barracuda::bio::bistable::{bifurcation_scan, run_bistable, BistableParams};
use wetspring_barracuda::bio::ode::steady_state_mean;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const DT: f64 = 0.001;
const SS_FRAC: f64 = 0.1;

fn main() {
    let mut v = Validator::new("Exp023: Fernandez 2020 Bistable Phenotypic Switching");

    // ── Scenario 1: Zero feedback (recover low-B monostable) ─────────
    v.section("── Scenario 1: Zero feedback (alpha_fb = 0) ──");
    let p = BistableParams {
        alpha_fb: 0.0,
        ..Default::default()
    };
    let r = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 48.0, DT, &p);
    let n_ss = steady_state_mean(&r, 0, SS_FRAC);
    let b_ss = steady_state_mean(&r, 4, SS_FRAC);
    let c_ss = steady_state_mean(&r, 3, SS_FRAC);

    v.check(
        "S1: N_ss (carrying capacity)",
        n_ss,
        0.975,
        tolerances::ODE_METHOD_PARITY,
    );
    v.check(
        "S1: B_ss (low biofilm)",
        b_ss,
        0.040,
        tolerances::GC_CONTENT,
    );
    v.check(
        "S1: C_ss (moderate c-di-GMP)",
        c_ss,
        0.454,
        tolerances::ODE_STEADY_STATE,
    );
    check_non_negative(&mut v, &r, "S1");

    // ── Scenario 2: Default feedback (high-B attractor) ──────────────
    v.section("── Scenario 2: Default feedback (alpha_fb = 3.0) ──");
    let p = BistableParams::default();
    let r = run_bistable(&[0.01, 0.0, 0.0, 3.0, 0.9], 48.0, DT, &p);
    let b_ss = steady_state_mean(&r, 4, SS_FRAC);
    let c_ss = steady_state_mean(&r, 3, SS_FRAC);

    v.check(
        "S2: B_ss matches Python (sessile)",
        b_ss,
        0.745,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "S2: C_ss matches Python",
        c_ss,
        1.634,
        tolerances::KMD_SPREAD,
    );
    check_non_negative(&mut v, &r, "S2");

    // ── Scenario 3: Strong feedback (alpha_fb = 8.0) ─────────────────
    v.section("── Scenario 3: Strong feedback (alpha_fb = 8.0) ──");
    let p = BistableParams {
        alpha_fb: 8.0,
        ..Default::default()
    };
    let r = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.8], 48.0, DT, &p);
    let b_ss = steady_state_mean(&r, 4, SS_FRAC);
    let c_ss = steady_state_mean(&r, 3, SS_FRAC);

    v.check(
        "S3: B_ss matches Python",
        b_ss,
        0.831,
        tolerances::ODE_STEADY_STATE,
    );
    v.check(
        "S3: C_ss matches Python",
        c_ss,
        3.967,
        tolerances::ODE_NEAR_ZERO,
    );
    check_non_negative(&mut v, &r, "S3");

    // ── Bifurcation / Hysteresis ─────────────────────────────────────
    v.section("── Bifurcation scan (alpha_fb: 0 → 10) ──");
    let p = BistableParams::default();
    let bif = bifurcation_scan(&p, 0.0, 10.0, 50, 0.01, 48.0);

    v.check(
        "Hysteresis width > 0 (bistability)",
        bif.hysteresis_width,
        bif.hysteresis_width.max(0.5),
        tolerances::ODE_STEADY_STATE,
    );

    let fwd_stays_low = bif.b_forward.iter().all(|&b| b < 0.1);
    v.check(
        "Forward sweep stays in low-B attractor",
        f64::from(u8::from(fwd_stays_low)),
        1.0,
        0.0,
    );

    let bwd_has_high = bif.b_backward.iter().any(|&b| b > 0.5);
    v.check(
        "Backward sweep visits high-B attractor",
        f64::from(u8::from(bwd_has_high)),
        1.0,
        0.0,
    );

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let p = BistableParams::default();
    let r1 = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, DT, &p);
    let r2 = run_bistable(&[0.01, 0.0, 0.0, 2.0, 0.5], 24.0, DT, &p);
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
        .copied()
        .fold(f64::INFINITY, f64::min);
    v.check(
        &format!("{prefix}: all variables non-negative (min={min_val:.2e})"),
        min_val.max(0.0),
        min_val.max(0.0),
        0.0,
    );
    if min_val < 0.0 {
        v.check(
            &format!("{prefix}: NEGATIVE VALUE DETECTED: {min_val:.6e}"),
            min_val,
            0.0,
            0.0,
        );
    }
}
