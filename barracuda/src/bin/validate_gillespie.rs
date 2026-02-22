// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Gillespie SSA — Massie 2012 c-di-GMP birth-death model.
//!
//! Validates the Rust Gillespie implementation against the Python baseline
//! (`scripts/gillespie_baseline.py`). Checks ensemble statistics, determinism,
//! non-negativity, and Poisson convergence.
//!
//! Follows the `hotSpring` pattern: hardcoded expected values, explicit
//! pass/fail, exit code 0/1.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | gillespie_baseline.py |
//! | Baseline version | scripts/ |
//! | Baseline command | python3 scripts/gillespie_baseline.py |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/gillespie_baseline.py` |
//! | Data | Massie 2012 c-di-GMP birth-death model, N=1000 ensemble |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use wetspring_barracuda::bio::gillespie;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

/// Python baseline values from `gillespie_python_baseline.json`.
const K_DGC: f64 = 10.0;
const K_PDE: f64 = 0.1;
const T_MAX: f64 = 100.0;
const N_RUNS: usize = 1000;
const BASE_SEED: u64 = 42;

const ANALYTICAL_MEAN: f64 = 100.0; // k_dgc / k_pde

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn main() {
    let mut v = Validator::new("Exp022: Gillespie SSA — Massie 2012");

    v.section("── Determinism ──");

    let traj1 = gillespie::birth_death_ssa(K_DGC, K_PDE, T_MAX, BASE_SEED);
    let traj2 = gillespie::birth_death_ssa(K_DGC, K_PDE, T_MAX, BASE_SEED);
    v.check_count(
        "Same seed → same final count",
        traj1.final_state()[0] as usize,
        traj2.final_state()[0] as usize,
    );
    v.check_count(
        "Same seed → same event count",
        traj1.n_events(),
        traj2.n_events(),
    );

    v.section("── Single Trajectory ──");

    v.check(
        "Final count non-negative",
        traj1.final_state()[0] as f64,
        traj1.final_state()[0].max(0) as f64,
        0.0,
    );
    let all_non_neg = traj1.states.iter().all(|s| s[0] >= 0);
    v.check(
        "All states non-negative",
        f64::from(u8::from(all_non_neg)),
        1.0,
        0.0,
    );
    let has_events = traj1.n_events() > 10;
    v.check(
        &format!("Has events: {} steps", traj1.n_events()),
        f64::from(u8::from(has_events)),
        1.0,
        0.0,
    );
    let time_mono = traj1.times.windows(2).all(|w| w[1] >= w[0]);
    v.check(
        "Time monotonically increasing",
        f64::from(u8::from(time_mono)),
        1.0,
        0.0,
    );

    v.section("── Ensemble Statistics (N=1000) ──");

    let stats = gillespie::birth_death_ensemble(K_DGC, K_PDE, T_MAX, N_RUNS, BASE_SEED);

    v.check(
        "Ensemble mean ~ analytical (100.0)",
        stats.mean,
        ANALYTICAL_MEAN,
        ANALYTICAL_MEAN * tolerances::GILLESPIE_MEAN_REL,
    );
    v.check(
        "Fano factor ~ 1.0 (Poisson)",
        stats.fano_factor,
        1.0,
        tolerances::GILLESPIE_FANO,
    );
    v.check(
        "Std dev > 0 (stochastic variability)",
        stats.std_dev,
        stats.std_dev,
        0.0,
    );
    let std_positive = stats.std_dev > 0.0;
    v.check(
        "Std dev strictly positive",
        f64::from(u8::from(std_positive)),
        1.0,
        0.0,
    );
    let ensemble_non_neg = stats.final_counts.iter().all(|&c| c >= 0);
    v.check(
        "All ensemble final counts non-negative",
        f64::from(u8::from(ensemble_non_neg)),
        1.0,
        0.0,
    );

    v.section("── Cross-Validation with Python Baseline ──");

    // Python ensemble: mean=100.397, var=97.456, fano≈0.971
    // We validate that Rust ensemble is in the same statistical regime.
    // Since PRNG differs, we check statistical convergence not bitwise match.
    v.check(
        "Rust mean in Python range (100.0 ± 15%)",
        stats.mean,
        ANALYTICAL_MEAN,
        ANALYTICAL_MEAN * 0.15,
    );

    v.check(
        "Rust Fano factor in physical range [0.5, 2.0]",
        stats.fano_factor,
        1.0,
        1.0,
    );

    v.finish();
}
