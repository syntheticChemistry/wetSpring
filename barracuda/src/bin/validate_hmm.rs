// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation: Liu 2014 HMM primitives (Exp026).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Liu et al. 2014, *`PLoS` Comp Bio* 10:e1003649 |
//! | Baseline script | `scripts/liu2014_hmm_baseline.py` |
//! | Baseline commit | `e4358c5` |
//! | Baseline output | `experiments/results/026_hmm/liu2014_hmm_python_baseline.json` |
//! | Date | 2026-02-21 |
//! | Exact command | `python3 scripts/liu2014_hmm_baseline.py` |

use wetspring_barracuda::bio::hmm::{forward, posterior, viterbi, HmmModel};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn weather_model() -> HmmModel {
    HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 3,
        log_emit: vec![
            0.1_f64.ln(),
            0.4_f64.ln(),
            0.5_f64.ln(),
            0.6_f64.ln(),
            0.3_f64.ln(),
            0.1_f64.ln(),
        ],
    }
}

fn three_state_model() -> HmmModel {
    let third = (1.0_f64 / 3.0).ln();
    HmmModel {
        n_states: 3,
        log_pi: vec![third, third, third],
        log_trans: vec![
            0.5_f64.ln(),
            0.3_f64.ln(),
            0.2_f64.ln(),
            0.2_f64.ln(),
            0.5_f64.ln(),
            0.3_f64.ln(),
            0.3_f64.ln(),
            0.2_f64.ln(),
            0.5_f64.ln(),
        ],
        n_symbols: 2,
        log_emit: vec![
            0.9_f64.ln(),
            0.1_f64.ln(),
            0.2_f64.ln(),
            0.8_f64.ln(),
            0.5_f64.ln(),
            0.5_f64.ln(),
        ],
    }
}

fn main() {
    let mut v = Validator::new("Exp026: Liu 2014 HMM Primitives");

    // ── 2-state weather model ───────────────────────────────────────
    v.section("── 2-state weather model (obs = [Walk, Shop, Clean, Walk, Shop]) ──");
    let model = weather_model();
    let obs = vec![0, 1, 2, 0, 1];

    let fwd = forward(&model, &obs);
    v.check(
        "2s: log-likelihood",
        fwd.log_likelihood,
        -5.758_240_012_089_921,
        tolerances::PYTHON_PARITY,
    );

    let vit = viterbi(&model, &obs);
    v.check(
        "2s: Viterbi log-prob",
        vit.log_probability,
        -7.739_116_800_070_987,
        tolerances::PYTHON_PARITY,
    );

    let expected_path_2 = [1, 0, 0, 1, 1];
    let path_match = vit.path.iter().zip(&expected_path_2).all(|(a, b)| a == b);
    v.check(
        "2s: Viterbi path matches Python",
        f64::from(u8::from(path_match)),
        1.0,
        0.0,
    );

    let gamma = posterior(&model, &obs);
    for t in 0..obs.len() {
        let row_sum: f64 = (0..model.n_states)
            .map(|i| gamma[t * model.n_states + i])
            .sum();
        v.check(
            &format!("2s: posterior t={t} sums to 1"),
            row_sum,
            1.0,
            tolerances::PYTHON_PARITY,
        );
    }

    // ── 3-state genomic model ───────────────────────────────────────
    v.section("── 3-state genomic model (obs = [0,1,0,0,1,1,0]) ──");
    let model3 = three_state_model();
    let obs3 = vec![0, 1, 0, 0, 1, 1, 0];

    let fwd3 = forward(&model3, &obs3);
    v.check(
        "3s: log-likelihood",
        fwd3.log_likelihood,
        -5.017_483_023_983_44,
        tolerances::PYTHON_PARITY,
    );

    let vit3 = viterbi(&model3, &obs3);
    v.check(
        "3s: Viterbi log-prob",
        vit3.log_probability,
        -9.202_600_799_882_006,
        tolerances::PYTHON_PARITY,
    );

    let expected_path_3 = [0, 1, 0, 0, 1, 1, 0];
    let path_match_3 = vit3.path.iter().zip(&expected_path_3).all(|(a, b)| a == b);
    v.check(
        "3s: Viterbi path matches Python",
        f64::from(u8::from(path_match_3)),
        1.0,
        0.0,
    );

    let gamma3 = posterior(&model3, &obs3);
    for t in 0..obs3.len() {
        let row_sum: f64 = (0..model3.n_states)
            .map(|i| gamma3[t * model3.n_states + i])
            .sum();
        v.check(
            &format!("3s: posterior t={t} sums to 1"),
            row_sum,
            1.0,
            tolerances::PYTHON_PARITY,
        );
    }

    // ── Viterbi ≤ Forward ───────────────────────────────────────────
    v.section("── Mathematical invariants ──");
    v.check(
        "2s: Viterbi ≤ forward (most likely path ≤ total)",
        f64::from(u8::from(vit.log_probability <= fwd.log_likelihood + 1e-10)),
        1.0,
        0.0,
    );
    v.check(
        "3s: Viterbi ≤ forward",
        f64::from(u8::from(
            vit3.log_probability <= fwd3.log_likelihood + 1e-10,
        )),
        1.0,
        0.0,
    );

    // ── Determinism ─────────────────────────────────────────────────
    v.section("── Determinism ──");
    let fwd_a = forward(&model, &obs);
    let fwd_b = forward(&model, &obs);
    v.check(
        "Forward bitwise deterministic",
        (fwd_a.log_likelihood - fwd_b.log_likelihood).abs(),
        0.0,
        0.0,
    );

    v.finish();
}
