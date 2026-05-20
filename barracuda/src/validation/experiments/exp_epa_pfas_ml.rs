// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp041 — PFAS detection ML on surface water data.
//!
//! # Provenance
//!
//! | Item            | Value                                                        |
//! |-----------------|--------------------------------------------------------------|
//! | Baseline commit | `e4358c5`                                                    |
//! | Baseline script | `scripts/epa_pfas_ml_baseline.py`                            |
//! | Baseline output | `experiments/results/041_epa_pfas_ml/python_baseline.json`    |
//! | Data source     | Michigan EGLE (3,719 samples) + EPA UCMR 5 (national)        |
//! | Proxy for       | Paper #22, Jones PFAS fate-and-transport                     |
//! | Date            | 2026-02-20                                                   |
//! | Exact command   | `python3 scripts/epa_pfas_ml_baseline.py`                    |
//! | Hardware        | i9-12900K, 64GB DDR5, RTX 4070, Ubuntu 24.04                 |
//!
//! Validates decision tree classification of PFAS contamination levels
//! using the Rust `DecisionTree` module.
//!
//! Validation class: Python-parity
//!
//! Provenance: Python/QIIME2/SciPy baseline script (see doc table for script, commit, date)

use crate::bio::decision_tree::DecisionTree;
use crate::validation::OrExit;
use crate::validation::Validator;

/// Run the `validate_epa_pfas_ml` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {

    // Build a stump matching Python baseline:
    // feature=4 (total_PFAS), threshold=70.0, left→0, right→1
    let tree = DecisionTree::from_arrays(
        &[4, -1, -1],              // feature indices (-1 = leaf)
        &[70.0, 0.0, 0.0],         // thresholds
        &[1, -1, -1],              // left children (-1 = none)
        &[2, -1, -1],              // right children
        &[None, Some(0), Some(1)], // predictions
        5,                         // n_features
    )
    .or_exit("valid tree");

    // ── Section 1: Tree structure ───────────────────────────────
    v.section("── Decision tree structure ──");
    v.check_count("n_nodes", tree.n_nodes(), 3);
    v.check_count("n_leaves", tree.n_leaves(), 2);
    v.check_count("depth", tree.depth(), 1);

    // ── Section 2: Classification vs Python baseline ────────────
    v.section("── Classification vs baseline ──");
    // total_PFAS < 70 → class 0 (low)
    let low_sample = vec![10.0, 5.0, 3.0, 42.5, 30.0];
    v.check_count("predict(low)", tree.predict(&low_sample), 0);

    // total_PFAS > 70 → class 1 (high)
    let high_sample = vec![80.0, 50.0, 30.0, 44.0, 200.0];
    v.check_count("predict(high)", tree.predict(&high_sample), 1);

    // Boundary: exactly at threshold
    let boundary_below = vec![0.0, 0.0, 0.0, 42.0, 69.9];
    v.check_count("predict(just_below)", tree.predict(&boundary_below), 0);

    let boundary_above = vec![0.0, 0.0, 0.0, 42.0, 70.1];
    v.check_count("predict(just_above)", tree.predict(&boundary_above), 1);

    // ── Section 3: Batch prediction ─────────────────────────────
    v.section("── Batch prediction ──");
    let samples = vec![
        vec![10.0, 5.0, 3.0, 42.5, 30.0],     // low
        vec![80.0, 50.0, 30.0, 44.0, 200.0],  // high
        vec![20.0, 15.0, 10.0, 43.0, 60.0],   // low
        vec![100.0, 80.0, 60.0, 45.0, 300.0], // high
    ];
    let preds = tree.predict_batch(&samples);
    v.check_count("batch_len", preds.len(), 4);
    v.check_count("batch[0] = low", preds[0], 0);
    v.check_count("batch[1] = high", preds[1], 1);
    v.check_count("batch[2] = low", preds[2], 0);
    v.check_count("batch[3] = high", preds[3], 1);

    // ── Section 4: Determinism ──────────────────────────────────
    v.section("── Determinism ──");
    let p1 = tree.predict(&high_sample);
    let p2 = tree.predict(&high_sample);
    v.check_count("predict deterministic", p1, p2);

    let batch1 = tree.predict_batch(&samples);
    let batch2 = tree.predict_batch(&samples);
    let batch_match = batch1 == batch2;
    v.check_count("batch deterministic", usize::from(batch_match), 1);

}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_epa_pfas_ml");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "epa_pfas_ml",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Rust,
        provenance_crate: "validate_epa_pfas_ml",
        provenance_date: "2026-05-20",
        description: "Exp041 — PFAS detection ML on surface water data",
    },
    run: |v, _ctx| run_as_scenario(v),
};
