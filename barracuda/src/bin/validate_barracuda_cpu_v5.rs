// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
//! Exp061/062: `BarraCUDA` CPU Parity v5 — Random Forest + GBM
//!
//! Validates the sovereign Random Forest and Gradient Boosting Machine
//! inference engines in pure Rust. Proves parity with the functional
//! specifications from sklearn.
//!
//! Combined: domains 24 (Random Forest) and 25 (GBM).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline tool | sklearn (RandomForest, GradientBoostingClassifier; functional spec) |
//! | Baseline version | sklearn 1.x |
//! | Baseline command | Hand-trace of sklearn inference (DT splits, RF majority vote, GBM additive model) |
//! | Baseline date | 2026-02-19 |
//! | Data | Synthetic test vectors (hand-computed majority votes / raw scores) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::time::Instant;
use wetspring_barracuda::bio::{
    decision_tree::DecisionTree,
    gbm::{GbmClassifier, GbmMultiClassifier, GbmTree},
    random_forest::RandomForest,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("BarraCUDA CPU v5 — RF + GBM (Domains 24-25)");
    let mut timings: Vec<(&str, f64)> = Vec::new();

    // ════════════════════════════════════════════════════════════════
    //  Domain 24: Random Forest Ensemble Inference
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 24: Random Forest Ensemble Inference ═══");
    let t0 = Instant::now();

    // Build 5-tree forest (2 features, 3 classes)
    let tree1 = DecisionTree::from_arrays(
        &[0, -2, 1, -2, -2],
        &[5.0, 0.0, 3.0, 0.0, 0.0],
        &[1, -1, 3, -1, -1],
        &[2, -1, 4, -1, -1],
        &[None, Some(0), None, Some(1), Some(2)],
        2,
    )
    .unwrap();

    let tree2 = DecisionTree::from_arrays(
        &[1, -2, -2],
        &[4.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(2)],
        2,
    )
    .unwrap();

    let tree3 = DecisionTree::from_arrays(
        &[0, -2, -2],
        &[6.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(1), Some(2)],
        2,
    )
    .unwrap();

    let tree4 = DecisionTree::from_arrays(
        &[0, 1, -2, -2, -2],
        &[4.0, 2.0, 0.0, 0.0, 0.0],
        &[1, 2, -1, -1, -1],
        &[4, 3, -1, -1, -1],
        &[None, None, Some(0), Some(1), Some(2)],
        2,
    )
    .unwrap();

    let tree5 = DecisionTree::from_arrays(
        &[1, -2, 0, -2, -2],
        &[5.0, 0.0, 7.0, 0.0, 0.0],
        &[1, -1, 3, -1, -1],
        &[2, -1, 4, -1, -1],
        &[None, Some(0), None, Some(1), Some(2)],
        2,
    )
    .unwrap();

    let rf = RandomForest::from_trees(vec![tree1, tree2, tree3, tree4, tree5], 3).unwrap();

    // Structural checks
    v.check("RF: n_trees = 5", rf.n_trees() as f64, 5.0, 0.0);
    v.check("RF: n_features = 2", rf.n_features() as f64, 2.0, 0.0);
    v.check("RF: n_classes = 3", rf.n_classes() as f64, 3.0, 0.0);

    // Prediction checks — hand-computed majority votes
    // Sample [3.0, 1.0]: tree1→0, tree2→0, tree3→1, tree4→0(≤4,≤2), tree5→0(≤5) → class 0 (4-1-0)
    let pred1 = rf.predict_with_votes(&[3.0, 1.0]);
    v.check("RF: [3,1] → class 0", pred1.class as f64, 0.0, 0.0);
    v.check(
        "RF: [3,1] confidence ≥ 0.6",
        f64::from(u8::from(pred1.confidence >= 0.6)),
        1.0,
        0.0,
    );

    // Sample [7.0, 6.0]: tree1→2, tree2→2, tree3→2, tree4→2, tree5→1 (f[0]=7≤7 → class 1)
    let pred2 = rf.predict_with_votes(&[7.0, 6.0]);
    v.check("RF: [7,6] → class 2", pred2.class as f64, 2.0, 0.0);
    v.check(
        "RF: [7,6] conf = 0.8",
        pred2.confidence,
        0.8,
        tolerances::ML_PREDICTION,
    );

    // Batch predict
    let batch = rf.predict_batch(&[vec![3.0, 1.0], vec![7.0, 6.0], vec![5.5, 3.5]]);
    v.check("RF: batch size = 3", batch.len() as f64, 3.0, 0.0);
    v.check("RF: batch[0] = class 0", batch[0] as f64, 0.0, 0.0);
    v.check("RF: batch[1] = class 2", batch[1] as f64, 2.0, 0.0);

    // Tree predictions debug
    let tree_preds = rf.tree_predictions(&[3.0, 1.0]);
    v.check("RF: tree_preds len = 5", tree_preds.len() as f64, 5.0, 0.0);

    // Metadata
    v.check(
        "RF: total_nodes > 0",
        f64::from(u8::from(rf.total_nodes() > 0)),
        1.0,
        0.0,
    );
    v.check(
        "RF: avg_depth > 0",
        f64::from(u8::from(rf.avg_depth() > 0.0)),
        1.0,
        0.0,
    );

    let rf_us = t0.elapsed().as_micros() as f64;
    timings.push(("RF (5 trees, 3 classes, batch=3)", rf_us));

    // ════════════════════════════════════════════════════════════════
    //  Domain 25: Gradient Boosting Machine Inference
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 25: GBM Inference ═══");
    let t0 = Instant::now();

    // Binary GBM: 3 stumps, learning_rate=0.1
    let stump1 = GbmTree::from_arrays(
        &[0, -2, -2],
        &[5.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -1.0, 1.0],
    )
    .unwrap();

    let stump2 = GbmTree::from_arrays(
        &[1, -2, -2],
        &[3.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.8, 0.8],
    )
    .unwrap();

    let stump3 = GbmTree::from_arrays(
        &[0, -2, -2],
        &[7.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 1.5],
    )
    .unwrap();

    let gbm = GbmClassifier::new(vec![stump1, stump2, stump3], 0.1, 0.0, 2).unwrap();

    // Structural checks
    v.check("GBM: n_estimators = 3", gbm.n_estimators() as f64, 3.0, 0.0);
    v.check(
        "GBM: lr = 0.1",
        gbm.learning_rate(),
        0.1,
        tolerances::ML_PREDICTION,
    );

    // [3.0, 2.0]: s1=-1.0, s2=-0.8, s3=-0.5 → score=0.1*(-2.3)=-0.23 → sigmoid<0.5 → class 0
    let pred_neg = gbm.predict_proba(&[3.0, 2.0]);
    v.check("GBM: [3,2] → class 0", pred_neg.class as f64, 0.0, 0.0);
    let expected_score_neg = 0.1 * (-1.0 + -0.8 + -0.5);
    v.check(
        "GBM: [3,2] raw score",
        pred_neg.raw_score,
        expected_score_neg,
        tolerances::ML_PREDICTION,
    );
    v.check(
        "GBM: [3,2] prob < 0.5",
        f64::from(u8::from(pred_neg.probability < 0.5)),
        1.0,
        0.0,
    );

    // [8.0, 5.0]: s1=1.0, s2=0.8, s3=1.5 → score=0.1*3.3=0.33 → sigmoid>0.5 → class 1
    let pred_pos = gbm.predict_proba(&[8.0, 5.0]);
    v.check("GBM: [8,5] → class 1", pred_pos.class as f64, 1.0, 0.0);
    let expected_score_pos = 0.1 * (1.0 + 0.8 + 1.5);
    v.check(
        "GBM: [8,5] raw score",
        pred_pos.raw_score,
        expected_score_pos,
        tolerances::ML_PREDICTION,
    );
    v.check(
        "GBM: [8,5] prob > 0.5",
        f64::from(u8::from(pred_pos.probability > 0.5)),
        1.0,
        0.0,
    );

    // Sigmoid correctness check
    let expected_prob = 1.0 / (1.0 + (-expected_score_pos).exp());
    v.check(
        "GBM: sigmoid(0.33) correct",
        pred_pos.probability,
        expected_prob,
        tolerances::ML_PREDICTION,
    );

    // Batch prediction
    let batch_preds = gbm.predict_batch(&[vec![3.0, 2.0], vec![8.0, 5.0]]);
    v.check("GBM: batch[0] = 0", batch_preds[0] as f64, 0.0, 0.0);
    v.check("GBM: batch[1] = 1", batch_preds[1] as f64, 1.0, 0.0);

    // Initial bias check
    let gbm_biased = GbmClassifier::new(
        vec![GbmTree::from_arrays(&[-2], &[0.0], &[-1], &[-1], &[0.0]).unwrap()],
        0.1,
        5.0,
        2,
    )
    .unwrap();
    let biased_pred = gbm_biased.predict_proba(&[0.0]);
    v.check(
        "GBM: biased initial → prob > 0.99",
        f64::from(u8::from(biased_pred.probability > 0.99)),
        1.0,
        0.0,
    );

    // Multi-class GBM
    v.section("═══ Domain 25b: GBM Multi-Class ═══");

    let mc_tree0 = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 2.0, -1.0],
    )
    .unwrap();
    let mc_tree1 = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -1.0, 2.0],
    )
    .unwrap();
    let mc_tree2 = GbmTree::from_arrays(
        &[1, -2, -2],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 2.0, -1.0],
    )
    .unwrap();

    let mgbm = GbmMultiClassifier::new(
        vec![vec![mc_tree0], vec![mc_tree1], vec![mc_tree2]],
        1.0,
        vec![0.0, 0.0, 0.0],
        2,
    )
    .unwrap();

    v.check("GBM-MC: n_classes = 3", mgbm.n_classes() as f64, 3.0, 0.0);

    // [0.3, 0.3]: class0 gets +2, class1 gets -1, class2 gets +2 → tie 0 vs 2, softmax picks 0 or 2
    let mc_pred = mgbm.predict_proba(&[0.3, 0.3]);
    v.check(
        "GBM-MC: probabilities sum to 1",
        mc_pred.probabilities.iter().sum::<f64>(),
        1.0,
        tolerances::ML_PREDICTION,
    );
    v.check(
        "GBM-MC: 3 probabilities",
        mc_pred.probabilities.len() as f64,
        3.0,
        0.0,
    );

    // [0.7, 0.3]: class0→-1, class1→+2, class2→+2 → softmax: class 1 or 2 wins
    let mc_pred2 = mgbm.predict_proba(&[0.7, 0.3]);
    v.check(
        "GBM-MC: [0.7,0.3] not class 0",
        f64::from(u8::from(mc_pred2.class != 0)),
        1.0,
        0.0,
    );

    let gbm_us = t0.elapsed().as_micros() as f64;
    timings.push(("GBM (3 stumps + multi-class)", gbm_us));

    // ════════════════════════════════════════════════════════════════
    //  Timing Summary
    // ════════════════════════════════════════════════════════════════
    v.section("═══ BarraCUDA CPU v5 Timing Summary ═══");
    println!();
    println!("  {:<40} {:>12}", "Domain", "Time (µs)");
    println!("  {}", "─".repeat(55));
    let mut total = 0.0;
    for (name, us) in &timings {
        println!("  {name:<40} {us:>12.0}");
        total += us;
    }
    println!("  {}", "─".repeat(55));
    println!("  {:<40} {total:>12.0}", "TOTAL");
    println!();

    v.finish();
}
