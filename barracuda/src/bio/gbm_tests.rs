// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::tolerances;

fn stump_positive() -> GbmTree {
    // f[0] <= 0.5 → -1.0 (push toward class 0), else → 1.0 (push toward class 1)
    GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.5, -2.0, -2.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -1.0, 1.0],
    )
    .unwrap()
}

fn stump_feature1() -> GbmTree {
    // f[1] <= 0.5 → -0.5, else → 0.5
    GbmTree::from_arrays(
        &[1, -2, -2],
        &[0.5, -2.0, -2.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 0.5],
    )
    .unwrap()
}

#[test]
fn binary_gbm_positive() {
    let gbm = GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
    let pred = gbm.predict_proba(&[0.7, 0.7]);
    // score = 0.0 + 0.1*1.0 + 0.1*0.5 = 0.15
    assert!((pred.raw_score - 0.15).abs() < tolerances::ANALYTICAL_LOOSE);
    assert!(pred.probability > 0.5);
    assert_eq!(pred.class, 1);
}

#[test]
fn binary_gbm_negative() {
    let gbm = GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
    let pred = gbm.predict_proba(&[0.3, 0.3]);
    // score = 0.0 + 0.1*(-1.0) + 0.1*(-0.5) = -0.15
    assert!((pred.raw_score - (-0.15)).abs() < tolerances::ANALYTICAL_LOOSE);
    assert!(pred.probability < 0.5);
    assert_eq!(pred.class, 0);
}

#[test]
fn binary_gbm_batch() {
    let gbm = GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
    let preds = gbm.predict_batch(&[vec![0.3, 0.3], vec![0.7, 0.7]]);
    assert_eq!(preds, vec![0, 1]);
}

#[test]
fn binary_gbm_initial_bias() {
    let gbm = GbmClassifier::new(vec![stump_positive()], 0.1, 2.0, 2).unwrap();
    let pred = gbm.predict_proba(&[0.3]);
    // score = 2.0 + 0.1*(-1.0) = 1.9 → sigmoid(1.9) ≈ 0.87
    assert!((pred.raw_score - 1.9).abs() < tolerances::ANALYTICAL_LOOSE);
    assert_eq!(pred.class, 1); // bias pulls toward 1
}

#[test]
fn multi_class_gbm() {
    // 3-class: each class has one stump
    let class0_tree = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 1.0, -0.5],
    )
    .unwrap();
    let class1_tree = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.6, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 0.5, 1.0],
    )
    .unwrap();
    let class2_tree = GbmTree::from_arrays(
        &[1, -2, -2],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 0.5],
    )
    .unwrap();

    let mgbm = GbmMultiClassifier::new(
        vec![vec![class0_tree], vec![class1_tree], vec![class2_tree]],
        1.0,
        vec![0.0, 0.0, 0.0],
        2,
    )
    .unwrap();

    let pred = mgbm.predict_proba(&[0.2, 0.3]);
    assert_eq!(pred.class, 0); // f[0]=0.2 ≤ 0.3 → class0 gets +1.0
    assert_eq!(pred.probabilities.len(), 3);
    let sum: f64 = pred.probabilities.iter().sum();
    assert!((sum - 1.0).abs() < tolerances::ANALYTICAL_LOOSE);
}

#[test]
fn gbm_metadata() {
    let gbm = GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
    assert_eq!(gbm.n_estimators(), 2);
    assert!((gbm.learning_rate() - 0.1).abs() < tolerances::ANALYTICAL_LOOSE);
    assert_eq!(gbm.n_features(), 2);
}

// ─── Additional edge-case coverage ────────────────────────────────────────

fn single_leaf_stump() -> GbmTree {
    GbmTree::from_arrays(&[-1], &[-1.0], &[-1], &[-1], &[0.5]).unwrap()
}

#[test]
fn gbm_single_node_stump() {
    let gbm = GbmClassifier::new(vec![single_leaf_stump()], 0.1, 0.0, 2).unwrap();
    let pred = gbm.predict_proba(&[0.0, 0.0]);
    assert_eq!(pred.class, 1);
    assert!((pred.raw_score - 0.05).abs() < tolerances::ANALYTICAL_LOOSE);
}

#[test]
fn gbm_negative_target_values() {
    let stump = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.5, -2.0, -2.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -2.0, -3.0],
    )
    .unwrap();
    let gbm = GbmClassifier::new(vec![stump], 0.1, -1.0, 2).unwrap();
    let pred = gbm.predict_proba(&[0.3, 0.3]);
    assert!(pred.raw_score < 0.0);
    assert_eq!(pred.class, 0);
}

#[test]
fn gbm_prediction_outside_training_range() {
    let gbm = GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
    let pred_high = gbm.predict_proba(&[999.0, 999.0]);
    let pred_low = gbm.predict_proba(&[-999.0, -999.0]);
    assert_eq!(pred_high.class, 1);
    assert_eq!(pred_low.class, 0);
}

#[test]
fn gbm_zero_learning_rate() {
    let gbm = GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.0, 0.5, 2).unwrap();
    let pred = gbm.predict_proba(&[0.7, 0.7]);
    assert!((pred.raw_score - 0.5).abs() < tolerances::ANALYTICAL_LOOSE);
}

#[test]
fn gbm_empty_trees_error() {
    assert!(GbmClassifier::new(vec![], 0.1, 0.0, 2).is_err());
}

#[test]
fn gbm_tree_inconsistent_arrays_error() {
    let err = GbmTree::from_arrays(&[0, -1], &[0.5], &[1, -1], &[2, -1], &[0.0, 1.0]);
    assert!(err.is_err());
}

#[test]
fn gbm_multi_class_single_class_error() {
    let tree = GbmTree::from_arrays(&[-1], &[0.0], &[-1], &[-1], &[0.0]).unwrap();
    assert!(GbmMultiClassifier::new(vec![vec![tree]], 0.1, vec![0.0], 1).is_err());
}

#[test]
fn gbm_multi_class_initial_predictions_mismatch_error() {
    let tree = GbmTree::from_arrays(&[-1], &[0.0], &[-1], &[-1], &[0.0]).unwrap();
    assert!(
        GbmMultiClassifier::new(vec![vec![tree.clone()], vec![tree]], 0.1, vec![0.0], 1).is_err()
    );
}

#[test]
fn gbm_predict_feature_missing_uses_zero() {
    let gbm = GbmClassifier::new(vec![stump_positive()], 0.1, 0.0, 2).unwrap();
    let pred = gbm.predict_proba(&[0.3]);
    assert_eq!(pred.class, 0);
}

#[test]
fn gbm_sigmoid_extreme_negative() {
    let stump = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.5, -2.0, -2.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -20.0, 0.0],
    )
    .unwrap();
    let gbm = GbmClassifier::new(vec![stump], 1.0, -5.0, 1).unwrap();
    let pred = gbm.predict_proba(&[0.0]);
    assert!(pred.probability < 1e-6);
}

#[test]
fn gbm_batch_proba_consistency() {
    let gbm = GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
    let batch = gbm.predict_batch_proba(&[vec![0.3, 0.3], vec![0.7, 0.7]]);
    let single0 = gbm.predict_proba(&[0.3, 0.3]);
    let single1 = gbm.predict_proba(&[0.7, 0.7]);
    assert!((batch[0].raw_score - single0.raw_score).abs() < tolerances::MATRIX_EPS);
    assert!((batch[1].raw_score - single1.raw_score).abs() < tolerances::MATRIX_EPS);
}

#[test]
fn gbm_multi_predict_batch() {
    let class0_tree = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, 1.0, -1.0],
    )
    .unwrap();
    let class1_tree = GbmTree::from_arrays(
        &[0, -2, -2],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -1.0, 1.0],
    )
    .unwrap();
    let mgbm = GbmMultiClassifier::new(
        vec![vec![class0_tree], vec![class1_tree]],
        1.0,
        vec![0.0, 0.0],
        2,
    )
    .unwrap();
    let preds = mgbm.predict_batch(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
    assert_eq!(preds.len(), 2);
}
