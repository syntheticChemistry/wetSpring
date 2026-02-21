// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Gradient Boosting Machine (GBM) inference engine.
//!
//! Sequential ensemble of regression trees. Each tree predicts a residual
//! correction to the previous cumulative prediction. The final output is
//! the sum of an initial prediction plus all tree contributions scaled
//! by a learning rate.
//!
//! # Design
//!
//! GBM for classification uses the same `DecisionTree` infrastructure
//! but differs from Random Forest:
//! - Trees are evaluated **sequentially** (each corrects the previous)
//! - Trees predict **log-odds residuals** (not class labels)
//! - Final prediction is sigmoid(sum of log-odds)
//!
//! For multi-class, we use one-vs-rest with K sets of trees.
//!
//! # GPU Strategy
//!
//! GBM is inherently sequential across trees (tree N depends on tree N-1
//! cumulative sum). However, within each tree, the batch of samples can
//! be dispatched in parallel. For multi-class, the K class chains can
//! also run in parallel. `ToadStool` absorption as `GbmBatchInferenceGpu`.

/// A regression tree for GBM (predicts f64 residuals, not class labels).
#[derive(Debug, Clone)]
pub struct GbmTree {
    nodes: Vec<GbmNode>,
}

/// Node in a GBM regression tree.
#[derive(Debug, Clone)]
pub struct GbmNode {
    /// Feature index to split on (negative = leaf node).
    pub feature: i32,
    /// Split threshold; samples with `feat_val <= threshold` go left.
    pub threshold: f64,
    /// Left child node index.
    pub left_child: i32,
    /// Right child node index.
    pub right_child: i32,
    /// Leaf value (predicted residual for this node).
    pub value: f64,
}

impl GbmTree {
    /// Build from parallel arrays.
    ///
    /// # Errors
    ///
    /// Returns `Err` if arrays have inconsistent lengths.
    pub fn from_arrays(
        features: &[i32],
        thresholds: &[f64],
        left_children: &[i32],
        right_children: &[i32],
        values: &[f64],
    ) -> Result<Self, String> {
        let n = features.len();
        if thresholds.len() != n
            || left_children.len() != n
            || right_children.len() != n
            || values.len() != n
        {
            return Err("inconsistent array lengths".into());
        }
        let nodes = (0..n)
            .map(|i| GbmNode {
                feature: features[i],
                threshold: thresholds[i],
                left_child: left_children[i],
                right_child: right_children[i],
                value: values[i],
            })
            .collect();
        Ok(Self { nodes })
    }

    /// Predict the residual for a single sample.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.feature < 0 {
                return node.value;
            }
            let feat_val = features.get(node.feature as usize).copied().unwrap_or(0.0);
            idx = if feat_val <= node.threshold {
                node.left_child as usize
            } else {
                node.right_child as usize
            };
        }
    }
}

/// Sigmoid function for converting log-odds to probabilities.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A GBM binary classifier.
#[derive(Debug, Clone)]
pub struct GbmClassifier {
    trees: Vec<GbmTree>,
    learning_rate: f64,
    initial_prediction: f64,
    n_features: usize,
}

/// GBM prediction result with probability.
#[derive(Debug, Clone)]
pub struct GbmPrediction {
    /// Predicted class (0 or 1 for binary).
    pub class: usize,
    /// Probability of positive class.
    pub probability: f64,
    /// Raw log-odds score.
    pub raw_score: f64,
}

impl GbmClassifier {
    /// Build a binary GBM classifier from pre-trained trees.
    ///
    /// `initial_prediction` is the log-odds baseline (e.g., log(p/(1-p)) for
    /// the training set base rate).
    ///
    /// # Errors
    ///
    /// Returns `Err` if trees is empty.
    pub fn new(
        trees: Vec<GbmTree>,
        learning_rate: f64,
        initial_prediction: f64,
        n_features: usize,
    ) -> Result<Self, String> {
        if trees.is_empty() {
            return Err("empty GBM".into());
        }
        Ok(Self {
            trees,
            learning_rate,
            initial_prediction,
            n_features,
        })
    }

    /// Predict with probability and raw score.
    #[must_use]
    pub fn predict_proba(&self, features: &[f64]) -> GbmPrediction {
        let mut score = self.initial_prediction;
        for tree in &self.trees {
            score += self.learning_rate * tree.predict(features);
        }
        let prob = sigmoid(score);
        GbmPrediction {
            class: usize::from(prob >= 0.5),
            probability: prob,
            raw_score: score,
        }
    }

    /// Predict class only.
    #[must_use]
    pub fn predict(&self, features: &[f64]) -> usize {
        self.predict_proba(features).class
    }

    /// Batch prediction with probabilities.
    #[must_use]
    pub fn predict_batch_proba(&self, samples: &[Vec<f64>]) -> Vec<GbmPrediction> {
        samples.iter().map(|s| self.predict_proba(s)).collect()
    }

    /// Batch prediction (class only).
    #[must_use]
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Number of boosting rounds (trees).
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Learning rate.
    #[must_use]
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Number of features.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }
}

/// A GBM multi-class classifier (one-vs-rest).
#[derive(Debug, Clone)]
pub struct GbmMultiClassifier {
    class_trees: Vec<Vec<GbmTree>>,
    learning_rate: f64,
    initial_predictions: Vec<f64>,
    n_features: usize,
    n_classes: usize,
}

/// Multi-class GBM prediction.
#[derive(Debug, Clone)]
pub struct GbmMultiPrediction {
    /// Predicted class index (argmax of `probabilities`).
    pub class: usize,
    /// Per-class probabilities (softmax over `raw_scores`).
    pub probabilities: Vec<f64>,
    /// Raw log-odds scores before softmax.
    pub raw_scores: Vec<f64>,
}

impl GbmMultiClassifier {
    /// Build a multi-class GBM.
    ///
    /// `class_trees[k]` contains the regression trees for class k.
    ///
    /// # Errors
    ///
    /// Returns `Err` if configuration is invalid.
    pub fn new(
        class_trees: Vec<Vec<GbmTree>>,
        learning_rate: f64,
        initial_predictions: Vec<f64>,
        n_features: usize,
    ) -> Result<Self, String> {
        let n_classes = class_trees.len();
        if n_classes < 2 {
            return Err("need at least 2 classes".into());
        }
        if initial_predictions.len() != n_classes {
            return Err("initial_predictions length must equal n_classes".into());
        }
        Ok(Self {
            class_trees,
            learning_rate,
            initial_predictions,
            n_features,
            n_classes,
        })
    }

    /// Predict with per-class probabilities (softmax).
    ///
    /// Returns the predicted class, per-class probabilities, and raw scores.
    /// Falls back to class 0 if the probability vector is empty.
    #[must_use]
    pub fn predict_proba(&self, features: &[f64]) -> GbmMultiPrediction {
        let mut scores = self.initial_predictions.clone();
        for (k, trees) in self.class_trees.iter().enumerate() {
            for tree in trees {
                scores[k] += self.learning_rate * tree.predict(features);
            }
        }

        // Softmax
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let probabilities: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        let (class, _) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, &0.0));

        GbmMultiPrediction {
            class,
            probabilities,
            raw_scores: scores,
        }
    }

    /// Predict class only.
    #[must_use]
    pub fn predict(&self, features: &[f64]) -> usize {
        self.predict_proba(features).class
    }

    /// Batch prediction.
    #[must_use]
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Number of classes.
    #[must_use]
    pub const fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Number of features.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

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
        let gbm =
            GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
        let pred = gbm.predict_proba(&[0.7, 0.7]);
        // score = 0.0 + 0.1*1.0 + 0.1*0.5 = 0.15
        assert!((pred.raw_score - 0.15).abs() < 1e-10);
        assert!(pred.probability > 0.5);
        assert_eq!(pred.class, 1);
    }

    #[test]
    fn binary_gbm_negative() {
        let gbm =
            GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
        let pred = gbm.predict_proba(&[0.3, 0.3]);
        // score = 0.0 + 0.1*(-1.0) + 0.1*(-0.5) = -0.15
        assert!((pred.raw_score - (-0.15)).abs() < 1e-10);
        assert!(pred.probability < 0.5);
        assert_eq!(pred.class, 0);
    }

    #[test]
    fn binary_gbm_batch() {
        let gbm =
            GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
        let preds = gbm.predict_batch(&[vec![0.3, 0.3], vec![0.7, 0.7]]);
        assert_eq!(preds, vec![0, 1]);
    }

    #[test]
    fn binary_gbm_initial_bias() {
        let gbm = GbmClassifier::new(vec![stump_positive()], 0.1, 2.0, 2).unwrap();
        let pred = gbm.predict_proba(&[0.3]);
        // score = 2.0 + 0.1*(-1.0) = 1.9 → sigmoid(1.9) ≈ 0.87
        assert!((pred.raw_score - 1.9).abs() < 1e-10);
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
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gbm_metadata() {
        let gbm =
            GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
        assert_eq!(gbm.n_estimators(), 2);
        assert!((gbm.learning_rate() - 0.1).abs() < 1e-10);
        assert_eq!(gbm.n_features(), 2);
    }
}
