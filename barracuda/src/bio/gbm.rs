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
//! also run in parallel. barraCuda absorption as `GbmBatchInferenceGpu`.

use crate::error;

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
    ) -> error::Result<Self> {
        let n = features.len();
        if thresholds.len() != n
            || left_children.len() != n
            || right_children.len() != n
            || values.len() != n
        {
            return Err(error::Error::InvalidInput(
                "inconsistent array lengths".into(),
            ));
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
    #[expect(clippy::cast_sign_loss)] // Sign: node.feature and child indices from tree structure
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.feature < 0 {
                return node.value;
            }
            let feat_val = features.get(node.feature as usize).map_or(0.0, |x| *x);
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
    ) -> error::Result<Self> {
        if trees.is_empty() {
            return Err(error::Error::InvalidInput("empty GBM".into()));
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
        self.predict_single_proba(features)
    }

    /// Predict class only.
    #[must_use]
    pub fn predict(&self, features: &[f64]) -> usize {
        self.predict_proba(features).class
    }

    /// Batch prediction with probabilities.
    #[must_use]
    pub fn predict_batch_proba(&self, samples: &[Vec<f64>]) -> Vec<GbmPrediction> {
        samples
            .iter()
            .map(|s| self.predict_single_proba(s))
            .collect()
    }

    /// Single-sample prediction with probability.
    #[must_use]
    fn predict_single_proba(&self, features: &[f64]) -> GbmPrediction {
        let mut score = self.initial_prediction;
        for tree in &self.trees {
            score = self.learning_rate.mul_add(tree.predict(features), score);
        }
        let prob = sigmoid(score);
        GbmPrediction {
            class: usize::from(prob >= 0.5),
            probability: prob,
            raw_score: score,
        }
    }

    /// Batch prediction (class only).
    #[must_use]
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Number of boosting rounds (trees).
    #[must_use]
    pub const fn n_estimators(&self) -> usize {
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
    ) -> error::Result<Self> {
        let n_classes = class_trees.len();
        if n_classes < 2 {
            return Err(error::Error::InvalidInput("need at least 2 classes".into()));
        }
        if initial_predictions.len() != n_classes {
            return Err(error::Error::InvalidInput(
                "initial_predictions length must equal n_classes".into(),
            ));
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
                scores[k] = self
                    .learning_rate
                    .mul_add(tree.predict(features), scores[k]);
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
#[path = "gbm_tests.rs"]
mod tests;
