// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Random Forest inference engine.
//!
//! Ensemble of [`DecisionTree`]s with majority voting. Each tree is
//! pre-trained in Python (sklearn) and serialized — this module
//! handles inference only (not training).
//!
//! # Design
//!
//! A Random Forest predicts by running each tree independently, then
//! aggregating via majority vote (classification) or mean (regression).
//! This is embarrassingly parallel — each tree is independent.
//!
//! # GPU Strategy
//!
//! RF batch inference maps to GPU dispatch:
//! - One workgroup per (sample, tree) pair
//! - Each thread traverses its tree for one sample
//! - Reduce via voting/averaging
//!
//! This module is a ToadStool absorption candidate: the same array-based
//! tree representation used by `TreeInferenceGpu` can be extended to
//! multi-tree dispatch.

use super::decision_tree::DecisionTree;

/// A Random Forest classifier/regressor.
#[derive(Debug, Clone)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_features: usize,
    n_classes: usize,
}

/// Result of Random Forest prediction with vote details.
#[derive(Debug, Clone)]
pub struct RfPrediction {
    /// Predicted class (majority vote).
    pub class: usize,
    /// Vote count per class.
    pub votes: Vec<usize>,
    /// Confidence (fraction of trees voting for winning class).
    pub confidence: f64,
}

impl RandomForest {
    /// Build a forest from a collection of pre-trained decision trees.
    ///
    /// # Errors
    ///
    /// Returns `Err` if trees have inconsistent feature counts.
    pub fn from_trees(trees: Vec<DecisionTree>, n_classes: usize) -> Result<Self, String> {
        if trees.is_empty() {
            return Err("empty forest".into());
        }
        let n_features = trees[0].n_features();
        if trees.iter().any(|t| t.n_features() != n_features) {
            return Err("inconsistent n_features across trees".into());
        }
        Ok(Self {
            trees,
            n_features,
            n_classes,
        })
    }

    /// Predict a single sample with vote details.
    #[must_use]
    pub fn predict_with_votes(&self, features: &[f64]) -> RfPrediction {
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            if pred < self.n_classes {
                votes[pred] += 1;
            }
        }

        let (class, &max_votes) = votes
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .unwrap_or((0, &0));

        let confidence = if self.trees.is_empty() {
            0.0
        } else {
            max_votes as f64 / self.trees.len() as f64
        };

        RfPrediction {
            class,
            votes,
            confidence,
        }
    }

    /// Predict a single sample (majority vote, returns class only).
    #[must_use]
    pub fn predict(&self, features: &[f64]) -> usize {
        self.predict_with_votes(features).class
    }

    /// Predict multiple samples, returning class labels.
    #[must_use]
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Predict multiple samples with vote details.
    #[must_use]
    pub fn predict_batch_with_votes(&self, samples: &[Vec<f64>]) -> Vec<RfPrediction> {
        samples.iter().map(|s| self.predict_with_votes(s)).collect()
    }

    /// Individual tree predictions for a single sample (useful for debugging).
    #[must_use]
    pub fn tree_predictions(&self, features: &[f64]) -> Vec<usize> {
        self.trees.iter().map(|t| t.predict(features)).collect()
    }

    /// Number of trees in the forest.
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Expected number of features per sample.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of output classes.
    #[must_use]
    pub const fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Access individual tree by index.
    #[must_use]
    pub fn tree_at(&self, index: usize) -> &DecisionTree {
        &self.trees[index]
    }

    /// Average tree depth across the forest.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_depth(&self) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        let total: usize = self.trees.iter().map(DecisionTree::depth).sum();
        total as f64 / self.trees.len() as f64
    }

    /// Total number of nodes across all trees.
    #[must_use]
    pub fn total_nodes(&self) -> usize {
        self.trees.iter().map(DecisionTree::n_nodes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_a() -> DecisionTree {
        // f[0] <= 0.5 → 0, else → 1
        DecisionTree::from_arrays(
            &[0, -2, -2],
            &[0.5, -2.0, -2.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(1)],
            2,
        )
        .unwrap()
    }

    fn tree_b() -> DecisionTree {
        // f[1] <= 0.5 → 0, else → 1
        DecisionTree::from_arrays(
            &[1, -2, -2],
            &[0.5, -2.0, -2.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(1)],
            2,
        )
        .unwrap()
    }

    fn tree_c() -> DecisionTree {
        // Always predicts class 1
        DecisionTree::from_arrays(&[-2], &[-2.0], &[-1], &[-1], &[Some(1)], 2).unwrap()
    }

    #[test]
    fn unanimous_vote() {
        let rf = RandomForest::from_trees(vec![tree_a(), tree_b(), tree_c()], 2).unwrap();
        let pred = rf.predict_with_votes(&[0.7, 0.7]);
        assert_eq!(pred.class, 1);
        assert_eq!(pred.votes, vec![0, 3]);
        assert!((pred.confidence - 1.0).abs() < 1e-10);
    }

    #[test]
    fn majority_vote() {
        let rf = RandomForest::from_trees(vec![tree_a(), tree_b(), tree_c()], 2).unwrap();
        let pred = rf.predict_with_votes(&[0.3, 0.7]);
        assert_eq!(pred.class, 1); // tree_a=0, tree_b=1, tree_c=1 → 1 wins 2-1
        assert_eq!(pred.votes, vec![1, 2]);
    }

    #[test]
    fn batch_predict() {
        let rf = RandomForest::from_trees(vec![tree_a(), tree_b(), tree_c()], 2).unwrap();
        let preds = rf.predict_batch(&[vec![0.3, 0.3], vec![0.7, 0.7]]);
        assert_eq!(preds, vec![0, 1]); // [0.3,0.3]: a=0,b=0,c=1→0 wins; [0.7,0.7]: all→1
    }

    #[test]
    fn forest_metadata() {
        let rf = RandomForest::from_trees(vec![tree_a(), tree_b(), tree_c()], 2).unwrap();
        assert_eq!(rf.n_trees(), 3);
        assert_eq!(rf.n_features(), 2);
        assert_eq!(rf.n_classes(), 2);
        assert_eq!(rf.total_nodes(), 7); // 3 + 3 + 1
    }

    #[test]
    fn tree_predictions_debug() {
        let rf = RandomForest::from_trees(vec![tree_a(), tree_b(), tree_c()], 2).unwrap();
        let preds = rf.tree_predictions(&[0.3, 0.7]);
        assert_eq!(preds, vec![0, 1, 1]); // a=0, b=1, c=1
    }

    #[test]
    fn empty_forest_error() {
        let result = RandomForest::from_trees(vec![], 2);
        assert!(result.is_err());
    }
}
