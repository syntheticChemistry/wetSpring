// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign decision tree inference engine.
//!
//! Loads a trained decision tree (exported from sklearn as JSON) and performs
//! inference in pure Rust. Proves that ML predictions are portable and
//! reproducible without Python runtime dependency.
//!
//! # Design
//!
//! Array-based tree representation with node traversal. The tree is
//! pre-trained in Python and serialized — this module handles inference
//! only (not training), which is the critical path for field deployment.
//!
//! # Usage
//!
//! ```text
//! let tree = DecisionTree::from_json(json_str)?;
//! let prediction = tree.predict(&features);
//! ```

/// A node in the decision tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Feature index to split on (`-2` for leaf nodes).
    pub feature: i32,
    /// Split threshold (features <= threshold go left).
    pub threshold: f64,
    /// Index of left child (`-1` for leaf).
    pub left_child: i32,
    /// Index of right child (`-1` for leaf).
    pub right_child: i32,
    /// Predicted class for leaf nodes.
    pub prediction: Option<usize>,
}

impl TreeNode {
    /// Returns `true` if this node is a leaf (no children).
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.feature < 0
    }
}

/// A decision tree classifier.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    nodes: Vec<TreeNode>,
    n_features: usize,
}

impl DecisionTree {
    /// Build a tree from parallel arrays (mirrors sklearn's tree structure).
    ///
    /// # Errors
    ///
    /// Returns `Err` if node arrays have inconsistent lengths.
    pub fn from_arrays(
        features: &[i32],
        thresholds: &[f64],
        left_children: &[i32],
        right_children: &[i32],
        predictions: &[Option<usize>],
        n_features: usize,
    ) -> Result<Self, String> {
        let n = features.len();
        if thresholds.len() != n
            || left_children.len() != n
            || right_children.len() != n
            || predictions.len() != n
        {
            return Err("inconsistent array lengths".into());
        }

        let nodes: Vec<TreeNode> = (0..n)
            .map(|i| TreeNode {
                feature: features[i],
                threshold: thresholds[i],
                left_child: left_children[i],
                right_child: right_children[i],
                prediction: predictions[i],
            })
            .collect();

        Ok(Self { nodes, n_features })
    }

    /// Classify a single sample.
    ///
    /// Traverses from root to leaf, returning the predicted class.
    /// Panics if the tree structure is invalid (broken child pointers).
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn predict(&self, features: &[f64]) -> usize {
        let mut idx = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf() {
                return node.prediction.unwrap_or(0);
            }
            let feat_val = features.get(node.feature as usize).copied().unwrap_or(0.0);
            idx = if feat_val <= node.threshold {
                node.left_child as usize
            } else {
                node.right_child as usize
            };
        }
    }

    /// Classify multiple samples, returning a vector of predictions.
    #[must_use]
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Number of nodes in the tree.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaf nodes.
    #[must_use]
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Expected number of features.
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.n_features
    }

    /// Access a node by index.
    #[must_use]
    pub fn node_at(&self, index: usize) -> &TreeNode {
        &self.nodes[index]
    }

    /// Tree depth (longest root-to-leaf path).
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        self.node_depth(0)
    }

    #[allow(clippy::cast_sign_loss)]
    fn node_depth(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        if node.is_leaf() {
            return 0;
        }
        let left_depth = self.node_depth(node.left_child as usize);
        let right_depth = self.node_depth(node.right_child as usize);
        1 + left_depth.max(right_depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> DecisionTree {
        // f[0] <= 0.5 → class 0, else → class 1
        DecisionTree::from_arrays(
            &[0, -2, -2],
            &[0.5, -2.0, -2.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(1)],
            1,
        )
        .unwrap()
    }

    #[test]
    fn simple_classify_left() {
        let tree = simple_tree();
        assert_eq!(tree.predict(&[0.3]), 0);
    }

    #[test]
    fn simple_classify_right() {
        let tree = simple_tree();
        assert_eq!(tree.predict(&[0.7]), 1);
    }

    #[test]
    fn simple_classify_boundary() {
        let tree = simple_tree();
        assert_eq!(tree.predict(&[0.5]), 0); // <= threshold goes left
    }

    #[test]
    fn tree_structure() {
        let tree = simple_tree();
        assert_eq!(tree.n_nodes(), 3);
        assert_eq!(tree.n_leaves(), 2);
        assert_eq!(tree.n_features(), 1);
        assert_eq!(tree.depth(), 1);
    }

    fn deeper_tree() -> DecisionTree {
        // f[0] <= 5.0:
        //   f[1] <= 3.0 → class 0
        //   else → class 1
        // else:
        //   f[0] <= 8.0 → class 1
        //   else → class 2
        DecisionTree::from_arrays(
            &[0, 1, -2, -2, 0, -2, -2],
            &[5.0, 3.0, -2.0, -2.0, 8.0, -2.0, -2.0],
            &[1, 2, -1, -1, 5, -1, -1],
            &[4, 3, -1, -1, 6, -1, -1],
            &[None, None, Some(0), Some(1), None, Some(1), Some(2)],
            2,
        )
        .unwrap()
    }

    #[test]
    fn deeper_tree_classify() {
        let tree = deeper_tree();
        assert_eq!(tree.predict(&[3.0, 2.0]), 0); // left-left
        assert_eq!(tree.predict(&[3.0, 4.0]), 1); // left-right
        assert_eq!(tree.predict(&[7.0, 0.0]), 1); // right-left
        assert_eq!(tree.predict(&[9.0, 0.0]), 2); // right-right
    }

    #[test]
    fn deeper_tree_structure() {
        let tree = deeper_tree();
        assert_eq!(tree.n_nodes(), 7);
        assert_eq!(tree.n_leaves(), 4);
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn batch_predict() {
        let tree = simple_tree();
        let samples = vec![vec![0.3], vec![0.7], vec![0.5]];
        let preds = tree.predict_batch(&samples);
        assert_eq!(preds, vec![0, 1, 0]);
    }

    #[test]
    fn single_leaf_tree() {
        let tree = DecisionTree::from_arrays(&[-2], &[-2.0], &[-1], &[-1], &[Some(1)], 1).unwrap();
        assert_eq!(tree.predict(&[42.0]), 1);
        assert_eq!(tree.depth(), 0);
        assert_eq!(tree.n_leaves(), 1);
    }

    #[test]
    fn inconsistent_arrays_error() {
        let result =
            DecisionTree::from_arrays(&[0, -2], &[0.5], &[1, -1], &[2, -1], &[None, Some(0)], 1);
        assert!(result.is_err());
    }
}
