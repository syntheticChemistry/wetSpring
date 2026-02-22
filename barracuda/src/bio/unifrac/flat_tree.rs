// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-compatible flat tree representation (CSR layout).

use std::collections::HashMap;

use super::tree::{PhyloTree, TreeNode};

/// GPU-compatible flat tree representation (CSR layout).
///
/// All tree topology in contiguous arrays for GPU buffer upload.
/// Children stored in compressed sparse row format: node `i` has children
/// at `children_flat[children_offset[i] .. children_offset[i] + n_children[i]]`.
#[derive(Debug, Clone)]
pub struct FlatTree {
    /// Parent index for each node (root points to itself).
    pub parent: Vec<u32>,
    /// Branch length from node to its parent.
    pub branch_length: Vec<f64>,
    /// Number of children per node.
    pub n_children: Vec<u32>,
    /// Offset into `children_flat` for each node.
    pub children_offset: Vec<u32>,
    /// Flattened children indices.
    pub children_flat: Vec<u32>,
    /// Number of nodes.
    pub n_nodes: u32,
    /// Root index.
    pub root: u32,
    /// Leaf node indices (for sample mapping).
    pub leaf_indices: Vec<u32>,
    /// Leaf labels in the same order as `leaf_indices`.
    pub leaf_labels: Vec<String>,
}

impl PhyloTree {
    /// Convert to a GPU-compatible flat tree (CSR layout).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_flat_tree(&self) -> FlatTree {
        let n = self.nodes.len();
        let mut parent = Vec::with_capacity(n);
        let mut branch_length = Vec::with_capacity(n);
        let mut n_children = Vec::with_capacity(n);
        let mut children_offset = Vec::with_capacity(n);
        let mut children_flat = Vec::new();

        for node in &self.nodes {
            parent.push(node.parent as u32);
            branch_length.push(node.branch_length);
            n_children.push(node.children.len() as u32);
            children_offset.push(children_flat.len() as u32);
            for &c in &node.children {
                children_flat.push(c as u32);
            }
        }

        let mut leaf_indices = Vec::new();
        let mut leaf_labels = Vec::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.children.is_empty() && !node.label.is_empty() {
                leaf_indices.push(idx as u32);
                leaf_labels.push(node.label.clone());
            }
        }

        FlatTree {
            parent,
            branch_length,
            n_children,
            children_offset,
            children_flat,
            n_nodes: n as u32,
            root: self.root as u32,
            leaf_indices,
            leaf_labels,
        }
    }
}

impl FlatTree {
    /// Reconstruct a [`PhyloTree`] from flat arrays.
    #[must_use]
    pub fn to_phylo_tree(&self) -> PhyloTree {
        let n = self.n_nodes as usize;
        let mut nodes = Vec::with_capacity(n);

        for i in 0..n {
            let nc = self.n_children[i] as usize;
            let off = self.children_offset[i] as usize;
            let children: Vec<usize> = self.children_flat[off..off + nc]
                .iter()
                .map(|&c| c as usize)
                .collect();

            let label = self
                .leaf_indices
                .iter()
                .position(|&li| li == i as u32)
                .map(|pos| self.leaf_labels[pos].clone())
                .unwrap_or_default();

            nodes.push(TreeNode {
                parent: self.parent[i] as usize,
                branch_length: self.branch_length[i],
                label,
                children,
            });
        }

        let mut leaf_index = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            if node.children.is_empty() && !node.label.is_empty() {
                leaf_index.insert(node.label.clone(), idx);
            }
        }

        PhyloTree {
            nodes,
            root: self.root as usize,
            leaf_index,
        }
    }
}
