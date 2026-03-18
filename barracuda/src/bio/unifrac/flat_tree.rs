// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-compatible flat tree representation (CSR layout).

use std::collections::HashMap;

use super::tree::{PhyloTree, TreeNode};
use crate::cast;

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
    ///
    /// Clones leaf labels from the tree. Use [`into_flat_tree`](Self::into_flat_tree) when
    /// the tree is no longer needed to avoid those clones.
    ///
    /// Tree node counts fit in u32 for any realistic phylogeny.
    #[must_use]
    pub fn to_flat_tree(&self) -> FlatTree {
        let n = self.nodes.len();
        let mut parent = Vec::with_capacity(n);
        let mut branch_length = Vec::with_capacity(n);
        let mut n_children = Vec::with_capacity(n);
        let mut children_offset = Vec::with_capacity(n);
        let mut children_flat = Vec::new();

        for node in &self.nodes {
            parent.push(cast::usize_u32(node.parent));
            branch_length.push(node.branch_length);
            n_children.push(cast::usize_u32(node.children.len()));
            children_offset.push(cast::usize_u32(children_flat.len()));
            for &c in &node.children {
                children_flat.push(cast::usize_u32(c));
            }
        }

        let mut leaf_indices = Vec::new();
        let mut leaf_labels = Vec::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.children.is_empty() && !node.label.is_empty() {
                leaf_indices.push(cast::usize_u32(idx));
                leaf_labels.push(node.label.clone());
            }
        }

        FlatTree {
            parent,
            branch_length,
            n_children,
            children_offset,
            children_flat,
            n_nodes: cast::usize_u32(n),
            root: cast::usize_u32(self.root),
            leaf_indices,
            leaf_labels,
        }
    }

    /// Convert to a GPU-compatible flat tree (CSR layout), consuming `self`.
    ///
    /// Avoids cloning leaf labels by moving them from the tree. Use this when
    /// the tree is no longer needed after conversion.
    ///
    /// Tree node counts fit in u32 for any realistic phylogeny.
    #[must_use]
    pub fn into_flat_tree(self) -> FlatTree {
        let n = self.nodes.len();
        let mut parent = Vec::with_capacity(n);
        let mut branch_length = Vec::with_capacity(n);
        let mut n_children = Vec::with_capacity(n);
        let mut children_offset = Vec::with_capacity(n);
        let mut children_flat = Vec::new();

        for node in &self.nodes {
            parent.push(cast::usize_u32(node.parent));
            branch_length.push(node.branch_length);
            n_children.push(cast::usize_u32(node.children.len()));
            children_offset.push(cast::usize_u32(children_flat.len()));
            for &c in &node.children {
                children_flat.push(cast::usize_u32(c));
            }
        }

        let mut leaf_indices = Vec::new();
        let mut leaf_labels = Vec::new();
        for (idx, node) in self.nodes.into_iter().enumerate() {
            if node.children.is_empty() && !node.label.is_empty() {
                leaf_indices.push(cast::usize_u32(idx));
                leaf_labels.push(node.label);
            }
        }

        FlatTree {
            parent,
            branch_length,
            n_children,
            children_offset,
            children_flat,
            n_nodes: cast::usize_u32(n),
            root: cast::usize_u32(self.root),
            leaf_indices,
            leaf_labels,
        }
    }
}

impl FlatTree {
    /// Reconstruct a [`PhyloTree`] from flat arrays.
    ///
    /// u32→usize is infallible widening on 64-bit; tree indices fit on 32-bit.
    #[must_use]
    pub fn to_phylo_tree(&self) -> PhyloTree {
        let n = cast::u32_usize(self.n_nodes);
        let mut nodes = Vec::with_capacity(n);

        for i in 0..n {
            let nc = cast::u32_usize(self.n_children[i]);
            let off = cast::u32_usize(self.children_offset[i]);
            let children: Vec<usize> = self.children_flat[off..off + nc]
                .iter()
                .map(|&c| cast::u32_usize(c))
                .collect();

            let label = self
                .leaf_indices
                .iter()
                .position(|&li| li == cast::usize_u32(i))
                .map(|pos| self.leaf_labels[pos].clone())
                .unwrap_or_default();

            nodes.push(TreeNode {
                parent: cast::u32_usize(self.parent[i]),
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
            root: cast::u32_usize(self.root),
            leaf_index,
        }
    }
}
