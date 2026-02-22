// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phylogenetic tree types and Newick parser.

use std::collections::HashMap;

/// A node in the phylogenetic tree (array-based representation).
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Index of the `parent` node (root has parent = itself).
    pub parent: usize,
    /// `branch_length` from this node to its parent.
    pub branch_length: f64,
    /// `label` (ASV/OTU ID), or empty for internal nodes.
    pub label: String,
    /// Child indices.
    pub children: Vec<usize>,
}

/// A phylogenetic tree in array representation.
#[derive(Debug, Clone)]
pub struct PhyloTree {
    /// All nodes in the tree (leaves and internal).
    pub nodes: Vec<TreeNode>,
    /// Index of the root node.
    pub root: usize,
    pub(crate) leaf_index: HashMap<String, usize>,
}

impl PhyloTree {
    /// Parse a Newick-format tree string.
    ///
    /// Handles the subset of Newick used by phylogenetic tools:
    /// `((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);`
    #[must_use]
    pub fn from_newick(newick: &str) -> Self {
        let trimmed = newick.trim().trim_end_matches(';');
        let mut nodes = Vec::new();
        let mut stack: Vec<usize> = Vec::new();

        nodes.push(TreeNode {
            parent: 0,
            branch_length: 0.0,
            label: String::new(),
            children: Vec::new(),
        });
        let root = 0;
        stack.push(root);

        let chars: Vec<char> = trimmed.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                '(' => {
                    let parent = *stack.last().unwrap_or(&root);
                    let new_idx = nodes.len();
                    nodes.push(TreeNode {
                        parent,
                        branch_length: 0.0,
                        label: String::new(),
                        children: Vec::new(),
                    });
                    nodes[parent].children.push(new_idx);
                    stack.push(new_idx);
                    i += 1;
                }
                ')' => {
                    let closed = stack.pop().unwrap_or(root);
                    i += 1;
                    let (label, bl, consumed) = parse_label_length(&chars[i..]);
                    if consumed > 0 {
                        nodes[closed].label = label;
                        nodes[closed].branch_length = bl;
                        i += consumed;
                    }
                }
                ',' => {
                    i += 1;
                }
                _ => {
                    let (label, bl, consumed) = parse_label_length(&chars[i..]);
                    if consumed > 0 {
                        let parent = *stack.last().unwrap_or(&root);
                        let new_idx = nodes.len();
                        nodes.push(TreeNode {
                            parent,
                            branch_length: bl,
                            label,
                            children: Vec::new(),
                        });
                        nodes[parent].children.push(new_idx);
                        i += consumed;
                    } else {
                        i += 1;
                    }
                }
            }
        }

        let mut leaf_index = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            if node.children.is_empty() && !node.label.is_empty() {
                leaf_index.insert(node.label.clone(), idx);
            }
        }

        Self {
            nodes,
            root,
            leaf_index,
        }
    }

    /// Get the index of a leaf by its `label`.
    #[must_use]
    pub fn leaf_idx(&self, label: &str) -> Option<usize> {
        self.leaf_index.get(label).copied()
    }

    /// Total branch length of the tree.
    #[must_use]
    pub fn total_branch_length(&self) -> f64 {
        self.nodes.iter().map(|n| n.branch_length).sum()
    }

    /// Number of leaves.
    #[must_use]
    pub fn n_leaves(&self) -> usize {
        self.leaf_index.len()
    }
}

fn parse_label_length(chars: &[char]) -> (String, f64, usize) {
    let mut label = String::new();
    let mut bl_str = String::new();
    let mut i = 0;
    let mut in_length = false;

    while i < chars.len() {
        match chars[i] {
            '(' | ')' | ',' | ';' => break,
            ':' => {
                in_length = true;
                i += 1;
            }
            c => {
                if in_length {
                    bl_str.push(c);
                } else {
                    label.push(c);
                }
                i += 1;
            }
        }
    }

    let bl = bl_str.parse::<f64>().unwrap_or(0.0);
    (label, bl, i)
}
