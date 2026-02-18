// SPDX-License-Identifier: AGPL-3.0-or-later
//! UniFrac distance — phylogeny-weighted beta diversity.
//!
//! Implements unweighted and weighted UniFrac metrics (Lozupone & Knight 2005,
//! Lozupone et al. 2007) for comparing microbial communities using phylogenetic
//! information.
//!
//! # Algorithm
//!
//! **Unweighted UniFrac**: Fraction of total branch length unique to either
//! community. Captures presence/absence differences weighted by evolutionary
//! distance.
//!
//! **Weighted UniFrac**: Branch lengths weighted by relative abundance
//! differences between communities. Captures quantitative composition changes.
//!
//! # Tree representation
//!
//! Uses a simple array-based tree (Newick-parsed) where each node stores
//! parent index, branch length, and optional leaf label (ASV/OTU ID).
//!
//! # References
//!
//! - Lozupone & Knight. "UniFrac: a new phylogenetic method for comparing
//!   microbial communities." Applied and Environmental Microbiology 71,
//!   8228–8235 (2005).
//! - Lozupone et al. "Quantitative and qualitative β diversity measures lead
//!   to different insights into factors that structure microbial communities."
//!   Applied and Environmental Microbiology 73, 1576–1585 (2007).

use std::collections::HashMap;

/// A node in the phylogenetic tree (array-based representation).
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Index of the parent node (root has parent = itself).
    pub parent: usize,
    /// Branch length from this node to its parent.
    pub branch_length: f64,
    /// Leaf label (ASV/OTU ID), or empty for internal nodes.
    pub label: String,
    /// Child indices.
    pub children: Vec<usize>,
}

/// A phylogenetic tree in array representation.
#[derive(Debug, Clone)]
pub struct PhyloTree {
    pub nodes: Vec<TreeNode>,
    pub root: usize,
    leaf_index: HashMap<String, usize>,
}

impl PhyloTree {
    /// Parse a Newick-format tree string.
    ///
    /// Handles the subset of Newick used by phylogenetic tools:
    /// `((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);`
    pub fn from_newick(newick: &str) -> Self {
        let trimmed = newick.trim().trim_end_matches(';');
        let mut nodes = Vec::new();
        let mut stack: Vec<usize> = Vec::new();

        // Create root node
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
                    // Start a new internal node
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
                    // Close the current internal node
                    let closed = stack.pop().unwrap_or(root);
                    i += 1;
                    // After ')' there may be label:length for the closed node
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
                    // Leaf node: label:length
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

        PhyloTree {
            nodes,
            root,
            leaf_index,
        }
    }

    /// Get the index of a leaf by its label.
    pub fn leaf_idx(&self, label: &str) -> Option<usize> {
        self.leaf_index.get(label).copied()
    }

    /// Total branch length of the tree.
    pub fn total_branch_length(&self) -> f64 {
        self.nodes.iter().map(|n| n.branch_length).sum()
    }

    /// Number of leaves.
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

/// An abundance table: sample_id -> (leaf_label -> count).
pub type AbundanceTable = HashMap<String, HashMap<String, f64>>;

/// Compute the unweighted UniFrac distance between two samples.
///
/// UniFrac_u = (unique branch length) / (total observed branch length)
///
/// "Unique" means branches leading to leaves present in only one sample.
#[allow(clippy::cast_precision_loss)]
pub fn unweighted_unifrac(
    tree: &PhyloTree,
    sample_a: &HashMap<String, f64>,
    sample_b: &HashMap<String, f64>,
) -> f64 {
    // For each node, determine if descendants include leaves from sample A, B, or both
    let n = tree.nodes.len();
    let mut has_a = vec![false; n];
    let mut has_b = vec![false; n];

    // Mark leaves
    for (label, idx) in &tree.leaf_index {
        let in_a = sample_a.get(label).is_some_and(|&v| v > 0.0);
        let in_b = sample_b.get(label).is_some_and(|&v| v > 0.0);
        has_a[*idx] = in_a;
        has_b[*idx] = in_b;
    }

    // Propagate up from leaves to root (nodes are in insertion order, so
    // processing in reverse gives a bottom-up traversal)
    for i in (0..n).rev() {
        for &child in &tree.nodes[i].children.clone() {
            if has_a[child] {
                has_a[i] = true;
            }
            if has_b[child] {
                has_b[i] = true;
            }
        }
    }

    let mut unique_length = 0.0_f64;
    let mut total_length = 0.0_f64;

    for i in 0..n {
        if i == tree.root {
            continue;
        }
        let bl = tree.nodes[i].branch_length;
        if has_a[i] || has_b[i] {
            total_length += bl;
            if has_a[i] != has_b[i] {
                unique_length += bl;
            }
        }
    }

    if total_length > 0.0 {
        unique_length / total_length
    } else {
        0.0
    }
}

/// Compute the weighted UniFrac distance between two samples.
///
/// UniFrac_w = Σ_branches |p_A - p_B| × branch_length / Σ_branches max(p_A, p_B) × branch_length
///
/// where p_A, p_B are the proportional abundances of descendants in each sample.
#[allow(clippy::cast_precision_loss)]
pub fn weighted_unifrac(
    tree: &PhyloTree,
    sample_a: &HashMap<String, f64>,
    sample_b: &HashMap<String, f64>,
) -> f64 {
    let total_a: f64 = sample_a.values().sum();
    let total_b: f64 = sample_b.values().sum();

    if total_a == 0.0 || total_b == 0.0 {
        return if total_a == 0.0 && total_b == 0.0 {
            0.0
        } else {
            1.0
        };
    }

    let n = tree.nodes.len();
    let mut prop_a = vec![0.0_f64; n];
    let mut prop_b = vec![0.0_f64; n];

    // Set leaf proportions
    for (label, &idx) in &tree.leaf_index {
        prop_a[idx] = sample_a.get(label).copied().unwrap_or(0.0) / total_a;
        prop_b[idx] = sample_b.get(label).copied().unwrap_or(0.0) / total_b;
    }

    // Propagate up
    for i in (0..n).rev() {
        for &child in &tree.nodes[i].children.clone() {
            prop_a[i] += prop_a[child];
            prop_b[i] += prop_b[child];
        }
    }

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;

    for i in 0..n {
        if i == tree.root {
            continue;
        }
        let bl = tree.nodes[i].branch_length;
        let diff = (prop_a[i] - prop_b[i]).abs();
        numerator += bl * diff;
        denominator += bl * prop_a[i].max(prop_b[i]);
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute a pairwise UniFrac distance matrix for multiple samples.
pub fn unifrac_distance_matrix(
    tree: &PhyloTree,
    samples: &AbundanceTable,
    weighted: bool,
) -> (Vec<String>, Vec<Vec<f64>>) {
    let sample_ids: Vec<String> = samples.keys().cloned().collect();
    let n = sample_ids.len();
    let mut matrix = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let sa = &samples[&sample_ids[i]];
            let sb = &samples[&sample_ids[j]];
            let dist = if weighted {
                weighted_unifrac(tree, sa, sb)
            } else {
                unweighted_unifrac(tree, sa, sb)
            };
            matrix[i][j] = dist;
            matrix[j][i] = dist;
        }
    }

    (sample_ids, matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> PhyloTree {
        // ((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6)
        PhyloTree::from_newick("((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6)")
    }

    #[test]
    fn newick_parsing() {
        let tree = simple_tree();
        assert_eq!(tree.n_leaves(), 4);
        assert!(tree.leaf_idx("A").is_some());
        assert!(tree.leaf_idx("B").is_some());
        assert!(tree.leaf_idx("C").is_some());
        assert!(tree.leaf_idx("D").is_some());
        assert!(tree.total_branch_length() > 0.0);
    }

    #[test]
    fn identical_samples_distance_zero() {
        let tree = simple_tree();
        let mut sample: HashMap<String, f64> = HashMap::new();
        sample.insert("A".to_string(), 10.0);
        sample.insert("B".to_string(), 20.0);

        let d = unweighted_unifrac(&tree, &sample, &sample);
        assert!((d - 0.0).abs() < 1e-10, "identical samples should have distance 0, got {d}");

        let d = weighted_unifrac(&tree, &sample, &sample);
        assert!((d - 0.0).abs() < 1e-10, "identical samples should have distance 0, got {d}");
    }

    #[test]
    fn disjoint_samples_high_distance() {
        let tree = simple_tree();
        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 10.0);

        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("C".to_string(), 10.0);

        let d = unweighted_unifrac(&tree, &sa, &sb);
        assert!(d > 0.5, "disjoint samples should have high distance, got {d}");
        assert!(d <= 1.0);
    }

    #[test]
    fn weighted_unifrac_abundance_sensitive() {
        let tree = simple_tree();

        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 100.0);
        sa.insert("B".to_string(), 1.0);

        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("A".to_string(), 1.0);
        sb.insert("B".to_string(), 100.0);

        let d = weighted_unifrac(&tree, &sa, &sb);
        assert!(d > 0.0, "different abundance should give non-zero distance");
        assert!(d <= 1.0);
    }

    #[test]
    fn empty_samples() {
        let tree = simple_tree();
        let empty: HashMap<String, f64> = HashMap::new();
        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 10.0);

        // Both empty → 0
        assert!((unweighted_unifrac(&tree, &empty, &empty) - 0.0).abs() < 1e-10);

        // One empty → maximum distance
        let d = weighted_unifrac(&tree, &sa, &empty);
        assert!((d - 1.0).abs() < 1e-10, "one empty should give distance 1.0, got {d}");
    }

    #[test]
    fn distance_matrix_symmetric() {
        let tree = simple_tree();
        let mut samples: AbundanceTable = HashMap::new();

        let mut s1: HashMap<String, f64> = HashMap::new();
        s1.insert("A".to_string(), 10.0);
        s1.insert("B".to_string(), 5.0);
        samples.insert("sample1".to_string(), s1);

        let mut s2: HashMap<String, f64> = HashMap::new();
        s2.insert("C".to_string(), 10.0);
        s2.insert("D".to_string(), 5.0);
        samples.insert("sample2".to_string(), s2);

        let (ids, matrix) = unifrac_distance_matrix(&tree, &samples, false);
        assert_eq!(ids.len(), 2);
        assert_eq!(matrix[0][0], 0.0);
        assert_eq!(matrix[1][1], 0.0);
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-10);
    }

    #[test]
    fn unweighted_unifrac_partial_overlap() {
        let tree = simple_tree();

        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 10.0);
        sa.insert("B".to_string(), 10.0);

        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("B".to_string(), 10.0);
        sb.insert("C".to_string(), 10.0);

        let d = unweighted_unifrac(&tree, &sa, &sb);
        assert!(d > 0.0, "partial overlap should give non-zero distance");
        assert!(d < 1.0, "partial overlap should give less than max distance");
    }

    #[test]
    fn star_tree() {
        // Simple star tree: all leaves directly off root
        let tree = PhyloTree::from_newick("(A:1,B:1,C:1)");
        assert_eq!(tree.n_leaves(), 3);

        let mut sa: HashMap<String, f64> = HashMap::new();
        sa.insert("A".to_string(), 1.0);
        let mut sb: HashMap<String, f64> = HashMap::new();
        sb.insert("B".to_string(), 1.0);

        let d = unweighted_unifrac(&tree, &sa, &sb);
        // All unique length is 2 out of total 2 → distance should be 1.0
        assert!(
            (d - 1.0).abs() < 1e-10,
            "completely disjoint on star tree should be 1.0, got {d}"
        );
    }
}
