// SPDX-License-Identifier: AGPL-3.0-or-later
//! Felsenstein pruning algorithm for phylogenetic likelihood.
//!
//! Computes the likelihood of observed nucleotide sequences at leaf nodes
//! given a tree topology, branch lengths, and a substitution model. This is
//! the core computational bottleneck in phylogenetics and a prime GPU target.
//!
//! # References
//!
//! - Felsenstein 1981, *J Mol Evol* 17:368-376
//! - Felsenstein 2004, *Inferring Phylogenies* (Sinauer)
//!
//! # GPU Promotion
//!
//! The pruning pass over sites is embarrassingly parallel — each site
//! can be computed independently. barraCuda can dispatch one workgroup
//! per site with shared memory for the transition matrix.

/// Nucleotide states: A=0, C=1, G=2, T=3.
pub const N_STATES: usize = 4;

/// A node in a binary phylogenetic tree.
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// Leaf node with observed nucleotide sequence (as state indices 0-3).
    Leaf {
        /// Name/label.
        name: String,
        /// Observed states at each site (0=A, 1=C, 2=G, 3=T).
        states: Vec<usize>,
    },
    /// Internal node with left child, right child, and branch lengths.
    Internal {
        /// Left child subtree.
        left: Box<Self>,
        /// Right child subtree.
        right: Box<Self>,
        /// Branch length to left child.
        left_branch: f64,
        /// Branch length to right child.
        right_branch: f64,
    },
}

/// Jukes-Cantor transition probability: `P(j|i, t)`.
///
/// `P(same) = 0.25 + 0.75 * exp(-4*mu*t/3)`
/// `P(diff) = 0.25 - 0.25 * exp(-4*mu*t/3)`
#[inline]
#[must_use]
pub fn jc69_prob(from: usize, to: usize, branch_len: f64, mu: f64) -> f64 {
    let e = (-4.0 * mu * branch_len / 3.0).exp();
    if from == to {
        0.25_f64.mul_add(3.0 * e, 0.25)
    } else {
        0.25 * (1.0 - e)
    }
}

/// Compute the 4x4 transition matrix for a given branch length.
#[must_use]
pub fn transition_matrix(branch_len: f64, mu: f64) -> [[f64; N_STATES]; N_STATES] {
    let mut mat = [[0.0_f64; N_STATES]; N_STATES];
    for (i, row) in mat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = jc69_prob(i, j, branch_len, mu);
        }
    }
    mat
}

/// Partial likelihood vector for a single site at an internal node.
///
/// `L_k(s) = [Σ_x P(x|s,t_left) * L_left(x)] * [Σ_y P(y|s,t_right) * L_right(y)]`
fn pruning_one_site(
    left_partial: &[f64; N_STATES],
    right_partial: &[f64; N_STATES],
    trans_left: &[[f64; N_STATES]; N_STATES],
    trans_right: &[[f64; N_STATES]; N_STATES],
) -> [f64; N_STATES] {
    let mut result = [0.0_f64; N_STATES];
    for (s, res) in result.iter_mut().enumerate() {
        let mut left_sum = 0.0_f64;
        let mut right_sum = 0.0_f64;
        for x in 0..N_STATES {
            left_sum = trans_left[s][x].mul_add(left_partial[x], left_sum);
            right_sum = trans_right[s][x].mul_add(right_partial[x], right_sum);
        }
        *res = left_sum * right_sum;
    }
    result
}

/// Compute partial likelihoods for all sites via post-order traversal.
///
/// Returns a Vec of length `n_sites`, each a `[f64; 4]` partial likelihood vector.
fn compute_partials(node: &TreeNode, mu: f64) -> Vec<[f64; N_STATES]> {
    match node {
        TreeNode::Leaf { states, .. } => states
            .iter()
            .map(|&s| {
                let mut partial = [0.0_f64; N_STATES];
                if s < N_STATES {
                    partial[s] = 1.0;
                } else {
                    partial = [0.25; N_STATES];
                }
                partial
            })
            .collect(),
        TreeNode::Internal {
            left,
            right,
            left_branch,
            right_branch,
        } => {
            let left_partials = compute_partials(left, mu);
            let right_partials = compute_partials(right, mu);
            let trans_l = transition_matrix(*left_branch, mu);
            let trans_r = transition_matrix(*right_branch, mu);

            left_partials
                .iter()
                .zip(&right_partials)
                .map(|(lp, rp)| pruning_one_site(lp, rp, &trans_l, &trans_r))
                .collect()
        }
    }
}

/// Compute total log-likelihood of the alignment given the tree (JC69 model).
///
/// Equal base frequencies (0.25 each) assumed.
#[must_use]
pub fn log_likelihood(tree: &TreeNode, mu: f64) -> f64 {
    let partials = compute_partials(tree, mu);
    let pi = 0.25_f64;
    partials
        .iter()
        .map(|partial| {
            let site_lik: f64 = partial.iter().map(|&p| pi * p).sum();
            site_lik.ln()
        })
        .sum()
}

/// Per-site log-likelihoods (for site-pattern analysis).
#[must_use]
pub fn site_log_likelihoods(tree: &TreeNode, mu: f64) -> Vec<f64> {
    let partials = compute_partials(tree, mu);
    let pi = 0.25_f64;
    partials
        .iter()
        .map(|partial| {
            let site_lik: f64 = partial.iter().map(|&p| pi * p).sum();
            site_lik.ln()
        })
        .collect()
}

// ─── GPU-ready flat representation ──────────────────────────────────
//
// The following types linearize the recursive `TreeNode` into flat arrays
// matching WGSL storage buffer layout. This is the data layout that
// barraCuda will dispatch to GPU: one workgroup per site, sequential
// post-order traversal over the flat node array.

/// GPU-ready flat tree layout (post-order node array).
///
/// All arrays are indexed by internal node index (0..`n_internal`).
/// Leaf data is stored in `leaf_states` (column-major: `[site][leaf]`).
/// Internal node `i` has children `left_child[i]` and `right_child[i]`.
/// Negative child indices indicate leaf nodes (encoded as `-(leaf_idx + 1)`).
#[derive(Debug, Clone)]
pub struct FlatTree {
    /// Number of leaf nodes.
    pub n_leaves: usize,
    /// Number of internal nodes.
    pub n_internal: usize,
    /// Number of alignment sites.
    pub n_sites: usize,
    /// Left child for each internal node. Negative = leaf `-(idx+1)`.
    pub left_child: Vec<i32>,
    /// Right child for each internal node. Negative = leaf `-(idx+1)`.
    pub right_child: Vec<i32>,
    /// Left branch length per internal node.
    pub left_branch: Vec<f64>,
    /// Right branch length per internal node.
    pub right_branch: Vec<f64>,
    /// Leaf states, column-major: `leaf_states[site * n_leaves + leaf]`.
    pub leaf_states: Vec<u8>,
    /// Precomputed 4x4 transition matrices per branch.
    /// `trans_left[node * 16 + from * 4 + to]`.
    pub trans_left: Vec<f64>,
    /// `trans_right[node * 16 + from * 4 + to]`.
    pub trans_right: Vec<f64>,
}

impl FlatTree {
    /// Flatten a recursive `TreeNode` into GPU-ready layout.
    #[must_use]
    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub fn from_tree(tree: &TreeNode, mu: f64) -> Self {
        struct Collector {
            leaf_seqs: Vec<Vec<usize>>,
            lc: Vec<i32>,
            rc: Vec<i32>,
            lb: Vec<f64>,
            rb: Vec<f64>,
        }

        fn flatten_node(node: &TreeNode, c: &mut Collector) -> i32 {
            match node {
                TreeNode::Leaf { states, .. } => {
                    let idx = c.leaf_seqs.len();
                    c.leaf_seqs.push(states.clone());
                    -(idx as i32 + 1)
                }
                TreeNode::Internal {
                    left,
                    right,
                    left_branch,
                    right_branch,
                } => {
                    let left_id = flatten_node(left, c);
                    let right_id = flatten_node(right, c);
                    let internal_idx = c.lc.len() as i32;
                    c.lc.push(left_id);
                    c.rc.push(right_id);
                    c.lb.push(*left_branch);
                    c.rb.push(*right_branch);
                    internal_idx
                }
            }
        }

        let mut c = Collector {
            leaf_seqs: Vec::new(),
            lc: Vec::new(),
            rc: Vec::new(),
            lb: Vec::new(),
            rb: Vec::new(),
        };
        flatten_node(tree, &mut c);

        let Collector {
            leaf_seqs,
            lc: internals_left_child,
            rc: internals_right_child,
            lb: internals_left_branch,
            rb: internals_right_branch,
        } = c;

        let n_leaves = leaf_seqs.len();
        let n_internal = internals_left_child.len();
        let n_sites = leaf_seqs.first().map_or(0, Vec::len);

        // Column-major leaf states
        let mut leaf_states = vec![0u8; n_sites * n_leaves];
        for (leaf_idx, states) in leaf_seqs.iter().enumerate() {
            for (site, &s) in states.iter().enumerate() {
                #[expect(clippy::cast_possible_truncation)] // Truncation: s is DNA state 0..4
                {
                    leaf_states[site * n_leaves + leaf_idx] = s as u8;
                }
            }
        }

        // Precompute transition matrices
        let mut trans_left = vec![0.0_f64; n_internal * 16];
        let mut trans_right = vec![0.0_f64; n_internal * 16];
        for i in 0..n_internal {
            let tl = transition_matrix(internals_left_branch[i], mu);
            let tr = transition_matrix(internals_right_branch[i], mu);
            for from in 0..N_STATES {
                for to in 0..N_STATES {
                    trans_left[i * 16 + from * 4 + to] = tl[from][to];
                    trans_right[i * 16 + from * 4 + to] = tr[from][to];
                }
            }
        }

        Self {
            n_leaves,
            n_internal,
            n_sites,
            left_child: internals_left_child,
            right_child: internals_right_child,
            left_branch: internals_left_branch,
            right_branch: internals_right_branch,
            leaf_states,
            trans_left,
            trans_right,
        }
    }

    /// Compute total log-likelihood using flat post-order traversal.
    ///
    /// This mirrors the GPU dispatch pattern: for each site, traverse
    /// the flat node array in post-order, computing partials from
    /// precomputed transition matrices.
    #[must_use]
    pub fn log_likelihood(&self) -> f64 {
        self.site_log_likelihoods().iter().sum()
    }

    /// Per-site log-likelihoods via flat traversal.
    ///
    /// Each site is independent — this is the kernel that maps to one
    /// GPU workgroup per site.
    #[must_use]
    pub fn site_log_likelihoods(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.n_sites);
        // Preallocated partial buffer: [node][state]
        let mut partials = vec![[0.0_f64; N_STATES]; self.n_internal];

        for site in 0..self.n_sites {
            // Post-order: internal nodes are already in post-order from flattening
            for node in 0..self.n_internal {
                let left_partial = self.get_partial(self.left_child[node], site, &partials);
                let right_partial = self.get_partial(self.right_child[node], site, &partials);

                let base = node * 16;
                for (s, partial) in partials[node].iter_mut().enumerate() {
                    let mut l_sum = 0.0_f64;
                    let mut r_sum = 0.0_f64;
                    for x in 0..N_STATES {
                        l_sum = self.trans_left[base + s * 4 + x].mul_add(left_partial[x], l_sum);
                        r_sum = self.trans_right[base + s * 4 + x].mul_add(right_partial[x], r_sum);
                    }
                    *partial = l_sum * r_sum;
                }
            }

            let root = &partials[self.n_internal - 1];
            let site_lik: f64 = root.iter().map(|&p| 0.25 * p).sum();
            result.push(site_lik.ln());
        }
        result
    }

    #[inline]
    #[expect(clippy::cast_sign_loss)] // Sign: leaf_idx from -child-1, child is negative leaf id
    fn get_partial(
        &self,
        child: i32,
        site: usize,
        partials: &[[f64; N_STATES]],
    ) -> [f64; N_STATES] {
        if child < 0 {
            let leaf_idx = (-child - 1) as usize;
            let state = usize::from(self.leaf_states[site * self.n_leaves + leaf_idx]);
            let mut p = [0.0_f64; N_STATES];
            if state < N_STATES {
                p[state] = 1.0;
            } else {
                p = [0.25; N_STATES];
            }
            p
        } else {
            partials[child as usize]
        }
    }
}

/// Encode a DNA string to state indices (A=0, C=1, G=2, T=3, else=4).
#[must_use]
pub fn encode_dna(seq: &str) -> Vec<usize> {
    seq.bytes()
        .map(|b| match b.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 4,
        })
        .collect()
}

#[cfg(test)]
#[path = "felsenstein_tests.rs"]
mod tests;
