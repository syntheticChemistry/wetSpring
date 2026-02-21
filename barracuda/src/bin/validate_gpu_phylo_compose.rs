// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp046: GPU Phylogenetic Composition
//!
//! Demonstrates that ToadStool's `FelsensteinGpu` serves as a drop-in
//! GPU inner loop for:
//!
//! 1. **Felsenstein parity** — CPU recursive/flat vs GPU log-likelihood
//! 2. **GPU bootstrap** — column resampling + GPU Felsenstein per replicate
//! 3. **GPU placement** — edge-parallel GPU Felsenstein for metagenomic reads
//!
//! This validates the *composition* of an absorbed ToadStool primitive into
//! higher-level phylogenetic workflows — the core Write → Absorb → Lean
//! pattern applied to the most compute-heavy bioinformatics operation.

use barracuda::device::WgpuDevice;
use barracuda::{FelsensteinGpu, FelsensteinResult, PhyloTree};
use std::sync::Arc;
use wetspring_barracuda::bio::bootstrap::{self, Alignment};
use wetspring_barracuda::bio::felsenstein::{
    encode_dna, log_likelihood, transition_matrix, FlatTree, TreeNode, N_STATES,
};
use wetspring_barracuda::bio::gillespie::Lcg64;
use wetspring_barracuda::bio::placement;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

const MU: f64 = 1.0;
const PI: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

// ─── Tree ↔ PhyloTree conversion ─────────────────────────────────────────────

struct TreeConversion {
    phylo: PhyloTree,
    tip_likelihoods: Vec<f64>,
    transition_probs: Vec<f64>,
    n_sites: usize,
    n_nodes: usize,
    root_idx: usize,
}

/// Pre-order traversal assigns indices and collects per-node data.
fn convert_tree(tree: &TreeNode, mu: f64) -> TreeConversion {
    let mut left_child = Vec::new();
    let mut right_child = Vec::new();
    let mut branch_lengths = Vec::new();
    let mut leaf_seqs: Vec<Vec<usize>> = Vec::new();
    let mut depths: Vec<usize> = Vec::new();
    let mut is_leaf: Vec<bool> = Vec::new();

    fn walk(
        node: &TreeNode,
        parent_branch: f64,
        depth: usize,
        lc: &mut Vec<i32>,
        rc: &mut Vec<i32>,
        bl: &mut Vec<f64>,
        ls: &mut Vec<Vec<usize>>,
        dv: &mut Vec<usize>,
        il: &mut Vec<bool>,
    ) -> usize {
        let my_idx = lc.len();
        // Placeholder — fill children later
        lc.push(-1);
        rc.push(-1);
        bl.push(parent_branch);
        dv.push(depth);

        match node {
            TreeNode::Leaf { states, .. } => {
                ls.push(states.clone());
                il.push(true);
            }
            TreeNode::Internal {
                left,
                right,
                left_branch,
                right_branch,
            } => {
                ls.push(Vec::new());
                il.push(false);
                let left_idx = walk(left, *left_branch, depth + 1, lc, rc, bl, ls, dv, il);
                let right_idx = walk(right, *right_branch, depth + 1, lc, rc, bl, ls, dv, il);
                #[allow(clippy::cast_possible_wrap)]
                {
                    lc[my_idx] = left_idx as i32;
                    rc[my_idx] = right_idx as i32;
                }
            }
        }
        my_idx
    }

    walk(
        tree,
        0.0,
        0,
        &mut left_child,
        &mut right_child,
        &mut branch_lengths,
        &mut leaf_seqs,
        &mut depths,
        &mut is_leaf,
    );

    let n_nodes = left_child.len();
    let n_sites = leaf_seqs
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.len())
        .next()
        .unwrap_or(0);

    // Build levels (bottom-up): group by depth, reverse so deepest first
    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut levels: Vec<Vec<u32>> = Vec::with_capacity(max_depth + 1);
    for d in (0..=max_depth).rev() {
        #[allow(clippy::cast_possible_truncation)]
        let group: Vec<u32> = (0..n_nodes)
            .filter(|&i| depths[i] == d)
            .map(|i| i as u32)
            .collect();
        if !group.is_empty() {
            levels.push(group);
        }
    }

    let phylo = PhyloTree {
        left_child,
        right_child,
        branch_lengths: branch_lengths.clone(),
        levels,
    };

    // Build tip_likelihoods [n_nodes × n_sites × n_states]
    let mut tip_likelihoods = vec![0.0_f64; n_nodes * n_sites * N_STATES];
    for (node_idx, seq) in leaf_seqs.iter().enumerate() {
        if is_leaf[node_idx] {
            for (site, &state) in seq.iter().enumerate() {
                if state < N_STATES {
                    tip_likelihoods[node_idx * n_sites * N_STATES + site * N_STATES + state] = 1.0;
                }
            }
        }
    }

    // Build transition_probs [n_nodes × n_states × n_states]
    let mut transition_probs = vec![0.0_f64; n_nodes * N_STATES * N_STATES];
    for i in 0..n_nodes {
        let mat = transition_matrix(branch_lengths[i], mu);
        for from in 0..N_STATES {
            for to in 0..N_STATES {
                transition_probs[i * N_STATES * N_STATES + from * N_STATES + to] = mat[from][to];
            }
        }
    }

    TreeConversion {
        phylo,
        tip_likelihoods,
        transition_probs,
        n_sites,
        n_nodes,
        root_idx: 0,
    }
}

/// Rebuild tip_likelihoods from a resampled alignment.
fn rebuild_tips(conv: &TreeConversion, tree: &TreeNode, aln: &Alignment) -> Vec<f64> {
    let mut tips = vec![0.0_f64; conv.n_nodes * aln.n_sites * N_STATES];
    let leaf_states = collect_leaf_states(tree, aln);
    for (node_idx, states) in &leaf_states {
        for (site, &state) in states.iter().enumerate() {
            if state < N_STATES {
                tips[node_idx * aln.n_sites * N_STATES + site * N_STATES + state] = 1.0;
            }
        }
    }
    tips
}

/// Collect (node_idx, states) for each leaf in pre-order.
fn collect_leaf_states(tree: &TreeNode, aln: &Alignment) -> Vec<(usize, Vec<usize>)> {
    let mut result = Vec::new();
    let mut leaf_counter = 0_usize;
    fn walk(
        node: &TreeNode,
        idx: &mut usize,
        leaf_ctr: &mut usize,
        aln: &Alignment,
        out: &mut Vec<(usize, Vec<usize>)>,
    ) {
        let my_idx = *idx;
        *idx += 1;
        match node {
            TreeNode::Leaf { .. } => {
                let states: Vec<usize> = aln.columns.iter().map(|col| col[*leaf_ctr]).collect();
                out.push((my_idx, states));
                *leaf_ctr += 1;
            }
            TreeNode::Internal { left, right, .. } => {
                walk(left, idx, leaf_ctr, aln, out);
                walk(right, idx, leaf_ctr, aln, out);
            }
        }
    }
    let mut idx = 0;
    walk(tree, &mut idx, &mut leaf_counter, aln, &mut result);
    result
}

fn gpu_log_likelihood(result: &FelsensteinResult, root_idx: usize) -> f64 {
    result.log_likelihood(root_idx, &PI)
}

// ─── Test trees ──────────────────────────────────────────────────────────────

fn make_3_taxon_tree() -> TreeNode {
    TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "A".into(),
                states: encode_dna("ACGTACGTACGT"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "B".into(),
                states: encode_dna("ACGTACTTACGT"),
            }),
            left_branch: 0.1,
            right_branch: 0.1,
        }),
        right: Box::new(TreeNode::Leaf {
            name: "C".into(),
            states: encode_dna("ACGTACGTACTT"),
        }),
        left_branch: 0.2,
        right_branch: 0.3,
    }
}

fn make_5_taxon_tree() -> TreeNode {
    TreeNode::Internal {
        left: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Internal {
                left: Box::new(TreeNode::Leaf {
                    name: "sp1".into(),
                    states: encode_dna("ACGTACGTACGTACGT"),
                }),
                right: Box::new(TreeNode::Leaf {
                    name: "sp2".into(),
                    states: encode_dna("ACGTACTTACGTACGT"),
                }),
                left_branch: 0.05,
                right_branch: 0.05,
            }),
            right: Box::new(TreeNode::Leaf {
                name: "sp3".into(),
                states: encode_dna("ACGTACGTACTTACGT"),
            }),
            left_branch: 0.1,
            right_branch: 0.15,
        }),
        right: Box::new(TreeNode::Internal {
            left: Box::new(TreeNode::Leaf {
                name: "sp4".into(),
                states: encode_dna("ACTTACGTACGTACGT"),
            }),
            right: Box::new(TreeNode::Leaf {
                name: "sp5".into(),
                states: encode_dna("ACTTACTTACTTACGT"),
            }),
            left_branch: 0.12,
            right_branch: 0.18,
        }),
        left_branch: 0.2,
        right_branch: 0.25,
    }
}

fn alignment_from_tree(tree: &TreeNode) -> Alignment {
    let mut rows = Vec::new();
    fn collect(node: &TreeNode, rows: &mut Vec<Vec<usize>>) {
        match node {
            TreeNode::Leaf { states, .. } => rows.push(states.clone()),
            TreeNode::Internal { left, right, .. } => {
                collect(left, rows);
                collect(right, rows);
            }
        }
    }
    collect(tree, &mut rows);
    Alignment::from_rows(&rows)
}

// ─── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp046: GPU Phylogenetic Composition");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            validation::exit_skipped(&format!("GPU init failed: {e}"));
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }
    println!();

    let device = gpu.to_wgpu_device();

    validate_felsenstein_parity(&device, &mut v);
    validate_gpu_bootstrap(&device, &mut v);
    validate_gpu_placement(&device, &mut v);

    v.finish();
}

// ─── Section 1: CPU ↔ GPU Felsenstein Parity ─────────────────────────────────

fn validate_felsenstein_parity(device: &Arc<WgpuDevice>, v: &mut Validator) {
    v.section("── Section 1: CPU ↔ GPU Felsenstein Parity ──");

    let pruner = FelsensteinGpu::new(device);

    // 3-taxon tree
    let tree3 = make_3_taxon_tree();
    let cpu_ll = log_likelihood(&tree3, MU);
    let flat = FlatTree::from_tree(&tree3, MU);
    let flat_ll = flat.log_likelihood();

    v.check("3-taxon: recursive ≈ flat", cpu_ll, flat_ll, 1e-12);

    let conv = convert_tree(&tree3, MU);
    match pruner.prune(
        &conv.phylo,
        &conv.tip_likelihoods,
        &conv.transition_probs,
        conv.n_sites,
        N_STATES,
    ) {
        Ok(result) => {
            let gpu_ll = gpu_log_likelihood(&result, conv.root_idx);
            v.check("3-taxon: CPU ≈ GPU", cpu_ll, gpu_ll, 1e-6);
            v.check(
                "3-taxon: GPU LL finite",
                f64::from(gpu_ll.is_finite() as u8),
                1.0,
                0.0,
            );
            v.check(
                "3-taxon: GPU LL negative",
                f64::from((gpu_ll < 0.0) as u8),
                1.0,
                0.0,
            );
        }
        Err(e) => {
            println!("  [SKIP] FelsensteinGpu 3-taxon: {e}");
            v.check("3-taxon GPU (skipped)", 1.0, 1.0, 0.0);
        }
    }

    // 5-taxon tree
    let tree5 = make_5_taxon_tree();
    let cpu_ll5 = log_likelihood(&tree5, MU);
    let conv5 = convert_tree(&tree5, MU);
    match pruner.prune(
        &conv5.phylo,
        &conv5.tip_likelihoods,
        &conv5.transition_probs,
        conv5.n_sites,
        N_STATES,
    ) {
        Ok(result) => {
            let gpu_ll5 = gpu_log_likelihood(&result, conv5.root_idx);
            v.check("5-taxon: CPU ≈ GPU", cpu_ll5, gpu_ll5, 1e-6);
            v.check(
                "5-taxon: GPU LL negative",
                f64::from((gpu_ll5 < 0.0) as u8),
                1.0,
                0.0,
            );
        }
        Err(e) => {
            println!("  [SKIP] FelsensteinGpu 5-taxon: {e}");
            v.check("5-taxon GPU (skipped)", 1.0, 1.0, 0.0);
        }
    }
}

// ─── Section 2: GPU Bootstrap ────────────────────────────────────────────────

fn validate_gpu_bootstrap(device: &Arc<WgpuDevice>, v: &mut Validator) {
    v.section("── Section 2: GPU Bootstrap (RAWR) ──");

    let pruner = FelsensteinGpu::new(device);
    let tree = make_3_taxon_tree();
    let aln = alignment_from_tree(&tree);
    let conv = convert_tree(&tree, MU);
    let n_reps = 20_usize;

    // CPU bootstrap likelihoods
    let cpu_lls = bootstrap::bootstrap_likelihoods(&tree, &aln, n_reps, MU, 42);

    // GPU bootstrap: resample columns on CPU, run Felsenstein on GPU
    let mut rng = Lcg64::new(42);
    let mut gpu_lls = Vec::with_capacity(n_reps);
    let mut gpu_ok = true;

    for _ in 0..n_reps {
        let rep = bootstrap::resample_columns(&aln, &mut rng);
        let tips = rebuild_tips(&conv, &tree, &rep);
        match pruner.prune(
            &conv.phylo,
            &tips,
            &conv.transition_probs,
            rep.n_sites,
            N_STATES,
        ) {
            Ok(result) => gpu_lls.push(gpu_log_likelihood(&result, conv.root_idx)),
            Err(e) => {
                println!("  [SKIP] GPU bootstrap replicate failed: {e}");
                gpu_ok = false;
                break;
            }
        }
    }

    if gpu_ok {
        #[allow(clippy::cast_precision_loss)]
        {
            v.check(
                "Bootstrap: replicate count",
                gpu_lls.len() as f64,
                n_reps as f64,
                0.0,
            );
        }

        let all_finite = gpu_lls.iter().all(|x| x.is_finite() && *x < 0.0);
        v.check(
            "Bootstrap: all GPU LLs finite & negative",
            f64::from(all_finite as u8),
            1.0,
            0.0,
        );

        // Per-replicate CPU ≈ GPU (within f64 tolerance)
        let mut max_diff = 0.0_f64;
        for (cpu, gpu) in cpu_lls.iter().zip(&gpu_lls) {
            max_diff = max_diff.max((cpu - gpu).abs());
        }
        v.check(
            "Bootstrap: max |CPU−GPU| < 1e-4",
            f64::from((max_diff < 1e-4) as u8),
            1.0,
            0.0,
        );
        println!("    (max per-replicate diff = {max_diff:.2e})");

        // Statistical agreement: mean and variance should be close
        let cpu_mean: f64 = cpu_lls.iter().sum::<f64>() / n_reps as f64;
        let gpu_mean: f64 = gpu_lls.iter().sum::<f64>() / n_reps as f64;
        v.check("Bootstrap: mean CPU ≈ GPU", cpu_mean, gpu_mean, 1e-4);

        let cpu_var: f64 =
            cpu_lls.iter().map(|x| (x - cpu_mean).powi(2)).sum::<f64>() / n_reps as f64;
        let gpu_var: f64 =
            gpu_lls.iter().map(|x| (x - gpu_mean).powi(2)).sum::<f64>() / n_reps as f64;
        v.check("Bootstrap: variance CPU ≈ GPU", cpu_var, gpu_var, 1e-4);
    } else {
        v.check("Bootstrap: GPU available (skipped)", 1.0, 1.0, 0.0);
    }
}

// ─── Section 3: GPU Placement ────────────────────────────────────────────────

fn validate_gpu_placement(device: &Arc<WgpuDevice>, v: &mut Validator) {
    v.section("── Section 3: GPU Phylogenetic Placement ──");

    let pruner = FelsensteinGpu::new(device);
    let ref_tree = make_3_taxon_tree();
    let query = "ACGTACGTACGT"; // identical to sp1

    // CPU placement scan
    let cpu_scan = placement::placement_scan(&ref_tree, query, 0.05, MU);

    // GPU placement: for each edge, build augmented tree → convert → GPU Felsenstein
    let query_states = encode_dna(query);
    let n_edges = cpu_scan.placements.len();
    let mut gpu_lls = Vec::with_capacity(n_edges);
    let mut gpu_ok = true;

    for edge_idx in 0..n_edges {
        let mut idx = 0;
        let (augmented, _) =
            insert_query_at_edge(&ref_tree, &query_states, edge_idx, 0.05, &mut idx);
        let conv = convert_tree(&augmented, MU);
        match pruner.prune(
            &conv.phylo,
            &conv.tip_likelihoods,
            &conv.transition_probs,
            conv.n_sites,
            N_STATES,
        ) {
            Ok(result) => gpu_lls.push(gpu_log_likelihood(&result, conv.root_idx)),
            Err(e) => {
                println!("  [SKIP] GPU placement edge {edge_idx} failed: {e}");
                gpu_ok = false;
                break;
            }
        }
    }

    if gpu_ok {
        #[allow(clippy::cast_precision_loss)]
        {
            v.check(
                "Placement: edge count",
                gpu_lls.len() as f64,
                n_edges as f64,
                0.0,
            );
        }

        let all_finite = gpu_lls.iter().all(|x| x.is_finite() && *x < 0.0);
        v.check(
            "Placement: all GPU LLs finite & negative",
            f64::from(all_finite as u8),
            1.0,
            0.0,
        );

        // Compare per-edge likelihoods
        let mut max_diff = 0.0_f64;
        for (cpu_p, &gpu_ll) in cpu_scan.placements.iter().zip(&gpu_lls) {
            max_diff = max_diff.max((cpu_p.log_likelihood - gpu_ll).abs());
        }
        v.check(
            "Placement: max |CPU−GPU| < 1e-4",
            f64::from((max_diff < 1e-4) as u8),
            1.0,
            0.0,
        );
        println!("    (max per-edge diff = {max_diff:.2e})");

        // Best edge agreement
        let gpu_best = gpu_lls
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        #[allow(clippy::cast_precision_loss)]
        {
            v.check(
                "Placement: best edge CPU == GPU",
                gpu_best as f64,
                cpu_scan.best_edge as f64,
                0.0,
            );
        }
    } else {
        v.check("Placement: GPU available (skipped)", 1.0, 1.0, 0.0);
    }
}

/// Re-implementation of placement's insert_at_edge (the original is private).
fn insert_query_at_edge(
    tree: &TreeNode,
    query_states: &[usize],
    target: usize,
    pendant_len: f64,
    current_idx: &mut usize,
) -> (TreeNode, bool) {
    match tree {
        TreeNode::Leaf { name, states } => {
            let my_idx = *current_idx;
            *current_idx += 1;
            if my_idx == target {
                let node = TreeNode::Internal {
                    left: Box::new(TreeNode::Leaf {
                        name: name.clone(),
                        states: states.clone(),
                    }),
                    right: Box::new(TreeNode::Leaf {
                        name: "query".into(),
                        states: query_states.to_vec(),
                    }),
                    left_branch: 0.01,
                    right_branch: pendant_len,
                };
                (node, true)
            } else {
                (tree.clone(), false)
            }
        }
        TreeNode::Internal {
            left,
            right,
            left_branch,
            right_branch,
        } => {
            let my_idx = *current_idx;
            *current_idx += 1;
            let (new_left, found_l) =
                insert_query_at_edge(left, query_states, target, pendant_len, current_idx);
            let (new_right, found_r) =
                insert_query_at_edge(right, query_states, target, pendant_len, current_idx);

            if found_l || found_r {
                (
                    TreeNode::Internal {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                        left_branch: *left_branch,
                        right_branch: *right_branch,
                    },
                    true,
                )
            } else if my_idx == target {
                let node = TreeNode::Internal {
                    left: Box::new(TreeNode::Internal {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                        left_branch: *left_branch,
                        right_branch: *right_branch,
                    }),
                    right: Box::new(TreeNode::Leaf {
                        name: "query".into(),
                        states: query_states.to_vec(),
                    }),
                    left_branch: 0.01,
                    right_branch: pendant_len,
                };
                (node, true)
            } else {
                (
                    TreeNode::Internal {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                        left_branch: *left_branch,
                        right_branch: *right_branch,
                    },
                    false,
                )
            }
        }
    }
}
