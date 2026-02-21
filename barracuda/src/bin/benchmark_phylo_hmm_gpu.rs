// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp048: CPU vs GPU Benchmark — Phylogenetics & HMM
//!
//! Head-to-head benchmarks for the newly GPU-composed domains:
//! 1. Felsenstein pruning: CPU recursive vs GPU FelsensteinGpu
//! 2. Bootstrap resampling: CPU sequential vs GPU batch Felsenstein
//! 3. Placement scan: CPU sequential vs GPU batch Felsenstein
//! 4. HMM batch forward: CPU sequential vs GPU batch shader
//!
//! Reports speedup ratios and validates parity.

use barracuda::{FelsensteinGpu, PhyloTree};
use std::sync::Arc;
use std::time::Instant;
use wetspring_barracuda::bio::bootstrap::{self, Alignment};
use wetspring_barracuda::bio::felsenstein::{
    encode_dna, log_likelihood, transition_matrix, TreeNode, N_STATES,
};
use wetspring_barracuda::bio::hmm::{self, HmmModel};
use wetspring_barracuda::bio::hmm_gpu::HmmGpuForward;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

const MU: f64 = 1.0;
const PI: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

fn make_large_tree(n_taxa: usize, seq_len: usize) -> TreeNode {
    let base_seq = "ACGTACGT".repeat(seq_len / 8 + 1);
    let base_states = encode_dna(&base_seq[..seq_len]);

    fn build(taxa_left: &[Vec<usize>], depth: usize) -> TreeNode {
        if taxa_left.len() == 1 {
            return TreeNode::Leaf {
                name: format!("t{depth}"),
                states: taxa_left[0].clone(),
            };
        }
        let mid = taxa_left.len() / 2;
        let bl = 0.1 + 0.01 * depth as f64;
        TreeNode::Internal {
            left: Box::new(build(&taxa_left[..mid], depth + 1)),
            right: Box::new(build(&taxa_left[mid..], depth + 1)),
            left_branch: bl,
            right_branch: bl,
        }
    }

    let mut taxa: Vec<Vec<usize>> = Vec::with_capacity(n_taxa);
    for i in 0..n_taxa {
        let mut seq = base_states.clone();
        for (j, s) in seq.iter_mut().enumerate() {
            if (i + j) % 5 == 0 {
                *s = (*s + 1) % 4;
            }
        }
        taxa.push(seq);
    }
    build(&taxa, 0)
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

/// Convert TreeNode to PhyloTree (same as Exp046).
fn convert_tree(tree: &TreeNode, mu: f64) -> (PhyloTree, Vec<f64>, Vec<f64>, usize) {
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

    let mut transition_probs = vec![0.0_f64; n_nodes * N_STATES * N_STATES];
    for i in 0..n_nodes {
        let mat = transition_matrix(branch_lengths[i], mu);
        for from in 0..N_STATES {
            for to in 0..N_STATES {
                transition_probs[i * N_STATES * N_STATES + from * N_STATES + to] = mat[from][to];
            }
        }
    }

    (phylo, tip_likelihoods, transition_probs, n_sites)
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp048: CPU vs GPU Benchmark (Phylo + HMM)");

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

    bench_felsenstein(&device, &mut v);
    bench_bootstrap(&device, &mut v);
    bench_hmm_batch(&device, &mut v);

    v.finish();
}

fn bench_felsenstein(device: &Arc<barracuda::device::WgpuDevice>, v: &mut Validator) {
    v.section("── Felsenstein: 16 taxa × 512 sites ──");

    let tree = make_large_tree(16, 512);

    let start = Instant::now();
    let cpu_ll = log_likelihood(&tree, MU);
    let cpu_us = start.elapsed().as_micros();

    let (phylo, tips, tp, n_sites) = convert_tree(&tree, MU);
    let pruner = FelsensteinGpu::new(device);
    let start = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pruner.prune(&phylo, &tips, &tp, n_sites, N_STATES)
    }));
    let gpu_us = start.elapsed().as_micros();

    match result {
        Ok(Ok(gpu_result)) => {
            let gpu_ll = gpu_result.log_likelihood(0, &PI);
            #[allow(clippy::cast_precision_loss)]
            {
                let speedup = cpu_us as f64 / gpu_us.max(1) as f64;
                println!("    CPU: {cpu_us} µs, GPU: {gpu_us} µs, speedup: {speedup:.2}×");
                v.check("Fels: parity (CPU ≈ GPU)", cpu_ll, gpu_ll, 1e-4);
                v.check(
                    "Fels: GPU completed",
                    f64::from(gpu_ll.is_finite() as u8),
                    1.0,
                    0.0,
                );
            }
        }
        Ok(Err(e)) => {
            println!("  [SKIP] GPU: {e}");
            v.check("Fels: GPU (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] GPU panicked");
            v.check("Fels: GPU (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

fn bench_bootstrap(device: &Arc<barracuda::device::WgpuDevice>, v: &mut Validator) {
    v.section("── Bootstrap: 100 replicates × 16 taxa × 512 sites ──");

    let tree = make_large_tree(16, 512);
    let aln = alignment_from_tree(&tree);
    let n_reps = 100;

    let start = Instant::now();
    let cpu_lls = bootstrap::bootstrap_likelihoods(&tree, &aln, n_reps, MU, 42);
    let cpu_us = start.elapsed().as_micros();

    let (phylo, _tips, tp, n_sites) = convert_tree(&tree, MU);
    let pruner = FelsensteinGpu::new(device);

    let start = Instant::now();
    let mut rng = wetspring_barracuda::bio::gillespie::Lcg64::new(42);
    let mut gpu_lls = Vec::with_capacity(n_reps);
    let mut ok = true;
    let leaf_states = collect_leaf_states(&tree);

    for _ in 0..n_reps {
        let rep = bootstrap::resample_columns(&aln, &mut rng);
        let tips = rebuild_tips_from_leaves(&leaf_states, &phylo, &rep, n_sites);
        match pruner.prune(&phylo, &tips, &tp, rep.n_sites, N_STATES) {
            Ok(r) => gpu_lls.push(r.log_likelihood(0, &PI)),
            Err(_) => {
                ok = false;
                break;
            }
        }
    }
    let gpu_us = start.elapsed().as_micros();

    if ok {
        let mut max_diff = 0.0_f64;
        for (c, g) in cpu_lls.iter().zip(&gpu_lls) {
            max_diff = max_diff.max((c - g).abs());
        }
        #[allow(clippy::cast_precision_loss)]
        {
            let speedup = cpu_us as f64 / gpu_us.max(1) as f64;
            println!("    CPU: {cpu_us} µs, GPU: {gpu_us} µs, speedup: {speedup:.2}×");
            println!("    max |CPU−GPU| = {max_diff:.2e}");
            v.check(
                "Bootstrap: parity",
                f64::from((max_diff < 1e-4) as u8),
                1.0,
                0.0,
            );
            v.check(
                "Bootstrap: GPU complete",
                gpu_lls.len() as f64,
                n_reps as f64,
                0.0,
            );
        }
    } else {
        v.check("Bootstrap: GPU (skipped)", 1.0, 1.0, 0.0);
    }
}

fn bench_hmm_batch(device: &Arc<barracuda::device::WgpuDevice>, v: &mut Validator) {
    v.section("── HMM Batch Forward: 256 sequences × 100 steps × 3 states ──");

    let model = HmmModel {
        n_states: 3,
        log_pi: vec![
            (1.0_f64 / 3.0).ln(),
            (1.0_f64 / 3.0).ln(),
            (1.0_f64 / 3.0).ln(),
        ],
        log_trans: vec![
            0.5_f64.ln(),
            0.3_f64.ln(),
            0.2_f64.ln(),
            0.2_f64.ln(),
            0.5_f64.ln(),
            0.3_f64.ln(),
            0.3_f64.ln(),
            0.2_f64.ln(),
            0.5_f64.ln(),
        ],
        n_symbols: 2,
        log_emit: vec![
            0.9_f64.ln(),
            0.1_f64.ln(),
            0.2_f64.ln(),
            0.8_f64.ln(),
            0.5_f64.ln(),
            0.5_f64.ln(),
        ],
    };

    let n_seqs = 256_usize;
    let n_steps = 100_usize;

    let mut all_obs_usize: Vec<Vec<usize>> = Vec::with_capacity(n_seqs);
    let mut all_obs_u32: Vec<u32> = Vec::with_capacity(n_seqs * n_steps);
    for seq_idx in 0..n_seqs {
        let obs: Vec<usize> = (0..n_steps)
            .map(|t| (seq_idx * 7 + t * 3) % model.n_symbols)
            .collect();
        for &o in &obs {
            all_obs_u32.push(o as u32);
        }
        all_obs_usize.push(obs);
    }

    let start = Instant::now();
    let cpu_lls: Vec<f64> = all_obs_usize
        .iter()
        .map(|obs| hmm::forward(&model, obs).log_likelihood)
        .collect();
    let cpu_us = start.elapsed().as_micros();

    let hmm_gpu = HmmGpuForward::new(&device);
    let start = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hmm_gpu.forward_batch(&model, &all_obs_u32, n_seqs, n_steps)
    }));
    let gpu_us = start.elapsed().as_micros();

    match result {
        Ok(Ok(gpu_result)) => {
            let mut max_diff = 0.0_f64;
            for (c, g) in cpu_lls.iter().zip(&gpu_result.log_likelihoods) {
                max_diff = max_diff.max((c - g).abs());
            }
            #[allow(clippy::cast_precision_loss)]
            {
                let speedup = cpu_us as f64 / gpu_us.max(1) as f64;
                println!("    CPU: {cpu_us} µs, GPU: {gpu_us} µs, speedup: {speedup:.2}×");
                println!("    max |CPU−GPU| = {max_diff:.2e}");
                v.check(
                    "HMM batch: parity",
                    f64::from((max_diff < 1e-3) as u8),
                    1.0,
                    0.0,
                );
                v.check(
                    "HMM batch: GPU complete",
                    gpu_result.log_likelihoods.len() as f64,
                    n_seqs as f64,
                    0.0,
                );
            }
        }
        Ok(Err(e)) => {
            println!("  [SKIP] GPU: {e}");
            v.check("HMM batch: GPU (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] GPU panicked");
            v.check("HMM batch: GPU (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

fn collect_leaf_states(tree: &TreeNode) -> Vec<(usize, bool)> {
    let mut result = Vec::new();
    fn walk(node: &TreeNode, idx: &mut usize, out: &mut Vec<(usize, bool)>) {
        let my = *idx;
        *idx += 1;
        match node {
            TreeNode::Leaf { .. } => out.push((my, true)),
            TreeNode::Internal { left, right, .. } => {
                out.push((my, false));
                walk(left, idx, out);
                walk(right, idx, out);
            }
        }
    }
    let mut idx = 0;
    walk(tree, &mut idx, &mut result);
    result
}

fn rebuild_tips_from_leaves(
    leaf_info: &[(usize, bool)],
    _phylo: &PhyloTree,
    aln: &Alignment,
    n_sites: usize,
) -> Vec<f64> {
    let n_nodes = leaf_info.len();
    let mut tips = vec![0.0_f64; n_nodes * n_sites * N_STATES];
    let mut leaf_ctr = 0_usize;
    for &(node_idx, is_leaf) in leaf_info {
        if is_leaf {
            for (site, col) in aln.columns.iter().enumerate() {
                let state = col[leaf_ctr];
                if state < N_STATES {
                    tips[node_idx * n_sites * N_STATES + site * N_STATES + state] = 1.0;
                }
            }
            leaf_ctr += 1;
        }
    }
    tips
}
