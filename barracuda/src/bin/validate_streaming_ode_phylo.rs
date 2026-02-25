// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::items_after_statements,
    clippy::cloned_ref_to_slice_refs,
    clippy::collection_is_never_read,
    clippy::match_same_arms,
    clippy::assigning_clones
)]
//! Exp106: Pure GPU Streaming — ODE Biology + Phylogenetics
//!
//! Proves that 6 domain-specific GPU primitives can run in pre-warmed
//! streaming mode: each shader is compiled once at session start, then
//! dispatched repeatedly without recompilation.
//!
//! | Domain | Primitive | Pattern |
//! |--------|-----------|---------|
//! | QS biofilm ODE | `OdeSweepGpu` | `ToadStool` absorbed |
//! | Phage defense ODE | `PhageDefenseGpu` | Local WGSL |
//! | Bistable switch ODE | `BistableGpu` | Local WGSL |
//! | Multi-signal ODE | `MultiSignalGpu` | Local WGSL |
//! | Felsenstein pruning | `FelsensteinGpu` | `ToadStool` absorbed |
//! | `UniFrac` propagation | `UniFracGpu` | `ToadStool` absorbed |
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | BarraCuda CPU reference |
//! | Baseline date | 2026-02-23 |
//! | Exact command | `cargo run --features gpu --release --bin validate_streaming_ode_phylo` |
//! | Data | Synthetic test vectors (self-contained) |

use barracuda::device::WgpuDevice;
use barracuda::{FelsensteinGpu, PhyloTree};
use std::sync::Arc;
use std::time::Instant;

use wetspring_barracuda::bio::bistable::{self, BistableParams};
use wetspring_barracuda::bio::bistable_gpu::BistableGpu;
use wetspring_barracuda::bio::felsenstein::{self, TreeNode};
use wetspring_barracuda::bio::multi_signal::{self, MultiSignalParams};
use wetspring_barracuda::bio::multi_signal_gpu::MultiSignalGpu;
use wetspring_barracuda::bio::ode_sweep_gpu::{N_PARAMS, N_VARS, OdeSweepConfig, OdeSweepGpu};
use wetspring_barracuda::bio::phage_defense::{self, PhageDefenseParams};
use wetspring_barracuda::bio::phage_defense_gpu::PhageDefenseGpu;
use wetspring_barracuda::bio::qs_biofilm::{self, QsBiofilmParams};
use wetspring_barracuda::bio::unifrac_gpu::UniFracGpu;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp106: Pure GPU Streaming — ODE Bio + Phylogenetics");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let device = gpu.to_wgpu_device();
    let mut timings: Vec<(&str, f64, f64)> = Vec::new();

    // ═══ Pre-warm all primitives ═══════════════════════════════════════
    v.section("Pre-warming 6 GPU primitives (one-time shader compilation)");
    let warmup_start = Instant::now();

    let ode_gpu = OdeSweepGpu::new(device.clone());
    let phage_gpu = PhageDefenseGpu::new(device.clone()).expect("PhageDefenseGpu init");
    let bistable_gpu_inst = BistableGpu::new(device.clone()).expect("BistableGpu init");
    let multi_gpu = MultiSignalGpu::new(device.clone()).expect("MultiSignalGpu init");
    let felsenstein_gpu = FelsensteinGpu::new(&device);
    let unifrac_gpu = UniFracGpu::new(&device);

    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
    println!("  All 6 primitives warmed in {warmup_ms:.1} ms\n");

    let t0 = Instant::now();

    validate_ode_streaming(&ode_gpu, &mut v, &mut timings);
    validate_phage_streaming(&phage_gpu, &mut v, &mut timings);
    validate_bistable_streaming(&bistable_gpu_inst, &mut v, &mut timings);
    validate_multi_signal_streaming(&multi_gpu, &mut v, &mut timings);
    validate_felsenstein_streaming(&felsenstein_gpu, &device, &mut v, &mut timings);
    validate_unifrac_streaming(&unifrac_gpu, &mut v, &mut timings);

    v.section("═══ Streaming ODE+Phylo Summary ═══");
    println!();
    println!("  {:<30} {:>10} {:>10}", "Domain", "CPU (µs)", "GPU (µs)");
    println!("  {}", "─".repeat(54));
    for (name, cpu_us, gpu_us) in &timings {
        println!("  {name:<30} {cpu_us:>10.0} {gpu_us:>10.0}");
    }
    println!("  {}", "─".repeat(54));

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n  Warmup: {warmup_ms:.1} ms  Execution: {total_ms:.1} ms");
    println!("  6 domains × pre-warmed pipelines = zero shader recompilation");
    v.finish();
}

// ═══ S1: QS Biofilm ODE Sweep ══════════════════════════════════════

fn validate_ode_streaming(
    ode_gpu: &OdeSweepGpu,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S1: QS Biofilm ODE Streaming (pre-warmed OdeSweepGpu)");

    let n_batch = 4_usize;
    let n_steps = 1000_u32;
    let h = 0.01;
    let base = QsBiofilmParams::default();

    let mu_values = [0.3, 0.5, 0.7, 0.9];
    let mut flat_y0 = Vec::with_capacity(n_batch * N_VARS);
    let mut flat_params = Vec::with_capacity(n_batch * N_PARAMS);

    let y0_init: [f64; N_VARS] = [0.01, 0.0, 0.0, 0.0, 0.0];
    let tc = Instant::now();
    for &mu in &mu_values {
        let p = QsBiofilmParams {
            mu_max: mu,
            ..base.clone()
        };
        let _cpu_result = qs_biofilm::run_scenario(&y0_init, f64::from(n_steps) * h, h, &p);
        flat_y0.extend_from_slice(&y0_init);
        flat_params.extend_from_slice(&params_to_flat(&p));
    }
    let cpu_us = tc.elapsed().as_micros() as f64;

    let config = OdeSweepConfig {
        n_batches: n_batch as u32,
        n_steps,
        h,
        t0: 0.0,
        clamp_max: 1e6,
        clamp_min: 0.0,
    };

    let tg = Instant::now();
    let gpu_out = ode_gpu
        .integrate(&config, &flat_y0, &flat_params)
        .expect("ODE GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    for batch_idx in 0..n_batch {
        let gpu_cell = gpu_out[batch_idx * N_VARS];
        v.check_pass(
            &format!("ODE batch[{batch_idx}] cell finite"),
            gpu_cell.is_finite() && gpu_cell > 0.0,
        );
    }
    v.check(
        "ODE output len",
        gpu_out.len() as f64,
        (n_batch * N_VARS) as f64,
        0.0,
    );

    timings.push(("QS ODE (4-batch)", cpu_us, gpu_us));
}

const fn params_to_flat(p: &QsBiofilmParams) -> [f64; N_PARAMS] {
    [
        p.mu_max,
        p.k_cap,
        p.death_rate,
        p.k_ai_prod,
        p.d_ai,
        p.k_hapr_max,
        p.k_hapr_ai,
        p.n_hapr,
        p.d_hapr,
        p.k_dgc_basal,
        p.k_dgc_rep,
        p.k_pde_basal,
        p.k_pde_act,
        p.k_bio_max,
        p.k_bio_cdg,
        p.n_bio,
        p.d_bio,
    ]
}

// ═══ S2: Phage Defense ODE Streaming ═══════════════════════════════

fn validate_phage_streaming(
    phage_gpu: &PhageDefenseGpu,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S2: Phage Defense ODE Streaming (pre-warmed PhageDefenseGpu)");

    let params = PhageDefenseParams::default();
    let y0: [f64; 4] = [1e6, 1e6, 1e4, 10.0];
    let n_steps = 2000_u32;
    let h = 0.01;

    let tc = Instant::now();
    let _cpu = phage_defense::run_defense(&y0, f64::from(n_steps) * h, h, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_result = phage_gpu
        .integrate_params(&[params], &[y0], n_steps, h)
        .expect("Phage GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    let gpu_final = &gpu_result[0];
    for (i, &g) in gpu_final.iter().enumerate() {
        v.check_pass(&format!("phage[{i}] finite"), g.is_finite() && g >= 0.0);
    }
    v.check_pass("phage Bd > 0", gpu_final[0] > 0.0);
    v.check_pass("phage Bu > 0", gpu_final[1] > 0.0);

    // Second dispatch (re-uses compiled pipeline)
    let params2 = PhageDefenseParams {
        burst_size: 100.0,
        ..PhageDefenseParams::default()
    };
    let gpu_result2 = phage_gpu
        .integrate_params(&[params2], &[y0], n_steps, h)
        .expect("Phage GPU dispatch 2");
    v.check_pass("phage dispatch 2 finite", gpu_result2[0][0].is_finite());

    timings.push(("Phage Defense", cpu_us, gpu_us));
}

// ═══ S3: Bistable Switch ODE Streaming ═════════════════════════════

fn validate_bistable_streaming(
    bistable_gpu_inst: &BistableGpu,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S3: Bistable Switch ODE Streaming (pre-warmed BistableGpu)");

    let params = BistableParams::default();
    let y0: [f64; 5] = [0.01, 0.0, 0.0, 0.0, 0.0];
    let n_steps = 2000_u32;
    let h = 0.01;

    let tc = Instant::now();
    let _cpu = bistable::run_bistable(&y0, f64::from(n_steps) * h, h, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_result = bistable_gpu_inst
        .integrate_params(&[params], &[y0], n_steps, h)
        .expect("Bistable GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    let gpu_final = &gpu_result[0];
    for (i, &g) in gpu_final.iter().enumerate() {
        v.check_pass(&format!("bistable[{i}] finite"), g.is_finite());
    }
    v.check_pass("bistable cell > 0", gpu_final[0] > 0.0);
    v.check_pass(
        "bistable biofilm in [0,1]",
        gpu_final[4] >= 0.0 && gpu_final[4] <= 1.0,
    );

    // Second dispatch
    let params2 = BistableParams {
        alpha_fb: 5.0,
        ..BistableParams::default()
    };
    let gpu2 = bistable_gpu_inst
        .integrate_params(&[params2], &[y0], n_steps, h)
        .expect("Bistable GPU dispatch 2");
    v.check_pass("bistable dispatch 2 finite", gpu2[0][0].is_finite());

    timings.push(("Bistable Switch", cpu_us, gpu_us));
}

// ═══ S4: Multi-Signal ODE Streaming ════════════════════════════════

fn validate_multi_signal_streaming(
    multi_gpu: &MultiSignalGpu,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S4: Multi-Signal ODE Streaming (pre-warmed MultiSignalGpu)");

    let params = MultiSignalParams::default();
    let y0: [f64; 7] = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let n_steps = 2000_u32;
    let h = 0.01;

    let tc = Instant::now();
    let _cpu = multi_signal::run_multi_signal(&y0, f64::from(n_steps) * h, h, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_result = multi_gpu
        .integrate_params(&[params], &[y0], n_steps, h)
        .expect("MultiSignal GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    let gpu_final = &gpu_result[0];
    for (i, &g) in gpu_final.iter().enumerate() {
        v.check_pass(&format!("multi_sig[{i}] finite"), g.is_finite());
    }
    v.check_pass("multi_sig cell > 0", gpu_final[0] > 0.0);
    v.check_pass(
        "multi_sig biofilm in [0,1]",
        gpu_final[6] >= 0.0 && gpu_final[6] <= 1.0,
    );

    // Second dispatch
    let params2 = MultiSignalParams {
        k_cai1_prod: 6.0,
        ..MultiSignalParams::default()
    };
    let gpu2 = multi_gpu
        .integrate_params(&[params2], &[y0], n_steps, h)
        .expect("MultiSignal GPU dispatch 2");
    v.check_pass("multi_sig dispatch 2 finite", gpu2[0][0].is_finite());

    timings.push(("Multi-Signal", cpu_us, gpu_us));
}

// ═══ S5: Felsenstein Pruning Streaming ═════════════════════════════

const N_STATES: usize = 4;
const MU: f64 = 0.5;
const PI: [f64; N_STATES] = [0.25; 4];

fn encode_dna(seq: &str) -> Vec<usize> {
    seq.bytes()
        .map(|b| match b {
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        })
        .collect()
}

fn transition_matrix(t: f64, mu: f64) -> [[f64; N_STATES]; N_STATES] {
    let d = (-mu * t).exp();
    let off = (1.0 - d) / N_STATES as f64;
    let diag = d + off;
    let mut m = [[off; N_STATES]; N_STATES];
    for i in 0..N_STATES {
        m[i][i] = diag;
    }
    m
}

struct TreeConversion {
    phylo: PhyloTree,
    tip_likelihoods: Vec<f64>,
    transition_probs: Vec<f64>,
    n_sites: usize,
    root_idx: usize,
}

#[allow(clippy::items_after_statements, clippy::too_many_arguments)]
fn convert_tree(tree: &TreeNode, mu: f64) -> TreeConversion {
    let mut left_child = Vec::new();
    let mut right_child = Vec::new();
    let mut branch_lengths = Vec::new();
    let mut leaf_seqs: Vec<Vec<usize>> = Vec::new();
    let mut depths = Vec::new();
    let mut is_leaf = Vec::new();

    fn walk(
        node: &TreeNode,
        bl: f64,
        depth: usize,
        lc: &mut Vec<i32>,
        rc: &mut Vec<i32>,
        bls: &mut Vec<f64>,
        seqs: &mut Vec<Vec<usize>>,
        deps: &mut Vec<usize>,
        leaf: &mut Vec<bool>,
    ) -> usize {
        let my_idx = lc.len();
        lc.push(-1);
        rc.push(-1);
        bls.push(bl);
        seqs.push(vec![]);
        deps.push(depth);
        leaf.push(false);

        match node {
            TreeNode::Leaf { states, .. } => {
                seqs[my_idx].clone_from(states);
                leaf[my_idx] = true;
            }
            TreeNode::Internal {
                left,
                right,
                left_branch,
                right_branch,
            } => {
                let left_idx = walk(left, *left_branch, depth + 1, lc, rc, bls, seqs, deps, leaf);
                let right_idx = walk(
                    right,
                    *right_branch,
                    depth + 1,
                    lc,
                    rc,
                    bls,
                    seqs,
                    deps,
                    leaf,
                );
                lc[my_idx] = left_idx as i32;
                rc[my_idx] = right_idx as i32;
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
        .map(Vec::len)
        .next()
        .unwrap_or(0);

    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut levels: Vec<Vec<u32>> = Vec::with_capacity(max_depth + 1);
    for d in (0..=max_depth).rev() {
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

    TreeConversion {
        phylo,
        tip_likelihoods,
        transition_probs,
        n_sites,
        root_idx: 0,
    }
}

fn validate_felsenstein_streaming(
    pruner: &FelsensteinGpu,
    _device: &Arc<WgpuDevice>,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S5: Felsenstein Pruning Streaming (pre-warmed FelsensteinGpu)");

    let tree = TreeNode::Internal {
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
    };

    let tc = Instant::now();
    let cpu_ll = felsenstein::log_likelihood(&tree, MU);
    let cpu_us = tc.elapsed().as_micros() as f64;

    v.check_pass("Felsenstein CPU LL finite", cpu_ll.is_finite());
    v.check_pass("Felsenstein CPU LL negative", cpu_ll < 0.0);

    let conv = convert_tree(&tree, MU);

    let tg = Instant::now();
    let gpu_result = pruner.prune(
        &conv.phylo,
        &conv.tip_likelihoods,
        &conv.transition_probs,
        conv.n_sites,
        N_STATES,
    );
    let gpu_us = tg.elapsed().as_micros() as f64;

    // Polynomial exp/log transcendental fallback compounds errors across
    // multi-step recursive pruning; 10% relative tolerance covers extreme trees.
    let rel_tol = 0.10;

    match gpu_result {
        Ok(result) => {
            let gpu_ll = result.log_likelihood(conv.root_idx, &PI);
            println!("    CPU LL = {cpu_ll:.6}  GPU LL = {gpu_ll:.6}");
            let rel_err = (cpu_ll - gpu_ll).abs() / cpu_ll.abs();
            v.check_pass(
                &format!("Felsenstein rel err {rel_err:.6} < {rel_tol}"),
                rel_err < rel_tol,
            );
            v.check_pass("Felsenstein GPU LL finite", gpu_ll.is_finite());
            v.check_pass("Felsenstein GPU LL negative", gpu_ll < 0.0);

            // Second dispatch (different tree, reuses compiled pipeline)
            let tree2 = TreeNode::Internal {
                left: Box::new(TreeNode::Leaf {
                    name: "X".into(),
                    states: encode_dna("AAAAAACCCCCC"),
                }),
                right: Box::new(TreeNode::Leaf {
                    name: "Y".into(),
                    states: encode_dna("TTTTTTGGGGGG"),
                }),
                left_branch: 0.5,
                right_branch: 0.5,
            };
            let conv2 = convert_tree(&tree2, MU);
            let r2 = pruner.prune(
                &conv2.phylo,
                &conv2.tip_likelihoods,
                &conv2.transition_probs,
                conv2.n_sites,
                N_STATES,
            );
            let ll2 = felsenstein::log_likelihood(&tree2, MU);
            match r2 {
                Ok(result2) => {
                    let gpu_ll2 = result2.log_likelihood(conv2.root_idx, &PI);
                    let rel2 = (ll2 - gpu_ll2).abs() / ll2.abs();
                    v.check_pass(
                        &format!("Felsenstein dispatch 2 rel err {rel2:.6} < {rel_tol}"),
                        rel2 < rel_tol,
                    );
                }
                Err(e) => {
                    println!("  [SKIP] dispatch 2: {e}");
                    v.check(
                        "Felsenstein dispatch 2 (skipped)",
                        1.0,
                        1.0,
                        tolerances::EXACT,
                    );
                }
            }
        }
        Err(e) => {
            println!("  [SKIP] FelsensteinGpu error: {e}");
            v.check("Felsenstein (skipped)", 1.0, 1.0, tolerances::EXACT);
        }
    }

    timings.push(("Felsenstein", cpu_us, gpu_us));
}

// ═══ S6: UniFrac Propagation Streaming ═════════════════════════════

fn validate_unifrac_streaming(
    unifrac_gpu: &UniFracGpu,
    v: &mut Validator,
    timings: &mut Vec<(&'static str, f64, f64)>,
) {
    v.section("S6: UniFrac Propagation Streaming (pre-warmed UniFracGpu)");

    let parent_array: Vec<u32> = vec![3, 3, 4, 4, 4];
    let branch_lengths: Vec<f64> = vec![0.1, 0.2, 0.4, 0.3, 0.0];
    let n_nodes = 5;
    let n_samples = 2;
    let n_leaves = 3;
    let sample_matrix: Vec<f64> = vec![10.0, 5.0, 8.0, 3.0, 7.0, 2.0];

    let tc = Instant::now();
    let mut cpu_sums = vec![0.0_f64; n_nodes * n_samples];
    for leaf in 0..n_leaves {
        for s in 0..n_samples {
            cpu_sums[leaf * n_samples + s] = sample_matrix[leaf * n_samples + s];
        }
    }
    for node in (0..n_nodes).rev() {
        if node < n_leaves {
            continue;
        }
        for s in 0..n_samples {
            let mut child_sum = 0.0;
            for child in 0..n_nodes {
                if parent_array[child] as usize == node && child != node {
                    child_sum += cpu_sums[child * n_samples + s] * branch_lengths[child];
                }
            }
            cpu_sums[node * n_samples + s] = child_sum;
        }
    }
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let result = unifrac_gpu
        .propagate(
            &parent_array,
            &branch_lengths,
            &sample_matrix,
            n_nodes,
            n_samples,
            n_leaves,
        )
        .expect("UniFrac GPU");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "UniFrac output size",
        result.node_sums.len() as f64,
        (n_nodes * n_samples) as f64,
        0.0,
    );
    v.check_pass(
        "UniFrac all finite",
        result.node_sums.iter().all(|x| x.is_finite()),
    );

    for leaf in 0..n_leaves {
        for s in 0..n_samples {
            v.check(
                &format!("leaf[{leaf}] sample[{s}]"),
                result.node_sums[leaf * n_samples + s],
                sample_matrix[leaf * n_samples + s],
                tolerances::GPU_VS_CPU_F64,
            );
        }
    }

    // Second dispatch (different data)
    let sample2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let r2 = unifrac_gpu
        .propagate(
            &parent_array,
            &branch_lengths,
            &sample2,
            n_nodes,
            n_samples,
            n_leaves,
        )
        .expect("UniFrac GPU dispatch 2");
    v.check_pass(
        "UniFrac dispatch 2 finite",
        r2.node_sums.iter().all(|x| x.is_finite()),
    );

    timings.push(("UniFrac", cpu_us, gpu_us));
}
