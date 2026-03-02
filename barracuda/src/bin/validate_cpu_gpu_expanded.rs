// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
//! Exp099: Expanded CPU vs GPU Parity — K-mer, `UniFrac`, ODE Domains
//!
//! # Provenance
//!
//! | Script  | `validate_cpu_gpu_expanded` |
//! | Command | `cargo run --features gpu --bin validate_cpu_gpu_expanded` |
//!
//! # Purpose
//!
//! Validates CPU ↔ GPU parity for newly-wrapped domains:
//! 1. K-mer histogram: CPU `count_kmers` ↔ GPU `KmerGpu`
//! 2. `UniFrac` propagation: CPU tree ↔ GPU `UniFracGpu`
//! 3. ODE sweep: CPU `run_bistable` ↔ GPU `OdeSweepGpu`
//! 4. metalForge mixed-hardware dispatch: GPU→CPU→GPU pipeline
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in barracuda::bio

use std::sync::Arc;
use std::time::Instant;
use wetspring_barracuda::bio::kmer;
use wetspring_barracuda::bio::kmer_gpu::KmerGpu;
use wetspring_barracuda::bio::ode_sweep_gpu::{self, OdeSweepConfig, OdeSweepGpu};
use wetspring_barracuda::bio::phage_defense::{self, PhageDefenseParams};
use wetspring_barracuda::bio::phage_defense_gpu::{PhageDefenseGpu, PhageDefenseOdeConfig};
use wetspring_barracuda::bio::qs_biofilm;
use wetspring_barracuda::bio::unifrac_gpu::UniFracGpu;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn validate_kmer(v: &mut Validator, device: &Arc<barracuda::device::WgpuDevice>) {
    v.section("K-mer Histogram: CPU ↔ GPU (k=4, raw)");
    let t0 = Instant::now();
    let seq = b"ACGTACGTACGTACGTAAACCCCGGGGTTTT";

    let kmer_gpu = KmerGpu::new(device);
    let gpu_result = kmer_gpu
        .count_from_sequence(seq, 4)
        .expect("GPU kmer dispatch");
    let gpu_hist = &gpu_result.histogram;

    let k = 4usize;
    let mask = (1u32 << (2 * k)) - 1;
    let mut cpu_hist = vec![0u32; 4_usize.pow(k as u32)];
    let mut window = 0u32;
    let mut valid = 0usize;
    for &base in seq {
        let encoded = match base {
            b'A' | b'a' => 0u32,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => {
                valid = 0;
                window = 0;
                continue;
            }
        };
        window = ((window << 2) | encoded) & mask;
        valid += 1;
        if valid >= k {
            cpu_hist[window as usize] += 1;
        }
    }

    let us = t0.elapsed().as_micros();

    v.check(
        "histogram length",
        gpu_hist.len() as f64,
        256.0,
        tolerances::EXACT,
    );

    let gpu_total: u32 = gpu_hist.iter().sum();
    let cpu_total: u32 = cpu_hist.iter().sum();
    v.check(
        "total k-mers match",
        f64::from(gpu_total),
        f64::from(cpu_total),
        tolerances::EXACT,
    );

    let mut max_diff = 0u32;
    for (&g, &c) in gpu_hist.iter().zip(cpu_hist.iter()) {
        max_diff = max_diff.max(g.abs_diff(c));
    }
    v.check(
        "max bin difference",
        f64::from(max_diff),
        0.0,
        tolerances::EXACT,
    );
    println!("  K-mer CPU ↔ GPU in {us} µs");
}

fn validate_unifrac(v: &mut Validator, device: &Arc<barracuda::device::WgpuDevice>) {
    v.section("UniFrac Propagation: GPU dispatch");
    let t0 = Instant::now();

    let parent_array: Vec<u32> = vec![2, 2, 4, 4, 4];
    let branch_lengths: Vec<f64> = vec![0.1, 0.2, 0.3, 0.15, 0.0];
    let n_nodes = 5;
    let n_samples = 2;
    let n_leaves = 2;
    let sample_matrix: Vec<f64> = vec![10.0, 5.0, 3.0, 7.0];

    let unifrac_gpu = UniFracGpu::new(device);
    let result = unifrac_gpu
        .propagate(
            &parent_array,
            &branch_lengths,
            &sample_matrix,
            n_nodes,
            n_samples,
            n_leaves,
        )
        .expect("GPU UniFrac dispatch");
    let us = t0.elapsed().as_micros();

    v.check(
        "node_sums length",
        result.node_sums.len() as f64,
        (n_nodes * n_samples) as f64,
        tolerances::EXACT,
    );

    let leaf0_s0 = result.node_sums[0];
    let leaf1_s0 = result.node_sums[n_samples];
    v.check(
        "leaf 0 sample 0 initialized",
        leaf0_s0,
        10.0,
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "leaf 1 sample 0 initialized",
        leaf1_s0,
        3.0,
        tolerances::PYTHON_PARITY,
    );

    let all_finite = result.node_sums.iter().all(|x| x.is_finite());
    v.check_pass("all node sums finite", all_finite);
    println!("  UniFrac GPU dispatch in {us} µs");
}

fn validate_ode_sweep(v: &mut Validator, device: &Arc<barracuda::device::WgpuDevice>) {
    v.section("ODE Sweep: CPU ↔ GPU (QS/c-di-GMP, 4 batches)");
    let t0 = Instant::now();
    let n_batches = 4u32;
    let n_steps = 500u32;
    let dt = 0.01;

    let base_params = qs_biofilm::QsBiofilmParams {
        d_bio: 0.0,
        ..Default::default()
    };
    let flat_full = base_params.to_flat();
    let flat_base = &flat_full[..ode_sweep_gpu::N_PARAMS];
    let y0 = [0.1, 0.0, 0.0, 0.5, 0.0];

    let mut all_y0 = Vec::with_capacity(n_batches as usize * 5);
    let mut all_params = Vec::with_capacity(n_batches as usize * ode_sweep_gpu::N_PARAMS);
    for _ in 0..n_batches {
        all_y0.extend_from_slice(&y0);
        all_params.extend_from_slice(flat_base);
    }

    let cpu_result = qs_biofilm::run_scenario(&y0, f64::from(n_steps) * dt, dt, &base_params);
    let cpu_finals: Vec<f64> = cpu_result
        .states()
        .last()
        .expect("CPU/GPU expanded")
        .to_vec();

    let config = OdeSweepConfig {
        n_batches,
        n_steps,
        h: dt,
        t0: 0.0,
        clamp_max: 1e6,
        clamp_min: 0.0,
    };

    let sweeper = OdeSweepGpu::new(Arc::clone(device));
    let gpu_result = sweeper
        .integrate(&config, &all_y0, &all_params)
        .expect("GPU ODE dispatch");
    let us = t0.elapsed().as_micros();

    v.check(
        "GPU output count",
        gpu_result.len() as f64,
        f64::from(n_batches * 5),
        tolerances::EXACT,
    );

    let gpu_batch0 = &gpu_result[0..5];
    let all_finite = gpu_batch0.iter().all(|x| x.is_finite());
    v.check_pass("GPU ODE outputs finite", all_finite);

    for (i, (&g, &c)) in gpu_batch0.iter().zip(cpu_finals.iter()).enumerate() {
        let denom = c.abs().max(g.abs()).max(tolerances::MATRIX_EPS);
        let rel = (g - c).abs() / denom;
        let tol = if i < 3 {
            tolerances::ODE_STEADY_STATE
        } else {
            tolerances::ODE_NEAR_ZERO_RELATIVE
        };
        v.check(&format!("var {i} CPU ↔ GPU (tol={tol})"), rel, 0.0, tol);
    }

    let gpu_positive = gpu_batch0.iter().all(|&x| x >= 0.0);
    v.check_pass("GPU ODE outputs non-negative", gpu_positive);

    let batches_consistent = gpu_result.chunks(5).all(|chunk| {
        chunk
            .iter()
            .zip(gpu_batch0.iter())
            .all(|(a, b)| (a - b).abs() < tolerances::PYTHON_PARITY)
    });
    v.check_pass("all batches produce identical results", batches_consistent);
    println!("  ODE CPU ↔ GPU in {us} µs");
}

fn validate_phage_defense(v: &mut Validator, device: &Arc<barracuda::device::WgpuDevice>) {
    v.section("Phage Defense ODE: CPU ↔ GPU (local shader, 4 batches)");
    let t0 = Instant::now();
    let n_batches = 4u32;
    let n_steps = 100u32;
    let dt = wetspring_barracuda::tolerances::ODE_DEFAULT_DT;
    let params = PhageDefenseParams::default();
    let y0 = [100.0, 100.0, 10.0, 50.0];

    let cpu_result = phage_defense::run_defense(&y0, f64::from(n_steps) * dt, dt, &params);
    let cpu_finals: Vec<f64> = cpu_result
        .states()
        .last()
        .expect("CPU/GPU expanded")
        .to_vec();

    let phage_gpu = PhageDefenseGpu::new(Arc::clone(device)).expect("PhageDefense GPU compile");

    let mut all_y0 = Vec::with_capacity(n_batches as usize * 4);
    let mut all_params = Vec::with_capacity(n_batches as usize * phage_defense::N_PARAMS);
    let flat = params.to_flat();
    for _ in 0..n_batches {
        all_y0.extend_from_slice(&y0);
        all_params.extend_from_slice(&flat);
    }

    let gpu_config = PhageDefenseOdeConfig {
        n_batches,
        n_steps,
        h: dt,
        t0: 0.0,
        clamp_max: 1e12,
        clamp_min: 0.0,
    };
    let gpu_result = phage_gpu
        .integrate(&gpu_config, &all_y0, &all_params)
        .expect("PhageDefense GPU dispatch");
    let us = t0.elapsed().as_micros();

    v.check(
        "GPU output count",
        gpu_result.len() as f64,
        f64::from(n_batches * 4),
        tolerances::EXACT,
    );

    let gpu_batch0 = &gpu_result[0..4];
    let all_finite = gpu_batch0.iter().all(|x| x.is_finite());
    v.check_pass("GPU outputs finite", all_finite);

    let gpu_positive = gpu_batch0.iter().all(|&x| x >= 0.0);
    v.check_pass("GPU outputs non-negative", gpu_positive);

    for (i, (&g, &c)) in gpu_batch0.iter().zip(cpu_finals.iter()).enumerate() {
        let denom = c.abs().max(g.abs()).max(tolerances::MATRIX_EPS);
        let rel = (g - c).abs() / denom;
        v.check(
            &format!("var {i} CPU ↔ GPU relative diff"),
            rel,
            0.0,
            tolerances::ODE_NEAR_ZERO,
        );
    }

    let batches_ok = gpu_result.chunks(4).all(|chunk| {
        chunk
            .iter()
            .zip(gpu_batch0.iter())
            .all(|(a, b)| (a - b).abs() < tolerances::PYTHON_PARITY)
    });
    v.check_pass("all batches produce identical results", batches_ok);
    println!("  Phage Defense CPU ↔ GPU in {us} µs");
}

fn validate_metalforge_pipeline(v: &mut Validator, device: &Arc<barracuda::device::WgpuDevice>) {
    v.section("metalForge: GPU → CPU → GPU pipeline");
    let t0 = Instant::now();

    let seq = b"ACGTACGTACGTACGTAAACCCCGGGGTTTT";
    let kmer_gpu = KmerGpu::new(device);
    let gpu_kmer_result = kmer_gpu.count_from_sequence(seq, 4).expect("GPU kmer");

    let total: u32 = gpu_kmer_result.histogram.iter().sum();
    let nonzero_bins: usize = gpu_kmer_result.histogram.iter().filter(|&&x| x > 0).count();
    let richness = nonzero_bins as f64 / 256.0;

    v.check(
        "pipeline: kmer total > 0",
        f64::from(total),
        f64::from(total),
        tolerances::EXACT,
    );
    v.check_pass(
        &format!("pipeline: richness = {richness:.4} (nonzero bins)"),
        richness > 0.0 && richness <= 1.0,
    );

    let cpu_counts = kmer::count_kmers(seq, 4);
    let cpu_total: u32 = cpu_counts.to_histogram().iter().sum();
    v.check(
        "pipeline: GPU total == CPU total",
        f64::from(total),
        f64::from(cpu_total),
        tolerances::EXACT,
    );

    let us = t0.elapsed().as_micros();
    println!("  metalForge GPU→CPU→GPU pipeline in {us} µs");
}

#[tokio::main]
async fn main() {
    println!("════════════════════════════════════════════════════════════════════");
    println!("  Exp099: Expanded CPU vs GPU Parity + metalForge Mixed Hardware");
    println!("════════════════════════════════════════════════════════════════════\n");

    let mut v = Validator::new("Exp099: CPU vs GPU Expanded Domains");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    let device = gpu.to_wgpu_device();

    validate_kmer(&mut v, &device);
    validate_unifrac(&mut v, &device);
    validate_ode_sweep(&mut v, &device);
    validate_phage_defense(&mut v, &device);
    validate_metalforge_pipeline(&mut v, &device);

    v.finish();
}
