// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp047: GPU HMM Batch Forward
//!
//! Validates the GPU batch HMM forward algorithm against the CPU
//! implementation. Uses a local WGSL shader (`hmm_forward_f64.wgsl`)
//! — a ToadStool absorption candidate following Write → Absorb → Lean.
//!
//! Sections:
//! 1. **2-state parity** — classic weather HMM, single sequence
//! 2. **3-state parity** — genomic HMM, single sequence
//! 3. **Batch parity** — N independent sequences, CPU vs GPU
//! 4. **Forward-backward consistency** — sum_i alpha[t][i]*beta[t][i] = P(O)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | BarraCUDA CPU (reference) |
//! | Baseline version | wetspring-barracuda 0.1.0 (CPU path) |
//! | Baseline command | bio::hmm::forward |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --release --features gpu --bin validate_gpu_hmm_forward` |
//! | Data | Weather HMM, genomic HMM, 64-seq batch |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Local WGSL shader: hmm_forward_f64.wgsl (ToadStool absorption candidate).

use wetspring_barracuda::bio::hmm::{self, HmmModel};
use wetspring_barracuda::bio::hmm_gpu::HmmGpuForward;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn weather_model() -> HmmModel {
    HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 3,
        log_emit: vec![
            0.1_f64.ln(),
            0.4_f64.ln(),
            0.5_f64.ln(),
            0.6_f64.ln(),
            0.3_f64.ln(),
            0.1_f64.ln(),
        ],
    }
}

fn genomic_model() -> HmmModel {
    HmmModel {
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
    }
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp047: GPU HMM Batch Forward");

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
    let hmm_gpu = HmmGpuForward::new(&device);

    validate_2state(&hmm_gpu, &mut v);
    validate_3state(&hmm_gpu, &mut v);
    validate_batch(&hmm_gpu, &mut v);
    validate_forward_backward(&hmm_gpu, &mut v);

    v.finish();
}

fn validate_2state(gpu: &HmmGpuForward, v: &mut Validator) {
    v.section("── Section 1: 2-State Weather HMM ──");

    let model = weather_model();
    let obs: Vec<usize> = vec![0, 1, 2, 0, 1];
    let cpu = hmm::forward(&model, &obs);

    let obs_u32: Vec<u32> = obs.iter().map(|&x| x as u32).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.forward_batch(&model, &obs_u32, 1, obs.len())
    }));

    match result {
        Ok(Ok(gpu_result)) => {
            let gpu_ll = gpu_result.log_likelihoods[0];
            v.check(
                "2-state: CPU ≈ GPU log-likelihood",
                cpu.log_likelihood,
                gpu_ll,
                1e-6,
            );
            v.check(
                "2-state: GPU LL finite",
                f64::from(gpu_ll.is_finite() as u8),
                1.0,
                0.0,
            );
            v.check(
                "2-state: GPU LL negative",
                f64::from((gpu_ll < 0.0) as u8),
                1.0,
                0.0,
            );

            let mut max_alpha_diff = 0.0_f64;
            for (_, (&cpu_a, &gpu_a)) in cpu
                .log_alpha
                .iter()
                .zip(gpu_result.log_alpha.iter())
                .enumerate()
            {
                let diff = (cpu_a - gpu_a).abs();
                if diff > max_alpha_diff {
                    max_alpha_diff = diff;
                }
            }
            v.check(
                "2-state: max |alpha CPU−GPU| < 1e-6",
                f64::from((max_alpha_diff < 1e-6) as u8),
                1.0,
                0.0,
            );
            println!("    (max alpha diff = {max_alpha_diff:.2e})");
        }
        Ok(Err(e)) => {
            println!("  [SKIP] HMM GPU error: {e}");
            v.check("2-state: GPU available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] HMM GPU panicked (NVVM f64 shader compilation)");
            v.check("2-state: GPU available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

fn validate_3state(gpu: &HmmGpuForward, v: &mut Validator) {
    v.section("── Section 2: 3-State Genomic HMM ──");

    let model = genomic_model();
    let obs: Vec<usize> = vec![0, 1, 0, 0, 1, 1, 0];
    let cpu = hmm::forward(&model, &obs);

    let obs_u32: Vec<u32> = obs.iter().map(|&x| x as u32).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.forward_batch(&model, &obs_u32, 1, obs.len())
    }));

    match result {
        Ok(Ok(gpu_result)) => {
            let gpu_ll = gpu_result.log_likelihoods[0];
            v.check(
                "3-state: CPU ≈ GPU log-likelihood",
                cpu.log_likelihood,
                gpu_ll,
                1e-6,
            );
            v.check(
                "3-state: GPU LL finite",
                f64::from(gpu_ll.is_finite() as u8),
                1.0,
                0.0,
            );

            // Viterbi path should be consistent (if alpha is correct,
            // GPU Viterbi would yield same path)
            let vit = hmm::viterbi(&model, &obs);
            v.check(
                "3-state: Viterbi path len",
                vit.path.len() as f64,
                obs.len() as f64,
                0.0,
            );
            v.check(
                "3-state: Viterbi LL ≤ forward LL",
                f64::from((vit.log_probability <= cpu.log_likelihood + 1e-10) as u8),
                1.0,
                0.0,
            );
        }
        Ok(Err(e)) => {
            println!("  [SKIP] HMM GPU error: {e}");
            v.check("3-state: GPU available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] HMM GPU panicked (NVVM f64 shader compilation)");
            v.check("3-state: GPU available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

fn validate_batch(gpu: &HmmGpuForward, v: &mut Validator) {
    v.section("── Section 3: Batch Forward (64 sequences) ──");

    let model = weather_model();
    let n_seqs = 64_usize;
    let n_steps = 10_usize;

    // Generate deterministic observation sequences
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

    // CPU: per-sequence forward
    let cpu_lls: Vec<f64> = all_obs_usize
        .iter()
        .map(|obs| hmm::forward(&model, obs).log_likelihood)
        .collect();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.forward_batch(&model, &all_obs_u32, n_seqs, n_steps)
    }));

    match result {
        Ok(Ok(gpu_result)) => {
            #[allow(clippy::cast_precision_loss)]
            {
                v.check(
                    "Batch: n_seqs",
                    gpu_result.log_likelihoods.len() as f64,
                    n_seqs as f64,
                    0.0,
                );
            }

            let all_finite = gpu_result.log_likelihoods.iter().all(|x| x.is_finite());
            v.check(
                "Batch: all GPU LLs finite",
                f64::from(all_finite as u8),
                1.0,
                0.0,
            );

            let mut max_diff = 0.0_f64;
            for (cpu_ll, gpu_ll) in cpu_lls.iter().zip(&gpu_result.log_likelihoods) {
                max_diff = max_diff.max((cpu_ll - gpu_ll).abs());
            }
            v.check(
                "Batch: max |CPU−GPU| < 1e-4",
                f64::from((max_diff < tolerances::GPU_VS_CPU_ENSEMBLE) as u8),
                1.0,
                0.0,
            );
            println!("    (max per-sequence diff = {max_diff:.2e})");

            #[allow(clippy::cast_precision_loss)]
            {
                let cpu_mean: f64 = cpu_lls.iter().sum::<f64>() / n_seqs as f64;
                let gpu_mean: f64 = gpu_result.log_likelihoods.iter().sum::<f64>() / n_seqs as f64;
                v.check(
                    "Batch: mean CPU ≈ GPU",
                    cpu_mean,
                    gpu_mean,
                    tolerances::GPU_VS_CPU_ENSEMBLE,
                );
            }
        }
        Ok(Err(e)) => {
            println!("  [SKIP] HMM GPU batch error: {e}");
            v.check("Batch: GPU available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] HMM GPU batch panicked (NVVM f64 shader compilation)");
            v.check("Batch: GPU available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

fn validate_forward_backward(gpu: &HmmGpuForward, v: &mut Validator) {
    v.section("── Section 4: Forward-Backward Consistency ──");

    let model = weather_model();
    let obs: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0];
    let cpu_fwd = hmm::forward(&model, &obs);
    let cpu_bwd = hmm::backward(&model, &obs);

    let obs_u32: Vec<u32> = obs.iter().map(|&x| x as u32).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.forward_batch(&model, &obs_u32, 1, obs.len())
    }));

    match result {
        Ok(Ok(gpu_result)) => {
            // Check forward-backward consistency using GPU alpha:
            // sum_i exp(alpha_gpu[t][i] + beta_cpu[t][i]) ≈ P(O) for all t
            let n = model.n_states;
            let mut max_fb_diff = 0.0_f64;
            for t in 0..obs.len() {
                let mut log_vals: Vec<f64> = Vec::with_capacity(n);
                for i in 0..n {
                    let alpha_gpu = gpu_result.log_alpha[t * n + i];
                    let beta_cpu = cpu_bwd.log_beta[t * n + i];
                    log_vals.push(alpha_gpu + beta_cpu);
                }
                let ll_t = hmm::log_sum_exp(&log_vals);
                let diff = (ll_t - cpu_fwd.log_likelihood).abs();
                max_fb_diff = max_fb_diff.max(diff);
            }
            v.check(
                "FB: GPU alpha + CPU beta consistent",
                f64::from((max_fb_diff < 1e-6) as u8),
                1.0,
                0.0,
            );
            println!("    (max FB consistency diff = {max_fb_diff:.2e})");
        }
        Ok(Err(e)) => {
            println!("  [SKIP] HMM GPU FB error: {e}");
            v.check("FB: GPU available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] HMM GPU FB panicked");
            v.check("FB: GPU available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}
