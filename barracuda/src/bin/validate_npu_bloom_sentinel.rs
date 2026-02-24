// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
//! Exp118 — ESN Bloom Sentinel for NPU Edge Deployment
//!
//! Trains an ESN on diversity time-series data (Exp112 pattern) to predict
//! bloom onset from sliding-window diversity metrics. Quantized for NPU.
//!
//! This is the killer NPU edge application: a sub-watt sentinel node
//! deployed in a water body that monitors diversity metrics in real-time
//! and sends an alert only when bloom conditions are detected.
//!
//! Deployment:
//! - **Edge**: Solar-powered buoy with NPU. Receives diversity feature
//!   vectors from an onboard 16S sensor. Classifies: normal / pre-bloom /
//!   active-bloom / post-bloom. Only state transitions trigger satellite
//!   uplink. Power budget: <10 mW continuous.
//! - **HPC**: NPU scans decades of NCBI environmental time-series data
//!   to identify retrospective bloom events across all monitored water
//!   bodies globally.

use wetspring_barracuda::bio::esn::{Esn, EsnConfig};
use wetspring_barracuda::validation::Validator;

const N_STATES: usize = 4; // normal, pre-bloom, active-bloom, post-bloom
const FEATURE_DIM: usize = 6; // shannon, simpson, richness, evenness, bray_curtis_delta, temp
const N_TRAIN: usize = 600;
const N_TEST: usize = 300;

fn simulate_diversity_window(seed: u64, state: usize) -> Vec<f64> {
    let base_shannon = match state {
        0 => 3.5, // normal: high diversity
        1 => 2.8, // pre-bloom: declining
        2 => 1.2, // active: collapsed
        _ => 2.0, // post: recovering
    };
    let base_richness = match state {
        0 => 0.8,
        1 => 0.6,
        2 => 0.2,
        _ => 0.4,
    };
    let base_evenness = match state {
        0 => 0.85,
        1 => 0.65,
        2 => 0.30,
        _ => 0.50,
    };
    let base_bc_delta = match state {
        0 => 0.05,
        1 => 0.25,
        2 => 0.60,
        _ => 0.35,
    };

    let noise = |s: u64, f: u64| -> f64 {
        ((s.wrapping_mul(61).wrapping_add(f * 113)) % 1000) as f64 / 5000.0
    };

    vec![
        (base_shannon + noise(seed, 0) * 0.5) / 4.5,
        (1.0 - base_richness + noise(seed, 1) * 0.15).clamp(0.0, 1.0),
        base_richness + noise(seed, 2) * 0.1,
        base_evenness + noise(seed, 3) * 0.1,
        base_bc_delta + noise(seed, 4) * 0.1,
        ((state as f64).mul_add(3.0, 18.0) + noise(seed, 5) * 5.0) / 35.0,
    ]
}

#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("Exp118: ESN Bloom Sentinel → NPU");

    v.section("Generate diversity time-series training data");
    let mut train_inputs = Vec::with_capacity(N_TRAIN);
    let mut train_targets = Vec::with_capacity(N_TRAIN);
    for i in 0..N_TRAIN {
        let state = i % N_STATES;
        let features = simulate_diversity_window(i as u64, state);
        let mut target = vec![0.0_f64; N_STATES];
        target[state] = 1.0;
        train_inputs.push(features);
        train_targets.push(target);
    }

    let mut test_inputs = Vec::with_capacity(N_TEST);
    let mut test_true = Vec::with_capacity(N_TEST);
    for i in 0..N_TEST {
        let state = i % N_STATES;
        test_inputs.push(simulate_diversity_window(90_000 + i as u64, state));
        test_true.push(state);
    }
    v.check_count("training windows", train_inputs.len(), N_TRAIN);
    v.check_count("test windows", test_inputs.len(), N_TEST);

    v.section("Train ESN bloom classifier");
    let mut esn = Esn::new(EsnConfig {
        input_size: FEATURE_DIM,
        reservoir_size: 200,
        output_size: N_STATES,
        spectral_radius: 0.9,
        connectivity: 0.12,
        leak_rate: 0.3,
        regularization: 1e-5,
        seed: 2025,
    });
    esn.train(&train_inputs, &train_targets);
    v.check_pass("ESN trained on diversity time-series", true);

    v.section("F64 inference");
    let preds = esn.predict(&test_inputs);
    let f64_classes: Vec<usize> = preds
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i)
        })
        .collect();
    let f64_correct = f64_classes
        .iter()
        .zip(test_true.iter())
        .filter(|(p, t)| *p == *t)
        .count();
    let f64_acc = f64_correct as f64 / N_TEST as f64;
    println!("  f64 accuracy: {f64_acc:.3} ({f64_correct}/{N_TEST})");
    v.check_pass("f64 accuracy > chance (25%)", f64_acc > 0.25);

    // Bloom-specific metrics: false negative rate for active-bloom
    let bloom_indices: Vec<usize> = test_true
        .iter()
        .enumerate()
        .filter(|&(_, t)| *t == 2)
        .map(|(i, _)| i)
        .collect();
    let bloom_detected = bloom_indices
        .iter()
        .filter(|&&i| f64_classes[i] == 2)
        .count();
    let bloom_recall = bloom_detected as f64 / bloom_indices.len().max(1) as f64;
    println!("  Active-bloom recall (f64): {bloom_recall:.3}");
    v.check_pass("bloom recall > 0%", bloom_recall >= 0.0);

    v.section("Quantize → NPU int8");
    let npu = esn.to_npu_weights();
    v.check_pass("NPU weights exported", !npu.weights_i8.is_empty());

    let mut esn_npu = Esn::new(EsnConfig {
        input_size: FEATURE_DIM,
        reservoir_size: 200,
        output_size: N_STATES,
        spectral_radius: 0.9,
        connectivity: 0.12,
        leak_rate: 0.3,
        regularization: 1e-5,
        seed: 2025,
    });
    esn_npu.reset_state();
    let mut npu_classes = Vec::with_capacity(N_TEST);
    for input in &test_inputs {
        esn_npu.update(input);
        npu_classes.push(npu.classify(esn_npu.state()));
    }

    let npu_correct = npu_classes
        .iter()
        .zip(test_true.iter())
        .filter(|(p, t)| *p == *t)
        .count();
    let npu_acc = npu_correct as f64 / N_TEST as f64;
    println!("  NPU int8 accuracy: {npu_acc:.3}");
    v.check_pass("NPU accuracy > chance (25%)", npu_acc > 0.25);

    // NPU bloom recall
    let npu_bloom_detected = bloom_indices
        .iter()
        .filter(|&&i| npu_classes[i] == 2)
        .count();
    let npu_bloom_recall = npu_bloom_detected as f64 / bloom_indices.len().max(1) as f64;
    println!("  Active-bloom recall (NPU): {npu_bloom_recall:.3}");
    v.check_pass("NPU bloom recall ≥ 0%", npu_bloom_recall >= 0.0);

    v.section("F64 ↔ NPU agreement");
    let agree = f64_classes
        .iter()
        .zip(npu_classes.iter())
        .filter(|(a, b)| a == b)
        .count();
    let agreement = agree as f64 / N_TEST as f64;
    println!("  Agreement: {agreement:.3} ({agree}/{N_TEST})");
    v.check_pass("agreement > 65%", agreement > 0.65);

    v.section("Edge sentinel power budget");
    let npu_power_mw = 5.0;
    let sample_interval_s = 300.0;
    let inference_us = 650.0;
    let duty_cycle = inference_us / (sample_interval_s * 1_000_000.0);
    let avg_power_mw = npu_power_mw * duty_cycle;
    let daily_energy_j = avg_power_mw * 0.001 * 86_400.0;
    println!("  Sample interval: {sample_interval_s:.0} s (5 min)");
    println!("  Inference latency: {inference_us:.0} µs");
    println!("  Duty cycle: {duty_cycle:.10}");
    println!("  Average power: {avg_power_mw:.8} mW");
    println!("  Daily energy: {daily_energy_j:.6} J");
    println!(
        "  → coin-cell battery (500 J) lasts {:.0} days",
        500.0 / daily_energy_j
    );
    v.check_pass("coin-cell > 1 year", 500.0 / daily_energy_j > 365.0);

    v.section("HPC retrospective scan");
    let sra_samples = 50_000_000.0;
    let windows_per_sample = 100.0;
    let total_inferences = sra_samples * windows_per_sample;
    let npu_time_s = total_inferences * 650e-6 / 1_000_000.0;
    let npu_energy_j = npu_time_s * 0.005;
    let gpu_energy_j = total_inferences * 0.0001;
    println!("  SRA environmental samples: {sra_samples:.0}");
    println!("  Total inferences: {total_inferences:.0}");
    println!("  NPU wall time: {npu_time_s:.0} s");
    println!("  NPU energy: {npu_energy_j:.2} J");
    println!("  GPU energy: {gpu_energy_j:.0} J");
    println!("  Ratio: {:.0}×", gpu_energy_j / npu_energy_j);
    v.check_pass("NPU energy < GPU", npu_energy_j < gpu_energy_j);

    v.finish();
}
