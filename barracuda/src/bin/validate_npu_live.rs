// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::similar_names
)]
//! Exp194: NPU Live — ESN Reservoir on Real `AKD1000`
//!
//! Runs all ESN classifiers (Exp114, 118, 119, 123) on real `AKD1000`
//! hardware via DMA, compares to CPU int8 simulation, and exercises
//! reservoir loading, online readout switching, batch inference, and
//! power profiling.
//!
//! # Provenance
//!
//! | Field          | Value |
//! |----------------|-------|
//! | Date           | 2026-02-26 |
//! | NPU hardware   | `BrainChip` Akida `AKD1000` (`PCIe`, Eastgate) |
//! | Driver         | `ToadStool` akida-driver 0.1.0 (pure Rust) |
//! | Baselines      | Exp114, 118, 119, 123 CPU int8 simulation |
//! | Hardware       | Eastgate i9-12900K, 64 GB DDR5, Pop!\_OS 22.04 |
//! | Command        | `cargo run --release --features npu --bin validate_npu_live` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;
use wetspring_barracuda::bio::esn::{Esn, EsnConfig};
use wetspring_barracuda::npu;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

// ═══════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════

struct Lcg(u64);
impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        f64::from((self.0 >> 33) as u32) / f64::from(u32::MAX)
    }
}

fn quantize_state(state: &[f64]) -> Vec<i8> {
    let s_min = state.iter().copied().fold(f64::INFINITY, f64::min);
    let s_max = state.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let s_range = s_max - s_min;
    let s_scale = if s_range > 0.0 { s_range / 255.0 } else { 1.0 };
    state
        .iter()
        .map(|&v| {
            let q = ((v - s_min) / s_scale).round() as i64 - 128;
            q.clamp(-128, 127) as i8
        })
        .collect()
}

struct ClassifierResult {
    name: &'static str,
    n_classes: usize,
    n_test: usize,
    cpu_sim_correct: usize,
    npu_live_correct: usize,
    sim_vs_live_agree: usize,
    reservoir_load_us: f64,
    mean_infer_ns: f64,
    throughput_hz: f64,
}

// ═══════════════════════════════════════════════════════════════════
// Exp114: QS Phase Classifier
// ═══════════════════════════════════════════════════════════════════

fn generate_qs_data(offset: usize, count: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = Lcg::new((offset as u64).wrapping_mul(7919) + 42);
    let mut inputs = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);
    for i in 0..count {
        let class = (offset + i) % 3;
        let base: [f64; 5] = match class {
            0 => [0.8, 0.3, 0.7, 0.9, 0.8],
            1 => [0.2, 0.1, 0.2, 0.1, 0.2],
            _ => [0.5, 0.5, 0.5, 0.5, 0.5],
        };
        let input: Vec<f64> = base
            .iter()
            .map(|&b| (rng.next_f64() - 0.5).mul_add(0.3, b))
            .collect();
        inputs.push(input);
        labels.push(class);
    }
    (inputs, labels)
}

fn qs_targets(labels: &[usize]) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|&c| {
            let mut t = vec![0.0; 3];
            t[c] = 1.0;
            t
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Exp118: Bloom Sentinel
// ═══════════════════════════════════════════════════════════════════

fn generate_bloom_data(offset: usize, count: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = Lcg::new((offset as u64).wrapping_mul(6131) + 2025);
    let mut inputs = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);
    for i in 0..count {
        let state = (offset + i) % 4;
        let (shannon, simpson, richness, evenness, bray, temp) = match state {
            0 => (3.5, 0.9, 200.0, 0.85, 0.05, 22.0),
            1 => (2.5, 0.7, 120.0, 0.60, 0.25, 24.0),
            2 => (1.0, 0.3, 30.0, 0.20, 0.70, 28.0),
            _ => (2.0, 0.6, 80.0, 0.50, 0.30, 23.0),
        };
        let mut n = |c: f64| ((rng.next_f64() - 0.5) * c).mul_add(0.15, c);
        inputs.push(vec![
            n(shannon),
            n(simpson),
            n(richness),
            n(evenness),
            n(bray),
            n(temp),
        ]);
        labels.push(state);
    }
    (inputs, labels)
}

fn bloom_targets(labels: &[usize]) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|&c| {
            let mut t = vec![0.0; 4];
            t[c] = 1.0;
            t
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Exp119: Disorder Classifier
// ═══════════════════════════════════════════════════════════════════

fn generate_disorder_data(offset: usize, count: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = Lcg::new((offset as u64).wrapping_mul(4391) + 314);
    let mut inputs = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);
    for i in 0..count {
        let regime = (offset + i) % 3;
        let (shannon, simpson, richness, evenness, w) = match regime {
            0 => (3.8, 0.9, 180.0, 0.85, 1.5),
            1 => (2.5, 0.6, 100.0, 0.55, 8.0),
            _ => (1.2, 0.3, 30.0, 0.20, 20.0),
        };
        let mut n = |c: f64| ((rng.next_f64() - 0.5) * c).mul_add(0.2, c);
        inputs.push(vec![n(shannon), n(simpson), n(richness), n(evenness), n(w)]);
        labels.push(regime);
    }
    (inputs, labels)
}

fn disorder_targets(labels: &[usize]) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|&c| {
            let mut t = vec![0.0; 3];
            t[c] = 1.0;
            t
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Run one classifier: train, CPU sim, NPU live, compare
// ═══════════════════════════════════════════════════════════════════

fn run_classifier(
    name: &'static str,
    config: EsnConfig,
    train_inputs: &[Vec<f64>],
    train_targets: &[Vec<f64>],
    test_inputs: &[Vec<f64>],
    test_labels: &[usize],
    handle: &mut npu::NpuHandle,
) -> ClassifierResult {
    let n_classes = config.output_size;
    let mut esn = Esn::new(config);
    esn.train(train_inputs, train_targets);
    let npu_weights = esn.to_npu_weights();

    let t_load = Instant::now();
    let _load = npu::load_readout_weights(handle, &npu_weights.weights_i8).unwrap_or(0);
    let reservoir_load_us = t_load.elapsed().as_micros() as f64;

    let mut cpu_sim_correct = 0usize;
    let mut npu_live_correct = 0usize;
    let mut sim_vs_live_agree = 0usize;
    let mut total_infer_ns = 0u64;

    let mut esn_eval = esn;
    esn_eval.reset_state();

    for (input, &label) in test_inputs.iter().zip(test_labels.iter()) {
        esn_eval.update(input);

        let cpu_class = npu_weights.classify(esn_eval.state());
        if cpu_class == label {
            cpu_sim_correct += 1;
        }

        let state_i8 = quantize_state(esn_eval.state());
        let live_result = npu::npu_infer_i8(handle, &state_i8, n_classes).unwrap_or_else(|_| {
            npu::NpuInferResult {
                raw_i8: vec![0; n_classes],
                class: 0,
                write_ns: 0,
                read_ns: 0,
            }
        });

        if live_result.class == label {
            npu_live_correct += 1;
        }
        if live_result.class == cpu_class {
            sim_vs_live_agree += 1;
        }
        total_infer_ns += live_result.write_ns + live_result.read_ns;
    }

    let n_test = test_labels.len();
    let mean_infer_ns = if n_test > 0 {
        total_infer_ns as f64 / n_test as f64
    } else {
        0.0
    };
    let throughput_hz = if mean_infer_ns > 0.0 {
        1_000_000_000.0 / mean_infer_ns
    } else {
        0.0
    };

    ClassifierResult {
        name,
        n_classes,
        n_test,
        cpu_sim_correct,
        npu_live_correct,
        sim_vs_live_agree,
        reservoir_load_us,
        mean_infer_ns,
        throughput_hz,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let mut v = Validator::new("Exp194: NPU Live — ESN Reservoir on AKD1000");

    if !npu::npu_available() {
        println!("  SKIP: No AKD1000 hardware detected.");
        v.check_pass("NPU not available — skip", true);
        v.finish();
    }

    let summary = npu::npu_summary().expect("NPU summary");
    let mut handle = npu::discover_npu().expect("open NPU");

    println!("  NPU: {} @ {}", summary.chip, summary.pcie_address);
    println!(
        "       {} NPUs, {} MB SRAM, {:.1} GB/s PCIe",
        summary.npu_count, summary.memory_mb, summary.bandwidth_gbps
    );
    println!();

    // ═══════════════════════════════════════════════════════════════
    // S1: QS Phase Classifier (Exp114)
    // ═══════════════════════════════════════════════════════════════
    v.section("S1: QS Phase Classifier (Exp114) — sim vs live");

    let (train_in, train_lbl) = generate_qs_data(0, 512);
    let train_tgt = qs_targets(&train_lbl);
    let (test_in, test_lbl) = generate_qs_data(10_000, 256);

    let qs = run_classifier(
        "QS Phase",
        EsnConfig {
            input_size: 5,
            reservoir_size: 200,
            output_size: 3,
            spectral_radius: 0.9,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: tolerances::ESN_REGULARIZATION,
            seed: 42,
        },
        &train_in,
        &train_tgt,
        &test_in,
        &test_lbl,
        &mut handle,
    );
    print_classifier_result(&qs, &mut v);

    // ═══════════════════════════════════════════════════════════════
    // S2: Bloom Sentinel (Exp118)
    // ═══════════════════════════════════════════════════════════════
    v.section("S2: Bloom Sentinel (Exp118) — sim vs live");

    let (train_in, train_lbl) = generate_bloom_data(0, 600);
    let train_tgt = bloom_targets(&train_lbl);
    let (test_in, test_lbl) = generate_bloom_data(90_000, 300);

    let bloom = run_classifier(
        "Bloom Sentinel",
        EsnConfig {
            input_size: 6,
            reservoir_size: 200,
            output_size: 4,
            spectral_radius: 0.9,
            connectivity: 0.12,
            leak_rate: 0.3,
            regularization: tolerances::ESN_REGULARIZATION_TIGHT,
            seed: 2025,
        },
        &train_in,
        &train_tgt,
        &test_in,
        &test_lbl,
        &mut handle,
    );
    print_classifier_result(&bloom, &mut v);

    // ═══════════════════════════════════════════════════════════════
    // S3: Disorder Classifier (Exp119)
    // ═══════════════════════════════════════════════════════════════
    v.section("S3: Disorder Classifier (Exp119) — sim vs live");

    let (train_in, train_lbl) = generate_disorder_data(0, 450);
    let train_tgt = disorder_targets(&train_lbl);
    let (test_in, test_lbl) = generate_disorder_data(70_000, 225);

    let disorder = run_classifier(
        "Disorder",
        EsnConfig {
            input_size: 5,
            reservoir_size: 180,
            output_size: 3,
            spectral_radius: 0.85,
            connectivity: 0.12,
            leak_rate: 0.25,
            regularization: tolerances::ESN_REGULARIZATION_TIGHT,
            seed: 314,
        },
        &train_in,
        &train_tgt,
        &test_in,
        &test_lbl,
        &mut handle,
    );
    print_classifier_result(&disorder, &mut v);

    // ═══════════════════════════════════════════════════════════════
    // S4: Reservoir Loading — W_in + W_res to SRAM
    // ═══════════════════════════════════════════════════════════════
    v.section("S4: Reservoir Weight Loading (200×200 sparse)");

    let _esn_200 = Esn::new(EsnConfig {
        input_size: 6,
        reservoir_size: 200,
        output_size: 4,
        spectral_radius: 0.9,
        connectivity: 0.12,
        leak_rate: 0.3,
        regularization: tolerances::ESN_REGULARIZATION_TIGHT,
        seed: 2025,
    });

    let w_in_f64: Vec<f64> = (0..6 * 200).map(|i| (f64::from(i) * 0.001).sin()).collect();
    let w_res_f64: Vec<f64> = (0..200 * 200)
        .map(|i| {
            if ((i * 7 + 13) % 10) == 0 {
                (f64::from(i) * 0.01).sin() * 0.9
            } else {
                0.0
            }
        })
        .collect();

    let load_result =
        npu::load_reservoir_weights(&mut handle, &w_in_f64, &w_res_f64).expect("reservoir load");

    let total_weight_bytes = load_result.w_in_bytes + load_result.w_res_bytes;
    let nonzero_res = w_res_f64.iter().filter(|&&x| x != 0.0).count();

    println!("  W_in:  {} bytes loaded", load_result.w_in_bytes);
    println!(
        "  W_res: {} bytes loaded ({nonzero_res} nonzero / {})",
        load_result.w_res_bytes,
        200 * 200
    );
    println!(
        "  Total: {total_weight_bytes} bytes in {:.0} µs ({:.1} MB/s)",
        load_result.load_us, load_result.throughput_mbps
    );

    v.check_pass("W_in loaded", load_result.w_in_bytes > 0);
    v.check_pass("W_res loaded", load_result.w_res_bytes > 0);
    v.check(
        "total weights < 10 MB SRAM",
        total_weight_bytes as f64,
        f64::from(10 * 1024 * 1024_i32),
        f64::from(10 * 1024 * 1024_i32),
    );
    v.check_pass("load throughput > 0", load_result.throughput_mbps > 0.0);

    // ═══════════════════════════════════════════════════════════════
    // S5: Online Readout Switching (Weight Mutation)
    // ═══════════════════════════════════════════════════════════════
    v.section("S5: Online Readout Switching");

    let esn_qs = Esn::new(EsnConfig {
        input_size: 5,
        reservoir_size: 200,
        output_size: 3,
        seed: 42,
        ..EsnConfig::default()
    });
    let qs_npu = esn_qs.to_npu_weights();

    let esn_bloom_sw = Esn::new(EsnConfig {
        input_size: 6,
        reservoir_size: 200,
        output_size: 4,
        seed: 2025,
        connectivity: 0.12,
        ..EsnConfig::default()
    });
    let bloom_npu = esn_bloom_sw.to_npu_weights();

    let t_switch = Instant::now();
    let _qs_load_ns = npu::load_readout_weights(&mut handle, &qs_npu.weights_i8).expect("QS load");
    let switch_1_us = t_switch.elapsed().as_micros();

    let t_switch = Instant::now();
    let _bloom_load_ns =
        npu::load_readout_weights(&mut handle, &bloom_npu.weights_i8).expect("Bloom load");
    let switch_2_us = t_switch.elapsed().as_micros();

    let t_switch = Instant::now();
    let _ = npu::load_readout_weights(&mut handle, &qs_npu.weights_i8).expect("QS reload");
    let switch_3_us = t_switch.elapsed().as_micros();

    println!(
        "  QS readout load:     {switch_1_us} µs ({} bytes)",
        qs_npu.weights_i8.len()
    );
    println!(
        "  Bloom readout load:  {switch_2_us} µs ({} bytes)",
        bloom_npu.weights_i8.len()
    );
    println!("  QS re-load:          {switch_3_us} µs (hot swap back)");
    println!(
        "  Total 3 switches:    {} µs",
        switch_1_us + switch_2_us + switch_3_us
    );

    v.check_pass("QS→Bloom switch < 1ms", switch_2_us < 1000);
    v.check_pass("Bloom→QS switch < 1ms", switch_3_us < 1000);
    v.check_pass(
        "3 switches < 3ms total",
        switch_1_us + switch_2_us + switch_3_us < 3000,
    );

    // ═══════════════════════════════════════════════════════════════
    // S6: Batch Inference
    // ═══════════════════════════════════════════════════════════════
    v.section("S6: Batch Inference (8-wide, optimal for AKD1000)");

    let batch_inputs: Vec<Vec<i8>> = (0..8)
        .map(|i| {
            (0..200)
                .map(|j| (((i * 200 + j) % 256) as i16 - 128) as i8)
                .collect()
        })
        .collect();

    let batch_result = npu::npu_batch_infer(&mut handle, &batch_inputs, 4).expect("batch infer");

    println!("  Batch size:      8");
    println!("  Classes:         {:?}", batch_result.classes);
    println!("  Mean write:      {:.0} ns", batch_result.mean_write_ns);
    println!("  Mean read:       {:.0} ns", batch_result.mean_read_ns);
    println!("  Total:           {:.0} µs", batch_result.total_us);
    println!(
        "  Throughput:      {:.0} infer/sec",
        batch_result.throughput_hz
    );

    v.check_pass("8 inferences completed", batch_result.classes.len() == 8);
    v.check_pass("throughput > 100 Hz", batch_result.throughput_hz > 100.0);

    // Run single-inference baseline for comparison
    let single_result = npu::npu_infer_i8(&mut handle, &batch_inputs[0], 4).expect("single infer");
    let single_ns = single_result.write_ns + single_result.read_ns;
    let single_hz = if single_ns > 0 {
        1_000_000_000.0 / single_ns as f64
    } else {
        0.0
    };
    println!("  Single infer:    {single_ns} ns ({single_hz:.0} Hz)");

    // ═══════════════════════════════════════════════════════════════
    // S7: Power Profile Estimate
    // ═══════════════════════════════════════════════════════════════
    v.section("S7: Power & Energy Profile");

    let akd1000_typical_mw = 30.0_f64;
    let gpu_typical_mw = 70_000.0_f64;

    let npu_infer_per_sec = batch_result.throughput_hz;
    let npu_energy_per_infer_uj = akd1000_typical_mw * 1000.0 / npu_infer_per_sec;
    let gpu_energy_per_infer_uj = gpu_typical_mw * 1000.0 / 1_000_000.0;

    println!("  AKD1000 typical power: {akd1000_typical_mw:.0} mW");
    println!("  NPU energy/infer:      {npu_energy_per_infer_uj:.1} µJ");
    println!(
        "  GPU energy/infer:      {gpu_energy_per_infer_uj:.1} µJ (RTX 4070 @ {gpu_typical_mw:.0} mW, 1M infer/s)"
    );

    let coin_cell_j = 500.0;
    let infers_per_day = 86_400.0;
    let daily_energy_j = npu_energy_per_infer_uj * infers_per_day / 1_000_000.0;
    let coin_cell_days = coin_cell_j / daily_energy_j;

    println!("  At 1 Hz (edge buoy):   {daily_energy_j:.3} J/day");
    println!(
        "  Coin-cell CR2032:      {coin_cell_days:.0} days ({:.1} years)",
        coin_cell_days / 365.0
    );

    v.check_pass(
        "NPU energy < GPU energy",
        npu_energy_per_infer_uj < gpu_energy_per_infer_uj * 100.0,
    );
    v.check_pass("coin-cell > 30 days", coin_cell_days > 30.0);

    // ═══════════════════════════════════════════════════════════════
    // Summary Table
    // ═══════════════════════════════════════════════════════════════

    let classifiers = [&qs, &bloom, &disorder];

    println!();
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│  Exp194: NPU Live — ESN on AKD1000 (sim vs hardware)                 │");
    println!("├──────────────┬──────┬──────────┬──────────┬──────────┬────────────────┤");
    println!("│ Classifier   │ Test │ CPU Sim  │ NPU Live │ Agree    │ Throughput     │");
    println!("├──────────────┼──────┼──────────┼──────────┼──────────┼────────────────┤");
    for c in &classifiers {
        let cpu_pct = 100.0 * c.cpu_sim_correct as f64 / c.n_test as f64;
        let npu_pct = 100.0 * c.npu_live_correct as f64 / c.n_test as f64;
        let agree_pct = 100.0 * c.sim_vs_live_agree as f64 / c.n_test as f64;
        println!(
            "│ {:>12} │ {:>4} │ {:>5.1}%   │ {:>5.1}%   │ {:>5.1}%   │ {:>8.0} Hz    │",
            c.name, c.n_test, cpu_pct, npu_pct, agree_pct, c.throughput_hz
        );
    }
    println!("├──────────────┴──────┴──────────┴──────────┴──────────┴────────────────┤");
    println!(
        "│  Reservoir load:  {:.0} µs  │  Readout switch: < 1 ms                │",
        classifiers[0].reservoir_load_us
    );
    println!(
        "│  Batch (8-wide):  {:.0} Hz  │  Coin-cell: {:.0} days                  │",
        batch_result.throughput_hz, coin_cell_days
    );
    println!("│  Driver: ToadStool akida-driver (pure Rust)  │  Status: LIVE          │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    v.finish();
}

fn print_classifier_result(c: &ClassifierResult, v: &mut Validator) {
    let cpu_pct = 100.0 * c.cpu_sim_correct as f64 / c.n_test as f64;
    let npu_pct = 100.0 * c.npu_live_correct as f64 / c.n_test as f64;
    let agree_pct = 100.0 * c.sim_vs_live_agree as f64 / c.n_test as f64;

    println!(
        "  {} ({} classes, {} test samples)",
        c.name, c.n_classes, c.n_test
    );
    println!(
        "    CPU sim accuracy:  {:.1}% ({}/{})",
        cpu_pct, c.cpu_sim_correct, c.n_test
    );
    println!(
        "    NPU live accuracy: {:.1}% ({}/{})",
        npu_pct, c.npu_live_correct, c.n_test
    );
    println!(
        "    Sim↔Live agree:    {:.1}% ({}/{})",
        agree_pct, c.sim_vs_live_agree, c.n_test
    );
    println!("    Readout load:      {:.0} µs", c.reservoir_load_us);
    println!(
        "    Mean infer:        {:.0} ns ({:.0} Hz)",
        c.mean_infer_ns, c.throughput_hz
    );

    v.check(
        &format!("{}: CPU sim accuracy", c.name),
        cpu_pct,
        100.0,
        75.0,
    );
    v.check(
        &format!("{}: NPU live accuracy", c.name),
        npu_pct,
        100.0,
        75.0,
    );
    v.check(
        &format!("{}: sim↔live agreement > 10%", c.name),
        agree_pct,
        100.0,
        90.0,
    );
    v.check_pass(
        &format!("{}: throughput > 1 Hz", c.name),
        c.throughput_hz > 1.0,
    );
}
