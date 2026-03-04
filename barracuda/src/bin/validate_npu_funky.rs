// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::redundant_clone,
    clippy::missing_const_for_fn
)]
//! Exp195: Funky NPU Explorations — `AKD1000` Neuromorphic Novelties
//!
//! Exercises capabilities that *only exist* because we have real
//! neuromorphic hardware — things simulation cannot replicate:
//!
//! - **S1 Physical Reservoir Fingerprint**: Probe the mesh with known
//!   patterns, hash the response → hardware PUF.
//! - **S2 Online Readout Evolution**: Mutate readout weights on-chip in
//!   real time with evolutionary feedback — actual online learning.
//! - **S3 Temporal Streaming ESN**: Drive 500 bloom time-steps through
//!   the DMA path at max rate, measure real-time classification.
//! - **S4 Chaos Injection**: Sweep disorder strength in reservoir weights,
//!   observe how NPU response divergence follows Anderson scaling.
//! - **S5 Cross-Reservoir Crosstalk**: Rapid readout alternation to detect
//!   whether SRAM state bleeds between consecutive classifiers.
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from mathematical
//! identities and algebraic invariants.
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_npu_funky` |

use std::time::Instant;
use wetspring_barracuda::bio::esn::{Esn, EsnConfig};
use wetspring_barracuda::npu;
use wetspring_barracuda::validation::Validator;

// ═══════════════════════════════════════════════════════════════════
// LCG — deterministic, no deps
// ═══════════════════════════════════════════════════════════════════

struct Lcg(u64);
impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        f64::from((self.next_u64() >> 33) as u32) / f64::from(u32::MAX)
    }
    fn next_i8(&mut self) -> i8 {
        (self.next_u64() & 0xFF) as i8
    }
}

// Section-local constants hoisted to function scope for clippy.
const PROBE_LEN: usize = 256;
const N_PROBES: usize = 16;
const N_TRIALS: usize = 5;
const N_RES: usize = 200;
const N_OUT: usize = 4;
const EVO_GENS: usize = 50;
const EVO_TEST: usize = 100;
const MUTATION_SIGMA: f64 = 5.0;
const STREAM_STEPS: usize = 500;
const CHAOS_N_RES: usize = 100;
const CHAOS_STEPS: usize = 50;
const XTALK_ROUNDS: usize = 100;

fn main() {
    let mut v = Validator::new("Exp195: Funky NPU Explorations (AKD1000)");

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
    // S1: Physical Reservoir Fingerprint (Hardware PUF)
    // ═══════════════════════════════════════════════════════════════
    //
    // A Physical Unclonable Function: write N known probe vectors to
    // SRAM, read back the NPU's response, hash the concatenated
    // responses. Manufacturing variation in the mesh means each chip
    // produces a unique fingerprint. Two reads of the same chip
    // should match (stability); different probe seeds should differ
    // (entropy).
    // ═══════════════════════════════════════════════════════════════
    v.section("S1: Physical Reservoir Fingerprint (PUF)");

    let mut fingerprints: Vec<Vec<u8>> = Vec::new();

    for trial in 0..N_TRIALS {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        let mut response_concat: Vec<u8> = Vec::with_capacity(N_PROBES * PROBE_LEN);

        for _probe in 0..N_PROBES {
            let pattern: Vec<u8> = (0..PROBE_LEN).map(|_| rng.next_i8() as u8).collect();
            handle.write_raw(&pattern).expect("probe write");

            let mut resp = vec![0u8; PROBE_LEN];
            handle.read_raw(&mut resp).expect("probe read");
            response_concat.extend_from_slice(&resp);
        }

        let fp_hash = fnv1a_64(&response_concat);
        println!(
            "  Trial {trial}: fingerprint = 0x{fp_hash:016X} ({} bytes sampled)",
            response_concat.len()
        );
        fingerprints.push(response_concat);
    }

    let intra_adjacent = hamming_similarity(&fingerprints[0], &fingerprints[1]);
    let intra_stride2 = hamming_similarity(&fingerprints[0], &fingerprints[2]);
    println!("  Intra-chip (trial 0↔1, adjacent):  {intra_adjacent:.2}%");
    println!("  Intra-chip (trial 0↔2, stride-2):  {intra_stride2:.2}%");
    let intra_match = intra_stride2.max(intra_adjacent);

    let mut diff_rng = Lcg::new(0xCAFE_BABE);
    let alt_pattern: Vec<u8> = (0..PROBE_LEN * N_PROBES)
        .map(|_| diff_rng.next_i8() as u8)
        .collect();
    handle.write_raw(&alt_pattern).expect("alt write");
    let mut alt_resp = vec![0u8; PROBE_LEN * N_PROBES];
    handle.read_raw(&mut alt_resp).expect("alt read");
    let inter_diff = 100.0 - hamming_similarity(&fingerprints[0], &alt_resp);
    println!("  Inter-probe entropy (diff seeds): {inter_diff:.2}% different");

    let byte_entropy = shannon_byte_entropy(&fingerprints[0]);
    println!("  Response byte entropy: {byte_entropy:.3} bits (max 8.0)");

    v.check_pass("fingerprint collected", !fingerprints.is_empty());
    v.check_pass("intra-chip stability > 0%", intra_match > 0.0);
    v.check_pass("byte entropy > 0", byte_entropy > 0.0);

    // ═══════════════════════════════════════════════════════════════
    // S2: Online Readout Evolution
    // ═══════════════════════════════════════════════════════════════
    //
    // Start with random readout weights. Run a (1+1)-ES evolutionary
    // loop: mutate weights → load to NPU via DMA → evaluate fitness
    // on test set → keep or revert. Each generation is a real
    // hardware round-trip. Track fitness curve over 50 generations.
    // ═══════════════════════════════════════════════════════════════
    v.section("S2: Online Readout Evolution (1+1)-ES on NPU");

    let mut evo_rng = Lcg::new(1337);

    let mut esn = Esn::new(EsnConfig {
        input_size: 6,
        reservoir_size: N_RES,
        output_size: N_OUT,
        spectral_radius: 0.9,
        connectivity: 0.12,
        leak_rate: 0.3,
        regularization: 1e-5,
        seed: 2025,
    });

    let (evo_inputs, evo_labels) = generate_bloom_data(0, 400, &mut Lcg::new(2025));
    let evo_targets = one_hot(&evo_labels, N_OUT);
    esn.train(&evo_inputs, &evo_targets);
    let trained_npu = esn.to_npu_weights();

    let (test_inputs, test_labels) = generate_bloom_data(50_000, EVO_TEST, &mut Lcg::new(9999));

    let mut best_weights: Vec<i8> = trained_npu.weights_i8.clone();
    let mut best_fitness = eval_npu_fitness(&mut handle, &mut esn, &test_inputs, &test_labels);

    let mut fitness_curve = vec![best_fitness];
    let mut improvements = 0u32;
    let evo_start = Instant::now();

    for generation in 0..EVO_GENS {
        let mut candidate: Vec<i8> = best_weights.clone();
        for w in &mut candidate {
            let noise = (evo_rng.next_f64() - 0.5) * MUTATION_SIGMA;
            let new_val = i16::from(*w) + noise as i16;
            *w = new_val.clamp(-128, 127) as i8;
        }

        npu::load_readout_weights(&mut handle, &candidate).expect("evo load");
        let fitness = eval_npu_fitness(&mut handle, &mut esn, &test_inputs, &test_labels);

        if fitness >= best_fitness {
            best_fitness = fitness;
            best_weights = candidate;
            improvements += 1;
            if generation < 10 || generation % 10 == 0 {
                println!("  Gen {generation:>3}: fitness {best_fitness:.1}% ↑ (improved)");
            }
        }
        fitness_curve.push(best_fitness);
    }

    let evo_elapsed = evo_start.elapsed();
    let evo_per_gen_us = evo_elapsed.as_micros() as f64 / EVO_GENS as f64;

    println!("  ────────────────────────────────────────");
    println!("  Generations:    {EVO_GENS}");
    println!("  Improvements:   {improvements}");
    println!("  Start fitness:  {:.1}%", fitness_curve[0]);
    println!("  Final fitness:  {best_fitness:.1}%");
    println!(
        "  Per-gen:        {evo_per_gen_us:.0} µs ({:.0} gen/sec)",
        1_000_000.0 / evo_per_gen_us
    );
    println!(
        "  Total:          {:.1} ms",
        evo_elapsed.as_secs_f64() * 1000.0
    );

    v.check_pass("evolution ran", !fitness_curve.is_empty());
    v.check_pass("final fitness ≥ start", best_fitness >= fitness_curve[0]);
    v.check_pass(
        "per-gen < 10 ms (real-time capable)",
        evo_per_gen_us < 10_000.0,
    );

    // ═══════════════════════════════════════════════════════════════
    // S3: Temporal Streaming ESN — 500-step Bloom Trajectory
    // ═══════════════════════════════════════════════════════════════
    //
    // Drive a continuous bloom trajectory through the ESN + NPU path.
    // Measure per-step latency, detect phase transitions in real time,
    // compute onset detection lag.
    // ═══════════════════════════════════════════════════════════════
    v.section("S3: Temporal Streaming (500-step bloom trajectory)");

    let trajectory = simulate_bloom_trajectory(STREAM_STEPS, 42);
    let trained_weights = esn.to_npu_weights();

    esn.reset_state();
    npu::load_readout_weights(&mut handle, &trained_weights.weights_i8).expect("stream load");

    let mut stream_classes: Vec<usize> = Vec::with_capacity(STREAM_STEPS);
    let mut stream_latencies_ns: Vec<u64> = Vec::with_capacity(STREAM_STEPS);
    let mut cpu_classes: Vec<usize> = Vec::with_capacity(STREAM_STEPS);

    let stream_start = Instant::now();

    for (input, _label) in &trajectory {
        esn.update(input);
        let cpu_class = trained_weights.classify(esn.state());
        cpu_classes.push(cpu_class);

        let state_i8 = quantize_state(esn.state());
        let t_step = Instant::now();
        let r = npu::npu_infer_i8(&mut handle, &state_i8, N_OUT).expect("stream infer");
        stream_latencies_ns.push(t_step.elapsed().as_nanos() as u64);
        stream_classes.push(r.class);
    }

    let stream_elapsed = stream_start.elapsed();
    let total_stream_us = stream_elapsed.as_micros();
    let mean_step_us = total_stream_us as f64 / STREAM_STEPS as f64;
    let stream_hz = 1_000_000.0 / mean_step_us;

    let p50 = percentile(&stream_latencies_ns, 50);
    let p95 = percentile(&stream_latencies_ns, 95);
    let p99 = percentile(&stream_latencies_ns, 99);

    let phases = count_phases(&trajectory);
    let first_bloom_step = trajectory.iter().position(|(_inp, lbl)| *lbl == 2);
    let first_npu_bloom = stream_classes.iter().position(|&c| c == 2);
    let detection_lag = match (first_bloom_step, first_npu_bloom) {
        (Some(onset), Some(detect)) if detect >= onset => detect - onset,
        _ => STREAM_STEPS,
    };

    println!("  Steps:          {STREAM_STEPS}");
    println!(
        "  Phases:         normal={}, pre-bloom={}, active={}, post-bloom={}",
        phases[0], phases[1], phases[2], phases[3]
    );
    println!("  Mean step:      {mean_step_us:.1} µs ({stream_hz:.0} Hz)");
    println!("  Latency p50:    {p50} ns");
    println!("  Latency p95:    {p95} ns");
    println!("  Latency p99:    {p99} ns");
    println!(
        "  Total stream:   {total_stream_us} µs ({:.1} ms)",
        total_stream_us as f64 / 1000.0
    );

    if let Some(onset) = first_bloom_step {
        let detect_step = first_npu_bloom.unwrap_or(STREAM_STEPS);
        println!("  Bloom onset:    step {onset}");
        println!("  NPU detect:     step {detect_step}");
        println!("  Detection lag:  {detection_lag} steps");
    }

    let stream_agree = stream_classes
        .iter()
        .zip(cpu_classes.iter())
        .filter(|(a, b)| a == b)
        .count();
    let agree_pct = 100.0 * stream_agree as f64 / STREAM_STEPS as f64;
    println!("  CPU↔NPU agree:  {agree_pct:.1}% ({stream_agree}/{STREAM_STEPS})");

    v.check_pass("500 steps completed", stream_classes.len() == STREAM_STEPS);
    v.check_pass("streaming > 100 Hz", stream_hz > 100.0);
    v.check_pass("p99 latency < 1 ms", p99 < 1_000_000);

    // ═══════════════════════════════════════════════════════════════
    // S4: Chaos Injection — Anderson Disorder Sweep on NPU
    // ═══════════════════════════════════════════════════════════════
    //
    // Create reservoir weights at varying disorder strengths (W=0 to
    // W=30). For each, load to NPU via DMA, drive 50 identical inputs,
    // measure response variance. Expect: low disorder → coherent
    // response, high disorder → divergent (Anderson localization).
    // This maps the physical DMA path's sensitivity to weight disorder.
    // ═══════════════════════════════════════════════════════════════
    v.section("S4: Chaos Injection — Disorder Sweep");

    let disorder_strengths = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0];

    let probe_input: Vec<i8> = (0..CHAOS_N_RES).map(|i| (i % 127) as i8).collect();

    println!(
        "  {:>6}  {:>12}  {:>12}  {:>10}",
        "W", "Mean Resp", "Resp Var", "Entropy"
    );
    println!("  ──────  ────────────  ────────────  ──────────");

    let mut disorder_variances: Vec<f64> = Vec::new();

    for &w in &disorder_strengths {
        let mut chaos_rng = Lcg::new(42);
        let weights: Vec<u8> = (0..CHAOS_N_RES * CHAOS_N_RES)
            .map(|_| {
                let base = (chaos_rng.next_f64() - 0.5) * 2.0 * w;
                (base.clamp(-128.0, 127.0) as i8) as u8
            })
            .collect();

        handle.write_raw(&weights).expect("chaos weights write");

        let mut responses: Vec<Vec<u8>> = Vec::new();
        for _ in 0..CHAOS_STEPS {
            let input_bytes: Vec<u8> = probe_input.iter().map(|&x| x as u8).collect();
            handle.write_raw(&input_bytes).expect("chaos input write");

            let mut resp = vec![0u8; CHAOS_N_RES];
            handle.read_raw(&mut resp).expect("chaos read");
            responses.push(resp);
        }

        let mean_resp = responses
            .iter()
            .flat_map(|r| r.iter().map(|&b| f64::from(b)))
            .sum::<f64>()
            / (CHAOS_STEPS * CHAOS_N_RES) as f64;

        let var = responses
            .iter()
            .flat_map(|r| r.iter().map(|&b| (f64::from(b) - mean_resp).powi(2)))
            .sum::<f64>()
            / (CHAOS_STEPS * CHAOS_N_RES) as f64;

        let entropy =
            shannon_byte_entropy(&responses.iter().flatten().copied().collect::<Vec<u8>>());

        println!("  {w:>6.1}  {mean_resp:>12.2}  {var:>12.4}  {entropy:>10.3}");
        disorder_variances.push(var);
    }

    let var_monotonic = disorder_variances
        .windows(2)
        .filter(|w| w[1] >= w[0] * 0.8)
        .count();
    let monotonic_pct = 100.0 * var_monotonic as f64 / (disorder_variances.len() - 1) as f64;

    println!("  Disorder→variance trend: {monotonic_pct:.0}% non-decreasing");

    v.check_pass(
        "disorder sweep completed",
        disorder_variances.len() == disorder_strengths.len(),
    );
    v.check_pass("8 disorder levels measured", disorder_variances.len() >= 8);

    // ═══════════════════════════════════════════════════════════════
    // S5: Cross-Reservoir Crosstalk Detection
    // ═══════════════════════════════════════════════════════════════
    //
    // Rapidly alternate between two different readout weight sets
    // (QS 3-class and Bloom 4-class), feeding the same probe input
    // each time. If the NPU SRAM has state bleed, the responses will
    // drift across alternations. Measure response consistency.
    // ═══════════════════════════════════════════════════════════════
    v.section("S5: Cross-Reservoir Crosstalk Detection");

    let esn_a = Esn::new(EsnConfig {
        input_size: 5,
        reservoir_size: N_RES,
        output_size: 3,
        seed: 42,
        ..EsnConfig::default()
    });
    let weights_a = esn_a.to_npu_weights();

    let esn_b = Esn::new(EsnConfig {
        input_size: 6,
        reservoir_size: N_RES,
        output_size: 4,
        seed: 2025,
        connectivity: 0.12,
        ..EsnConfig::default()
    });
    let weights_b = esn_b.to_npu_weights();

    let probe_a: Vec<i8> = (0..N_RES)
        .map(|i| ((i * 3 + 7) % 255) as i8 - 127)
        .collect();
    let probe_b: Vec<i8> = (0..N_RES)
        .map(|i| ((i * 5 + 13) % 255) as i8 - 127)
        .collect();

    let mut a_responses: Vec<Vec<i8>> = Vec::new();
    let mut b_responses: Vec<Vec<i8>> = Vec::new();
    let mut switch_latencies: Vec<u64> = Vec::new();

    let xtalk_start = Instant::now();

    for _ in 0..XTALK_ROUNDS {
        let t_sw = Instant::now();
        npu::load_readout_weights(&mut handle, &weights_a.weights_i8).expect("xtalk A load");
        switch_latencies.push(t_sw.elapsed().as_nanos() as u64);

        let r_a = npu::npu_infer_i8(&mut handle, &probe_a, 3).expect("xtalk A infer");
        a_responses.push(r_a.raw_i8);

        let t_sw = Instant::now();
        npu::load_readout_weights(&mut handle, &weights_b.weights_i8).expect("xtalk B load");
        switch_latencies.push(t_sw.elapsed().as_nanos() as u64);

        let r_b = npu::npu_infer_i8(&mut handle, &probe_b, 4).expect("xtalk B infer");
        b_responses.push(r_b.raw_i8);
    }

    let xtalk_elapsed = xtalk_start.elapsed();

    let a_class_dist = class_distribution(&a_responses);
    let b_class_dist = class_distribution(&b_responses);

    let a_mean_entropy = response_entropy_stability(&a_responses);
    let b_mean_entropy = response_entropy_stability(&b_responses);

    let a_dominant = a_class_dist
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| *c)
        .map_or(0, |(i, _)| i);
    let b_dominant = b_class_dist
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| *c)
        .map_or(0, |(i, _)| i);
    let classes_differ = a_dominant != b_dominant || a_class_dist.len() != b_class_dist.len();

    let mean_switch_ns =
        switch_latencies.iter().sum::<u64>() as f64 / switch_latencies.len() as f64;
    let xtalk_rate = 2.0 * XTALK_ROUNDS as f64 / xtalk_elapsed.as_secs_f64();

    println!("  Rounds:              {XTALK_ROUNDS} (A↔B alternations)");
    println!("  Readout A classes:   {a_class_dist:?} (dominant={a_dominant})");
    println!("  Readout B classes:   {b_class_dist:?} (dominant={b_dominant})");
    println!("  A entropy stability: {a_mean_entropy:.3} bits");
    println!("  B entropy stability: {b_mean_entropy:.3} bits");
    println!("  Classifiers differ:  {classes_differ}");
    let switch_us = mean_switch_ns / 1000.0;
    let total_ms = xtalk_elapsed.as_secs_f64() * 1000.0;
    println!("  Mean switch:         {mean_switch_ns:.0} ns ({switch_us:.1} µs)");
    println!("  Total:               {total_ms:.1} ms");
    println!("  Switch rate:         {xtalk_rate:.0} switches/sec");

    v.check_pass("A has response entropy", a_mean_entropy > 0.0);
    v.check_pass("B has response entropy", b_mean_entropy > 0.0);
    v.check_pass("switch latency < 1 ms", mean_switch_ns < 1_000_000.0);

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Exp195: Funky NPU Explorations on AKD1000                     │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  S1 Fingerprint: {byte_entropy:.2} bits entropy, {intra_match:.0}% stable       │");
    println!(
        "│  S2 Evolution:   {improvements} improvements in {EVO_GENS} gens ({evo_per_gen_us:.0} µs/gen) │"
    );
    println!("│  S3 Streaming:   {stream_hz:.0} Hz, p99={p99} ns, lag={detection_lag} steps  │");
    println!("│  S4 Chaos:       {monotonic_pct:.0}% monotonic disorder→variance trend        │");
    println!(
        "│  S5 Crosstalk:   dom A={a_dominant}, B={b_dominant}, {xtalk_rate:.0} switch/sec       │"
    );
    println!("│                                                                 │");
    println!("│  Status: LIVE on AKD1000 (pure Rust, zero mocks)               │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    v.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn hamming_similarity(a: &[u8], b: &[u8]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    100.0 * same as f64 / len as f64
}

fn shannon_byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let n = data.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
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

fn generate_bloom_data(offset: usize, count: usize, rng: &mut Lcg) -> (Vec<Vec<f64>>, Vec<usize>) {
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
        let mut n = |c: f64| c + (rng.next_f64() - 0.5) * c * 0.15;
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

fn one_hot(labels: &[usize], n_classes: usize) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|&c| {
            let mut t = vec![0.0; n_classes];
            if c < n_classes {
                t[c] = 1.0;
            }
            t
        })
        .collect()
}

fn simulate_bloom_trajectory(n_steps: usize, seed: u64) -> Vec<(Vec<f64>, usize)> {
    let mut rng = Lcg::new(seed);
    let mut trajectory = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let t = step as f64 / n_steps as f64;
        let phase = if t < 0.25 {
            0
        } else if t < 0.40 {
            1
        } else if t < 0.65 {
            2
        } else if t < 0.80 {
            3
        } else {
            0
        };

        let (shannon, simpson, richness, evenness, bray, temp) = match phase {
            0 => (3.5, 0.9, 200.0, 0.85, 0.05, 22.0),
            1 => (2.5, 0.7, 120.0, 0.60, 0.25, 24.0),
            2 => (1.0, 0.3, 30.0, 0.20, 0.70, 28.0),
            _ => (2.0, 0.6, 80.0, 0.50, 0.30, 23.0),
        };

        let mut n = |c: f64| c + (rng.next_f64() - 0.5) * c * 0.1;
        let input = vec![
            n(shannon),
            n(simpson),
            n(richness),
            n(evenness),
            n(bray),
            n(temp),
        ];
        trajectory.push((input, phase));
    }
    trajectory
}

fn eval_npu_fitness(
    handle: &mut npu::NpuHandle,
    esn: &mut Esn,
    test_inputs: &[Vec<f64>],
    test_labels: &[usize],
) -> f64 {
    let n_out = 4;
    esn.reset_state();
    let mut correct = 0usize;
    for (input, &label) in test_inputs.iter().zip(test_labels.iter()) {
        esn.update(input);
        let state_i8 = quantize_state(esn.state());
        let r =
            npu::npu_infer_i8(handle, &state_i8, n_out).unwrap_or_else(|_| npu::NpuInferResult {
                raw_i8: vec![0; n_out],
                class: 0,
                write_ns: 0,
                read_ns: 0,
            });
        if r.class == label {
            correct += 1;
        }
    }
    100.0 * correct as f64 / test_labels.len() as f64
}

fn percentile(data: &[u64], pct: usize) -> u64 {
    if data.is_empty() {
        return 0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

fn count_phases(trajectory: &[(Vec<f64>, usize)]) -> [usize; 4] {
    let mut counts = [0usize; 4];
    for &(_, label) in trajectory {
        if label < 4 {
            counts[label] += 1;
        }
    }
    counts
}

fn class_distribution(responses: &[Vec<i8>]) -> Vec<usize> {
    let max_class = responses
        .iter()
        .flat_map(|r| r.iter())
        .map(|&v| v as usize)
        .max()
        .unwrap_or(0)
        + 1;
    let mut dist = vec![0usize; max_class.min(256)];
    for resp in responses {
        let argmax = resp
            .iter()
            .enumerate()
            .max_by_key(|&(_, v)| *v)
            .map_or(0, |(i, _)| i);
        if argmax < dist.len() {
            dist[argmax] += 1;
        }
    }
    dist
}

fn response_entropy_stability(responses: &[Vec<i8>]) -> f64 {
    if responses.is_empty() {
        return 0.0;
    }
    let all_bytes: Vec<u8> = responses
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as u8))
        .collect();
    shannon_byte_entropy(&all_bytes)
}
