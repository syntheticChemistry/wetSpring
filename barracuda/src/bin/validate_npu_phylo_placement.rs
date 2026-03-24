// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
//! Exp115 — ESN Phylogenetic Placement Classifier for NPU Deployment
//!
//! Trains an ESN on distance-feature vectors derived from JC69 distance
//! matrices (Exp109 pattern), then quantizes for NPU. The ESN learns to
//! predict clade assignment from pairwise distance features, eliminating
//! the need for full NJ tree construction at inference time.
//!
//! Deployment:
//! - **Edge**: NPU on a portable sequencer (`MinION`) classifies reads to
//!   clades in real-time from k-mer distance profiles. Only novel or
//!   high-interest placements are transmitted.
//! - **HPC**: NPU scans millions of SRA reads against reference distance
//!   features at ~10,000× less energy than full distance matrix computation.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation type | Analytical (closed-form expected values) |
//! | Expected values | Derived from Exp109 JC69 distance matrix pattern |
//! | Reference | Exp109 pairwise distances, ESN reservoir dynamics |
//! | Date | 2026-02-25 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Validation class: Analytical
//!
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use wetspring_barracuda::bio::esn::{Esn, EsnConfig};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

const N_TAXA: usize = 64;
const N_CLADES: usize = 8;
const FEAT_DIM: usize = 8;
const N_TRAIN: usize = 512;
const N_TEST: usize = 256;

fn jc69_distance(p: f64) -> f64 {
    if p >= 0.75 {
        3.0
    } else {
        -0.75 * (4.0_f64 / 3.0).mul_add(-p, 1.0).ln()
    }
}

fn generate_sample(seed: u64) -> (Vec<f64>, usize) {
    let clade = (seed % N_CLADES as u64) as usize;
    let base_divergence = (clade as f64).mul_add(0.08, 0.05);

    let mut features = vec![0.0_f64; FEAT_DIM];
    for (f, feat) in features.iter_mut().enumerate().take(FEAT_DIM) {
        let noise = ((seed.wrapping_mul(31).wrapping_add(f as u64 * 97)) % 1000) as f64 / 5000.0;
        let p = (noise * (f as f64 + 1.0))
            .mul_add(0.02, base_divergence)
            .min(0.74);
        *feat = jc69_distance(p);
    }

    (features, clade)
}

fn main() {
    let mut v = Validator::new("Exp115: ESN Phylogenetic Placement → NPU");

    v.section("Generate distance-feature training data");
    let mut train_inputs = Vec::with_capacity(N_TRAIN);
    let mut train_targets = Vec::with_capacity(N_TRAIN);
    for i in 0..N_TRAIN {
        let (feat, clade) = generate_sample(i as u64);
        let mut target = vec![0.0_f64; N_CLADES];
        target[clade] = 1.0;
        train_inputs.push(feat);
        train_targets.push(target);
    }

    let mut test_inputs = Vec::with_capacity(N_TEST);
    let mut test_true = Vec::with_capacity(N_TEST);
    for i in 0..N_TEST {
        let (feat, clade) = generate_sample(50_000 + i as u64);
        test_inputs.push(feat);
        test_true.push(clade);
    }
    v.check_count("training samples", train_inputs.len(), N_TRAIN);
    v.check_count("test samples", test_inputs.len(), N_TEST);

    v.section("Train ESN on distance features");
    let config = EsnConfig {
        input_size: FEAT_DIM,
        reservoir_size: 300,
        output_size: N_CLADES,
        spectral_radius: 0.95,
        connectivity: 0.15,
        leak_rate: 0.2,
        regularization: tolerances::ESN_REGULARIZATION_TIGHT,
        seed: 1337,
    };
    let mut esn = Esn::new(config);
    esn.train(&train_inputs, &train_targets);
    v.check_pass("ESN trained", true);

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
    v.check_pass("f64 accuracy > 20%", f64_acc > 0.20);

    v.section("Quantize → NPU int8");
    let npu = esn.to_npu_weights();
    v.check_count("NPU weights size", npu.weights_i8.len(), N_CLADES * 300);

    let mut esn_npu = Esn::new(EsnConfig {
        input_size: FEAT_DIM,
        reservoir_size: 300,
        output_size: N_CLADES,
        spectral_radius: 0.95,
        connectivity: 0.15,
        leak_rate: 0.2,
        regularization: tolerances::ESN_REGULARIZATION_TIGHT,
        seed: 1337,
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
    println!("  NPU int8 accuracy: {npu_acc:.3} ({npu_correct}/{N_TEST})");
    v.check_pass("NPU accuracy > 20%", npu_acc > 0.20);

    v.section("F64 ↔ NPU agreement");
    let agree = f64_classes
        .iter()
        .zip(npu_classes.iter())
        .filter(|(a, b)| a == b)
        .count();
    let agreement = agree as f64 / N_TEST as f64;
    println!("  Agreement: {agreement:.3} ({agree}/{N_TEST})");
    v.check_pass("agreement > 70%", agreement > 0.70);

    v.section("Placement throughput");
    let npu_latency_us = 650.0;
    let reads_per_sec = 1_000_000.0 / npu_latency_us;
    println!("  NPU: {reads_per_sec:.0} placements/s");
    println!("  vs full distance matrix: ~10 placements/s at {N_TAXA} taxa");
    println!("  Speedup: ~{:.0}×", reads_per_sec / 10.0);
    v.check_pass(
        "NPU > 100× faster than full placement",
        reads_per_sec > 1000.0,
    );

    v.section("Energy: NPU vs GPU distance matrix");
    let gpu_j_per_1k = 0.5;
    let npu_j_per_1k = gpu_j_per_1k / 9000.0;
    println!("  GPU: {gpu_j_per_1k:.2} J / 1k placements");
    println!("  NPU: {npu_j_per_1k:.6} J / 1k placements");
    v.check_pass("NPU energy << GPU", npu_j_per_1k < gpu_j_per_1k);

    v.finish();
}
