// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! # Exp196c: Int8 Quantization from Noisy Nanopore Reads
//!
//! Validates the NPU-ready int8 quantization pipeline for nanopore data:
//! - Feature extraction from nanopore signal (mean pA, `std_dev`, duration, GC%)
//! - f64 → int8 affine quantization preserving classification fidelity
//! - ESN-like reservoir feature extraction (simple Euler map)
//! - Classification agreement between f64 and int8 paths
//!
//! This experiment proves that the nanopore → NPU classification chain
//! works end-to-end on synthetic data, validating the architecture
//! before hardware arrives.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Source | Synthetic (SyntheticSignalGenerator seed=42, int8 from synthetic reads) |
//! | Date | 2026-02-26 |
//! | Commit | wetSpring Phase 61 |
//! | Hardware | CPU only (int8 simulation) |
//! | Command | `cargo run --release --bin validate_nanopore_int8_quantization` |

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::io::nanopore::{NanoporeRead, SyntheticSignalGenerator};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

/// Feature vector extracted from a nanopore read (f64 precision).
#[derive(Debug, Clone)]
struct ReadFeatures {
    mean_pa: f64,
    std_pa: f64,
    duration_s: f64,
    signal_range: f64,
    complexity: f64,
    gc_estimate: f64,
    energy: f64,
    kurtosis_proxy: f64,
}

/// Extract features from calibrated nanopore signal.
fn extract_features(calibrated: &[f64], sample_rate: f64) -> ReadFeatures {
    let n = calibrated.len() as f64;
    if calibrated.is_empty() {
        return ReadFeatures {
            mean_pa: 0.0,
            std_pa: 0.0,
            duration_s: 0.0,
            signal_range: 0.0,
            complexity: 0.0,
            gc_estimate: 0.0,
            energy: 0.0,
            kurtosis_proxy: 0.0,
        };
    }

    let mean = barracuda::stats::mean(calibrated);
    let std_dev = barracuda::stats::correlation::variance(calibrated)
        .map(|var_sample| (var_sample * (n - 1.0) / n).sqrt())
        .unwrap_or(0.0);
    let (min, max) = calibrated
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));

    let transitions = calibrated
        .windows(2)
        .filter(|w| (w[0] > mean) != (w[1] > mean))
        .count();
    let complexity = transitions as f64 / n;

    let high_current = calibrated.iter().filter(|&&x| x > mean + std_dev).count();
    let gc_estimate = high_current as f64 / n;

    let energy = calibrated.iter().map(|&x| x.powi(2)).sum::<f64>() / n;

    let outliers = calibrated
        .iter()
        .filter(|&&x| (x - mean).abs() > 2.0 * std_dev)
        .count();
    let kurtosis_proxy = outliers as f64 / n;

    ReadFeatures {
        mean_pa: mean,
        std_pa: std_dev,
        duration_s: n / sample_rate,
        signal_range: max - min,
        complexity,
        gc_estimate,
        energy,
        kurtosis_proxy,
    }
}

/// Quantize f64 features to int8 using affine mapping.
fn quantize_features(features: &ReadFeatures) -> [i8; 8] {
    let quantize = |val: f64, lo: f64, hi: f64| -> i8 {
        let normalized = ((val - lo) / (hi - lo)).clamp(0.0, 1.0);
        (normalized * 127.0) as i8
    };

    [
        quantize(features.mean_pa, 150.0, 350.0),
        quantize(features.std_pa, 0.0, 30.0),
        quantize(features.duration_s, 0.0, 5.0),
        quantize(features.signal_range, 0.0, 200.0),
        quantize(features.complexity, 0.0, 1.0),
        quantize(features.gc_estimate, 0.0, 1.0),
        quantize(features.energy, 30000.0, 80000.0),
        quantize(features.kurtosis_proxy, 0.0, 0.2),
    ]
}

/// Simple int8 classifier (simulating NPU SNN inference).
///
/// Classifies into 4 categories based on signal characteristics:
/// - 0: Low-quality (short, noisy)
/// - 1: Standard pore (moderate signal)
/// - 2: High-quality (long, clean signal)
/// - 3: Anomalous (unusual signal pattern)
fn int8_classify(features: [i8; 8]) -> usize {
    let mean = features[0];
    let std_dev = features[1];
    let duration = features[2];
    let complexity = features[4];

    if duration < 10 || std_dev > 100 {
        return 0;
    }
    if complexity > 80 || !(20..=110).contains(&mean) {
        return 3;
    }
    if duration > 60 && std_dev < 50 && (41..=89).contains(&mean) {
        return 2;
    }
    1
}

/// f64 classifier (reference path for accuracy comparison).
fn f64_classify(features: &ReadFeatures) -> usize {
    if features.duration_s < 0.2 || features.std_pa > 25.0 {
        return 0;
    }
    if features.complexity > 0.63 || !(165.0..=325.0).contains(&features.mean_pa) {
        return 3;
    }
    if features.duration_s > 1.2
        && features.std_pa < 12.0
        && (190.0..=310.0).contains(&features.mean_pa)
    {
        return 2;
    }
    1
}

/// Simple reservoir transform: Euler map chaos for feature mixing.
fn reservoir_transform(features: [i8; 8]) -> [i8; 8] {
    let mut state = [0i8; 8];
    for (i, &f) in features.iter().enumerate() {
        let x = f64::from(f) / 127.0;
        let euler = (1.0 + x).powi(3).fract();
        state[i] = (euler * 127.0).clamp(-128.0, 127.0) as i8;
    }
    state
}

fn validate_feature_extraction(v: &mut Validator, features: &ReadFeatures) {
    v.section("── S1: Feature extraction from nanopore signal ──");

    println!("  Mean pA:     {:.2}", features.mean_pa);
    println!("  Std pA:      {:.2}", features.std_pa);
    println!("  Duration:    {:.3} s", features.duration_s);
    println!("  Range:       {:.2} pA", features.signal_range);
    println!("  Complexity:  {:.4}", features.complexity);
    println!("  GC estimate: {:.4}", features.gc_estimate);
    println!("  Energy:      {:.2}", features.energy);
    println!("  Kurtosis:    {:.4}", features.kurtosis_proxy);

    v.check_pass(
        "mean_pa in plausible range (150-350)",
        (150.0..350.0).contains(&features.mean_pa),
    );
    v.check_pass("std_pa > 0 (signal has noise)", features.std_pa > 0.0);
    v.check(
        "duration = 1.0 s (4000 samples / 4000 Hz)",
        features.duration_s,
        1.0,
        tolerances::NANOPORE_CALIBRATION,
    );
    v.check_pass(
        "complexity in [0, 1]",
        (0.0..=1.0).contains(&features.complexity),
    );
}

fn validate_quantization(v: &mut Validator, features: &ReadFeatures) {
    v.section("── S2: Int8 quantization round-trip ──");

    let quantized = quantize_features(features);
    println!("  Quantized: {quantized:?}");

    v.check_pass(
        "quantization produced non-trivial values",
        quantized.iter().any(|&v| v != 0),
    );

    let dequant_mean = (f64::from(quantized[0]) / 127.0).mul_add(350.0 - 150.0, 150.0);
    let recon_error = (dequant_mean - features.mean_pa).abs();
    println!(
        "  Reconstruction error (mean_pa): {recon_error:.4} pA ({:.2}%)",
        recon_error / features.mean_pa * 100.0
    );
    v.check_pass("quantization error < 5 pA", recon_error < 5.0);
}

fn validate_classification(v: &mut Validator, batch: &[NanoporeRead]) {
    v.section("── S3: f64 ↔ int8 classification agreement ──");

    let mut agree = 0usize;
    let mut class_counts = [0usize; 4];

    for read in batch {
        let cal = read.calibrated_signal();
        let feats = extract_features(&cal, 4000.0);
        let f64_class = f64_classify(&feats);
        let int8_feats = quantize_features(&feats);
        let int8_class = int8_classify(int8_feats);

        class_counts[f64_class] += 1;
        if f64_class == int8_class {
            agree += 1;
        }
    }

    let agreement = agree as f64 / batch.len() as f64;
    println!(
        "  Agreement: {agree}/{} ({:.1}%)",
        batch.len(),
        agreement * 100.0
    );
    println!(
        "  Class distribution: low={}, standard={}, high={}, anomalous={}",
        class_counts[0], class_counts[1], class_counts[2], class_counts[3]
    );

    v.check(
        "f64 ↔ int8 agreement",
        agreement,
        1.0,
        1.0 - tolerances::NANOPORE_INT8_FIDELITY,
    );
}

fn validate_reservoir(v: &mut Validator, features: &ReadFeatures) {
    v.section("── S4: Reservoir transform determinism ──");

    let quantized = quantize_features(features);
    let reservoir1 = reservoir_transform(quantized);
    let reservoir2 = reservoir_transform(quantized);

    v.check_pass(
        "reservoir deterministic (same input → same output)",
        reservoir1 == reservoir2,
    );

    v.check_pass(
        "reservoir produced non-trivial values",
        reservoir1.iter().any(|&v| v != 0),
    );

    let features_alt = [10i8, 20, 30, 40, 50, 60, 70, 80];
    let reservoir_alt = reservoir_transform(features_alt);
    v.check_pass(
        "different input → different reservoir state",
        reservoir1 != reservoir_alt,
    );
}

fn validate_diversity(v: &mut Validator, batch: &[NanoporeRead]) {
    v.section("── S5: Diversity metrics from int8 pipeline ──");

    let mut int8_otu = vec![0.0_f64; 4];
    let mut f64_otu = vec![0.0_f64; 4];

    for read in batch {
        let cal = read.calibrated_signal();
        let feats = extract_features(&cal, 4000.0);
        f64_otu[f64_classify(&feats)] += 1.0;
        int8_otu[int8_classify(quantize_features(&feats))] += 1.0;
    }

    let h_f64 = diversity::shannon(&f64_otu);
    let h_int8 = diversity::shannon(&int8_otu);

    println!("  Shannon (f64 path): {h_f64:.4}");
    println!("  Shannon (int8 path): {h_int8:.4}");

    v.check(
        "Shannon diversity preserved through int8",
        h_int8,
        h_f64,
        tolerances::NANOPORE_DIVERSITY_TOLERANCE,
    );

    let bc = diversity::bray_curtis(&f64_otu, &int8_otu);
    println!("  Bray-Curtis(f64 vs int8): {bc:.4}");
    v.check(
        "Bray-Curtis(f64 vs int8) < 0.3",
        bc,
        0.0,
        tolerances::NANOPORE_DIVERSITY_TOLERANCE,
    );
}

fn validate_throughput(v: &mut Validator, batch: &[NanoporeRead]) {
    v.section("── S6: Int8 pipeline throughput ──");

    let start = std::time::Instant::now();
    let n_iters = 50_u32;
    for _ in 0..n_iters {
        for read in batch {
            let cal = read.calibrated_signal();
            let feats = extract_features(&cal, 4000.0);
            let q = quantize_features(&feats);
            let _r = reservoir_transform(q);
            let _c = int8_classify(q);
        }
    }
    let elapsed = start.elapsed();
    let total_reads = u64::from(n_iters) * batch.len() as u64;
    let reads_per_sec = total_reads as f64 / elapsed.as_secs_f64();
    println!(
        "  Throughput: {reads_per_sec:.0} reads/sec (extract + quantize + reservoir + classify)"
    );
    v.check_pass("throughput > 1000 reads/sec", reads_per_sec > 1000.0);
}

fn main() {
    let mut v = Validator::new("Exp196c: Int8 Quantization from Noisy Nanopore Reads");

    let sig = SyntheticSignalGenerator::new(42);
    let read = sig.generate_read(1, 4000, 4000.0);
    let cal = read.calibrated_signal();
    let features = extract_features(&cal, 4000.0);
    let batch = sig.generate_batch(200, 4000, 4000.0);

    validate_feature_extraction(&mut v, &features);
    validate_quantization(&mut v, &features);
    validate_classification(&mut v, &batch);
    validate_reservoir(&mut v, &features);
    validate_diversity(&mut v, &batch);
    validate_throughput(&mut v, &batch);

    v.finish();
}
