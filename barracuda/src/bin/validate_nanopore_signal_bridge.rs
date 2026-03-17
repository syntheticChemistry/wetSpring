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
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
//! # Exp196a: Nanopore Signal Bridge — POD5 Parser Validation
//!
//! Validates the `io::nanopore` module with synthetic nanopore data:
//! - NRS wire format round-trip (write → read → bit-exact signal)
//! - Calibration affine transform (raw ADC → picoamperes)
//! - Signal statistics (mean, `std_dev`, min, max)
//! - Synthetic basecalling (sequence → signal → basecall → accuracy)
//! - Batch generation and streaming iteration
//!
//! This is a pre-hardware experiment: no `MinION` required.
//! The synthetic signal generator produces known-answer data
//! for deterministic validation.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Source | Synthetic (`SyntheticSignalGenerator` seed=42) |
//! | Date | 2026-02-26 |
//! | Commit | wetSpring Phase 61 |
//! | Hardware | CPU only (pre-hardware) |
//! | Command | `cargo run --release --bin validate_nanopore_signal_bridge` |
//!
//! Validation class: Synthetic
//! Provenance: Generated data with known statistical properties

use wetspring_barracuda::io::nanopore::{
    self, NanoporeIter, NanoporeRead, SyntheticSignalGenerator,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn temp_nrs(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("wetspring_exp196a_{name}.nrs"))
}

fn validate_nrs_roundtrip(v: &mut Validator) {
    v.section("── S1: NRS wire format round-trip ──");

    let sig = SyntheticSignalGenerator::new(42);
    let reads = sig.generate_batch(20, 4000, 4000.0);

    let path = temp_nrs("roundtrip");
    nanopore::write_nrs(&path, &reads).or_exit("unexpected error");

    let loaded: Vec<NanoporeRead> = NanoporeIter::open(&path)
        .or_exit("unexpected error")
        .map(|r| r.or_exit("unexpected error"))
        .collect();

    v.check_count("read count round-trip", loaded.len(), 20);

    let mut signal_exact = true;
    let mut cal_exact = true;
    let mut id_exact = true;
    for (orig, rt) in reads.iter().zip(loaded.iter()) {
        if orig.read_id != rt.read_id {
            id_exact = false;
        }
        if orig.signal != rt.signal {
            signal_exact = false;
        }
        if (orig.calibration_offset - rt.calibration_offset).abs()
            > tolerances::NANOPORE_SIGNAL_ROUNDTRIP
            || (orig.calibration_scale - rt.calibration_scale).abs()
                > tolerances::NANOPORE_SIGNAL_ROUNDTRIP
            || (orig.sample_rate - rt.sample_rate).abs() > tolerances::NANOPORE_SIGNAL_ROUNDTRIP
        {
            cal_exact = false;
        }
    }

    v.check_pass("read_id bit-exact round-trip", id_exact);
    v.check_pass(
        "signal bit-exact round-trip (20 reads × 4000 samples)",
        signal_exact,
    );
    v.check_pass("calibration values bit-exact round-trip", cal_exact);

    let total_samples: usize = reads.iter().map(|r| r.signal.len()).sum();
    v.check_count("total signal samples", total_samples, 80_000);
}

fn validate_calibration(v: &mut Validator) {
    v.section("── S2: Calibration affine transform ──");

    let known_read = NanoporeRead {
        read_id: [0; 16],
        signal: vec![0, 100, 200, 400, -100, -200, i16::MAX, i16::MIN],
        channel: 1,
        sample_rate: 4000.0,
        calibration_offset: 200.0,
        calibration_scale: 0.15,
    };

    let cal = known_read.calibrated_signal();

    let cases: &[(&str, usize, f64)] = &[
        ("calibration(0) = 200.0 pA", 0, 200.0),
        ("calibration(100) = 215.0 pA", 1, 215.0),
        ("calibration(200) = 230.0 pA", 2, 230.0),
        ("calibration(400) = 260.0 pA", 3, 260.0),
        ("calibration(-100) = 185.0 pA", 4, 185.0),
        ("calibration(-200) = 170.0 pA", 5, 170.0),
    ];
    for &(label, idx, expected) in cases {
        v.check(label, cal[idx], expected, tolerances::NANOPORE_CALIBRATION);
    }
}

fn validate_signal_stats(v: &mut Validator) {
    v.section("── S3: Signal statistics ──");

    let stat_read = NanoporeRead {
        read_id: [0; 16],
        signal: vec![100, 200, 300, 400, 500],
        channel: 1,
        sample_rate: 4000.0,
        calibration_offset: 0.0,
        calibration_scale: 1.0,
    };

    let stats = stat_read.signal_stats();
    v.check(
        "mean(100..500) = 300.0",
        stats.mean,
        300.0,
        tolerances::NANOPORE_SIGNAL_STATS,
    );
    v.check_count("min = 100", stats.min as usize, 100);
    v.check_count("max = 500", stats.max as usize, 500);
    v.check_count("n_samples = 5", stats.n_samples, 5);

    let samples = [100.0_f64, 200.0, 300.0, 400.0, 500.0];
    let n = samples.len() as f64;
    let expected_std = barracuda::stats::correlation::variance(&samples[..])
        .map(|var_sample| (var_sample * (n - 1.0) / n).sqrt())
        .unwrap_or(0.0);
    v.check(
        "std_dev = sqrt(var(100..500))",
        stats.std_dev,
        expected_std,
        tolerances::NANOPORE_SIGNAL_STATS,
    );

    v.check(
        "duration(8000 samples, 4000 Hz) = 2.0 s",
        NanoporeRead {
            read_id: [0; 16],
            signal: vec![0; 8000],
            channel: 1,
            sample_rate: 4000.0,
            calibration_offset: 0.0,
            calibration_scale: 1.0,
        }
        .duration_seconds(),
        2.0,
        tolerances::NANOPORE_CALIBRATION,
    );
}

fn validate_basecalling(v: &mut Validator) {
    v.section("── S4: Synthetic basecalling ──");

    let sequence = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
    let sig = SyntheticSignalGenerator::new(42);
    let read = sig.generate_from_sequence(1, sequence, 200, 4000.0);

    v.check_count(
        "signal length = 32 bases × 200 spb = 6400",
        read.signal.len(),
        6400,
    );

    let cal = read.calibrated_signal();
    let called = nanopore::simple_basecall(&cal, 32);
    v.check_count("basecalled length = 32", called.len(), 32);

    let matches = called
        .iter()
        .zip(sequence.iter())
        .filter(|(a, b)| a == b)
        .count();
    let accuracy = matches as f64 / sequence.len() as f64;
    println!(
        "  Basecall accuracy: {matches}/{} ({accuracy:.1}%)",
        sequence.len()
    );

    v.check_pass(
        &format!(
            "basecall accuracy >= {:.0}%",
            tolerances::NANOPORE_BASECALL_ACCURACY * 100.0
        ),
        accuracy >= tolerances::NANOPORE_BASECALL_ACCURACY,
    );

    let a_mean: f64 = cal[..200].iter().sum::<f64>() / 200.0;
    let c_mean: f64 = cal[200..400].iter().sum::<f64>() / 200.0;
    let g_mean: f64 = cal[400..600].iter().sum::<f64>() / 200.0;
    let t_mean: f64 = cal[600..800].iter().sum::<f64>() / 200.0;

    println!("  Base levels (pA): A={a_mean:.1}, C={c_mean:.1}, G={g_mean:.1}, T={t_mean:.1}");

    v.check_pass(
        "A < C < G < T level ordering",
        a_mean < c_mean && c_mean < g_mean && g_mean < t_mean,
    );
}

fn validate_uuid_format(v: &mut Validator) {
    v.section("── S5: UUID v4 format compliance ──");

    let sig = SyntheticSignalGenerator::new(99);
    let test_read = sig.generate_read(1, 100, 4000.0);

    v.check_pass("UUID version nibble = 4", test_read.read_id[6] >> 4 == 4);
    v.check_pass(
        "UUID variant bits = 10xxxxxx",
        test_read.read_id[8] >> 6 == 2,
    );

    let hex = test_read.read_id_hex();
    v.check_count("UUID hex length = 32", hex.len(), 32);
}

fn validate_edge_cases(v: &mut Validator) {
    v.section("── S6: Edge cases ──");

    let empty_reads: Vec<NanoporeRead> = Vec::new();
    let p2 = temp_nrs("empty");
    nanopore::write_nrs(&p2, &empty_reads).or_exit("unexpected error");
    let empty_count = NanoporeIter::open(&p2).or_exit("unexpected error").count();
    v.check_count("empty NRS round-trip", empty_count, 0);

    let tiny = NanoporeRead {
        read_id: [0xFF; 16],
        signal: vec![42],
        channel: 512,
        sample_rate: 5000.0,
        calibration_offset: 100.0,
        calibration_scale: 0.2,
    };
    let p3 = temp_nrs("single");
    nanopore::write_nrs(&p3, &[tiny]).or_exit("unexpected error");
    let loaded3: Vec<NanoporeRead> = NanoporeIter::open(&p3)
        .or_exit("unexpected error")
        .map(|r| r.or_exit("unexpected error"))
        .collect();
    v.check_count("single-sample read round-trip", loaded3.len(), 1);
    v.check_count(
        "single-sample signal preserved",
        loaded3[0].signal[0] as usize,
        42,
    );

    let p4 = temp_nrs("badmagic");
    std::fs::write(&p4, b"BAD!").or_exit("unexpected error");
    v.check_pass(
        "invalid NRS magic rejected",
        NanoporeIter::open(&p4).is_err(),
    );
}

fn main() {
    let mut v = Validator::new("Exp196a: Nanopore Signal Bridge (POD5 Parser Validation)");
    validate_nrs_roundtrip(&mut v);
    validate_calibration(&mut v);
    validate_signal_stats(&mut v);
    validate_basecalling(&mut v);
    validate_uuid_format(&mut v);
    validate_edge_cases(&mut v);
    v.finish();
}
