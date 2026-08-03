// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::bio::ref_index::FmIndex;

fn test_reference() -> Vec<u8> {
    // 500bp synthetic reference with varied sequence to avoid ambiguous mapping
    let mut seq = Vec::with_capacity(500);
    let bases = [b'A', b'C', b'G', b'T'];
    let mut rng = SimpleRng::new(12345);
    for _ in 0..500 {
        seq.push(bases[rng.next_usize() % 4]);
    }
    seq
}

#[test]
fn simulate_reads_produces_correct_count() {
    let reference = test_reference();
    let config = CalibrationConfig {
        n_reads: 50,
        read_length: 30,
        error_rate: 0.01,
        seed: 99,
        ..CalibrationConfig::default()
    };
    let reads = simulate_reads(&reference, &config);
    assert_eq!(reads.len(), 50);
}

#[test]
fn simulated_reads_within_reference_bounds() {
    let reference = test_reference();
    let config = CalibrationConfig {
        n_reads: 100,
        read_length: 30,
        error_rate: 0.0,
        seed: 77,
        ..CalibrationConfig::default()
    };
    let reads = simulate_reads(&reference, &config);
    for read in &reads {
        assert!(read.true_position + config.read_length <= reference.len());
        assert_eq!(read.sequence.len(), config.read_length);
        assert_eq!(read.quality.len(), config.read_length);
    }
}

#[test]
fn zero_error_reads_match_reference_exactly() {
    let reference = test_reference();
    let config = CalibrationConfig {
        n_reads: 20,
        read_length: 30,
        error_rate: 0.0,
        seed: 42,
        ..CalibrationConfig::default()
    };
    let reads = simulate_reads(&reference, &config);
    for read in &reads {
        let expected = &reference[read.true_position..read.true_position + config.read_length];
        assert_eq!(&read.sequence, expected);
    }
}

#[test]
fn error_reads_differ_from_reference() {
    let reference = test_reference();
    let config = CalibrationConfig {
        n_reads: 100,
        read_length: 50,
        error_rate: 0.2, // High error rate to guarantee differences
        seed: 42,
        ..CalibrationConfig::default()
    };
    let reads = simulate_reads(&reference, &config);
    let mismatches: usize = reads
        .iter()
        .map(|r| {
            let expected = &reference[r.true_position..r.true_position + config.read_length];
            r.sequence
                .iter()
                .zip(expected.iter())
                .filter(|(a, b)| a != b)
                .count()
        })
        .sum();
    assert!(mismatches > 0, "high error rate should produce mismatches");
}

#[test]
fn mapq_model_linear_fallback() {
    let model = MapqModel::linear_fallback();
    assert_eq!(model.lookup(0), 0);
    assert_eq!(model.lookup(1), 6);
    assert_eq!(model.lookup(5), 30);
    assert_eq!(model.lookup(10), 60);
    assert_eq!(model.lookup(100), 60); // capped
}

#[test]
fn mapq_model_from_perfect_training() {
    // All samples correct at all gaps — should give MAPQ 60 everywhere
    let samples: Vec<CalibrationSample> = (0..20)
        .map(|gap| CalibrationSample {
            score_gap: gap,
            n_candidates: 2,
            correct: true,
        })
        .collect();

    let model = MapqModel::from_training_data(&samples);
    // P(wrong) = 0 → clamped to 1e-7 → MAPQ = -10*log10(1e-7) = 70, capped at 60
    for gap in 0..20 {
        assert_eq!(model.lookup(gap), 60);
    }
}

#[test]
fn mapq_model_from_all_wrong_training() {
    // All samples wrong — MAPQ should be 0
    let samples: Vec<CalibrationSample> = (0..10)
        .map(|gap| CalibrationSample {
            score_gap: gap,
            n_candidates: 5,
            correct: false,
        })
        .collect();

    let model = MapqModel::from_training_data(&samples);
    for gap in 0..10 {
        assert_eq!(model.lookup(gap), 0);
    }
}

#[test]
fn mapq_model_from_mixed_training() {
    // Gap 0: 50% correct → MAPQ = -10*log10(0.5) ≈ 3
    // Gap 5: 99% correct → MAPQ = -10*log10(0.01) = 20
    let mut samples = Vec::new();
    for _ in 0..50 {
        samples.push(CalibrationSample {
            score_gap: 0,
            n_candidates: 3,
            correct: true,
        });
        samples.push(CalibrationSample {
            score_gap: 0,
            n_candidates: 3,
            correct: false,
        });
    }
    for _ in 0..99 {
        samples.push(CalibrationSample {
            score_gap: 5,
            n_candidates: 2,
            correct: true,
        });
    }
    samples.push(CalibrationSample {
        score_gap: 5,
        n_candidates: 2,
        correct: false,
    });

    let model = MapqModel::from_training_data(&samples);
    let mapq_0 = model.lookup(0);
    let mapq_5 = model.lookup(5);
    assert_eq!(mapq_0, 3, "50% correct → MAPQ 3");
    assert_eq!(mapq_5, 20, "99% correct → MAPQ 20");
    assert!(mapq_5 > mapq_0);
}

#[test]
fn calibrate_small_reference() {
    let reference = test_reference();
    let index = FmIndex::build(&reference);
    let mapper_config = MapperConfig {
        seed_k: 10,
        min_score: 15,
        extension_window: 20,
        max_seed_hits: 100,
        ..MapperConfig::default()
    };
    let cal_config = CalibrationConfig {
        n_reads: 50,
        read_length: 30,
        error_rate: 0.0,
        seed: 42,
        position_tolerance: 5,
    };

    let (model, stats) = calibrate(&reference, "test_ref", &index, &mapper_config, &cal_config);

    assert_eq!(stats.total_reads, 50);
    assert!(stats.mapped > 0, "at least some reads should map");
    assert!(
        stats.accuracy() > 0.5,
        "accuracy should be reasonable: {}",
        stats.accuracy()
    );
    assert!(model.max_gap() > 0);
}

#[test]
fn calibration_stats_methods() {
    let stats = CalibrationStats {
        total_reads: 100,
        mapped: 80,
        correct: 72,
        samples_collected: 80,
    };
    assert!((stats.accuracy() - 0.9).abs() < 1e-10);
    assert!((stats.mapping_rate() - 0.8).abs() < 1e-10);

    let empty = CalibrationStats {
        total_reads: 0,
        mapped: 0,
        correct: 0,
        samples_collected: 0,
    };
    assert!((empty.accuracy() - 0.0).abs() < 1e-10);
    assert!((empty.mapping_rate() - 0.0).abs() < 1e-10);
}

#[test]
fn simple_rng_deterministic() {
    let mut rng1 = SimpleRng::new(42);
    let mut rng2 = SimpleRng::new(42);
    for _ in 0..100 {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
}

#[test]
fn empty_reference_produces_no_reads() {
    let reference = vec![];
    let config = CalibrationConfig::default();
    let reads = simulate_reads(&reference, &config);
    assert!(reads.is_empty());
}
