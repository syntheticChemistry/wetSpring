// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

fn make_call(pos: usize, vtype: VariantType) -> CalledVariant {
    CalledVariant {
        position: pos,
        variant_type: vtype,
        ref_allele: b'A',
        alt_allele: b'T',
        depth: 50,
        frequency: 0.8,
        quality: 30.0,
        gene: None,
    }
}

fn make_ref_snp(pos: usize) -> (String, usize, String) {
    ("SNP".to_string(), pos, "T".to_string())
}

fn make_ref_del(pos: usize) -> (String, usize, String) {
    ("DEL".to_string(), pos, String::new())
}

fn make_ref_ins(pos: usize) -> (String, usize, String) {
    ("INS".to_string(), pos, "AA".to_string())
}

#[test]
fn perfect_concordance() {
    let sovereign = vec![
        make_call(100, VariantType::Snp),
        make_call(200, VariantType::Snp),
        make_call(300, VariantType::Deletion),
    ];
    let reference = vec![make_ref_snp(100), make_ref_snp(200), make_ref_del(300)];

    let result = cross_validate(&sovereign, &reference, 5);
    assert_eq!(result.overall.true_positives, 3);
    assert_eq!(result.overall.false_negatives, 0);
    assert_eq!(result.overall.false_positives, 0);
    assert!((result.overall.sensitivity() - 1.0).abs() < 1e-10);
    assert!((result.overall.precision() - 1.0).abs() < 1e-10);
    assert!((result.overall.f1_score() - 1.0).abs() < 1e-10);
}

#[test]
fn missed_mutations_reduce_sensitivity() {
    let sovereign = vec![make_call(100, VariantType::Snp)];
    let reference = vec![make_ref_snp(100), make_ref_snp(200), make_ref_snp(300)];

    let result = cross_validate(&sovereign, &reference, 5);
    assert_eq!(result.overall.true_positives, 1);
    assert_eq!(result.overall.false_negatives, 2);
    assert_eq!(result.overall.false_positives, 0);
    assert!((result.overall.sensitivity() - 1.0 / 3.0).abs() < 1e-10);
    assert!((result.overall.precision() - 1.0).abs() < 1e-10);
}

#[test]
fn extra_calls_reduce_precision() {
    let sovereign = vec![
        make_call(100, VariantType::Snp),
        make_call(500, VariantType::Snp),
        make_call(600, VariantType::Snp),
    ];
    let reference = vec![make_ref_snp(100)];

    let result = cross_validate(&sovereign, &reference, 5);
    assert_eq!(result.overall.true_positives, 1);
    assert_eq!(result.overall.false_negatives, 0);
    assert_eq!(result.overall.false_positives, 2);
    assert!((result.overall.sensitivity() - 1.0).abs() < 1e-10);
    assert!((result.overall.precision() - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn window_tolerance_matches_nearby() {
    let sovereign = vec![make_call(103, VariantType::Snp)]; // 3bp away
    let reference = vec![make_ref_snp(100)];

    let within = cross_validate(&sovereign, &reference, 5);
    assert_eq!(within.overall.true_positives, 1);

    let outside = cross_validate(&sovereign, &reference, 2);
    assert_eq!(outside.overall.true_positives, 0);
    assert_eq!(outside.overall.false_positives, 1);
    assert_eq!(outside.overall.false_negatives, 1);
}

#[test]
fn per_type_breakdown() {
    let sovereign = vec![
        make_call(100, VariantType::Snp),
        make_call(200, VariantType::Deletion),
        make_call(300, VariantType::Insertion),
        make_call(400, VariantType::Snp), // extra FP
    ];
    let reference = vec![
        make_ref_snp(100),
        make_ref_del(200),
        make_ref_ins(300),
        make_ref_snp(500), // missed by us
    ];

    let result = cross_validate(&sovereign, &reference, 5);
    assert_eq!(result.snp.true_positives, 1);
    assert_eq!(result.snp.false_positives, 1); // pos 400
    assert_eq!(result.snp.false_negatives, 1); // pos 500
    assert_eq!(result.del.true_positives, 1);
    assert_eq!(result.del.false_positives, 0);
    assert_eq!(result.del.false_negatives, 0);
    assert_eq!(result.ins.true_positives, 1);
    assert_eq!(result.ins.false_positives, 0);
    assert_eq!(result.ins.false_negatives, 0);
}

#[test]
fn empty_inputs() {
    let result = cross_validate(&[], &[], 5);
    assert_eq!(result.overall.true_positives, 0);
    assert!((result.overall.sensitivity() - 0.0).abs() < 1e-10);
    assert!((result.overall.precision() - 0.0).abs() < 1e-10);
    assert!((result.overall.f1_score() - 0.0).abs() < 1e-10);
}

#[test]
fn f1_score_harmonic_mean() {
    let sovereign = vec![
        make_call(100, VariantType::Snp),
        make_call(200, VariantType::Snp),
        make_call(999, VariantType::Snp), // FP
    ];
    let reference = vec![
        make_ref_snp(100),
        make_ref_snp(200),
        make_ref_snp(300), // FN
    ];

    let result = cross_validate(&sovereign, &reference, 5);
    // sensitivity = 2/3, precision = 2/3, F1 = 2/3
    let expected_f1 = 2.0 / 3.0;
    assert!(
        (result.overall.f1_score() - expected_f1).abs() < 1e-10,
        "F1 = {}, expected {expected_f1}",
        result.overall.f1_score()
    );
}

#[test]
fn config_for_generation_varies() {
    let cfg_early = config_for_generation(0);
    let cfg_mid = config_for_generation(5_000);
    let cfg_late = config_for_generation(50_000);

    assert!(
        cfg_early.min_alt_frequency < cfg_mid.min_alt_frequency,
        "early < mid: {} vs {}",
        cfg_early.min_alt_frequency,
        cfg_mid.min_alt_frequency
    );
    assert!(
        cfg_mid.min_alt_frequency < cfg_late.min_alt_frequency,
        "mid < late: {} vs {}",
        cfg_mid.min_alt_frequency,
        cfg_late.min_alt_frequency
    );
    assert!(cfg_early.quality_weighted);
    assert!(cfg_late.binomial_quality);
}

#[test]
fn unsupported_reference_types_do_not_affect_concordance() {
    let sovereign = vec![make_call(100, VariantType::Snp)];
    let reference = vec![
        make_ref_snp(100),
        ("MOB".to_string(), 200, String::new()),
        ("AMP".to_string(), 300, String::new()),
        ("CON".to_string(), 400, String::new()),
        ("INV".to_string(), 500, String::new()),
    ];

    let result = cross_validate(&sovereign, &reference, 5);
    assert_eq!(result.overall.true_positives, 1);
    assert_eq!(result.overall.false_negatives, 0);
    assert_eq!(result.overall.false_positives, 0);
    assert_eq!(result.snp.reference_total(), 1);
}

#[test]
fn compare_threshold_strategies_returns_both() {
    let sovereign = vec![make_call(100, VariantType::Snp)];
    let reference = vec![make_ref_snp(100)];

    let (gen_aware, fixed) = compare_threshold_strategies(&sovereign, &sovereign, &reference, 5);
    assert_eq!(gen_aware.overall.true_positives, 1);
    assert_eq!(fixed.overall.true_positives, 1);
}
