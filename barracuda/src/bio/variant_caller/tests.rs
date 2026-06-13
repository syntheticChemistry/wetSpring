// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used, reason = "test assertions")]
use super::*;
use crate::bio::pileup::PileupColumn;

fn make_snp_column(position: usize, ref_count: u32, alt_count: u32, ref_idx: usize, alt_idx: usize) -> PileupColumn {
    let total = ref_count + alt_count;
    let mut col = PileupColumn {
        position,
        depth: total,
        ..PileupColumn::default()
    };
    col.base_counts[ref_idx] = ref_count;
    col.base_counts[alt_idx] = alt_count;
    // Q30 per base so quality_weighted_freq produces meaningful results
    col.quality_sums[ref_idx] = u64::from(ref_count) * 30;
    col.quality_sums[alt_idx] = u64::from(alt_count) * 30;
    col.forward_depth = total / 2;
    col.reverse_depth = total - col.forward_depth;
    col
}

#[test]
fn call_clear_snp() {
    let reference = b"ACGTACGT";
    let pileup = vec![make_snp_column(0, 2, 48, 0, 3)]; // A→T, 96% alt
    let config = CallerConfig::default();
    let variants = call_variants(&pileup, reference, &[], &config);

    assert_eq!(variants.len(), 1);
    assert_eq!(variants[0].variant_type, VariantType::Snp);
    assert_eq!(variants[0].position, 1); // 1-based
    assert_eq!(variants[0].ref_allele, b'A');
    assert_eq!(variants[0].alt_allele, b'T');
    assert!(variants[0].frequency > 0.9);
    assert!(variants[0].quality > 0.0);
}

#[test]
fn skip_low_frequency() {
    let reference = b"ACGT";
    let pileup = vec![make_snp_column(0, 95, 5, 0, 1)]; // 5% alt
    let config = CallerConfig {
        min_alt_frequency: 0.1,
        ..CallerConfig::permissive()
    };
    let variants = call_variants(&pileup, reference, &[], &config);
    assert!(variants.is_empty());
}

#[test]
fn skip_low_depth() {
    let reference = b"ACGT";
    let pileup = vec![make_snp_column(0, 1, 1, 0, 1)]; // depth 2
    let config = CallerConfig {
        min_depth: 5,
        ..CallerConfig::default()
    };
    let variants = call_variants(&pileup, reference, &[], &config);
    assert!(variants.is_empty());
}

#[test]
fn call_deletion() {
    let reference = b"ACGT";
    let mut col = PileupColumn {
        position: 1,
        depth: 40,
        deletions: 30,
        ..PileupColumn::default()
    };
    col.base_counts[1] = 40;
    col.quality_sums[1] = 40 * 30;
    col.forward_depth = 20;
    col.reverse_depth = 20;

    let config = CallerConfig::default();
    let variants = call_variants(&[col], reference, &[], &config);

    let del = variants.iter().find(|v| v.variant_type == VariantType::Deletion);
    assert!(del.is_some());
    let del = del.unwrap();
    assert_eq!(del.position, 2);
}

#[test]
fn call_insertion() {
    let reference = b"ACGT";
    let mut col = PileupColumn {
        position: 1,
        depth: 50,
        insertions: 20,
        ..PileupColumn::default()
    };
    col.base_counts[1] = 50;
    col.quality_sums[1] = 50 * 30;
    col.forward_depth = 25;
    col.reverse_depth = 25;

    let config = CallerConfig::default();
    let variants = call_variants(&[col], reference, &[], &config);

    let ins = variants.iter().find(|v| v.variant_type == VariantType::Insertion);
    assert!(ins.is_some());
}

#[test]
fn gene_annotation() {
    let reference = b"ACGTACGT";
    let features = vec![GenBankFeature {
        feature_type: "CDS".into(),
        start: 1,
        end: 4,
        forward: true,
        gene: Some("testGene".into()),
        product: None,
        locus_tag: None,
    }];

    let pileup = vec![make_snp_column(1, 2, 48, 1, 3)]; // pos 1 (0-based) in CDS
    let config = CallerConfig::default();
    let variants = call_variants(&pileup, reference, &features, &config);

    assert_eq!(variants.len(), 1);
    assert_eq!(variants[0].gene.as_deref(), Some("testGene"));
}

#[test]
fn gd_line_format() {
    let var = CalledVariant {
        position: 100,
        variant_type: VariantType::Snp,
        ref_allele: b'A',
        alt_allele: b'T',
        depth: 50,
        frequency: 0.96,
        quality: 100.0,
        gene: Some("fooB".into()),
    };
    let line = var.to_gd_line("REL606");
    assert!(line.contains("SNP"));
    assert!(line.contains("REL606"));
    assert!(line.contains("100"));
    assert!(line.contains("fooB"));
}

#[test]
fn parse_gd_file_basic() {
    let gd = "#version    GenomeDiff 1.0\nSNP\t1\t.\tREL606\t100\tA\n\
              DEL\t2\t.\tREL606\t200\t.\n\
              INS\t3\t.\tREL606\t300\t+T\n";
    let mutations = parse_gd_file(gd);
    assert_eq!(mutations.len(), 3);
    assert_eq!(mutations[0], ("SNP".into(), 100, "A".into()));
    assert_eq!(mutations[1], ("DEL".into(), 200, ".".into()));
    assert_eq!(mutations[2], ("INS".into(), 300, "+T".into()));
}

#[test]
fn compare_calls_basic() {
    let sovereign = vec![
        CalledVariant {
            position: 100,
            variant_type: VariantType::Snp,
            ref_allele: b'A',
            alt_allele: b'T',
            depth: 50,
            frequency: 1.0,
            quality: 100.0,
            gene: None,
        },
        CalledVariant {
            position: 200,
            variant_type: VariantType::Snp,
            ref_allele: b'C',
            alt_allele: b'G',
            depth: 50,
            frequency: 1.0,
            quality: 100.0,
            gene: None,
        },
    ];

    let baseline = vec![
        ("SNP".into(), 100, "T".into()),
        ("DEL".into(), 300, ".".into()),
    ];

    let (matches, only_sov, only_base) = compare_calls(&sovereign, &baseline);
    assert_eq!(matches, 1); // position 100 matches
    assert_eq!(only_sov, 1); // position 200 only in sovereign
    assert_eq!(only_base, 1); // position 300 only in baseline
}

#[test]
fn variant_quality_zero_for_noise() {
    let q = variant_quality(1, 1000, 0.001);
    assert!(q.abs() < 1e-10);
}

#[test]
fn variant_quality_positive_for_signal() {
    let q = variant_quality(50, 100, 0.5);
    assert!(q > 10.0);
}

#[test]
fn variant_type_display() {
    assert_eq!(VariantType::Snp.to_string(), "SNP");
    assert_eq!(VariantType::Deletion.to_string(), "DEL");
    assert_eq!(VariantType::Insertion.to_string(), "INS");
}

// ── WS-11: Binomial quality model tests ─────────────────────────

#[test]
fn binomial_quality_high_for_strong_variant() {
    let col = make_snp_column(0, 5, 45, 0, 3); // 90% alt at Q30
    let q = binomial_quality(&col, 3, 45);
    assert!(q > 100.0, "strong variant at Q30 should have high quality: {q}");
}

#[test]
fn binomial_quality_low_for_noise_level() {
    let col = make_snp_column(0, 99, 1, 0, 1); // 1% alt at Q30
    let q = binomial_quality(&col, 1, 1);
    // Q30 error rate is 0.001, seeing 1/100 at 1% is still above error
    // but should be much lower quality than a strong variant
    assert!(q < 50.0, "noise-level variant should have moderate quality: {q}");
}

#[test]
fn binomial_quality_zero_when_at_error_rate() {
    // Construct a column with Q10 quality (10% error rate)
    let mut col = PileupColumn {
        position: 0,
        depth: 100,
        ..PileupColumn::default()
    };
    col.base_counts[0] = 90; // ref A
    col.base_counts[1] = 10; // alt C
    col.quality_sums[0] = 90 * 10; // Q10 ref
    col.quality_sums[1] = 10 * 10; // Q10 alt = 10% error rate
    col.forward_depth = 50;
    col.reverse_depth = 50;

    let q = binomial_quality(&col, 1, 10);
    // At Q10, p_err = 0.1, seeing 10% is expected under null
    assert!(q < 5.0, "variant at error rate should have near-zero quality: {q}");
}

#[test]
fn binomial_quality_zero_for_empty_column() {
    let col = PileupColumn::default();
    let q = binomial_quality(&col, 0, 0);
    assert!((q - 0.0).abs() < f64::EPSILON);
}

#[test]
fn binomial_quality_scales_with_depth() {
    // Same 80% alt frequency but different depths
    let shallow = make_snp_column(0, 4, 16, 0, 3); // depth 20
    let deep = make_snp_column(0, 20, 80, 0, 3); // depth 100
    let q_shallow = binomial_quality(&shallow, 3, 16);
    let q_deep = binomial_quality(&deep, 3, 80);
    assert!(
        q_deep > q_shallow,
        "deeper coverage should yield higher quality: {q_deep} vs {q_shallow}"
    );
}

#[test]
fn binomial_model_calls_same_clear_variant() {
    let reference = b"ACGTACGT";
    let pileup = vec![make_snp_column(0, 2, 48, 0, 3)]; // A→T, 96% alt

    // Default config uses binomial
    let config_binom = CallerConfig::default();
    assert!(config_binom.binomial_quality);
    let variants = call_variants(&pileup, reference, &[], &config_binom);
    assert_eq!(variants.len(), 1, "binomial model should call clear variant");

    // Legacy model
    let config_legacy = CallerConfig {
        binomial_quality: false,
        ..CallerConfig::default()
    };
    let variants_legacy = call_variants(&pileup, reference, &[], &config_legacy);
    assert_eq!(variants_legacy.len(), 1, "legacy model should also call it");
}

#[test]
fn binomial_model_suppresses_low_quality_noise() {
    // Low quality (Q5) alt bases — error rate ~31.6%
    let mut col = PileupColumn {
        position: 0,
        depth: 50,
        ..PileupColumn::default()
    };
    col.base_counts[0] = 35; // ref A
    col.base_counts[1] = 15; // alt C (30%)
    col.quality_sums[0] = 35 * 30; // Q30 ref
    col.quality_sums[1] = 15 * 5; // Q5 alt — very low quality
    col.forward_depth = 25;
    col.reverse_depth = 25;

    let reference = b"A";

    // With binomial model: Q5 error rate is ~0.316, so 30% alt is noise
    let config_binom = CallerConfig {
        min_alt_frequency: 0.1,
        min_alt_count: 3,
        binomial_quality: true,
        ..CallerConfig::default()
    };
    let variants = call_variants(&[col], reference, &[], &config_binom);
    // Binomial should give very low quality for this
    if !variants.is_empty() {
        assert!(
            variants[0].quality < 20.0,
            "Q5 noise should be low quality with binomial: {}",
            variants[0].quality
        );
    }
}

#[test]
fn log_gamma_basic_values() {
    // Γ(1) = 1, ln(1) = 0
    assert!((log_gamma(1.0) - 0.0).abs() < 0.01);
    // Γ(2) = 1, ln(1) = 0
    assert!((log_gamma(2.0) - 0.0).abs() < 0.01);
    // Γ(5) = 24, ln(24) ≈ 3.178
    assert!((log_gamma(5.0) - 24.0_f64.ln()).abs() < 0.01);
    // Γ(10) = 362880, ln(362880) ≈ 12.80
    assert!((log_gamma(10.0) - 362_880.0_f64.ln()).abs() < 0.01);
}

#[test]
fn binomial_log_sf_trivial() {
    // P(X >= 0) = 1, ln(1) = 0
    let p = binomial_log_sf(0, 10, 0.5);
    assert!(
        p.exp_m1().abs() < 0.1,
        "P(X>=0) should be ~1: {}",
        p.exp()
    );
}

#[test]
fn binomial_log_sf_extreme() {
    // P(X >= 10) where X ~ Bin(10, 0.001) should be vanishingly small
    let p = binomial_log_sf(10, 10, 0.001);
    assert!(p < -50.0, "P(X>=10|n=10,p=0.001) should be near-zero: {p}");
}

// ── WS-11: MAPQ-aware binomial quality tests ────────────────────

#[test]
fn binomial_quality_mapq_reduces_quality_for_low_mapq() {
    // Same base quality (Q30) but low MAPQ (Q10 = 10% mapping error)
    let mut col = make_snp_column(0, 5, 45, 0, 3); // 90% alt at Q30
    // Add low MAPQ to alt bases: Q10 mapping quality
    col.mapq_sums[3] = 45 * 10; // alt bases have MAPQ 10

    let q_low_mapq = binomial_quality(&col, 3, 45);

    // Compare against no MAPQ (zero mapq_sums = fallback to base-only)
    let mut col_no_mapq = make_snp_column(0, 5, 45, 0, 3);
    col_no_mapq.mapq_sums = [0; 5]; // explicit: no MAPQ data
    let q_no_mapq = binomial_quality(&col_no_mapq, 3, 45);

    assert!(
        q_low_mapq < q_no_mapq,
        "low MAPQ should reduce quality: {q_low_mapq} vs {q_no_mapq} (base-only)"
    );
}

#[test]
fn binomial_quality_mapq_high_mapq_negligible_effect() {
    // Q30 base + Q60 MAPQ: mapping error is negligible (1e-6)
    let mut col = make_snp_column(0, 5, 45, 0, 3);
    col.mapq_sums[3] = 45 * 60; // alt bases have MAPQ 60

    let q_high_mapq = binomial_quality(&col, 3, 45);

    let mut col_no_mapq = make_snp_column(0, 5, 45, 0, 3);
    col_no_mapq.mapq_sums = [0; 5];
    let q_no_mapq = binomial_quality(&col_no_mapq, 3, 45);

    // High MAPQ should barely change the quality (within ~1%)
    let ratio = q_high_mapq / q_no_mapq;
    assert!(
        ratio > 0.95,
        "high MAPQ should have negligible effect: ratio={ratio:.4} ({q_high_mapq} vs {q_no_mapq})"
    );
}

#[test]
fn binomial_quality_mapq_zero_suppresses_with_low_mapq() {
    // Q30 base quality but MAPQ 0 (completely unconfident mapping)
    // Combined error ≈ 0.001 + 1.0 - 0.001 = ~1.0
    let mut col = make_snp_column(0, 50, 50, 0, 3);
    // MAPQ 0 means mapq_sums = 0 which triggers the fallback to base-only.
    // Use MAPQ 1 to test very-low-MAPQ behavior: P(map_err) = 10^(-0.1) ≈ 0.79
    col.mapq_sums[3] = 50; // 50 reads × MAPQ 1

    let q = binomial_quality(&col, 3, 50);
    // With P(map_err) ≈ 0.79, combined P(err) is very high → quality near zero
    assert!(q < 5.0, "MAPQ 1 should suppress variant quality: {q}");
}

#[test]
fn pileup_column_mean_mapq() {
    let mut col = PileupColumn::default();
    col.base_counts[2] = 10; // 10 G bases
    col.mapq_sums[2] = 10 * 40; // mean MAPQ 40
    assert!((col.mean_mapq(2) - 40.0).abs() < f64::EPSILON);
    assert!((col.mean_mapq(0) - 0.0).abs() < f64::EPSILON); // no A bases
}
