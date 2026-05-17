// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::bio::pileup::PileupColumn;

fn make_snp_column(position: usize, ref_count: u32, alt_count: u32, ref_idx: usize, alt_idx: usize) -> PileupColumn {
    let mut col = PileupColumn {
        position,
        depth: ref_count + alt_count,
        ..PileupColumn::default()
    };
    col.base_counts[ref_idx] = ref_count;
    col.base_counts[alt_idx] = alt_count;
    col.forward_depth = (ref_count + alt_count) / 2;
    col.reverse_depth = (ref_count + alt_count) - col.forward_depth;
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
        ..CallerConfig::default()
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
    col.base_counts[1] = 40; // C
    col.forward_depth = 20;
    col.reverse_depth = 20;

    let config = CallerConfig::default();
    let variants = call_variants(&[col], reference, &[], &config);

    let del = variants.iter().find(|v| v.variant_type == VariantType::Deletion);
    assert!(del.is_some());
    let del = del.unwrap();
    assert_eq!(del.position, 2); // 1-based
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
