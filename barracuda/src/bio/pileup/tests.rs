// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::io::sam::{CigarOp, CigarType, SamRecord, FLAG_REVERSE, FLAG_UNMAPPED};

fn simple_record(pos: u64, seq: &[u8], cigar_len: u32) -> SamRecord {
    SamRecord {
        qname: "r1".into(),
        flag: 0,
        rname: "ref".into(),
        pos,
        mapq: 60,
        cigar: vec![CigarOp {
            len: cigar_len,
            op: CigarType::Match,
        }],
        rnext: "*".into(),
        pnext: 0,
        tlen: 0,
        seq: seq.to_vec(),
        qual: vec![b'I'; seq.len()], // Q40
    }
}

#[test]
fn single_read_pileup() {
    let records = vec![simple_record(1, b"ACGT", 4)];
    let pileup = generate_pileup(&records, 10);
    assert_eq!(pileup.len(), 4);
    assert_eq!(pileup[0].position, 0);
    assert_eq!(pileup[0].depth, 1);
    assert_eq!(pileup[0].base_counts[0], 1); // A
    assert_eq!(pileup[1].base_counts[1], 1); // C
    assert_eq!(pileup[2].base_counts[2], 1); // G
    assert_eq!(pileup[3].base_counts[3], 1); // T
}

#[test]
fn overlapping_reads() {
    let records = vec![
        simple_record(1, b"ACGT", 4),
        simple_record(3, b"GTAA", 4),
    ];
    let pileup = generate_pileup(&records, 10);

    // Position 2 (0-based) should have depth 2 (overlap of "GT")
    let pos2 = pileup.iter().find(|c| c.position == 2).unwrap();
    assert_eq!(pos2.depth, 2);
    assert_eq!(pos2.base_counts[2], 2); // both reads have G at this position
}

#[test]
fn unmapped_reads_skipped() {
    let records = vec![SamRecord {
        qname: "unmapped".into(),
        flag: FLAG_UNMAPPED,
        rname: "*".into(),
        pos: 0,
        mapq: 0,
        cigar: vec![],
        rnext: "*".into(),
        pnext: 0,
        tlen: 0,
        seq: b"ACGT".to_vec(),
        qual: b"IIII".to_vec(),
    }];
    let pileup = generate_pileup(&records, 10);
    assert!(pileup.is_empty());
}

#[test]
fn strand_tracking() {
    let mut rev_record = simple_record(1, b"ACGT", 4);
    rev_record.flag = FLAG_REVERSE;

    let records = vec![simple_record(1, b"ACGT", 4), rev_record];
    let pileup = generate_pileup(&records, 10);

    assert_eq!(pileup[0].depth, 2);
    assert_eq!(pileup[0].forward_depth, 1);
    assert_eq!(pileup[0].reverse_depth, 1);
    assert!((pileup[0].strand_bias() - 0.5).abs() < 1e-10);
}

#[test]
fn deletion_in_cigar() {
    let records = vec![SamRecord {
        qname: "r1".into(),
        flag: 0,
        rname: "ref".into(),
        pos: 1,
        mapq: 60,
        cigar: vec![
            CigarOp { len: 2, op: CigarType::Match },
            CigarOp { len: 3, op: CigarType::Deletion },
            CigarOp { len: 2, op: CigarType::Match },
        ],
        rnext: "*".into(),
        pnext: 0,
        tlen: 0,
        seq: b"ACGT".to_vec(),
        qual: b"IIII".to_vec(),
    }];
    let pileup = generate_pileup(&records, 10);

    // Positions 0,1 have depth 1 (first 2M)
    // Positions 2,3,4 have deletions but no bases
    // Positions 5,6 have depth 1 (second 2M)
    let del_pos = pileup.iter().filter(|c| c.deletions > 0).count();
    assert_eq!(del_pos, 3);
}

#[test]
fn insertion_in_cigar() {
    let records = vec![SamRecord {
        qname: "r1".into(),
        flag: 0,
        rname: "ref".into(),
        pos: 1,
        mapq: 60,
        cigar: vec![
            CigarOp { len: 3, op: CigarType::Match },
            CigarOp { len: 2, op: CigarType::Insertion },
            CigarOp { len: 3, op: CigarType::Match },
        ],
        rnext: "*".into(),
        pnext: 0,
        tlen: 0,
        seq: b"ACGTTACG".to_vec(),
        qual: b"IIIIIIII".to_vec(),
    }];
    let pileup = generate_pileup(&records, 10);

    // Position 2 (0-based) should record 1 insertion
    let ins_cols: Vec<_> = pileup.iter().filter(|c| c.insertions > 0).collect();
    assert_eq!(ins_cols.len(), 1);
}

#[test]
fn major_allele() {
    let mut col = PileupColumn::default();
    col.depth = 10;
    col.base_counts = [1, 2, 5, 2, 0]; // G is major
    assert_eq!(col.major_allele(), b'G');
    assert!((col.major_allele_frequency() - 0.5).abs() < 1e-10);
}

#[test]
fn empty_column_defaults() {
    let col = PileupColumn::default();
    assert_eq!(col.depth, 0);
    assert!(col.major_allele_frequency().abs() < 1e-10);
    assert!((col.strand_bias() - 0.5).abs() < 1e-10);
    assert!(col.mean_quality().abs() < 1e-10);
}

#[test]
fn coverage_stats_basic() {
    let records = vec![simple_record(1, b"ACGT", 4)];
    let pileup = generate_pileup(&records, 20);
    let stats = coverage_stats(&pileup, 20);

    assert_eq!(stats.covered_positions, 4);
    assert!((stats.mean_depth - 1.0).abs() < 1e-10);
    assert_eq!(stats.median_depth, 1);
    assert_eq!(stats.max_depth, 1);
    assert!((stats.coverage_fraction - 0.2).abs() < 1e-10);
}

#[test]
fn coverage_stats_empty() {
    let stats = coverage_stats(&[], 100);
    assert_eq!(stats.covered_positions, 0);
    assert!(stats.mean_depth.abs() < 1e-10);
}
