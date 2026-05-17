// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::bio::ref_index::FmIndex;

fn test_reference() -> Vec<u8> {
    // 200bp reference with a unique region
    let mut seq = Vec::with_capacity(200);
    for _ in 0..40 {
        seq.extend_from_slice(b"ACGTA");
    }
    seq
}

fn make_index_and_ref() -> (FmIndex, Vec<u8>) {
    let reference = test_reference();
    let index = FmIndex::build(&reference);
    (index, reference)
}

#[test]
fn map_exact_read() {
    let reference = b"ACGTACGTAAACCCGGGTTTACGTACGT";
    let index = FmIndex::build(reference);

    let read = b"AAACCCGGGTTT";
    let qual = vec![b'I'; read.len()];
    let config = MapperConfig {
        seed_k: 6,
        min_score: 10,
        ..MapperConfig::default()
    };

    let result = map_read("test_read", read, &qual, &index, reference, "ref", &config);
    assert!(result.is_some());
    let rec = result.unwrap();
    assert!(rec.is_mapped());
    assert_eq!(rec.qname, "test_read");
    assert_eq!(rec.rname, "ref");
    assert!(rec.pos > 0); // 1-based
}

#[test]
fn map_unmapped_read() {
    let reference = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    let index = FmIndex::build(reference);

    let read = b"CCCCCCCCCCCCCCCCCCCCCC";
    let qual = vec![b'I'; read.len()];
    let config = MapperConfig {
        seed_k: 8,
        min_score: 20,
        ..MapperConfig::default()
    };

    let result = map_read("unmapped", read, &qual, &index, reference, "ref", &config);
    assert!(result.is_some());
    let rec = result.unwrap();
    assert!(!rec.is_mapped());
    assert_eq!(rec.rname, "*");
}

#[test]
fn map_batch() {
    let reference = b"ACGTACGTAAACCCGGGTTTACGTACGT";
    let index = FmIndex::build(reference);
    let config = MapperConfig {
        seed_k: 6,
        min_score: 10,
        ..MapperConfig::default()
    };

    let reads = vec![
        (
            "r1".to_string(),
            b"AAACCCGGG".to_vec(),
            vec![b'I'; 9],
        ),
        (
            "r2".to_string(),
            b"CCCCCCCCCC".to_vec(),
            vec![b'I'; 10],
        ),
    ];

    let results = map_reads(&reads, &index, reference, "ref", &config);
    assert_eq!(results.len(), 2);
    assert!(results[0].is_mapped());
    // r2 may or may not map depending on score threshold
}

#[test]
fn alignment_to_cigar_pure_match() {
    let alignment = AlignmentResult {
        score: 10,
        aligned_query: b"ACGT".to_vec(),
        aligned_target: b"ACGT".to_vec(),
        query_start: 0,
        target_start: 0,
    };
    let cigar = alignment_to_cigar(&alignment);
    assert_eq!(cigar.len(), 1);
    assert_eq!(cigar[0].op, CigarType::Match);
    assert_eq!(cigar[0].len, 4);
}

#[test]
fn alignment_to_cigar_with_indel() {
    let alignment = AlignmentResult {
        score: 10,
        aligned_query: b"AC-GT".to_vec(),
        aligned_target: b"ACAGT".to_vec(),
        query_start: 0,
        target_start: 0,
    };
    let cigar = alignment_to_cigar(&alignment);
    assert_eq!(cigar.len(), 3);
    assert_eq!(cigar[0].op, CigarType::Match);
    assert_eq!(cigar[0].len, 2);
    assert_eq!(cigar[1].op, CigarType::Deletion);
    assert_eq!(cigar[1].len, 1);
    assert_eq!(cigar[2].op, CigarType::Match);
    assert_eq!(cigar[2].len, 2);
}

#[test]
fn alignment_to_cigar_with_insertion() {
    let alignment = AlignmentResult {
        score: 10,
        aligned_query: b"ACAGT".to_vec(),
        aligned_target: b"AC-GT".to_vec(),
        query_start: 0,
        target_start: 0,
    };
    let cigar = alignment_to_cigar(&alignment);
    assert_eq!(cigar.len(), 3);
    assert_eq!(cigar[0].op, CigarType::Match);
    assert_eq!(cigar[1].op, CigarType::Insertion);
    assert_eq!(cigar[2].op, CigarType::Match);
}

#[test]
fn reverse_complement_works() {
    assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
    assert_eq!(reverse_complement(b"AACG"), b"CGTT");
    assert_eq!(reverse_complement(b""), b"");
    assert_eq!(reverse_complement(b"A"), b"T");
}

#[test]
fn mapq_unique_mapping() {
    let candidates = vec![MappingCandidate {
        ref_start: 0,
        score: 100,
        alignment: AlignmentResult {
            score: 100,
            aligned_query: vec![],
            aligned_target: vec![],
            query_start: 0,
            target_start: 0,
        },
    }];
    assert_eq!(compute_mapq(&candidates), 60);
}

#[test]
fn mapq_ambiguous_mapping() {
    let candidates = vec![
        MappingCandidate {
            ref_start: 0,
            score: 100,
            alignment: AlignmentResult {
                score: 100,
                aligned_query: vec![],
                aligned_target: vec![],
                query_start: 0,
                target_start: 0,
            },
        },
        MappingCandidate {
            ref_start: 1000,
            score: 100,
            alignment: AlignmentResult {
                score: 100,
                aligned_query: vec![],
                aligned_target: vec![],
                query_start: 0,
                target_start: 0,
            },
        },
    ];
    assert_eq!(compute_mapq(&candidates), 0);
}

#[test]
fn mapq_differentiated() {
    let candidates = vec![
        MappingCandidate {
            ref_start: 0,
            score: 100,
            alignment: AlignmentResult {
                score: 100,
                aligned_query: vec![],
                aligned_target: vec![],
                query_start: 0,
                target_start: 0,
            },
        },
        MappingCandidate {
            ref_start: 1000,
            score: 50,
            alignment: AlignmentResult {
                score: 50,
                aligned_query: vec![],
                aligned_target: vec![],
                query_start: 0,
                target_start: 0,
            },
        },
    ];
    let mapq = compute_mapq(&candidates);
    assert!(mapq > 0);
    assert!(mapq <= 60);
}
