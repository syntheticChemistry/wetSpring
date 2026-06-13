// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used, reason = "test assertions")]
use super::*;
use crate::bio::ref_index::FmIndex;

#[allow(dead_code)]
fn test_reference() -> Vec<u8> {
    // 200bp reference with a unique region
    let mut seq = Vec::with_capacity(200);
    for _ in 0..40 {
        seq.extend_from_slice(b"ACGTA");
    }
    seq
}

#[allow(dead_code)]
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
    assert_eq!(compute_mapq(&candidates, None), 60);
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
    assert_eq!(compute_mapq(&candidates, None), 0);
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
    let mapq = compute_mapq(&candidates, None);
    assert!(mapq > 0);
    assert!(mapq <= 60);
}

// --- Post-alignment dedup tests ---

fn make_candidate(ref_start: usize, score: i32) -> MappingCandidate {
    MappingCandidate {
        ref_start,
        score,
        alignment: AlignmentResult {
            score,
            aligned_query: vec![],
            aligned_target: vec![],
            query_start: 0,
            target_start: 0,
        },
    }
}

#[test]
fn dedup_removes_nearby_lower_scoring_candidates() {
    let mut candidates = vec![
        make_candidate(100, 80),
        make_candidate(120, 70), // within 50bp of first
        make_candidate(130, 60), // within 50bp of first
    ];
    dedup_candidates(&mut candidates, 50);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].ref_start, 100);
    assert_eq!(candidates[0].score, 80);
}

#[test]
fn dedup_preserves_distant_candidates() {
    let mut candidates = vec![
        make_candidate(100, 80),
        make_candidate(200, 70), // 100bp away, > 50bp distance
        make_candidate(500, 60),
    ];
    dedup_candidates(&mut candidates, 50);
    assert_eq!(candidates.len(), 3);
}

#[test]
fn dedup_no_op_with_single_candidate() {
    let mut candidates = vec![make_candidate(100, 80)];
    dedup_candidates(&mut candidates, 50);
    assert_eq!(candidates.len(), 1);
}

#[test]
fn dedup_disabled_when_distance_zero() {
    let mut candidates = vec![
        make_candidate(100, 80),
        make_candidate(101, 70),
    ];
    dedup_candidates(&mut candidates, 0);
    // distance=0 means abs_diff < 0 is never true, so nothing is deduped
    assert_eq!(candidates.len(), 2);
}

#[test]
fn dedup_repetitive_region_improves_mapq() {
    // Simulate repetitive region: 5 candidates all near position 500
    // with similar scores, plus one distant candidate at position 5000
    let mut candidates_no_dedup = vec![
        make_candidate(500, 80),
        make_candidate(510, 79),
        make_candidate(520, 78),
        make_candidate(530, 77),
        make_candidate(540, 76),
        make_candidate(5000, 50),
    ];
    let mapq_no_dedup = compute_mapq(&candidates_no_dedup, None);

    // After dedup, the cluster collapses to one candidate
    dedup_candidates(&mut candidates_no_dedup, 50);
    let mapq_after_dedup = compute_mapq(&candidates_no_dedup, None);

    assert_eq!(candidates_no_dedup.len(), 2, "cluster should collapse to 1 + distant 1");
    assert!(
        mapq_after_dedup >= mapq_no_dedup,
        "dedup should improve or maintain MAPQ: {mapq_after_dedup} vs {mapq_no_dedup}"
    );
}

#[test]
fn dedup_keeps_best_from_each_cluster() {
    // Two clusters: [100, 120, 140] and [1000, 1020]
    let mut candidates = vec![
        make_candidate(100, 90),
        make_candidate(1000, 85),
        make_candidate(120, 80),
        make_candidate(1020, 75),
        make_candidate(140, 70),
    ];
    dedup_candidates(&mut candidates, 50);
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].ref_start, 100);
    assert_eq!(candidates[0].score, 90);
    assert_eq!(candidates[1].ref_start, 1000);
    assert_eq!(candidates[1].score, 85);
}
