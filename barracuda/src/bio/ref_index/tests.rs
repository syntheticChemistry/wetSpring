// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn exact_match_single() {
    let reference = b"ACGTACGTACGT";
    let index = FmIndex::build(reference);
    let hits = index.exact_match(b"ACGT");
    assert_eq!(hits.len(), 3);
    assert!(hits.contains(&0));
    assert!(hits.contains(&4));
    assert!(hits.contains(&8));
}

#[test]
fn exact_match_unique() {
    let reference = b"ACGTTTTTGGGG";
    let index = FmIndex::build(reference);
    let hits = index.exact_match(b"ACGTTTTT");
    assert_eq!(hits, vec![0]);
}

#[test]
fn exact_match_no_hit() {
    let reference = b"AAAAAAAAAA";
    let index = FmIndex::build(reference);
    let hits = index.exact_match(b"CG");
    assert!(hits.is_empty());
}

#[test]
fn count_matches() {
    let reference = b"ACGTACGTACGT";
    let index = FmIndex::build(reference);
    assert_eq!(index.count(b"ACGT"), 3);
    assert_eq!(index.count(b"CG"), 3);
    assert_eq!(index.count(b"A"), 3);
}

#[test]
fn empty_pattern() {
    let index = FmIndex::build(b"ACGT");
    assert!(index.exact_match(b"").is_empty());
    assert_eq!(index.count(b""), 0);
}

#[test]
fn single_base_reference() {
    let index = FmIndex::build(b"A");
    let hits = index.exact_match(b"A");
    assert_eq!(hits, vec![0]);
    assert!(index.exact_match(b"C").is_empty());
}

#[test]
fn reference_len() {
    let reference = b"ACGTACGT";
    let index = FmIndex::build(reference);
    assert_eq!(index.reference_len(), 8);
}

#[test]
fn full_reference_match() {
    let reference = b"ACGT";
    let index = FmIndex::build(reference);
    let hits = index.exact_match(b"ACGT");
    assert_eq!(hits, vec![0]);
}

#[test]
fn seed_kmers_basic() {
    let reference = b"ACGTACGTACGT";
    let index = FmIndex::build(reference);
    let seeds = index.seed_kmers(b"ACGTACGT", 4, 100);
    assert!(!seeds.is_empty());
    for (offset, positions) in &seeds {
        assert!(*offset <= 4); // 5 possible 4-mers in an 8bp read
        assert!(!positions.is_empty());
    }
}

#[test]
fn seed_kmers_max_hits_filter() {
    let reference = b"AAAAAAAAAA";
    let index = FmIndex::build(reference);
    let seeds = index.seed_kmers(b"AAAA", 2, 1);
    // "AA" appears many times; max_hits=1 should filter it
    assert!(seeds.is_empty());
}

#[test]
fn repetitive_reference() {
    let reference = b"ATATATATATAT";
    let index = FmIndex::build(reference);
    let hits = index.exact_match(b"AT");
    // "ATATATATATAT" has 11 chars → "AT" at 0,2,4,6,8 and also 9 (last "AT")
    assert_eq!(hits.len(), 6);

    // Verify sorted
    for w in hits.windows(2) {
        assert!(w[0] < w[1]);
    }
}

#[test]
fn case_insensitive() {
    let reference = b"ACGTACGT";
    let index = FmIndex::build(reference);
    let hits_upper = index.exact_match(b"ACGT");
    let hits_lower = index.exact_match(b"acgt");
    assert_eq!(hits_upper, hits_lower);
}

#[test]
fn longer_reference() {
    let mut reference = Vec::with_capacity(1000);
    for _ in 0..250 {
        reference.extend_from_slice(b"ACGT");
    }
    reference.extend_from_slice(b"GGCC");
    let index = FmIndex::build(&reference);
    let hits = index.exact_match(b"GGCC");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0], 1000);

    assert_eq!(index.count(b"ACGT"), 250);
    assert_eq!(index.reference_len(), 1004);
}

#[test]
fn sais_correctness() {
    let text = b"ABRACADABRA";
    let mut text_with_sentinel: Vec<usize> = text.iter().map(|&b| match b {
        b'A' => 1, b'B' => 2, b'C' => 3, b'D' => 4, b'R' => 5, _ => 0,
    }).collect();
    text_with_sentinel.push(0);

    let sa = sais(&text_with_sentinel, 6);
    assert_eq!(sa.len(), text_with_sentinel.len());

    // Verify SA is a permutation
    let mut sorted = sa.clone();
    sorted.sort_unstable();
    let expected: Vec<usize> = (0..text_with_sentinel.len()).collect();
    assert_eq!(sorted, expected);

    // Verify lexicographic ordering
    for w in sa.windows(2) {
        assert!(text_with_sentinel[w[0]..] < text_with_sentinel[w[1]..]);
    }
}
