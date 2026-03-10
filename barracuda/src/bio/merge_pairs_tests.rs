// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used)]

use super::*;

fn make_record(id: &str, seq: &[u8], qual: &[u8]) -> FastqRecord {
    FastqRecord {
        id: id.to_string(),
        sequence: seq.to_vec(),
        quality: qual.to_vec(),
    }
}

fn phred(scores: &[u8]) -> Vec<u8> {
    scores.iter().map(|&q| q + 33).collect()
}

#[test]
fn reverse_complement_basic() {
    assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
    assert_eq!(reverse_complement(b"AAAA"), b"TTTT");
    assert_eq!(reverse_complement(b"GCTA"), b"TAGC");
    assert_eq!(reverse_complement(b"ACGN"), b"NCGT");
}

#[test]
fn reverse_complement_case_insensitive() {
    assert_eq!(reverse_complement(b"acgt"), b"ACGT");
}

#[test]
fn merge_perfect_overlap() {
    // 30bp forward, 30bp reverse with 15bp perfect overlap
    let shared = b"ACGTACGTACGTACG"; // 15bp overlap
    let fwd_seq: Vec<u8> = [b"AAAAAAAAAAAAAAAA" as &[u8], shared].concat(); // 16 + 15 = 31
    let rev_seq_rc: Vec<u8> = [shared as &[u8], b"TTTTTTTTTTTTTTTT"].concat(); // 15 + 16 = 31
    // Reverse of the reverse-complement = what the sequencer gives
    let rev_seq = reverse_complement(&rev_seq_rc);

    let fwd = make_record("fwd", &fwd_seq, &phred(&vec![30; fwd_seq.len()]));
    let rev = make_record("rev", &rev_seq, &phred(&vec![30; rev_seq.len()]));

    let result = merge_pair(&fwd, &rev, &MergeParams::default());

    assert!(result.merged.is_some(), "Should merge successfully");
    let merged = result.merged.unwrap();
    assert_eq!(result.overlap, 15);
    assert_eq!(result.mismatches, 0);
    // Merged = 16 prefix + 15 overlap + 16 suffix = 47
    assert_eq!(merged.sequence.len(), 47, "len = {}", merged.sequence.len());
}

#[test]
fn merge_with_mismatches() {
    // Overlap with 1 mismatch
    let fwd_seq = b"AAAAAAAAAAACGTACGT"; // 18bp
    let rev_seq_rc = b"ACGTACGTTTTTTTTTT"; // first 8 overlap; note T vs A at position 8 from end
    let rev_seq = reverse_complement(rev_seq_rc);

    // Introduce a mismatch: change one base in the overlap
    let mut fwd_mod = fwd_seq.to_vec();
    fwd_mod[10] = b'T'; // was A, should be A in overlap

    let fwd = make_record("fwd", &fwd_mod, &phred(&vec![30; fwd_mod.len()]));
    let rev = make_record("rev", &rev_seq, &phred(&vec![30; rev_seq.len()]));

    let params = MergeParams {
        min_overlap: 5,
        max_mismatches: 3,
        ..MergeParams::default()
    };

    let result = merge_pair(&fwd, &rev, &params);
    assert!(result.merged.is_some());
    assert!(result.mismatches <= 3);
}

#[test]
fn merge_no_overlap() {
    // Completely different sequences
    let fwd = make_record("fwd", b"AAAAAAAAAAAAAAAA", &phred(&[30; 16]));
    let rev = make_record("rev", b"CCCCCCCCCCCCCCCC", &phred(&[30; 16]));

    let result = merge_pair(&fwd, &rev, &MergeParams::default());
    assert!(result.merged.is_none());
}

#[test]
fn merge_too_short_overlap() {
    // Only 5bp overlap, but min_overlap = 10
    let fwd_seq = b"AAAAAAAAAAACGTA"; // 15bp
    let rev_rc = b"ACGTATTTTTTTTTT"; // 5bp overlap
    let rev_seq = reverse_complement(rev_rc);

    let fwd = make_record("fwd", fwd_seq, &phred(&[30; 15]));
    let rev = make_record("rev", &rev_seq, &phred(&vec![30; rev_rc.len()]));

    let params = MergeParams {
        min_overlap: 10,
        ..MergeParams::default()
    };
    let result = merge_pair(&fwd, &rev, &params);
    assert!(result.merged.is_none());
}

#[test]
fn merge_quality_based_resolution() {
    // Overlap with a mismatch; higher-quality base should win
    let overlap = 10;
    let prefix = b"AAAAAAAAAA"; // 10bp prefix

    let fwd_seq: Vec<u8> = [prefix as &[u8], b"ACGTACGTAC"].concat(); // 20bp
    let mut rev_rc_seq = b"ACGTACGTACTTTTTTTTTT".to_vec(); // 10bp overlap + 10bp suffix
    rev_rc_seq[0] = b'T'; // mismatch at overlap position 0: fwd=A, rev_rc=T
    let rev_seq = reverse_complement(&rev_rc_seq);

    let mut fwd_qual = phred(&[30; 20]);
    fwd_qual[10] = 33 + 35; // High quality for fwd at mismatch position

    let mut rev_qual_rc = phred(&[30; 20]);
    rev_qual_rc[0] = 33 + 5; // Low quality for rev at mismatch position
    let rev_qual: Vec<u8> = rev_qual_rc.iter().rev().copied().collect();

    let fwd = make_record("fwd", &fwd_seq, &fwd_qual);
    let rev = make_record("rev", &rev_seq, &rev_qual);

    let params = MergeParams {
        min_overlap: overlap,
        max_mismatches: 5,
        ..MergeParams::default()
    };

    let result = merge_pair(&fwd, &rev, &params);
    assert!(result.merged.is_some());
    let merged = result.merged.unwrap();

    // At the mismatch position (merged index 10): should take fwd 'A' (Q35 > Q5)
    assert_eq!(merged.sequence[10], b'A');
}

#[test]
fn merge_batch() {
    let overlap_seq = b"ACGTACGTAC"; // 10bp
    let fwd_prefix = b"TTTTTTTTTTTTTTTTTTTT"; // 20bp
    let rev_suffix = b"GGGGGGGGGGGGGGGGGGGG"; // 20bp

    let fwd_seq: Vec<u8> = [fwd_prefix as &[u8], overlap_seq].concat(); // 30bp
    let rev_rc: Vec<u8> = [overlap_seq as &[u8], rev_suffix].concat(); // 30bp
    let rev_seq = reverse_complement(&rev_rc);

    let fwd = make_record("pair1", &fwd_seq, &phred(&[30; 30]));
    let rev = make_record("pair1", &rev_seq, &phred(&[30; 30]));

    let (merged, stats) = merge_pairs(
        &[fwd.clone(), fwd],
        &[rev.clone(), rev],
        &MergeParams::default(),
    );

    assert_eq!(stats.input_pairs, 2);
    assert_eq!(stats.merged_count, 2);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].sequence.len(), 50); // 20 + 10 + 20
}

#[test]
fn posterior_quality_agree_increases() {
    let q_agree = posterior_quality_agree(33 + 30, 33 + 30, 33);
    // Two Q30 bases agreeing should produce > Q30
    assert!(
        q_agree > 33 + 30,
        "agree should boost quality: {}",
        q_agree - 33
    );
}

#[test]
fn posterior_quality_disagree_decreases() {
    let q_disagree = posterior_quality_disagree(33 + 30, 33 + 25, 33);
    // The winning base should have lower quality than its original
    assert!(
        q_disagree < 33 + 30,
        "disagree should reduce quality: {}",
        q_disagree - 33
    );
}
