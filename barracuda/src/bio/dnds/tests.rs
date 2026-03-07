// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::tolerances;

#[test]
fn translate_standard_codons() {
    assert_eq!(translate_codon(b"ATG"), Some(b'M'));
    assert_eq!(translate_codon(b"TAA"), Some(b'*'));
    assert_eq!(translate_codon(b"TGG"), Some(b'W'));
    assert_eq!(translate_codon(b"GCT"), Some(b'A'));
}

#[test]
fn identical_sequences_zero() {
    let seq = b"ATGATGATG";
    let result = pairwise_dnds(seq, seq).unwrap();
    assert!((result.dn - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert!((result.ds - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert_eq!(result.omega, None);
}

#[test]
fn synonymous_only_ds_positive() {
    // TTT (Phe) → TTC (Phe): synonymous at position 3
    let seq1 = b"TTTGCTAAA";
    let seq2 = b"TTCGCTAAA";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.ds > 0.0);
    assert!((result.dn - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert_eq!(result.omega, Some(0.0));
}

#[test]
fn nonsynonymous_only_dn_positive() {
    // AAA (Lys) → GAA (Glu): nonsynonymous at position 1
    let seq1 = b"AAAGCTGCT";
    let seq2 = b"GAAGCTGCT";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.dn > 0.0);
}

#[test]
fn mixed_changes() {
    // Multiple codon differences
    let seq1 = b"ATGGCTAAA";
    let seq2 = b"ATGGCCGAA";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.syn_sites > 0.0);
    assert!(result.nonsyn_sites > 0.0);
}

#[test]
fn gap_codons_skipped() {
    let seq1 = b"ATG---GCT";
    let seq2 = b"ATG---GCT";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!((result.dn - 0.0).abs() < tolerances::ANALYTICAL_F64);
}

#[test]
fn unequal_length_error() {
    assert!(pairwise_dnds(b"ATG", b"ATGA").is_err());
}

#[test]
fn not_divisible_by_3_error() {
    assert!(pairwise_dnds(b"AT", b"AT").is_err());
}

#[test]
fn jukes_cantor_zero() {
    assert!((jukes_cantor(0.0) - 0.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn jukes_cantor_small_p() {
    let d = jukes_cantor(0.01);
    assert!(d > 0.0 && d < 0.02);
}

#[test]
fn codon_sites_met() {
    // ATG: all changes are nonsynonymous (Met is unique codon)
    let (s, n) = codon_sites(b"ATG");
    assert!((s - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert!((n - 3.0).abs() < tolerances::ANALYTICAL_F64);
}

#[test]
fn codon_sites_fourfold_degenerate() {
    // GCT (Ala): position 3 is 4-fold degenerate → 1 syn site there
    let (s, _n) = codon_sites(b"GCT");
    assert!(s > 0.5);
}

#[test]
fn deterministic() {
    // Long enough to avoid Jukes-Cantor saturation; few differences
    let seq1 = b"ATGGCTAAATTTGCTGCTGCTGCTGCTGCT";
    let seq2 = b"ATGGCCAAATTTGCTGCTGCTGCTGCCGCT";
    let r1 = pairwise_dnds(seq1, seq2).unwrap();
    let r2 = pairwise_dnds(seq1, seq2).unwrap();
    assert!(
        (r1.dn - r2.dn).abs() < tolerances::ANALYTICAL_F64,
        "dN not deterministic: {} vs {}",
        r1.dn,
        r2.dn
    );
    assert!(
        (r1.ds - r2.ds).abs() < tolerances::ANALYTICAL_F64,
        "dS not deterministic: {} vs {}",
        r1.ds,
        r2.ds
    );
}

#[test]
fn batch_matches_individual() {
    let s1 = b"ATGGCTAAATTT";
    let s2 = b"ATGGCCAAATTT";
    let s3 = b"ATGATGAAA";
    let pairs: Vec<(&[u8], &[u8])> = vec![(s1, s2), (s1, s1), (s3, s3)];
    let batch = pairwise_dnds_batch(&pairs);
    assert_eq!(batch.len(), 3);
    let individual = pairwise_dnds(s1, s2).unwrap();
    let batch_0 = batch[0].as_ref().unwrap();
    assert!((batch_0.dn - individual.dn).abs() < tolerances::EXACT_F64);
    assert!((batch_0.ds - individual.ds).abs() < tolerances::EXACT_F64);
}

#[test]
fn batch_propagates_errors() {
    let ok_pair: (&[u8], &[u8]) = (b"ATGATG", b"CTGATG");
    let bad_pair: (&[u8], &[u8]) = (b"ATG", b"ATGATG");
    let batch = pairwise_dnds_batch(&[ok_pair, bad_pair]);
    assert!(batch[0].is_ok());
    assert!(batch[1].is_err());
}

// ─── Additional edge-case coverage ────────────────────────────────────────

#[test]
fn translate_nonstandard_codon_returns_none() {
    // Ambiguous bases (N, R, Y, etc.) and invalid codons → None
    assert_eq!(translate_codon(b"NNN"), None);
    assert_eq!(translate_codon(b"XXX"), None);
    assert_eq!(translate_codon(b"ZZZ"), None);
}

#[test]
fn translate_partial_codon_len_not_3() {
    assert_eq!(translate_codon(b"AT"), None);
    assert_eq!(translate_codon(b"A"), None);
    assert_eq!(translate_codon(b""), None);
    assert_eq!(translate_codon(b"ATGA"), None);
}

#[test]
fn codon_sites_stop_codon_returns_zero() {
    let (s, n) = codon_sites(b"TAA");
    assert!((s - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert!((n - 0.0).abs() < tolerances::ANALYTICAL_F64);
}

#[test]
fn codon_sites_invalid_codon_returns_zero() {
    let (s, n) = codon_sites(b"NNN");
    assert!((s - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert!((n - 0.0).abs() < tolerances::ANALYTICAL_F64);
}

#[test]
fn sequences_too_short_single_codon() {
    // Single codon (3 bp): valid length, but all sites compare same codon pair
    // Result: dN=0, dS=0 (identical or counted as 0 differences)
    let result = pairwise_dnds(b"ATG", b"ATG").unwrap();
    assert!((result.dn - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert!((result.ds - 0.0).abs() < tolerances::ANALYTICAL_F64);
}

#[test]
fn sequences_minimum_length_two_codons_with_difference() {
    let seq1 = b"ATGAAA";
    let seq2 = b"ATGGAA"; // Lys→Glu at codon 2
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.dn > 0.0);
}

#[test]
fn jukes_cantor_saturation_p_ge_75() {
    let d = jukes_cantor(0.75);
    assert!(d.is_infinite());
    let d = jukes_cantor(1.0);
    assert!(d.is_infinite());
}

#[test]
fn jukes_cantor_negative_p_returns_zero() {
    assert!((jukes_cantor(-0.1) - 0.0).abs() < tolerances::EXACT_F64);
}

#[test]
fn very_divergent_sequences_jc_saturation() {
    // Maximize observed p for JC: need p_n approaching 0.75
    // AAA→GGG (Lys→Gly): 3 different positions, all nonsynonymous
    // With many such codons, p_n can get high; JC returns INF if p_n >= 0.75
    let seq1 = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // 15 codons
    let seq2 = b"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"; // 15 codons
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.dn.is_infinite() || result.dn > 1.0);
}

#[test]
fn synonymous_only_ds_positive_omega_zero() {
    // TTT (Phe)→TTC (Phe): synonymous only; exercises omega = dN/dS = 0 when dS>0
    let seq1 = b"TTTGCTAAA";
    let seq2 = b"TTCGCTAAA";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.ds > 0.0);
    assert!((result.dn - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert_eq!(result.omega, Some(0.0));
}

#[test]
fn nonsynonymous_only_ds_zero_omega_none() {
    // All nonsynonymous changes: dS=0 (no synonymous sites used or p_s=0)
    // When dS ≈ 0, omega is None
    let seq1 = b"ATGATGATG";
    let seq2 = b"GTGGTGGTG"; // Met→Val at every codon
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.dn > 0.0);
    // dS may be 0 or very small; omega = None when dS <= DNDS_OMEGA_GUARD
    if result.ds <= crate::tolerances::DNDS_OMEGA_GUARD {
        assert_eq!(result.omega, None);
    }
}

#[test]
fn count_codon_diffs_two_positions_pathway() {
    // Codons differing at 2 positions: exercises pathway_diffs and permutations
    let seq1 = b"AAAGCT";
    let seq2 = b"GAACCT"; // AAA→GAA (syn?) and GCT→CCT (Ala→Pro, non)
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.syn_diffs + result.nonsyn_diffs > 0.0);
}

#[test]
fn count_codon_diffs_three_positions() {
    let seq1 = b"AAATTT";
    let seq2 = b"GGGCCC";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!(result.syn_diffs + result.nonsyn_diffs > 0.0);
}

#[test]
fn gap_and_dot_codons_skipped() {
    let seq1 = b"ATG...GCT";
    let seq2 = b"ATG...GCT";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!((result.dn - 0.0).abs() < tolerances::ANALYTICAL_F64);
}

#[test]
fn lowercase_bases_handled() {
    let seq1 = b"atggctaaa";
    let seq2 = b"atggctaaa";
    let result = pairwise_dnds(seq1, seq2).unwrap();
    assert!((result.dn - 0.0).abs() < tolerances::ANALYTICAL_F64);
    assert!((result.ds - 0.0).abs() < tolerances::ANALYTICAL_F64);
}
