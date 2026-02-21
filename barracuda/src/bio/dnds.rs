// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pairwise dN/dS estimation via the Nei-Gojobori (1986) method.
//!
//! dN/dS (omega) measures the ratio of nonsynonymous (amino-acid changing)
//! to synonymous (silent) substitution rates. Values:
//! - omega < 1: purifying selection (most genes)
//! - omega ≈ 1: neutral evolution
//! - omega > 1: positive selection
//!
//! Reference: Nei & Gojobori (1986) Mol Biol Evol 3:418-426.

use crate::error::{Error, Result};

/// Standard genetic code: codon → amino acid (single letter).
///
/// Stop codons map to `'*'`.
fn translate_codon(codon: &[u8]) -> Option<u8> {
    if codon.len() != 3 {
        return None;
    }
    let c = [
        codon[0].to_ascii_uppercase(),
        codon[1].to_ascii_uppercase(),
        codon[2].to_ascii_uppercase(),
    ];
    Some(match &c {
        b"TTT" | b"TTC" => b'F',
        b"TTA" | b"TTG" | b"CTT" | b"CTC" | b"CTA" | b"CTG" => b'L',
        b"ATT" | b"ATC" | b"ATA" => b'I',
        b"ATG" => b'M',
        b"GTT" | b"GTC" | b"GTA" | b"GTG" => b'V',
        b"TCT" | b"TCC" | b"TCA" | b"TCG" | b"AGT" | b"AGC" => b'S',
        b"CCT" | b"CCC" | b"CCA" | b"CCG" => b'P',
        b"ACT" | b"ACC" | b"ACA" | b"ACG" => b'T',
        b"GCT" | b"GCC" | b"GCA" | b"GCG" => b'A',
        b"TAT" | b"TAC" => b'Y',
        b"TAA" | b"TAG" | b"TGA" => b'*',
        b"CAT" | b"CAC" => b'H',
        b"CAA" | b"CAG" => b'Q',
        b"AAT" | b"AAC" => b'N',
        b"AAA" | b"AAG" => b'K',
        b"GAT" | b"GAC" => b'D',
        b"GAA" | b"GAG" => b'E',
        b"TGT" | b"TGC" => b'C',
        b"TGG" => b'W',
        b"CGT" | b"CGC" | b"CGA" | b"CGG" | b"AGA" | b"AGG" => b'R',
        b"GGT" | b"GGC" | b"GGA" | b"GGG" => b'G',
        _ => return None,
    })
}

const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// Count synonymous and nonsynonymous sites for a single codon.
///
/// Uses the Nei-Gojobori method: for each position, count how many of
/// the 3 possible single-nucleotide changes are synonymous.
fn codon_sites(codon: &[u8]) -> (f64, f64) {
    let Some(orig_aa) = translate_codon(codon) else {
        return (0.0, 0.0);
    };
    if orig_aa == b'*' {
        return (0.0, 0.0);
    }

    let mut syn_sites = 0.0;
    for pos in 0..3 {
        let mut syn_changes = 0;
        let mut total_changes = 0;
        for &base in &BASES {
            if base == codon[pos].to_ascii_uppercase() {
                continue;
            }
            let mut mutant = [
                codon[0].to_ascii_uppercase(),
                codon[1].to_ascii_uppercase(),
                codon[2].to_ascii_uppercase(),
            ];
            mutant[pos] = base;
            total_changes += 1;
            if let Some(new_aa) = translate_codon(&mutant) {
                if new_aa != b'*' && new_aa == orig_aa {
                    syn_changes += 1;
                }
            }
        }
        if total_changes > 0 {
            syn_sites += f64::from(syn_changes) / f64::from(total_changes);
        }
    }

    (syn_sites, 3.0 - syn_sites)
}

/// Result of pairwise dN/dS estimation.
#[derive(Debug, Clone)]
pub struct DnDsResult {
    /// dN: nonsynonymous substitutions per nonsynonymous site.
    pub dn: f64,
    /// dS: synonymous substitutions per synonymous site.
    pub ds: f64,
    /// omega = dN/dS (None if dS ≈ 0).
    pub omega: Option<f64>,
    /// Total synonymous sites.
    pub syn_sites: f64,
    /// Total nonsynonymous sites.
    pub nonsyn_sites: f64,
    /// Observed synonymous differences.
    pub syn_diffs: f64,
    /// Observed nonsynonymous differences.
    pub nonsyn_diffs: f64,
}

/// GPU uniform parameters for batched dN/dS dispatch.
///
/// Maps directly to WGSL `var<uniform>` for future GPU absorption.
/// Each thread processes one sequence pair independently.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DnDsParams {
    /// Number of sequence pairs to process.
    pub n_pairs: u32,
    /// Sequence length in codons (all pairs must be same length).
    pub n_codons: u32,
}

/// Compute dN/dS for a batch of sequence pairs (GPU-friendly API).
///
/// Each pair is processed independently — maps to one GPU thread per pair.
/// Returns `Ok` results only; pairs that fail validation are `Err` in the Vec.
#[must_use]
pub fn pairwise_dnds_batch(pairs: &[(&[u8], &[u8])]) -> Vec<Result<DnDsResult>> {
    pairs.iter().map(|(s1, s2)| pairwise_dnds(s1, s2)).collect()
}

/// Pairwise dN/dS via Nei-Gojobori (1986).
///
/// Both sequences must be coding DNA of equal length divisible by 3.
/// Gap characters (`-` or `.`) in either sequence cause that codon to
/// be skipped.
///
/// # Errors
///
/// Returns `Error::InvalidInput` if sequences differ in length or if
/// the length is not divisible by 3.
pub fn pairwise_dnds(seq1: &[u8], seq2: &[u8]) -> Result<DnDsResult> {
    if seq1.len() != seq2.len() {
        return Err(Error::InvalidInput(
            "dN/dS: sequences must be equal length".into(),
        ));
    }
    if seq1.len() % 3 != 0 {
        return Err(Error::InvalidInput(
            "dN/dS: sequence length must be divisible by 3".into(),
        ));
    }

    let mut total_syn_sites = 0.0;
    let mut total_nonsyn_sites = 0.0;
    let mut total_syn_diffs = 0.0;
    let mut total_nonsyn_diffs = 0.0;

    for i in (0..seq1.len()).step_by(3) {
        let c1 = &seq1[i..i + 3];
        let c2 = &seq2[i..i + 3];

        if c1.iter().any(|&b| b == b'-' || b == b'.') || c2.iter().any(|&b| b == b'-' || b == b'.')
        {
            continue;
        }

        let (s1_syn, s1_non) = codon_sites(c1);
        let (s2_syn, s2_non) = codon_sites(c2);
        total_syn_sites += (s1_syn + s2_syn) / 2.0;
        total_nonsyn_sites += (s1_non + s2_non) / 2.0;

        let diffs = count_codon_diffs(c1, c2);
        total_syn_diffs += diffs.0;
        total_nonsyn_diffs += diffs.1;
    }

    let p_s = if total_syn_sites > 0.0 {
        total_syn_diffs / total_syn_sites
    } else {
        0.0
    };
    let p_n = if total_nonsyn_sites > 0.0 {
        total_nonsyn_diffs / total_nonsyn_sites
    } else {
        0.0
    };

    let ds = jukes_cantor(p_s);
    let dn = jukes_cantor(p_n);

    let omega = if ds > 1e-10 { Some(dn / ds) } else { None };

    Ok(DnDsResult {
        dn,
        ds,
        omega,
        syn_sites: total_syn_sites,
        nonsyn_sites: total_nonsyn_sites,
        syn_diffs: total_syn_diffs,
        nonsyn_diffs: total_nonsyn_diffs,
    })
}

/// Count synonymous and nonsynonymous differences between two codons.
///
/// For codons differing at 1 site, classification is direct.
/// For 2-3 site differences, averages over all shortest pathways.
fn count_codon_diffs(c1: &[u8], c2: &[u8]) -> (f64, f64) {
    let c1_up = [
        c1[0].to_ascii_uppercase(),
        c1[1].to_ascii_uppercase(),
        c1[2].to_ascii_uppercase(),
    ];
    let c2_up = [
        c2[0].to_ascii_uppercase(),
        c2[1].to_ascii_uppercase(),
        c2[2].to_ascii_uppercase(),
    ];

    let diff_positions: Vec<usize> = (0..3).filter(|&i| c1_up[i] != c2_up[i]).collect();
    let n_diffs = diff_positions.len();

    if n_diffs == 0 {
        return (0.0, 0.0);
    }

    if n_diffs == 1 {
        let aa1 = translate_codon(&c1_up);
        let aa2 = translate_codon(&c2_up);
        return match (aa1, aa2) {
            (Some(a), Some(b)) if a == b => (1.0, 0.0),
            _ => (0.0, 1.0),
        };
    }

    // For 2+ differences: average over all permutations of mutation order
    let perms = permutations(&diff_positions);
    let mut total_syn = 0.0;
    let mut total_non = 0.0;

    for perm in &perms {
        let (s, n) = pathway_diffs(c1_up, c2_up, perm);
        total_syn += s;
        total_non += n;
    }

    let count = f64::from(u32::try_from(perms.len()).unwrap_or(u32::MAX));
    (total_syn / count, total_non / count)
}

/// Walk a single pathway from `c1` to `c2`, mutating positions in the given order.
fn pathway_diffs(c1: [u8; 3], c2: [u8; 3], order: &[usize]) -> (f64, f64) {
    let mut current = c1;
    let mut syn = 0.0;
    let mut non = 0.0;

    for &pos in order {
        let aa_before = translate_codon(&current);
        current[pos] = c2[pos];
        let aa_after = translate_codon(&current);
        match (aa_before, aa_after) {
            (Some(a), Some(b)) if a == b => syn += 1.0,
            _ => non += 1.0,
        }
    }
    (syn, non)
}

fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        let rest: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &v)| v)
            .collect();
        for mut perm in permutations(&rest) {
            perm.insert(0, item);
            result.push(perm);
        }
    }
    result
}

/// Jukes-Cantor correction: d = -(3/4) * ln(1 - 4p/3).
///
/// Returns 0 if p is 0, and `f64::INFINITY` if p >= 0.75 (saturation).
fn jukes_cantor(p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    let arg = 1.0 - 4.0 * p / 3.0;
    if arg <= 0.0 {
        return f64::INFINITY;
    }
    -0.75 * arg.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((result.dn - 0.0).abs() < 1e-12);
        assert!((result.ds - 0.0).abs() < 1e-12);
        assert_eq!(result.omega, None);
    }

    #[test]
    fn synonymous_only_ds_positive() {
        // TTT (Phe) → TTC (Phe): synonymous at position 3
        let seq1 = b"TTTGCTAAA";
        let seq2 = b"TTCGCTAAA";
        let result = pairwise_dnds(seq1, seq2).unwrap();
        assert!(result.ds > 0.0);
        assert!((result.dn - 0.0).abs() < 1e-12);
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
        assert!((result.dn - 0.0).abs() < 1e-12);
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
        assert!((jukes_cantor(0.0) - 0.0).abs() < 1e-15);
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
        assert!((s - 0.0).abs() < 1e-12);
        assert!((n - 3.0).abs() < 1e-12);
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
            (r1.dn - r2.dn).abs() < 1e-12,
            "dN not deterministic: {} vs {}",
            r1.dn,
            r2.dn
        );
        assert!(
            (r1.ds - r2.ds).abs() < 1e-12,
            "dS not deterministic: {} vs {}",
            r1.ds,
            r2.ds
        );
    }
}
