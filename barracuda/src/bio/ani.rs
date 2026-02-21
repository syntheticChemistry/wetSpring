// SPDX-License-Identifier: AGPL-3.0-or-later
//! Average Nucleotide Identity (ANI) for genome comparison.
//!
//! ANI is the standard measure for prokaryotic species delineation.
//! Genomes with ANI >= 95% are considered the same species.
//!
//! Reference: Goris et al. (2007) Int J Syst Evol Microbiol 57:81-91.

/// Result of a pairwise ANI calculation.
#[derive(Debug, Clone)]
pub struct AniResult {
    /// ANI value in [0.0, 1.0].
    pub ani: f64,
    /// Number of aligned nucleotide positions.
    pub aligned_length: usize,
    /// Number of identical positions.
    pub identical_positions: usize,
}

/// Compute pairwise ANI from two nucleotide sequences.
///
/// Aligns sequences positionally (assumes pre-aligned or equal-length).
/// Gaps (`-` or `.`) and `N`s are excluded from both numerator and denominator.
#[must_use]
pub fn pairwise_ani(seq1: &[u8], seq2: &[u8]) -> AniResult {
    const fn is_gap_or_n(b: u8) -> bool {
        matches!(b, b'-' | b'.' | b'N')
    }

    let (identical, aligned) = seq1
        .iter()
        .zip(seq2.iter())
        .map(|(&a, &b)| (a.to_ascii_uppercase(), b.to_ascii_uppercase()))
        .filter(|&(a, b)| !is_gap_or_n(a) && !is_gap_or_n(b))
        .fold((0usize, 0usize), |(ident, aln), (a, b)| {
            (ident + usize::from(a == b), aln + 1)
        });

    let ani = if aligned > 0 {
        identical as f64 / aligned as f64
    } else {
        0.0
    };

    AniResult {
        ani,
        aligned_length: aligned,
        identical_positions: identical,
    }
}

/// Compute pairwise ANI matrix (condensed, upper-triangular).
///
/// Returns `n*(n-1)/2` elements in row-major condensed order.
#[must_use]
pub fn ani_matrix(sequences: &[&[u8]]) -> Vec<f64> {
    let n = sequences.len();
    let mut matrix = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in 0..i {
            let result = pairwise_ani(sequences[i], sequences[j]);
            matrix.push(result.ani);
        }
    }
    matrix
}

/// GPU uniform parameters for batched ANI dispatch.
///
/// Maps directly to WGSL `var<uniform>` for future GPU absorption.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AniParams {
    /// Number of sequence pairs to process.
    pub n_pairs: u32,
    /// Maximum sequence length (all sequences padded to this).
    pub max_seq_len: u32,
}

/// Compute ANI for a batch of sequence pairs (GPU-friendly API).
///
/// Each pair is processed independently — maps to one GPU thread per pair.
/// This is the absorption target: `pairwise_ani` is the per-thread kernel.
#[must_use]
pub fn pairwise_ani_batch(pairs: &[(&[u8], &[u8])]) -> Vec<AniResult> {
    pairs.iter().map(|(s1, s2)| pairwise_ani(s1, s2)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sequences() {
        let seq = b"ATGATGATGATG";
        let r = pairwise_ani(seq, seq);
        assert!((r.ani - 1.0).abs() < 1e-15);
        assert_eq!(r.aligned_length, 12);
        assert_eq!(r.identical_positions, 12);
    }

    #[test]
    fn completely_different() {
        let r = pairwise_ani(b"AAAA", b"TTTT");
        assert!((r.ani - 0.0).abs() < 1e-15);
        assert_eq!(r.aligned_length, 4);
    }

    #[test]
    fn half_identical() {
        let r = pairwise_ani(b"AATT", b"AAGC");
        assert!((r.ani - 0.5).abs() < 1e-15);
    }

    #[test]
    fn gaps_excluded() {
        let r = pairwise_ani(b"A-TG", b"ACTG");
        assert_eq!(r.aligned_length, 3);
        assert_eq!(r.identical_positions, 3);
        assert!((r.ani - 1.0).abs() < 1e-15);
    }

    #[test]
    fn n_excluded() {
        let r = pairwise_ani(b"ANTG", b"ACTG");
        assert_eq!(r.aligned_length, 3);
    }

    #[test]
    fn symmetric() {
        let r1 = pairwise_ani(b"ATGATG", b"ATGTTG");
        let r2 = pairwise_ani(b"ATGTTG", b"ATGATG");
        assert!((r1.ani - r2.ani).abs() < 1e-15);
    }

    #[test]
    fn same_species_threshold() {
        // 96% identical → same species (> 95%)
        let seq1 = b"ATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG";
        let mut seq2 = seq1.to_vec();
        seq2[0] = b'C';
        seq2[10] = b'C';
        let r = pairwise_ani(seq1, &seq2);
        assert!(r.ani > 0.95);
    }

    #[test]
    fn matrix_size() {
        let seqs: Vec<&[u8]> = vec![b"ATG", b"ATG", b"CTG"];
        let m = ani_matrix(&seqs);
        assert_eq!(m.len(), 3); // 3*(3-1)/2
    }

    #[test]
    fn empty_sequences() {
        let r = pairwise_ani(b"", b"");
        assert!((r.ani - 0.0).abs() < 1e-15);
        assert_eq!(r.aligned_length, 0);
    }
}
