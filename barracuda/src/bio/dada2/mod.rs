// SPDX-License-Identifier: AGPL-3.0-or-later
//! DADA2-style amplicon sequence variant (ASV) denoising.
//!
//! Special math functions (`ln_gamma`, `regularized_gamma_lower`) are
//! provided by [`crate::special`] — the shared sovereign math module.
//!
//! Implements the core algorithm from Callahan et al. "DADA2: High-resolution
//! sample inference from Illumina amplicon data." Nature Methods 13, 581–583
//! (2016).
//!
//! # Algorithm
//!
//! 1. **Error model**: For each nucleotide substitution (A→C, A→G, etc.) and
//!    quality score, estimate the probability of that error occurring. The
//!    initial model uses Phred quality: `P_error = 10^(-Q/10)`.
//!
//! 2. **Divisive partitioning**: Starting from abundance-sorted dereplicated
//!    sequences, iteratively split partitions when a sequence's abundance
//!    exceeds what the error model predicts as errors from the partition center.
//!    Uses a Poisson abundance p-value test (`OMEGA_A` threshold).
//!
//! 3. **Error model refinement**: After partitioning, re-estimate error rates
//!    from observed substitution patterns and iterate.
//!
//! 4. **Output**: Each final partition center is an ASV with its total
//!    abundance (sum of all member sequences).
//!
//! # Input
//!
//! Takes `UniqueSequence` from `bio::derep` (sequences sorted by abundance,
//! with per-base quality scores from the best representative read).
//!
//! # References
//!
//! - Callahan et al. Nature Methods 13, 581–583 (2016).
//! - QIIME2 `dada2 denoise-paired` / `denoise-single`.

mod core;
mod types;

use std::fmt::Write;

#[allow(unused_imports)]
pub(crate) use core::{
    ErrorModel, base_to_idx, err_model_converged, estimate_error_model, init_error_model,
};
pub use core::{MAX_ERR_ITERS, denoise, poisson_pvalue};
#[allow(unused_imports)]
pub(crate) use core::{MAX_QUAL, MIN_ERR, NUM_BASES};
pub use types::{Asv, Dada2Params, Dada2Stats};

/// Write ASVs to FASTA format suitable for downstream analysis.
#[must_use]
pub fn asvs_to_fasta(asvs: &[Asv]) -> String {
    let mut out = String::new();
    for (i, asv) in asvs.iter().enumerate() {
        let _ = writeln!(out, ">ASV_{};size={}", i + 1, asv.abundance);
        out.push_str(&String::from_utf8_lossy(&asv.sequence));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bio::derep::UniqueSequence;

    fn make_unique(seq: &[u8], abundance: usize, q: u8) -> UniqueSequence {
        UniqueSequence {
            sequence: seq.to_vec(),
            abundance,
            best_quality: f64::from(q),
            representative_id: String::new(),
            representative_quality: vec![33 + q; seq.len()],
        }
    }

    #[test]
    fn empty_input() {
        let (asvs, stats) = denoise(&[], &Dada2Params::default());
        assert!(asvs.is_empty());
        assert_eq!(stats.input_uniques, 0);
        assert_eq!(stats.output_asvs, 0);
    }

    #[test]
    fn single_sequence() {
        let seqs = vec![make_unique(b"ACGTACGT", 100, 30)];
        let (asvs, stats) = denoise(&seqs, &Dada2Params::default());
        assert_eq!(asvs.len(), 1);
        assert_eq!(asvs[0].abundance, 100);
        assert_eq!(stats.output_asvs, 1);
    }

    #[test]
    fn identical_sequences_collapse() {
        let seqs = vec![
            make_unique(b"ACGTACGT", 100, 30),
            make_unique(b"ACGTACGT", 50, 30),
        ];
        let (_asvs, stats) = denoise(&seqs, &Dada2Params::default());
        assert_eq!(stats.input_reads, 150);
        assert_eq!(stats.output_reads, 150);
    }

    #[test]
    fn distinct_sequences_separate() {
        let seqs = vec![
            make_unique(b"AAAAAAAAAA", 1000, 35),
            make_unique(b"CCCCCCCCCC", 1000, 35),
        ];
        let (asvs, stats) = denoise(&seqs, &Dada2Params::default());
        assert_eq!(asvs.len(), 2);
        assert_eq!(stats.output_asvs, 2);
        assert_eq!(stats.output_reads, 2000);
    }

    #[test]
    fn error_variant_absorbed() {
        // A very abundant center and a low-abundance variant with 1 mismatch
        // at high quality — should be absorbed
        let mut variant = b"ACGTACGT".to_vec();
        variant[3] = b'A'; // one mismatch
        let seqs = vec![
            make_unique(b"ACGTACGT", 10000, 35),
            make_unique(&variant, 2, 35),
        ];
        let (asvs, _) = denoise(&seqs, &Dada2Params::default());
        // The variant should be absorbed into the center
        assert_eq!(asvs.len(), 1);
        assert_eq!(asvs[0].abundance, 10002);
    }

    #[test]
    fn abundant_variant_becomes_asv() {
        // Two quite different sequences, both highly abundant
        let seqs = vec![
            make_unique(b"AAAAAAAAAA", 5000, 35),
            make_unique(b"TTTTTTTTTT", 5000, 35),
        ];
        let (asvs, _) = denoise(&seqs, &Dada2Params::default());
        assert_eq!(asvs.len(), 2);
    }

    #[test]
    fn reads_conserved() {
        let seqs = vec![
            make_unique(b"ACGTACGTAC", 500, 30),
            make_unique(b"GCTAGCTAGC", 300, 30),
            make_unique(b"ACGTACGTCC", 5, 30),
        ];
        let total_in: usize = seqs.iter().map(|s| s.abundance).sum();
        let (asvs, stats) = denoise(&seqs, &Dada2Params::default());
        let total_out: usize = asvs.iter().map(|a| a.abundance).sum();
        assert_eq!(total_in, total_out);
        assert_eq!(stats.input_reads, stats.output_reads);
    }

    #[test]
    fn fasta_output_format() {
        let asvs = vec![
            Asv {
                sequence: b"ACGT".to_vec(),
                abundance: 100,
                n_members: 3,
            },
            Asv {
                sequence: b"GCTA".to_vec(),
                abundance: 50,
                n_members: 1,
            },
        ];
        let fasta = asvs_to_fasta(&asvs);
        assert!(fasta.contains(">ASV_1;size=100"));
        assert!(fasta.contains(">ASV_2;size=50"));
        assert!(fasta.contains("ACGT"));
        assert!(fasta.contains("GCTA"));
    }

    #[test]
    fn min_abundance_filter() {
        let seqs = vec![
            make_unique(b"ACGTACGT", 100, 30),
            make_unique(b"GCTAGCTA", 1, 30),
        ];
        let params = Dada2Params {
            min_abundance: 2,
            ..Dada2Params::default()
        };
        let (_, stats) = denoise(&seqs, &params);
        assert_eq!(stats.input_uniques, 1);
    }
}
