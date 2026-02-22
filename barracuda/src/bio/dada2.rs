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

use crate::bio::derep::UniqueSequence;
use crate::special::regularized_gamma_lower;
use std::fmt::Write;

/// Number of canonical nucleotide bases (A, C, G, T).
pub(crate) const NUM_BASES: usize = 4;

/// Maximum Phred quality score tracked in the error matrix.
///
/// Illumina instruments report Q0–Q41; Q42 covers edge cases.
/// Matches DADA2 R package `MAX_QUAL` (Callahan et al. 2016, §Methods).
pub(crate) const MAX_QUAL: usize = 42;

/// Abundance p-value threshold for partition splitting.
///
/// A sequence is split from its current partition when the Poisson
/// probability of observing its abundance as errors from the center
/// falls below this threshold. Default from DADA2 R package `OMEGA_A`
/// (Callahan et al. 2016, §Methods; R source `dada.cpp:22`).
const OMEGA_A: f64 = 1e-40;

/// Maximum outer iterations of denoise → re-estimate error model.
///
/// DADA2 R package default `MAX_CONSIST = 10` (`dada.R:113`).
const MAX_DADA_ITERS: usize = 10;

/// Maximum iterations for error-rate self-consistency loop.
///
/// DADA2 R package `learnErrors()` default convergence rounds.
pub(crate) const MAX_ERR_ITERS: usize = 6;

/// Floor for per-substitution error rate.
///
/// Prevents log(0) in Poisson abundance calculations. DADA2 R package
/// `MIN_ERR_RATE` (`dada.cpp:24`).
pub(crate) const MIN_ERR: f64 = 1e-7;

/// Ceiling for per-substitution error rate.
///
/// Caps unreliable estimates from low-coverage substitution classes.
/// DADA2 R package `MAX_ERR_RATE` (`dada.cpp:25`).
pub(crate) const MAX_ERR: f64 = 0.25;

/// An Amplicon Sequence Variant — the output of denoising.
#[derive(Debug, Clone)]
pub struct Asv {
    /// The denoised sequence (uppercase ACGT).
    pub sequence: Vec<u8>,
    /// Total abundance (sum of all sequences assigned to this ASV).
    pub abundance: usize,
    /// Number of unique sequences merged into this ASV.
    pub n_members: usize,
}

/// Parameters for DADA2 denoising.
#[derive(Debug, Clone)]
pub struct Dada2Params {
    /// Abundance p-value threshold. Sequences with `p < omega_a` are promoted
    /// to new ASVs instead of being absorbed. Default: 1e-40.
    pub omega_a: f64,
    /// Maximum rounds of the partition–refine loop. Default: 10.
    pub max_iterations: usize,
    /// Maximum rounds of error model refinement per partition step. Default: 6.
    pub max_err_iterations: usize,
    /// Minimum abundance for a unique sequence to be considered. Default: 1.
    pub min_abundance: usize,
}

impl Default for Dada2Params {
    fn default() -> Self {
        Self {
            omega_a: OMEGA_A,
            max_iterations: MAX_DADA_ITERS,
            max_err_iterations: MAX_ERR_ITERS,
            min_abundance: 1,
        }
    }
}

/// Statistics from a denoising run.
#[derive(Debug, Clone)]
pub struct Dada2Stats {
    /// Number of input unique sequences.
    pub input_uniques: usize,
    /// Total input reads (sum of abundances).
    pub input_reads: usize,
    /// Number of ASVs produced.
    pub output_asvs: usize,
    /// Total output reads (sum of ASV abundances).
    pub output_reads: usize,
    /// Number of partition–refine iterations performed.
    pub iterations: usize,
}

/// 4×4 error matrix indexed by quality score.
/// `err[from][to][qual]` = P(observing `to` when truth is `from` at quality `qual`).
pub(crate) type ErrorModel = [[[f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];

/// Denoise a set of dereplicated sequences into ASVs.
///
/// This is the main entry point. Takes abundance-sorted `UniqueSequence`s
/// (from `bio::derep::dereplicate`) and returns ASVs.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn denoise(seqs: &[UniqueSequence], params: &Dada2Params) -> (Vec<Asv>, Dada2Stats) {
    let seqs: Vec<&UniqueSequence> = seqs
        .iter()
        .filter(|s| s.abundance >= params.min_abundance && !s.sequence.is_empty())
        .collect();

    let input_uniques = seqs.len();
    let input_reads: usize = seqs.iter().map(|s| s.abundance).sum();

    if seqs.is_empty() {
        return (
            vec![],
            Dada2Stats {
                input_uniques,
                input_reads,
                output_asvs: 0,
                output_reads: 0,
                iterations: 0,
            },
        );
    }

    let mut err = init_error_model();

    // Partition assignments: partition[i] = index of center sequence
    let mut partition: Vec<usize> = vec![0; seqs.len()];
    let mut centers: Vec<usize> = vec![0];

    let mut last_n_centers = 0;
    let mut iters = 0;

    for _ in 0..params.max_iterations {
        iters += 1;

        // E-step: assign each sequence to the nearest center
        assign_to_centers(&seqs, &centers, &err, &mut partition);

        // M-step: update error model from assignments
        for _ in 0..params.max_err_iterations {
            let new_err = estimate_error_model(&seqs, &partition, &centers);
            let converged = err_model_converged(&err, &new_err);
            err = new_err;
            if converged {
                break;
            }
        }

        // Split step: find sequences whose abundance is too high to be errors
        let new_centers = find_new_centers(&seqs, &partition, &centers, &err, params.omega_a);

        for &c in &new_centers {
            if !centers.contains(&c) {
                centers.push(c);
            }
        }
        centers.sort_unstable();

        if centers.len() == last_n_centers {
            break;
        }
        last_n_centers = centers.len();
    }

    // Final assignment
    assign_to_centers(&seqs, &centers, &err, &mut partition);

    // Build ASVs
    let mut asvs = build_asvs(&seqs, &partition, &centers);
    asvs.sort_by(|a, b| b.abundance.cmp(&a.abundance));

    let output_reads: usize = asvs.iter().map(|a| a.abundance).sum();
    let output_asvs = asvs.len();

    (
        asvs,
        Dada2Stats {
            input_uniques,
            input_reads,
            output_asvs,
            output_reads,
            iterations: iters,
        },
    )
}

/// Initialize error model from Phred quality scores (no prior data).
#[allow(clippy::needless_range_loop)] // 3D array requires indexing by from/to/q
pub(crate) fn init_error_model() -> ErrorModel {
    let mut err = [[[0.0_f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];
    for q in 0..MAX_QUAL {
        #[allow(clippy::cast_precision_loss)] // q is 0..42, exact
        let p_err = (10.0_f64).powf(-(q as f64) / 10.0).clamp(MIN_ERR, MAX_ERR);
        for from in 0..NUM_BASES {
            for to in 0..NUM_BASES {
                if from == to {
                    err[from][to][q] = 1.0 - p_err;
                } else {
                    err[from][to][q] = p_err / 3.0;
                }
            }
        }
    }
    err
}

#[allow(clippy::match_same_arms)]
/// Map nucleotide to error-matrix index: A=0, C=1, G=2, T=3.
///
/// Ambiguous/unknown bases (N, IUPAC degenerate) map to 0 (A).
/// This matches the DADA2 R package behavior where non-ACGT bases
/// are treated as A for error-rate indexing — rare in quality-filtered
/// reads and has negligible effect on the error model.
pub(crate) const fn base_to_idx(b: u8) -> usize {
    match b {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 0,
    }
}

/// Compute log-probability that `seq` arose from `center` by errors.
fn log_p_error(seq: &UniqueSequence, center: &UniqueSequence, err: &ErrorModel) -> f64 {
    let len = seq.sequence.len().min(center.sequence.len());
    let mut log_p = 0.0_f64;
    for i in 0..len {
        let from = base_to_idx(center.sequence[i]);
        let to = base_to_idx(seq.sequence[i]);
        let q = seq
            .representative_quality
            .get(i)
            .map_or(0, |&v| v.saturating_sub(33) as usize)
            .min(MAX_QUAL - 1);
        let p = err[from][to][q].max(MIN_ERR);
        log_p += p.ln();
    }
    log_p
}

/// Assign each sequence to its most likely center.
fn assign_to_centers(
    seqs: &[&UniqueSequence],
    centers: &[usize],
    err: &ErrorModel,
    partition: &mut [usize],
) {
    for (i, seq) in seqs.iter().enumerate() {
        let mut best_center = centers[0];
        let mut best_log_p = f64::NEG_INFINITY;

        for &c in centers {
            let lp = log_p_error(seq, seqs[c], err);
            if lp > best_log_p {
                best_log_p = lp;
                best_center = c;
            }
        }
        partition[i] = best_center;
    }
}

/// Re-estimate error model from observed substitution patterns.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn estimate_error_model(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    _centers: &[usize],
) -> ErrorModel {
    // Count observed transitions
    let mut counts = [[[0.0_f64; MAX_QUAL]; NUM_BASES]; NUM_BASES];
    let mut totals = [[0.0_f64; MAX_QUAL]; NUM_BASES];

    for (i, seq) in seqs.iter().enumerate() {
        let center_idx = partition[i];
        let center = seqs[center_idx];
        let len = seq.sequence.len().min(center.sequence.len());
        let weight = seq.abundance as f64;

        for pos in 0..len {
            let from = base_to_idx(center.sequence[pos]);
            let to = base_to_idx(seq.sequence[pos]);
            let q = seq
                .representative_quality
                .get(pos)
                .map_or(0, |&v| v.saturating_sub(33) as usize)
                .min(MAX_QUAL - 1);

            counts[from][to][q] += weight;
            totals[from][q] += weight;
        }
    }

    let mut err = init_error_model();
    #[allow(clippy::needless_range_loop)] // 3D array requires indexing by from/to/q
    for from in 0..NUM_BASES {
        for q in 0..MAX_QUAL {
            if totals[from][q] > 0.0 {
                for to in 0..NUM_BASES {
                    let rate = counts[from][to][q] / totals[from][q];
                    err[from][to][q] = rate.clamp(MIN_ERR, 1.0 - MIN_ERR);
                }
            }
        }
    }

    // Ensure rows sum to ~1
    #[allow(clippy::needless_range_loop)] // 3D array requires indexing by from/to/q
    for from in 0..NUM_BASES {
        for q in 0..MAX_QUAL {
            let sum: f64 = (0..NUM_BASES).map(|to| err[from][to][q]).sum();
            if sum > 0.0 {
                for to in 0..NUM_BASES {
                    err[from][to][q] /= sum;
                }
            }
        }
    }

    err
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn err_model_converged(old: &ErrorModel, new: &ErrorModel) -> bool {
    let mut max_diff = 0.0_f64;
    for from in 0..NUM_BASES {
        for to in 0..NUM_BASES {
            for q in 0..MAX_QUAL {
                let diff = (old[from][to][q] - new[from][to][q]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }
    max_diff < crate::tolerances::DADA2_ERR_CONVERGENCE
}

/// Find sequences that should become new centers (abundance p-value test).
///
/// For each non-center sequence, computes the expected abundance under the
/// error model and tests whether the observed abundance is significantly
/// higher using a Poisson CDF approximation.
#[allow(clippy::cast_precision_loss)]
fn find_new_centers(
    seqs: &[&UniqueSequence],
    partition: &[usize],
    centers: &[usize],
    err: &ErrorModel,
    omega_a: f64,
) -> Vec<usize> {
    let mut new_centers = Vec::new();

    for (i, seq) in seqs.iter().enumerate() {
        if centers.contains(&i) {
            continue;
        }

        let center_idx = partition[i];
        let center = seqs[center_idx];

        // Expected number of reads that would look like seq if they came from center
        let log_p = log_p_error(seq, center, err);
        let lambda = (center.abundance as f64) * log_p.exp();

        if lambda <= 0.0 {
            new_centers.push(i);
            continue;
        }

        let observed = seq.abundance;

        // Poisson survival function: P(X >= observed) where X ~ Poisson(lambda)
        let p_value = poisson_pvalue(observed, lambda);

        if p_value < omega_a {
            new_centers.push(i);
        }
    }

    new_centers
}

/// Upper-tail Poisson p-value: P(X >= k) for X ~ Poisson(lambda).
/// Uses the identity: P(X >= k | Poisson(λ)) = P(k, λ) where P is the
/// regularized lower incomplete gamma function.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn poisson_pvalue(k: usize, lambda: f64) -> f64 {
    if lambda <= 0.0 || k == 0 {
        return 1.0;
    }
    regularized_gamma_lower(k as f64, lambda)
}

/// Build ASV structs from final partition assignments.
fn build_asvs(seqs: &[&UniqueSequence], partition: &[usize], centers: &[usize]) -> Vec<Asv> {
    let mut asvs: Vec<Asv> = centers
        .iter()
        .map(|&c| Asv {
            sequence: seqs[c].sequence.clone(),
            abundance: 0,
            n_members: 0,
        })
        .collect();

    for (i, &center_idx) in partition.iter().enumerate() {
        if let Some(asv_pos) = centers.iter().position(|&c| c == center_idx) {
            asvs[asv_pos].abundance += seqs[i].abundance;
            asvs[asv_pos].n_members += 1;
        }
    }

    asvs
}

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
