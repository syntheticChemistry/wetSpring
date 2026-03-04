// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired-end read merging for amplicon sequencing.
//!
//! Replaces VSEARCH `--fastq_mergepairs` / FLASH / QIIME2 `join-pairs`
//! in the 16S pipeline (Exp001/002). Pure Rust, zero external dependencies.
//!
//! # Algorithm
//!
//! 1. Find the best overlap between the 3' end of the forward read and
//!    the reverse-complement of the reverse read.
//! 2. Score overlaps using quality-weighted alignment.
//! 3. Resolve mismatches in the overlap region by choosing the base
//!    with higher quality.
//! 4. Compute merged quality scores using the posterior probability
//!    method (Edgar & Flyvbjerg, 2015).
//!
//! # References
//!
//! - Edgar, R.C. & Flyvbjerg, H. (2015). Error filtering, pair assembly
//!   and error correction for next-generation sequencing reads.
//!   Bioinformatics 31(21): 3476-3482.
//! - Magoč, T. & Salzberg, S.L. (2011). FLASH: fast length adjustment of
//!   short reads. Bioinformatics 27(21): 2957-2963.

use crate::bio::phred::{decode_qual, error_prob_to_phred, phred_to_error_prob};
use crate::io::fastq::FastqRecord;

/// Configuration for paired-end merging.
#[derive(Debug, Clone)]
pub struct MergeParams {
    /// Minimum overlap length (bp).
    /// Default: 10 (VSEARCH default).
    pub min_overlap: usize,
    /// Maximum overlap length (bp). `None` = no limit.
    /// Default: None.
    pub max_overlap: Option<usize>,
    /// Maximum fraction of mismatches in the overlap region.
    /// Default: 0.25 (VSEARCH `--fastq_maxdiffpct`).
    pub max_mismatch_fraction: f64,
    /// Maximum absolute mismatches in the overlap region.
    /// Default: 10.
    pub max_mismatches: usize,
    /// Phred encoding offset. Default: 33 (Illumina 1.8+).
    pub phred_offset: u8,
    /// Minimum merged read quality.
    /// Default: 0 (no filter).
    pub min_merged_quality: u8,
}

/// Manual impl intentional: non-zero defaults (VSEARCH/Edgar-Flyvbjerg style).
impl Default for MergeParams {
    fn default() -> Self {
        Self {
            min_overlap: 10,
            max_overlap: None,
            max_mismatches: 10,
            max_mismatch_fraction: 0.25,
            phred_offset: 33,
            min_merged_quality: 0,
        }
    }
}

/// Result of merging a read pair.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Merged record (if successful).
    pub merged: Option<FastqRecord>,
    /// Overlap length found.
    pub overlap: usize,
    /// Number of mismatches in the overlap.
    pub mismatches: usize,
}

/// Statistics from merging a batch of read pairs.
#[derive(Debug, Clone)]
pub struct MergeStats {
    /// Total input pairs.
    pub input_pairs: usize,
    /// Successfully merged pairs.
    pub merged_count: usize,
    /// Pairs that failed to merge (no acceptable overlap).
    pub no_overlap_count: usize,
    /// Pairs with too many mismatches.
    pub too_many_mismatches: usize,
    /// Mean overlap length of merged pairs.
    pub mean_overlap: f64,
    /// Mean merged length.
    pub mean_merged_length: f64,
}

/// Reverse-complement a DNA sequence.
#[must_use]
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&b| match b {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'N' | b'n' => b'N',
            other => other,
        })
        .collect()
}

/// Find the best overlap between forward and reverse-complement reads.
///
/// Slides the reverse-complement read along the forward read's 3' end
/// and scores each overlap position.
fn find_best_overlap(
    fwd: &[u8],
    rev_rc: &[u8],
    fwd_qual: &[u8],
    rev_rc_qual: &[u8],
    params: &MergeParams,
) -> Option<(usize, usize, usize)> {
    // offset = position in fwd where the rev_rc starts overlapping
    // overlap region: fwd[offset..fwd_len] aligns with rev_rc[0..overlap]
    let fwd_len = fwd.len();
    let rev_len = rev_rc.len();

    let max_overlap = params.max_overlap.unwrap_or_else(|| fwd_len.min(rev_len));
    let max_overlap = max_overlap.min(fwd_len).min(rev_len);

    let mut best_score = i64::MIN;
    let mut best_offset = 0_usize;
    let mut best_mismatches = 0_usize;
    let mut found = false;

    // Try all possible overlap lengths from max_overlap down to min_overlap
    for overlap in (params.min_overlap..=max_overlap).rev() {
        let offset = fwd_len - overlap;

        let mut score: i64 = 0;
        let mut mismatches = 0_usize;

        for k in 0..overlap {
            let fb = fwd[offset + k];
            let rb = rev_rc[k];
            let fq = i64::from(fwd_qual[offset + k].saturating_sub(params.phred_offset));
            let rq = i64::from(rev_rc_qual[k].saturating_sub(params.phred_offset));

            if bases_equal(fb, rb) {
                // Match: add quality bonus
                score += fq + rq;
            } else {
                // Mismatch: penalise
                mismatches += 1;
                score -= fq.max(rq);
            }
        }

        // Check mismatch constraints
        #[allow(clippy::cast_precision_loss)]
        let mismatch_frac = mismatches as f64 / overlap as f64;

        if mismatches > params.max_mismatches || mismatch_frac > params.max_mismatch_fraction {
            continue;
        }

        if score > best_score {
            best_score = score;
            best_offset = offset;
            best_mismatches = mismatches;
            found = true;
        }
    }

    if found {
        let overlap = fwd_len - best_offset;
        Some((best_offset, overlap, best_mismatches))
    } else {
        None
    }
}

/// Merge a single read pair.
///
/// The forward read and reverse read are expected as they come off the
/// sequencer (reverse read is NOT pre-complemented).
///
/// # Examples
///
/// ```
/// use wetspring_barracuda::bio::merge_pairs::{merge_pair, reverse_complement, MergeParams};
/// use wetspring_barracuda::io::fastq::FastqRecord;
///
/// let shared: &[u8] = b"ACGTACGTACGTACG";
/// let fwd_seq: Vec<u8> = [&b"AAAAAAAAAAAAAAAA"[..], shared].concat();
/// let rev_rc: Vec<u8> = [shared, &b"TTTTTTTTTTTTTTTT"[..]].concat();
/// let rev_seq = reverse_complement(&rev_rc);
/// let phred = |len: usize| (0..len).map(|_| 33 + 30u8).collect::<Vec<_>>();
///
/// let fwd = FastqRecord {
///     id: "fwd".into(),
///     sequence: fwd_seq.clone(),
///     quality: phred(fwd_seq.len()),
/// };
/// let rev = FastqRecord {
///     id: "rev".into(),
///     sequence: rev_seq,
///     quality: phred(rev_rc.len()),
/// };
/// let result = merge_pair(&fwd, &rev, &MergeParams::default());
/// assert!(result.merged.is_some());
/// assert_eq!(result.overlap, 15);
/// ```
#[must_use]
pub fn merge_pair(fwd: &FastqRecord, rev: &FastqRecord, params: &MergeParams) -> MergeResult {
    let rev_rc_seq = reverse_complement(&rev.sequence);
    let rev_rc_qual: Vec<u8> = rev.quality.iter().rev().copied().collect();

    let overlap_result = find_best_overlap(
        &fwd.sequence,
        &rev_rc_seq,
        &fwd.quality,
        &rev_rc_qual,
        params,
    );

    let Some((offset, overlap, mismatches)) = overlap_result else {
        return MergeResult {
            merged: None,
            overlap: 0,
            mismatches: 0,
        };
    };

    // Build merged sequence:
    // [fwd_prefix][overlap_consensus][rev_suffix]
    let fwd_prefix_len = offset;
    let rev_suffix_start = overlap;
    let merged_len = fwd_prefix_len + overlap + (rev_rc_seq.len() - overlap);

    let mut merged_seq = Vec::with_capacity(merged_len);
    let mut merged_qual = Vec::with_capacity(merged_len);

    // Forward prefix (before overlap)
    merged_seq.extend_from_slice(&fwd.sequence[..fwd_prefix_len]);
    merged_qual.extend_from_slice(&fwd.quality[..fwd_prefix_len]);

    // Overlap region: consensus
    for k in 0..overlap {
        let fi = offset + k;
        let fb = fwd.sequence[fi];
        let rb = rev_rc_seq[k];
        let fq = fwd.quality[fi];
        let rq = rev_rc_qual[k];

        if bases_equal(fb, rb) {
            // Agree: take the base, compute posterior quality
            merged_seq.push(fb.to_ascii_uppercase());
            merged_qual.push(posterior_quality_agree(fq, rq, params.phred_offset));
        } else {
            // Disagree: take the higher-quality base
            if fq >= rq {
                merged_seq.push(fb.to_ascii_uppercase());
                merged_qual.push(posterior_quality_disagree(fq, rq, params.phred_offset));
            } else {
                merged_seq.push(rb.to_ascii_uppercase());
                merged_qual.push(posterior_quality_disagree(rq, fq, params.phred_offset));
            }
        }
    }

    // Reverse suffix (after overlap)
    merged_seq.extend_from_slice(&rev_rc_seq[rev_suffix_start..]);
    merged_qual.extend_from_slice(&rev_rc_qual[rev_suffix_start..]);

    // Quality filter on merged read
    if params.min_merged_quality > 0 {
        #[allow(clippy::cast_precision_loss)]
        let mean_q: f64 = merged_qual
            .iter()
            .map(|&q| f64::from(q.saturating_sub(params.phred_offset)))
            .sum::<f64>()
            / merged_qual.len() as f64;
        if mean_q < f64::from(params.min_merged_quality) {
            return MergeResult {
                merged: None,
                overlap,
                mismatches,
            };
        }
    }

    MergeResult {
        merged: Some(FastqRecord {
            id: fwd.id.clone(),
            sequence: merged_seq,
            quality: merged_qual,
        }),
        overlap,
        mismatches,
    }
}

/// Merge a batch of read pairs.
///
/// Forward and reverse reads must be in the same order (paired).
///
/// # Panics
///
/// Panics if `fwd_reads.len() != rev_reads.len()` (forward and reverse read
/// counts must match).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn merge_pairs(
    fwd_reads: &[FastqRecord],
    rev_reads: &[FastqRecord],
    params: &MergeParams,
) -> (Vec<FastqRecord>, MergeStats) {
    assert_eq!(
        fwd_reads.len(),
        rev_reads.len(),
        "Forward and reverse read counts must match"
    );

    let mut merged = Vec::with_capacity(fwd_reads.len());
    let mut total_overlap = 0_usize;
    let mut total_merged_len = 0_usize;
    let mut no_overlap = 0_usize;
    let mut too_many_mm = 0_usize;

    for (fwd, rev) in fwd_reads.iter().zip(rev_reads.iter()) {
        let result = merge_pair(fwd, rev, params);

        if let Some(rec) = result.merged {
            total_overlap += result.overlap;
            total_merged_len += rec.sequence.len();
            merged.push(rec);
        } else if result.overlap == 0 {
            no_overlap += 1;
        } else {
            too_many_mm += 1;
        }
    }

    let n_merged = merged.len();
    let stats = MergeStats {
        input_pairs: fwd_reads.len(),
        merged_count: n_merged,
        no_overlap_count: no_overlap,
        too_many_mismatches: too_many_mm,
        mean_overlap: if n_merged > 0 {
            total_overlap as f64 / n_merged as f64
        } else {
            0.0
        },
        mean_merged_length: if n_merged > 0 {
            total_merged_len as f64 / n_merged as f64
        } else {
            0.0
        },
    };

    (merged, stats)
}

/// Case-insensitive DNA base comparison.
const fn bases_equal(a: u8, b: u8) -> bool {
    a.eq_ignore_ascii_case(&b)
}

/// Posterior quality when two bases agree.
///
/// Edgar & Flyvbjerg (2015), Eq. 2:
/// `P_merged = P_f * P_r / (1 - P_f)(1 - P_r) + P_f * P_r`
///
/// Where P = 10^(-Q/10).
#[must_use]
fn posterior_quality_agree(fq: u8, rq: u8, offset: u8) -> u8 {
    let qf = decode_qual(fq, offset);
    let qr = decode_qual(rq, offset);
    let pf = phred_to_error_prob(qf);
    let pr = phred_to_error_prob(qr);

    let p_err = pf * pr / 3.0;
    let p_correct = (1.0 - pf).mul_add(1.0 - pr, p_err);
    let p_merged = p_err / p_correct;

    let q_merged = error_prob_to_phred(p_merged);

    // Cap at Q41 (Illumina max), floor at 0
    let q_capped = q_merged.clamp(0.0, 41.0);

    q_capped.round() as u8 + offset
}

/// Posterior quality when two bases disagree.
///
/// Take the difference of the error probabilities.
#[must_use]
fn posterior_quality_disagree(higher_q: u8, lower_q: u8, offset: u8) -> u8 {
    let qh = decode_qual(higher_q, offset);
    let ql = decode_qual(lower_q, offset);
    let ph = phred_to_error_prob(qh);
    let pl = phred_to_error_prob(ql);

    let p_merged = ph * (1.0 - pl / 3.0) / ph.mul_add(1.0 - pl / 3.0, (1.0 - ph) * pl / 3.0);

    let q_merged = error_prob_to_phred(p_merged);

    let q_capped = q_merged.clamp(0.0, 41.0);

    q_capped.round() as u8 + offset
}

#[cfg(test)]
#[path = "merge_pairs_tests.rs"]
mod tests;
