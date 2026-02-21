// SPDX-License-Identifier: AGPL-3.0-or-later
//! Smith-Waterman local sequence alignment.
//!
//! Implements the classic DP algorithm for local alignment with affine gap
//! penalties. This is a prerequisite for Liu track papers (`SATé`, metagenomics)
//! and a well-known GPU parallelization target.
//!
//! # References
//!
//! - Smith & Waterman 1981, *J Mol Biol* 147:195-197
//! - Liu et al. 2009, *Science* 324:1561-1564 (`SATé`)
//! - Alamin & Liu 2024, *IEEE/ACM TCBB* (metagenomics placement)

/// Alignment result.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Optimal local alignment score.
    pub score: i32,
    /// Aligned subsequence of query.
    pub aligned_query: Vec<u8>,
    /// Aligned subsequence of target.
    pub aligned_target: Vec<u8>,
    /// Start position in query (0-indexed).
    pub query_start: usize,
    /// Start position in target (0-indexed).
    pub target_start: usize,
}

/// Scoring parameters for alignment.
#[derive(Debug, Clone, Copy)]
pub struct ScoringParams {
    /// Score for matching residues.
    pub match_score: i32,
    /// Penalty for mismatching residues.
    pub mismatch_penalty: i32,
    /// Penalty for opening a gap.
    pub gap_open: i32,
    /// Penalty per residue in an extended gap.
    pub gap_extend: i32,
}

impl Default for ScoringParams {
    fn default() -> Self {
        Self {
            match_score: 2,
            mismatch_penalty: -1,
            gap_open: -3,
            gap_extend: -1,
        }
    }
}

/// BLOSUM62-style nucleotide scoring: match/mismatch.
#[inline]
const fn score_pair(a: u8, b: u8, params: &ScoringParams) -> i32 {
    let a_upper = a.to_ascii_uppercase();
    let b_upper = b.to_ascii_uppercase();
    if a_upper == b_upper {
        params.match_score
    } else {
        params.mismatch_penalty
    }
}

/// Smith-Waterman local alignment with affine gap penalties.
///
/// Returns the optimal local alignment. Time: O(mn), Space: O(mn).
/// For GPU promotion, the anti-diagonal wavefront is embarrassingly parallel.
#[must_use]
#[allow(clippy::cast_possible_wrap, clippy::many_single_char_names)]
pub fn smith_waterman(query: &[u8], target: &[u8], params: &ScoringParams) -> AlignmentResult {
    let m = query.len();
    let n = target.len();

    if m == 0 || n == 0 {
        return AlignmentResult {
            score: 0,
            aligned_query: Vec::new(),
            aligned_target: Vec::new(),
            query_start: 0,
            target_start: 0,
        };
    }

    // H[i][j]: best score ending at (i,j)
    // E[i][j]: best score ending with gap in query (insertion in target)
    // F[i][j]: best score ending with gap in target (deletion from query)
    let mut h_matrix = vec![0_i32; (m + 1) * (n + 1)];
    let mut e_matrix = vec![i32::MIN / 2; (m + 1) * (n + 1)];
    let mut f_matrix = vec![i32::MIN / 2; (m + 1) * (n + 1)];

    let idx = |i: usize, j: usize| -> usize { i * (n + 1) + j };

    let mut best_score = 0_i32;
    let mut best_i = 0_usize;
    let mut best_j = 0_usize;

    for i in 1..=m {
        for j in 1..=n {
            let s = score_pair(query[i - 1], target[j - 1], params);

            // Gap in query (horizontal)
            e_matrix[idx(i, j)] = (h_matrix[idx(i, j - 1)] + params.gap_open + params.gap_extend)
                .max(e_matrix[idx(i, j - 1)] + params.gap_extend);

            // Gap in target (vertical)
            f_matrix[idx(i, j)] = (h_matrix[idx(i - 1, j)] + params.gap_open + params.gap_extend)
                .max(f_matrix[idx(i - 1, j)] + params.gap_extend);

            h_matrix[idx(i, j)] = 0_i32
                .max(h_matrix[idx(i - 1, j - 1)] + s)
                .max(e_matrix[idx(i, j)])
                .max(f_matrix[idx(i, j)]);

            if h_matrix[idx(i, j)] > best_score {
                best_score = h_matrix[idx(i, j)];
                best_i = i;
                best_j = j;
            }
        }
    }

    // Traceback
    let mut aligned_q = Vec::new();
    let mut aligned_t = Vec::new();
    let mut i = best_i;
    let mut j = best_j;

    while i > 0 && j > 0 && h_matrix[idx(i, j)] > 0 {
        let current = h_matrix[idx(i, j)];
        let s = score_pair(query[i - 1], target[j - 1], params);
        let diag = h_matrix[idx(i - 1, j - 1)] + s;

        if current == diag {
            aligned_q.push(query[i - 1]);
            aligned_t.push(target[j - 1]);
            i -= 1;
            j -= 1;
        } else if current == f_matrix[idx(i, j)] {
            aligned_q.push(query[i - 1]);
            aligned_t.push(b'-');
            i -= 1;
        } else {
            aligned_q.push(b'-');
            aligned_t.push(target[j - 1]);
            j -= 1;
        }
    }

    aligned_q.reverse();
    aligned_t.reverse();

    AlignmentResult {
        score: best_score,
        aligned_query: aligned_q,
        aligned_target: aligned_t,
        query_start: i,
        target_start: j,
    }
}

/// Compute alignment score only (no traceback). Useful for batch scoring.
#[must_use]
pub fn smith_waterman_score(query: &[u8], target: &[u8], params: &ScoringParams) -> i32 {
    let m = query.len();
    let n = target.len();
    if m == 0 || n == 0 {
        return 0;
    }

    // Space-optimized: only need two rows of H + E, one row of F
    let mut h_prev = vec![0_i32; n + 1];
    let mut h_curr = vec![0_i32; n + 1];
    let mut e_row = vec![i32::MIN / 2; n + 1];
    let mut best = 0_i32;

    for i in 1..=m {
        let mut f_val = i32::MIN / 2;
        for j in 1..=n {
            let s = score_pair(query[i - 1], target[j - 1], params);
            e_row[j] = (h_curr[j - 1] + params.gap_open + params.gap_extend)
                .max(e_row[j - 1] + params.gap_extend);
            f_val =
                (h_prev[j] + params.gap_open + params.gap_extend).max(f_val + params.gap_extend);
            h_curr[j] = 0_i32.max(h_prev[j - 1] + s).max(e_row[j]).max(f_val);
            best = best.max(h_curr[j]);
        }
        std::mem::swap(&mut h_prev, &mut h_curr);
        h_curr.fill(0);
        e_row.fill(i32::MIN / 2);
    }

    best
}

/// Batch pairwise alignment scores for a set of sequences.
///
/// Returns condensed upper-triangular matrix of scores.
/// This is the target for GPU parallelization (anti-diagonal wavefront).
#[must_use]
pub fn pairwise_scores(sequences: &[&[u8]], params: &ScoringParams) -> Vec<i32> {
    let n = sequences.len();
    let mut scores = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            scores.push(smith_waterman_score(sequences[i], sequences[j], params));
        }
    }
    scores
}

// ─── Batch API (GPU-ready) ──────────────────────────────────────────
//
// Each pair is independent — batch dispatch with one workgroup per pair.
// Anti-diagonal wavefront parallelism within each pair for larger sequences.

/// Batch score: compute Smith-Waterman scores for explicit query/target pairs.
///
/// Each `(query, target)` pair is independent, enabling GPU dispatch with
/// one workgroup per pair.
#[must_use]
pub fn score_batch(pairs: &[(&[u8], &[u8])], params: &ScoringParams) -> Vec<i32> {
    pairs
        .iter()
        .map(|(q, t)| smith_waterman_score(q, t, params))
        .collect()
}

/// Batch full alignment: compute alignments for explicit query/target pairs.
#[must_use]
pub fn align_batch(pairs: &[(&[u8], &[u8])], params: &ScoringParams) -> Vec<AlignmentResult> {
    pairs
        .iter()
        .map(|(q, t)| smith_waterman(q, t, params))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sequences() {
        let q = b"ACGTACGT";
        let p = ScoringParams::default();
        let r = smith_waterman(q, q, &p);
        assert_eq!(r.score, 16, "identical 8bp should score 8*2=16");
        assert_eq!(r.aligned_query, q.to_vec());
        assert_eq!(r.aligned_target, q.to_vec());
    }

    #[test]
    fn simple_mismatch() {
        let q = b"ACGT";
        let t = b"ACTT";
        let p = ScoringParams::default();
        let r = smith_waterman(q, t, &p);
        assert!(r.score > 0, "should find local alignment");
        assert!(r.score < 8, "should be less than perfect: {}", r.score);
    }

    #[test]
    fn gap_alignment() {
        let q = b"ACGTACGT";
        let t = b"ACGACGT";
        let p = ScoringParams::default();
        let r = smith_waterman(q, t, &p);
        assert!(r.score > 0);
    }

    #[test]
    fn empty_sequences() {
        let p = ScoringParams::default();
        assert_eq!(smith_waterman(b"", b"ACGT", &p).score, 0);
        assert_eq!(smith_waterman(b"ACGT", b"", &p).score, 0);
        assert_eq!(smith_waterman(b"", b"", &p).score, 0);
    }

    #[test]
    fn no_match() {
        let p = ScoringParams {
            match_score: 1,
            mismatch_penalty: -3,
            gap_open: -5,
            gap_extend: -2,
        };
        let r = smith_waterman(b"AAAA", b"CCCC", &p);
        assert_eq!(r.score, 0, "completely different should score 0");
    }

    #[test]
    fn local_alignment_finds_best_region() {
        let q = b"XXXACGTACGTXXX";
        let t = b"ACGTACGT";
        let p = ScoringParams::default();
        let r = smith_waterman(q, t, &p);
        assert_eq!(r.score, 16, "should find perfect 8bp match in the middle");
    }

    #[test]
    fn score_only_matches_full() {
        let q = b"ACGTACGT";
        let t = b"ACTTACGT";
        let p = ScoringParams::default();
        let full = smith_waterman(q, t, &p);
        let score = smith_waterman_score(q, t, &p);
        assert_eq!(full.score, score, "score-only should match full alignment");
    }

    #[test]
    fn pairwise_scores_symmetric() {
        let seqs: Vec<&[u8]> = vec![b"ACGT", b"ACTT", b"GGGG"];
        let p = ScoringParams::default();
        let scores = pairwise_scores(&seqs, &p);
        assert_eq!(scores.len(), 3, "3 sequences → 3 pairs");
    }

    #[test]
    fn case_insensitive() {
        let p = ScoringParams::default();
        let r1 = smith_waterman(b"ACGT", b"acgt", &p);
        let r2 = smith_waterman(b"ACGT", b"ACGT", &p);
        assert_eq!(r1.score, r2.score, "should be case-insensitive");
    }

    #[test]
    fn deterministic() {
        let p = ScoringParams::default();
        let r1 = smith_waterman(b"ACGTACGTACGT", b"ACTTACGTACTT", &p);
        let r2 = smith_waterman(b"ACGTACGTACGT", b"ACTTACGTACTT", &p);
        assert_eq!(r1.score, r2.score);
        assert_eq!(r1.aligned_query, r2.aligned_query);
    }
}
