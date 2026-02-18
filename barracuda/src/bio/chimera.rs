// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference-free chimera detection for amplicon sequences.
//!
//! Implements a UCHIME-style algorithm (Edgar et al. 2011) for detecting
//! chimeric sequences formed during PCR amplification. Chimeras are
//! artificial sequences composed of two or more parent sequences joined
//! at crossover points.
//!
//! # Algorithm
//!
//! 1. Sort sequences by abundance (descending).
//! 2. For each query, find the two best-matching parents from the pool
//!    of more-abundant sequences.
//! 3. For each possible crossover point, compute a score comparing the
//!    two-parent chimera model against the best single-parent model.
//! 4. If the best chimera score exceeds the threshold, flag the sequence.
//!
//! # References
//!
//! - Edgar et al. "UCHIME improves sensitivity and speed of chimera
//!   detection." Bioinformatics 27, 2194–2200 (2011).
//! - DADA2 `removeBimeraDenovo` (Callahan et al. 2016).

use crate::bio::dada2::Asv;

const DEFAULT_MIN_SCORE: f64 = 2.0;
const DEFAULT_MIN_PARENT_ABUNDANCE: f64 = 2.0;
const MIN_SEGMENT_LEN: usize = 3;

/// Result of chimera detection for a single sequence.
#[derive(Debug, Clone)]
pub struct ChimeraResult {
    /// Index of the query sequence.
    pub query_idx: usize,
    /// Whether this sequence is flagged as chimeric.
    pub is_chimera: bool,
    /// Chimera score (higher = more chimeric).
    pub score: f64,
    /// Index of the left parent (in the input list), if chimeric.
    pub left_parent: Option<usize>,
    /// Index of the right parent (in the input list), if chimeric.
    pub right_parent: Option<usize>,
    /// Crossover position (0-based index in the alignment).
    pub crossover: Option<usize>,
}

/// Parameters for chimera detection.
#[derive(Debug, Clone)]
pub struct ChimeraParams {
    /// Minimum chimera score to flag a sequence. Default: 2.0.
    pub min_score: f64,
    /// Minimum fold-abundance of a parent relative to the query.
    /// A parent must be at least this many times more abundant. Default: 2.0.
    pub min_parent_fold: f64,
    /// Minimum number of differences between a query and each parent
    /// in their respective non-matching segments. Default: 3.
    pub min_diffs: usize,
}

impl Default for ChimeraParams {
    fn default() -> Self {
        Self {
            min_score: DEFAULT_MIN_SCORE,
            min_parent_fold: DEFAULT_MIN_PARENT_ABUNDANCE,
            min_diffs: MIN_SEGMENT_LEN,
        }
    }
}

/// Statistics from chimera detection.
#[derive(Debug, Clone)]
pub struct ChimeraStats {
    /// Total sequences evaluated.
    pub input_sequences: usize,
    /// Number of chimeras detected.
    pub chimeras_found: usize,
    /// Number of non-chimeric sequences retained.
    pub retained: usize,
}

/// Detect chimeras in a set of ASVs (or abundance-sorted unique sequences).
///
/// Sequences must be sorted by abundance (descending). Returns chimera
/// results for each sequence and the filtered non-chimeric sequences.
#[allow(clippy::cast_precision_loss)]
pub fn detect_chimeras(
    seqs: &[Asv],
    params: &ChimeraParams,
) -> (Vec<ChimeraResult>, ChimeraStats) {
    let n = seqs.len();
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        if i < 2 {
            // First two sequences cannot be chimeras (no two parents available)
            results.push(ChimeraResult {
                query_idx: i,
                is_chimera: false,
                score: 0.0,
                left_parent: None,
                right_parent: None,
                crossover: None,
            });
            continue;
        }

        let query = &seqs[i];

        // Find chimera among more-abundant parents
        let parents: Vec<usize> = (0..i)
            .filter(|&j| {
                seqs[j].abundance as f64 >= query.abundance as f64 * params.min_parent_fold
            })
            .collect();

        if parents.len() < 2 {
            results.push(ChimeraResult {
                query_idx: i,
                is_chimera: false,
                score: 0.0,
                left_parent: None,
                right_parent: None,
                crossover: None,
            });
            continue;
        }

        let result = test_chimera(query, &seqs, &parents, i, params);
        results.push(result);
    }

    let chimeras_found = results.iter().filter(|r| r.is_chimera).count();
    let stats = ChimeraStats {
        input_sequences: n,
        chimeras_found,
        retained: n - chimeras_found,
    };

    (results, stats)
}

/// Remove chimeric sequences, returning only non-chimeric ASVs.
pub fn remove_chimeras(seqs: &[Asv], params: &ChimeraParams) -> (Vec<Asv>, ChimeraStats) {
    let (results, stats) = detect_chimeras(seqs, params);
    let filtered: Vec<Asv> = results
        .iter()
        .filter(|r| !r.is_chimera)
        .map(|r| seqs[r.query_idx].clone())
        .collect();
    (filtered, stats)
}

/// Test a single query for chimera formation.
fn test_chimera(
    query: &Asv,
    seqs: &[Asv],
    parents: &[usize],
    query_idx: usize,
    params: &ChimeraParams,
) -> ChimeraResult {
    let qseq = &query.sequence;
    let qlen = qseq.len();

    if qlen < MIN_SEGMENT_LEN * 2 {
        return ChimeraResult {
            query_idx,
            is_chimera: false,
            score: 0.0,
            left_parent: None,
            right_parent: None,
            crossover: None,
        };
    }

    let mut best_score = 0.0_f64;
    let mut best_left = None;
    let mut best_right = None;
    let mut best_cross = None;

    // Try all pairs of parents
    for (pi, &pa) in parents.iter().enumerate() {
        let aseq = &seqs[pa].sequence;
        if aseq.len() < qlen {
            continue;
        }

        for &pb in &parents[pi + 1..] {
            let bseq = &seqs[pb].sequence;
            if bseq.len() < qlen {
                continue;
            }

            // Try each crossover point
            for cross in MIN_SEGMENT_LEN..qlen.saturating_sub(MIN_SEGMENT_LEN) {
                // Left from A, right from B
                let score_ab = chimera_score(qseq, aseq, bseq, cross, params.min_diffs);
                if score_ab > best_score {
                    best_score = score_ab;
                    best_left = Some(pa);
                    best_right = Some(pb);
                    best_cross = Some(cross);
                }

                // Left from B, right from A
                let score_ba = chimera_score(qseq, bseq, aseq, cross, params.min_diffs);
                if score_ba > best_score {
                    best_score = score_ba;
                    best_left = Some(pb);
                    best_right = Some(pa);
                    best_cross = Some(cross);
                }
            }
        }
    }

    ChimeraResult {
        query_idx,
        is_chimera: best_score >= params.min_score,
        score: best_score,
        left_parent: best_left,
        right_parent: best_right,
        crossover: best_cross,
    }
}

/// Compute chimera score for query = left_parent[..cross] + right_parent[cross..].
///
/// Score = (mismatches to best single parent) / (mismatches to chimera model).
/// Higher score means the chimera model explains the query much better
/// than any single parent.
#[allow(clippy::cast_precision_loss)]
fn chimera_score(
    query: &[u8],
    left_parent: &[u8],
    right_parent: &[u8],
    crossover: usize,
    min_diffs: usize,
) -> f64 {
    let len = query.len().min(left_parent.len()).min(right_parent.len());
    if crossover >= len {
        return 0.0;
    }

    // Count mismatches in left segment
    let mut left_match_l = 0_usize; // matches with left_parent in left segment
    let mut left_match_r = 0_usize; // matches with right_parent in left segment
    for i in 0..crossover {
        if query[i] == left_parent[i] {
            left_match_l += 1;
        }
        if query[i] == right_parent[i] {
            left_match_r += 1;
        }
    }

    // Count mismatches in right segment
    let mut right_match_l = 0_usize;
    let mut right_match_r = 0_usize;
    for i in crossover..len {
        if query[i] == left_parent[i] {
            right_match_l += 1;
        }
        if query[i] == right_parent[i] {
            right_match_r += 1;
        }
    }

    // Chimera model: left from left_parent, right from right_parent
    let chimera_matches = left_match_l + right_match_r;
    let chimera_mismatches = len - chimera_matches;

    // Best single parent: whichever parent matches more overall
    let parent_l_total = left_match_l + right_match_l;
    let parent_r_total = left_match_r + right_match_r;
    let best_single = parent_l_total.max(parent_r_total);
    let best_single_mismatches = len - best_single;

    // The chimera model should show distinct parent contributions
    let _left_diffs = crossover - left_match_l.min(crossover);
    let _right_diffs = (len - crossover) - right_match_r.min(len - crossover);
    if left_match_l <= left_match_r || right_match_r <= right_match_l {
        return 0.0;
    }

    // Need minimum differences from the "wrong" parent in each segment
    let wrong_left = crossover.saturating_sub(left_match_r);
    let wrong_right = (len - crossover).saturating_sub(right_match_l);
    if wrong_left < min_diffs || wrong_right < min_diffs {
        return 0.0;
    }

    if chimera_mismatches == 0 {
        if best_single_mismatches > 0 {
            return best_single_mismatches as f64 + 1.0;
        }
        return 0.0;
    }

    best_single_mismatches as f64 / chimera_mismatches as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asv(seq: &[u8], abundance: usize) -> Asv {
        Asv {
            sequence: seq.to_vec(),
            abundance,
            n_members: 1,
        }
    }

    #[test]
    fn no_chimera_in_small_set() {
        let asvs = vec![make_asv(b"AAAAAAAAAA", 100)];
        let (results, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert_eq!(stats.chimeras_found, 0);
        assert_eq!(stats.retained, 1);
        assert!(!results[0].is_chimera);
    }

    #[test]
    fn two_distinct_sequences_no_chimera() {
        let asvs = vec![
            make_asv(b"AAAAAAAAAA", 1000),
            make_asv(b"CCCCCCCCCC", 500),
        ];
        let (_, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert_eq!(stats.chimeras_found, 0);
        assert_eq!(stats.retained, 2);
    }

    #[test]
    fn obvious_chimera_detected() {
        // Parent A: AAAAAACCCCCC (12 bases)
        // Parent B: CCCCCCAAAAAA (12 bases)
        // Chimera:  AAAAAAAAAAAA (left from A, right from B) — nah, that's all the same
        //
        // Better: Parent A: AAAAAGGGGG, Parent B: CCCCCTTTTTT
        //         Chimera:  AAAAATTTTT (left from A, right from B)
        let parent_a = b"AAAAAGGGGG";
        let parent_b = b"CCCCCTTTTTT";
        let chimera = b"AAAAATTTTT"; // left 5 from A, right 5 from B

        let asvs = vec![
            make_asv(parent_a, 1000),
            make_asv(parent_b, 800),
            make_asv(chimera, 10),
        ];

        let (results, stats) = detect_chimeras(&asvs, &ChimeraParams::default());

        // The chimera should be detected
        assert!(
            results[2].is_chimera,
            "chimera not detected, score={}",
            results[2].score
        );
        assert_eq!(stats.chimeras_found, 1);
    }

    #[test]
    fn real_looking_chimera() {
        // Two 20bp parent sequences that differ in the first and second half
        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
        // Chimera: first 10 from A, last 10 from B
        let chimera = b"ACGTACGTACACGTACGTAC";

        let asvs = vec![
            make_asv(parent_a, 5000),
            make_asv(parent_b, 3000),
            make_asv(chimera, 50),
        ];

        let (results, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert!(results[2].is_chimera, "score={}", results[2].score);
        assert_eq!(stats.chimeras_found, 1);
    }

    #[test]
    fn non_chimeric_variant_not_flagged() {
        // A sequence with minor variation should not be flagged
        let parent_a = b"AAAAAAAAAA";
        let parent_b = b"CCCCCCCCCC";
        let variant = b"AAAAAAAAAT"; // 1-base variant of parent_a

        let asvs = vec![
            make_asv(parent_a, 1000),
            make_asv(parent_b, 800),
            make_asv(variant, 50),
        ];

        let (results, _) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert!(!results[2].is_chimera);
    }

    #[test]
    fn remove_chimeras_filters_correctly() {
        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
        let chimera = b"ACGTACGTACACGTACGTAC";

        let asvs = vec![
            make_asv(parent_a, 5000),
            make_asv(parent_b, 3000),
            make_asv(chimera, 50),
        ];

        let (filtered, stats) = remove_chimeras(&asvs, &ChimeraParams::default());
        assert_eq!(stats.chimeras_found, 1);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn empty_input() {
        let (results, stats) = detect_chimeras(&[], &ChimeraParams::default());
        assert!(results.is_empty());
        assert_eq!(stats.chimeras_found, 0);
    }

    #[test]
    fn min_parent_fold_filter() {
        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
        let chimera = b"ACGTACGTACACGTACGTAC";

        // If chimera abundance is similar to parents, parents don't qualify
        let asvs = vec![
            make_asv(parent_a, 100),
            make_asv(parent_b, 100),
            make_asv(chimera, 90),
        ];

        let params = ChimeraParams {
            min_parent_fold: 2.0,
            ..ChimeraParams::default()
        };
        let (results, _) = detect_chimeras(&asvs, &params);
        // Parents aren't 2x more abundant, so no chimera detection
        assert!(!results[2].is_chimera);
    }
}
