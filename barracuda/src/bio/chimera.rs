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
//! 2. For each query, use k-mer sketch to find the top-K most similar
//!    parents from the pool of more-abundant sequences.
//! 3. For the top parent pairs, compute chimera scores using prefix-sum
//!    match vectors for O(1) crossover evaluation.
//! 4. If the best chimera score exceeds the threshold, flag the sequence.
//!
//! # Performance
//!
//! Uses k-mer sketching (8-mers) for parent candidate selection instead
//! of exhaustive all-pairs. For N ASVs with K candidate parents per query
//! and L base sequence length:
//!
//! - Previous: O(N × P² × L) where P ≈ N → O(N³L)
//! - Current:  O(N × K² × L) where K = min(8, P) → O(N × 64 × L)
//!
//! # References
//!
//! - Edgar et al. "UCHIME improves sensitivity and speed of chimera
//!   detection." Bioinformatics 27, 2194–2200 (2011).
//! - DADA2 `removeBimeraDenovo` (Callahan et al. 2016).

use crate::bio::dada2::Asv;
use std::collections::HashMap;

const DEFAULT_MIN_SCORE: f64 = 2.0;
const DEFAULT_MIN_PARENT_ABUNDANCE: f64 = 2.0;
const MIN_SEGMENT_LEN: usize = 3;
const SKETCH_K: usize = 8;
const MAX_PARENT_CANDIDATES: usize = 8;

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

type KmerSketch = HashMap<u64, u16>;

/// Build a k-mer sketch (8-mer presence counts) for a DNA sequence.
fn build_sketch(seq: &[u8]) -> KmerSketch {
    let mut sketch = HashMap::new();
    if seq.len() < SKETCH_K {
        return sketch;
    }
    let mut kmer = 0_u64;
    let mask = (1_u64 << (2 * SKETCH_K)) - 1;
    let mut valid = 0_usize;

    for (i, &b) in seq.iter().enumerate() {
        let enc = match b {
            b'A' | b'a' => 0_u64,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => {
                valid = 0;
                continue;
            }
        };
        kmer = ((kmer << 2) | enc) & mask;
        valid += 1;
        if valid >= SKETCH_K {
            *sketch.entry(kmer).or_insert(0) += 1;
            if i >= seq.len().saturating_sub(1) {
                break;
            }
        }
    }
    sketch
}

/// Count shared k-mers between two sketches (Jaccard-like similarity).
#[must_use]
fn sketch_similarity(a: &KmerSketch, b: &KmerSketch) -> u32 {
    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    smaller
        .iter()
        .filter_map(|(k, &va)| larger.get(k).map(|&vb| u32::from(va.min(vb))))
        .sum()
}

/// Detect chimeras in a set of ASVs (or abundance-sorted unique sequences).
///
/// Sequences must be sorted by abundance (descending). Returns chimera
/// results for each sequence and the filtered non-chimeric sequences.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn detect_chimeras(seqs: &[Asv], params: &ChimeraParams) -> (Vec<ChimeraResult>, ChimeraStats) {
    let n = seqs.len();
    let mut results = Vec::with_capacity(n);

    // Precompute k-mer sketches for all sequences
    let sketches: Vec<KmerSketch> = seqs.iter().map(|a| build_sketch(&a.sequence)).collect();

    for i in 0..n {
        if i < 2 {
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

        // Find eligible parents (more abundant by min_parent_fold)
        let eligible: Vec<usize> = (0..i)
            .filter(|&j| {
                seqs[j].abundance as f64 >= query.abundance as f64 * params.min_parent_fold
            })
            .collect();

        if eligible.len() < 2 {
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

        // K-mer sketch: rank parents by shared k-mer count with query
        let query_sketch = &sketches[i];
        let mut scored: Vec<(usize, u32)> = eligible
            .iter()
            .map(|&j| (j, sketch_similarity(query_sketch, &sketches[j])))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let top_k = scored.len().min(MAX_PARENT_CANDIDATES);
        let candidates: Vec<usize> = scored[..top_k].iter().map(|&(j, _)| j).collect();

        let result = test_chimera_fast(query, seqs, &candidates, i, params);
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
#[must_use]
pub fn remove_chimeras(seqs: &[Asv], params: &ChimeraParams) -> (Vec<Asv>, ChimeraStats) {
    let (results, stats) = detect_chimeras(seqs, params);
    let filtered: Vec<Asv> = results
        .iter()
        .filter(|r| !r.is_chimera)
        .map(|r| seqs[r.query_idx].clone())
        .collect();
    (filtered, stats)
}

/// Test a single query for chimera formation using prefix-sum optimization.
///
/// For each parent pair, precomputes cumulative match vectors once (O(L)),
/// then evaluates all crossover points in O(1) each. Early termination
/// when score exceeds threshold.
#[allow(clippy::cast_precision_loss)]
fn test_chimera_fast(
    query: &Asv,
    seqs: &[Asv],
    candidates: &[usize],
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

    for (pi, &pa) in candidates.iter().enumerate() {
        let aseq = &seqs[pa].sequence;
        if aseq.len() < qlen {
            continue;
        }

        for &pb in &candidates[pi + 1..] {
            let bseq = &seqs[pb].sequence;
            if bseq.len() < qlen {
                continue;
            }

            let len = qlen.min(aseq.len()).min(bseq.len());

            // Precompute cumulative match counts (O(L), done once per pair)
            let mut cum_a = vec![0_u32; len + 1]; // cumulative matches with parent A
            let mut cum_b = vec![0_u32; len + 1]; // cumulative matches with parent B
            for j in 0..len {
                cum_a[j + 1] = cum_a[j] + u32::from(qseq[j] == aseq[j]);
                cum_b[j + 1] = cum_b[j] + u32::from(qseq[j] == bseq[j]);
            }

            // Evaluate all crossover points in O(1) each
            for cross in MIN_SEGMENT_LEN..len.saturating_sub(MIN_SEGMENT_LEN) {
                // Left from A, right from B
                let s_ab = score_from_prefix(&cum_a, &cum_b, cross, len, params.min_diffs);
                if s_ab > best_score {
                    best_score = s_ab;
                    best_left = Some(pa);
                    best_right = Some(pb);
                    best_cross = Some(cross);
                    if best_score >= params.min_score {
                        return ChimeraResult {
                            query_idx,
                            is_chimera: true,
                            score: best_score,
                            left_parent: best_left,
                            right_parent: best_right,
                            crossover: best_cross,
                        };
                    }
                }

                // Left from B, right from A
                let s_ba = score_from_prefix(&cum_b, &cum_a, cross, len, params.min_diffs);
                if s_ba > best_score {
                    best_score = s_ba;
                    best_left = Some(pb);
                    best_right = Some(pa);
                    best_cross = Some(cross);
                    if best_score >= params.min_score {
                        return ChimeraResult {
                            query_idx,
                            is_chimera: true,
                            score: best_score,
                            left_parent: best_left,
                            right_parent: best_right,
                            crossover: best_cross,
                        };
                    }
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

/// O(1) chimera score evaluation using precomputed prefix sums.
#[allow(clippy::cast_precision_loss)]
fn score_from_prefix(
    cum_left: &[u32],
    cum_right: &[u32],
    crossover: usize,
    len: usize,
    min_diffs: usize,
) -> f64 {
    let left_match_l = cum_left[crossover] as usize;
    let left_match_r = cum_right[crossover] as usize;
    let right_match_l = (cum_left[len] - cum_left[crossover]) as usize;
    let right_match_r = (cum_right[len] - cum_right[crossover]) as usize;

    // Chimera model: left from left_parent, right from right_parent
    let chimera_matches = left_match_l + right_match_r;
    let chimera_mismatches = len - chimera_matches;

    // Best single parent
    let total_from_left_parent = left_match_l + right_match_l;
    let total_from_right_parent = left_match_r + right_match_r;
    let best_single = total_from_left_parent.max(total_from_right_parent);
    let best_single_mismatches = len - best_single;

    // Must show distinct parent contributions in each segment
    if left_match_l <= left_match_r || right_match_r <= right_match_l {
        return 0.0;
    }

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
        let asvs = vec![make_asv(b"AAAAAAAAAA", 1000), make_asv(b"CCCCCCCCCC", 500)];
        let (_, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert_eq!(stats.chimeras_found, 0);
        assert_eq!(stats.retained, 2);
    }

    #[test]
    fn obvious_chimera_detected() {
        let parent_a = b"AAAAAGGGGG";
        let parent_b = b"CCCCCTTTTTT";
        let chimera = b"AAAAATTTTT";

        let asvs = vec![
            make_asv(parent_a, 1000),
            make_asv(parent_b, 800),
            make_asv(chimera, 10),
        ];

        let (results, stats) = detect_chimeras(&asvs, &ChimeraParams::default());
        assert!(
            results[2].is_chimera,
            "chimera not detected, score={}",
            results[2].score
        );
        assert_eq!(stats.chimeras_found, 1);
    }

    #[test]
    fn real_looking_chimera() {
        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
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
        let parent_a = b"AAAAAAAAAA";
        let parent_b = b"CCCCCCCCCC";
        let variant = b"AAAAAAAAAT";

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
        assert!(!results[2].is_chimera);
    }

    #[test]
    fn default_params_values() {
        let p = ChimeraParams::default();
        assert!((p.min_score - DEFAULT_MIN_SCORE).abs() < f64::EPSILON);
        assert!((p.min_parent_fold - DEFAULT_MIN_PARENT_ABUNDANCE).abs() < f64::EPSILON);
        assert_eq!(p.min_diffs, MIN_SEGMENT_LEN);
    }

    #[test]
    fn high_min_score_suppresses_detection() {
        let parent_a = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let parent_b = b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC";
        let chimera = b"AAAAAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCCCCC";

        let asvs = vec![
            make_asv(parent_a, 1000),
            make_asv(parent_b, 1000),
            make_asv(chimera, 10),
        ];

        let strict = ChimeraParams {
            min_score: 100.0,
            ..ChimeraParams::default()
        };
        let (results, _) = detect_chimeras(&asvs, &strict);
        assert!(
            !results[2].is_chimera,
            "high min_score should suppress detection"
        );
    }
}
