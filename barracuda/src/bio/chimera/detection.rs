// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chimera detection algorithm (UCHIME-style).
//!
//! For each query sequence, uses k-mer sketch to find top-K parent candidates,
//! then evaluates crossover points using prefix-sum match vectors for O(1)
//! score evaluation per crossover.

use crate::bio::dada2::Asv;
use crate::cast::{u32_usize, usize_f64};

use super::kmer_sketch::{KmerSketch, build_sketch, sketch_similarity};
use super::{ChimeraParams, ChimeraResult};

const MAX_PARENT_CANDIDATES: usize = 8;

/// Detect chimeras in a set of ASVs (or abundance-sorted unique sequences).
///
/// Sequences must be sorted by abundance (descending). Returns chimera
/// results for each sequence and the filtered non-chimeric sequences.
#[must_use]
pub fn detect_chimeras(
    seqs: &[Asv],
    params: &ChimeraParams,
) -> (Vec<ChimeraResult>, super::ChimeraStats) {
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
                usize_f64(seqs[j].abundance) >= usize_f64(query.abundance) * params.min_parent_fold
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
    let stats = super::ChimeraStats {
        input_sequences: n,
        chimeras_found,
        retained: n - chimeras_found,
    };

    (results, stats)
}

/// Remove chimeric sequences, returning only non-chimeric ASVs.
#[must_use]
pub fn remove_chimeras(seqs: &[Asv], params: &ChimeraParams) -> (Vec<Asv>, super::ChimeraStats) {
    let (results, stats) = detect_chimeras(seqs, params);
    let filtered: Vec<Asv> = results
        .iter()
        .filter(|r| !r.is_chimera)
        .map(|r| seqs[r.query_idx].clone()) // ownership transfer: borrowed input requires clone
        .collect();
    (filtered, stats)
}

/// Test a single query for chimera formation using prefix-sum optimization.
///
/// For each parent pair, precomputes cumulative match vectors once (O(L)),
/// then evaluates all crossover points in O(1) each. Early termination
/// when score exceeds threshold.
pub fn test_chimera_fast(
    query: &Asv,
    seqs: &[Asv],
    candidates: &[usize],
    query_idx: usize,
    params: &ChimeraParams,
) -> ChimeraResult {
    let qseq = &query.sequence;
    let qlen = qseq.len();

    if qlen < super::MIN_SEGMENT_LEN * 2 {
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
            for cross in super::MIN_SEGMENT_LEN..len.saturating_sub(super::MIN_SEGMENT_LEN) {
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
fn score_from_prefix(
    cum_left: &[u32],
    cum_right: &[u32],
    crossover: usize,
    len: usize,
    min_diffs: usize,
) -> f64 {
    let left_match_l = u32_usize(cum_left[crossover]);
    let left_match_r = u32_usize(cum_right[crossover]);
    let right_match_l = u32_usize(cum_left[len] - cum_left[crossover]);
    let right_match_r = u32_usize(cum_right[len] - cum_right[crossover]);

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
            return usize_f64(best_single_mismatches) + 1.0;
        }
        return 0.0;
    }

    usize_f64(best_single_mismatches) / usize_f64(chimera_mismatches)
}
