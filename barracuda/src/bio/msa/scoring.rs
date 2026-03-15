// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scoring utilities for MSA.
//!
//! Score-to-distance conversion for guide tree construction. Substitution
//! matrices and profile-profile scoring are reserved for future refinement.

/// Convert pairwise alignment scores to distances for tree building.
///
/// Uses `distance = max_score - score` so higher similarity yields lower distance.
#[must_use]
pub(super) fn scores_to_distances(scores: &[i32], n: usize) -> Vec<f64> {
    let max_score = scores.iter().copied().max().unwrap_or(0);
    let mut dist = vec![0.0_f64; n * n];
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = f64::from(max_score - scores[idx]);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
            idx += 1;
        }
    }
    dist
}
