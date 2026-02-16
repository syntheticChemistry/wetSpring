// SPDX-License-Identifier: AGPL-3.0-or-later
//! MS2 spectral similarity scoring for compound identification.
//!
//! Computes cosine similarity between MS2 spectra, the standard metric
//! for spectral library matching in metabolomics and PFAS screening.
//!
//! # Algorithm
//!
//! 1. Align peaks between query and reference spectra within ±Da tolerance
//! 2. Compute cosine similarity: cos(θ) = (A·B) / (||A|| × ||B||)
//!    where A, B are aligned intensity vectors
//! 3. Optionally weight by m/z (Stein & Scott weighting)
//!
//! # References
//!
//! - Stein, S.E. & Scott, D.R. (1994). J. Am. Soc. Mass Spectrom. 5: 859–866.
//! - `matchms` Python library (Netherlands eScience Center).
//! - `ToadStool`'s `cosine_similarity_f64.wgsl` for GPU promotion.

/// Result of a spectral similarity comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchResult {
    /// Cosine similarity score \[0, 1\].
    pub score: f64,
    /// Number of matched peak pairs.
    pub matched_peaks: usize,
    /// Indices of matched peaks in the query spectrum.
    pub query_indices: Vec<usize>,
    /// Indices of matched peaks in the reference spectrum.
    pub reference_indices: Vec<usize>,
}

/// Compute cosine similarity between two MS2 spectra.
///
/// Peaks are matched greedily by closest m/z within `tolerance_da`.
/// Unmatched peaks are treated as zero intensity in the other spectrum.
///
/// # Arguments
///
/// * `query_mz`, `query_intensity` — Query spectrum (m/z, intensity pairs).
/// * `ref_mz`, `ref_intensity` — Reference spectrum.
/// * `tolerance_da` — Maximum m/z difference for peak matching (Daltons).
///
/// # Returns
///
/// [`MatchResult`] with cosine score and matched peak info.
#[must_use]
pub fn cosine_similarity(
    query_mz: &[f64],
    query_intensity: &[f64],
    ref_mz: &[f64],
    ref_intensity: &[f64],
    tolerance_da: f64,
) -> MatchResult {
    if query_mz.is_empty() || ref_mz.is_empty() {
        return MatchResult {
            score: 0.0,
            matched_peaks: 0,
            query_indices: vec![],
            reference_indices: vec![],
        };
    }

    // Greedy matching: for each query peak, find closest reference peak
    let mut ref_used = vec![false; ref_mz.len()];
    let mut query_indices = Vec::new();
    let mut reference_indices = Vec::new();

    for (qi, &qmz) in query_mz.iter().enumerate() {
        let mut best_ri = None;
        let mut best_diff = tolerance_da;

        for (ri, &rmz) in ref_mz.iter().enumerate() {
            if ref_used[ri] {
                continue;
            }
            let diff = (qmz - rmz).abs();
            if diff < best_diff {
                best_diff = diff;
                best_ri = Some(ri);
            }
        }

        if let Some(ri) = best_ri {
            ref_used[ri] = true;
            query_indices.push(qi);
            reference_indices.push(ri);
        }
    }

    let matched_peaks = query_indices.len();

    if matched_peaks == 0 {
        return MatchResult {
            score: 0.0,
            matched_peaks: 0,
            query_indices: vec![],
            reference_indices: vec![],
        };
    }

    // Compute cosine similarity on matched peaks
    let mut dot = 0.0;
    let mut norm_q = 0.0;
    let mut norm_r = 0.0;

    for (&qi, &ri) in query_indices.iter().zip(reference_indices.iter()) {
        dot += query_intensity[qi] * ref_intensity[ri];
        norm_q += query_intensity[qi] * query_intensity[qi];
        norm_r += ref_intensity[ri] * ref_intensity[ri];
    }

    // Include unmatched peaks in norms for proper cosine
    for (i, &int) in query_intensity.iter().enumerate() {
        if !query_indices.contains(&i) {
            norm_q += int * int;
        }
    }
    for (i, &int) in ref_intensity.iter().enumerate() {
        if !reference_indices.contains(&i) {
            norm_r += int * int;
        }
    }

    let denom = norm_q.sqrt() * norm_r.sqrt();
    let score = if denom > 0.0 { dot / denom } else { 0.0 };

    MatchResult {
        score,
        matched_peaks,
        query_indices,
        reference_indices,
    }
}

/// Compute modified cosine similarity with Stein & Scott m/z weighting.
///
/// Weight = mz^`mz_power` × intensity^`int_power`
///
/// Standard values: `mz_power` = 0.0 (no m/z weighting) or 2.0 (NIST),
/// `int_power` = 0.5 (square root scaling).
#[must_use]
pub fn cosine_similarity_weighted(
    query_mz: &[f64],
    query_intensity: &[f64],
    ref_mz: &[f64],
    ref_intensity: &[f64],
    tolerance_da: f64,
    mz_power: f64,
    int_power: f64,
) -> MatchResult {
    // Apply weighting
    let weight = |mz: f64, int: f64| -> f64 { mz.powf(mz_power) * int.powf(int_power) };

    let q_weighted: Vec<f64> = query_mz
        .iter()
        .zip(query_intensity.iter())
        .map(|(&mz, &int)| weight(mz, int))
        .collect();

    let r_weighted: Vec<f64> = ref_mz
        .iter()
        .zip(ref_intensity.iter())
        .map(|(&mz, &int)| weight(mz, int))
        .collect();

    cosine_similarity(query_mz, &q_weighted, ref_mz, &r_weighted, tolerance_da)
}

/// Score all-pairs cosine similarity for a batch of spectra.
///
/// Returns a condensed similarity matrix (same format as
/// [`super::diversity::bray_curtis_condensed`]).
///
/// # Arguments
///
/// * `spectra` — Vector of (mz, intensity) pairs.
/// * `tolerance_da` — Peak matching tolerance.
///
/// # Returns
///
/// Condensed upper-triangle: N*(N-1)/2 similarity scores.
#[must_use]
pub fn pairwise_cosine(spectra: &[(Vec<f64>, Vec<f64>)], tolerance_da: f64) -> Vec<f64> {
    let n = spectra.len();
    let mut result = Vec::with_capacity(n * (n - 1) / 2);

    for i in 1..n {
        for j in 0..i {
            let score = cosine_similarity(
                &spectra[i].0,
                &spectra[i].1,
                &spectra[j].0,
                &spectra[j].1,
                tolerance_da,
            )
            .score;
            result.push(score);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_spectra_score_one() {
        let mz = vec![100.0, 200.0, 300.0];
        let int = vec![50.0, 100.0, 75.0];
        let result = cosine_similarity(&mz, &int, &mz, &int, 0.5);
        assert!((result.score - 1.0).abs() < 1e-10);
        assert_eq!(result.matched_peaks, 3);
    }

    #[test]
    fn orthogonal_spectra_score_zero() {
        let q_mz = vec![100.0, 200.0];
        let q_int = vec![50.0, 100.0];
        let r_mz = vec![500.0, 600.0]; // No overlap
        let r_int = vec![50.0, 100.0];
        let result = cosine_similarity(&q_mz, &q_int, &r_mz, &r_int, 0.5);
        assert!((result.score).abs() < 1e-10);
        assert_eq!(result.matched_peaks, 0);
    }

    #[test]
    fn partial_overlap() {
        let q_mz = vec![100.0, 200.0, 300.0];
        let q_int = vec![50.0, 100.0, 75.0];
        let r_mz = vec![100.0, 200.0, 400.0]; // 300 vs 400 — no match
        let r_int = vec![50.0, 100.0, 75.0];
        let result = cosine_similarity(&q_mz, &q_int, &r_mz, &r_int, 0.5);
        assert!(result.score > 0.5);
        assert!(result.score < 1.0);
        assert_eq!(result.matched_peaks, 2);
    }

    #[test]
    fn tolerance_controls_matching() {
        let q_mz = vec![100.0];
        let q_int = vec![100.0];
        let r_mz = vec![100.3];
        let r_int = vec![100.0];

        // Tight tolerance: no match
        let tight = cosine_similarity(&q_mz, &q_int, &r_mz, &r_int, 0.1);
        assert_eq!(tight.matched_peaks, 0);

        // Loose tolerance: match
        let loose = cosine_similarity(&q_mz, &q_int, &r_mz, &r_int, 0.5);
        assert_eq!(loose.matched_peaks, 1);
        assert!((loose.score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn empty_spectra() {
        let result = cosine_similarity(&[], &[], &[100.0], &[50.0], 0.5);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn weighted_cosine() {
        let mz = vec![100.0, 200.0, 300.0];
        let int = vec![50.0, 100.0, 75.0];
        let result = cosine_similarity_weighted(&mz, &int, &mz, &int, 0.5, 0.0, 0.5);
        assert!((result.score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pairwise_three_spectra() {
        let spectra = vec![
            (vec![100.0, 200.0], vec![50.0, 100.0]),
            (vec![100.0, 200.0], vec![50.0, 100.0]), // identical to first
            (vec![500.0, 600.0], vec![50.0, 100.0]), // no overlap
        ];
        let scores = pairwise_cosine(&spectra, 0.5);
        assert_eq!(scores.len(), 3); // 3*(3-1)/2
        assert!((scores[0] - 1.0).abs() < 1e-10); // (1,0) identical
        assert!(scores[1] < 0.01); // (2,0) no overlap
        assert!(scores[2] < 0.01); // (2,1) no overlap
    }
}
