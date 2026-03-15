// SPDX-License-Identifier: AGPL-3.0-or-later
//! Peak detection in 1D signals.
//!
//! Implements `find_peaks` equivalent to `scipy.signal.find_peaks` with
//! configurable height, prominence, width, and distance filters.

use super::prominence;
use super::{Peak, PeakParams};

/// Detect peaks in a 1D signal with configurable filtering.
///
/// Equivalent to `scipy.signal.find_peaks` with `height`, `prominence`,
/// `width`, and `distance` parameters.
///
/// # Arguments
///
/// * `data` — 1D signal (e.g., chromatogram intensities).
/// * `params` — Peak detection parameters.
///
/// # Returns
///
/// Vector of detected [`Peak`] structs, sorted by index.
#[must_use]
pub fn find_peaks(data: &[f64], params: &PeakParams) -> Vec<Peak> {
    if data.len() < 3 {
        return vec![];
    }

    // 1. Find all local maxima
    let mut candidates: Vec<usize> = Vec::new();
    for i in 1..data.len() - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            candidates.push(i);
        }
    }

    // Also handle flat-topped peaks: find runs where consecutive values are equal
    // and both edges are strictly lower. Pick the midpoint.
    #[expect(clippy::float_cmp)] // exact equality is intentional for plateau detection
    let eq = |a: f64, b: f64| a == b;

    let mut i = 1;
    while i < data.len() - 1 {
        if eq(data[i], data[i - 1]) {
            i += 1;
            continue;
        }
        if data[i] > data[i - 1] {
            let start = i;
            while i + 1 < data.len() && eq(data[i + 1], data[i]) {
                i += 1;
            }
            if i + 1 < data.len() && data[i + 1] < data[i] {
                let mid = usize::midpoint(start, i);
                if !candidates.contains(&mid) {
                    candidates.push(mid);
                }
            }
        }
        i += 1;
    }

    candidates.sort_unstable();
    candidates.dedup();

    // 2. Height filter
    if let Some(min_h) = params.min_height {
        candidates.retain(|&idx| data[idx] >= min_h);
    }

    // 3. Compute prominence and width for each candidate
    let mut peaks: Vec<Peak> = candidates
        .iter()
        .map(|&idx| prominence::compute_peak_properties(data, idx))
        .collect();

    // 4. Prominence filter
    if let Some(min_p) = params.min_prominence {
        peaks.retain(|p| p.prominence >= min_p);
    }

    // 5. Width filter
    if let Some(min_w) = params.min_width {
        peaks.retain(|p| p.width >= min_w);
    }
    if let Some(max_w) = params.max_width {
        peaks.retain(|p| p.width <= max_w);
    }

    // 6. Distance filter (keep highest peak within each window)
    if params.distance > 1 {
        peaks = filter_by_distance(peaks, data, params.distance);
    }

    peaks
}

/// Keep only the highest peak within each `distance` window.
pub(super) fn filter_by_distance(mut peaks: Vec<Peak>, data: &[f64], distance: usize) -> Vec<Peak> {
    // Sort by height descending (highest priority first)
    peaks.sort_by(|a, b| {
        data[b.index]
            .partial_cmp(&data[a.index])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![true; peaks.len()];
    for i in 0..peaks.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..peaks.len() {
            if !keep[j] {
                continue;
            }
            let dist = peaks[i].index.abs_diff(peaks[j].index);
            if dist < distance {
                keep[j] = false;
            }
        }
    }

    let mut result: Vec<Peak> = peaks
        .into_iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| p)
        .collect();

    result.sort_by_key(|p| p.index);
    result
}

/// Integrate a single peak's area under the curve via trapezoidal rule.
///
/// Uses the peak's `left_base`/`right_base` indices to slice the signal
/// and x-axis arrays, then delegates to `barracuda::numerical::trapz`.
///
/// # Arguments
///
/// * `x` — X-axis values (e.g. retention time, index positions).
/// * `y` — Y-axis values (signal intensities, same length as `x`).
/// * `peak` — Detected peak whose `left_base`..=`right_base` defines the
///   integration window.
///
/// # Returns
///
/// Integrated area (y-units × x-units), or `0.0` if the window is invalid.
#[must_use]
pub fn integrate_peak(x: &[f64], y: &[f64], peak: &Peak) -> f64 {
    if peak.left_base >= peak.right_base || peak.right_base >= y.len() || x.len() != y.len() {
        return 0.0;
    }
    let x_slice = &x[peak.left_base..=peak.right_base];
    let y_slice = &y[peak.left_base..=peak.right_base];
    barracuda::numerical::trapz(y_slice, x_slice).unwrap_or(0.0)
}

/// Detect peaks and compute their integrated areas in one pass.
///
/// Convenience wrapper that calls [`find_peaks`] followed by
/// [`integrate_peak`] for each detected peak. Returns `(peaks, areas)`
/// where `areas[i]` is the integrated area for `peaks[i]`.
///
/// # Arguments
///
/// * `x` — X-axis values (e.g. retention time).
/// * `y` — Y-axis values (signal intensities).
/// * `params` — Peak detection parameters.
#[must_use]
pub fn find_peaks_with_area(x: &[f64], y: &[f64], params: &PeakParams) -> Vec<(Peak, f64)> {
    let peaks = find_peaks(y, params);
    peaks
        .into_iter()
        .map(|p| {
            let area = integrate_peak(x, y, &p);
            (p, area)
        })
        .collect()
}
