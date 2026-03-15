// SPDX-License-Identifier: AGPL-3.0-or-later
//! Peak prominence and property calculation.
//!
//! Computes prominence (height above the higher surrounding valley) and
//! width at half-prominence via linear interpolation of threshold crossings.

use super::Peak;

/// Compute prominence and width for a single peak.
pub(super) fn compute_peak_properties(data: &[f64], peak_idx: usize) -> Peak {
    let height = data[peak_idx];

    // Find left valley: scan left to find minimum before a higher peak or boundary
    let mut left_min = height;
    let mut left_base = peak_idx;
    for i in (0..peak_idx).rev() {
        if data[i] < left_min {
            left_min = data[i];
            left_base = i;
        }
        if data[i] > height {
            break;
        }
    }

    // Find right valley: scan right to find minimum before a higher peak or boundary
    let mut right_min = height;
    let mut right_base = peak_idx;
    for (i, &val) in data.iter().enumerate().skip(peak_idx + 1) {
        if val < right_min {
            right_min = val;
            right_base = i;
        }
        if val > height {
            break;
        }
    }

    let reference = left_min.max(right_min);
    let prominence = height - reference;

    // Width at half-prominence
    let half_height = height - prominence / 2.0;
    let left_ips = interpolate_crossing(data, peak_idx, half_height, true);
    let right_ips = interpolate_crossing(data, peak_idx, half_height, false);
    let width = right_ips - left_ips;

    Peak {
        index: peak_idx,
        height,
        prominence,
        left_base,
        right_base,
        width,
        left_ips,
        right_ips,
    }
}

/// Linearly interpolate where the signal crosses a threshold.
#[expect(clippy::cast_precision_loss)] // indices are tiny relative to f64 range
pub(super) fn interpolate_crossing(
    data: &[f64],
    peak_idx: usize,
    threshold: f64,
    go_left: bool,
) -> f64 {
    if go_left {
        for i in (1..=peak_idx).rev() {
            if data[i - 1] <= threshold && data[i] > threshold {
                let frac = (threshold - data[i - 1]) / (data[i] - data[i - 1]);
                return (i - 1) as f64 + frac;
            }
        }
        0.0
    } else {
        for i in peak_idx..data.len().saturating_sub(1) {
            if data[i] > threshold && data[i + 1] <= threshold {
                let frac = (data[i] - threshold) / (data[i] - data[i + 1]);
                return i as f64 + frac;
            }
        }
        (data.len() - 1) as f64
    }
}
