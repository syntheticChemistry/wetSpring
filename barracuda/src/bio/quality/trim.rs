// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quality trim primitives: leading, trailing, sliding window.
//!
//! Low-level functions used by [`super::trim_read`] and [`super::filter_reads_flat`].

/// Trim low-quality leading bases from a read.
///
/// Removes bases from the 5' end where quality < `min_quality`.
/// Returns the index of the first base that passes.
#[must_use]
pub(super) fn trim_leading(quality: &[u8], min_quality: u8, phred_offset: u8) -> usize {
    quality
        .iter()
        .position(|&q| q.saturating_sub(phred_offset) >= min_quality)
        .unwrap_or(quality.len())
}

/// Trim low-quality trailing bases from a read.
///
/// Removes bases from the 3' end where quality < `min_quality`.
/// Returns the index one past the last base that passes.
#[must_use]
pub(super) fn trim_trailing(quality: &[u8], min_quality: u8, phred_offset: u8) -> usize {
    quality
        .iter()
        .rposition(|&q| q.saturating_sub(phred_offset) >= min_quality)
        .map_or(0, |i| i + 1)
}

/// Sliding window quality trim from 5' to 3'.
///
/// Scans the read with a window of `window_size` bases. When the average
/// quality within the window drops below `min_quality`, the read is
/// truncated at that position.
///
/// Returns the trim position (index one past the last retained base).
#[must_use]
#[expect(clippy::cast_precision_loss)] // window sizes are small
pub(super) fn trim_sliding_window(
    quality: &[u8],
    window_size: usize,
    min_quality: u8,
    phred_offset: u8,
) -> usize {
    if quality.len() < window_size {
        // Can't form a full window — check the whole read as one window
        let avg: f64 = quality
            .iter()
            .map(|&q| f64::from(q.saturating_sub(phred_offset)))
            .sum::<f64>()
            / quality.len() as f64;
        return if avg >= f64::from(min_quality) {
            quality.len()
        } else {
            0
        };
    }

    // Initial window sum
    let mut window_sum: u32 = quality[..window_size]
        .iter()
        .map(|&q| u32::from(q.saturating_sub(phred_offset)))
        .sum();

    #[expect(clippy::cast_possible_truncation)] // read lengths are always small
    let threshold = u32::from(min_quality) * window_size as u32;

    if window_sum < threshold {
        return 0;
    }

    for i in 1..=(quality.len() - window_size) {
        // Slide: remove left, add right
        window_sum -= u32::from(quality[i - 1].saturating_sub(phred_offset));
        window_sum += u32::from(quality[i + window_size - 1].saturating_sub(phred_offset));

        if window_sum < threshold {
            return i;
        }
    }

    quality.len()
}
