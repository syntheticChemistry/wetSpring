// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference-coordinate pileup from aligned reads.
//!
//! Walks SAM records sorted by position and accumulates per-position
//! base counts, quality sums, and strand information. The GPU composition
//! target is `Tensor::scan` (prefix sum) for cumulative coverage tracking.
//!
//! # Pipeline Position
//!
//! ```text
//! SAM records → pileup → variant_caller
//! ```

#[cfg(test)]
mod tests;

use crate::io::sam::{CigarType, SamRecord};

#[cfg(feature = "gpu")]
use barracuda::device::WgpuDevice;
#[cfg(feature = "gpu")]
use barracuda::tensor::Tensor;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Per-position pileup column.
#[derive(Debug, Clone, Default)]
pub struct PileupColumn {
    /// 0-based reference position.
    pub position: usize,
    /// Total read depth at this position.
    pub depth: u32,
    /// Base counts: A, C, G, T, N.
    pub base_counts: [u32; 5],
    /// Sum of quality scores per base: A, C, G, T, N.
    pub quality_sums: [u64; 5],
    /// Forward strand depth.
    pub forward_depth: u32,
    /// Reverse strand depth.
    pub reverse_depth: u32,
    /// Number of insertions starting at this position.
    pub insertions: u32,
    /// Number of deletions spanning this position.
    pub deletions: u32,
}

impl PileupColumn {
    /// Reference allele frequency (most common base).
    #[must_use]
    pub fn major_allele_frequency(&self) -> f64 {
        if self.depth == 0 {
            return 0.0;
        }
        let max_count = self.base_counts.iter().max().copied().unwrap_or(0);
        f64::from(max_count) / f64::from(self.depth)
    }

    /// The most common base at this position.
    #[must_use]
    pub fn major_allele(&self) -> u8 {
        let bases = [b'A', b'C', b'G', b'T', b'N'];
        let max_idx = self
            .base_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map_or(0, |(i, _)| i);
        bases[max_idx]
    }

    /// Mean base quality at this position.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Precision: quality sums bounded by coverage"
    )]
    pub fn mean_quality(&self) -> f64 {
        if self.depth == 0 {
            return 0.0;
        }
        let total_qual: u64 = self.quality_sums.iter().sum();
        total_qual as f64 / f64::from(self.depth)
    }

    /// Strand bias ratio (forward / total). 0.5 = balanced.
    #[must_use]
    pub fn strand_bias(&self) -> f64 {
        if self.depth == 0 {
            return 0.5;
        }
        f64::from(self.forward_depth) / f64::from(self.depth)
    }
}

const fn base_to_idx(b: u8) -> usize {
    match b.to_ascii_uppercase() {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4, // N
    }
}

/// Generate pileup from sorted SAM records.
///
/// Records should be sorted by position (as in coordinate-sorted SAM/BAM).
/// Returns pileup columns only for positions with non-zero depth.
///
/// # Arguments
///
/// * `records` - SAM records (should be position-sorted for efficiency)
/// * `ref_len` - Length of the reference sequence
#[must_use]
pub fn generate_pileup(records: &[SamRecord], ref_len: usize) -> Vec<PileupColumn> {
    let mut columns: Vec<PileupColumn> = (0..ref_len)
        .map(|i| PileupColumn {
            position: i,
            ..PileupColumn::default()
        })
        .collect();

    for record in records {
        if !record.is_mapped() || record.pos == 0 {
            continue;
        }

        let is_reverse = record.is_reverse();
        let ref_start = (record.pos - 1) as usize; // convert 1-based to 0-based
        let mut ref_pos = ref_start;
        let mut query_pos = 0usize;

        for op in &record.cigar {
            let len = op.len as usize;
            match op.op {
                CigarType::Match | CigarType::SeqMatch | CigarType::SeqMismatch => {
                    for i in 0..len {
                        if ref_pos + i < ref_len && query_pos + i < record.seq.len() {
                            let col = &mut columns[ref_pos + i];
                            let base = record.seq[query_pos + i];
                            let qual = if query_pos + i < record.qual.len() {
                                record.qual[query_pos + i].saturating_sub(33) // Phred33
                            } else {
                                30 // default Q30
                            };
                            let idx = base_to_idx(base);
                            col.depth += 1;
                            col.base_counts[idx] += 1;
                            col.quality_sums[idx] += u64::from(qual);
                            if is_reverse {
                                col.reverse_depth += 1;
                            } else {
                                col.forward_depth += 1;
                            }
                        }
                    }
                    ref_pos += len;
                    query_pos += len;
                }
                CigarType::Insertion => {
                    if ref_pos > 0 && ref_pos - 1 < ref_len {
                        columns[ref_pos - 1].insertions += 1;
                    }
                    query_pos += len;
                }
                CigarType::Deletion => {
                    for i in 0..len {
                        if ref_pos + i < ref_len {
                            columns[ref_pos + i].deletions += 1;
                        }
                    }
                    ref_pos += len;
                }
                CigarType::SoftClip => {
                    query_pos += len;
                }
                CigarType::HardClip | CigarType::Padding => {}
                CigarType::Skip => {
                    ref_pos += len;
                }
            }
        }
    }

    // Filter to positions with coverage (depth, deletions, or insertions)
    columns.retain(|c| c.depth > 0 || c.deletions > 0 || c.insertions > 0);
    columns
}

/// Compute coverage statistics from pileup.
#[derive(Debug, Clone)]
pub struct CoverageStats {
    /// Total reference positions with depth > 0.
    pub covered_positions: usize,
    /// Mean depth across covered positions.
    pub mean_depth: f64,
    /// Median depth across covered positions.
    pub median_depth: u32,
    /// Maximum depth.
    pub max_depth: u32,
    /// Fraction of reference covered.
    pub coverage_fraction: f64,
}

/// Compute coverage statistics from pileup columns.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "Precision: position counts bounded by genome"
)]
pub fn coverage_stats(columns: &[PileupColumn], ref_len: usize) -> CoverageStats {
    if columns.is_empty() {
        return CoverageStats {
            covered_positions: 0,
            mean_depth: 0.0,
            median_depth: 0,
            max_depth: 0,
            coverage_fraction: 0.0,
        };
    }

    let mut depths: Vec<u32> = columns.iter().map(|c| c.depth).collect();
    depths.sort_unstable();

    let total_depth: u64 = depths.iter().map(|&d| u64::from(d)).sum();
    let mean = total_depth as f64 / depths.len() as f64;
    let median = depths[depths.len() / 2];
    let max = depths.last().copied().unwrap_or(0);

    CoverageStats {
        covered_positions: columns.len(),
        mean_depth: mean,
        median_depth: median,
        max_depth: max,
        coverage_fraction: columns.len() as f64 / ref_len as f64,
    }
}

// ── GPU-accelerated cumulative coverage via Tensor::scan ─────────

/// Compute cumulative depth track on GPU using `Tensor::scan` (prefix sum).
///
/// Takes per-position depth values and returns inclusive prefix sums,
/// useful for rapid coverage queries over windows.
///
/// # Errors
///
/// Returns [`crate::error::Error::Gpu`] if GPU tensor creation or scan fails.
#[cfg(feature = "gpu")]
pub fn cumulative_coverage_gpu(
    columns: &[PileupColumn],
    device: &Arc<WgpuDevice>,
) -> crate::error::Result<Vec<f64>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }

    let depths: Vec<f32> = columns.iter().map(|c| c.depth as f32).collect();
    let shape = vec![depths.len()];

    let tensor = Tensor::from_data(&depths, shape, Arc::clone(device))
        .map_err(|e| crate::error::Error::Gpu(format!("scan tensor create: {e}")))?;

    let scanned = tensor
        .scan(false) // inclusive prefix sum
        .map_err(|e| crate::error::Error::Gpu(format!("scan dispatch: {e}")))?;

    let result_f32 = scanned
        .to_vec()
        .map_err(|e| crate::error::Error::Gpu(format!("scan readback: {e}")))?;

    Ok(result_f32.into_iter().map(f64::from).collect())
}

/// Compute per-base coverage array for the full reference on GPU.
///
/// Returns a dense f64 vector of length `ref_len` with cumulative depth
/// at each position. Positions without coverage have value 0.
///
/// # Errors
///
/// Returns [`crate::error::Error::Gpu`] if GPU dispatch fails.
#[cfg(feature = "gpu")]
pub fn full_coverage_track_gpu(
    columns: &[PileupColumn],
    ref_len: usize,
    device: &Arc<WgpuDevice>,
) -> crate::error::Result<Vec<f64>> {
    let mut depth_track: Vec<f32> = vec![0.0; ref_len];
    for col in columns {
        if col.position < ref_len {
            depth_track[col.position] = col.depth as f32;
        }
    }

    let shape = vec![ref_len];
    let tensor = Tensor::from_data(&depth_track, shape, Arc::clone(device))
        .map_err(|e| crate::error::Error::Gpu(format!("coverage tensor create: {e}")))?;

    let scanned = tensor
        .scan(false)
        .map_err(|e| crate::error::Error::Gpu(format!("coverage scan: {e}")))?;

    let result_f32 = scanned
        .to_vec()
        .map_err(|e| crate::error::Error::Gpu(format!("coverage readback: {e}")))?;

    Ok(result_f32.into_iter().map(f64::from).collect())
}
