// SPDX-License-Identifier: AGPL-3.0-or-later
//! FASTQ quality filtering and adapter trimming.
//!
//! Replaces Trimmomatic/Cutadapt for the 16S amplicon pipeline (Exp001/004).
//! Pure Rust — no external dependencies beyond [`crate::io::fastq`].
//!
//! # Operations
//!
//! 1. **Sliding window quality trim**: Scan from 5' to 3' with a fixed-size
//!    window; trim when average window quality drops below threshold.
//! 2. **Leading/trailing low-quality trim**: Remove bases below a threshold
//!    from either end.
//! 3. **Adapter trimming**: Simple semi-global alignment of adapter sequences
//!    against read ends with configurable mismatch tolerance.
//! 4. **Minimum length filter**: Discard reads shorter than a threshold after
//!    trimming.
//!
//! # References
//!
//! - Bolger, A.M. et al. (2014). Trimmomatic. Bioinformatics 30(15): 2114-2120.
//! - Martin, M. (2011). Cutadapt removes adapter sequences. EMBnet.journal 17(1): 10-12.

mod trim;

use crate::io::fastq::FastqRecord;

pub use super::adapter::{find_adapter_3prime, trim_adapter_3prime};

/// Configuration for quality filtering operations.
#[derive(Debug, Clone)]
pub struct QualityParams {
    /// Sliding window size for quality trim (Trimmomatic `SLIDINGWINDOW`).
    /// Default: 4.
    pub window_size: usize,
    /// Minimum average quality within the sliding window.
    /// Default: 20 (Phred33).
    pub window_min_quality: u8,
    /// Minimum quality for leading bases (trim from 5' end).
    /// Default: 3 (Trimmomatic `LEADING`).
    pub leading_min_quality: u8,
    /// Minimum quality for trailing bases (trim from 3' end).
    /// Default: 3 (Trimmomatic `TRAILING`).
    pub trailing_min_quality: u8,
    /// Minimum read length after all trimming.
    /// Default: 36 (Trimmomatic `MINLEN`).
    pub min_length: usize,
    /// Phred encoding offset (33 for Illumina 1.8+, 64 for older).
    /// Default: 33.
    pub phred_offset: u8,
}

/// Manual impl intentional: all fields use non-zero defaults (Trimmomatic-style).
impl Default for QualityParams {
    fn default() -> Self {
        Self {
            window_size: 4,
            window_min_quality: 20,
            leading_min_quality: 3,
            trailing_min_quality: 3,
            min_length: 36,
            phred_offset: 33,
        }
    }
}

/// GPU-compatible quality parameters for uniform buffer binding.
///
/// Maps directly to WGSL `var<uniform>` layout. Uses `u32` fields
/// for GPU alignment (WGSL does not have `u8` or `usize`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QualityGpuParams {
    /// Sliding window size.
    pub window_size: u32,
    /// Minimum average quality within the sliding window.
    pub window_min_quality: u32,
    /// Minimum quality for leading bases.
    pub leading_min_quality: u32,
    /// Minimum quality for trailing bases.
    pub trailing_min_quality: u32,
    /// Minimum read length after all trimming.
    pub min_length: u32,
    /// Phred encoding offset.
    pub phred_offset: u32,
}

impl From<&QualityParams> for QualityGpuParams {
    fn from(p: &QualityParams) -> Self {
        Self {
            window_size: p.window_size as u32,
            window_min_quality: u32::from(p.window_min_quality),
            leading_min_quality: u32::from(p.leading_min_quality),
            trailing_min_quality: u32::from(p.trailing_min_quality),
            min_length: p.min_length as u32,
            phred_offset: u32::from(p.phred_offset),
        }
    }
}

/// Result of quality filtering a batch of reads.
#[derive(Debug, Clone)]
pub struct FilterStats {
    /// Total input reads.
    pub input_reads: usize,
    /// Reads passing all filters.
    pub output_reads: usize,
    /// Reads discarded (too short after trimming).
    pub discarded_reads: usize,
    /// Total bases trimmed from leading end.
    pub leading_bases_trimmed: u64,
    /// Total bases trimmed from trailing end.
    pub trailing_bases_trimmed: u64,
    /// Total bases trimmed by sliding window.
    pub window_bases_trimmed: u64,
    /// Total bases trimmed by adapter removal.
    pub adapter_bases_trimmed: u64,
}

/// Apply all quality trimming operations to a single read.
///
/// Order: leading trim → trailing trim → sliding window → min length check.
///
/// Returns `Some((start, end))` if the read passes, or `None` if it's too short.
#[inline]
#[must_use]
pub fn trim_read(record: &FastqRecord, params: &QualityParams) -> Option<(usize, usize)> {
    let len = record.quality.len();
    if len == 0 {
        return None;
    }

    // 1. Leading trim
    let start = trim::trim_leading(
        &record.quality,
        params.leading_min_quality,
        params.phred_offset,
    );
    if start >= len {
        return None;
    }

    // 2. Trailing trim
    let end = trim::trim_trailing(
        &record.quality[..len],
        params.trailing_min_quality,
        params.phred_offset,
    );
    if end <= start {
        return None;
    }

    // 3. Sliding window on the remaining region
    let sub_quality = &record.quality[start..end];
    let window_end = trim::trim_sliding_window(
        sub_quality,
        params.window_size,
        params.window_min_quality,
        params.phred_offset,
    );
    let final_end = start + window_end;
    if final_end <= start {
        return None;
    }

    // 4. Minimum length check
    let trimmed_len = final_end - start;
    if trimmed_len < params.min_length {
        return None;
    }

    Some((start, final_end))
}

/// Apply a trim result to create a new trimmed record.
#[inline]
#[must_use]
pub fn apply_trim(record: &FastqRecord, start: usize, end: usize) -> FastqRecord {
    FastqRecord {
        id: record.id.clone(), // ownership transfer: borrowed input requires clone
        sequence: record.sequence[start..end].to_vec(),
        quality: record.quality[start..end].to_vec(),
    }
}

/// Filter a batch of reads by quality.
///
/// Applies all trimming operations and returns passing reads + statistics.
#[must_use]
pub fn filter_reads(
    records: &[FastqRecord],
    params: &QualityParams,
) -> (Vec<FastqRecord>, FilterStats) {
    let mut output = Vec::with_capacity(records.len());
    let mut stats = FilterStats {
        input_reads: records.len(),
        output_reads: 0,
        discarded_reads: 0,
        leading_bases_trimmed: 0,
        trailing_bases_trimmed: 0,
        window_bases_trimmed: 0,
        adapter_bases_trimmed: 0,
    };

    for record in records {
        let orig_len = record.sequence.len();

        if let Some((start, end)) = trim_read(record, params) {
            stats.leading_bases_trimmed += start as u64;
            stats.trailing_bases_trimmed += (orig_len - end) as u64;
            output.push(apply_trim(record, start, end));
            stats.output_reads += 1;
        } else {
            stats.discarded_reads += 1;
        }
    }

    (output, stats)
}

/// Flat quality filter results for a batch of reads.
///
/// Parallel arrays (`SoA`) where index `i` corresponds to read `i`.
/// Maps directly to GPU storage buffer output.
#[derive(Debug, Clone)]
pub struct QualityFlatResult {
    /// Trim start position for each read (0-based).
    pub starts: Vec<u32>,
    /// Trim end position for each read (exclusive).
    pub ends: Vec<u32>,
    /// Whether each read passed all filters (1 = pass, 0 = fail).
    pub pass: Vec<u8>,
}

/// Batch quality filter using flat arrays.
///
/// Takes quality scores as contiguous flat arrays with per-read offsets,
/// matching the GPU dispatch model where all reads are in a single buffer.
///
/// # Arguments
///
/// * `qualities` — Concatenated quality bytes for all reads.
/// * `offsets` — Start offset of each read in `qualities`.
/// * `lengths` — Length of each read.
/// * `params` — Quality filtering parameters.
#[must_use]
#[expect(clippy::cast_possible_truncation)]
pub fn filter_reads_flat(
    qualities: &[u8],
    offsets: &[usize],
    lengths: &[usize],
    params: &QualityParams,
) -> QualityFlatResult {
    let n = offsets.len();
    let mut starts = Vec::with_capacity(n);
    let mut ends = Vec::with_capacity(n);
    let mut pass = Vec::with_capacity(n);

    for (&off, &len) in offsets.iter().zip(lengths.iter()) {
        if off + len > qualities.len() || len == 0 {
            starts.push(0);
            ends.push(0);
            pass.push(0);
            continue;
        }
        let qual = &qualities[off..off + len];

        // Leading trim
        let start = trim::trim_leading(qual, params.leading_min_quality, params.phred_offset);
        if start >= len {
            starts.push(0);
            ends.push(0);
            pass.push(0);
            continue;
        }

        // Trailing trim
        let end = trim::trim_trailing(qual, params.trailing_min_quality, params.phred_offset);
        if end <= start {
            starts.push(0);
            ends.push(0);
            pass.push(0);
            continue;
        }

        // Sliding window on remaining region
        let sub_qual = &qual[start..end];
        let window_end = trim::trim_sliding_window(
            sub_qual,
            params.window_size,
            params.window_min_quality,
            params.phred_offset,
        );
        let final_end = start + window_end;

        // Min length check
        starts.push(start as u32);
        ends.push(final_end as u32);
        let passed = final_end > start && (final_end - start) >= params.min_length;
        pass.push(u8::from(passed));
    }

    QualityFlatResult { starts, ends, pass }
}

#[cfg(test)]
#[path = "quality_tests.rs"]
mod tests;
