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

use crate::io::fastq::FastqRecord;

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

/// Trim low-quality leading bases from a read.
///
/// Removes bases from the 5' end where quality < `min_quality`.
/// Returns the index of the first base that passes.
#[must_use]
fn trim_leading(quality: &[u8], min_quality: u8, phred_offset: u8) -> usize {
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
fn trim_trailing(quality: &[u8], min_quality: u8, phred_offset: u8) -> usize {
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
#[allow(clippy::cast_precision_loss)] // window sizes are small
fn trim_sliding_window(
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

    #[allow(clippy::cast_possible_truncation)] // read lengths are always small
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
    let start = trim_leading(
        &record.quality,
        params.leading_min_quality,
        params.phred_offset,
    );
    if start >= len {
        return None;
    }

    // 2. Trailing trim
    let end = trim_trailing(
        &record.quality[..len],
        params.trailing_min_quality,
        params.phred_offset,
    );
    if end <= start {
        return None;
    }

    // 3. Sliding window on the remaining region
    let sub_quality = &record.quality[start..end];
    let window_end = trim_sliding_window(
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
        id: record.id.clone(),
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
            #[allow(clippy::cast_possible_truncation)]
            {
                stats.leading_bases_trimmed += start as u64;
                stats.trailing_bases_trimmed += (orig_len - end) as u64;
            }
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
#[allow(clippy::cast_possible_truncation)]
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

    for i in 0..n {
        let off = offsets[i];
        let len = lengths[i];
        if off + len > qualities.len() || len == 0 {
            starts.push(0);
            ends.push(0);
            pass.push(0);
            continue;
        }
        let qual = &qualities[off..off + len];

        // Leading trim
        let start = qual
            .iter()
            .position(|&q| q.saturating_sub(params.phred_offset) >= params.leading_min_quality)
            .unwrap_or(len);
        if start >= len {
            starts.push(0);
            ends.push(0);
            pass.push(0);
            continue;
        }

        // Trailing trim
        let end = qual
            .iter()
            .rposition(|&q| q.saturating_sub(params.phred_offset) >= params.trailing_min_quality)
            .map_or(0, |i| i + 1);
        if end <= start {
            starts.push(0);
            ends.push(0);
            pass.push(0);
            continue;
        }

        // Sliding window on remaining region
        let sub_qual = &qual[start..end];
        let window_end = trim_sliding_window(
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
use super::adapter::bases_match;
pub use super::adapter::{find_adapter_3prime, trim_adapter_3prime};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::naive_bytecount)]
mod tests {
    use super::*;

    fn make_record(seq: &[u8], qual: &[u8]) -> FastqRecord {
        FastqRecord {
            id: "test".to_string(),
            sequence: seq.to_vec(),
            quality: qual.to_vec(),
        }
    }

    fn qual_from_phred(scores: &[u8]) -> Vec<u8> {
        scores.iter().map(|&q| q + 33).collect()
    }

    #[test]
    fn trim_leading_removes_low_quality() {
        let qual = qual_from_phred(&[2, 2, 2, 30, 30, 30]);
        assert_eq!(trim_leading(&qual, 3, 33), 3);
    }

    #[test]
    fn trim_leading_all_low() {
        let qual = qual_from_phred(&[2, 2, 2]);
        assert_eq!(trim_leading(&qual, 3, 33), 3);
    }

    #[test]
    fn trim_leading_all_high() {
        let qual = qual_from_phred(&[30, 30, 30]);
        assert_eq!(trim_leading(&qual, 3, 33), 0);
    }

    #[test]
    fn trim_trailing_removes_low_quality() {
        let qual = qual_from_phred(&[30, 30, 30, 2, 2, 2]);
        assert_eq!(trim_trailing(&qual, 3, 33), 3);
    }

    #[test]
    fn trim_trailing_all_low() {
        let qual = qual_from_phred(&[2, 2, 2]);
        assert_eq!(trim_trailing(&qual, 3, 33), 0);
    }

    #[test]
    fn trim_trailing_all_high() {
        let qual = qual_from_phred(&[30, 30, 30]);
        assert_eq!(trim_trailing(&qual, 3, 33), 3);
    }

    #[test]
    fn sliding_window_high_quality() {
        let qual = qual_from_phred(&[30, 30, 30, 30, 30, 30]);
        assert_eq!(trim_sliding_window(&qual, 4, 20, 33), 6);
    }

    #[test]
    fn sliding_window_drops_at_end() {
        let qual = qual_from_phred(&[30, 30, 30, 30, 5, 5, 5, 5]);
        // Window of 4 starting at position 3: [30, 5, 5, 5] avg = 11.25 < 20
        // Window of 4 starting at position 2: [30, 30, 5, 5] avg = 17.5 < 20
        // Window of 4 starting at position 1: [30, 30, 30, 5] avg = 23.75 >= 20
        // So trim at position where window first fails
        let pos = trim_sliding_window(&qual, 4, 20, 33);
        assert!((2..=5).contains(&pos), "pos={pos}");
    }

    #[test]
    fn sliding_window_all_low() {
        let qual = qual_from_phred(&[5, 5, 5, 5, 5]);
        assert_eq!(trim_sliding_window(&qual, 4, 20, 33), 0);
    }

    #[test]
    fn trim_read_full_pipeline() {
        // Simulate: 2 low-quality leading, 3 low-quality trailing, good middle
        let qual = qual_from_phred(&[2, 2, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 2, 2, 2]);
        let seq: Vec<u8> = vec![b'A'; 15];
        let record = make_record(&seq, &qual);

        let params = QualityParams {
            min_length: 5,
            ..QualityParams::default()
        };

        let result = trim_read(&record, &params);
        assert!(result.is_some());
        let (start, end) = result.unwrap();
        assert_eq!(start, 2); // skip 2 leading
        assert_eq!(end, 12); // skip 3 trailing
    }

    #[test]
    fn trim_read_too_short() {
        let qual = qual_from_phred(&[30, 30, 30]);
        let seq: Vec<u8> = vec![b'A'; 3];
        let record = make_record(&seq, &qual);

        let params = QualityParams {
            min_length: 36, // longer than read
            ..QualityParams::default()
        };

        assert!(trim_read(&record, &params).is_none());
    }

    #[test]
    fn filter_reads_batch() {
        let records = vec![
            // Good read
            make_record(&[b'A'; 50], &qual_from_phred(&[30; 50])),
            // Bad read (all low quality)
            make_record(&[b'A'; 50], &qual_from_phred(&[2; 50])),
            // Short but good
            make_record(&[b'A'; 10], &qual_from_phred(&[30; 10])),
        ];

        let params = QualityParams {
            min_length: 36,
            ..QualityParams::default()
        };

        let (filtered, stats) = filter_reads(&records, &params);
        assert_eq!(stats.input_reads, 3);
        assert_eq!(stats.output_reads, 1); // only the good 50bp read
        assert_eq!(stats.discarded_reads, 2);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].sequence.len(), 50);
    }

    #[test]
    fn adapter_exact_match() {
        let seq = b"ACGTACGTAACTAGTCGA"; // adapter at end
        let adapter = b"AACTAGTCGA";
        let pos = find_adapter_3prime(seq, adapter, 0, 5);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn adapter_with_mismatches() {
        let seq = b"ACGTACGTAACTXGTCGA"; // 1 mismatch in adapter region
        let adapter = b"AACTAGTCGA";
        let pos = find_adapter_3prime(seq, adapter, 1, 5);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn adapter_not_found() {
        let seq = b"ACGTACGTACGTACGT";
        let adapter = b"TTTTTTTTTT";
        let pos = find_adapter_3prime(seq, adapter, 0, 5);
        assert!(pos.is_none());
    }

    #[test]
    fn adapter_partial_overlap() {
        // Adapter partially overlaps read end (first 5 bases of adapter match end)
        let seq = b"ACGTACGTAACTA"; // last 5 match start of adapter
        let adapter = b"AACTAGTCGA";
        let pos = find_adapter_3prime(seq, adapter, 0, 5);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn trim_adapter_3prime_found() {
        let record = make_record(b"ACGTACGTAACTAGTCGA", &[33 + 30; 18]);
        let trimmed = trim_adapter_3prime(&record, b"AACTAGTCGA", 0, 5);
        assert!(trimmed.is_some());
        assert_eq!(trimmed.unwrap().sequence.len(), 8);
    }

    #[test]
    fn trim_adapter_3prime_not_found() {
        let record = make_record(b"ACGTACGT", &[33 + 30; 8]);
        let trimmed = trim_adapter_3prime(&record, b"TTTTTTTTTT", 0, 5);
        assert!(trimmed.is_none());
    }

    #[test]
    fn n_bases_match_anything() {
        assert!(bases_match(b'N', b'A'));
        assert!(bases_match(b'A', b'N'));
        assert!(bases_match(b'N', b'N'));
    }

    #[test]
    fn empty_record_returns_none() {
        let record = make_record(b"", &[]);
        assert!(trim_read(&record, &QualityParams::default()).is_none());
    }

    #[test]
    fn gpu_params_from_quality_params() {
        let params = QualityParams::default();
        let gpu: QualityGpuParams = QualityGpuParams::from(&params);
        assert_eq!(gpu.window_size, 4);
        assert_eq!(gpu.window_min_quality, 20);
        assert_eq!(gpu.phred_offset, 33);
    }

    #[test]
    fn filter_reads_flat_matches_structured() {
        let records = vec![
            make_record(&[b'A'; 50], &qual_from_phred(&[30; 50])),
            make_record(&[b'A'; 50], &qual_from_phred(&[2; 50])),
            make_record(&[b'A'; 10], &qual_from_phred(&[30; 10])),
        ];

        let params = QualityParams {
            min_length: 36,
            ..QualityParams::default()
        };

        // Build flat arrays
        let mut qualities = Vec::new();
        let mut offsets = Vec::new();
        let mut lengths = Vec::new();
        for r in &records {
            offsets.push(qualities.len());
            lengths.push(r.quality.len());
            qualities.extend_from_slice(&r.quality);
        }

        let flat = filter_reads_flat(&qualities, &offsets, &lengths, &params);
        let (structured, stats) = filter_reads(&records, &params);

        // Flat pass count should match structured output count
        let flat_pass_count = flat.pass.iter().filter(|&&p| p == 1).count();
        assert_eq!(flat_pass_count, stats.output_reads);
        assert_eq!(flat_pass_count, structured.len());
    }

    #[test]
    fn filter_reads_flat_empty() {
        let result = filter_reads_flat(&[], &[], &[], &QualityParams::default());
        assert!(result.starts.is_empty());
        assert!(result.ends.is_empty());
        assert!(result.pass.is_empty());
    }

    #[test]
    fn filter_reads_flat_out_of_bounds_safe() {
        let result = filter_reads_flat(&[30; 5], &[10], &[5], &QualityParams::default());
        assert_eq!(result.pass[0], 0);
    }
}
