// SPDX-License-Identifier: AGPL-3.0-or-later
//! FASTQ statistics computation — from records or streaming.

use super::{FastqRecord, FastqStats, header_error_bytes, open_reader, read_byte_line, trim_end};
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

/// Incremental accumulator for FASTQ statistics.
///
/// Single implementation shared by [`compute_stats`] (from records) and
/// [`stats_from_file`] (streaming). Eliminates duplication and guarantees
/// both paths produce identical results.
struct StatsAccumulator {
    num_sequences: usize,
    total_bases: u64,
    total_quality_sum: u64,
    total_quality_count: u64,
    gc_count: u64,
    min_len: usize,
    max_len: usize,
    q30_count: usize,
    length_dist: HashMap<usize, usize>,
}

impl StatsAccumulator {
    fn new() -> Self {
        Self {
            num_sequences: 0,
            total_bases: 0,
            total_quality_sum: 0,
            total_quality_count: 0,
            gc_count: 0,
            min_len: usize::MAX,
            max_len: 0,
            q30_count: 0,
            length_dist: HashMap::new(),
        }
    }

    #[inline]
    fn add_record(&mut self, sequence: &[u8], quality: &[u8]) {
        let gc = sequence
            .iter()
            .filter(|&&b| b == b'G' || b == b'C' || b == b'g' || b == b'c')
            .count();
        self.add_record_borrowed(sequence.len(), gc, quality);
    }

    /// Zero-copy path: accepts pre-computed sequence length and GC count
    /// so the caller can borrow directly from the I/O buffer without
    /// allocating a `Vec<u8>` copy.
    #[inline]
    fn add_record_borrowed(&mut self, seq_len: usize, gc: usize, quality: &[u8]) {
        self.num_sequences += 1;
        self.total_bases += seq_len as u64;
        self.min_len = self.min_len.min(seq_len);
        self.max_len = self.max_len.max(seq_len);
        *self.length_dist.entry(seq_len).or_insert(0) += 1;
        self.gc_count += gc as u64;

        if !quality.is_empty() {
            let mut q_sum: u64 = 0;
            for &q in quality {
                q_sum += u64::from(q.saturating_sub(33));
            }
            self.total_quality_sum += q_sum;
            self.total_quality_count += quality.len() as u64;

            #[expect(clippy::cast_precision_loss)] // Precision: quality.len() and q_sum bounded by read
            let mean_q = q_sum as f64 / quality.len() as f64;
            if mean_q >= 30.0 {
                self.q30_count += 1;
            }
        }
    }

    #[expect(clippy::cast_precision_loss)] // Precision: num_sequences, total_bases, etc. fit f64
    fn finish(self) -> FastqStats {
        let n = self.num_sequences;
        FastqStats {
            num_sequences: n,
            total_bases: self.total_bases,
            min_length: if n == 0 { 0 } else { self.min_len },
            max_length: self.max_len,
            mean_length: if n > 0 {
                self.total_bases as f64 / n as f64
            } else {
                0.0
            },
            mean_quality: if self.total_quality_count > 0 {
                self.total_quality_sum as f64 / self.total_quality_count as f64
            } else {
                0.0
            },
            gc_content: if self.total_bases > 0 {
                self.gc_count as f64 / self.total_bases as f64
            } else {
                0.0
            },
            q30_count: self.q30_count,
            length_distribution: self.length_dist,
        }
    }
}

/// Compute summary statistics from parsed records.
///
/// # Examples
///
/// ```
/// use wetspring_barracuda::io::fastq::{compute_stats, FastqRecord};
///
/// let records = vec![FastqRecord {
///     id: "read1".into(),
///     sequence: b"ATGC".to_vec(),
///     quality: b"IIII".to_vec(),  // Q40
/// }];
/// let stats = compute_stats(&records);
/// assert_eq!(stats.num_sequences, 1);
/// assert_eq!(stats.total_bases, 4);
/// assert!((stats.gc_content - 0.5).abs() < 1e-6);
/// ```
#[must_use]
pub fn compute_stats(records: &[FastqRecord]) -> FastqStats {
    let mut acc = StatsAccumulator::new();
    for rec in records {
        acc.add_record(&rec.sequence, &rec.quality);
    }
    acc.finish()
}

/// Compute FASTQ statistics in a single streaming pass (byte-native path).
///
/// No per-record allocation — lines are processed in-place from a
/// reusable byte buffer. No UTF-8 requirement for sequence or quality
/// data. Use this for large files where only aggregate statistics are needed.
///
/// # Errors
///
/// Returns [`crate::error::Error::Io`] if the file cannot be opened, or
/// [`crate::error::Error::Fastq`] if a record is malformed.
#[must_use = "computed stats are discarded if not used"]
pub fn stats_from_file(path: &Path) -> Result<FastqStats> {
    let mut reader = open_reader(path)?;
    let mut buf = Vec::new();
    let mut acc = StatsAccumulator::new();

    loop {
        buf.clear();
        if read_byte_line(reader.as_mut(), &mut buf, path)? == 0 {
            break;
        }
        if trim_end(&buf).is_empty() {
            break;
        }
        if buf.first() != Some(&b'@') {
            return Err(header_error_bytes(&buf));
        }

        // Sequence line — borrow directly from buf, no .to_vec()
        buf.clear();
        read_byte_line(reader.as_mut(), &mut buf, path)?;
        let seq_len = trim_end(&buf).len();
        let seq_gc = trim_end(&buf)
            .iter()
            .filter(|&&b| b == b'G' || b == b'C' || b == b'g' || b == b'c')
            .count();

        // Separator line
        buf.clear();
        read_byte_line(reader.as_mut(), &mut buf, path)?;

        // Quality line
        buf.clear();
        read_byte_line(reader.as_mut(), &mut buf, path)?;
        let qual = trim_end(&buf);

        acc.add_record_borrowed(seq_len, seq_gc, qual);
    }

    Ok(acc.finish())
}
