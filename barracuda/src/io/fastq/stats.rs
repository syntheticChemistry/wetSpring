// SPDX-License-Identifier: AGPL-3.0-or-later
//! FASTQ statistics computation — from records or streaming.

use super::{FastqRecord, FastqStats, open_reader};
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

use super::{header_error, read_line, trimmed_len};

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
        self.num_sequences += 1;
        let len = sequence.len();
        self.total_bases += len as u64;
        self.min_len = self.min_len.min(len);
        self.max_len = self.max_len.max(len);
        *self.length_dist.entry(len).or_insert(0) += 1;

        for &base in sequence {
            if base == b'G' || base == b'C' || base == b'g' || base == b'c' {
                self.gc_count += 1;
            }
        }

        if !quality.is_empty() {
            let mut q_sum: u64 = 0;
            for &q in quality {
                q_sum += u64::from(q.saturating_sub(33));
            }
            self.total_quality_sum += q_sum;
            self.total_quality_count += quality.len() as u64;

            #[allow(clippy::cast_precision_loss)]
            let mean_q = q_sum as f64 / quality.len() as f64;
            if mean_q >= 30.0 {
                self.q30_count += 1;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
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

/// Compute FASTQ statistics in a single streaming pass (zero-copy path).
///
/// No per-record allocation — lines are processed in-place from a
/// reusable `String` buffer.  Use this for large files where only
/// aggregate statistics are needed.
///
/// # Errors
///
/// Returns [`crate::error::Error::Io`] if the file cannot be opened, or
/// [`crate::error::Error::Fastq`] if a record is malformed.
#[must_use = "computed stats are discarded if not used"]
pub fn stats_from_file(path: &Path) -> Result<FastqStats> {
    let mut reader = open_reader(path)?;
    let mut header_buf = String::new();
    let mut seq_buf = String::new();
    let mut separator_buf = String::new();
    let mut qual_buf = String::new();
    let mut acc = StatsAccumulator::new();

    loop {
        header_buf.clear();
        if read_line(reader.as_mut(), &mut header_buf, path)? == 0 {
            break;
        }
        if header_buf.trim_end().is_empty() {
            break;
        }
        if !header_buf.starts_with('@') {
            return Err(header_error(&header_buf));
        }

        seq_buf.clear();
        read_line(reader.as_mut(), &mut seq_buf, path)?;
        let seq_len = trimmed_len(&seq_buf);

        separator_buf.clear();
        read_line(reader.as_mut(), &mut separator_buf, path)?;

        qual_buf.clear();
        read_line(reader.as_mut(), &mut qual_buf, path)?;
        let qual_len = trimmed_len(&qual_buf);

        acc.add_record(
            &seq_buf.as_bytes()[..seq_len],
            &qual_buf.as_bytes()[..qual_len],
        );
    }

    Ok(acc.finish())
}
