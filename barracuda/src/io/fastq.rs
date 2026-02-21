// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign FASTQ parser — zero external parsing dependencies.
//!
//! Streams records from disk via [`BufReader`]. Handles both plain and
//! gzip-compressed files (`.gz` extension, via `flate2::read::GzDecoder`).
//!
//! # Format (standard 4-line FASTQ)
//!
//! ```text
//! @identifier description
//! SEQUENCE
//! +
//! QUALITY (Phred33 ASCII)
//! ```
//!
//! [`parse_fastq`] collects all records for multi-pass analysis (k-mer
//! counting, diversity). For single-pass statistics on large files,
//! prefer [`stats_from_file`] which processes lines in-place and never
//! allocates per-record storage.

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Summary statistics from parsing a FASTQ file.
#[derive(Debug, Clone)]
pub struct FastqStats {
    /// Total number of sequences.
    pub num_sequences: usize,
    /// Total number of bases.
    pub total_bases: u64,
    /// Minimum sequence length.
    pub min_length: usize,
    /// Maximum sequence length.
    pub max_length: usize,
    /// Mean sequence length.
    pub mean_length: f64,
    /// Mean Phred quality score (Phred33).
    pub mean_quality: f64,
    /// GC content as fraction \[0, 1\].
    pub gc_content: f64,
    /// Number of sequences with mean Q >= 30.
    pub q30_count: usize,
    /// Length distribution: length -> count.
    pub length_distribution: HashMap<usize, usize>,
}

/// A parsed FASTQ record with owned data.
#[derive(Debug, Clone)]
pub struct FastqRecord {
    /// Record identifier.
    pub id: String,
    /// Nucleotide sequence.
    pub sequence: Vec<u8>,
    /// Phred33 quality scores (raw ASCII bytes).
    pub quality: Vec<u8>,
}

// ── Internal helpers ─────────────────────────────────────────────

/// Open a FASTQ file for buffered reading.
///
/// Detects gzip compression from the `.gz` file extension and
/// wraps the stream with [`flate2::read::GzDecoder`] when needed.
fn open_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let ext = path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("");
    if ext.eq_ignore_ascii_case("gz") {
        let decoder = flate2::read::GzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Read one line into `buf`, returning bytes read.  Wraps I/O errors
/// with path context.
fn read_line(reader: &mut dyn BufRead, buf: &mut String, path: &Path) -> Result<usize> {
    reader.read_line(buf).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Return an error for a malformed header line.
fn header_error(buf: &str) -> Error {
    let snippet = buf.trim_end();
    let end = snippet.len().min(40);
    Error::Fastq(format!("expected '@' header, got: {}", &snippet[..end]))
}

/// Count the trimmed length of `buf` (excluding trailing `\n` / `\r`).
fn trimmed_len(buf: &str) -> usize {
    buf.trim_end().len()
}

// ── Public API ───────────────────────────────────────────────────

/// Parse a FASTQ file and collect all records.
///
/// Handles plain `.fastq` and gzip-compressed `.fastq.gz` files.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, or
/// [`Error::Fastq`] if a record is malformed.
pub fn parse_fastq(path: &Path) -> Result<Vec<FastqRecord>> {
    let mut reader = open_reader(path)?;
    let mut records = Vec::new();
    let mut buf = String::new();

    loop {
        // Line 1: @identifier
        buf.clear();
        if read_line(reader.as_mut(), &mut buf, path)? == 0 {
            break;
        }
        if buf.trim_end().is_empty() {
            break;
        }
        if !buf.starts_with('@') {
            return Err(header_error(&buf));
        }
        let id = buf.trim_end()[1..]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();

        // Line 2: sequence
        buf.clear();
        read_line(reader.as_mut(), &mut buf, path)?;
        let sequence = buf.trim_end().as_bytes().to_vec();

        // Line 3: + separator (consumed, not validated beyond presence)
        buf.clear();
        read_line(reader.as_mut(), &mut buf, path)?;

        // Line 4: quality scores
        buf.clear();
        read_line(reader.as_mut(), &mut buf, path)?;
        let quality = buf.trim_end().as_bytes().to_vec();

        records.push(FastqRecord {
            id,
            sequence,
            quality,
        });
    }

    Ok(records)
}

/// Streaming FASTQ iterator — yields one record at a time without buffering.
///
/// For large files where you don't need all records in memory simultaneously.
/// Handles both plain `.fastq` and gzip-compressed `.fastq.gz` files.
///
/// # Errors
///
/// The iterator yields `Result<FastqRecord>` — each item may fail with
/// [`Error::Io`] or [`Error::Fastq`].
///
/// # Example
///
/// ```no_run
/// use wetspring_barracuda::io::fastq;
/// use std::path::Path;
///
/// let iter = fastq::FastqIter::open(Path::new("reads.fastq")).unwrap();
/// for result in iter {
///     let record = result.unwrap();
///     // process record without buffering all reads
/// }
/// ```
pub struct FastqIter {
    reader: Box<dyn BufRead>,
    path: std::path::PathBuf,
    buf: String,
    done: bool,
}

impl FastqIter {
    /// Open a FASTQ file for streaming iteration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        let reader = open_reader(path)?;
        Ok(Self {
            reader,
            path: path.to_path_buf(),
            buf: String::new(),
            done: false,
        })
    }
}

impl Iterator for FastqIter {
    type Item = Result<FastqRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Line 1: @identifier
        self.buf.clear();
        match read_line(self.reader.as_mut(), &mut self.buf, &self.path) {
            Ok(0) => {
                self.done = true;
                return None;
            }
            Ok(_) => {}
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        }
        if self.buf.trim_end().is_empty() {
            self.done = true;
            return None;
        }
        if !self.buf.starts_with('@') {
            self.done = true;
            return Some(Err(header_error(&self.buf)));
        }
        let id = self.buf.trim_end()[1..]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();

        // Line 2: sequence
        self.buf.clear();
        if let Err(e) = read_line(self.reader.as_mut(), &mut self.buf, &self.path) {
            self.done = true;
            return Some(Err(e));
        }
        let sequence = self.buf.trim_end().as_bytes().to_vec();

        // Line 3: + separator
        self.buf.clear();
        if let Err(e) = read_line(self.reader.as_mut(), &mut self.buf, &self.path) {
            self.done = true;
            return Some(Err(e));
        }

        // Line 4: quality scores
        self.buf.clear();
        if let Err(e) = read_line(self.reader.as_mut(), &mut self.buf, &self.path) {
            self.done = true;
            return Some(Err(e));
        }
        let quality = self.buf.trim_end().as_bytes().to_vec();

        Some(Ok(FastqRecord {
            id,
            sequence,
            quality,
        }))
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
    if records.is_empty() {
        return FastqStats {
            num_sequences: 0,
            total_bases: 0,
            min_length: 0,
            max_length: 0,
            mean_length: 0.0,
            mean_quality: 0.0,
            gc_content: 0.0,
            q30_count: 0,
            length_distribution: HashMap::new(),
        };
    }

    let mut total_bases: u64 = 0;
    let mut total_quality_sum: u64 = 0;
    let mut total_quality_count: u64 = 0;
    let mut gc_count: u64 = 0;
    let mut min_len = usize::MAX;
    let mut max_len = 0_usize;
    let mut q30_count = 0_usize;
    let mut length_dist: HashMap<usize, usize> = HashMap::new();

    for rec in records {
        let len = rec.sequence.len();
        total_bases += len as u64;
        min_len = min_len.min(len);
        max_len = max_len.max(len);
        *length_dist.entry(len).or_insert(0) += 1;

        for &base in &rec.sequence {
            if base == b'G' || base == b'C' || base == b'g' || base == b'c' {
                gc_count += 1;
            }
        }

        if !rec.quality.is_empty() {
            let mut q_sum: u64 = 0;
            for &q in &rec.quality {
                q_sum += u64::from(q.saturating_sub(33));
            }
            total_quality_sum += q_sum;
            total_quality_count += rec.quality.len() as u64;

            #[allow(clippy::cast_precision_loss)] // quality sums fit in f64 mantissa
            let mean_q = q_sum as f64 / rec.quality.len() as f64;
            if mean_q >= 30.0 {
                q30_count += 1;
            }
        }
    }

    let n = records.len();
    #[allow(clippy::cast_precision_loss)] // all values are small enough for exact f64
    FastqStats {
        num_sequences: n,
        total_bases,
        min_length: min_len,
        max_length: max_len,
        mean_length: total_bases as f64 / n as f64,
        mean_quality: if total_quality_count > 0 {
            total_quality_sum as f64 / total_quality_count as f64
        } else {
            0.0
        },
        gc_content: if total_bases > 0 {
            gc_count as f64 / total_bases as f64
        } else {
            0.0
        },
        q30_count,
        length_distribution: length_dist,
    }
}

/// Compute FASTQ statistics in a single streaming pass (zero-copy path).
///
/// No per-record allocation — lines are processed in-place from a
/// reusable `String` buffer.  Use this for large files where only
/// aggregate statistics are needed.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, or
/// [`Error::Fastq`] if a record is malformed.
pub fn stats_from_file(path: &Path) -> Result<FastqStats> {
    let mut reader = open_reader(path)?;
    let mut buf = String::new();

    let mut num_sequences = 0_usize;
    let mut total_bases: u64 = 0;
    let mut total_quality_sum: u64 = 0;
    let mut total_quality_count: u64 = 0;
    let mut gc_count: u64 = 0;
    let mut min_len = usize::MAX;
    let mut max_len = 0_usize;
    let mut q30_count = 0_usize;
    let mut length_dist: HashMap<usize, usize> = HashMap::new();

    loop {
        // Line 1: @identifier (or EOF)
        buf.clear();
        if read_line(reader.as_mut(), &mut buf, path)? == 0 {
            break;
        }
        if buf.trim_end().is_empty() {
            break;
        }
        if !buf.starts_with('@') {
            return Err(header_error(&buf));
        }
        num_sequences += 1;

        // Line 2: sequence — process GC and length in-place
        buf.clear();
        read_line(reader.as_mut(), &mut buf, path)?;
        let seq_len = trimmed_len(&buf);
        total_bases += seq_len as u64;
        min_len = min_len.min(seq_len);
        max_len = max_len.max(seq_len);
        *length_dist.entry(seq_len).or_insert(0) += 1;
        for &base in &buf.as_bytes()[..seq_len] {
            if base == b'G' || base == b'C' || base == b'g' || base == b'c' {
                gc_count += 1;
            }
        }

        // Line 3: + separator (skip)
        buf.clear();
        read_line(reader.as_mut(), &mut buf, path)?;

        // Line 4: quality — process Phred33 scores in-place
        buf.clear();
        read_line(reader.as_mut(), &mut buf, path)?;
        let qual_len = trimmed_len(&buf);
        if qual_len > 0 {
            let mut q_sum: u64 = 0;
            for &q in &buf.as_bytes()[..qual_len] {
                q_sum += u64::from(q.saturating_sub(33));
            }
            total_quality_sum += q_sum;
            total_quality_count += qual_len as u64;

            #[allow(clippy::cast_precision_loss)]
            let mean_q = q_sum as f64 / qual_len as f64;
            if mean_q >= 30.0 {
                q30_count += 1;
            }
        }
    }

    if num_sequences == 0 {
        min_len = 0;
    }

    #[allow(clippy::cast_precision_loss)]
    Ok(FastqStats {
        num_sequences,
        total_bases,
        min_length: min_len,
        max_length: max_len,
        mean_length: if num_sequences > 0 {
            total_bases as f64 / num_sequences as f64
        } else {
            0.0
        },
        mean_quality: if total_quality_count > 0 {
            total_quality_sum as f64 / total_quality_count as f64
        } else {
            0.0
        },
        gc_content: if total_bases > 0 {
            gc_count as f64 / total_bases as f64
        } else {
            0.0
        },
        q30_count,
        length_distribution: length_dist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal FASTQ file and return its path.
    fn write_fastq(dir: &tempfile::TempDir, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_parse_fastq_two_records() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@seq1 description\nACGTACGT\n+\nIIIIIIII\n@seq2\nGGCC\n+\n!!!!\n";
        let path = write_fastq(&dir, "test.fastq", content);
        let records = parse_fastq(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "seq1");
        assert_eq!(records[0].sequence, b"ACGTACGT");
        assert_eq!(records[0].quality, b"IIIIIIII");
        assert_eq!(records[1].id, "seq2");
        assert_eq!(records[1].sequence, b"GGCC");
    }

    #[test]
    fn test_parse_fastq_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fastq(&dir, "empty.fastq", "");
        let records = parse_fastq(&path).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn test_parse_fastq_trailing_blank_line() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@seq1\nACGT\n+\nIIII\n\n";
        let path = write_fastq(&dir, "trailing.fastq", content);
        let records = parse_fastq(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "seq1");
    }

    #[test]
    fn test_parse_fastq_bad_header() {
        let dir = tempfile::tempdir().unwrap();
        let content = "NOT_A_HEADER\nACGT\n+\nIIII\n";
        let path = write_fastq(&dir, "bad.fastq", content);
        let result = parse_fastq(&path);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("expected '@' header"));
    }

    #[test]
    fn test_parse_fastq_nonexistent_file() {
        let path = std::env::temp_dir().join("nonexistent_wetspring_9f8a2.fastq");
        let result = parse_fastq(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_from_file_two_records() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nGGCC\n+\n!!!!\n";
        let path = write_fastq(&dir, "stats.fastq", content);
        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_sequences, 2);
        assert_eq!(stats.total_bases, 12);
        assert_eq!(stats.min_length, 4);
        assert_eq!(stats.max_length, 8);
        assert!(stats.gc_content > 0.0);
        assert!(stats.mean_quality > 0.0);
    }

    #[test]
    fn test_stats_from_file_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fastq(&dir, "empty.fastq", "");
        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_sequences, 0);
        assert_eq!(stats.min_length, 0);
    }

    #[test]
    fn test_stats_from_file_bad_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fastq(&dir, "bad.fastq", "BAD\nACGT\n+\nIIII\n");
        let result = stats_from_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_from_file_nonexistent() {
        let path = std::env::temp_dir().join("nonexistent_wetspring_9f8a2.fastq");
        let result = stats_from_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats_from_file_single_record_quality() {
        let dir = tempfile::tempdir().unwrap();
        // Q=40 for 'I' (73-33=40)
        let content = "@s1\nACGT\n+\nIIII\n";
        let path = write_fastq(&dir, "q40.fastq", content);
        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_sequences, 1);
        assert!((stats.mean_quality - 40.0).abs() < 1e-10);
        assert!((stats.gc_content - 0.5).abs() < 1e-10);
        assert_eq!(stats.q30_count, 1);
    }

    #[test]
    fn test_stats_from_file_low_quality() {
        let dir = tempfile::tempdir().unwrap();
        // Q=0 for '!' (33-33=0)
        let content = "@s1\nAAAA\n+\n!!!!\n";
        let path = write_fastq(&dir, "q0.fastq", content);
        let stats = stats_from_file(&path).unwrap();
        assert!((stats.mean_quality - 0.0).abs() < 1e-10);
        assert_eq!(stats.q30_count, 0);
    }

    #[test]
    fn test_stats_gc_lowercase() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@s1\ngcgc\n+\nIIII\n";
        let path = write_fastq(&dir, "lc.fastq", content);
        let stats = stats_from_file(&path).unwrap();
        assert!((stats.gc_content - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_and_stats_agree() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@a\nACGTAC\n+\nIIIIII\n@b\nGGCC\n+\n!I!I\n";
        let path = write_fastq(&dir, "agree.fastq", content);
        let records = parse_fastq(&path).unwrap();
        let stats_from_records = compute_stats(&records);
        let stats_from_stream = stats_from_file(&path).unwrap();
        assert_eq!(
            stats_from_records.num_sequences,
            stats_from_stream.num_sequences
        );
        assert_eq!(
            stats_from_records.total_bases,
            stats_from_stream.total_bases
        );
        assert_eq!(stats_from_records.min_length, stats_from_stream.min_length);
        assert_eq!(stats_from_records.max_length, stats_from_stream.max_length);
        assert!((stats_from_records.gc_content - stats_from_stream.gc_content).abs() < 1e-12);
        assert!((stats_from_records.mean_quality - stats_from_stream.mean_quality).abs() < 1e-12);
    }

    #[test]
    fn test_empty_stats() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.num_sequences, 0);
        assert_eq!(stats.total_bases, 0);
        assert_eq!(stats.min_length, 0);
    }

    #[test]
    fn test_single_record() {
        let rec = FastqRecord {
            id: "test".to_string(),
            sequence: b"ACGTACGT".to_vec(),
            quality: b"IIIIIIII".to_vec(), // Q=40
        };
        let stats = compute_stats(&[rec]);
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.total_bases, 8);
        assert!((stats.gc_content - 0.5).abs() < 1e-10);
        assert!((stats.mean_quality - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_all_gc() {
        let rec = FastqRecord {
            id: "gc".to_string(),
            sequence: b"GGGGCCCC".to_vec(),
            quality: b"IIIIIIII".to_vec(),
        };
        let stats = compute_stats(&[rec]);
        assert!((stats.gc_content - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gc_content_no_gc() {
        let rec = FastqRecord {
            id: "at".to_string(),
            sequence: b"AAAATTTT".to_vec(),
            quality: b"IIIIIIII".to_vec(),
        };
        let stats = compute_stats(&[rec]);
        assert!(stats.gc_content.abs() < 1e-10);
    }

    #[test]
    fn test_q30_threshold() {
        let high_q = FastqRecord {
            id: "high".to_string(),
            sequence: b"ACGT".to_vec(),
            quality: b"IIII".to_vec(), // Q=40
        };
        let low_q = FastqRecord {
            id: "low".to_string(),
            sequence: b"ACGT".to_vec(),
            quality: b"!!!!".to_vec(), // Q=0
        };
        let stats = compute_stats(&[high_q, low_q]);
        assert_eq!(stats.q30_count, 1);
    }

    /// Write a gzip-compressed FASTQ file and return its path.
    fn write_fastq_gz(dir: &tempfile::TempDir, name: &str, content: &str) -> std::path::PathBuf {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let path = dir.path().join(name);
        let file = File::create(&path).unwrap();
        let mut gz = GzEncoder::new(file, Compression::default());
        gz.write_all(content.as_bytes()).unwrap();
        gz.finish().unwrap();
        path
    }

    #[test]
    fn test_parse_fastq_gzip() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@gz1\nACGT\n+\nIIII\n@gz2\nTTTT\n+\n!!!!\n";
        let path = write_fastq_gz(&dir, "reads.fastq.gz", content);
        let records = parse_fastq(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "gz1");
        assert_eq!(records[0].sequence, b"ACGT");
        assert_eq!(records[1].id, "gz2");
    }

    #[test]
    fn test_stats_from_file_gzip() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@gz1\nACGT\n+\nIIII\n";
        let path = write_fastq_gz(&dir, "stats.fastq.gz", content);
        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.total_bases, 4);
        assert!((stats.gc_content - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_from_file_trailing_blank_line() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@r1\nACGT\n+\nIIII\n\n";
        let path = write_fastq(&dir, "trailing.fastq", content);
        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_sequences, 1);
    }

    #[test]
    fn test_stats_from_file_empty_quality_line() {
        let dir = tempfile::tempdir().unwrap();
        // quality line is present but empty — exercises qual_len == 0 path
        let content = "@r1\nACGT\n+\n\n";
        let path = write_fastq(&dir, "noq.fastq", content);
        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_sequences, 1);
        assert!((stats.mean_quality - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_zero_gc_and_quality() {
        // All A/T, empty quality — exercises both gc_content==0 and mean_quality==0
        let rec = FastqRecord {
            id: "at".to_string(),
            sequence: b"AATTAATT".to_vec(),
            quality: vec![],
        };
        let stats = compute_stats(&[rec]);
        assert!(stats.gc_content.abs() < 1e-10);
        assert!(stats.mean_quality.abs() < 1e-10);
        assert_eq!(stats.q30_count, 0);
    }

    #[test]
    fn test_compute_stats_empty_sequence_record() {
        // Non-empty records with zero-length sequence — exercises total_bases==0 fallback
        let rec = FastqRecord {
            id: "empty_seq".to_string(),
            sequence: vec![],
            quality: vec![],
        };
        let stats = compute_stats(&[rec]);
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.total_bases, 0);
        assert!(stats.gc_content.abs() < 1e-10);
        assert!(stats.mean_quality.abs() < 1e-10);
        assert_eq!(stats.min_length, 0);
        assert_eq!(stats.max_length, 0);
    }

    #[test]
    fn test_length_distribution() {
        let r1 = FastqRecord {
            id: "a".into(),
            sequence: b"ACGT".to_vec(),
            quality: vec![],
        };
        let r2 = FastqRecord {
            id: "b".into(),
            sequence: b"ACGTAC".to_vec(),
            quality: vec![],
        };
        let r3 = FastqRecord {
            id: "c".into(),
            sequence: b"ACGT".to_vec(),
            quality: vec![],
        };
        let stats = compute_stats(&[r1, r2, r3]);
        assert_eq!(stats.length_distribution[&4], 2);
        assert_eq!(stats.length_distribution[&6], 1);
        assert_eq!(stats.min_length, 4);
        assert_eq!(stats.max_length, 6);
    }

    #[test]
    fn fastq_iter_matches_parse() {
        let dir = tempfile::tempdir().unwrap();
        let content = "@r1 desc\nACGTACGT\n+\nIIIIIIII\n@r2\nGGCC\n+\n!!!!\n";
        let path = write_fastq(&dir, "iter.fastq", content);

        let buffered = parse_fastq(&path).unwrap();
        let streamed: Vec<FastqRecord> = FastqIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(buffered.len(), streamed.len());
        for (b, s) in buffered.iter().zip(streamed.iter()) {
            assert_eq!(b.id, s.id);
            assert_eq!(b.sequence, s.sequence);
            assert_eq!(b.quality, s.quality);
        }
    }

    #[test]
    fn fastq_iter_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fastq(&dir, "empty.fastq", "");
        let records: Vec<FastqRecord> = FastqIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn fastq_iter_gzip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iter.fastq.gz");
        let file = File::create(&path).unwrap();
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        gz.write_all(b"@g1\nACGT\n+\nIIII\n").unwrap();
        gz.finish().unwrap();

        let records: Vec<FastqRecord> = FastqIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "g1");
        assert_eq!(records[0].sequence, b"ACGT");
    }
}
