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
//! All I/O is byte-native — no UTF-8 assumption for sequence or quality
//! data. Header identifiers are extracted as lossy UTF-8.
//!
//! [`parse_fastq`] collects records via [`FastqIter`] for multi-pass
//! analysis (k-mer counting, diversity). For single-pass statistics on
//! large files, prefer [`stats_from_file`] which processes lines in-place
//! and never allocates per-record storage.

mod stats;
#[cfg(test)]
mod tests;

pub use stats::{compute_stats, stats_from_file};

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

/// Borrowed FASTQ record — references data in the parser's internal buffer.
///
/// Avoids allocation for high-throughput single-pass processing. The references
/// are only valid during the callback invocation in [`for_each_record`].
#[derive(Debug, Clone, Copy)]
pub struct FastqRefRecord<'a> {
    /// Record identifier (first word after `@`).
    pub id: &'a str,
    /// Nucleotide sequence.
    pub sequence: &'a [u8],
    /// Phred33 quality scores (raw ASCII bytes).
    pub quality: &'a [u8],
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
pub(crate) fn open_reader(path: &Path) -> Result<Box<dyn BufRead>> {
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

/// Read one byte-line into `buf`, returning bytes read. Wraps I/O errors
/// with path context. Uses `read_until(b'\n')` — no UTF-8 requirement.
pub(crate) fn read_byte_line(
    reader: &mut dyn BufRead,
    buf: &mut Vec<u8>,
    path: &Path,
) -> Result<usize> {
    reader.read_until(b'\n', buf).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Trim trailing `\n` and `\r` from a byte slice.
pub(crate) fn trim_end(buf: &[u8]) -> &[u8] {
    let mut end = buf.len();
    while end > 0 && matches!(buf[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    &buf[..end]
}

/// Return an error for a malformed header line (bytes input, lossy display).
pub(crate) fn header_error_bytes(buf: &[u8]) -> Error {
    let trimmed = trim_end(buf);
    let end = trimmed.len().min(40);
    let s = String::from_utf8_lossy(&trimmed[..end]);
    Error::Fastq(format!("expected '@' header, got: {s}"))
}

/// Extract the record identifier from a header line (bytes after `@`,
/// up to the first ASCII whitespace). Returns lossy UTF-8.
fn extract_id(header: &[u8]) -> &str {
    let trimmed = trim_end(header);
    let after_at = if trimmed.first() == Some(&b'@') {
        &trimmed[1..]
    } else {
        trimmed
    };
    let id_end = after_at
        .iter()
        .position(|&b| b == b' ' || b == b'\t')
        .unwrap_or(after_at.len());
    // FASTQ identifiers are ASCII by spec; this is infallible for well-formed input
    std::str::from_utf8(&after_at[..id_end])
        .unwrap_or_else(|e| std::str::from_utf8(&after_at[..e.valid_up_to()]).unwrap_or(""))
}

// ── Public API ───────────────────────────────────────────────────

/// Collect all records from a FASTQ file into memory via [`FastqIter`].
///
/// Convenience wrapper — streams records from disk, then collects.
/// For single-pass processing prefer [`FastqIter`] or [`for_each_record`].
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, or
/// [`Error::Fastq`] if a record is malformed.
#[must_use = "parsed records are discarded if not used"]
pub fn parse_fastq(path: &Path) -> Result<Vec<FastqRecord>> {
    FastqIter::open(path)?.collect()
}

/// Process each record without per-record allocation.
///
/// The callback receives borrowed slices into the parser's internal buffers.
/// For single-pass processing (statistics, filtering), this avoids the
/// per-record `Vec` allocation of [`FastqIter`].
///
/// All I/O is byte-native — no UTF-8 requirement for sequence or quality
/// lines. Header identifiers are extracted as lossy UTF-8.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, [`Error::Fastq`] if a
/// record is malformed, or propagates the callback's [`Result`].
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use wetspring_barracuda::io::fastq::{for_each_record, FastqRefRecord};
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let path = Path::new("reads.fastq");
/// let mut count = 0_usize;
/// for_each_record(path, |record: FastqRefRecord<'_>| {
///     count += 1;
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
pub fn for_each_record<F>(path: &Path, mut f: F) -> Result<()>
where
    F: FnMut(FastqRefRecord<'_>) -> Result<()>,
{
    let mut reader = open_reader(path)?;
    let mut buf_header = Vec::new();
    let mut buf_sequence = Vec::new();
    let mut buf_sep = Vec::new();
    let mut buf_quality = Vec::new();

    loop {
        buf_header.clear();
        if read_byte_line(reader.as_mut(), &mut buf_header, path)? == 0 {
            break;
        }
        if trim_end(&buf_header).is_empty() {
            break;
        }
        if buf_header.first() != Some(&b'@') {
            return Err(header_error_bytes(&buf_header));
        }
        let id = extract_id(&buf_header);

        buf_sequence.clear();
        read_byte_line(reader.as_mut(), &mut buf_sequence, path)?;

        buf_sep.clear();
        read_byte_line(reader.as_mut(), &mut buf_sep, path)?;

        buf_quality.clear();
        read_byte_line(reader.as_mut(), &mut buf_quality, path)?;

        let record = FastqRefRecord {
            id,
            sequence: trim_end(&buf_sequence),
            quality: trim_end(&buf_quality),
        };
        f(record)?;
    }
    Ok(())
}

/// Streaming FASTQ iterator — yields one record at a time without buffering.
///
/// All I/O is byte-native — no UTF-8 requirement for sequence or quality
/// lines. Handles both plain `.fastq` and gzip-compressed `.fastq.gz` files.
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
    buf: Vec<u8>,
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
            buf: Vec::new(),
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
        match read_byte_line(self.reader.as_mut(), &mut self.buf, &self.path) {
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
        if trim_end(&self.buf).is_empty() {
            self.done = true;
            return None;
        }
        if self.buf.first() != Some(&b'@') {
            self.done = true;
            return Some(Err(header_error_bytes(&self.buf)));
        }
        let id = extract_id(&self.buf).to_string();

        // Line 2: sequence
        self.buf.clear();
        if let Err(e) = read_byte_line(self.reader.as_mut(), &mut self.buf, &self.path) {
            self.done = true;
            return Some(Err(e));
        }
        let sequence = trim_end(&self.buf).to_vec();

        // Line 3: + separator
        self.buf.clear();
        if let Err(e) = read_byte_line(self.reader.as_mut(), &mut self.buf, &self.path) {
            self.done = true;
            return Some(Err(e));
        }

        // Line 4: quality scores
        self.buf.clear();
        if let Err(e) = read_byte_line(self.reader.as_mut(), &mut self.buf, &self.path) {
            self.done = true;
            return Some(Err(e));
        }
        let quality = trim_end(&self.buf).to_vec();

        Some(Ok(FastqRecord {
            id,
            sequence,
            quality,
        }))
    }
}
