// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign SAM (Sequence Alignment/Map) format reader and writer.
//!
//! Implements the SAM text format for alignment interchange. BAM (binary)
//! is deferred — text SAM is sufficient for validation against breseq output.
//!
//! # SAM Format (tab-delimited)
//!
//! ```text
//! QNAME FLAG RNAME POS MAPQ CIGAR RNEXT PNEXT TLEN SEQ QUAL
//! ```
//!
//! # Reference
//!
//! SAM/BAM spec v1: <https://samtools.github.io/hts-specs/SAMv1.pdf>

#[cfg(test)]
mod tests;

use crate::error::{Error, Result};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

// ── SAM flags (bitfield) ─────────────────────────────────────────

/// Read is paired in sequencing.
pub const FLAG_PAIRED: u16 = 0x1;
/// Read mapped in proper pair.
pub const FLAG_PROPER_PAIR: u16 = 0x2;
/// Read unmapped.
pub const FLAG_UNMAPPED: u16 = 0x4;
/// Mate unmapped.
pub const FLAG_MATE_UNMAPPED: u16 = 0x8;
/// Read reverse strand.
pub const FLAG_REVERSE: u16 = 0x10;
/// Mate reverse strand.
pub const FLAG_MATE_REVERSE: u16 = 0x20;
/// First in pair.
pub const FLAG_READ1: u16 = 0x40;
/// Second in pair.
pub const FLAG_READ2: u16 = 0x80;
/// Not primary alignment.
pub const FLAG_SECONDARY: u16 = 0x100;
/// Fails platform/vendor QC.
pub const FLAG_FAILED_QC: u16 = 0x200;
/// PCR or optical duplicate.
pub const FLAG_DUPLICATE: u16 = 0x400;
/// Supplementary alignment.
pub const FLAG_SUPPLEMENTARY: u16 = 0x800;

// ── CIGAR ────────────────────────────────────────────────────────

/// A single CIGAR operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CigarOp {
    /// Length of this operation.
    pub len: u32,
    /// Operation type.
    pub op: CigarType,
}

/// CIGAR operation types per SAM spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarType {
    /// Alignment match (can be sequence match or mismatch).
    Match,
    /// Insertion to reference.
    Insertion,
    /// Deletion from reference.
    Deletion,
    /// Skipped region from reference.
    Skip,
    /// Soft clipping (bases present in SEQ).
    SoftClip,
    /// Hard clipping (bases not present in SEQ).
    HardClip,
    /// Padding (silent deletion from padded reference).
    Padding,
    /// Sequence match.
    SeqMatch,
    /// Sequence mismatch.
    SeqMismatch,
}

impl CigarType {
    /// Whether this operation consumes query bases.
    #[must_use]
    pub const fn consumes_query(self) -> bool {
        matches!(
            self,
            Self::Match
                | Self::Insertion
                | Self::SoftClip
                | Self::SeqMatch
                | Self::SeqMismatch
        )
    }

    /// Whether this operation consumes reference bases.
    #[must_use]
    pub const fn consumes_reference(self) -> bool {
        matches!(
            self,
            Self::Match | Self::Deletion | Self::Skip | Self::SeqMatch | Self::SeqMismatch
        )
    }
}

impl fmt::Display for CigarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Match => "M",
            Self::Insertion => "I",
            Self::Deletion => "D",
            Self::Skip => "N",
            Self::SoftClip => "S",
            Self::HardClip => "H",
            Self::Padding => "P",
            Self::SeqMatch => "=",
            Self::SeqMismatch => "X",
        })
    }
}

fn char_to_cigar_type(c: u8) -> Result<CigarType> {
    match c {
        b'M' => Ok(CigarType::Match),
        b'I' => Ok(CigarType::Insertion),
        b'D' => Ok(CigarType::Deletion),
        b'N' => Ok(CigarType::Skip),
        b'S' => Ok(CigarType::SoftClip),
        b'H' => Ok(CigarType::HardClip),
        b'P' => Ok(CigarType::Padding),
        b'=' => Ok(CigarType::SeqMatch),
        b'X' => Ok(CigarType::SeqMismatch),
        _ => Err(Error::Sam(format!("unknown CIGAR op: {}", c as char))),
    }
}

/// Parse a CIGAR string into operations.
///
/// # Errors
///
/// Returns [`Error::Sam`] if the CIGAR string contains invalid characters.
pub fn parse_cigar(cigar: &str) -> Result<Vec<CigarOp>> {
    if cigar == "*" {
        return Ok(Vec::new());
    }
    let mut ops = Vec::new();
    let mut num_start = 0;
    for (i, b) in cigar.bytes().enumerate() {
        if b.is_ascii_alphabetic() || b == b'=' {
            let len_str = &cigar[num_start..i];
            let len = len_str
                .parse::<u32>()
                .map_err(|_| Error::Sam(format!("invalid CIGAR length: {len_str}")))?;
            ops.push(CigarOp {
                len,
                op: char_to_cigar_type(b)?,
            });
            num_start = i + 1;
        }
    }
    Ok(ops)
}

/// Format CIGAR operations back to a string.
#[must_use]
pub fn format_cigar(ops: &[CigarOp]) -> String {
    if ops.is_empty() {
        return "*".to_string();
    }
    let mut s = String::with_capacity(ops.len() * 4);
    for op in ops {
        s.push_str(&op.len.to_string());
        s.push_str(&op.op.to_string());
    }
    s
}

/// Total reference bases consumed by a CIGAR string.
#[must_use]
pub fn reference_length(ops: &[CigarOp]) -> u64 {
    ops.iter()
        .filter(|op| op.op.consumes_reference())
        .map(|op| u64::from(op.len))
        .sum()
}

/// Total query bases consumed by a CIGAR string.
#[must_use]
pub fn query_length(ops: &[CigarOp]) -> u64 {
    ops.iter()
        .filter(|op| op.op.consumes_query())
        .map(|op| u64::from(op.len))
        .sum()
}

// ── SAM Record ───────────────────────────────────────────────────

/// A parsed SAM alignment record.
#[derive(Debug, Clone)]
pub struct SamRecord {
    /// Query template name.
    pub qname: String,
    /// Bitwise FLAG.
    pub flag: u16,
    /// Reference sequence name.
    pub rname: String,
    /// 1-based leftmost mapping position (0 if unmapped).
    pub pos: u64,
    /// Mapping quality (255 = unavailable).
    pub mapq: u8,
    /// CIGAR operations.
    pub cigar: Vec<CigarOp>,
    /// Reference name of mate/next read.
    pub rnext: String,
    /// 1-based position of mate/next read.
    pub pnext: u64,
    /// Template length.
    pub tlen: i64,
    /// Read sequence.
    pub seq: Vec<u8>,
    /// Quality string (Phred33 ASCII).
    pub qual: Vec<u8>,
}

impl SamRecord {
    /// Whether this read is mapped.
    #[must_use]
    pub const fn is_mapped(&self) -> bool {
        self.flag & FLAG_UNMAPPED == 0
    }

    /// Whether this read is on the reverse strand.
    #[must_use]
    pub const fn is_reverse(&self) -> bool {
        self.flag & FLAG_REVERSE != 0
    }

    /// Whether this is a primary alignment.
    #[must_use]
    pub const fn is_primary(&self) -> bool {
        self.flag & FLAG_SECONDARY == 0 && self.flag & FLAG_SUPPLEMENTARY == 0
    }

    /// End position on the reference (1-based, exclusive).
    #[must_use]
    pub fn end_pos(&self) -> u64 {
        self.pos + reference_length(&self.cigar)
    }

    /// Write this record as a SAM line (no trailing newline).
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        write!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.qname,
            self.flag,
            self.rname,
            self.pos,
            self.mapq,
            format_cigar(&self.cigar),
            self.rnext,
            self.pnext,
            self.tlen,
            String::from_utf8_lossy(&self.seq),
            String::from_utf8_lossy(&self.qual),
        )
    }
}

/// Parse a single SAM line into a `SamRecord`.
///
/// # Errors
///
/// Returns [`Error::Sam`] if the line has too few fields or invalid values.
pub fn parse_record(line: &str) -> Result<SamRecord> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 11 {
        return Err(Error::Sam(format!(
            "expected 11+ fields, got {}",
            fields.len()
        )));
    }

    let flag = fields[1]
        .parse::<u16>()
        .map_err(|_| Error::Sam(format!("invalid FLAG: {}", fields[1])))?;
    let pos = fields[3]
        .parse::<u64>()
        .map_err(|_| Error::Sam(format!("invalid POS: {}", fields[3])))?;
    let mapq = fields[4]
        .parse::<u8>()
        .map_err(|_| Error::Sam(format!("invalid MAPQ: {}", fields[4])))?;
    let cigar = parse_cigar(fields[5])?;
    let pnext = fields[7]
        .parse::<u64>()
        .map_err(|_| Error::Sam(format!("invalid PNEXT: {}", fields[7])))?;
    let tlen = fields[8]
        .parse::<i64>()
        .map_err(|_| Error::Sam(format!("invalid TLEN: {}", fields[8])))?;

    Ok(SamRecord {
        qname: fields[0].to_string(),
        flag,
        rname: fields[2].to_string(),
        pos,
        mapq,
        cigar,
        rnext: fields[6].to_string(),
        pnext,
        tlen,
        seq: fields[9].as_bytes().to_vec(),
        qual: fields[10].as_bytes().to_vec(),
    })
}

// ── SAM Header ───────────────────────────────────────────────────

/// A SAM header `@SQ` reference sequence entry.
#[derive(Debug, Clone)]
pub struct SamRefSeq {
    /// Sequence name.
    pub name: String,
    /// Sequence length.
    pub length: u64,
}

/// SAM file header.
#[derive(Debug, Clone, Default)]
pub struct SamHeader {
    /// Header lines (raw `@HD`, `@PG`, etc.).
    pub header_lines: Vec<String>,
    /// Reference sequences from `@SQ` lines.
    pub references: Vec<SamRefSeq>,
}

impl SamHeader {
    /// Write the header to a SAM output.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        for line in &self.header_lines {
            writeln!(w, "{line}")?;
        }
        for sq in &self.references {
            writeln!(w, "@SQ\tSN:{}\tLN:{}", sq.name, sq.length)?;
        }
        Ok(())
    }
}

// ── SAM Reader ───────────────────────────────────────────────────

/// Streaming SAM reader.
pub struct SamReader {
    reader: BufReader<File>,
    path: std::path::PathBuf,
    /// Parsed header (populated after `open`).
    pub header: SamHeader,
    line_buf: String,
    done: bool,
}

impl SamReader {
    /// Open a SAM file, parsing the header.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut reader = BufReader::new(file);
        let mut header = SamHeader::default();
        let mut line_buf = String::new();
        let mut first_record_line = None;

        loop {
            line_buf.clear();
            let n = reader.read_line(&mut line_buf).map_err(|e| Error::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            if n == 0 {
                break;
            }
            let trimmed = line_buf.trim_end();
            if trimmed.starts_with('@') {
                if trimmed.starts_with("@SQ") {
                    let mut name = String::new();
                    let mut length = 0u64;
                    for field in trimmed.split('\t').skip(1) {
                        if let Some(val) = field.strip_prefix("SN:") {
                            name = val.to_string();
                        } else if let Some(val) = field.strip_prefix("LN:") {
                            length = val.parse().unwrap_or(0);
                        }
                    }
                    header.references.push(SamRefSeq { name, length });
                }
                header.header_lines.push(trimmed.to_string());
            } else {
                first_record_line = Some(trimmed.to_string());
                break;
            }
        }

        let mut sam_reader = Self {
            reader,
            path: path.to_path_buf(),
            header,
            line_buf: String::new(),
            done: false,
        };

        if let Some(line) = first_record_line {
            sam_reader.line_buf = line;
        } else {
            sam_reader.done = true;
        }

        Ok(sam_reader)
    }
}

impl Iterator for SamReader {
    type Item = Result<SamRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if !self.line_buf.is_empty() {
            let line = std::mem::take(&mut self.line_buf);
            return Some(parse_record(&line));
        }

        let mut buf = String::new();
        loop {
            buf.clear();
            match self.reader.read_line(&mut buf) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {
                    let trimmed = buf.trim_end();
                    if trimmed.is_empty() {
                        continue;
                    }
                    return Some(parse_record(trimmed));
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(Error::Io {
                        path: self.path.clone(),
                        source: e,
                    }));
                }
            }
        }
    }
}

// ── SAM Writer ───────────────────────────────────────────────────

/// Buffered SAM writer.
pub struct SamWriter<W: Write> {
    writer: W,
}

impl<W: Write> SamWriter<W> {
    /// Create a new SAM writer.
    pub const fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Write the SAM header.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_header(&mut self, header: &SamHeader) -> std::io::Result<()> {
        header.write_to(&mut self.writer)
    }

    /// Write a single alignment record.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn write_record(&mut self, record: &SamRecord) -> std::io::Result<()> {
        record.write_to(&mut self.writer)?;
        writeln!(self.writer)
    }

    /// Flush the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// Sort SAM records by reference position (coordinate order).
pub fn sort_by_position(records: &mut [SamRecord]) {
    records.sort_by(|a, b| a.rname.cmp(&b.rname).then(a.pos.cmp(&b.pos)));
}
