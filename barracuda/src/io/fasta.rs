// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign FASTA parser — zero external parsing dependencies.
//!
//! Streams records from disk via [`BufReader`]. Handles both plain and
//! gzip-compressed files (`.gz` extension, via `flate2::read::GzDecoder`).
//!
//! # Format
//!
//! ```text
//! >identifier description
//! SEQUENCE (may span multiple lines)
//! ```
//!
//! All I/O is byte-native — no UTF-8 assumption for sequence data.
//! Header identifiers are extracted as lossy UTF-8.
//!
//! For reference genome loading (e.g. REL606, 4.63 Mb), prefer
//! [`FastaRecord::load_all`] which reads the entire file into memory.
//! For streaming large multi-FASTA files, use [`FastaIter`].

#[cfg(test)]
mod tests;

use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Concrete reader for plain or gzip-compressed FASTA files.
enum FastaReader {
    Plain(BufReader<File>),
    Gz(BufReader<flate2::read::GzDecoder<File>>),
}

impl Read for FastaReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(r) => r.read(buf),
            Self::Gz(r) => r.read(buf),
        }
    }
}

impl BufRead for FastaReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        match self {
            Self::Plain(r) => r.fill_buf(),
            Self::Gz(r) => r.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            Self::Plain(r) => r.consume(amt),
            Self::Gz(r) => r.consume(amt),
        }
    }
}

fn open_reader(path: &Path) -> Result<FastaReader> {
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
        Ok(FastaReader::Gz(BufReader::new(decoder)))
    } else {
        Ok(FastaReader::Plain(BufReader::new(file)))
    }
}

/// A parsed FASTA record with owned data.
#[derive(Debug, Clone)]
pub struct FastaRecord {
    /// Record identifier (first word after `>`).
    pub id: String,
    /// Full description line (everything after `>`).
    pub description: String,
    /// Concatenated sequence (all continuation lines, uppercase, no whitespace).
    pub sequence: Vec<u8>,
}

impl FastaRecord {
    /// Sequence length in bases.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Whether this record has an empty sequence.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// GC content as a fraction in \[0, 1\].
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "Precision: sequence length bounded by genome size"
    )]
    pub fn gc_content(&self) -> f64 {
        if self.sequence.is_empty() {
            return 0.0;
        }
        let gc = self
            .sequence
            .iter()
            .filter(|&&b| matches!(b, b'G' | b'C' | b'g' | b'c'))
            .count();
        gc as f64 / self.sequence.len() as f64
    }

    /// Load all records from a FASTA file into memory.
    ///
    /// Suitable for reference genomes (typically < 20 MB).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] or [`Error::Fasta`] on parse failure.
    pub fn load_all(path: &Path) -> Result<Vec<Self>> {
        let iter = FastaIter::open(path)?;
        iter.collect()
    }
}

/// A GenBank feature (CDS, gene, rRNA, etc.) parsed from a `.gbk` file.
#[derive(Debug, Clone)]
pub struct GenBankFeature {
    /// Feature type (e.g. "CDS", "gene", "rRNA").
    pub feature_type: String,
    /// 1-based start position.
    pub start: usize,
    /// 1-based end position (inclusive).
    pub end: usize,
    /// Strand: true = forward (+), false = reverse complement (-).
    pub forward: bool,
    /// Gene name from `/gene=` qualifier, if present.
    pub gene: Option<String>,
    /// Product from `/product=` qualifier, if present.
    pub product: Option<String>,
    /// Locus tag from `/locus_tag=` qualifier, if present.
    pub locus_tag: Option<String>,
}

/// Parsed GenBank file with sequence and feature table.
#[derive(Debug, Clone)]
pub struct GenBankRecord {
    /// Locus name from the LOCUS line.
    pub locus: String,
    /// Full nucleotide sequence.
    pub sequence: Vec<u8>,
    /// Parsed feature table entries.
    pub features: Vec<GenBankFeature>,
}

impl GenBankRecord {
    /// Load a GenBank flat file (`.gbk` / `.gb`).
    ///
    /// Parses the FEATURES table for CDS/gene annotations and the ORIGIN
    /// section for the nucleotide sequence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] or [`Error::Fasta`] on parse failure.
    pub fn load(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::parse(&contents, path)
    }

    fn parse(contents: &str, path: &Path) -> Result<Self> {
        let mut locus = String::new();
        let mut features = Vec::new();
        let mut sequence = Vec::new();
        let mut in_features = false;
        let mut in_origin = false;
        let mut current_feature: Option<GenBankFeature> = None;
        let mut current_qualifier_value = String::new();
        let mut current_qualifier_key = String::new();

        for line in contents.lines() {
            if in_origin {
                if line.starts_with("//") {
                    break;
                }
                for ch in line.bytes() {
                    if ch.is_ascii_alphabetic() {
                        sequence.push(ch.to_ascii_uppercase());
                    }
                }
                continue;
            }

            if line.starts_with("LOCUS") {
                locus = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .to_string();
                continue;
            }

            if line.starts_with("FEATURES") {
                in_features = true;
                continue;
            }

            if line.starts_with("ORIGIN") {
                if let Some(feat) = current_feature.take() {
                    apply_qualifier(&mut features, feat, &current_qualifier_key, &current_qualifier_value);
                }
                in_features = false;
                in_origin = true;
                continue;
            }

            if !in_features {
                continue;
            }

            let indent = line.len() - line.trim_start().len();
            let trimmed = line.trim();

            // GenBank: feature keys at column 6 (indent ~5), qualifiers at column 22 (indent ~21)
            if indent < 21 && !trimmed.is_empty() && !trimmed.starts_with('/') {
                if let Some(feat) = current_feature.take() {
                    apply_qualifier(&mut features, feat, &current_qualifier_key, &current_qualifier_value);
                }
                current_qualifier_key.clear();
                current_qualifier_value.clear();

                let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
                if parts.len() == 2 {
                    let (start, end, forward) = parse_location(parts[1].trim());
                    current_feature = Some(GenBankFeature {
                        feature_type: parts[0].to_string(),
                        start,
                        end,
                        forward,
                        gene: None,
                        product: None,
                        locus_tag: None,
                    });
                }
            } else if trimmed.starts_with('/') {
                if let Some(ref mut feat) = current_feature {
                    if !current_qualifier_key.is_empty() {
                        apply_qualifier_to_feat(feat, &current_qualifier_key, &current_qualifier_value);
                    }
                    if let Some(eq_pos) = trimmed.find('=') {
                        current_qualifier_key = trimmed[1..eq_pos].to_string();
                        current_qualifier_value = trimmed[eq_pos + 1..].trim_matches('"').to_string();
                    } else {
                        current_qualifier_key = trimmed[1..].to_string();
                        current_qualifier_value.clear();
                    }
                }
            } else if current_feature.is_some() && !current_qualifier_key.is_empty() {
                current_qualifier_value.push(' ');
                current_qualifier_value.push_str(trimmed.trim_matches('"'));
            }
        }

        if let Some(feat) = current_feature.take() {
            apply_qualifier(&mut features, feat, &current_qualifier_key, &current_qualifier_value);
        }

        if sequence.is_empty() {
            return Err(Error::Fasta(format!(
                "{}: no ORIGIN sequence found",
                path.display()
            )));
        }

        Ok(Self {
            locus,
            sequence,
            features,
        })
    }

    /// Number of CDS features.
    #[must_use]
    pub fn cds_count(&self) -> usize {
        self.features
            .iter()
            .filter(|f| f.feature_type == "CDS")
            .count()
    }

    /// Find the feature (typically CDS) that contains a given 1-based position.
    #[must_use]
    pub fn feature_at(&self, position: usize) -> Option<&GenBankFeature> {
        self.features
            .iter()
            .find(|f| f.feature_type == "CDS" && position >= f.start && position <= f.end)
    }
}

fn apply_qualifier(
    features: &mut Vec<GenBankFeature>,
    mut feat: GenBankFeature,
    key: &str,
    value: &str,
) {
    if !key.is_empty() {
        apply_qualifier_to_feat(&mut feat, key, value);
    }
    features.push(feat);
}

fn apply_qualifier_to_feat(feat: &mut GenBankFeature, key: &str, value: &str) {
    match key {
        "gene" => feat.gene = Some(value.to_string()),
        "product" => feat.product = Some(value.to_string()),
        "locus_tag" => feat.locus_tag = Some(value.to_string()),
        _ => {}
    }
}

fn parse_location(loc: &str) -> (usize, usize, bool) {
    let (inner, forward) = if let Some(stripped) = loc.strip_prefix("complement(") {
        (stripped.trim_end_matches(')'), false)
    } else {
        (loc, true)
    };
    let inner = inner
        .strip_prefix("join(")
        .unwrap_or(inner)
        .trim_end_matches(')');

    let inner = inner.replace(['<', '>'], "");

    let (start, end) = if let Some((s, e)) = inner.split_once("..") {
        let s_val = s
            .split(',')
            .next()
            .and_then(|x| x.trim().parse::<usize>().ok())
            .unwrap_or(1);
        let e_parts: Vec<&str> = e.split(',').collect();
        let e_val = e_parts
            .last()
            .and_then(|x| {
                x.trim()
                    .split("..")
                    .last()
                    .and_then(|y| y.parse::<usize>().ok())
            })
            .unwrap_or(s_val);
        (s_val, e_val)
    } else {
        let val = inner.trim().parse::<usize>().unwrap_or(1);
        (val, val)
    };

    (start, end, forward)
}

/// Streaming FASTA iterator — yields one record at a time.
///
/// Handles multi-line sequences and both plain and gzip-compressed files.
pub struct FastaIter {
    reader: FastaReader,
    path: std::path::PathBuf,
    line_buf: Vec<u8>,
    pending_header: Option<Vec<u8>>,
    done: bool,
}

impl FastaIter {
    /// Open a FASTA file for streaming iteration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        let reader = open_reader(path)?;
        Ok(Self {
            reader,
            path: path.to_path_buf(),
            line_buf: Vec::new(),
            pending_header: None,
            done: false,
        })
    }
}

impl Iterator for FastaIter {
    type Item = Result<FastaRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let header = if let Some(h) = self.pending_header.take() {
            h
        } else {
            loop {
                self.line_buf.clear();
                match read_line(&mut self.reader, &mut self.line_buf, &self.path) {
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
                let trimmed = trim_end(&self.line_buf);
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed[0] == b'>' {
                    break trimmed.to_vec();
                }
                self.done = true;
                return Some(Err(Error::Fasta(format!(
                    "expected '>' header, got: {}",
                    String::from_utf8_lossy(&trimmed[..trimmed.len().min(40)])
                ))));
            }
        };

        let desc_bytes = &header[1..];
        let description = String::from_utf8_lossy(desc_bytes).to_string();
        let id = description
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();

        let mut sequence = Vec::new();

        loop {
            self.line_buf.clear();
            match read_line(&mut self.reader, &mut self.line_buf, &self.path) {
                Ok(0) => {
                    self.done = true;
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
            let trimmed = trim_end(&self.line_buf);
            if trimmed.is_empty() {
                continue;
            }
            if trimmed[0] == b'>' {
                self.pending_header = Some(trimmed.to_vec());
                break;
            }
            for &b in trimmed {
                if b.is_ascii_alphabetic() {
                    sequence.push(b.to_ascii_uppercase());
                }
            }
        }

        Some(Ok(FastaRecord {
            id,
            description,
            sequence,
        }))
    }
}

fn read_line(reader: &mut impl BufRead, buf: &mut Vec<u8>, path: &Path) -> Result<usize> {
    reader.read_until(b'\n', buf).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

fn trim_end(buf: &[u8]) -> &[u8] {
    let mut end = buf.len();
    while end > 0 && matches!(buf[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    &buf[..end]
}
