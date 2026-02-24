// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzML parser — vendor-neutral mass spectrometry data I/O.
//!
//! Parses mzML XML files **streaming from disk** using a sovereign
//! pull parser (internal `xml` module) with a `BufReader`.  Never loads
//! the entire file into memory.  Decodes base64-encoded and
//! optionally zlib-compressed m/z + intensity arrays.

mod decode;
mod stats;
#[cfg(test)]
mod tests;

pub use stats::{compute_stats, stats_from_file};

use crate::error::{Error, Result};
use crate::io::xml::{XmlEvent, XmlReader};
use decode::{BinaryArrayState, DecodeBuffer, SpectrumBuilder};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

// ── Public types ────────────────────────────────────────────────

/// A single mass spectrum parsed from mzML.
#[derive(Debug, Clone)]
pub struct MzmlSpectrum {
    /// Spectrum index (0-based).
    pub index: usize,
    /// MS level (1 = MS1, 2 = MS2, etc.).
    pub ms_level: u32,
    /// Retention time in minutes.
    pub rt_minutes: f64,
    /// Total ion current.
    pub tic: f64,
    /// Base peak m/z.
    pub base_peak_mz: f64,
    /// Base peak intensity.
    pub base_peak_intensity: f64,
    /// Lowest observed m/z.
    pub lowest_mz: f64,
    /// Highest observed m/z.
    pub highest_mz: f64,
    /// m/z array.
    pub mz_array: Vec<f64>,
    /// Intensity array.
    pub intensity_array: Vec<f64>,
}

/// Summary statistics from an mzML file.
#[derive(Debug, Clone)]
pub struct MzmlStats {
    /// Total number of spectra.
    pub num_spectra: usize,
    /// Number of MS1 spectra.
    pub num_ms1: usize,
    /// Number of MS2 spectra.
    pub num_ms2: usize,
    /// Minimum retention time (minutes), or `None` if empty.
    pub min_rt: Option<f64>,
    /// Maximum retention time (minutes), or `None` if empty.
    pub max_rt: Option<f64>,
    /// Minimum observed m/z, or `None` if empty.
    pub min_mz: Option<f64>,
    /// Maximum observed m/z, or `None` if empty.
    pub max_mz: Option<f64>,
    /// Total number of peaks across all spectra.
    pub total_peaks: usize,
}

// ── Public API ──────────────────────────────────────────────────

/// Streaming iterator that yields one [`MzmlSpectrum`] at a time without
/// buffering the entire file.
///
/// Wraps the sovereign `XmlReader` pull parser and decodes binary arrays
/// on demand for each `<spectrum>` element.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use wetspring_barracuda::io::mzml;
///
/// let iter = mzml::MzmlIter::open(Path::new("data.mzML")).unwrap();
/// for result in iter {
///     let spectrum = result.unwrap();
///     println!("index {} — {} peaks", spectrum.index, spectrum.mz_array.len());
/// }
/// ```
pub struct MzmlIter {
    reader: XmlReader<BufReader<File>>,
    decode_buffer: DecodeBuffer,
    done: bool,
}

impl MzmlIter {
    /// Open an mzML file for streaming iteration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let buf_reader = BufReader::new(file);
        let mut reader = XmlReader::new(buf_reader);
        reader.set_trim_text(true);
        Ok(Self {
            reader,
            decode_buffer: DecodeBuffer::new(),
            done: false,
        })
    }
}

impl Iterator for MzmlIter {
    type Item = Result<MzmlSpectrum>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut builder: Option<SpectrumBuilder> = None;
        let mut binary_state = BinaryArrayState::new();
        let mut in_binary_data_array = false;
        let mut in_binary = false;

        loop {
            match self.reader.next_event() {
                Ok(event) => match event {
                    XmlEvent::StartElement {
                        ref name,
                        ref attrs,
                    } => match name.as_str() {
                        "spectrum" => {
                            let idx = attrs
                                .iter()
                                .find(|(k, _)| k == "index")
                                .and_then(|(_, v)| v.parse().ok())
                                .unwrap_or(0);
                            builder = Some(SpectrumBuilder::new(idx));
                        }
                        "binaryDataArray" => {
                            in_binary_data_array = true;
                            binary_state.reset();
                        }
                        "binary" => {
                            in_binary = true;
                            binary_state.text.clear();
                        }
                        "cvParam" if builder.is_some() => {
                            let accession = attrs
                                .iter()
                                .find(|(k, _)| k == "accession")
                                .map_or("", |(_, v)| v.as_str());
                            let cv_value = attrs
                                .iter()
                                .find(|(k, _)| k == "value")
                                .map_or("", |(_, v)| v.as_str());

                            if in_binary_data_array {
                                binary_state.apply_cv_param(accession);
                            } else if let Some(ref mut b) = builder {
                                b.apply_cv_param(accession, cv_value);
                            }
                        }
                        _ => {}
                    },
                    XmlEvent::Text(ref text) => {
                        if in_binary {
                            binary_state.text.push_str(text);
                        }
                    }
                    XmlEvent::EndElement { ref name } => match name.as_str() {
                        "spectrum" => {
                            return builder.take().map(|b| Ok(b.build()));
                        }
                        "binary" => in_binary = false,
                        "binaryDataArray" => {
                            if let Some(ref mut b) = builder {
                                if let Err(e) = binary_state
                                    .decode_into_with_buffer(b, Some(&mut self.decode_buffer))
                                {
                                    return Some(Err(e));
                                }
                            }
                            in_binary_data_array = false;
                        }
                        _ => {}
                    },
                    XmlEvent::Eof => {
                        self.done = true;
                        return None;
                    }
                },
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

/// Parse an mzML file and collect **all** spectra into memory.
///
/// Delegates to [`MzmlIter`] — XML streams from disk, but results are
/// collected into a `Vec`. For large files, prefer iterating with
/// [`MzmlIter`] directly.
///
/// # Examples
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use std::path::Path;
/// use wetspring_barracuda::io::mzml;
///
/// let spectra = mzml::parse_mzml(Path::new("data.mzML"))?;
/// assert!(!spectra.is_empty());
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, [`Error::Xml`]
/// for XML structure errors, or [`Error::Base64`] / [`Error::Zlib`]
/// for binary array decoding failures.
#[deprecated(
    since = "0.2.0",
    note = "collects all spectra; use `MzmlIter` for streaming"
)]
pub fn parse_mzml(path: &Path) -> Result<Vec<MzmlSpectrum>> {
    MzmlIter::open(path)?.collect()
}
