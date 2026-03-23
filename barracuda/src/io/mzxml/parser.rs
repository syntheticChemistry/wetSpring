// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzXML streaming pull parser.
//!
//! Parses mzXML files from disk using `XmlReader` with a `BufReader`.
//! Never loads the entire file into memory. Decodes base64-encoded
//! and optionally zlib-compressed peak arrays.

use super::types::{ByteOrder, Compression, MzxmlSpectrum, ScanBuilder, ZlibBuffer};
use crate::error::{Error, Result};
use crate::io::xml::{XmlEvent, XmlReader};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Streaming iterator that yields one [`MzxmlSpectrum`] at a time.
pub struct MzxmlIter {
    reader: XmlReader<BufReader<File>>,
    zlib_buf: ZlibBuffer,
    done: bool,
}

impl MzxmlIter {
    /// Open an mzXML file for streaming iteration.
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
            zlib_buf: ZlibBuffer::default(),
            done: false,
        })
    }
}

impl Iterator for MzxmlIter {
    type Item = Result<MzxmlSpectrum>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut builder: Option<ScanBuilder> = None;
        let mut in_peaks = false;
        let mut peaks_text = String::new();
        let mut peaks_precision = 32_u8;
        let mut peaks_byte_order = ByteOrder::Network;
        let mut peaks_compression = Compression::None;

        loop {
            match self.reader.next_event() {
                Ok(event) => match event {
                    XmlEvent::StartElement {
                        ref name,
                        ref attrs,
                    } => {
                        if name.as_ref() == "scan" {
                            let mut b = ScanBuilder::default();
                            for (k, v) in attrs {
                                match k.as_str() {
                                    "num" => b.index = v.parse().unwrap_or(0),
                                    "msLevel" => b.ms_level = v.parse().unwrap_or(1),
                                    "retentionTime" => b.rt = parse_retention_time(v),
                                    "totIonCurrent" => b.tic = v.parse().unwrap_or(0.0),
                                    "basePeakMz" => b.base_peak_mz = v.parse().unwrap_or(0.0),
                                    "basePeakIntensity" => {
                                        b.base_peak_intensity = v.parse().unwrap_or(0.0);
                                    }
                                    "lowMz" => b.lowest_mz = v.parse().unwrap_or(0.0),
                                    "highMz" => b.highest_mz = v.parse().unwrap_or(0.0),
                                    _ => {}
                                }
                            }
                            builder = Some(b);
                        } else if name.as_ref() == "peaks" && builder.is_some() {
                            in_peaks = true;
                            peaks_text.clear();
                            peaks_precision = 32;
                            peaks_byte_order = ByteOrder::Network;
                            peaks_compression = Compression::None;
                            for (k, v) in attrs {
                                match k.as_str() {
                                    "precision" => {
                                        peaks_precision = v.parse().unwrap_or(32);
                                    }
                                    "byteOrder" => {
                                        peaks_byte_order = if v == "network" {
                                            ByteOrder::Network
                                        } else {
                                            ByteOrder::Little
                                        };
                                    }
                                    "compressionType" => {
                                        peaks_compression = if v == "zlib" {
                                            Compression::Zlib
                                        } else {
                                            Compression::None
                                        };
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    XmlEvent::Text(ref text) => {
                        if in_peaks {
                            peaks_text.push_str(text);
                        }
                    }
                    XmlEvent::EndElement { ref name } => {
                        if name.as_ref() == "peaks" {
                            in_peaks = false;
                            if let Some(ref mut b) = builder {
                                match decode_peaks(
                                    &peaks_text,
                                    peaks_precision,
                                    peaks_byte_order,
                                    peaks_compression,
                                    &mut self.zlib_buf,
                                ) {
                                    Ok((mz, intensity)) => {
                                        b.mz_array = mz;
                                        b.intensity_array = intensity;
                                    }
                                    Err(e) => return Some(Err(e)),
                                }
                            }
                        } else if name.as_ref() == "scan" {
                            return builder.take().map(|b| Ok(b.build()));
                        }
                    }
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

/// Collect all spectra from an mzXML file via [`MzxmlIter`].
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, or
/// [`Error::Base64`] / [`Error::BinaryFormat`] for decode failures.
pub fn parse_mzxml(path: &Path) -> Result<Vec<MzxmlSpectrum>> {
    MzxmlIter::open(path)?.collect()
}

/// Process each scan without collecting.
///
/// # Errors
///
/// Returns parse or callback errors.
pub fn for_each_scan<F>(path: &Path, mut f: F) -> Result<()>
where
    F: FnMut(MzxmlSpectrum) -> Result<()>,
{
    for result in MzxmlIter::open(path)? {
        f(result?)?;
    }
    Ok(())
}

/// Parse mzXML retention time string (ISO 8601 duration like "PT1.234S").
pub(super) fn parse_retention_time(s: &str) -> f64 {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("PT") {
        if let Some(secs_str) = rest.strip_suffix('S') {
            if let Ok(secs) = secs_str.parse::<f64>() {
                return secs / 60.0;
            }
        }
    }
    s.parse().unwrap_or(0.0)
}

/// Decode base64 mzXML `<peaks>` content into (mz, intensity) arrays.
///
/// mzXML stores interleaved m/z-intensity pairs in network byte order
/// (big-endian) by default, unlike mzML which uses little-endian.
/// Reuses `zlib_buf` across scans to amortize zlib allocation.
pub(super) fn decode_peaks(
    encoded: &str,
    precision: u8,
    byte_order: ByteOrder,
    compression: Compression,
    zlib_buf: &mut ZlibBuffer,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let trimmed = encoded.trim();
    if trimmed.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let raw = crate::encoding::base64_decode(trimmed)?;

    let bytes: &[u8] = match compression {
        Compression::Zlib => {
            use flate2::read::ZlibDecoder;
            use std::io::Read;
            zlib_buf.buf.clear();
            let mut dec = ZlibDecoder::new(&raw[..]);
            dec.read_to_end(&mut zlib_buf.buf)
                .map_err(|e| Error::Zlib(format!("{e}")))?;
            &zlib_buf.buf
        }
        Compression::None => &raw,
    };

    let elem_size: usize = if precision == 64 { 8 } else { 4 };
    let pair_size = elem_size * 2;

    if !bytes.len().is_multiple_of(pair_size) {
        return Err(Error::BinaryFormat(format!(
            "mzXML peaks length {} not divisible by pair size {pair_size}",
            bytes.len()
        )));
    }

    let n_pairs = bytes.len() / pair_size;
    let mut mz = Vec::with_capacity(n_pairs);
    let mut intensity = Vec::with_capacity(n_pairs);

    for i in 0..n_pairs {
        let offset = i * pair_size;
        let mz_val = decode_float(&bytes[offset..offset + elem_size], precision, byte_order);
        let int_val = decode_float(
            &bytes[offset + elem_size..offset + pair_size],
            precision,
            byte_order,
        );
        mz.push(mz_val);
        intensity.push(int_val);
    }

    Ok((mz, intensity))
}

fn decode_float(chunk: &[u8], precision: u8, byte_order: ByteOrder) -> f64 {
    match (precision, byte_order) {
        (64, ByteOrder::Network) => {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&chunk[..8]);
            f64::from_be_bytes(arr)
        }
        (64, ByteOrder::Little) => {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&chunk[..8]);
            f64::from_le_bytes(arr)
        }
        (_, ByteOrder::Network) => {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(&chunk[..4]);
            f64::from(f32::from_be_bytes(arr))
        }
        (_, ByteOrder::Little) => {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(&chunk[..4]);
            f64::from(f32::from_le_bytes(arr))
        }
    }
}
