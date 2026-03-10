// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzXML parser — legacy mass spectrometry data I/O.
//!
//! Parses mzXML files **streaming from disk** using the sovereign
//! `XmlReader` pull parser with a `BufReader`. Never loads the entire
//! file into memory. Decodes base64-encoded and optionally
//! zlib-compressed peak arrays.
//!
//! mzXML is a legacy format predating mzML. It uses `<scan>` elements
//! with `<peaks>` children containing base64 network-byte-order
//! (big-endian) float arrays. Each `<scan>` carries attributes
//! directly (msLevel, retentionTime, peaksCount, etc.) rather than
//! using CV-param references.
//!
//! # References
//!
//! - Pedrioli et al. 2004, *Nat Biotechnol* 22:1459-1466 (mzXML spec)
//! - `<http://sashimi.sourceforge.net/schema_revision/mzXML_3.2/>`

use crate::error::{Error, Result};
use crate::io::xml::{XmlEvent, XmlReader};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// A single mass spectrum parsed from mzXML.
///
/// Field layout matches [`crate::io::mzml::MzmlSpectrum`] for downstream
/// interoperability — callers can use either parser and feed results
/// into the same EIC / feature-detection pipeline.
#[derive(Debug, Clone)]
pub struct MzxmlSpectrum {
    /// Scan number (1-based in file, stored 0-based here).
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
    /// m/z array (deinterleaved from `<peaks>`).
    pub mz_array: Vec<f64>,
    /// Intensity array (deinterleaved from `<peaks>`).
    pub intensity_array: Vec<f64>,
}

/// Streaming iterator that yields one [`MzxmlSpectrum`] at a time.
pub struct MzxmlIter {
    reader: XmlReader<BufReader<File>>,
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
                        if name == "scan" {
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
                        } else if name == "peaks" && builder.is_some() {
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
                        if name == "peaks" {
                            in_peaks = false;
                            if let Some(ref mut b) = builder {
                                match decode_peaks(
                                    &peaks_text,
                                    peaks_precision,
                                    peaks_byte_order,
                                    peaks_compression,
                                ) {
                                    Ok((mz, intensity)) => {
                                        b.mz_array = mz;
                                        b.intensity_array = intensity;
                                    }
                                    Err(e) => return Some(Err(e)),
                                }
                            }
                        } else if name == "scan" {
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

// ── Internal types ──────────────────────────────────────────────

#[derive(Default)]
struct ScanBuilder {
    index: usize,
    ms_level: u32,
    rt: f64,
    tic: f64,
    base_peak_mz: f64,
    base_peak_intensity: f64,
    lowest_mz: f64,
    highest_mz: f64,
    mz_array: Vec<f64>,
    intensity_array: Vec<f64>,
}

impl ScanBuilder {
    fn build(self) -> MzxmlSpectrum {
        MzxmlSpectrum {
            index: self.index.saturating_sub(1), // 1-based → 0-based
            ms_level: self.ms_level,
            rt_minutes: self.rt,
            tic: self.tic,
            base_peak_mz: self.base_peak_mz,
            base_peak_intensity: self.base_peak_intensity,
            lowest_mz: self.lowest_mz,
            highest_mz: self.highest_mz,
            mz_array: self.mz_array,
            intensity_array: self.intensity_array,
        }
    }
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Network, // big-endian (mzXML default)
    Little,
}

#[derive(Clone, Copy)]
enum Compression {
    None,
    Zlib,
}

/// Parse mzXML retention time string (ISO 8601 duration like "PT1.234S").
fn parse_retention_time(s: &str) -> f64 {
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
fn decode_peaks(
    encoded: &str,
    precision: u8,
    byte_order: ByteOrder,
    compression: Compression,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let trimmed = encoded.trim();
    if trimmed.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let raw = crate::encoding::base64_decode(trimmed)?;

    let bytes: Vec<u8> = match compression {
        Compression::Zlib => {
            use flate2::read::ZlibDecoder;
            use std::io::Read;
            let mut dec = ZlibDecoder::new(&raw[..]);
            let mut buf = Vec::new();
            dec.read_to_end(&mut buf)
                .map_err(|e| Error::Zlib(format!("{e}")))?;
            buf
        }
        Compression::None => raw,
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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_peaks_b64_32be(pairs: &[(f32, f32)]) -> String {
        let bytes: Vec<u8> = pairs
            .iter()
            .flat_map(|(mz, int)| {
                let mut v = Vec::with_capacity(8);
                v.extend_from_slice(&mz.to_be_bytes());
                v.extend_from_slice(&int.to_be_bytes());
                v
            })
            .collect();
        crate::encoding::base64_encode(&bytes)
    }

    fn make_peaks_b64_64be(pairs: &[(f64, f64)]) -> String {
        let bytes: Vec<u8> = pairs
            .iter()
            .flat_map(|(mz, int)| {
                let mut v = Vec::with_capacity(16);
                v.extend_from_slice(&mz.to_be_bytes());
                v.extend_from_slice(&int.to_be_bytes());
                v
            })
            .collect();
        crate::encoding::base64_encode(&bytes)
    }

    fn minimal_mzxml(peaks_b64: &str, precision: u8) -> String {
        format!(
            r#"<?xml version="1.0" encoding="ISO-8859-1"?>
<mzXML>
  <msRun>
    <scan num="1" msLevel="1" retentionTime="PT60.0S" totIonCurrent="5000" basePeakMz="200.0" basePeakIntensity="3000" lowMz="100.0" highMz="300.0">
      <peaks precision="{precision}" byteOrder="network" contentType="m/z-int">{peaks_b64}</peaks>
    </scan>
  </msRun>
</mzXML>"#
        )
    }

    fn write_temp(dir: &tempfile::TempDir, name: &str, content: &str) -> std::path::PathBuf {
        use std::io::Write;
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parse_retention_time_iso8601() {
        assert!((parse_retention_time("PT60.0S") - 1.0).abs() < 1e-6);
        assert!((parse_retention_time("PT120.0S") - 2.0).abs() < 1e-6);
        assert!((parse_retention_time("PT0.0S") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn decode_peaks_32bit_be() {
        let pairs = [(100.0_f32, 500.0_f32), (200.0, 1500.0), (300.0, 1000.0)];
        let b64 = make_peaks_b64_32be(&pairs);
        let (mz, int) = decode_peaks(&b64, 32, ByteOrder::Network, Compression::None).unwrap();
        assert_eq!(mz.len(), 3);
        assert!((mz[0] - 100.0).abs() < 0.01);
        assert!((mz[1] - 200.0).abs() < 0.01);
        assert!((mz[2] - 300.0).abs() < 0.01);
        assert!((int[0] - 500.0).abs() < 0.01);
        assert!((int[1] - 1500.0).abs() < 0.01);
    }

    #[test]
    fn decode_peaks_64bit_be() {
        let pairs = [(100.0_f64, 500.0_f64), (200.0, 1500.0)];
        let b64 = make_peaks_b64_64be(&pairs);
        let (mz, int) = decode_peaks(&b64, 64, ByteOrder::Network, Compression::None).unwrap();
        assert_eq!(mz.len(), 2);
        assert!((mz[0] - 100.0).abs() < f64::EPSILON);
        assert!((int[1] - 1500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn decode_empty_peaks() {
        let (mz, int) = decode_peaks("", 32, ByteOrder::Network, Compression::None).unwrap();
        assert!(mz.is_empty());
        assert!(int.is_empty());
    }

    #[test]
    fn decode_peaks_wrong_length() {
        let b64 = crate::encoding::base64_encode(&[1, 2, 3]);
        let result = decode_peaks(&b64, 32, ByteOrder::Network, Compression::None);
        assert!(result.is_err());
    }

    #[test]
    fn parse_mzxml_32bit() {
        let pairs = [(100.0_f32, 500.0_f32), (200.0, 1500.0), (300.0, 1000.0)];
        let b64 = make_peaks_b64_32be(&pairs);
        let xml = minimal_mzxml(&b64, 32);

        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "test.mzXML", &xml);

        let spectra = parse_mzxml(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        let s = &spectra[0];
        assert_eq!(s.index, 0); // num=1 → 0-based
        assert_eq!(s.ms_level, 1);
        assert!((s.rt_minutes - 1.0).abs() < 1e-6);
        assert!((s.tic - 5000.0).abs() < 1e-6);
        assert_eq!(s.mz_array.len(), 3);
        assert!((s.mz_array[0] - 100.0).abs() < 0.01);
        assert!((s.mz_array[2] - 300.0).abs() < 0.01);
        assert!((s.intensity_array[1] - 1500.0).abs() < 0.01);
    }

    #[test]
    fn parse_mzxml_64bit() {
        let pairs = [(150.5_f64, 750.0_f64), (250.5, 2000.0)];
        let b64 = make_peaks_b64_64be(&pairs);
        let xml = minimal_mzxml(&b64, 64);

        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "test64.mzXML", &xml);

        let spectra = parse_mzxml(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].mz_array.len(), 2);
        assert!((spectra[0].mz_array[0] - 150.5).abs() < f64::EPSILON);
    }

    #[test]
    fn streaming_iter_matches_collect() {
        let pairs = [(100.0_f32, 500.0_f32), (200.0, 1000.0)];
        let b64 = make_peaks_b64_32be(&pairs);
        let xml = minimal_mzxml(&b64, 32);

        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "stream.mzXML", &xml);

        let collected = parse_mzxml(&path).unwrap();
        let mut iterated = Vec::new();
        for result in MzxmlIter::open(&path).unwrap() {
            iterated.push(result.unwrap());
        }
        assert_eq!(collected.len(), iterated.len());
        assert_eq!(collected[0].mz_array, iterated[0].mz_array);
    }

    #[test]
    fn for_each_scan_works() {
        let pairs = [(100.0_f32, 500.0_f32)];
        let b64 = make_peaks_b64_32be(&pairs);
        let xml = minimal_mzxml(&b64, 32);

        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "callback.mzXML", &xml);

        let mut count = 0_usize;
        for_each_scan(&path, |_| {
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn nonexistent_file_returns_error() {
        let path = std::env::temp_dir().join("nonexistent_mzxml_test_file.mzXML");
        assert!(MzxmlIter::open(&path).is_err());
    }

    #[test]
    fn multi_scan_mzxml() {
        let b64_1 = make_peaks_b64_32be(&[(100.0, 500.0)]);
        let b64_2 = make_peaks_b64_32be(&[(200.0, 1000.0), (300.0, 750.0)]);
        let xml = format!(
            r#"<?xml version="1.0" encoding="ISO-8859-1"?>
<mzXML>
  <msRun>
    <scan num="1" msLevel="1" retentionTime="PT30.0S">
      <peaks precision="32" byteOrder="network" contentType="m/z-int">{b64_1}</peaks>
    </scan>
    <scan num="2" msLevel="2" retentionTime="PT60.0S">
      <peaks precision="32" byteOrder="network" contentType="m/z-int">{b64_2}</peaks>
    </scan>
  </msRun>
</mzXML>"#
        );

        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "multi.mzXML", &xml);

        let spectra = parse_mzxml(&path).unwrap();
        assert_eq!(spectra.len(), 2);
        assert_eq!(spectra[0].ms_level, 1);
        assert_eq!(spectra[1].ms_level, 2);
        assert_eq!(spectra[0].mz_array.len(), 1);
        assert_eq!(spectra[1].mz_array.len(), 2);
    }
}
