// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzML parser — vendor-neutral mass spectrometry data I/O.
//!
//! Parses mzML XML files **streaming from disk** using a sovereign
//! pull parser (internal `xml` module) with a `BufReader`.  Never loads
//! the entire file into memory.  Decodes base64-encoded and
//! optionally zlib-compressed m/z + intensity arrays.

mod decode;

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
                                    eprintln!("Warning: binary decode failed: {e}");
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

/// Parse an mzML file and return all spectra, **streaming from disk**.
///
/// Delegates to [`MzmlIter`] — the full XML is never held in memory.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, [`Error::Xml`]
/// for XML structure errors, or [`Error::Base64`] / [`Error::Zlib`]
/// for binary array decoding failures.
pub fn parse_mzml(path: &Path) -> Result<Vec<MzmlSpectrum>> {
    MzmlIter::open(path)?.collect()
}

/// Compute summary statistics in a single streaming pass (zero-copy path).
///
/// Skips binary array decoding entirely — only spectrum-level `cvParam`
/// metadata is inspected.  Use this for large files where only aggregate
/// statistics are needed.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened or [`Error::Xml`]
/// for XML structure errors.
pub fn stats_from_file(path: &Path) -> Result<MzmlStats> {
    let file = File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let buf_reader = BufReader::new(file);
    let mut reader = XmlReader::new(buf_reader);
    reader.set_trim_text(true);

    let mut ms1 = 0_usize;
    let mut ms2 = 0_usize;
    let mut min_rt: Option<f64> = None;
    let mut max_rt: Option<f64> = None;
    let mut min_mz: Option<f64> = None;
    let mut max_mz: Option<f64> = None;
    let mut total_peaks = 0_usize;
    let mut num_spectra = 0_usize;

    let mut in_spectrum = false;
    let mut current_ms_level: u32 = 1;
    let mut current_array_len: usize = 0;

    loop {
        match reader.next_event()? {
            XmlEvent::StartElement {
                ref name,
                ref attrs,
            } => match name.as_str() {
                "spectrum" => {
                    in_spectrum = true;
                    current_ms_level = 1;
                    current_array_len = attrs
                        .iter()
                        .find(|(k, _)| k == "defaultArrayLength")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(0);
                }
                "cvParam" if in_spectrum => {
                    let accession = attrs
                        .iter()
                        .find(|(k, _)| k == "accession")
                        .map_or("", |(_, v)| v.as_str());
                    let cv_value = attrs
                        .iter()
                        .find(|(k, _)| k == "value")
                        .map_or("", |(_, v)| v.as_str());
                    match accession {
                        "MS:1000511" => {
                            current_ms_level = cv_value.parse().unwrap_or(1);
                        }
                        "MS:1000016" => {
                            if let Ok(rt) = cv_value.parse::<f64>() {
                                min_rt = Some(min_rt.map_or(rt, |v: f64| v.min(rt)));
                                max_rt = Some(max_rt.map_or(rt, |v: f64| v.max(rt)));
                            }
                        }
                        "MS:1000528" => {
                            if let Ok(lo) = cv_value.parse::<f64>() {
                                if lo > 0.0 {
                                    min_mz = Some(min_mz.map_or(lo, |v: f64| v.min(lo)));
                                }
                            }
                        }
                        "MS:1000527" => {
                            if let Ok(hi) = cv_value.parse::<f64>() {
                                if hi > 0.0 {
                                    max_mz = Some(max_mz.map_or(hi, |v: f64| v.max(hi)));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            XmlEvent::EndElement { ref name } if name == "spectrum" => {
                num_spectra += 1;
                match current_ms_level {
                    1 => ms1 += 1,
                    2 => ms2 += 1,
                    _ => {}
                }
                total_peaks += current_array_len;
                in_spectrum = false;
            }
            XmlEvent::Eof => break,
            _ => {}
        }
    }

    Ok(MzmlStats {
        num_spectra,
        num_ms1: ms1,
        num_ms2: ms2,
        min_rt,
        max_rt,
        min_mz,
        max_mz,
        total_peaks,
    })
}

/// Compute summary statistics from parsed spectra.
#[must_use]
pub fn compute_stats(spectra: &[MzmlSpectrum]) -> MzmlStats {
    let mut ms1 = 0_usize;
    let mut ms2 = 0_usize;
    let mut min_rt: Option<f64> = None;
    let mut max_rt: Option<f64> = None;
    let mut min_mz: Option<f64> = None;
    let mut max_mz: Option<f64> = None;
    let mut total_peaks = 0_usize;

    for s in spectra {
        match s.ms_level {
            1 => ms1 += 1,
            2 => ms2 += 1,
            _ => {}
        }

        min_rt = Some(min_rt.map_or(s.rt_minutes, |v: f64| v.min(s.rt_minutes)));
        max_rt = Some(max_rt.map_or(s.rt_minutes, |v: f64| v.max(s.rt_minutes)));

        if s.lowest_mz > 0.0 {
            min_mz = Some(min_mz.map_or(s.lowest_mz, |v: f64| v.min(s.lowest_mz)));
        }
        if s.highest_mz > 0.0 {
            max_mz = Some(max_mz.map_or(s.highest_mz, |v: f64| v.max(s.highest_mz)));
        }

        total_peaks += s.mz_array.len();
    }

    MzmlStats {
        num_spectra: spectra.len(),
        num_ms1: ms1,
        num_ms2: ms2,
        min_rt,
        max_rt,
        min_mz,
        max_mz,
        total_peaks,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal valid mzML document with one MS1 spectrum.
    fn minimal_mzml(mz_b64: &str, int_b64: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
<mzML>
<run>
<spectrumList count="1">
  <spectrum index="0" defaultArrayLength="3">
    <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
    <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="1.5" unitName="minute"/>
    <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="3000"/>
    <cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="200.0"/>
    <cvParam cvRef="MS" accession="MS:1000505" name="base peak intensity" value="1500"/>
    <cvParam cvRef="MS" accession="MS:1000528" name="lowest observed m/z" value="100.0"/>
    <cvParam cvRef="MS" accession="MS:1000527" name="highest observed m/z" value="300.0"/>
    <binaryDataArrayList count="2">
      <binaryDataArray>
        <cvParam accession="MS:1000514" name="m/z array"/>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <binary>{mz_b64}</binary>
      </binaryDataArray>
      <binaryDataArray>
        <cvParam accession="MS:1000515" name="intensity array"/>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <binary>{int_b64}</binary>
      </binaryDataArray>
    </binaryDataArrayList>
  </spectrum>
</spectrumList>
</run>
</mzML>
</indexedmzML>"#
        )
    }

    #[test]
    fn test_parse_mzml_minimal() {
        let mz_vals = [100.0_f64, 200.0, 300.0];
        let int_vals = [500.0_f64, 1500.0, 1000.0];
        let mz_bytes: Vec<u8> = mz_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let int_bytes: Vec<u8> = int_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mz_b64 = crate::encoding::base64_encode(&mz_bytes);
        let int_b64 = crate::encoding::base64_encode(&int_bytes);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(minimal_mzml(&mz_b64, &int_b64).as_bytes())
            .unwrap();

        let spectra = parse_mzml(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].ms_level, 1);
        assert_eq!(spectra[0].mz_array.len(), 3);
        assert!((spectra[0].mz_array[0] - 100.0).abs() < f64::EPSILON);
        assert!((spectra[0].mz_array[2] - 300.0).abs() < f64::EPSILON);
        assert!((spectra[0].rt_minutes - 1.5).abs() < f64::EPSILON);
        assert_eq!(spectra[0].intensity_array.len(), 3);
    }

    #[test]
    fn test_parse_mzml_stats_integration() {
        let mz_vals = [100.0_f64, 200.0, 300.0];
        let int_vals = [500.0_f64, 1500.0, 1000.0];
        let mz_bytes: Vec<u8> = mz_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let int_bytes: Vec<u8> = int_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mz_b64 = crate::encoding::base64_encode(&mz_bytes);
        let int_b64 = crate::encoding::base64_encode(&int_bytes);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stats.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(minimal_mzml(&mz_b64, &int_b64).as_bytes())
            .unwrap();

        let spectra = parse_mzml(&path).unwrap();
        let stats = compute_stats(&spectra);
        assert_eq!(stats.num_spectra, 1);
        assert_eq!(stats.num_ms1, 1);
        assert_eq!(stats.total_peaks, 3);
    }

    #[test]
    fn test_parse_mzml_nonexistent() {
        let path = std::env::temp_dir().join("nonexistent_wetspring_9f8a2.mzML");
        let result = parse_mzml(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_mzml_empty_spectrumlist() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
<mzML><run><spectrumList count="0">
</spectrumList></run></mzML></indexedmzML>"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(xml.as_bytes()).unwrap();

        let spectra = parse_mzml(&path).unwrap();
        assert!(spectra.is_empty());
    }

    #[test]
    fn compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.num_spectra, 0);
        assert!(stats.min_rt.is_none());
        assert!(stats.max_mz.is_none());
    }

    #[test]
    fn compute_stats_single_ms1() {
        let spectrum = MzmlSpectrum {
            index: 0,
            ms_level: 1,
            rt_minutes: 5.0,
            tic: 1000.0,
            base_peak_mz: 500.0,
            base_peak_intensity: 900.0,
            lowest_mz: 100.0,
            highest_mz: 900.0,
            mz_array: vec![100.0, 500.0, 900.0],
            intensity_array: vec![100.0, 900.0, 200.0],
        };
        let stats = compute_stats(&[spectrum]);
        assert_eq!(stats.num_spectra, 1);
        assert_eq!(stats.num_ms1, 1);
        assert_eq!(stats.num_ms2, 0);
        assert_eq!(stats.total_peaks, 3);
        assert!((stats.min_rt.unwrap() - 5.0).abs() < f64::EPSILON);
        assert!((stats.min_mz.unwrap() - 100.0).abs() < f64::EPSILON);
        assert!((stats.max_mz.unwrap() - 900.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_stats_unknown_ms_level() {
        let s = MzmlSpectrum {
            index: 0,
            ms_level: 3,
            rt_minutes: 1.0,
            tic: 0.0,
            base_peak_mz: 0.0,
            base_peak_intensity: 0.0,
            lowest_mz: 0.0,
            highest_mz: 0.0,
            mz_array: vec![],
            intensity_array: vec![],
        };
        let stats = compute_stats(&[s]);
        assert_eq!(stats.num_spectra, 1);
        assert_eq!(stats.num_ms1, 0);
        assert_eq!(stats.num_ms2, 0);
    }

    #[test]
    fn compute_stats_zero_mz_range_skipped() {
        let s = MzmlSpectrum {
            index: 0,
            ms_level: 1,
            rt_minutes: 1.0,
            tic: 0.0,
            base_peak_mz: 0.0,
            base_peak_intensity: 0.0,
            lowest_mz: 0.0,
            highest_mz: 0.0,
            mz_array: vec![],
            intensity_array: vec![],
        };
        let stats = compute_stats(&[s]);
        assert!(stats.min_mz.is_none());
        assert!(stats.max_mz.is_none());
    }

    #[test]
    fn compute_stats_mixed_ms_levels() {
        let ms1 = MzmlSpectrum {
            index: 0,
            ms_level: 1,
            rt_minutes: 1.0,
            tic: 0.0,
            base_peak_mz: 0.0,
            base_peak_intensity: 0.0,
            lowest_mz: 80.0,
            highest_mz: 1000.0,
            mz_array: vec![100.0, 200.0],
            intensity_array: vec![50.0, 60.0],
        };
        let ms2 = MzmlSpectrum {
            index: 1,
            ms_level: 2,
            rt_minutes: 2.0,
            tic: 0.0,
            base_peak_mz: 0.0,
            base_peak_intensity: 0.0,
            lowest_mz: 100.0,
            highest_mz: 500.0,
            mz_array: vec![150.0],
            intensity_array: vec![70.0],
        };
        let stats = compute_stats(&[ms1, ms2]);
        assert_eq!(stats.num_ms1, 1);
        assert_eq!(stats.num_ms2, 1);
        assert_eq!(stats.total_peaks, 3);
        assert!((stats.min_mz.unwrap() - 80.0).abs() < f64::EPSILON);
        assert!((stats.max_mz.unwrap() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mzml_iter_matches_parse() {
        let mz_vals = [100.0_f64, 200.0, 300.0];
        let int_vals = [500.0_f64, 1500.0, 1000.0];
        let mz_bytes: Vec<u8> = mz_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let int_bytes: Vec<u8> = int_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mz_b64 = crate::encoding::base64_encode(&mz_bytes);
        let int_b64 = crate::encoding::base64_encode(&int_bytes);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iter.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(minimal_mzml(&mz_b64, &int_b64).as_bytes())
            .unwrap();

        let buffered = parse_mzml(&path).unwrap();
        let streamed: Vec<MzmlSpectrum> = MzmlIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(buffered.len(), streamed.len());
        for (b, s) in buffered.iter().zip(streamed.iter()) {
            assert_eq!(b.index, s.index);
            assert_eq!(b.ms_level, s.ms_level);
            assert!((b.rt_minutes - s.rt_minutes).abs() < f64::EPSILON);
            assert_eq!(b.mz_array.len(), s.mz_array.len());
            for (bm, sm) in b.mz_array.iter().zip(s.mz_array.iter()) {
                assert!((bm - sm).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn mzml_iter_empty_spectrumlist() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
<mzML><run><spectrumList count="0">
</spectrumList></run></mzML></indexedmzML>"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_iter.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(xml.as_bytes()).unwrap();

        let spectra: Vec<MzmlSpectrum> = MzmlIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(spectra.is_empty());
    }

    /// Build mzML with two spectra: one MS1, one MS2.
    fn mixed_ms1_ms2_mzml(mz1_b64: &str, int1_b64: &str, mz2_b64: &str, int2_b64: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
<mzML>
<run>
<spectrumList count="2">
  <spectrum index="0" defaultArrayLength="3">
    <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
    <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="1.0" unitName="minute"/>
    <cvParam cvRef="MS" accession="MS:1000528" name="lowest observed m/z" value="80.0"/>
    <cvParam cvRef="MS" accession="MS:1000527" name="highest observed m/z" value="500.0"/>
    <binaryDataArrayList count="2">
      <binaryDataArray>
        <cvParam accession="MS:1000514" name="m/z array"/>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <binary>{mz1_b64}</binary>
      </binaryDataArray>
      <binaryDataArray>
        <cvParam accession="MS:1000515" name="intensity array"/>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <binary>{int1_b64}</binary>
      </binaryDataArray>
    </binaryDataArrayList>
  </spectrum>
  <spectrum index="1" defaultArrayLength="2">
    <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2"/>
    <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="2.5" unitName="minute"/>
    <cvParam cvRef="MS" accession="MS:1000528" name="lowest observed m/z" value="100.0"/>
    <cvParam cvRef="MS" accession="MS:1000527" name="highest observed m/z" value="300.0"/>
    <binaryDataArrayList count="2">
      <binaryDataArray>
        <cvParam accession="MS:1000514" name="m/z array"/>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <binary>{mz2_b64}</binary>
      </binaryDataArray>
      <binaryDataArray>
        <cvParam accession="MS:1000515" name="intensity array"/>
        <cvParam accession="MS:1000523" name="64-bit float"/>
        <binary>{int2_b64}</binary>
      </binaryDataArray>
    </binaryDataArrayList>
  </spectrum>
</spectrumList>
</run>
</mzML>
</indexedmzML>"#
        )
    }

    #[test]
    fn stats_from_file_basic() {
        let mz_vals = [100.0_f64, 200.0, 300.0];
        let int_vals = [500.0_f64, 1500.0, 1000.0];
        let mz_bytes: Vec<u8> = mz_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let int_bytes: Vec<u8> = int_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mz_b64 = crate::encoding::base64_encode(&mz_bytes);
        let int_b64 = crate::encoding::base64_encode(&int_bytes);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stats.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(minimal_mzml(&mz_b64, &int_b64).as_bytes())
            .unwrap();

        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_spectra, 1);
        assert_eq!(stats.num_ms1, 1);
        assert_eq!(stats.num_ms2, 0);
        assert_eq!(stats.total_peaks, 3);
        assert!((stats.min_rt.unwrap() - 1.5).abs() < f64::EPSILON);
        assert!((stats.min_mz.unwrap() - 100.0).abs() < f64::EPSILON);
        assert!((stats.max_mz.unwrap() - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_from_file_mixed_ms1_ms2() {
        let mz1 = [80.0_f64, 200.0, 500.0];
        let int1 = [100.0_f64, 500.0, 200.0];
        let mz2 = [100.0_f64, 300.0];
        let int2 = [150.0_f64, 250.0];
        let mz1_b64 = crate::encoding::base64_encode(
            &mz1.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let int1_b64 = crate::encoding::base64_encode(
            &int1
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let mz2_b64 = crate::encoding::base64_encode(
            &mz2.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let int2_b64 = crate::encoding::base64_encode(
            &int2
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(mixed_ms1_ms2_mzml(&mz1_b64, &int1_b64, &mz2_b64, &int2_b64).as_bytes())
            .unwrap();

        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_spectra, 2);
        assert_eq!(stats.num_ms1, 1);
        assert_eq!(stats.num_ms2, 1);
        assert_eq!(stats.total_peaks, 5);
        assert!((stats.min_rt.unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((stats.max_rt.unwrap() - 2.5).abs() < f64::EPSILON);
        assert!((stats.min_mz.unwrap() - 80.0).abs() < f64::EPSILON);
        assert!((stats.max_mz.unwrap() - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_from_file_nonexistent() {
        let path = std::env::temp_dir().join("nonexistent_wetspring_mzml_stats_9f8a2.mzML");
        let result = stats_from_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn mzml_iter_multiple_spectra() {
        let mz1 = [100.0_f64, 200.0];
        let int1 = [500.0_f64, 1500.0];
        let mz2 = [150.0_f64, 250.0, 350.0];
        let int2 = [100.0_f64, 200.0, 300.0];
        let mz1_b64 = crate::encoding::base64_encode(
            &mz1.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let int1_b64 = crate::encoding::base64_encode(
            &int1
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let mz2_b64 = crate::encoding::base64_encode(
            &mz2.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let int2_b64 = crate::encoding::base64_encode(
            &int2
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi_iter.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(mixed_ms1_ms2_mzml(&mz1_b64, &int1_b64, &mz2_b64, &int2_b64).as_bytes())
            .unwrap();

        let streamed: Vec<MzmlSpectrum> = MzmlIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(streamed.len(), 2);
        assert_eq!(streamed[0].ms_level, 1);
        assert_eq!(streamed[0].mz_array.len(), 2);
        assert_eq!(streamed[1].ms_level, 2);
        assert_eq!(streamed[1].mz_array.len(), 3);
    }

    #[test]
    fn mzml_iter_zlib_compressed() {
        use flate2::write::ZlibEncoder;
        use flate2::Compression;

        let mz_vals = [100.0_f64, 200.0, 300.0];
        let int_vals = [500.0_f64, 1500.0, 1000.0];
        let mz_bytes: Vec<u8> = mz_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let int_bytes: Vec<u8> = int_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compress = |data: &[u8]| -> Vec<u8> {
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
            std::io::Write::write_all(&mut enc, data).unwrap();
            enc.finish().unwrap()
        };
        let mz_b64 = crate::encoding::base64_encode(&compress(&mz_bytes));
        let int_b64 = crate::encoding::base64_encode(&compress(&int_bytes));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zlib_iter.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(
            decode::custom_binary_mzml(&mz_b64, &int_b64, true, true, true, true).as_bytes(),
        )
        .unwrap();

        let spectra: Vec<MzmlSpectrum> = MzmlIter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].mz_array.len(), 3);
        assert!((spectra[0].mz_array[1] - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mzml_iter_xml_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("malformed.mzML");
        let mut f = std::fs::File::create(&path).unwrap();
        // Unclosed element - XML parse error
        f.write_all(b"<?xml version=\"1.0\"?><indexedmzML><mzML><run><spectrumList><spectrum")
            .unwrap();

        let results: Vec<Result<MzmlSpectrum>> = MzmlIter::open(&path).unwrap().collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }
}
