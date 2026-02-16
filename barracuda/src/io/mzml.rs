// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzML parser — vendor-neutral mass spectrometry data I/O.
//!
//! Parses mzML XML files **streaming from disk** using a sovereign
//! pull parser (internal `xml` module) with a `BufReader`.  Never loads
//! the entire file into memory.  Decodes base64-encoded and
//! optionally zlib-compressed m/z + intensity arrays.

use crate::encoding::base64_decode;
use crate::error::{Error, Result};
use crate::io::xml::{XmlEvent, XmlReader};
use flate2::read::ZlibDecoder;
use std::fs::File;
use std::io::{BufReader, Read};
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

// ── Internal parser state ───────────────────────────────────────

/// Builder that accumulates fields while parsing a single `<spectrum>`.
struct SpectrumBuilder {
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

impl SpectrumBuilder {
    fn new(index: usize) -> Self {
        Self {
            index,
            ms_level: 1,
            rt: 0.0,
            tic: 0.0,
            base_peak_mz: 0.0,
            base_peak_intensity: 0.0,
            lowest_mz: 0.0,
            highest_mz: 0.0,
            mz_array: Vec::new(),
            intensity_array: Vec::new(),
        }
    }

    fn apply_cv_param(&mut self, accession: &str, value: &str) {
        match accession {
            "MS:1000511" => self.ms_level = value.parse().unwrap_or(1),
            "MS:1000016" => self.rt = value.parse().unwrap_or(0.0),
            "MS:1000285" => self.tic = value.parse().unwrap_or(0.0),
            "MS:1000504" => self.base_peak_mz = value.parse().unwrap_or(0.0),
            "MS:1000505" => self.base_peak_intensity = value.parse().unwrap_or(0.0),
            "MS:1000528" => self.lowest_mz = value.parse().unwrap_or(0.0),
            "MS:1000527" => self.highest_mz = value.parse().unwrap_or(0.0),
            _ => {}
        }
    }

    fn build(self) -> MzmlSpectrum {
        MzmlSpectrum {
            index: self.index,
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

/// Tracks encoding properties for a `<binaryDataArray>`.
#[allow(clippy::struct_excessive_bools)] // maps directly to mzML cvParam flags
struct BinaryArrayState {
    is_mz: bool,
    is_intensity: bool,
    is_zlib: bool,
    is_64bit: bool,
    text: String,
}

impl BinaryArrayState {
    fn new() -> Self {
        Self {
            is_mz: false,
            is_intensity: false,
            is_zlib: false,
            is_64bit: true,
            text: String::new(),
        }
    }

    fn reset(&mut self) {
        self.is_mz = false;
        self.is_intensity = false;
        self.is_zlib = false;
        self.is_64bit = true;
        self.text.clear();
    }

    fn apply_cv_param(&mut self, accession: &str) {
        match accession {
            "MS:1000514" => self.is_mz = true,
            "MS:1000515" => self.is_intensity = true,
            "MS:1000574" => self.is_zlib = true,
            "MS:1000576" => self.is_zlib = false,
            "MS:1000523" => self.is_64bit = true,
            "MS:1000521" => self.is_64bit = false,
            _ => {}
        }
    }

    fn decode_into(&self, builder: &mut SpectrumBuilder) -> Result<()> {
        if self.text.is_empty() {
            return Ok(());
        }
        let arr = decode_binary_array(&self.text, self.is_zlib, self.is_64bit)?;
        if self.is_mz {
            builder.mz_array = arr;
        } else if self.is_intensity {
            builder.intensity_array = arr;
        }
        Ok(())
    }
}

// ── Binary array decoder ────────────────────────────────────────

/// Decode a base64 + optional zlib-compressed array of f64 values.
fn decode_binary_array(encoded: &str, is_zlib: bool, is_64bit: bool) -> Result<Vec<f64>> {
    let bytes = base64_decode(encoded.trim()).map_err(Error::Base64)?;

    let decompressed = if is_zlib {
        let mut decoder = ZlibDecoder::new(&bytes[..]);
        let mut buf = Vec::new();
        decoder
            .read_to_end(&mut buf)
            .map_err(|e| Error::Zlib(format!("{e}")))?;
        buf
    } else {
        bytes
    };

    if is_64bit {
        if decompressed.len() % 8 != 0 {
            return Err(Error::BinaryFormat(format!(
                "f64 array length {} not divisible by 8",
                decompressed.len()
            )));
        }
        Ok(decompressed
            .chunks_exact(8)
            .map(|chunk| {
                let mut arr = [0_u8; 8];
                arr.copy_from_slice(chunk);
                f64::from_le_bytes(arr)
            })
            .collect())
    } else {
        if decompressed.len() % 4 != 0 {
            return Err(Error::BinaryFormat(format!(
                "f32 array length {} not divisible by 4",
                decompressed.len()
            )));
        }
        Ok(decompressed
            .chunks_exact(4)
            .map(|chunk| {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(chunk);
                f64::from(f32::from_le_bytes(arr))
            })
            .collect())
    }
}

// ── Public API ──────────────────────────────────────────────────

/// Parse an mzML file and return all spectra, **streaming from disk**.
///
/// Uses `BufReader` — the full XML is never held in memory.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, [`Error::Xml`]
/// for XML structure errors, or [`Error::Base64`] / [`Error::Zlib`]
/// for binary array decoding failures.
pub fn parse_mzml(path: &Path) -> Result<Vec<MzmlSpectrum>> {
    let file = File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let buf_reader = BufReader::new(file);
    let mut reader = XmlReader::new(buf_reader);
    reader.set_trim_text(true);

    let mut spectra = Vec::new();
    let mut builder: Option<SpectrumBuilder> = None;
    let mut binary_state = BinaryArrayState::new();
    let mut in_binary_data_array = false;
    let mut in_binary = false;

    loop {
        match reader.next_event()? {
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
                    if let Some(b) = builder.take() {
                        spectra.push(b.build());
                    }
                }
                "binary" => in_binary = false,
                "binaryDataArray" => {
                    if let Some(ref mut b) = builder {
                        if let Err(e) = binary_state.decode_into(b) {
                            eprintln!("Warning: binary decode failed: {e}");
                        }
                    }
                    in_binary_data_array = false;
                }
                _ => {}
            },
            XmlEvent::Eof => break,
        }
    }

    Ok(spectra)
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
    fn decode_f64_uncompressed() {
        let values = [100.0_f64, 200.0, 300.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoded = crate::encoding::base64_encode(&bytes);
        let decoded = decode_binary_array(&encoded, false, true).unwrap();
        assert_eq!(decoded.len(), 3);
        for (a, &b) in decoded.iter().zip(values.iter()) {
            assert!((*a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn decode_f32_uncompressed() {
        let values = [1.5_f32, 2.5, 3.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoded = crate::encoding::base64_encode(&bytes);
        let decoded = decode_binary_array(&encoded, false, false).unwrap();
        assert_eq!(decoded.len(), 3);
        for (a, v) in decoded.iter().zip(values.iter()) {
            assert!((*a - f64::from(*v)).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn decode_bad_base64() {
        let result = decode_binary_array("!!!invalid!!!", false, true);
        assert!(result.is_err());
    }

    #[test]
    fn decode_wrong_length() {
        let encoded = crate::encoding::base64_encode(&[1, 2, 3]);
        let result = decode_binary_array(&encoded, false, true);
        assert!(result.is_err());
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

    /// Build an mzML with configurable compression and precision per array.
    #[allow(clippy::fn_params_excessive_bools)] // maps directly to mzML encoding flags
    fn custom_binary_mzml(
        mz_b64: &str,
        int_b64: &str,
        mz_zlib: bool,
        mz_64bit: bool,
        int_zlib: bool,
        int_64bit: bool,
    ) -> String {
        let mz_zlib_cv = if mz_zlib {
            r#"<cvParam accession="MS:1000574" name="zlib compression"/>"#
        } else {
            r#"<cvParam accession="MS:1000576" name="no compression"/>"#
        };
        let mz_prec_cv = if mz_64bit {
            r#"<cvParam accession="MS:1000523" name="64-bit float"/>"#
        } else {
            r#"<cvParam accession="MS:1000521" name="32-bit float"/>"#
        };
        let int_zlib_cv = if int_zlib {
            r#"<cvParam accession="MS:1000574" name="zlib compression"/>"#
        } else {
            r#"<cvParam accession="MS:1000576" name="no compression"/>"#
        };
        let int_prec_cv = if int_64bit {
            r#"<cvParam accession="MS:1000523" name="64-bit float"/>"#
        } else {
            r#"<cvParam accession="MS:1000521" name="32-bit float"/>"#
        };
        format!(
            r#"<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
<mzML>
<run>
<spectrumList count="1">
  <spectrum index="0" defaultArrayLength="3">
    <cvParam accession="MS:1000511" name="ms level" value="1"/>
    <binaryDataArrayList count="2">
      <binaryDataArray>
        <cvParam accession="MS:1000514" name="m/z array"/>
        {mz_zlib_cv}
        {mz_prec_cv}
        <binary>{mz_b64}</binary>
      </binaryDataArray>
      <binaryDataArray>
        <cvParam accession="MS:1000515" name="intensity array"/>
        {int_zlib_cv}
        {int_prec_cv}
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

    /// Write text to a temp mzML file and return the path.
    fn write_temp_mzml(dir: &tempfile::TempDir, name: &str, xml: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(xml.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parse_mzml_f32_arrays() {
        let mz_vals = [100.0_f32, 200.0, 300.0];
        let int_vals = [500.0_f32, 1500.0, 1000.0];
        let mz_bytes: Vec<u8> = mz_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let int_bytes: Vec<u8> = int_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mz_b64 = crate::encoding::base64_encode(&mz_bytes);
        let int_b64 = crate::encoding::base64_encode(&int_bytes);

        let dir = tempfile::tempdir().unwrap();
        let xml = custom_binary_mzml(&mz_b64, &int_b64, false, false, false, false);
        let path = write_temp_mzml(&dir, "f32.mzML", &xml);

        let spectra = parse_mzml(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].mz_array.len(), 3);
        assert!((spectra[0].mz_array[0] - 100.0).abs() < 0.01);
        assert!((spectra[0].mz_array[2] - 300.0).abs() < 0.01);
    }

    #[test]
    fn parse_mzml_zlib_f64_arrays() {
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
        let xml = custom_binary_mzml(&mz_b64, &int_b64, true, true, true, true);
        let path = write_temp_mzml(&dir, "zlib.mzML", &xml);

        let spectra = parse_mzml(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].mz_array.len(), 3);
        assert!((spectra[0].mz_array[1] - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn decode_f32_wrong_length() {
        // 3 bytes is not divisible by 4
        let encoded = crate::encoding::base64_encode(&[1, 2, 3]);
        let result = decode_binary_array(&encoded, false, false);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("f32 array length"));
    }

    #[test]
    fn decode_zlib_f64() {
        use flate2::write::ZlibEncoder;
        use flate2::Compression;

        let values = [42.0_f64, 99.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut enc, &raw).unwrap();
        let compressed = enc.finish().unwrap();
        let encoded = crate::encoding::base64_encode(&compressed);

        let decoded = decode_binary_array(&encoded, true, true).unwrap();
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0] - 42.0).abs() < f64::EPSILON);
        assert!((decoded[1] - 99.0).abs() < f64::EPSILON);
    }

    #[test]
    fn binary_state_empty_text_skips_decode() {
        let state = BinaryArrayState::new();
        assert!(state.text.is_empty());
        let mut builder = SpectrumBuilder::new(0);
        state.decode_into(&mut builder).unwrap();
        assert!(builder.mz_array.is_empty());
    }

    #[test]
    fn binary_state_unknown_cv_params_ignored() {
        let mut state = BinaryArrayState::new();
        state.apply_cv_param("MS:9999999");
        // Should not change any flags
        assert!(!state.is_mz);
        assert!(!state.is_intensity);
        assert!(!state.is_zlib);
        assert!(state.is_64bit);
    }

    #[test]
    fn spectrum_builder_unknown_cv_param_ignored() {
        let mut b = SpectrumBuilder::new(0);
        b.apply_cv_param("MS:9999999", "42");
        assert_eq!(b.ms_level, 1); // unchanged
        assert!((b.rt - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_mzml_empty_binary_element() {
        let dir = tempfile::tempdir().unwrap();
        let xml = custom_binary_mzml("", "", false, true, false, true);
        let path = write_temp_mzml(&dir, "emptybinary.mzML", &xml);
        let spectra = parse_mzml(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert!(spectra[0].mz_array.is_empty());
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
}
