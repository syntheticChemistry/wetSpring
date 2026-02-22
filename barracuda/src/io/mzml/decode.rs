// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binary array decoding for mzML spectra.
//!
//! Handles base64 + optional zlib decompression and f32/f64 parsing.

use super::MzmlSpectrum;
use crate::encoding::base64_decode;
use crate::error::{Error, Result};
use flate2::read::ZlibDecoder;
use std::io::Read;

/// Builder that accumulates fields while parsing a single `<spectrum>`.
pub struct SpectrumBuilder {
    pub index: usize,
    pub ms_level: u32,
    pub rt: f64,
    pub tic: f64,
    pub base_peak_mz: f64,
    pub base_peak_intensity: f64,
    pub lowest_mz: f64,
    pub highest_mz: f64,
    pub mz_array: Vec<f64>,
    pub intensity_array: Vec<f64>,
}

impl SpectrumBuilder {
    pub const fn new(index: usize) -> Self {
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

    pub fn apply_cv_param(&mut self, accession: &str, value: &str) {
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

    pub fn build(self) -> MzmlSpectrum {
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

/// Reusable buffer for binary array decoding, avoiding per-spectrum allocation
/// for the intermediate decompression step.
///
/// Use with [`BinaryArrayState::decode_into_with_buffer`] when streaming many
/// spectra to amortize allocations.
#[derive(Debug, Default)]
pub struct DecodeBuffer {
    decompressed: Vec<u8>,
}

impl DecodeBuffer {
    /// Create a new empty decode buffer.
    pub const fn new() -> Self {
        Self {
            decompressed: Vec::new(),
        }
    }

    /// Decode a base64 + optional zlib-compressed array, reusing the internal buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Base64`] for invalid base64, [`Error::Zlib`] for decompression
    /// failures, or [`Error::BinaryFormat`] for length mismatches.
    pub fn decode(&mut self, encoded: &str, is_zlib: bool, is_64bit: bool) -> Result<Vec<f64>> {
        let bytes = base64_decode(encoded.trim()).map_err(Error::Base64)?;

        let decompressed: &[u8] = if is_zlib {
            self.decompressed.clear();
            let mut decoder = ZlibDecoder::new(&bytes[..]);
            decoder
                .read_to_end(&mut self.decompressed)
                .map_err(|e| Error::Zlib(format!("{e}")))?;
            &self.decompressed[..]
        } else {
            &bytes[..]
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
}

/// Tracks encoding properties for a `<binaryDataArray>`.
#[allow(clippy::struct_excessive_bools)] // maps directly to mzML cvParam flags
pub struct BinaryArrayState {
    pub is_mz: bool,
    pub is_intensity: bool,
    pub is_zlib: bool,
    pub is_64bit: bool,
    pub text: String,
}

impl BinaryArrayState {
    pub const fn new() -> Self {
        Self {
            is_mz: false,
            is_intensity: false,
            is_zlib: false,
            is_64bit: true,
            text: String::new(),
        }
    }

    pub fn reset(&mut self) {
        self.is_mz = false;
        self.is_intensity = false;
        self.is_zlib = false;
        self.is_64bit = true;
        self.text.clear();
    }

    pub fn apply_cv_param(&mut self, accession: &str) {
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

    /// Decode the binary array into the spectrum builder.
    ///
    /// For backward compatibility, allocates per call. Prefer
    /// [`decode_into_with_buffer`](Self::decode_into_with_buffer) when streaming many spectra.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Base64`], [`Error::Zlib`], or [`Error::BinaryFormat`] on decode failure.
    #[allow(dead_code)] // public API, prefer decode_into_with_buffer in streaming paths
    pub fn decode_into(&self, builder: &mut SpectrumBuilder) -> Result<()> {
        self.decode_into_with_buffer(builder, None)
    }

    /// Decode the binary array into the spectrum builder, optionally reusing a buffer.
    ///
    /// When `buffer` is `Some`, reuses the buffer for zlib decompression to avoid
    /// per-spectrum allocation. Use with [`DecodeBuffer`] when streaming.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Base64`], [`Error::Zlib`], or [`Error::BinaryFormat`] on decode failure.
    pub fn decode_into_with_buffer(
        &self,
        builder: &mut SpectrumBuilder,
        buffer: Option<&mut DecodeBuffer>,
    ) -> Result<()> {
        if self.text.is_empty() {
            return Ok(());
        }
        let arr = match buffer {
            Some(buf) => buf.decode(&self.text, self.is_zlib, self.is_64bit)?,
            None => decode_binary_array(&self.text, self.is_zlib, self.is_64bit)?,
        };
        if self.is_mz {
            builder.mz_array = arr;
        } else if self.is_intensity {
            builder.intensity_array = arr;
        }
        Ok(())
    }
}

/// Decode a base64 + optional zlib-compressed array of f64 values.
///
/// For streaming use, prefer [`DecodeBuffer::decode`] to reuse the decompression buffer.
///
/// # Errors
///
/// Returns [`Error::Base64`] for invalid base64, [`Error::Zlib`] for decompression
/// failures, or [`Error::BinaryFormat`] for length mismatches.
pub fn decode_binary_array(encoded: &str, is_zlib: bool, is_64bit: bool) -> Result<Vec<f64>> {
    let mut buffer = DecodeBuffer::new();
    buffer.decode(encoded, is_zlib, is_64bit)
}

/// Build an mzML with configurable compression and precision per array.
/// Test helper used by both decode and mod tests.
#[cfg(test)]
#[allow(clippy::fn_params_excessive_bools)] // maps directly to mzML encoding flags
pub fn custom_binary_mzml(
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
#[cfg(test)]
pub fn write_temp_mzml(dir: &tempfile::TempDir, name: &str, xml: &str) -> std::path::PathBuf {
    use std::io::Write;
    let path = dir.path().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(xml.as_bytes()).unwrap();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn spectrum_builder_parse_fallbacks() {
        let mut b = SpectrumBuilder::new(0);
        b.apply_cv_param("MS:1000511", "invalid"); // ms_level parse fails -> 1
        b.apply_cv_param("MS:1000016", "x"); // rt parse fails -> 0
        b.apply_cv_param("MS:1000285", "x"); // tic parse fails -> 0
        b.apply_cv_param("MS:1000504", "x"); // base_peak_mz fails -> 0
        b.apply_cv_param("MS:1000505", "x"); // base_peak_intensity fails -> 0
        b.apply_cv_param("MS:1000528", "x"); // lowest_mz fails -> 0
        b.apply_cv_param("MS:1000527", "x"); // highest_mz fails -> 0
        let spec = b.build();
        assert_eq!(spec.ms_level, 1);
        assert!((spec.rt_minutes - 0.0).abs() < f64::EPSILON);
        assert!((spec.tic - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_mzml_f32_arrays() {
        use super::super::parse_mzml;

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

        use super::super::parse_mzml;

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
    fn decode_buffer_matches_decode_binary_array() {
        let values = [100.0_f64, 200.0, 300.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoded = crate::encoding::base64_encode(&bytes);

        let mut buffer = DecodeBuffer::new();
        let decoded = buffer.decode(&encoded, false, true).unwrap();
        assert_eq!(decoded.len(), 3);
        for (a, &b) in decoded.iter().zip(values.iter()) {
            assert!((*a - b).abs() < f64::EPSILON);
        }

        // Compare with decode_binary_array
        let legacy = decode_binary_array(&encoded, false, true).unwrap();
        assert_eq!(decoded, legacy);
    }

    #[test]
    fn decode_buffer_reuse_zlib() {
        use flate2::write::ZlibEncoder;
        use flate2::Compression;

        let values1 = [42.0_f64, 99.0];
        let values2 = [1.5_f64, 2.5, 3.5];
        let raw1: Vec<u8> = values1.iter().flat_map(|v| v.to_le_bytes()).collect();
        let raw2: Vec<u8> = values2.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut enc, &raw1).unwrap();
        let compressed1 = enc.finish().unwrap();
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut enc, &raw2).unwrap();
        let compressed2 = enc.finish().unwrap();
        let encoded1 = crate::encoding::base64_encode(&compressed1);
        let encoded2 = crate::encoding::base64_encode(&compressed2);

        let mut buffer = DecodeBuffer::new();
        let decoded1 = buffer.decode(&encoded1, true, true).unwrap();
        let decoded2 = buffer.decode(&encoded2, true, true).unwrap();

        assert_eq!(decoded1.len(), 2);
        assert!((decoded1[0] - 42.0).abs() < f64::EPSILON);
        assert!((decoded1[1] - 99.0).abs() < f64::EPSILON);
        assert_eq!(decoded2.len(), 3);
        assert!((decoded2[0] - 1.5).abs() < f64::EPSILON);
        assert!((decoded2[1] - 2.5).abs() < f64::EPSILON);
        assert!((decoded2[2] - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn decode_into_with_buffer_matches_decode_into() {
        let values = [10.0_f64, 20.0, 30.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoded = crate::encoding::base64_encode(&bytes);

        let mut state = BinaryArrayState::new();
        state.text = encoded.clone();
        state.is_mz = true;
        state.is_zlib = false;
        state.is_64bit = true;

        let mut builder_without = SpectrumBuilder::new(0);
        state.decode_into(&mut builder_without).unwrap();

        let mut builder_with = SpectrumBuilder::new(0);
        state.text = encoded;
        let mut buffer = DecodeBuffer::new();
        state
            .decode_into_with_buffer(&mut builder_with, Some(&mut buffer))
            .unwrap();

        assert_eq!(builder_without.mz_array.len(), builder_with.mz_array.len());
        for (a, b) in builder_without
            .mz_array
            .iter()
            .zip(builder_with.mz_array.iter())
        {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn parse_mzml_empty_binary_element() {
        use super::super::parse_mzml;

        let dir = tempfile::tempdir().unwrap();
        let xml = custom_binary_mzml("", "", false, true, false, true);
        let path = write_temp_mzml(&dir, "emptybinary.mzML", &xml);
        let spectra = parse_mzml(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert!(spectra[0].mz_array.is_empty());
    }
}
