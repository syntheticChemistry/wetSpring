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

mod parser;
mod types;

pub use parser::{for_each_scan, parse_mzxml, MzxmlIter};
pub use types::MzxmlSpectrum;

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::parser::{decode_peaks, parse_retention_time};
    use super::parser::{for_each_scan, parse_mzxml, MzxmlIter};
    use super::types::{ByteOrder, Compression, ZlibBuffer};

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
        let mut zb = ZlibBuffer::default();
        let (mz, int) =
            decode_peaks(&b64, 32, ByteOrder::Network, Compression::None, &mut zb).unwrap();
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
        let mut zb = ZlibBuffer::default();
        let (mz, int) =
            decode_peaks(&b64, 64, ByteOrder::Network, Compression::None, &mut zb).unwrap();
        assert_eq!(mz.len(), 2);
        assert!((mz[0] - 100.0).abs() < f64::EPSILON);
        assert!((int[1] - 1500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn decode_empty_peaks() {
        let mut zb = ZlibBuffer::default();
        let (mz, int) =
            decode_peaks("", 32, ByteOrder::Network, Compression::None, &mut zb).unwrap();
        assert!(mz.is_empty());
        assert!(int.is_empty());
    }

    #[test]
    fn decode_peaks_wrong_length() {
        let b64 = crate::encoding::base64_encode(&[1, 2, 3]);
        let mut zb = ZlibBuffer::default();
        let result = decode_peaks(&b64, 32, ByteOrder::Network, Compression::None, &mut zb);
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
