// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzML parser tests.
#![expect(clippy::unwrap_used)]

use super::*;

fn collect_mzml(path: &std::path::Path) -> crate::error::Result<Vec<MzmlSpectrum>> {
    MzmlIter::open(path)?.collect()
}
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

    let spectra = collect_mzml(&path).unwrap();
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

    let spectra = collect_mzml(&path).unwrap();
    let stats = compute_stats(&spectra);
    assert_eq!(stats.num_spectra, 1);
    assert_eq!(stats.num_ms1, 1);
    assert_eq!(stats.total_peaks, 3);
}

#[test]
fn test_parse_mzml_nonexistent() {
    let path = std::env::temp_dir().join("nonexistent_wetspring_9f8a2.mzML");
    let result = collect_mzml(&path);
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

    let spectra = collect_mzml(&path).unwrap();
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

    let buffered = collect_mzml(&path).unwrap();
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
    use flate2::Compression;
    use flate2::write::ZlibEncoder;

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
    f.write_all(decode::custom_binary_mzml(&mz_b64, &int_b64, true, true, true, true).as_bytes())
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

#[test]
fn mzml_nonexistent_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nonexistent.mzML");
    let result = collect_mzml(&path);
    assert!(result.is_err());
}

#[test]
fn mzml_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.mzML");
    std::fs::File::create(&path).unwrap();
    let spectra = collect_mzml(&path).unwrap();
    assert!(spectra.is_empty());
}

#[test]
fn mzml_stats_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.mzML");
    std::fs::File::create(&path).unwrap();
    let stats = stats_from_file(&path).unwrap();
    assert_eq!(stats.num_spectra, 0);
    assert_eq!(stats.total_peaks, 0);
}
