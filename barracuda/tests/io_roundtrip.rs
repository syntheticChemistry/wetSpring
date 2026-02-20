// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for I/O parsers: round-trip, determinism, edge cases.
//!
//! Each test creates synthetic data in a temporary directory,
//! parses it with our production code, and verifies correctness.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;
use wetspring_barracuda::encoding::base64_encode;
use wetspring_barracuda::io::{fastq, ms2, mzml};

use flate2::write::GzEncoder;
use flate2::Compression;

// ── MS2 round-trip ──────────────────────────────────────────────

fn create_ms2_file(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("roundtrip.ms2");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "H\tCreationDate\t2026-02-16").unwrap();
    writeln!(f, "H\tExtractor\twetSpring-test").unwrap();
    // Spectrum 1: 3 peaks
    writeln!(f, "S\t100\t100\t450.25").unwrap();
    writeln!(f, "I\tRTime\t3.50").unwrap();
    writeln!(f, "I\tTIC\t5000.0").unwrap();
    writeln!(f, "I\tBPI\t3000.0").unwrap();
    writeln!(f, "Z\t2\t899.49").unwrap();
    writeln!(f, "100.1\t500.0").unwrap();
    writeln!(f, "200.2\t300.0").unwrap();
    writeln!(f, "300.3\t100.0").unwrap();
    // Spectrum 2: 1 peak
    writeln!(f, "S\t200\t200\t550.50").unwrap();
    writeln!(f, "I\tRTime\t5.00").unwrap();
    writeln!(f, "Z\t3\t1650.47").unwrap();
    writeln!(f, "150.5\t800.0").unwrap();
    // Spectrum 3: empty (no peaks)
    writeln!(f, "S\t300\t300\t650.75").unwrap();
    writeln!(f, "I\tRTime\t7.25").unwrap();
    path
}

#[test]
fn ms2_roundtrip_structure() {
    let dir = TempDir::new().unwrap();
    let path = create_ms2_file(dir.path());
    let spectra = ms2::parse_ms2(&path).unwrap();

    assert_eq!(spectra.len(), 3, "should parse 3 spectra");
    assert_eq!(spectra[0].scan, 100);
    assert!((spectra[0].precursor_mz - 450.25).abs() < f64::EPSILON);
    assert!((spectra[0].rt_minutes - 3.5).abs() < f64::EPSILON);
    assert!((spectra[0].tic - 5000.0).abs() < f64::EPSILON);
    assert_eq!(spectra[0].charge, 2);
    assert_eq!(spectra[0].mz_array.len(), 3);
    assert_eq!(spectra[0].intensity_array.len(), 3);

    assert_eq!(spectra[1].scan, 200);
    assert_eq!(spectra[1].charge, 3);
    assert_eq!(spectra[1].mz_array.len(), 1);

    // Empty spectrum: no peaks
    assert_eq!(spectra[2].scan, 300);
    assert!(spectra[2].mz_array.is_empty());
}

#[test]
fn ms2_determinism() {
    let dir = TempDir::new().unwrap();
    let path = create_ms2_file(dir.path());

    let run1 = ms2::parse_ms2(&path).unwrap();
    let run2 = ms2::parse_ms2(&path).unwrap();

    assert_eq!(run1.len(), run2.len());
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.scan, b.scan);
        assert!((a.precursor_mz - b.precursor_mz).abs() < f64::EPSILON);
        assert_eq!(a.mz_array.len(), b.mz_array.len());
        for (ma, mb) in a.mz_array.iter().zip(b.mz_array.iter()) {
            assert!((ma - mb).abs() < f64::EPSILON);
        }
    }
}

#[test]
fn ms2_stats_correctness() {
    let dir = TempDir::new().unwrap();
    let path = create_ms2_file(dir.path());
    let spectra = ms2::parse_ms2(&path).unwrap();
    let stats = ms2::compute_stats(&spectra);

    assert_eq!(stats.num_spectra, 3);
    assert_eq!(stats.total_peaks, 4); // 3 + 1 + 0
    assert!((stats.min_precursor_mz.unwrap() - 450.25).abs() < f64::EPSILON);
    assert!((stats.max_precursor_mz.unwrap() - 650.75).abs() < f64::EPSILON);
    assert!((stats.min_rt.unwrap() - 3.5).abs() < f64::EPSILON);
    assert!((stats.max_rt.unwrap() - 7.25).abs() < f64::EPSILON);
}

#[test]
fn ms2_streaming_stats_match_collected() {
    let dir = TempDir::new().unwrap();
    let path = create_ms2_file(dir.path());

    let spectra = ms2::parse_ms2(&path).unwrap();
    let collected = ms2::compute_stats(&spectra);
    let streamed = ms2::stats_from_file(&path).unwrap();

    assert_eq!(collected.num_spectra, streamed.num_spectra);
    assert_eq!(collected.total_peaks, streamed.total_peaks);
    assert!(
        (collected.min_precursor_mz.unwrap() - streamed.min_precursor_mz.unwrap()).abs()
            < f64::EPSILON
    );
    assert!(
        (collected.max_precursor_mz.unwrap() - streamed.max_precursor_mz.unwrap()).abs()
            < f64::EPSILON
    );
    assert!((collected.min_rt.unwrap() - streamed.min_rt.unwrap()).abs() < f64::EPSILON);
    assert!((collected.max_rt.unwrap() - streamed.max_rt.unwrap()).abs() < f64::EPSILON);
}

// ── mzML synthetic file round-trip ───────────────────────────────

fn encode_f64_array(values: &[f64]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    base64_encode(&bytes)
}

fn create_mzml_file(dir: &Path) -> std::path::PathBuf {
    let mz_values = [100.0_f64, 250.0, 500.0, 750.0, 1000.0];
    let int_values = [100.0_f64, 500.0, 3000.0, 200.0, 50.0];
    let mz_b64 = encode_f64_array(&mz_values);
    let int_b64 = encode_f64_array(&int_values);

    let mz_values_ms2 = [150.0_f64, 300.0, 450.0];
    let int_values_ms2 = [200.0_f64, 800.0, 150.0];
    let mz_b64_ms2 = encode_f64_array(&mz_values_ms2);
    let int_b64_ms2 = encode_f64_array(&int_values_ms2);

    let path = dir.join("test.mzML");
    let mut f = File::create(&path).unwrap();
    write!(
        f,
        r#"<?xml version="1.0" encoding="utf-8"?>
<indexedmzML>
<mzML>
  <run>
    <spectrumList count="2" defaultDataProcessingRef="dp">
      <spectrum index="0" defaultArrayLength="5">
        <cvParam accession="MS:1000511" value="1"/>
        <cvParam accession="MS:1000016" value="3.50"/>
        <cvParam accession="MS:1000285" value="5000"/>
        <cvParam accession="MS:1000504" value="500.0"/>
        <cvParam accession="MS:1000505" value="3000"/>
        <cvParam accession="MS:1000528" value="100.0"/>
        <cvParam accession="MS:1000527" value="1000.0"/>
        <binaryDataArray encodedLength="0">
          <cvParam accession="MS:1000514"/>
          <cvParam accession="MS:1000576"/>
          <cvParam accession="MS:1000523"/>
          <binary>{mz_b64}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="0">
          <cvParam accession="MS:1000515"/>
          <cvParam accession="MS:1000576"/>
          <cvParam accession="MS:1000523"/>
          <binary>{int_b64}</binary>
        </binaryDataArray>
      </spectrum>
      <spectrum index="1" defaultArrayLength="3">
        <cvParam accession="MS:1000511" value="2"/>
        <cvParam accession="MS:1000016" value="4.00"/>
        <cvParam accession="MS:1000285" value="2000"/>
        <cvParam accession="MS:1000528" value="150.0"/>
        <cvParam accession="MS:1000527" value="450.0"/>
        <binaryDataArray encodedLength="0">
          <cvParam accession="MS:1000514"/>
          <cvParam accession="MS:1000576"/>
          <cvParam accession="MS:1000523"/>
          <binary>{mz_b64_ms2}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="0">
          <cvParam accession="MS:1000515"/>
          <cvParam accession="MS:1000576"/>
          <cvParam accession="MS:1000523"/>
          <binary>{int_b64_ms2}</binary>
        </binaryDataArray>
      </spectrum>
    </spectrumList>
  </run>
</mzML>
</indexedmzML>"#
    )
    .unwrap();
    path
}

#[test]
fn mzml_synthetic_file_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = create_mzml_file(dir.path());
    let spectra = mzml::parse_mzml(&path).unwrap();

    assert_eq!(spectra.len(), 2, "should parse 2 spectra");

    // Spectrum 0: MS1
    assert_eq!(spectra[0].index, 0);
    assert_eq!(spectra[0].ms_level, 1);
    assert!((spectra[0].rt_minutes - 3.5).abs() < f64::EPSILON);
    assert_eq!(spectra[0].mz_array.len(), 5);
    assert!((spectra[0].mz_array[0] - 100.0).abs() < f64::EPSILON);
    assert!((spectra[0].mz_array[4] - 1000.0).abs() < f64::EPSILON);
    assert_eq!(spectra[0].intensity_array.len(), 5);
    assert!((spectra[0].intensity_array[2] - 3000.0).abs() < f64::EPSILON);

    // Spectrum 1: MS2
    assert_eq!(spectra[1].index, 1);
    assert_eq!(spectra[1].ms_level, 2);
    assert_eq!(spectra[1].mz_array.len(), 3);
    assert!((spectra[1].mz_array[0] - 150.0).abs() < f64::EPSILON);
}

#[test]
fn mzml_determinism() {
    let dir = TempDir::new().unwrap();
    let path = create_mzml_file(dir.path());
    let run1 = mzml::parse_mzml(&path).unwrap();
    let run2 = mzml::parse_mzml(&path).unwrap();

    assert_eq!(run1.len(), run2.len());
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.index, b.index);
        assert_eq!(a.ms_level, b.ms_level);
        assert_eq!(a.mz_array, b.mz_array);
        assert_eq!(a.intensity_array, b.intensity_array);
    }
}

#[test]
fn mzml_stats_from_synthetic() {
    let dir = TempDir::new().unwrap();
    let path = create_mzml_file(dir.path());
    let spectra = mzml::parse_mzml(&path).unwrap();
    let stats = mzml::compute_stats(&spectra);

    assert_eq!(stats.num_spectra, 2);
    assert_eq!(stats.num_ms1, 1);
    assert_eq!(stats.num_ms2, 1);
    assert_eq!(stats.total_peaks, 8); // 5 + 3
    assert!((stats.min_rt.unwrap() - 3.5).abs() < f64::EPSILON);
    assert!((stats.max_rt.unwrap() - 4.0).abs() < f64::EPSILON);
}

#[test]
fn mzml_streaming_stats_match_collected() {
    let dir = TempDir::new().unwrap();
    let path = create_mzml_file(dir.path());

    let spectra = mzml::parse_mzml(&path).unwrap();
    let collected = mzml::compute_stats(&spectra);
    let streamed = mzml::stats_from_file(&path).unwrap();

    assert_eq!(collected.num_spectra, streamed.num_spectra);
    assert_eq!(collected.num_ms1, streamed.num_ms1);
    assert_eq!(collected.num_ms2, streamed.num_ms2);
    assert!((collected.min_rt.unwrap() - streamed.min_rt.unwrap()).abs() < f64::EPSILON);
    assert!((collected.max_rt.unwrap() - streamed.max_rt.unwrap()).abs() < f64::EPSILON);
}

#[test]
fn mzml_nonexistent_file() {
    assert!(mzml::parse_mzml(Path::new("/nonexistent/file.mzML")).is_err());
}

// ── FASTQ synthetic file round-trip ─────────────────────────────

fn create_fastq_file(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("test.fastq");
    let mut f = File::create(&path).unwrap();
    // 3 records with different lengths and qualities
    writeln!(f, "@seq1 description").unwrap();
    writeln!(f, "ACGTACGT").unwrap();
    writeln!(f, "+").unwrap();
    writeln!(f, "IIIIIIII").unwrap(); // Q=40 each
    writeln!(f, "@seq2").unwrap();
    writeln!(f, "GGCCAATTNN").unwrap();
    writeln!(f, "+").unwrap();
    writeln!(f, "BBBBBBBBBB").unwrap(); // Q=33 each
    writeln!(f, "@seq3").unwrap();
    writeln!(f, "AAAA").unwrap();
    writeln!(f, "+").unwrap();
    writeln!(f, "!!!!").unwrap(); // Q=0 each
    path
}

#[test]
fn fastq_synthetic_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = create_fastq_file(dir.path());
    let records = fastq::parse_fastq(&path).unwrap();

    assert_eq!(records.len(), 3);
    assert_eq!(records[0].sequence, b"ACGTACGT");
    assert_eq!(records[1].sequence, b"GGCCAATTNN");
    assert_eq!(records[2].sequence, b"AAAA");
}

#[test]
fn fastq_stats_synthetic() {
    let dir = TempDir::new().unwrap();
    let path = create_fastq_file(dir.path());
    let records = fastq::parse_fastq(&path).unwrap();
    let stats = fastq::compute_stats(&records);

    assert_eq!(stats.num_sequences, 3);
    assert_eq!(stats.total_bases, 22); // 8 + 10 + 4
    assert_eq!(stats.min_length, 4);
    assert_eq!(stats.max_length, 10);
}

#[test]
fn fastq_streaming_stats_match_collected() {
    let dir = TempDir::new().unwrap();
    let path = create_fastq_file(dir.path());

    let records = fastq::parse_fastq(&path).unwrap();
    let collected_stats = fastq::compute_stats(&records);
    let streaming_stats = fastq::stats_from_file(&path).unwrap();

    assert_eq!(collected_stats.num_sequences, streaming_stats.num_sequences);
    assert_eq!(collected_stats.total_bases, streaming_stats.total_bases);
    assert_eq!(collected_stats.min_length, streaming_stats.min_length);
    assert_eq!(collected_stats.max_length, streaming_stats.max_length);
    assert!((collected_stats.mean_quality - streaming_stats.mean_quality).abs() < 1e-10);
    assert!((collected_stats.gc_content - streaming_stats.gc_content).abs() < 1e-10);
    assert_eq!(collected_stats.q30_count, streaming_stats.q30_count);
}

#[test]
fn fastq_determinism() {
    let dir = TempDir::new().unwrap();
    let path = create_fastq_file(dir.path());
    let run1 = fastq::parse_fastq(&path).unwrap();
    let run2 = fastq::parse_fastq(&path).unwrap();

    assert_eq!(run1.len(), run2.len());
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.sequence, b.sequence);
        assert_eq!(a.quality, b.quality);
    }
}

#[test]
fn fastq_nonexistent_file() {
    assert!(fastq::parse_fastq(Path::new("/nonexistent/file.fastq")).is_err());
}

#[test]
fn fastq_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.fastq");
    File::create(&path).unwrap();

    let records = fastq::parse_fastq(&path).unwrap();
    assert!(records.is_empty());

    let stats = fastq::stats_from_file(&path).unwrap();
    assert_eq!(stats.num_sequences, 0);
    assert_eq!(stats.min_length, 0);
    assert_eq!(stats.max_length, 0);
}

#[test]
fn fastq_trailing_newlines() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("trailing.fastq");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "@seq1").unwrap();
    writeln!(f, "ACGT").unwrap();
    writeln!(f, "+").unwrap();
    writeln!(f, "IIII").unwrap();
    // Extra trailing newlines
    writeln!(f).unwrap();
    writeln!(f).unwrap();

    let records = fastq::parse_fastq(&path).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].sequence, b"ACGT");
}

#[test]
fn fastq_malformed_header() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("malformed.fastq");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "NOTAHEADER").unwrap();
    writeln!(f, "ACGT").unwrap();
    writeln!(f, "+").unwrap();
    writeln!(f, "IIII").unwrap();

    let result = fastq::parse_fastq(&path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("expected '@'"),
        "Error should mention missing '@': {err_msg}"
    );
}

#[test]
fn fastq_id_extraction() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ids.fastq");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "@read1 length=250 instrument=MiSeq").unwrap();
    writeln!(f, "ACGT").unwrap();
    writeln!(f, "+").unwrap();
    writeln!(f, "IIII").unwrap();
    writeln!(f, "@read2/1").unwrap();
    writeln!(f, "GGCC").unwrap();
    writeln!(f, "+").unwrap();
    writeln!(f, "JJJJ").unwrap();

    let records = fastq::parse_fastq(&path).unwrap();
    assert_eq!(records[0].id, "read1");
    assert_eq!(records[1].id, "read2/1");
}

#[test]
fn fastq_gzip_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.fastq.gz");

    // Write gzip-compressed FASTQ
    let file = File::create(&path).unwrap();
    let mut gz = GzEncoder::new(file, Compression::default());
    writeln!(gz, "@gz_seq1").unwrap();
    writeln!(gz, "ACGTACGTACGT").unwrap();
    writeln!(gz, "+").unwrap();
    writeln!(gz, "IIIIIIIIIIII").unwrap();
    writeln!(gz, "@gz_seq2").unwrap();
    writeln!(gz, "GGCCGG").unwrap();
    writeln!(gz, "+").unwrap();
    writeln!(gz, "JJJJJJ").unwrap();
    gz.finish().unwrap();

    // Parse with our sovereign parser
    let records = fastq::parse_fastq(&path).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, "gz_seq1");
    assert_eq!(records[0].sequence, b"ACGTACGTACGT");
    assert_eq!(records[1].id, "gz_seq2");
    assert_eq!(records[1].sequence, b"GGCCGG");

    // Verify streaming stats match collected stats
    let collected = fastq::compute_stats(&records);
    let streaming = fastq::stats_from_file(&path).unwrap();
    assert_eq!(collected.num_sequences, streaming.num_sequences);
    assert_eq!(collected.total_bases, streaming.total_bases);
    assert_eq!(collected.min_length, streaming.min_length);
    assert_eq!(collected.max_length, streaming.max_length);
    assert!((collected.mean_quality - streaming.mean_quality).abs() < 1e-10);
    assert!((collected.gc_content - streaming.gc_content).abs() < 1e-10);
}

#[test]
fn fastq_gzip_stats_correctness() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("stats.fastq.gz");

    let file = File::create(&path).unwrap();
    let mut gz = GzEncoder::new(file, Compression::default());
    writeln!(gz, "@s1").unwrap();
    writeln!(gz, "GGCCGGCC").unwrap(); // 100% GC
    writeln!(gz, "+").unwrap();
    writeln!(gz, "IIIIIIII").unwrap(); // Q=40
    writeln!(gz, "@s2").unwrap();
    writeln!(gz, "AAAATTTT").unwrap(); // 0% GC
    writeln!(gz, "+").unwrap();
    writeln!(gz, "!!!!!!!!").unwrap(); // Q=0
    gz.finish().unwrap();

    let stats = fastq::stats_from_file(&path).unwrap();
    assert_eq!(stats.num_sequences, 2);
    assert_eq!(stats.total_bases, 16);
    assert!((stats.gc_content - 0.5).abs() < 1e-10);
    assert!((stats.mean_quality - 20.0).abs() < 1e-10);
    assert_eq!(stats.q30_count, 1);
}

// ── mzML known-value decoding ───────────────────────────────────

#[test]
fn mzml_stats_with_constructed_spectra() {
    let spectra = vec![
        mzml::MzmlSpectrum {
            index: 0,
            ms_level: 1,
            rt_minutes: 1.0,
            tic: 5000.0,
            base_peak_mz: 500.0,
            base_peak_intensity: 3000.0,
            lowest_mz: 80.0,
            highest_mz: 1000.0,
            mz_array: vec![80.0, 250.0, 500.0, 750.0, 1000.0],
            intensity_array: vec![100.0, 500.0, 3000.0, 200.0, 50.0],
        },
        mzml::MzmlSpectrum {
            index: 1,
            ms_level: 2,
            rt_minutes: 1.5,
            tic: 2000.0,
            base_peak_mz: 250.0,
            base_peak_intensity: 1500.0,
            lowest_mz: 100.0,
            highest_mz: 500.0,
            mz_array: vec![100.0, 250.0, 400.0],
            intensity_array: vec![500.0, 1500.0, 300.0],
        },
    ];

    let stats = mzml::compute_stats(&spectra);
    assert_eq!(stats.num_spectra, 2);
    assert_eq!(stats.num_ms1, 1);
    assert_eq!(stats.num_ms2, 1);
    assert_eq!(stats.total_peaks, 8);
    assert!((stats.min_rt.unwrap() - 1.0).abs() < f64::EPSILON);
    assert!((stats.max_rt.unwrap() - 1.5).abs() < f64::EPSILON);
    assert!((stats.min_mz.unwrap() - 80.0).abs() < f64::EPSILON);
    assert!((stats.max_mz.unwrap() - 1000.0).abs() < f64::EPSILON);
}
