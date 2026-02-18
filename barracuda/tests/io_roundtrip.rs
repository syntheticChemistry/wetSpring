// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for I/O parsers: round-trip, determinism, edge cases.
//!
//! Each test creates synthetic data in a temporary directory,
//! parses it with our production code, and verifies correctness.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;
use wetspring_barracuda::bio::{
    derep, diversity, eic, feature_table, kmd, kmer, merge_pairs, quality, signal, spectral_match,
    tolerance_search,
};
use wetspring_barracuda::encoding::base64_encode;
use wetspring_barracuda::io::{fastq, ms2, mzml};

// Re-export for gzip tests
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

// ── Diversity analytical edge cases ─────────────────────────────

#[test]
fn diversity_empty_community() {
    let empty: Vec<f64> = vec![];
    assert!(diversity::shannon(&empty).abs() < f64::EPSILON);
    assert!(diversity::simpson(&empty).abs() < f64::EPSILON);
    assert!(diversity::observed_features(&empty).abs() < f64::EPSILON);
}

#[test]
fn diversity_all_zeros() {
    let zeros = vec![0.0, 0.0, 0.0];
    assert!(diversity::shannon(&zeros).abs() < f64::EPSILON);
    assert!(diversity::simpson(&zeros).abs() < f64::EPSILON);
    assert!(diversity::observed_features(&zeros).abs() < f64::EPSILON);
}

#[test]
fn diversity_bray_curtis_triangle_inequality() {
    let a = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let b = vec![15.0, 25.0, 5.0, 45.0, 10.0];
    let c = vec![50.0, 0.0, 50.0, 0.0, 50.0];

    let ab = diversity::bray_curtis(&a, &b);
    let bc = diversity::bray_curtis(&b, &c);
    let ac = diversity::bray_curtis(&a, &c);

    // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    assert!(ac <= ab + bc + f64::EPSILON);
}

#[test]
fn diversity_matrix_is_symmetric() {
    let samples = vec![
        vec![10.0, 20.0, 0.0, 5.0],
        vec![15.0, 10.0, 5.0, 0.0],
        vec![0.0, 30.0, 20.0, 10.0],
    ];
    let dm = diversity::bray_curtis_matrix(&samples);

    for i in 0..3 {
        for j in 0..3 {
            let ij = dm[i * 3 + j];
            let ji = dm[j * 3 + i];
            assert!(
                (ij - ji).abs() < 1e-15,
                "Matrix not symmetric at ({i},{j}): {ij} != {ji}"
            );
        }
    }
}

// ── K-mer determinism ───────────────────────────────────────────

#[test]
fn kmer_deterministic_across_runs() {
    let seq = b"ACGTACGTNNACGTACGT";
    let r1 = kmer::count_kmers(seq, 5);
    let r2 = kmer::count_kmers(seq, 5);

    assert_eq!(r1.total_valid_kmers, r2.total_valid_kmers);
    assert_eq!(r1.skipped_ambiguous, r2.skipped_ambiguous);
    assert_eq!(r1.counts.len(), r2.counts.len());
    for (k, v) in &r1.counts {
        assert_eq!(r2.counts.get(k), Some(v));
    }
}

#[test]
fn kmer_canonical_form_consistent() {
    // Forward and reverse complement of same sequence should give same canonical counts
    let fwd = b"ACGTACGT";
    let rev = b"ACGTACGT"; // ACGTACGT is a palindrome
    let c1 = kmer::count_kmers(fwd, 4);
    let c2 = kmer::count_kmers(rev, 4);
    assert_eq!(c1.counts, c2.counts);
}

// ── Tolerance search edge cases ─────────────────────────────────

#[test]
fn tolerance_search_boundary_values() {
    let mz = vec![100.0, 200.0, 300.0];
    // Exactly at boundary of 10 ppm for 200.0 = ±0.002
    let matches = tolerance_search::find_within_ppm(&mz, 200.002, 10.0);
    assert!(
        !matches.is_empty(),
        "Should find 200.0 within 10 ppm of 200.002"
    );
}

#[test]
fn tolerance_search_large_ppm() {
    let mz: Vec<f64> = (1..=100).map(|i| f64::from(i) * 10.0).collect();
    let matches = tolerance_search::find_within_ppm(&mz, 500.0, 100_000.0);
    // 100,000 ppm = 10% = ±50 Da, so 450-550 range
    assert!(matches.len() >= 10, "Should find many matches at 100K ppm");
}

#[test]
fn pfas_screen_multiple_fragment_types() {
    let frags = tolerance_search::PfasFragments::default();
    // Build peaks with CF2, C2F4, and HF differences
    let mz = vec![
        100.0,
        100.0 + frags.cf2,  // CF2 pair
        100.0 + frags.c2f4, // C2F4 pair
        200.0,
        200.0 + frags.hf, // HF pair
    ];
    let intensity = vec![1000.0; 5];

    let result = tolerance_search::screen_pfas_fragments(&mz, &intensity, 300.0, 5.0, 0.01, 1.0);

    assert!(result.is_some());
    let r = result.unwrap();
    assert!(r.cf2_count >= 1, "Should find at least 1 CF2 pair");
    assert!(r.c2f4_count >= 1, "Should find at least 1 C2F4 pair");
    assert!(r.hf_count >= 1, "Should find at least 1 HF pair");
}

// ── Quality filtering integration ─────────────────────────────

#[test]
fn quality_filter_fastq_pipeline() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.fastq");
    let mut f = File::create(&path).unwrap();

    // Write 3 reads: 1 good, 1 with low trailing quality, 1 all bad
    writeln!(f, "@good_read").unwrap();
    writeln!(f, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
    writeln!(f, "+").unwrap();
    // All Q30+ (ASCII 63 = 30+33)
    writeln!(f, "???????????????????????????????????????????????????").unwrap();

    writeln!(f, "@trim_read").unwrap();
    writeln!(f, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
    writeln!(f, "+").unwrap();
    // Good then bad: 40 Q30 then 11 Q2 (ASCII 35 = 2+33)
    let qual: String = std::iter::repeat('?')
        .take(40)
        .chain(std::iter::repeat('#').take(11))
        .collect();
    writeln!(f, "{qual}").unwrap();

    writeln!(f, "@bad_read").unwrap();
    writeln!(f, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT").unwrap();
    writeln!(f, "+").unwrap();
    // All Q2
    writeln!(f, "###################################################").unwrap();

    let records = fastq::parse_fastq(&path).unwrap();
    assert_eq!(records.len(), 3);

    let params = quality::QualityParams {
        min_length: 36,
        ..quality::QualityParams::default()
    };

    let (filtered, stats) = quality::filter_reads(&records, &params);
    assert_eq!(stats.input_reads, 3);
    assert!(stats.output_reads >= 1, "at least the good read passes");
    assert!(filtered.iter().all(|r| r.sequence.len() >= 36));
}

#[test]
fn quality_adapter_trim_integration() {
    let adapter = b"AGATCGGAAGAG"; // Illumina TruSeq adapter prefix
    let seq = b"ACGTACGTACGTACGTACGTACGTACGTAGATCGGAAGAG";
    let record = fastq::FastqRecord {
        id: "test".to_string(),
        sequence: seq.to_vec(),
        quality: vec![33 + 30; seq.len()],
    };

    let (trimmed, found) = quality::trim_adapter_3prime(&record, adapter, 1, 8);
    assert!(found);
    assert!(trimmed.sequence.len() < seq.len());
}

// ── Signal processing integration ─────────────────────────────

#[test]
fn signal_chromatographic_pipeline() {
    // Simulate a chromatogram with 3 Gaussian peaks
    let n = 300;
    #[allow(clippy::cast_precision_loss)]
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64;
            let p1 = 10000.0 * (-0.5 * ((x - 50.0) / 5.0).powi(2)).exp();
            let p2 = 50000.0 * (-0.5 * ((x - 150.0) / 8.0).powi(2)).exp();
            let p3 = 20000.0 * (-0.5 * ((x - 250.0) / 6.0).powi(2)).exp();
            p1 + p2 + p3 + 100.0 // baseline
        })
        .collect();

    let params = signal::PeakParams {
        min_height: Some(5000.0),
        min_prominence: Some(3000.0),
        ..signal::PeakParams::default()
    };

    let peaks = signal::find_peaks(&data, &params);
    assert_eq!(peaks.len(), 3, "Should detect 3 peaks");
    assert!(
        peaks[1].height > peaks[0].height,
        "Middle peak should be highest"
    );
    assert!(
        peaks[1].height > peaks[2].height,
        "Middle peak should be highest"
    );
}

// ── Spectral matching integration ─────────────────────────────

#[test]
fn spectral_match_library_search() {
    // Simulate a query spectrum matching one library entry
    let query_mz = vec![100.0, 150.0, 200.0, 250.0, 300.0];
    let query_int = vec![1000.0, 500.0, 250.0, 100.0, 50.0];

    let lib = vec![
        // Match: same peaks
        (query_mz.clone(), query_int.clone()),
        // Partial: shifted m/z
        (
            vec![100.0, 150.0, 200.0, 260.0, 310.0],
            vec![1000.0, 500.0, 250.0, 100.0, 50.0],
        ),
        // No overlap
        (vec![400.0, 500.0, 600.0], vec![1000.0, 500.0, 250.0]),
    ];

    let scores = spectral_match::pairwise_cosine(
        &[
            (query_mz, query_int),
            lib[0].clone(),
            lib[1].clone(),
            lib[2].clone(),
        ],
        0.5,
    );

    // (1,0) should be 1.0 (identical)
    assert!(
        (scores[0] - 1.0).abs() < 1e-10,
        "Identical should score 1.0"
    );
}

// ── KMD integration ───────────────────────────────────────────

#[test]
fn kmd_pfas_series_detection() {
    // Real PFAS masses: perfluoroalkyl carboxylic acids [M-H]-
    let pfoa = 412.966; // C8F15O2-  (PFOA)
    let pfhxa = 312.973; // C6F11O2- (PFHxA) = PFOA - CF2
    let pfba = 212.979; // C4F7O2-   (PFBA) = PFHxA - CF2

    let (results, groups) = kmd::pfas_kmd_screen(&[pfoa, pfhxa, pfba], 0.01);
    assert_eq!(results.len(), 3);

    // All three should be in the same group (homologous series)
    let max_group = groups.iter().map(Vec::len).max().unwrap_or(0);
    assert_eq!(max_group, 3, "All 3 PFCA should group together");
}

// ── Feature table integration ─────────────────────────────────

#[test]
fn feature_table_synthetic_lcms() {
    // Simulate 2 compounds across 50 MS1 scans
    let n_scans = 50;
    let spectra: Vec<mzml::MzmlSpectrum> = (0..n_scans)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let rt = 2.0 + (i as f64) * 0.2; // 2.0 to 11.8 min
            let x1 = (rt - 5.0) / 0.5;
            let x2 = (rt - 8.0) / 0.6;
            let int1 = 100000.0 * (-0.5 * x1 * x1).exp() + 200.0;
            let int2 = 50000.0 * (-0.5 * x2 * x2).exp() + 200.0;
            mzml::MzmlSpectrum {
                index: i,
                ms_level: 1,
                rt_minutes: rt,
                tic: int1 + int2,
                base_peak_mz: if int1 > int2 { 200.0 } else { 400.0 },
                base_peak_intensity: int1.max(int2),
                lowest_mz: 200.0,
                highest_mz: 400.0,
                mz_array: vec![200.0, 400.0],
                intensity_array: vec![int1, int2],
            }
        })
        .collect();

    let params = feature_table::FeatureParams {
        eic_ppm: 10.0,
        min_scans: 2,
        peak_params: signal::PeakParams {
            min_prominence: Some(5000.0),
            ..signal::PeakParams::default()
        },
        min_height: 5000.0,
        min_snr: 2.0,
    };

    let ft = feature_table::extract_features(&spectra, &params);
    assert!(
        ft.features.len() >= 2,
        "Should find at least 2 features, got {}",
        ft.features.len()
    );
    assert!(ft.mass_tracks_evaluated >= 2);

    // First feature should be at m/z 200, second at m/z 400
    if ft.features.len() >= 2 {
        assert!((ft.features[0].mz - 200.0).abs() < 1.0);
        assert!((ft.features[1].mz - 400.0).abs() < 1.0);
        assert!(
            (ft.features[0].rt_apex - 5.0).abs() < 0.5,
            "First peak RT near 5.0"
        );
        assert!(
            (ft.features[1].rt_apex - 8.0).abs() < 0.5,
            "Second peak RT near 8.0"
        );
    }
}

// ── EIC integration ───────────────────────────────────────────

#[test]
fn eic_extract_and_integrate() {
    // 3 scans with peaks at m/z 500
    let spectra = vec![
        mzml::MzmlSpectrum {
            index: 0,
            ms_level: 1,
            rt_minutes: 1.0,
            tic: 100.0,
            base_peak_mz: 500.0,
            base_peak_intensity: 100.0,
            lowest_mz: 500.0,
            highest_mz: 500.0,
            mz_array: vec![500.0],
            intensity_array: vec![100.0],
        },
        mzml::MzmlSpectrum {
            index: 1,
            ms_level: 1,
            rt_minutes: 2.0,
            tic: 500.0,
            base_peak_mz: 500.0,
            base_peak_intensity: 500.0,
            lowest_mz: 500.0,
            highest_mz: 500.0,
            mz_array: vec![500.0],
            intensity_array: vec![500.0],
        },
        mzml::MzmlSpectrum {
            index: 2,
            ms_level: 1,
            rt_minutes: 3.0,
            tic: 100.0,
            base_peak_mz: 500.0,
            base_peak_intensity: 100.0,
            lowest_mz: 500.0,
            highest_mz: 500.0,
            mz_array: vec![500.0],
            intensity_array: vec![100.0],
        },
    ];

    let eics = eic::extract_eics(&spectra, &[500.0], 10.0);
    assert_eq!(eics.len(), 1);
    assert_eq!(eics[0].intensity, vec![100.0, 500.0, 100.0]);

    // Integrate: trapezoid over 3 points
    let area = eic::integrate_peak(&eics[0].rt, &eics[0].intensity, 0, 2);
    // (100+500)/2*1 + (500+100)/2*1 = 300 + 300 = 600
    assert!((area - 600.0).abs() < 1e-10);
}

// ── Diversity pipeline integration ────────────────────────────

#[test]
fn diversity_full_pipeline() {
    // Simulate: FASTQ → k-mer → diversity → PCoA
    let community_a = vec![100.0, 50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 1.0];
    let community_b = vec![90.0, 60.0, 25.0, 25.0, 15.0, 8.0, 4.0, 3.0, 1.0, 1.0];
    let community_c = vec![1.0, 1.0, 1.0, 1.0, 1.0, 200.0, 1.0, 1.0, 1.0, 1.0];

    // Alpha diversity
    let alpha_a = diversity::alpha_diversity(&community_a);
    let alpha_b = diversity::alpha_diversity(&community_b);
    let alpha_c = diversity::alpha_diversity(&community_c);

    assert!(alpha_a.evenness < 1.0);
    assert!(alpha_b.evenness < 1.0);
    assert!(alpha_c.evenness < alpha_a.evenness, "C is very uneven");

    // Beta diversity
    let samples = vec![community_a, community_b, community_c];
    let bc = diversity::bray_curtis_condensed(&samples);
    assert_eq!(bc.len(), 3); // 3 pairs

    // A and B should be more similar than A and C
    let bc_ab = bc[diversity::condensed_index(1, 0)];
    let bc_ac = bc[diversity::condensed_index(2, 0)];
    assert!(
        bc_ab < bc_ac,
        "A-B ({bc_ab:.4}) should be < A-C ({bc_ac:.4})"
    );

    // Rarefaction
    let depths = vec![10.0, 50.0, 100.0, 222.0];
    let curve = diversity::rarefaction_curve(&samples[0], &depths);
    assert_eq!(curve.len(), 4);
    // Should be monotonically increasing
    for i in 1..curve.len() {
        assert!(curve[i] >= curve[i - 1] - 1e-10);
    }
}

// ── Paired-end merging integration ────────────────────────────

#[test]
fn merge_pairs_16s_simulation() {
    // Simulate 16S V4 amplicon: ~253bp, forward read 250bp, reverse read 250bp
    // Overlap should be ~247bp (2*250 - 253 = 247)

    // Generate a "16S amplicon" sequence (253bp)
    let amplicon: Vec<u8> = (0..253)
        .map(|i| match i % 4 {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            _ => b'T',
        })
        .collect();

    // Forward read: first 250bp
    let fwd_seq = amplicon[..250].to_vec();
    let fwd_qual: Vec<u8> = vec![33 + 30; 250];

    // Reverse read: last 250bp, reverse-complemented
    let rev_region = &amplicon[3..]; // 253-3 = 250bp from position 3
    let rev_seq = merge_pairs::reverse_complement(rev_region);
    let rev_qual: Vec<u8> = vec![33 + 30; 250];

    let fwd = fastq::FastqRecord {
        id: "pair1".to_string(),
        sequence: fwd_seq,
        quality: fwd_qual,
    };
    let rev = fastq::FastqRecord {
        id: "pair1".to_string(),
        sequence: rev_seq,
        quality: rev_qual,
    };

    let params = merge_pairs::MergeParams {
        min_overlap: 10,
        ..merge_pairs::MergeParams::default()
    };

    let result = merge_pairs::merge_pair(&fwd, &rev, &params);
    assert!(result.merged.is_some(), "Should merge successfully");
    let merged = result.merged.unwrap();
    assert_eq!(
        merged.sequence.len(),
        253,
        "Should reconstruct full amplicon"
    );
    assert_eq!(result.overlap, 247, "Overlap should be 247bp");
}

#[test]
fn merge_pairs_batch_with_quality() {
    // Create a batch of read pairs with varying quality
    let n_pairs = 5;
    let overlap_seq = b"ACGTACGTACGTACGTACGT"; // 20bp overlap
    let fwd_prefix = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // 28bp
    let rev_suffix = b"TTTTTTTTTTTTTTTTTTTTTTTTTTTT"; // 28bp

    let fwd_seq: Vec<u8> = [fwd_prefix as &[u8], overlap_seq].concat();
    let rev_rc: Vec<u8> = [overlap_seq as &[u8], rev_suffix].concat();
    let rev_seq = merge_pairs::reverse_complement(&rev_rc);

    let fwd_reads: Vec<fastq::FastqRecord> = (0..n_pairs)
        .map(|i| fastq::FastqRecord {
            id: format!("pair{i}"),
            sequence: fwd_seq.clone(),
            quality: vec![33 + 30; fwd_seq.len()],
        })
        .collect();

    let rev_reads: Vec<fastq::FastqRecord> = (0..n_pairs)
        .map(|i| fastq::FastqRecord {
            id: format!("pair{i}"),
            sequence: rev_seq.clone(),
            quality: vec![33 + 30; rev_seq.len()],
        })
        .collect();

    let (merged, stats) =
        merge_pairs::merge_pairs(&fwd_reads, &rev_reads, &merge_pairs::MergeParams::default());

    assert_eq!(stats.input_pairs, 5);
    assert_eq!(stats.merged_count, 5);
    assert_eq!(merged.len(), 5);
    assert!((stats.mean_overlap - 20.0).abs() < 1.0);
    assert!((stats.mean_merged_length - 76.0).abs() < 1.0); // 28 + 20 + 28 = 76
}

// ── Dereplication integration ─────────────────────────────────

#[test]
fn derep_simulated_community() {
    // Simulate a sequencing run: 100 reads from 5 ASVs with realistic abundances
    let asvs = vec![
        (b"ACGTACGTACGTACGT".to_vec(), 40), // dominant species
        (b"GCTAGCTAGCTAGCTA".to_vec(), 25),
        (b"TTTTAAAACCCCGGGG".to_vec(), 20),
        (b"AACCGGTTAACCGGTT".to_vec(), 10),
        (b"GGGCCCAAATTTGGGC".to_vec(), 5), // rare species
    ];

    let mut records = Vec::new();
    for (seq, count) in &asvs {
        for i in 0..*count {
            records.push(fastq::FastqRecord {
                id: format!("read_{}", records.len()),
                sequence: seq.clone(),
                quality: vec![33 + 25 + (i % 10) as u8; seq.len()],
            });
        }
    }

    let (uniques, stats) = derep::dereplicate(&records, derep::DerepSort::Abundance, 0);

    assert_eq!(stats.input_sequences, 100);
    assert_eq!(stats.unique_sequences, 5);
    assert_eq!(stats.max_abundance, 40);
    assert_eq!(stats.singletons, 0);

    // Sorted by abundance — first should be the dominant ASV
    assert_eq!(uniques[0].abundance, 40);
    assert_eq!(uniques[0].sequence, b"ACGTACGTACGTACGT");

    // Test abundance vector for diversity
    let counts = derep::abundance_vector(&uniques);
    assert_eq!(counts, vec![40.0, 25.0, 20.0, 10.0, 5.0]);

    // Diversity on dereplicated abundances
    let shannon = diversity::shannon(&counts);
    assert!(shannon > 0.0);
    assert!(shannon < f64::ln(5.0) + 0.01); // can't exceed ln(S)
}

#[test]
fn derep_singleton_filtering() {
    // In real pipelines, singletons are often removed as noise
    let records = vec![
        fastq::FastqRecord {
            id: "r1".into(),
            sequence: b"AAAA".to_vec(),
            quality: vec![33 + 30; 4],
        },
        fastq::FastqRecord {
            id: "r2".into(),
            sequence: b"AAAA".to_vec(),
            quality: vec![33 + 30; 4],
        },
        fastq::FastqRecord {
            id: "r3".into(),
            sequence: b"CCCC".to_vec(),
            quality: vec![33 + 30; 4],
        },
        fastq::FastqRecord {
            id: "r4".into(),
            sequence: b"GGGG".to_vec(),
            quality: vec![33 + 30; 4],
        },
    ];

    let (uniques, stats) = derep::dereplicate(&records, derep::DerepSort::Abundance, 2);
    assert_eq!(stats.unique_sequences, 1); // only AAAA kept
    assert_eq!(uniques[0].abundance, 2);
}

#[test]
fn derep_to_fasta_output() {
    let records = vec![
        fastq::FastqRecord {
            id: "s1".into(),
            sequence: b"ACGT".to_vec(),
            quality: vec![33 + 30; 4],
        },
        fastq::FastqRecord {
            id: "s2".into(),
            sequence: b"ACGT".to_vec(),
            quality: vec![33 + 30; 4],
        },
        fastq::FastqRecord {
            id: "s3".into(),
            sequence: b"TTTT".to_vec(),
            quality: vec![33 + 30; 4],
        },
    ];

    let (uniques, _) = derep::dereplicate(&records, derep::DerepSort::Abundance, 0);
    let fasta = derep::to_fasta_with_abundance(&uniques);

    assert!(fasta.contains(";size=2"));
    assert!(fasta.contains(";size=1"));
    assert!(fasta.contains("ACGT\n"));
}

// ── End-to-end 16S pipeline integration ───────────────────────

#[test]
fn pipeline_16s_fastq_to_diversity() {
    // Simulate the 16S pipeline: quality filter → merge → derep → diversity
    // This is the Track 1 sovereign pipeline end-to-end (minus denoising)

    // Generate forward + reverse read pairs
    let overlap = b"ACGTACGTACGTACGTACGT"; // 20bp shared
    let prefix = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // 40bp

    let species = vec![
        (b"CCCCCCCCCC".to_vec(), 30),
        (b"GGGGGGGGGG".to_vec(), 20),
        (b"TTTTTTTTTT".to_vec(), 10),
    ];

    let mut fwd_reads = Vec::new();
    let mut rev_reads = Vec::new();

    for (suffix, count) in &species {
        let fwd_seq: Vec<u8> = [prefix as &[u8], overlap].concat(); // 60bp
        let rev_rc: Vec<u8> = [overlap as &[u8], suffix.as_slice()].concat(); // 30bp
        let rev_seq = merge_pairs::reverse_complement(&rev_rc);

        for i in 0..*count {
            fwd_reads.push(fastq::FastqRecord {
                id: format!("r{}", fwd_reads.len()),
                sequence: fwd_seq.clone(),
                quality: vec![33 + 30; fwd_seq.len()],
            });
            rev_reads.push(fastq::FastqRecord {
                id: format!("r{}", rev_reads.len()),
                sequence: rev_seq.clone(),
                quality: vec![33 + 28 + (i % 5) as u8; rev_seq.len()],
            });
        }
    }

    // 1. Quality filter forward reads
    let qparams = quality::QualityParams {
        min_length: 20,
        ..quality::QualityParams::default()
    };
    let (fwd_filtered, _fstats) = quality::filter_reads(&fwd_reads, &qparams);
    let (rev_filtered, _rstats) = quality::filter_reads(&rev_reads, &qparams);

    // Pair up (keep matched pairs)
    let n_pairs = fwd_filtered.len().min(rev_filtered.len());
    assert!(n_pairs >= 50, "Most reads should pass quality filter");

    // 2. Merge pairs
    let (merged, mstats) = merge_pairs::merge_pairs(
        &fwd_filtered[..n_pairs],
        &rev_filtered[..n_pairs],
        &merge_pairs::MergeParams::default(),
    );
    assert!(mstats.merged_count > 40, "Most pairs should merge");

    // 3. Dereplicate
    let (uniques, dstats) = derep::dereplicate(&merged, derep::DerepSort::Abundance, 0);
    assert_eq!(dstats.unique_sequences, 3, "Should find 3 unique sequences");

    // 4. Diversity from abundance vector
    let counts = derep::abundance_vector(&uniques);
    let alpha = diversity::alpha_diversity(&counts);

    assert!(alpha.observed > 0.0);
    assert!(alpha.shannon > 0.0);
    assert!(alpha.evenness > 0.0);
    assert!(alpha.evenness <= 1.0);
}
