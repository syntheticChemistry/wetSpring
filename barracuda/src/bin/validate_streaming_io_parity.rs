// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::redundant_closure_for_method_calls,
    reason = "validation harness: required for domain validation"
)]
//! # Exp209: Streaming I/O Parity Validation
//!
//! Validates that the byte-native I/O evolution preserves all existing results:
//!
//! - **S1**: FASTQ `stats_from_file` (streaming byte-native) matches
//!   `compute_stats(FastqIter::collect())` (batch → accumulate)
//! - **S2**: FASTQ `for_each_record` (streaming borrowed) yields identical
//!   records to `FastqIter::collect()` (batch owned)
//! - **S3**: Multi-byte UTF-8 in FASTQ headers does not panic (fuzz fix)
//! - **S4**: Nanopore bulk `bytemuck::cast_slice_mut` read matches per-field
//!   reference (NRS round-trip with synthetic data)
//! - **S5**: MS2 `for_each_spectrum` streaming matches `Ms2Iter::collect()` batch
//!
//! This is a post-audit "trust but verify" binary — no external data needed.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | V65 audit (byte-native FASTQ, bytemuck nanopore, streaming APIs) |
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 66 (V66 post-audit) |
//! | Hardware | CPU only (I/O validation) |
//! | Command | `cargo run --release --bin validate_streaming_io_parity` |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::io::Write;
use std::path::PathBuf;

use wetspring_barracuda::io::fastq::{self, FastqRefRecord};
use wetspring_barracuda::io::ms2;
use wetspring_barracuda::io::nanopore::{self, NanoporeIter, SyntheticSignalGenerator};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;
use wetspring_barracuda::validation::OrExit;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("wetspring_exp209_{name}"))
}

// ── S1: FASTQ byte-native stats parity ──────────────────────────────────────

fn write_synthetic_fastq(path: &std::path::Path) {
    let mut f = std::fs::File::create(path).or_exit("unexpected error");
    let reads: &[(&str, &[u8], &[u8])] = &[
        ("read1", b"ATGCATGCATGCATGC", b"IIIIIIIIIIIIIIII"),
        ("read2", b"GGGGCCCCAAAA", b"FFFFFFFFFFFF"),
        (
            "read3",
            b"TTTTTTTTTTTTTTTTTTTTTTTTTT",
            b"5555555555555555555555555I",
        ),
        ("read4", b"ACGTACGT", b"IIIIIIII"),
        (
            "read5_long",
            b"ATCGATCGATCGATCGATCGATCGATCGATCG",
            b"HHHHHHHHHHHHHHHHIIIIIIIIIIIIIIII",
        ),
    ];
    for (id, seq, qual) in reads {
        writeln!(f, "@{id}").or_exit("unexpected error");
        f.write_all(seq).or_exit("unexpected error");
        writeln!(f).or_exit("unexpected error");
        writeln!(f, "+").or_exit("unexpected error");
        f.write_all(qual).or_exit("unexpected error");
        writeln!(f).or_exit("unexpected error");
    }
}

fn validate_fastq_stats_parity(v: &mut Validator) {
    v.section("═══ S1: FASTQ byte-native stats parity ═══");

    let path = temp_path("stats_parity.fastq");
    write_synthetic_fastq(&path);

    let batch_records: Vec<_> = fastq::FastqIter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");
    let batch_stats = fastq::compute_stats(&batch_records);

    let stream_stats = fastq::stats_from_file(&path).or_exit("unexpected error");

    v.check_count(
        "num_sequences batch==stream",
        batch_stats.num_sequences,
        stream_stats.num_sequences,
    );
    v.check_count_u64(
        "total_bases batch==stream",
        batch_stats.total_bases,
        stream_stats.total_bases,
    );
    v.check_count(
        "min_length batch==stream",
        batch_stats.min_length,
        stream_stats.min_length,
    );
    v.check_count(
        "max_length batch==stream",
        batch_stats.max_length,
        stream_stats.max_length,
    );
    v.check(
        "mean_length batch==stream",
        batch_stats.mean_length,
        stream_stats.mean_length,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "mean_quality batch==stream",
        batch_stats.mean_quality,
        stream_stats.mean_quality,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "gc_content batch==stream",
        batch_stats.gc_content,
        stream_stats.gc_content,
        tolerances::ANALYTICAL_F64,
    );
    v.check_count(
        "q30_count batch==stream",
        batch_stats.q30_count,
        stream_stats.q30_count,
    );

    v.check_count("num_sequences == 5", batch_stats.num_sequences, 5);
    v.check_count("min_length == 8", batch_stats.min_length, 8);
    v.check_count("max_length == 32", batch_stats.max_length, 32);

    let _ = std::fs::remove_file(&path);
}

// ── S2: FASTQ for_each_record ↔ FastqIter parity ─────────────────────────

fn validate_fastq_record_parity(v: &mut Validator) {
    v.section("═══ S2: FASTQ for_each_record ↔ FastqIter record parity ═══");

    let path = temp_path("record_parity.fastq");
    write_synthetic_fastq(&path);

    let batch_records: Vec<_> = fastq::FastqIter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");

    let mut stream_ids = Vec::new();
    let mut stream_seqs: Vec<Vec<u8>> = Vec::new();
    let mut stream_quals: Vec<Vec<u8>> = Vec::new();

    fastq::for_each_record(&path, |rec: FastqRefRecord<'_>| {
        stream_ids.push(rec.id.to_string());
        stream_seqs.push(rec.sequence.to_vec());
        stream_quals.push(rec.quality.to_vec());
        Ok(())
    })
    .or_exit("unexpected error");

    v.check_count(
        "record count batch==stream",
        batch_records.len(),
        stream_ids.len(),
    );

    let mut all_ids_match = true;
    let mut all_seqs_match = true;
    let mut all_quals_match = true;

    for (i, batch) in batch_records.iter().enumerate() {
        if batch.id != stream_ids[i] {
            all_ids_match = false;
        }
        if batch.sequence != stream_seqs[i] {
            all_seqs_match = false;
        }
        if batch.quality != stream_quals[i] {
            all_quals_match = false;
        }
    }

    v.check_pass("all record IDs match", all_ids_match);
    v.check_pass("all sequences match", all_seqs_match);
    v.check_pass("all quality scores match", all_quals_match);

    let _ = std::fs::remove_file(&path);
}

// ── S3: Multi-byte UTF-8 header safety ──────────────────────────────────────

fn validate_utf8_header_safety(v: &mut Validator) {
    v.section("═══ S3: Multi-byte UTF-8 header safety ═══");

    let path = temp_path("utf8_headers.fastq");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        writeln!(f, "@read_with_émojis_and_über_chars description here").or_exit("unexpected error");
        writeln!(f, "ATGCATGC").or_exit("unexpected error");
        writeln!(f, "+").or_exit("unexpected error");
        writeln!(f, "IIIIIIII").or_exit("unexpected error");
        writeln!(f, "@read_with_日本語_header").or_exit("unexpected error");
        writeln!(f, "GCGCGCGC").or_exit("unexpected error");
        writeln!(f, "+").or_exit("unexpected error");
        writeln!(f, "FFFFFFFF").or_exit("unexpected error");
    }

    let result = fastq::FastqIter::open(&path).and_then(|i| i.collect::<Result<Vec<_>, _>>());
    v.check_pass(
        "multi-byte UTF-8 headers parse without panic",
        result.is_ok(),
    );

    if let Ok(records) = &result {
        v.check_count("parsed 2 records with UTF-8 headers", records.len(), 2);
    } else {
        v.check_count("parsed 2 records with UTF-8 headers", 0, 2);
    }

    let malformed_path = temp_path("malformed_header.fastq");
    {
        let mut f = std::fs::File::create(&malformed_path).or_exit("unexpected error");
        writeln!(f, "NOT_A_HEADER_LINE").or_exit("unexpected error");
        writeln!(f, "ATGC").or_exit("unexpected error");
        writeln!(f, "+").or_exit("unexpected error");
        writeln!(f, "IIII").or_exit("unexpected error");
    }

    let malformed_result =
        fastq::FastqIter::open(&malformed_path).and_then(|i| i.collect::<Result<Vec<_>, _>>());
    v.check_pass(
        "malformed header returns Err (not panic)",
        malformed_result.is_err(),
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&malformed_path);
}

// ── S4: Nanopore bulk read parity (bytemuck) ────────────────────────────────

fn validate_nanopore_bulk_parity(v: &mut Validator) {
    v.section("═══ S4: Nanopore bulk read parity (bytemuck cast_slice) ═══");

    let sig = SyntheticSignalGenerator::new(42);

    let reads = sig.generate_batch(50, 8000, 4000.0);

    let path = temp_path("bulk_parity.nrs");
    nanopore::write_nrs(&path, &reads).or_exit("unexpected error");

    let loaded: Vec<_> = NanoporeIter::open(&path)
        .or_exit("unexpected error")
        .map(|r| r.or_exit("unexpected error"))
        .collect();

    v.check_count("read count round-trip", loaded.len(), 50);

    let mut signal_exact = true;
    let mut cal_exact = true;
    let mut id_exact = true;
    let mut rate_exact = true;

    for (orig, rt) in reads.iter().zip(loaded.iter()) {
        if orig.read_id != rt.read_id {
            id_exact = false;
        }
        if orig.signal != rt.signal {
            signal_exact = false;
        }
        if (orig.calibration_offset - rt.calibration_offset).abs()
            > tolerances::NANOPORE_SIGNAL_ROUNDTRIP
        {
            cal_exact = false;
        }
        if (orig.calibration_scale - rt.calibration_scale).abs()
            > tolerances::NANOPORE_SIGNAL_ROUNDTRIP
        {
            cal_exact = false;
        }
        if (orig.sample_rate - rt.sample_rate).abs() > tolerances::NANOPORE_SIGNAL_ROUNDTRIP {
            rate_exact = false;
        }
    }

    v.check_pass("read_id bit-exact (50 reads)", id_exact);
    v.check_pass(
        "signal bit-exact via bytemuck bulk read (50×8000 samples)",
        signal_exact,
    );
    v.check_pass("calibration parameters round-trip", cal_exact);
    v.check_pass("sample_rate round-trip", rate_exact);

    let orig_cal = reads[0].calibrated_signal();
    let rt_cal = loaded[0].calibrated_signal();
    v.check_count("calibrated signal length", rt_cal.len(), orig_cal.len());

    let cal_max_err = orig_cal
        .iter()
        .zip(rt_cal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check(
        "calibrated signal max error",
        cal_max_err,
        0.0,
        tolerances::NANOPORE_SIGNAL_ROUNDTRIP,
    );

    let stats_orig = reads[0].signal_stats();
    let stats_rt = loaded[0].signal_stats();
    v.check(
        "signal stats mean parity",
        stats_rt.mean,
        stats_orig.mean,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "signal stats std_dev parity",
        stats_rt.std_dev,
        stats_orig.std_dev,
        tolerances::ANALYTICAL_F64,
    );

    let _ = std::fs::remove_file(&path);
}

// ── S5: MS2 streaming ↔ batch parity ────────────────────────────────────────

fn validate_ms2_streaming_parity(v: &mut Validator) {
    v.section("═══ S5: MS2 streaming ↔ batch parity ═══");

    let path = temp_path("stream_parity.ms2");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        writeln!(f, "H\tCreatedBy\twetSpring Exp209").or_exit("unexpected error");
        writeln!(f, "H\tExtractor\tsynthetic").or_exit("unexpected error");
        writeln!(f, "S\t1\t1\t500.250").or_exit("unexpected error");
        writeln!(f, "I\tRTime\t1.5").or_exit("unexpected error");
        writeln!(f, "I\tBPI\t10000.0").or_exit("unexpected error");
        writeln!(f, "I\tTIC\t50000.0").or_exit("unexpected error");
        writeln!(f, "Z\t2\t999.493").or_exit("unexpected error");
        writeln!(f, "100.0\t1000.0").or_exit("unexpected error");
        writeln!(f, "200.0\t5000.0").or_exit("unexpected error");
        writeln!(f, "300.0\t3000.0").or_exit("unexpected error");
        writeln!(f, "S\t2\t2\t750.500").or_exit("unexpected error");
        writeln!(f, "I\tRTime\t3.2").or_exit("unexpected error");
        writeln!(f, "I\tBPI\t8000.0").or_exit("unexpected error");
        writeln!(f, "I\tTIC\t35000.0").or_exit("unexpected error");
        writeln!(f, "Z\t3\t1124.730").or_exit("unexpected error");
        writeln!(f, "150.0\t2000.0").or_exit("unexpected error");
        writeln!(f, "250.0\t8000.0").or_exit("unexpected error");
        writeln!(f, "350.0\t1500.0").or_exit("unexpected error");
        writeln!(f, "450.0\t500.0").or_exit("unexpected error");
    }

    let batch: Vec<_> = ms2::Ms2Iter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");

    let mut stream_count = 0_usize;
    let mut stream_mz_sums = Vec::new();
    ms2::for_each_spectrum(&path, |spec| {
        let mz_sum: f64 = spec.mz_array.iter().sum();
        stream_mz_sums.push(mz_sum);
        stream_count += 1;
        Ok(())
    })
    .or_exit("unexpected error");

    v.check_count(
        "MS2 spectrum count batch==stream",
        batch.len(),
        stream_count,
    );
    v.check_count("MS2 spectrum count == 2", batch.len(), 2);

    for (i, spec) in batch.iter().enumerate() {
        let batch_mz_sum: f64 = spec.mz_array.iter().sum();
        v.check(
            &format!("MS2 spec{i} mz_sum batch==stream"),
            batch_mz_sum,
            stream_mz_sums[i],
            tolerances::ANALYTICAL_F64,
        );
    }

    v.check_count("MS2 spec0 peaks", batch[0].mz_array.len(), 3);
    v.check_count("MS2 spec1 peaks", batch[1].mz_array.len(), 4);
    v.check(
        "MS2 spec0 precursor_mz",
        batch[0].precursor_mz,
        500.25,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "MS2 spec1 precursor_mz",
        batch[1].precursor_mz,
        750.5,
        tolerances::ANALYTICAL_F64,
    );

    let batch_stats = ms2::compute_stats(&batch);
    let stream_stats = ms2::stats_from_file(&path).or_exit("unexpected error");
    v.check_count(
        "MS2 stats num_spectra batch==stream",
        batch_stats.num_spectra,
        stream_stats.num_spectra,
    );
    v.check_count(
        "MS2 stats total_peaks batch==stream",
        batch_stats.total_peaks,
        stream_stats.total_peaks,
    );

    let _ = std::fs::remove_file(&path);
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let mut v = Validator::new("Exp209: Streaming I/O Parity Validation (V66 post-audit)");

    validate_fastq_stats_parity(&mut v);
    validate_fastq_record_parity(&mut v);
    validate_utf8_header_safety(&mut v);
    validate_nanopore_bulk_parity(&mut v);
    validate_ms2_streaming_parity(&mut v);

    v.finish();
}
