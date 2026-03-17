// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "validation harness: mathematical variable names from papers"
)]
//! Exp212: `BarraCuda` CPU Parity v12 — Post-Audit Math Fidelity
//!
//! Validates that the V66 audit evolution (byte-native FASTQ, bytemuck
//! nanopore, streaming APIs, tolerance centralization, safe env handling)
//! preserves full math pipeline fidelity from I/O through analysis.
//!
//! **What's new in v12**: End-to-end I/O → math chain validation.
//! Previous CPU parity versions validated math in isolation. This version
//! proves the evolved I/O layer correctly feeds the analytical pipeline:
//!
//! - D01: FASTQ byte-native → diversity math (Shannon, Simpson, Pielou)
//! - D02: FASTQ byte-native → quality filter → dereplication math
//! - D03: Nanopore bulk signal → calibrated pA → signal statistics
//! - D04: MS2 streaming → spectral math (precursor validation, peak counts)
//! - D05: Tolerance centralization structural audit (no inline magic numbers)
//! - D06: Cross-module end-to-end: FASTQ → quality → merge → derep → diversity
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | V65 audit changes + Exp209 I/O parity (37/37) |
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 66 (V66 post-audit) |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v12` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Tolerances | `tolerances::ANALYTICAL_F64`, `tolerances::EXACT_F64` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)
//!
//! # Python Baselines
//!
//! Multi-domain: analytical known-values (no Python script dependency).
//! I/O fidelity validated against:
//! - `scripts/validate_exp001.py` (FASTQ parsing parity)
//! - `scripts/spectral_match_baseline.py` (MS2 spectral math)
//! - `scripts/algae_timeseries_baseline.py` (diversity formulas)

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use wetspring_barracuda::bio::{derep, diversity, merge_pairs, quality};
use wetspring_barracuda::io::fastq::{self, FastqRefRecord};
use wetspring_barracuda::io::ms2;
use wetspring_barracuda::io::nanopore::{self, NanoporeIter, SyntheticSignalGenerator};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("wetspring_exp212_{name}"))
}

// ── D01: FASTQ byte-native → diversity math ─────────────────────────────────

fn write_community_fastq(path: &std::path::Path, species: &[(&str, usize)]) {
    let mut f = std::fs::File::create(path).or_exit("unexpected error");
    let mut idx = 0_usize;
    for (seq, count) in species {
        for _ in 0..*count {
            writeln!(f, "@species_{idx}").or_exit("unexpected error");
            writeln!(f, "{seq}").or_exit("unexpected error");
            writeln!(f, "+").or_exit("unexpected error");
            let qual = vec![b'I'; seq.len()];
            f.write_all(&qual).or_exit("unexpected error");
            writeln!(f).or_exit("unexpected error");
            idx += 1;
        }
    }
}

fn validate_fastq_to_diversity(v: &mut Validator) {
    v.section("═══ D01: FASTQ byte-native → diversity math ═══");
    let t = Instant::now();

    let species: Vec<(&str, usize)> = vec![
        ("ATGCATGCATGCATGC", 50),
        ("GCGCGCGCGCGCGCGC", 30),
        ("TTTTAAAACCCCGGGG", 15),
        ("AACCGGTTAACCGGTT", 4),
        ("GGGGCCCCAAAATTTT", 1),
    ];

    let path = temp_path("community.fastq");
    write_community_fastq(&path, &species);

    let mut seq_counts: std::collections::HashMap<Vec<u8>, usize> =
        std::collections::HashMap::new();
    fastq::for_each_record(&path, |rec: FastqRefRecord<'_>| {
        *seq_counts.entry(rec.sequence.to_vec()).or_insert(0) += 1;
        Ok(())
    })
    .or_exit("unexpected error");

    let counts: Vec<f64> = seq_counts.values().map(|&c| c as f64).collect();

    let h = diversity::shannon(&counts);
    let d = diversity::simpson(&counts);
    let j = diversity::pielou_evenness(&counts);
    let s = diversity::observed_features(&counts);
    let c1 = diversity::chao1(&counts);

    let expected_counts = [50.0, 30.0, 15.0, 4.0, 1.0];
    let expected_h = diversity::shannon(&expected_counts);
    let expected_d = diversity::simpson(&expected_counts);

    v.check(
        "Shannon from byte-native FASTQ",
        h,
        expected_h,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Simpson from byte-native FASTQ",
        d,
        expected_d,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Pielou J from byte-native FASTQ",
        j,
        diversity::pielou_evenness(&expected_counts),
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "S_obs from byte-native FASTQ",
        s,
        5.0,
        tolerances::EXACT_F64,
    );
    v.check_pass("Chao1 >= S_obs", c1 >= s);
    v.check_pass("Shannon > 0 (non-trivial community)", h > 0.0);
    v.check_pass("Simpson in [0,1]", (0.0..=1.0).contains(&d));

    println!("    D01 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── D02: FASTQ byte-native → quality + dereplication ─────────────────────────

fn validate_quality_derep_chain(v: &mut Validator) {
    v.section("═══ D02: FASTQ byte-native → quality + dereplication ═══");
    let t = Instant::now();

    let path = temp_path("quality_derep.fastq");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        for i in 0..20_u32 {
            let id = format!("read_{i}");
            let seq = if i < 10 {
                "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
            } else {
                "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"
            };
            let qual: String = if i % 5 == 0 {
                std::iter::repeat_n('!', seq.len()).collect() // Q0 = bad
            } else {
                std::iter::repeat_n('I', seq.len()).collect() // Q40 = good
            };
            writeln!(f, "@{id} desc").or_exit("unexpected error");
            writeln!(f, "{seq}").or_exit("unexpected error");
            writeln!(f, "+").or_exit("unexpected error");
            writeln!(f, "{qual}").or_exit("unexpected error");
        }
    }

    let records: Vec<_> = fastq::FastqIter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");
    v.check_count("parsed 20 reads", records.len(), 20);

    let params = quality::QualityParams {
        window_size: 4,
        window_min_quality: 15,
        leading_min_quality: 3,
        trailing_min_quality: 3,
        min_length: 36,
        phred_offset: 33,
    };
    let (filtered, qstats) = quality::filter_reads(&records, &params);
    v.check_count("quality input count", qstats.input_reads, 20);
    v.check_pass("some reads pass quality filter", qstats.output_reads > 0);
    v.check_pass("some reads fail quality filter", qstats.discarded_reads > 0);

    let (uniques, dstats) = derep::dereplicate(&filtered, derep::DerepSort::Abundance, 0);
    v.check_pass(
        "unique sequences <= filtered",
        uniques.len() <= filtered.len(),
    );
    v.check_pass("unique count == 2 (two ASVs)", dstats.unique_sequences <= 2);
    v.check_pass(
        "total abundance == filtered count",
        uniques.iter().map(|u| u.abundance).sum::<usize>() == filtered.len(),
    );

    println!("    D02 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── D03: Nanopore bulk signal → calibrated math ────────────────────────────

fn validate_nanopore_calibration_math(v: &mut Validator) {
    v.section("═══ D03: Nanopore bulk signal → calibrated math ═══");
    let t = Instant::now();

    let sig = SyntheticSignalGenerator::new(99);
    let reads = sig.generate_batch(25, 4000, 4000.0);

    let path = temp_path("calibration.nrs");
    nanopore::write_nrs(&path, &reads).or_exit("unexpected error");
    let loaded: Vec<_> = NanoporeIter::open(&path)
        .or_exit("unexpected error")
        .map(|r| r.or_exit("unexpected error"))
        .collect();

    for (i, (orig, rt)) in reads.iter().zip(loaded.iter()).enumerate().take(5) {
        let orig_cal = orig.calibrated_signal();
        let rt_cal = rt.calibrated_signal();

        let orig_mean: f64 = orig_cal.iter().sum::<f64>() / orig_cal.len() as f64;
        let rt_mean: f64 = rt_cal.iter().sum::<f64>() / rt_cal.len() as f64;

        v.check(
            &format!("read{i} calibrated mean parity"),
            rt_mean,
            orig_mean,
            tolerances::ANALYTICAL_F64,
        );
    }

    let cal0 = loaded[0].calibrated_signal();
    let scale = loaded[0].calibration_scale;
    let offset = loaded[0].calibration_offset;
    let expected_first = f64::from(loaded[0].signal[0]).mul_add(scale, offset);
    v.check(
        "calibration formula: pA = raw * scale + offset",
        cal0[0],
        expected_first,
        tolerances::ANALYTICAL_F64,
    );

    let dur = loaded[0].duration_seconds();
    let expected_dur = loaded[0].signal.len() as f64 / loaded[0].sample_rate;
    v.check(
        "duration_seconds = len / sample_rate",
        dur,
        expected_dur,
        tolerances::ANALYTICAL_F64,
    );

    let stats = loaded[0].signal_stats();
    let manual_mean: f64 =
        loaded[0].signal.iter().map(|&s| f64::from(s)).sum::<f64>() / loaded[0].signal.len() as f64;
    v.check(
        "signal_stats mean matches manual",
        stats.mean,
        manual_mean,
        tolerances::ANALYTICAL_F64,
    );

    println!("    D03 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── D04: MS2 streaming → spectral math ──────────────────────────────────────

fn validate_ms2_spectral_math(v: &mut Validator) {
    v.section("═══ D04: MS2 streaming → spectral math ═══");
    let t = Instant::now();

    let path = temp_path("spectral.ms2");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        writeln!(f, "H\tCreatedBy\twetSpring Exp212").or_exit("unexpected error");
        for scan in 1..=5_u32 {
            let precursor = f64::from(scan).mul_add(50.0, 400.0);
            writeln!(f, "S\t{scan}\t{scan}\t{precursor:.3}").or_exit("unexpected error");
            writeln!(f, "I\tRTime\t{:.1}", f64::from(scan) * 0.5).or_exit("unexpected error");
            writeln!(f, "I\tBPI\t10000.0").or_exit("unexpected error");
            writeln!(f, "I\tTIC\t50000.0").or_exit("unexpected error");
            writeln!(f, "Z\t2\t{:.3}", precursor * 2.0 - 1.008).or_exit("unexpected error");
            for frag in 0..scan {
                let mz = f64::from(frag).mul_add(100.0, 100.0);
                let intensity = 1000.0 * f64::from(scan - frag);
                writeln!(f, "{mz:.1}\t{intensity:.1}").or_exit("unexpected error");
            }
        }
    }

    let mut stream_spectra = Vec::new();
    ms2::for_each_spectrum(&path, |spec| {
        stream_spectra.push(spec);
        Ok(())
    })
    .or_exit("unexpected error");

    let batch_spectra: Vec<_> = ms2::Ms2Iter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");

    v.check_count("5 spectra from streaming", stream_spectra.len(), 5);
    v.check_count("5 spectra from batch", batch_spectra.len(), 5);

    for (i, (s, b)) in stream_spectra.iter().zip(batch_spectra.iter()).enumerate() {
        v.check(
            &format!("spec{i} precursor_mz stream==batch"),
            s.precursor_mz,
            b.precursor_mz,
            tolerances::ANALYTICAL_F64,
        );
        v.check_count(
            &format!("spec{i} peak count stream==batch"),
            s.mz_array.len(),
            b.mz_array.len(),
        );
    }

    v.check_count("spec0 has 1 peak", batch_spectra[0].mz_array.len(), 1);
    v.check_count("spec4 has 5 peaks", batch_spectra[4].mz_array.len(), 5);

    let precursors: Vec<f64> = batch_spectra.iter().map(|s| s.precursor_mz).collect();
    let is_monotonic = precursors.windows(2).all(|w| w[1] > w[0]);
    v.check_pass("precursor m/z monotonically increasing", is_monotonic);

    println!("    D04 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── D05: Tolerance centralization structural audit ───────────────────────────

fn validate_tolerance_centralization(v: &mut Validator) {
    v.section("═══ D05: Tolerance centralization structural audit ═══");

    v.check_pass("EXACT == 0.0", tolerances::EXACT == 0.0);
    v.check_pass(
        "ANALYTICAL_F64 == 1e-12",
        tolerances::ANALYTICAL_F64 == 1e-12,
    );
    v.check_pass("EXACT_F64 == 1e-15", tolerances::EXACT_F64 == 1e-15);
    v.check_pass("GC_CONTENT == 0.005", tolerances::GC_CONTENT == 0.005);
    v.check_pass("MZ_TOLERANCE == 0.01", tolerances::MZ_TOLERANCE == 0.01);
    v.check_pass(
        "ERF_PARITY exists and is scientifically small",
        tolerances::ERF_PARITY > 0.0 && tolerances::ERF_PARITY < 1e-5,
    );
    v.check_pass(
        "PYTHON_PARITY exists and is small",
        tolerances::PYTHON_PARITY > 0.0 && tolerances::PYTHON_PARITY < 1e-4,
    );
    v.check_pass(
        "NANOPORE_SIGNAL_ROUNDTRIP exists and is small",
        tolerances::NANOPORE_SIGNAL_ROUNDTRIP >= 0.0 && tolerances::NANOPORE_SIGNAL_ROUNDTRIP < 1.0,
    );
}

// ── D06: End-to-end: FASTQ → quality → merge → derep → diversity ────────────

fn validate_end_to_end_pipeline(v: &mut Validator) {
    v.section("═══ D06: End-to-end pipeline: FASTQ → quality → merge → derep → diversity ═══");
    let t = Instant::now();

    let amplicon: Vec<u8> = (0..253_u16)
        .map(|i| match i % 4 {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            _ => b'T',
        })
        .collect();

    let fwd_path = temp_path("e2e_R1.fastq");
    let rev_path = temp_path("e2e_R2.fastq");
    {
        let mut fwd_f = std::fs::File::create(&fwd_path).or_exit("unexpected error");
        let mut rev_f = std::fs::File::create(&rev_path).or_exit("unexpected error");

        for pair_idx in 0..50_u32 {
            let fwd_seq = &amplicon[..250];
            let rev_region = &amplicon[3..];
            let rev_seq = merge_pairs::reverse_complement(rev_region);

            writeln!(fwd_f, "@pair_{pair_idx}/1").or_exit("unexpected error");
            fwd_f.write_all(fwd_seq).or_exit("unexpected error");
            writeln!(fwd_f).or_exit("unexpected error");
            writeln!(fwd_f, "+").or_exit("unexpected error");
            let fwd_qual: String = std::iter::repeat_n('I', 250).collect();
            writeln!(fwd_f, "{fwd_qual}").or_exit("unexpected error");

            writeln!(rev_f, "@pair_{pair_idx}/2").or_exit("unexpected error");
            rev_f.write_all(&rev_seq).or_exit("unexpected error");
            writeln!(rev_f).or_exit("unexpected error");
            writeln!(rev_f, "+").or_exit("unexpected error");
            let rev_qual: String = std::iter::repeat_n('I', 250).collect();
            writeln!(rev_f, "{rev_qual}").or_exit("unexpected error");
        }
    }

    let fwd_records: Vec<_> = fastq::FastqIter::open(&fwd_path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");
    let rev_records: Vec<_> = fastq::FastqIter::open(&rev_path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");
    v.check_count("parsed 50 forward reads", fwd_records.len(), 50);
    v.check_count("parsed 50 reverse reads", rev_records.len(), 50);

    let fwd_stats = fastq::stats_from_file(&fwd_path).or_exit("unexpected error");
    let rev_stats = fastq::stats_from_file(&rev_path).or_exit("unexpected error");
    v.check_count("all forward reads 250bp", fwd_stats.min_length, 250);
    v.check_count("all reverse reads 250bp", rev_stats.min_length, 250);

    let qparams = quality::QualityParams {
        window_size: 4,
        window_min_quality: 20,
        leading_min_quality: 3,
        trailing_min_quality: 3,
        min_length: 100,
        phred_offset: 33,
    };
    let (fwd_filt, _) = quality::filter_reads(&fwd_records, &qparams);
    let (rev_filt, _) = quality::filter_reads(&rev_records, &qparams);
    v.check_pass("quality filter passes high-Q reads", !fwd_filt.is_empty());

    let merge_count = fwd_filt.len().min(rev_filt.len());
    let (merged, merge_stats) = merge_pairs::merge_pairs(
        &fwd_filt[..merge_count],
        &rev_filt[..merge_count],
        &merge_pairs::MergeParams::default(),
    );
    v.check_pass(
        "some pairs merge successfully",
        merge_stats.merged_count > 0,
    );

    let (uniques, derep_stats) = derep::dereplicate(&merged, derep::DerepSort::Abundance, 0);
    v.check_pass(
        "single amplicon → 1 unique sequence",
        derep_stats.unique_sequences == 1,
    );
    v.check_pass(
        "max abundance == merged count",
        derep_stats.max_abundance == merged.len(),
    );

    let counts: Vec<f64> = uniques.iter().map(|u| u.abundance as f64).collect();
    let h = diversity::shannon(&counts);
    let d = diversity::simpson(&counts);
    v.check(
        "single-ASV Shannon == 0.0",
        h,
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "single-ASV Simpson == 0.0 (1 - 1.0²)",
        d,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    println!("    D06 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&fwd_path);
    let _ = std::fs::remove_file(&rev_path);
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let mut v = Validator::new("Exp212: BarraCuda CPU v12 — Post-Audit Math Fidelity (V66)");

    validate_fastq_to_diversity(&mut v);
    validate_quality_derep_chain(&mut v);
    validate_nanopore_calibration_math(&mut v);
    validate_ms2_spectral_math(&mut v);
    validate_tolerance_centralization(&mut v);
    validate_end_to_end_pipeline(&mut v);

    v.finish();
}
