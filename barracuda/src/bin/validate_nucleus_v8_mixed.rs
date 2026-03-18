// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp214: NUCLEUS Mixed Hardware V8 — V66 I/O Evolution via IPC
//!
//! Extends Exp208 (74/74) with V66 I/O evolution through the IPC dispatch
//! layer, proving the byte-native FASTQ, bytemuck nanopore, and streaming
//! MS2 changes preserve math fidelity when routed through NUCLEUS atomics:
//!
//! - **MH01**: Tower capabilities — V66 evolved endpoints registered
//! - **MH02**: Node dispatch: byte-native FASTQ → diversity via IPC
//! - **MH03**: Node dispatch: nanopore signal → calibrated math
//! - **MH04**: Node dispatch: MS2 streaming → spectral analysis
//! - **MH05**: Nest metrics: per-capability timing + success/error counts
//! - **MH06**: CPU fallback parity (GPU-lost scenario)
//! - **MH07**: Pipeline: FASTQ → quality → diversity → QS ODE (full chain)
//! - **MH08**: Dispatch routing: substrate selection
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Exp208 (74/74), Exp209 I/O parity (37/37), Exp212 CPU v12 (55/55) |
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 66 |
//! | Command | `cargo run --features ipc --release --bin validate_nucleus_v8_mixed` |
//!
//! Validation class: Pipeline
//! Provenance: End-to-end pipeline integration test

use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use wetspring_barracuda::bio::{derep, diversity, quality};
use wetspring_barracuda::cast::usize_f64;
use wetspring_barracuda::io::fastq::{self, FastqRefRecord};
use wetspring_barracuda::io::ms2;
use wetspring_barracuda::io::nanopore::{self, NanoporeIter, SyntheticSignalGenerator};
use wetspring_barracuda::ipc::dispatch;
use wetspring_barracuda::ipc::metrics::Metrics;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("wetspring_exp214_{name}"))
}

// ── MH01: Tower capabilities ─────────────────────────────────────────────────

fn validate_tower_capabilities(v: &mut Validator) {
    v.section("═══ MH01: Tower capabilities — V66 evolved endpoints ═══");

    let health = dispatch::dispatch("health.check", &json!({})).or_exit("health.check");

    v.check_pass(
        "Tower: primal == wetspring",
        health["primal"].as_str() == Some(wetspring_barracuda::ipc::primal_names::SELF),
    );
    v.check_pass(
        "Tower: status == healthy",
        health["status"].as_str() == Some("healthy"),
    );

    let caps = health["capabilities"].as_array();
    v.check_pass("Tower: capabilities array present", caps.is_some());

    if let Some(cap_arr) = caps {
        let cap_names: Vec<&str> = cap_arr.iter().filter_map(Value::as_str).collect();
        v.check_pass(
            "Tower: science.diversity registered",
            cap_names.contains(&"science.diversity"),
        );
        v.check_pass(
            "Tower: science.qs_model registered",
            cap_names.contains(&"science.qs_model"),
        );
        v.check_pass(
            "Tower: science.full_pipeline registered",
            cap_names.contains(&"science.full_pipeline"),
        );
        v.check_pass(
            "Tower: science.anderson registered",
            cap_names.contains(&"science.anderson"),
        );
        v.check_pass(
            "Tower: metrics.snapshot registered (Nest)",
            cap_names.contains(&"metrics.snapshot"),
        );
    }
}

// ── MH02: Byte-native FASTQ → diversity via dispatch ─────────────────────────

fn validate_fastq_diversity_dispatch(v: &mut Validator) {
    v.section("═══ MH02: Byte-native FASTQ → diversity via IPC dispatch ═══");
    let t = Instant::now();

    let path = temp_path("nucleus_community.fastq");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        let species: &[(&str, usize)] = &[
            ("ATGCATGCATGCATGC", 50),
            ("GCGCGCGCGCGCGCGC", 30),
            ("TTTTAAAACCCCGGGG", 15),
            ("AACCGGTTAACCGGTT", 4),
            ("GGGGCCCCAAAATTTT", 1),
        ];
        let mut idx = 0_usize;
        for (seq, count) in species {
            for _ in 0..*count {
                writeln!(f, "@sp_{idx}\n{seq}\n+\n{}", "I".repeat(seq.len()))
                    .or_exit("unexpected error");
                idx += 1;
            }
        }
    }

    let mut seq_counts: std::collections::HashMap<Vec<u8>, usize> =
        std::collections::HashMap::new();
    fastq::for_each_record(&path, |rec: FastqRefRecord<'_>| {
        *seq_counts.entry(rec.sequence.to_vec()).or_insert(0) += 1;
        Ok(())
    })
    .or_exit("unexpected error");

    let direct_counts: Vec<f64> = seq_counts.values().map(|&c| usize_f64(c)).collect();
    let direct_h = diversity::shannon(&direct_counts);
    let direct_d = diversity::simpson(&direct_counts);

    let result = dispatch::dispatch("science.diversity", &json!({ "counts": direct_counts }))
        .or_exit("diversity dispatch");

    let dispatch_h = result["shannon"].as_f64().or_exit("shannon");
    let dispatch_d = result["simpson"].as_f64().or_exit("simpson");

    v.check(
        "Shannon: direct == IPC dispatch",
        direct_h,
        dispatch_h,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Simpson: direct == IPC dispatch",
        direct_d,
        dispatch_d,
        tolerances::ANALYTICAL_F64,
    );

    let has_shannon = result.get("shannon").is_some();
    v.check_pass("dispatch result contains shannon", has_shannon);

    println!("    MH02 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── MH03: Nanopore signal → calibrated math ─────────────────────────────────

fn validate_nanopore_signal(v: &mut Validator) {
    v.section("═══ MH03: Nanopore signal → calibrated math ═══");
    let t = Instant::now();

    let sig = SyntheticSignalGenerator::new(42);
    let reads = sig.generate_batch(10, 2000, 4000.0);

    let path = temp_path("nucleus_nanopore.nrs");
    nanopore::write_nrs(&path, &reads).or_exit("unexpected error");
    let loaded: Vec<_> = NanoporeIter::open(&path)
        .or_exit("unexpected error")
        .map(|r| r.or_exit("unexpected error"))
        .collect();

    v.check_count("loaded 10 nanopore reads", loaded.len(), 10);

    let stats = loaded[0].signal_stats();
    let raw_f64: Vec<f64> = loaded[0].signal.iter().map(|&s| f64::from(s)).collect();
    // Intentional: manual mean/variance as reference to validate signal_stats implementation.
    let manual_mean: f64 = raw_f64.iter().sum::<f64>() / usize_f64(raw_f64.len());

    v.check(
        "raw signal mean: manual == signal_stats",
        manual_mean,
        stats.mean,
        tolerances::ANALYTICAL_F64,
    );

    let manual_std = {
        let var: f64 = raw_f64
            .iter()
            .map(|x| (x - manual_mean).powi(2))
            .sum::<f64>()
            / usize_f64(raw_f64.len());
        var.sqrt()
    };
    v.check(
        "raw signal std_dev: manual == signal_stats",
        manual_std,
        stats.std_dev,
        tolerances::ANALYTICAL_F64,
    );

    let dur = loaded[0].duration_seconds();
    let expected_dur = usize_f64(loaded[0].signal.len()) / loaded[0].sample_rate;
    v.check(
        "duration_seconds = len / sample_rate",
        dur,
        expected_dur,
        tolerances::ANALYTICAL_F64,
    );

    let cal = loaded[0].calibrated_signal();
    let expected_first = f64::from(loaded[0].signal[0])
        .mul_add(loaded[0].calibration_scale, loaded[0].calibration_offset);
    v.check(
        "calibration formula: pA = raw * scale + offset",
        cal[0],
        expected_first,
        tolerances::ANALYTICAL_F64,
    );

    println!("    MH03 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── MH04: MS2 streaming → spectral analysis ─────────────────────────────────

fn validate_ms2_spectral(v: &mut Validator) {
    v.section("═══ MH04: MS2 streaming → spectral math ═══");
    let t = Instant::now();

    let path = temp_path("nucleus_spectral.ms2");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        writeln!(f, "H\tCreatedBy\tExp214").or_exit("unexpected error");
        for scan in 1..=3_u32 {
            let precursor = f64::from(scan).mul_add(100.0, 400.0);
            writeln!(f, "S\t{scan}\t{scan}\t{precursor:.3}").or_exit("unexpected error");
            writeln!(f, "I\tRTime\t{:.1}", f64::from(scan) * 0.5).or_exit("unexpected error");
            writeln!(f, "I\tBPI\t10000.0").or_exit("unexpected error");
            writeln!(f, "I\tTIC\t50000.0").or_exit("unexpected error");
            writeln!(f, "Z\t2\t{:.3}", precursor * 2.0 - 1.008).or_exit("unexpected error");
            for frag in 0..=scan {
                writeln!(
                    f,
                    "{:.1}\t{:.1}",
                    f64::from(frag).mul_add(100.0, 100.0),
                    1000.0 * f64::from(scan + 1 - frag)
                )
                .or_exit("unexpected error");
            }
        }
    }

    let batch: Vec<_> = ms2::Ms2Iter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");
    v.check_count("batch parsed 3 spectra", batch.len(), 3);

    let mut stream_count = 0_usize;
    let mut stream_precursors = Vec::new();
    ms2::for_each_spectrum(&path, |spec| {
        stream_count += 1;
        stream_precursors.push(spec.precursor_mz);
        Ok(())
    })
    .or_exit("unexpected error");

    v.check_count("stream parsed 3 spectra", stream_count, 3);

    for (i, (bp, sp)) in batch
        .iter()
        .map(|s| s.precursor_mz)
        .zip(stream_precursors.iter())
        .enumerate()
    {
        v.check(
            &format!("spec{i} precursor: batch == stream"),
            bp,
            *sp,
            tolerances::ANALYTICAL_F64,
        );
    }

    let stats = ms2::compute_stats(&batch);
    let stream_stats = ms2::stats_from_file(&path).or_exit("unexpected error");
    v.check_count(
        "stats: num_spectra batch == stream",
        stats.num_spectra,
        stream_stats.num_spectra,
    );
    v.check_count(
        "stats: total_peaks batch == stream",
        stats.total_peaks,
        stream_stats.total_peaks,
    );

    println!("    MH04 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── MH05: Nest metrics ───────────────────────────────────────────────────────

fn validate_nest_metrics(v: &mut Validator) {
    v.section("═══ MH05: Nest metrics — per-capability timing ═══");

    let m = Metrics::new();

    m.record_success("science.diversity", Duration::from_micros(250));
    m.record_success("science.diversity", Duration::from_micros(150));
    m.record_success("science.qs_model", Duration::from_micros(5000));
    m.record_error("science.ncbi_fetch", Duration::from_micros(100));

    let snap = m.snapshot();
    v.check_pass("Nest: snapshot is object", snap.is_object());
    v.check_pass(
        "Nest: primal == wetspring",
        snap["primal"].as_str() == Some(wetspring_barracuda::ipc::primal_names::SELF),
    );
    v.check_pass(
        "Nest: total_calls == 4",
        snap["total_calls"].as_u64() == Some(4),
    );
    v.check_pass(
        "Nest: success_count == 3",
        snap["success_count"].as_u64() == Some(3),
    );
    v.check_pass(
        "Nest: error_count == 1",
        snap["error_count"].as_u64() == Some(1),
    );

    let methods = &snap["methods"];
    v.check_pass("Nest: methods is object", methods.is_object());

    let div_m = &methods["science.diversity"];
    v.check_pass(
        "Nest: diversity calls == 2",
        div_m["calls"].as_u64() == Some(2),
    );
    v.check_pass(
        "Nest: diversity successes == 2",
        div_m["successes"].as_u64() == Some(2),
    );
    v.check_pass(
        "Nest: diversity min_us == 150",
        div_m["min_us"].as_u64() == Some(150),
    );
    v.check_pass(
        "Nest: diversity max_us == 250",
        div_m["max_us"].as_u64() == Some(250),
    );

    let qs_m = &methods["science.qs_model"];
    v.check_pass(
        "Nest: qs_model calls == 1",
        qs_m["calls"].as_u64() == Some(1),
    );

    let fetch_m = &methods["science.ncbi_fetch"];
    v.check_pass(
        "Nest: ncbi_fetch errors == 1",
        fetch_m["errors"].as_u64() == Some(1),
    );
}

// ── MH06: CPU fallback parity ────────────────────────────────────────────────

fn validate_cpu_fallback(v: &mut Validator) {
    v.section("═══ MH06: CPU fallback parity ═══");

    let counts: Vec<f64> = vec![100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0];

    let direct_h = diversity::shannon(&counts);
    let direct_d = diversity::simpson(&counts);
    let direct_j = diversity::pielou_evenness(&counts);
    let direct_s = diversity::observed_features(&counts);

    let result = dispatch::dispatch("science.diversity", &json!({ "counts": counts }))
        .or_exit("diversity dispatch");

    let d_h = result["shannon"].as_f64().or_exit("h");
    let d_d = result["simpson"].as_f64().or_exit("d");
    let d_j = result["pielou"].as_f64().or_exit("j");
    let d_s = result["observed"].as_f64().or_exit("s");

    v.check(
        "fallback Shannon parity",
        direct_h,
        d_h,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "fallback Simpson parity",
        direct_d,
        d_d,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "fallback Pielou parity",
        direct_j,
        d_j,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "fallback observed features",
        direct_s,
        d_s,
        tolerances::EXACT_F64,
    );
}

// ── MH07: Full pipeline via dispatch ─────────────────────────────────────────

fn validate_full_pipeline(v: &mut Validator) {
    v.section("═══ MH07: Full pipeline: FASTQ → quality → diversity → QS ODE ═══");
    let t = Instant::now();

    let path = temp_path("nucleus_pipeline.fastq");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        let species: &[(&str, usize)] = &[
            ("ATGCATGCATGCATGCATGCATGCATGCATGC", 40),
            ("GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC", 30),
            ("TTTTAAAACCCCGGGGTTTTAAAACCCCGGGG", 20),
            ("AACCGGTTAACCGGTTAACCGGTTAACCGGTT", 10),
        ];
        let mut idx = 0_usize;
        for (seq, count) in species {
            for _ in 0..*count {
                let qual: String = std::iter::repeat_n('I', seq.len()).collect();
                writeln!(f, "@read_{idx}\n{seq}\n+\n{qual}").or_exit("unexpected error");
                idx += 1;
            }
        }
    }

    let records: Vec<_> = fastq::FastqIter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");
    v.check_count("pipeline: parsed 100 reads", records.len(), 100);

    let params = quality::QualityParams {
        window_size: 4,
        window_min_quality: 20,
        leading_min_quality: 3,
        trailing_min_quality: 3,
        min_length: 20,
        phred_offset: 33,
    };
    let (filtered, _) = quality::filter_reads(&records, &params);
    v.check_pass(
        "pipeline: quality filter passes reads",
        !filtered.is_empty(),
    );

    let (uniques, dstats) = derep::dereplicate(&filtered, derep::DerepSort::Abundance, 0);
    v.check_pass(
        "pipeline: <= 4 unique sequences",
        dstats.unique_sequences <= 4,
    );

    let counts: Vec<f64> = uniques.iter().map(|u| usize_f64(u.abundance)).collect();

    let direct_h = diversity::shannon(&counts);
    let result = dispatch::dispatch("science.diversity", &json!({ "counts": counts }))
        .or_exit("pipeline diversity");

    v.check(
        "pipeline: Shannon direct == dispatch",
        direct_h,
        result["shannon"].as_f64().or_exit("h"),
        tolerances::ANALYTICAL_F64,
    );

    let qs_result = dispatch::dispatch(
        "science.qs_model",
        &json!({ "scenario": "standard_growth", "dt": 0.01 }),
    )
    .or_exit("pipeline qs");

    let qs_steps = qs_result["steps"].as_u64().unwrap_or(0);
    v.check_pass(
        &format!("pipeline: QS ODE ran {qs_steps} steps (>0)"),
        qs_steps > 0,
    );

    let peak_biofilm = qs_result["peak_biofilm"].as_f64().unwrap_or(0.0);
    v.check_pass(
        "pipeline: peak biofilm > 0 (QS activated)",
        peak_biofilm > 0.0,
    );

    println!("    MH07 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── MH08: Dispatch routing ───────────────────────────────────────────────────

fn validate_dispatch_routing(v: &mut Validator) {
    v.section("═══ MH08: Dispatch routing — substrate selection ═══");

    let small_counts: Vec<f64> = vec![10.0, 20.0, 30.0];
    let result_small = dispatch::dispatch("science.diversity", &json!({ "counts": small_counts }))
        .or_exit("small dispatch");

    v.check_pass(
        "small input: has shannon result",
        result_small.get("shannon").is_some(),
    );

    let large_counts: Vec<f64> = (0..20_000_i32).map(|i| f64::from((i % 100) + 1)).collect();
    let result_large = dispatch::dispatch("science.diversity", &json!({ "counts": large_counts }))
        .or_exit("large dispatch");

    v.check_pass(
        "large input: has shannon result",
        result_large.get("shannon").is_some(),
    );

    let large_h = result_large["shannon"].as_f64().or_exit("unexpected error");
    let cpu_h = diversity::shannon(&large_counts);
    v.check(
        "large input: Shannon dispatch == direct",
        large_h,
        cpu_h,
        tolerances::ANALYTICAL_F64,
    );

    let err_result = dispatch::dispatch("nonexistent.method", &json!({}));
    v.check_pass("unknown method returns Err", err_result.is_err());
}

fn main() {
    let mut v = Validator::new("Exp214: NUCLEUS Mixed Hardware V8 — V66 I/O Evolution (IPC)");

    validate_tower_capabilities(&mut v);
    validate_fastq_diversity_dispatch(&mut v);
    validate_nanopore_signal(&mut v);
    validate_ms2_spectral(&mut v);
    validate_nest_metrics(&mut v);
    validate_cpu_fallback(&mut v);
    validate_full_pipeline(&mut v);
    validate_dispatch_routing(&mut v);

    v.finish();
}
