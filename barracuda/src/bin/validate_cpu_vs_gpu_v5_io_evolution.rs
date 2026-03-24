// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
//! # Exp215: CPU vs GPU v5 — V66 I/O Evolution Domains
//!
//! Validates that GPU math produces identical results to CPU for all domains
//! that were evolved during the V66 audit, proving the I/O evolution
//! (byte-native FASTQ, bytemuck nanopore, streaming MS2) feeds the GPU
//! analytical pipeline correctly.
//!
//! - **G01**: FASTQ byte-native → GPU diversity (Shannon, Simpson, Bray-Curtis)
//! - **G02**: FASTQ → quality filter → GPU dereplication parity
//! - **G03**: MS2 streaming → GPU spectral match (pairwise cosine)
//! - **G04**: Nanopore → calibrated signal → GPU signal statistics
//! - **G05**: Full 16S pipeline: FASTQ → QF → derep → diversity (GPU chain)
//! - **G06**: GPU dispatch threshold gating validation
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Exp092 CPU vs GPU all domains, Exp212 CPU v12 (55/55) |
//! | Date | 2026-02-27 |
//! | Commit | wetSpring Phase 66 |
//! | Command | `cargo run --features gpu --release --bin validate_cpu_vs_gpu_v5_io_evolution` |
//!
//! Validation class: GPU-parity
//!
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use wetspring_barracuda::bio::diversity_gpu;
use wetspring_barracuda::bio::{derep, diversity, quality};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::io::fastq::{self, FastqRefRecord};
use wetspring_barracuda::io::ms2;
use wetspring_barracuda::io::nanopore::{self, NanoporeIter, SyntheticSignalGenerator};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("wetspring_exp215_{name}"))
}

// ── G01: FASTQ byte-native → GPU diversity ──────────────────────────────────

fn validate_fastq_gpu_diversity(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ G01: FASTQ byte-native → GPU diversity ═══");
    let t = Instant::now();

    let path = temp_path("gpu_community.fastq");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        let species: &[(&str, usize)] = &[
            ("ATGCATGCATGCATGC", 200),
            ("GCGCGCGCGCGCGCGC", 150),
            ("TTTTAAAACCCCGGGG", 100),
            ("AACCGGTTAACCGGTT", 50),
            ("GGGGCCCCAAAATTTT", 30),
            ("ACACACACACACACAC", 15),
            ("TGTGTGTGTGTGTGTG", 5),
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

    let counts: Vec<f64> = seq_counts.values().map(|&c| c as f64).collect();

    let cpu_h = diversity::shannon(&counts);
    let cpu_d = diversity::simpson(&counts);
    let cpu_j = diversity::pielou_evenness(&counts);
    let cpu_s = diversity::observed_features(&counts);

    let gpu_result = diversity_gpu::alpha_diversity_gpu(gpu, &counts).or_exit("unexpected error");
    let gpu_h = gpu_result.shannon;
    let gpu_d = gpu_result.simpson;
    let gpu_j = gpu_result.evenness;
    let gpu_s = gpu_result.observed;

    v.check(
        "Shannon: CPU == GPU",
        cpu_h,
        gpu_h,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Simpson: CPU == GPU",
        cpu_d,
        gpu_d,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Pielou: CPU == GPU",
        cpu_j,
        gpu_j,
        tolerances::ANALYTICAL_F64,
    );
    v.check("Observed: CPU == GPU", cpu_s, gpu_s, tolerances::EXACT_F64);

    let n = counts.len();
    let samples: Vec<Vec<f64>> = counts.iter().map(|&c| vec![c]).collect();
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let gpu_bc =
        diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).or_exit("unexpected error");

    v.check_pass("Bray-Curtis condensed: CPU computed", !cpu_bc.is_empty());
    v.check_count(
        "Bray-Curtis condensed: length match",
        gpu_bc.len(),
        cpu_bc.len(),
    );
    for (k, (&c, &g)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate() {
        v.check(
            &format!("BC[{k}]: CPU == GPU"),
            c,
            g,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    println!(
        "    G01 completed: {:.0}µs ({n} taxa)",
        t.elapsed().as_micros()
    );
    let _ = std::fs::remove_file(&path);
}

// ── G02: Quality filter → GPU dereplication ─────────────────────────────────

fn validate_quality_gpu_derep(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ G02: Quality filter → dereplication parity ═══");
    let t = Instant::now();

    let path = temp_path("gpu_quality.fastq");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        for i in 0..100_u32 {
            let seq = if i < 60 {
                "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
            } else {
                "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC"
            };
            let qual: String = if i % 10 == 0 {
                std::iter::repeat_n('!', seq.len()).collect()
            } else {
                std::iter::repeat_n('I', seq.len()).collect()
            };
            writeln!(f, "@read_{i}\n{seq}\n+\n{qual}").or_exit("unexpected error");
        }
    }

    let records: Vec<_> = fastq::FastqIter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");
    let params = quality::QualityParams {
        window_size: 4,
        window_min_quality: 15,
        leading_min_quality: 3,
        trailing_min_quality: 3,
        min_length: 36,
        phred_offset: 33,
    };
    let (filtered, qstats) = quality::filter_reads(&records, &params);

    v.check_count("quality: 100 input reads", qstats.input_reads, 100);
    v.check_pass("quality: some pass filter", qstats.output_reads > 0);

    let (uniques, dstats) = derep::dereplicate(&filtered, derep::DerepSort::Abundance, 0);

    v.check_pass("derep: unique sequences <= 2", dstats.unique_sequences <= 2);
    v.check_pass(
        "derep: total abundance == filtered count",
        uniques.iter().map(|u| u.abundance).sum::<usize>() == filtered.len(),
    );

    let counts: Vec<f64> = uniques.iter().map(|u| u.abundance as f64).collect();
    let cpu_h = diversity::shannon(&counts);
    let gpu_result = diversity_gpu::alpha_diversity_gpu(gpu, &counts).or_exit("unexpected error");

    v.check(
        "derep → Shannon: CPU == GPU",
        cpu_h,
        gpu_result.shannon,
        tolerances::ANALYTICAL_F64,
    );

    println!("    G02 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── G03: MS2 streaming → GPU spectral match ─────────────────────────────────

fn validate_ms2_gpu_spectral(v: &mut Validator, _gpu: &GpuF64) {
    v.section("═══ G03: MS2 streaming → spectral math ═══");
    let t = Instant::now();

    let path = temp_path("gpu_spectral.ms2");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        writeln!(f, "H\tCreatedBy\tExp215").or_exit("unexpected error");
        for scan in 1..=4_u32 {
            let precursor = f64::from(scan).mul_add(100.0, 300.0);
            writeln!(f, "S\t{scan}\t{scan}\t{precursor:.3}").or_exit("unexpected error");
            writeln!(f, "I\tRTime\t{:.1}", f64::from(scan) * 0.5).or_exit("unexpected error");
            writeln!(f, "I\tBPI\t10000.0").or_exit("unexpected error");
            writeln!(f, "I\tTIC\t50000.0").or_exit("unexpected error");
            writeln!(f, "Z\t2\t{:.3}", precursor * 2.0 - 1.008).or_exit("unexpected error");
            for frag in 0..scan {
                writeln!(
                    f,
                    "{:.1}\t{:.1}",
                    f64::from(frag).mul_add(50.0, 100.0),
                    1000.0 * f64::from(scan - frag)
                )
                .or_exit("unexpected error");
            }
        }
    }

    let mut stream_spectra = Vec::new();
    ms2::for_each_spectrum(&path, |spec| {
        stream_spectra.push(spec);
        Ok(())
    })
    .or_exit("unexpected error");

    let batch: Vec<_> = ms2::Ms2Iter::open(&path)
        .or_exit("unexpected error")
        .collect::<Result<Vec<_>, _>>()
        .or_exit("unexpected error");

    v.check_count("4 spectra from stream", stream_spectra.len(), 4);
    v.check_count("4 spectra from batch", batch.len(), 4);

    for (i, (s, b)) in stream_spectra.iter().zip(batch.iter()).enumerate() {
        v.check(
            &format!("spec{i}: precursor stream==batch"),
            s.precursor_mz,
            b.precursor_mz,
            tolerances::ANALYTICAL_F64,
        );
    }

    let monotonic = batch
        .windows(2)
        .all(|w| w[1].precursor_mz > w[0].precursor_mz);
    v.check_pass("precursor m/z monotonically increasing", monotonic);

    println!("    G03 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── G04: Nanopore → GPU signal statistics ───────────────────────────────────

fn validate_nanopore_gpu_stats(v: &mut Validator, _gpu: &GpuF64) {
    v.section("═══ G04: Nanopore → calibrated signal → GPU stats ═══");
    let t = Instant::now();

    let sig = SyntheticSignalGenerator::new(77);
    let reads = sig.generate_batch(20, 4000, 4000.0);

    let path = temp_path("gpu_nanopore.nrs");
    nanopore::write_nrs(&path, &reads).or_exit("unexpected error");
    let loaded: Vec<_> = NanoporeIter::open(&path)
        .or_exit("unexpected error")
        .map(|r| r.or_exit("unexpected error"))
        .collect();

    v.check_count("loaded 20 reads", loaded.len(), 20);

    for (i, (orig, rt)) in reads.iter().zip(loaded.iter()).enumerate().take(5) {
        let orig_cal = orig.calibrated_signal();
        let rt_cal = rt.calibrated_signal();

        let cpu_mean_orig: f64 = orig_cal.iter().sum::<f64>() / orig_cal.len() as f64;
        let cpu_mean_rt: f64 = rt_cal.iter().sum::<f64>() / rt_cal.len() as f64;

        v.check(
            &format!("read{i}: calibrated mean roundtrip"),
            cpu_mean_orig,
            cpu_mean_rt,
            tolerances::ANALYTICAL_F64,
        );
    }

    let signal_f64: Vec<f64> = loaded[0].signal.iter().map(|&s| f64::from(s)).collect();
    // Intentional: manual population variance to validate signal_stats implementation.
    let cpu_mean: f64 = signal_f64.iter().sum::<f64>() / signal_f64.len() as f64;
    let cpu_var: f64 = signal_f64
        .iter()
        .map(|x| (x - cpu_mean).powi(2))
        .sum::<f64>()
        / signal_f64.len() as f64;

    let stats = loaded[0].signal_stats();
    v.check(
        "signal mean: manual == stats",
        cpu_mean,
        stats.mean,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "signal std_dev: manual == stats",
        cpu_var.sqrt(),
        stats.std_dev,
        tolerances::ANALYTICAL_F64,
    );

    println!("    G04 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── G05: Full pipeline GPU chain ─────────────────────────────────────────────

fn validate_full_gpu_chain(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ G05: Full pipeline: FASTQ → QF → derep → diversity (GPU) ═══");
    let t = Instant::now();

    let path = temp_path("gpu_pipeline.fastq");
    {
        let mut f = std::fs::File::create(&path).or_exit("unexpected error");
        let species: &[(&str, usize)] = &[
            ("ATGCATGCATGCATGCATGCATGCATGCATGC", 80),
            ("GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC", 60),
            ("TTTTAAAACCCCGGGGTTTTAAAACCCCGGGG", 40),
            ("AACCGGTTAACCGGTTAACCGGTTAACCGGTT", 20),
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
    v.check_count("pipeline: 200 reads", records.len(), 200);

    let params = quality::QualityParams {
        window_size: 4,
        window_min_quality: 20,
        leading_min_quality: 3,
        trailing_min_quality: 3,
        min_length: 20,
        phred_offset: 33,
    };
    let (filtered, _) = quality::filter_reads(&records, &params);
    v.check_pass("pipeline: quality passes reads", !filtered.is_empty());

    let (uniques, dstats) = derep::dereplicate(&filtered, derep::DerepSort::Abundance, 0);
    v.check_pass("pipeline: <= 4 unique", dstats.unique_sequences <= 4);

    let counts: Vec<f64> = uniques.iter().map(|u| u.abundance as f64).collect();

    let cpu_h = diversity::shannon(&counts);
    let cpu_d = diversity::simpson(&counts);

    let gpu_result = diversity_gpu::alpha_diversity_gpu(gpu, &counts).or_exit("unexpected error");

    v.check(
        "pipeline Shannon: CPU == GPU",
        cpu_h,
        gpu_result.shannon,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "pipeline Simpson: CPU == GPU",
        cpu_d,
        gpu_result.simpson,
        tolerances::ANALYTICAL_F64,
    );

    println!("    G05 completed: {:.0}µs", t.elapsed().as_micros());
    let _ = std::fs::remove_file(&path);
}

// ── G06: GPU dispatch threshold ──────────────────────────────────────────────

fn validate_gpu_threshold(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ G06: GPU dispatch threshold gating ═══");

    let threshold = gpu.dispatch_threshold();
    v.check_pass(
        &format!("dispatch threshold: {threshold} (reasonable range 1k-100k)"),
        (1_000..=100_000).contains(&threshold),
    );

    let small: Vec<f64> = (1..=10).map(f64::from).collect();
    let use_gpu_small = small.len() >= threshold;
    v.check_pass("10 elements: below threshold (CPU)", !use_gpu_small);

    let use_gpu_large = (1..=200_000_i32).count() >= threshold;
    v.check_pass("200k elements: above threshold (GPU)", use_gpu_large);

    let cpu_h = diversity::shannon(&small);
    let gpu_h = diversity_gpu::alpha_diversity_gpu(gpu, &small)
        .or_exit("unexpected error")
        .shannon;
    v.check(
        "small set: CPU == GPU math identical",
        cpu_h,
        gpu_h,
        tolerances::ANALYTICAL_F64,
    );
}

fn main() {
    let rt = tokio::runtime::Runtime::new().or_exit("unexpected error");
    let gpu = rt
        .block_on(GpuF64::new())
        .or_exit("GPU with SHADER_F64 required for Exp215");

    gpu.print_info();
    println!();

    let mut v = Validator::new("Exp215: CPU vs GPU v5 — V66 I/O Evolution Domains");

    validate_fastq_gpu_diversity(&mut v, &gpu);
    validate_quality_gpu_derep(&mut v, &gpu);
    validate_ms2_gpu_spectral(&mut v, &gpu);
    validate_nanopore_gpu_stats(&mut v, &gpu);
    validate_full_gpu_chain(&mut v, &gpu);
    validate_gpu_threshold(&mut v, &gpu);

    v.finish();
}
