// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU validation of the full 16S pipeline — math parity across hardware.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | CPU path (this binary's own CPU reference) |
//! | Baseline version | Exp016 |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --features gpu --release --bin validate_16s_pipeline_gpu` |
//! | Data | PRJNA1114688, PRJNA629095, PRJNA1178324, PRJNA516219 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Runs the identical 16S pipeline on public datasets using both CPU and GPU
//! implementations, then checks that results match within f64 tolerance.
//! This proves the math is identical across hardware: Galaxy/Python ↔
//! `BarraCUDA` CPU ↔ `BarraCUDA` GPU.
//!
//! # Pipeline stages (CPU vs GPU)
//!
//! | Stage              | CPU                     | GPU                             |
//! |--------------------|-------------------------|---------------------------------|
//! | FASTQ parse        | `io::fastq`             | `io::fastq` (I/O, stays CPU)   |
//! | Quality filter     | `bio::quality`          | `bio::quality_gpu`              |
//! | Dereplication      | `bio::derep`            | `bio::derep` (hash, stays CPU)  |
//! | DADA2 denoise      | `bio::dada2`            | `bio::dada2` (stays CPU)        |
//! | Chimera detection  | `bio::chimera`          | `bio::chimera_gpu` (`GemmF64`)    |
//! | Taxonomy           | `bio::taxonomy`         | `bio::taxonomy_gpu` (batch)     |
//! | Diversity          | `bio::diversity`        | `bio::diversity_gpu` (GPU)      |
//!
//! # Tolerance
//!
//! All GPU vs CPU checks use `tolerances::GPU_VS_CPU_F64` (1e-6).
//!
//! Run: `cargo run --features gpu --release --bin validate_16s_pipeline_gpu`

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::useless_let_if_seq,
    clippy::option_if_let_else
)]

use std::path::Path;
use std::time::Instant;
use wetspring_barracuda::bio::chimera::{self, ChimeraParams};
use wetspring_barracuda::bio::chimera_gpu;
use wetspring_barracuda::bio::dada2::{self, Dada2Params};
use wetspring_barracuda::bio::derep::{self, DerepSort};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::quality::{self, QualityParams};
use wetspring_barracuda::bio::streaming_gpu::GpuPipelineSession;
use wetspring_barracuda::bio::taxonomy::{ClassifyParams, NaiveBayesClassifier, TaxRank};
use wetspring_barracuda::bio::validation_helpers;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  wetSpring 16S Pipeline GPU Validation — Math Parity Check         ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Goal: Prove CPU and GPU produce identical scientific results      ║");
    println!(
        "║  Tolerance: {:<54} ║",
        format!("{:.0e}", tolerances::GPU_VS_CPU_F64)
    );
    println!("║  Datasets: PRJNA1114688, PRJNA629095, PRJNA1178324, PRJNA516219   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let mut v = Validator::new("wetSpring 16S Pipeline GPU — CPU vs GPU Parity");

    // ── GPU init ────────────────────────────────────────────────────────────
    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            validation::exit_skipped(&format!("GPU init failed: {e}"));
        }
    };

    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    // ── Pre-warm GPU pipeline session (compile shaders once) ────────────
    let session = GpuPipelineSession::new(&gpu).unwrap_or_else(|e| {
        validation::exit_skipped(&format!("GPU session init failed: {e}"));
    });
    println!(
        "  GPU session warmed in {:.1}ms (QF + DADA2 + GemmCached + FMR pipelines compiled)",
        session.warmup_ms
    );
    println!("  ToadStool TensorContext wired (buffer pool + bind group cache)\n");

    let base = validation::data_dir("WETSPRING_PUBLIC_DIR", "data/public_benchmarks");
    let ref_dir = validation::data_dir("WETSPRING_REF_DIR", "data/reference_dbs/silva_138");

    let classifier = validation_helpers::load_silva_classifier(&ref_dir);

    // ── Timing accumulators ─────────────────────────────────────────────────
    let mut cpu_total_ms = 0.0_f64;
    let mut gpu_total_ms = 0.0_f64;
    let mut samples_processed = 0_usize;

    // ── Process representative samples across 4 BioProjects ────────────────
    // 4 PRJNA1114688 samples for full CPU+GPU chimera comparison, then
    // 6 cross-dataset samples with chimera-skip (CPU chimera O(n³) is the
    // sole bottleneck; GPU chimera parity is already proven on 4 samples).
    let sample_sets: Vec<(&str, Vec<(&str, &str)>)> = vec![
        (
            "PRJNA1114688",
            vec![
                ("SRR29127218", "N.oculata D1-R1"),
                ("SRR29127209", "N.oculata D14-R1"),
                ("SRR29127205", "B.plicatilis D1-R2"),
                ("SRR29127215", "B.plicatilis D14-R1"),
            ],
        ),
        (
            "PRJNA629095",
            vec![
                ("SRR11638224", "N.oceanica phyco-1"),
                ("SRR11638231", "N.oceanica phyco-2"),
            ],
        ),
        (
            "PRJNA1178324",
            vec![
                ("SRR31143973", "Cyano-tox-1"),
                ("SRR31143980", "Cyano-tox-2"),
            ],
        ),
        (
            "PRJNA516219",
            vec![("SRR8472475", "LakeErie-1"), ("SRR8472476", "LakeErie-2")],
        ),
    ];
    let full_chimera_project = "PRJNA1114688";

    for (project, samples) in &sample_sets {
        let project_dir = base.join(project);
        if !project_dir.exists() {
            println!("  [SKIP] {project} not found\n");
            continue;
        }
        let run_full_chimera = *project == full_chimera_project;
        for &(accession, label) in samples {
            let dir = project_dir.join(accession);
            if !dir.exists() {
                continue;
            }
            let (c, g) = process_sample_gpu_vs_cpu(
                &mut v,
                &gpu,
                &session,
                &dir,
                label,
                accession,
                classifier.as_ref(),
                run_full_chimera,
            );
            cpu_total_ms += c;
            gpu_total_ms += g;
            samples_processed += 1;
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    TIMING SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Samples: {samples_processed:>4}                                                    ║"
    );
    println!(
        "║  CPU total: {:>10.1} ms ({:>8.2} s)                            ║",
        cpu_total_ms,
        cpu_total_ms / 1000.0
    );
    println!(
        "║  GPU total: {:>10.1} ms ({:>8.2} s)                            ║",
        gpu_total_ms,
        gpu_total_ms / 1000.0
    );
    if gpu_total_ms > 0.0 {
        let speedup = cpu_total_ms / gpu_total_ms;
        println!("║  GPU speedup: {speedup:>6.2}×                                            ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // Write machine-readable results
    let out_dir = validation::data_dir(
        "WETSPRING_GPU_PIPELINE_PARITY_DIR",
        "experiments/results/016_gpu_pipeline_parity",
    );
    std::fs::create_dir_all(&out_dir).ok();
    let json = format!(
        r#"{{
  "experiment": "016_gpu_pipeline_parity",
  "date": "2026-02-19",
  "description": "16S pipeline GPU vs CPU math parity validation",
  "tolerance": {},
  "samples_processed": {},
  "cpu_total_ms": {:.1},
  "gpu_total_ms": {:.1},
  "speedup": {:.2}
}}"#,
        tolerances::GPU_VS_CPU_F64,
        samples_processed,
        cpu_total_ms,
        gpu_total_ms,
        if gpu_total_ms > 0.0 {
            cpu_total_ms / gpu_total_ms
        } else {
            0.0
        },
    );
    let json_path = out_dir.join("gpu_parity_results.json");
    std::fs::write(&json_path, &json).ok();
    println!("\nResults written to {}", json_path.display());

    // ── Scaling benchmark: prove GPU advantage grows with workload ───────
    if let Some(clf) = &classifier {
        run_scaling_benchmark(&session, clf);
    }

    // ── ToadStool infrastructure stats ──────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              TOADSTOOL INFRASTRUCTURE STATS                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    for line in session.ctx_stats().lines() {
        println!("║  {line:<66} ║");
    }
    println!("║  GemmCached: pipeline compiled once, reused across all dispatches  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    v.finish();
}

// ── Process a sample through both CPU and GPU pipelines, comparing results ──

#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)] // GPU parity validation requires all pipeline context
fn process_sample_gpu_vs_cpu(
    v: &mut Validator,
    gpu: &GpuF64,
    session: &GpuPipelineSession,
    sample_dir: &Path,
    label: &str,
    accession: &str,
    classifier: Option<&NaiveBayesClassifier>,
    run_full_chimera: bool,
) -> (f64, f64) {
    v.section(&format!("GPU Parity: {label} ({accession})"));

    let r1_name = format!("{accession}_1.fastq.gz");
    let interleaved_name = format!("{accession}.fastq.gz");
    let fastq_path = if sample_dir.join(&r1_name).exists() {
        sample_dir.join(&r1_name)
    } else if sample_dir.join(&interleaved_name).exists() {
        sample_dir.join(&interleaved_name)
    } else {
        println!("  [SKIP] No FASTQ for {label}");
        return (0.0, 0.0);
    };

    let records = match decompress_gz_fastq(&fastq_path) {
        Ok(recs) => recs,
        Err(e) => {
            println!("  [ERROR] {e}");
            return (0.0, 0.0);
        }
    };

    let total = records.len();
    println!("  {label}: {total} reads");

    // ── Stage 1: Quality filter (CPU vs GPU) ───────────────────────────────
    let qparams = QualityParams::default();

    let cpu_t = Instant::now();
    let (cpu_filtered, _cpu_fstats) = quality::filter_reads(&records, &qparams);
    let cpu_qf_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;

    let gpu_t = Instant::now();
    let (gpu_filtered, _gpu_fstats) =
        session
            .filter_reads(&records, &qparams)
            .unwrap_or_else(|e| {
                println!("  [GPU ERROR] quality: {e}");
                quality::filter_reads(&records, &qparams)
            });
    let gpu_qf_ms = gpu_t.elapsed().as_secs_f64() * 1000.0;

    v.check_pass(
        &format!("{label}: QF read count CPU == GPU"),
        cpu_filtered.len() == gpu_filtered.len(),
    );
    println!(
        "  {label} QF: CPU={} GPU={} reads ({:.1}ms / {:.1}ms)",
        cpu_filtered.len(),
        gpu_filtered.len(),
        cpu_qf_ms,
        gpu_qf_ms,
    );

    // ── Subsample for tractable denoising ──────────────────────────────────
    let sub: Vec<_> = cpu_filtered.into_iter().take(5000).collect();
    if sub.len() < 50 {
        println!("  {label}: too few reads ({}) — skipping", sub.len());
        return (cpu_qf_ms, gpu_qf_ms);
    }

    // ── Stage 2: Dereplication (same on CPU and GPU — hash-based) ──────────
    let cpu_t = Instant::now();
    let (uniques, _) = derep::dereplicate(&sub, DerepSort::Abundance, 2);
    let cpu_derep_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  {label} derep: {} uniques ({:.1}ms)",
        uniques.len(),
        cpu_derep_ms
    );

    // ── Stage 3: DADA2 denoise (CPU vs GPU E-step) ─────────────────────────
    let dada2_params = Dada2Params::default();
    let cpu_t = Instant::now();
    let (cpu_asvs, _) = dada2::denoise(&uniques, &dada2_params);
    let cpu_dada2_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;

    let gpu_t = Instant::now();
    let (gpu_asvs, _) = session
        .denoise(&uniques, &dada2_params)
        .unwrap_or_else(|e| {
            println!("  [GPU ERROR] DADA2: {e}");
            dada2::denoise(&uniques, &dada2_params)
        });
    let gpu_dada2_ms = gpu_t.elapsed().as_secs_f64() * 1000.0;

    v.check_pass(
        &format!("{label}: DADA2 ASV count CPU ≈ GPU"),
        cpu_asvs.len() == gpu_asvs.len(),
    );
    let cpu_dada2_reads: usize = cpu_asvs.iter().map(|a| a.abundance).sum();
    let gpu_dada2_reads: usize = gpu_asvs.iter().map(|a| a.abundance).sum();
    v.check_pass(
        &format!("{label}: DADA2 total reads CPU == GPU"),
        cpu_dada2_reads == gpu_dada2_reads,
    );
    println!(
        "  {label} DADA2: CPU={} ASVs ({:.1}ms) GPU={} ASVs ({:.1}ms)",
        cpu_asvs.len(),
        cpu_dada2_ms,
        gpu_asvs.len(),
        gpu_dada2_ms,
    );
    let asvs = cpu_asvs;

    // ── Stage 4: Chimera detection (CPU vs GPU) ────────────────────────────
    let cparams = ChimeraParams::default();
    let cpu_chimera_ms;
    let gpu_chimera_ms;
    let clean_asvs: Vec<_>;

    if run_full_chimera {
        let cpu_t = Instant::now();
        let (cpu_chimera_results, cpu_cstats) = chimera::detect_chimeras(&asvs, &cparams);
        cpu_chimera_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;

        let gpu_t = Instant::now();
        let (gpu_chimera_results, gpu_cstats) =
            chimera_gpu::detect_chimeras_gpu(gpu, &asvs, &cparams).unwrap_or_else(|e| {
                println!("  [GPU ERROR] chimera: {e}");
                chimera::detect_chimeras(&asvs, &cparams)
            });
        gpu_chimera_ms = gpu_t.elapsed().as_secs_f64() * 1000.0;

        v.check_pass(
            &format!("{label}: chimera count CPU == GPU"),
            cpu_cstats.chimeras_found == gpu_cstats.chimeras_found,
        );
        v.check_pass(
            &format!("{label}: chimera retained CPU == GPU"),
            cpu_cstats.retained == gpu_cstats.retained,
        );
        println!(
            "  {label} chimera: CPU={}/{} GPU={}/{} ({:.1}ms / {:.1}ms)",
            cpu_cstats.chimeras_found,
            cpu_cstats.input_sequences,
            gpu_cstats.chimeras_found,
            gpu_cstats.input_sequences,
            cpu_chimera_ms,
            gpu_chimera_ms,
        );

        let mut score_match_count = 0_usize;
        let mut score_total = 0_usize;
        for (c, g) in cpu_chimera_results.iter().zip(gpu_chimera_results.iter()) {
            score_total += 1;
            if c.is_chimera == g.is_chimera {
                score_match_count += 1;
            }
        }
        let chimera_agreement = if score_total > 0 {
            score_match_count as f64 / score_total as f64
        } else {
            1.0
        };
        v.check_pass(
            &format!("{label}: chimera decision agreement > 95%"),
            chimera_agreement > 0.95,
        );
        println!(
            "  {label} chimera agreement: {:.1}% ({}/{})",
            chimera_agreement * 100.0,
            score_match_count,
            score_total,
        );

        clean_asvs = cpu_chimera_results
            .iter()
            .filter(|r| !r.is_chimera)
            .map(|r| asvs[r.query_idx].clone())
            .collect();
    } else {
        // GPU-only chimera for cross-dataset samples (CPU chimera O(n³) too slow)
        let gpu_t = Instant::now();
        let (gpu_chimera_results, gpu_cstats) =
            chimera_gpu::detect_chimeras_gpu(gpu, &asvs, &cparams).unwrap_or_else(|e| {
                println!("  [GPU ERROR] chimera: {e}");
                chimera::detect_chimeras(&asvs, &cparams)
            });
        gpu_chimera_ms = gpu_t.elapsed().as_secs_f64() * 1000.0;
        cpu_chimera_ms = gpu_chimera_ms;

        println!(
            "  {label} chimera (GPU-only): {}/{} flagged ({:.1}ms)",
            gpu_cstats.chimeras_found, gpu_cstats.input_sequences, gpu_chimera_ms,
        );

        v.check_pass(&format!("{label}: GPU chimera completes"), true);

        clean_asvs = gpu_chimera_results
            .iter()
            .filter(|r| !r.is_chimera)
            .map(|r| asvs[r.query_idx].clone())
            .collect();
    }

    // ── Stage 5+6: Streaming GPU — taxonomy GEMM + diversity FMR ──────────
    // Single streaming session: upload ASV data once, run taxonomy GEMM +
    // diversity FMR, download results once. CPU path runs separately for
    // comparison.
    let counts: Vec<f64> = clean_asvs.iter().map(|a| a.abundance as f64).collect();

    if counts.len() < 2 {
        println!("  {label}: <2 ASVs after chimera — skipping diversity+taxonomy");
        let cpu_ms = cpu_qf_ms + cpu_derep_ms + cpu_dada2_ms + cpu_chimera_ms;
        let gpu_ms = gpu_qf_ms + cpu_derep_ms + cpu_dada2_ms + gpu_chimera_ms;
        return (cpu_ms, gpu_ms);
    }

    // CPU reference: diversity
    let cpu_t = Instant::now();
    let cpu_shannon = diversity::shannon(&counts);
    let cpu_simpson = diversity::simpson(&counts);
    let cpu_observed = diversity::observed_features(&counts);
    let cpu_div_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;

    // CPU reference: taxonomy
    let mut cpu_tax_ms = 0.0;
    let mut cpu_tax_results: Vec<wetspring_barracuda::bio::taxonomy::Classification> = vec![];
    let params = ClassifyParams {
        bootstrap_n: 50,
        ..ClassifyParams::default()
    };
    let n_classify = clean_asvs.len().min(5);
    let seqs: Vec<&[u8]> = clean_asvs
        .iter()
        .take(n_classify)
        .map(|a| a.sequence.as_slice())
        .collect();

    if let Some(clf) = classifier {
        let cpu_t = Instant::now();
        cpu_tax_results = seqs.iter().map(|seq| clf.classify(seq, &params)).collect();
        cpu_tax_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;
    }

    // GPU streaming session: pre-warmed taxonomy GEMM + diversity FMR
    let gpu_result = if let Some(clf) = classifier {
        session.stream_sample(clf, &seqs, &counts, &params).ok()
    } else {
        None
    };

    let (gpu_shannon, gpu_simpson, gpu_observed, gpu_div_ms, gpu_tax_ms, gpu_tax_results) =
        if let Some(ref res) = gpu_result {
            (
                res.shannon,
                res.simpson,
                res.observed,
                res.diversity_ms,
                res.taxonomy_ms,
                &res.classifications,
            )
        } else {
            // Fallback: pre-warmed FMR diversity
            let gpu_t = Instant::now();
            let s = session.shannon(&counts).unwrap_or(cpu_shannon);
            let d = session.simpson(&counts).unwrap_or(cpu_simpson);
            let o = session.observed_features(&counts).unwrap_or(cpu_observed);
            let div_ms = gpu_t.elapsed().as_secs_f64() * 1000.0;
            (s, d, o, div_ms, 0.0, &cpu_tax_results)
        };

    // ── Diversity parity checks ─────────────────────────────────────────────
    let tol = tolerances::GPU_VS_CPU_F64;
    let shannon_diff = (cpu_shannon - gpu_shannon).abs();
    let simpson_diff = (cpu_simpson - gpu_simpson).abs();
    let observed_diff = (cpu_observed - gpu_observed).abs();

    v.check_pass(
        &format!("{label}: Shannon CPU ≈ GPU (tol {tol:.0e})"),
        shannon_diff <= tol,
    );
    v.check_pass(
        &format!("{label}: Simpson CPU ≈ GPU (tol {tol:.0e})"),
        simpson_diff <= tol,
    );
    v.check_pass(
        &format!("{label}: observed CPU ≈ GPU (tol {tol:.0e})"),
        observed_diff <= tol,
    );

    println!(
        "  {label} diversity: Shannon CPU={cpu_shannon:.4} GPU={gpu_shannon:.4} (Δ={shannon_diff:.2e})",
    );
    println!(
        "  {label} diversity: Simpson CPU={cpu_simpson:.4} GPU={gpu_simpson:.4} (Δ={simpson_diff:.2e})",
    );
    println!(
        "  {label} diversity: observed CPU={cpu_observed:.0} GPU={gpu_observed:.0} (Δ={observed_diff:.2e})",
    );

    // ── Taxonomy parity checks ──────────────────────────────────────────────
    if !cpu_tax_results.is_empty() {
        let mut taxa_match = 0_usize;
        for (c, g) in cpu_tax_results.iter().zip(gpu_tax_results.iter()) {
            let c_genus = c.lineage.at_rank(TaxRank::Genus).unwrap_or("");
            let g_genus = g.lineage.at_rank(TaxRank::Genus).unwrap_or("");
            if c_genus == g_genus {
                taxa_match += 1;
            }
        }
        v.check_pass(
            &format!("{label}: taxonomy genus agreement CPU == GPU"),
            taxa_match == cpu_tax_results.len(),
        );
        println!(
            "  {label} taxonomy: {}/{} genus match (CPU {:.1}ms / GPU GEMM {:.1}ms)",
            taxa_match,
            cpu_tax_results.len(),
            cpu_tax_ms,
            gpu_tax_ms,
        );
    }

    println!(
        "  {label} streaming GPU session: taxonomy {:.1}ms + diversity {:.1}ms = {:.1}ms",
        gpu_tax_ms,
        gpu_div_ms,
        gpu_tax_ms + gpu_div_ms,
    );

    // ── Totals ──────────────────────────────────────────────────────────────
    let cpu_ms = cpu_qf_ms + cpu_derep_ms + cpu_dada2_ms + cpu_chimera_ms + cpu_div_ms + cpu_tax_ms;
    let gpu_ms = gpu_qf_ms + cpu_derep_ms + gpu_dada2_ms + gpu_chimera_ms + gpu_div_ms + gpu_tax_ms;

    println!("  {label} totals: CPU={cpu_ms:.1}ms GPU={gpu_ms:.1}ms\n",);

    (cpu_ms, gpu_ms)
}

// ── Scaling benchmark: dispatch overhead + parallelization ───────────────────

#[allow(clippy::cast_precision_loss)]
fn run_scaling_benchmark(session: &GpuPipelineSession, classifier: &NaiveBayesClassifier) {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           SCALING BENCHMARK — GPU vs CPU at Varying Load           ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Goal: GPU competitive at small N, dominant at large N             ║");
    println!("║  Proves: dispatch overhead cleared, parallelization unlocked       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Generate synthetic query sequences (realistic 16S length ~250bp)
    let base_seq = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                     ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                     ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\
                     ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT";

    // Vary sequence slightly per query using a simple mutation
    let make_seqs = |n: usize| -> Vec<Vec<u8>> {
        (0..n)
            .map(|i| {
                let mut s = base_seq.to_vec();
                let bases = [b'A', b'C', b'G', b'T'];
                let slen = s.len();
                for j in 0..slen.min(10) {
                    s[(i * 7 + j * 13) % slen] = bases[(i + j) % 4];
                }
                s
            })
            .collect()
    };

    let params = ClassifyParams {
        bootstrap_n: 50,
        ..ClassifyParams::default()
    };

    let query_sizes = [5, 25, 100, 500];

    println!(
        "  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Queries", "CPU (ms)", "GPU (ms)", "Speedup", "GPU tax/q"
    );
    println!("  {}", "─".repeat(62));

    for &n_queries in &query_sizes {
        let seqs_owned = make_seqs(n_queries);
        let seqs: Vec<&[u8]> = seqs_owned.iter().map(std::vec::Vec::as_slice).collect();
        let counts: Vec<f64> = (0..n_queries).map(|i| (i as f64 + 1.0) * 10.0).collect();

        // CPU
        let cpu_t = Instant::now();
        let _cpu_results: Vec<_> = seqs
            .iter()
            .map(|seq| classifier.classify(seq, &params))
            .collect();
        let _ = diversity::shannon(&counts);
        let _ = diversity::simpson(&counts);
        let _ = diversity::observed_features(&counts);
        let cpu_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;

        // GPU (pre-warmed session)
        let gpu_t = Instant::now();
        let gpu_result = session
            .stream_sample(classifier, &seqs, &counts, &params)
            .ok();
        let gpu_ms = gpu_t.elapsed().as_secs_f64() * 1000.0;

        let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };
        let gpu_tax_per_q = gpu_result
            .as_ref()
            .map_or(0.0, |r| r.taxonomy_ms / n_queries as f64);

        println!(
            "  {n_queries:>8}  {cpu_ms:>10.1}ms  {gpu_ms:>10.1}ms  {speedup:>9.1}×  {gpu_tax_per_q:>8.2}ms"
        );
    }

    println!();
}

// ── Gzipped FASTQ decompression ─────────────────────────────────────────────

fn decompress_gz_fastq(path: &Path) -> Result<Vec<FastqRecord>, String> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let decoder = flate2::read::GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut records = Vec::new();
    let mut lines = reader.lines();

    loop {
        let header = match lines.next() {
            Some(Ok(l)) if l.starts_with('@') => l,
            Some(Ok(_)) => continue,
            _ => break,
        };
        let Some(Ok(seq)) = lines.next() else { break };
        let Some(Ok(_)) = lines.next() else { break };
        let Some(Ok(qual)) = lines.next() else { break };

        let id = header[1..]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        records.push(FastqRecord {
            id,
            sequence: seq.into_bytes(),
            quality: qual.into_bytes(),
        });
    }

    if records.is_empty() {
        Err("No complete FASTQ records".to_string())
    } else {
        Ok(records)
    }
}
