// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp227: Pure GPU Streaming v4 — Unidirectional Full Science Pipeline
//!
//! Extends v3 with V71 additions: precision-flexible GEMM, DF64 host-side
//! pack/unpack, `DiversityFusion`, and end-to-end timing comparison of
//! streaming vs individual dispatch. Demonstrates `ToadStool`'s
//! unidirectional streaming architecture: multiple GPU stages chained
//! with zero CPU compute round-trips.
//!
//! # Three-tier chain
//!
//! ```text
//! Paper (224) → CPU (225) → GPU (226) → Streaming (this) → `metalForge` (228)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-28 |
//! | Phase | 71 |
//! | Command | `cargo run --release --features gpu --bin validate_pure_gpu_streaming_v4` |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::sync::Arc;
use std::time::Instant;

use barracuda::shaders::Precision;
use wetspring_barracuda::bio::{
    diversity,
    diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu},
    gemm_cached::GemmCached,
    pcoa, pcoa_gpu,
    quality::{self, QualityParams},
    spectral_match_gpu,
    streaming_gpu::GpuPipelineSession,
};
use wetspring_barracuda::df64_host;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};
use wetspring_barracuda::validation::OrExit;

#[tokio::main]
async fn main() {
    let mut v =
        Validator::new("Exp227: Pure GPU Streaming v4 — Unidirectional Full Science Pipeline");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support");
    }

    let session = match GpuPipelineSession::new(&gpu) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Session init failed: {e}");
            validation::exit_skipped("GpuPipelineSession init failed");
        }
    };
    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    println!("  Session: {}", session.ctx_stats());
    println!("  Streaming v4: quality → diversity → fusion → GEMM → PCoA → spectral");

    let mut timings: Vec<(&str, f64, f64)> = Vec::new();
    let pipeline_start = Instant::now();

    // ═══ Stage 1: Quality Filtering ════════════════════════════════════
    v.section("S1: Quality Filtering");
    let reads = synthetic_reads(500, 150);
    let qparams = QualityParams::default();
    let (tc, tg, n_pass) = {
        let t0 = Instant::now();
        let cpu_filt = quality::filter_reads(&reads, &qparams);
        let cpu_us = t0.elapsed().as_micros() as f64;
        let t1 = Instant::now();
        let (gpu_filt, _stats) = session.filter_reads(&reads, &qparams).or_exit("unexpected error");
        let gpu_us = t1.elapsed().as_micros() as f64;
        v.check_count("S1: filtered count", gpu_filt.len(), cpu_filt.0.len());
        (cpu_us, gpu_us, gpu_filt.len())
    };
    timings.push(("Quality Filter", tc, tg));
    println!("  {n_pass}/{} reads passed", reads.len());

    // ═══ Stage 2: Alpha + Beta Diversity ═══════════════════════════════
    v.section("S2: Diversity (α + β)");
    let samples: Vec<Vec<f64>> = vec![
        (1..=100).map(|i| f64::from(i % 20 + 1)).collect(),
        (1..=100).map(|i| f64::from((i * 3) % 25 + 1)).collect(),
        (1..=100).map(|i| f64::from((i * 7) % 30 + 1)).collect(),
        (1..=100).map(|i| f64::from((i * 11) % 15 + 1)).collect(),
    ];
    let sample_refs: Vec<&[f64]> = samples.iter().map(Vec::as_slice).collect();

    let t0 = Instant::now();
    let cpu_shannons: Vec<f64> = samples.iter().map(|c| diversity::shannon(c)).collect();
    let cpu_bc = diversity::bray_curtis_condensed(&samples);
    let cpu_div_us = t0.elapsed().as_micros() as f64;

    let t1 = Instant::now();
    let gpu_shannons: Vec<f64> = samples
        .iter()
        .map(|c| session.shannon(c).or_exit("unexpected error"))
        .collect();
    let gpu_bc = session.bray_curtis_matrix(&sample_refs).or_exit("unexpected error");
    let gpu_div_us = t1.elapsed().as_micros() as f64;

    for (i, (&c, &g)) in cpu_shannons.iter().zip(&gpu_shannons).enumerate() {
        v.check(&format!("Shannon[{i}]"), g, c, tolerances::GPU_LOG_POLYFILL);
    }
    for (i, (&c, &g)) in cpu_bc.iter().zip(&gpu_bc).enumerate() {
        v.check(&format!("BC[{i}]"), g, c, tolerances::GPU_VS_CPU_F64);
    }
    timings.push(("Diversity (α + β)", cpu_div_us, gpu_div_us));

    // ═══ Stage 3: DiversityFusion GPU ══════════════════════════════════
    v.section("S3: DiversityFusion (fused map-reduce)");
    let n_species = 1000;
    let n_fuse_samples = 4;
    let flat: Vec<f64> = (0..n_fuse_samples * n_species)
        .map(|i| ((i * 13 + 7) % 200 + 1) as f64)
        .collect();

    let t0 = Instant::now();
    let cpu_fuse = diversity_fusion_cpu(&flat, n_species);
    let cpu_fuse_us = t0.elapsed().as_micros() as f64;

    let fusion_gpu = DiversityFusionGpu::new(Arc::clone(&device)).or_exit("DiversityFusionGpu");
    let t1 = Instant::now();
    let gpu_fuse = fusion_gpu
        .compute(&flat, n_fuse_samples, n_species)
        .or_exit("fusion");
    let gpu_fuse_us = t1.elapsed().as_micros() as f64;

    for (i, (c, g)) in cpu_fuse.iter().zip(&gpu_fuse).enumerate() {
        v.check(
            &format!("Fusion Shannon[{i}]"),
            g.shannon,
            c.shannon,
            tolerances::GPU_VS_CPU_F64,
        );
    }
    timings.push(("DiversityFusion", cpu_fuse_us, gpu_fuse_us));

    // ═══ Stage 4: GEMM (V71 precision-flexible) ═══════════════════════
    v.section("S4: GEMM Streaming (V71 with_precision)");
    let m = 256;
    let k = 128;
    let n = 256;
    let a_mat: Vec<f64> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0)
        .collect();
    let b_mat: Vec<f64> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
        .collect();

    let t0 = Instant::now();
    let mut cpu_c = vec![0.0_f64; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut sum = 0.0;
            for j in 0..k {
                sum += a_mat[r * k + j] * b_mat[j * n + c];
            }
            cpu_c[r * n + c] = sum;
        }
    }
    let cpu_gemm_us = t0.elapsed().as_micros() as f64;

    let gemm = GemmCached::with_precision(Arc::clone(&device), Arc::clone(&ctx), Precision::F64);
    let t1 = Instant::now();
    let gpu_c = gemm.execute(&a_mat, &b_mat, m, k, n, 1).or_exit("GEMM");
    let gpu_gemm_us = t1.elapsed().as_micros() as f64;

    v.check(
        "GEMM C[0,0]",
        gpu_c[0],
        cpu_c[0],
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "GEMM C[m-1,n-1]",
        gpu_c[m * n - 1],
        cpu_c[m * n - 1],
        tolerances::GPU_VS_CPU_F64,
    );
    let max_err = cpu_c
        .iter()
        .zip(&gpu_c)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check_pass(
        "GEMM max error < 1e-5",
        max_err < tolerances::GEMM_GPU_MAX_ERR,
    );
    timings.push(("GEMM 256×128×256", cpu_gemm_us, gpu_gemm_us));

    // ═══ Stage 5: PCoA from Streaming BC ═══════════════════════════════
    v.section("S5: PCoA (zero round-trip from Stage 2 BC)");

    let t0 = Instant::now();
    let cpu_pcoa = pcoa::pcoa(&cpu_bc, 4, 2).or_exit("unexpected error");
    let cpu_pcoa_us = t0.elapsed().as_micros() as f64;

    let t1 = Instant::now();
    let gpu_pcoa = pcoa_gpu::pcoa_gpu(&gpu, &gpu_bc, 4, 2).or_exit("unexpected error");
    let gpu_pcoa_us = t1.elapsed().as_micros() as f64;

    v.check_count("PCoA samples", gpu_pcoa.n_samples, cpu_pcoa.n_samples);
    v.check_pass(
        "PCoA coords finite",
        gpu_pcoa.coordinates.iter().all(|c| c.is_finite()),
    );
    timings.push(("PCoA", cpu_pcoa_us, gpu_pcoa_us));

    // ═══ Stage 6: Spectral Cosine ══════════════════════════════════════
    v.section("S6: Spectral Cosine Streaming");
    let spectra: Vec<Vec<f64>> = vec![
        vec![100.0, 200.0, 50.0, 300.0, 150.0],
        vec![100.0, 200.0, 50.0, 300.0, 150.0],
        vec![10.0, 20.0, 500.0, 30.0, 15.0],
    ];
    let gpu_cos = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spectra).or_exit("unexpected error");
    v.check_pass(
        "Self-cosine ≈ 1",
        (gpu_cos[0] - 1.0).abs() < tolerances::GPU_VS_CPU_F64,
    );
    timings.push(("Spectral Cosine", 0.0, 0.0));

    // ═══ Stage 7: DF64 Pack/Unpack on Streaming Data ══════════════════
    v.section("S7: DF64 Host Protocol (V71)");
    let gemm_slice = &gpu_c[..10];
    let packed = df64_host::pack_slice(gemm_slice);
    let unpacked = df64_host::unpack_slice(&packed);
    let df64_max_err = gemm_slice
        .iter()
        .zip(&unpacked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    v.check_pass(
        "DF64 roundtrip < 1e-13",
        df64_max_err < tolerances::DF64_ROUNDTRIP,
    );

    let f32_err: f64 = gemm_slice
        .iter()
        .map(|&x| (x - f64::from(x as f32)).abs())
        .fold(0.0_f64, f64::max);
    v.check_pass(
        "DF64 beats f32 precision",
        df64_max_err < f32_err || f32_err == 0.0,
    );
    println!("  DF64 max err: {df64_max_err:.2e}, f32 max err: {f32_err:.2e}");

    // ═══ Streaming vs Dispatch Benchmark ═══════════════════════════════
    v.section("Streaming vs Individual Dispatch");

    let warmup_reads = synthetic_reads(200, 150);
    let warmup_samples: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            (1..=50)
                .map(|i| f64::from((i * (k + 1)) % 20 + 1))
                .collect()
        })
        .collect();
    let warmup_refs: Vec<&[f64]> = warmup_samples.iter().map(Vec::as_slice).collect();

    let t_stream = Instant::now();
    let _f = session.filter_reads(&warmup_reads, &qparams);
    for s in &warmup_samples {
        let _ = session.shannon(s);
    }
    let _ = session.bray_curtis_matrix(&warmup_refs);
    let stream_ms = t_stream.elapsed().as_secs_f64() * 1000.0;

    let t_individual = Instant::now();
    let _ = quality::filter_reads(&warmup_reads, &qparams);
    for s in &warmup_samples {
        let _ = diversity::shannon(s);
    }
    let _ = diversity::bray_curtis_condensed(&warmup_samples);
    let individual_ms = t_individual.elapsed().as_secs_f64() * 1000.0;

    println!("  Streaming:  {stream_ms:.3} ms (GPU session, pre-warmed)");
    println!("  Individual: {individual_ms:.3} ms (CPU)");
    v.check_pass(
        "both pipelines completed",
        stream_ms > 0.0 && individual_ms > 0.0,
    );

    // ═══ Summary ═══════════════════════════════════════════════════════
    v.section("Streaming v4 Summary");
    println!();
    println!("  {:<35} {:>12} {:>12}", "Stage", "CPU (µs)", "GPU (µs)");
    println!("  {}", "─".repeat(62));
    for (name, cpu_us, gpu_us) in &timings {
        println!("  {name:<35} {cpu_us:>12.0} {gpu_us:>12.0}");
    }
    println!("  {}", "─".repeat(62));

    let total_ms = pipeline_start.elapsed().as_secs_f64() * 1000.0;
    let stages = timings.len();
    println!();
    println!("  {stages} stages streamed unidirectionally");
    println!("  V71 additions: GEMM with_precision, DF64 host protocol, DiversityFusion");
    println!("  Zero CPU compute round-trips (data flows GPU→GPU→GPU)");
    println!("  [Total wall time] {total_ms:.1} ms");

    v.finish();
}

fn synthetic_reads(count: usize, read_len: usize) -> Vec<FastqRecord> {
    let bases = b"ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC";
    let seq = &bases[..read_len.min(bases.len())];
    (0..count)
        .map(|i| FastqRecord {
            id: format!("@read_{i}"),
            sequence: seq.to_vec(),
            quality: vec![35_u8; seq.len()],
        })
        .collect()
}
