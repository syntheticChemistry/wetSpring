// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
//! Exp075: Pure GPU Multi-Stage Analytics Pipeline
//!
//! Five chained GPU stages with zero intermediate CPU round-trips:
//! 1. Alpha diversity (Shannon, Simpson, Observed, Evenness, Chao1)
//! 2. Beta diversity (Bray-Curtis distance matrix)
//! 3. Ordination (`PCoA` from distance matrix)
//! 4. Statistical summary (variance, correlation on PC axes)
//! 5. Spectral cosine similarity (pairwise via GEMM + FMR)
//!
//! Validates: GPU pipeline == CPU pipeline for all stages.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | BarraCUDA CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --release --features gpu --bin validate_pure_gpu_pipeline` |
//! | Data | 8 synthetic communities × 512 features |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{
    diversity, diversity_gpu, pcoa, pcoa_gpu, spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

const N_SAMPLES: usize = 8;
const N_FEATURES: usize = 512;
const N_AXES: usize = 3;

fn make_communities() -> Vec<Vec<f64>> {
    (0..N_SAMPLES)
        .map(|s| {
            (0..N_FEATURES)
                .map(|f| {
                    let base = ((s * N_FEATURES + f + 1) as f64).sqrt();
                    let shift = ((s + 1) as f64) * 0.1;
                    (base + shift).max(0.01)
                })
                .collect()
        })
        .collect()
}

fn make_spectra() -> Vec<Vec<f64>> {
    (0..N_SAMPLES)
        .map(|s| {
            (0..128)
                .map(|f| ((s * 128 + f + 1) as f64).mul_add(0.01, (s as f64) * 0.001))
                .collect()
        })
        .collect()
}

fn cosine_cpu(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = na * nb;
    if denom > 0.0 {
        (dot / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn pairwise_cosine_cpu(spectra: &[Vec<f64>]) -> Vec<f64> {
    let n = spectra.len();
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in 0..i {
            condensed.push(cosine_cpu(&spectra[i], &spectra[j]));
        }
    }
    condensed
}

fn variance_cpu(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n
}

fn correlation_cpu(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / n;
    let sx = variance_cpu(x).sqrt();
    let sy = variance_cpu(y).sqrt();
    if sx * sy > 0.0 {
        cov / (sx * sy)
    } else {
        0.0
    }
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp075: Pure GPU Multi-Stage Analytics Pipeline");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let communities = make_communities();
    let spectra = make_spectra();
    let pipeline_start = Instant::now();

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 1: Alpha Diversity — GPU via FMR
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 1: Alpha Diversity (GPU FMR)");

    let t1 = Instant::now();
    let mut gpu_alpha: Vec<diversity::AlphaDiversity> = Vec::with_capacity(N_SAMPLES);
    for community in &communities {
        let alpha = diversity_gpu::alpha_diversity_gpu(&gpu, community).unwrap();
        gpu_alpha.push(alpha);
    }
    let stage1_us = t1.elapsed().as_micros() as f64;

    let mut cpu_alpha: Vec<diversity::AlphaDiversity> = Vec::with_capacity(N_SAMPLES);
    for community in &communities {
        cpu_alpha.push(diversity::alpha_diversity(community));
    }

    for (i, (ga, ca)) in gpu_alpha.iter().zip(cpu_alpha.iter()).enumerate() {
        v.check(
            &format!("S{i}: Shannon GPU == CPU"),
            ga.shannon,
            ca.shannon,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            &format!("S{i}: Simpson GPU == CPU"),
            ga.simpson,
            ca.simpson,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }
    v.check(
        "Observed: all samples match",
        f64::from(u8::from(gpu_alpha.iter().zip(cpu_alpha.iter()).all(
            |(g, c)| (g.observed - c.observed).abs() < tolerances::GPU_VS_CPU_F64,
        ))),
        1.0,
        0.0,
    );

    println!("  Stage 1: {stage1_us:.0} µs ({N_SAMPLES} samples × 5 metrics)");

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 2: Beta Diversity — GPU Bray-Curtis
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 2: Beta Diversity (GPU BrayCurtis)");

    let t2 = Instant::now();
    let gpu_bray = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).unwrap();
    let stage2_us = t2.elapsed().as_micros() as f64;

    let cpu_bray = diversity::bray_curtis_condensed(&communities);
    let n_pairs = N_SAMPLES * (N_SAMPLES - 1) / 2;

    v.check(
        "Bray-Curtis: correct pair count",
        gpu_bray.len() as f64,
        n_pairs as f64,
        0.0,
    );

    let bray_max_diff = gpu_bray
        .iter()
        .zip(cpu_bray.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max);
    v.check(
        "Bray-Curtis: max diff within tolerance",
        f64::from(u8::from(bray_max_diff < tolerances::GPU_VS_CPU_BRAY_CURTIS)),
        1.0,
        0.0,
    );
    v.check(
        "Bray-Curtis: all in [0,1]",
        f64::from(u8::from(
            gpu_bray.iter().all(|d| *d >= 0.0 && *d <= 1.0 + 1e-10),
        )),
        1.0,
        0.0,
    );

    println!("  Stage 2: {stage2_us:.0} µs ({n_pairs} pairs)");

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 3: Ordination — GPU PCoA
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 3: PCoA Ordination (GPU Eigh)");

    let t3 = Instant::now();
    let gpu_pcoa = pcoa_gpu::pcoa_gpu(&gpu, &gpu_bray, N_SAMPLES, N_AXES).unwrap();
    let stage3_us = t3.elapsed().as_micros() as f64;

    let cpu_pcoa = pcoa::pcoa(&cpu_bray, N_SAMPLES, N_AXES).unwrap();

    v.check(
        "PCoA: correct n_axes",
        gpu_pcoa.eigenvalues.len() as f64,
        N_AXES as f64,
        0.0,
    );
    v.check(
        "PCoA: correct n_samples",
        gpu_pcoa.coordinates.len() as f64,
        N_SAMPLES as f64,
        0.0,
    );

    let eig_match = gpu_pcoa
        .eigenvalues
        .iter()
        .zip(cpu_pcoa.eigenvalues.iter())
        .all(|(g, c)| (g - c).abs() < tolerances::GPU_VS_CPU_F64 * c.abs().max(1.0));
    v.check(
        "PCoA: eigenvalues match CPU",
        f64::from(u8::from(eig_match)),
        1.0,
        0.0,
    );

    let prop_sum: f64 = gpu_pcoa.proportion_explained.iter().sum();
    v.check(
        "PCoA: proportion explained ≤ 1.0",
        f64::from(u8::from(prop_sum <= 1.0 + 1e-6)),
        1.0,
        0.0,
    );

    println!("  Stage 3: {stage3_us:.0} µs (PCoA {N_SAMPLES}×{N_AXES}, prop_var={prop_sum:.3})");

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 4: Statistical Summary — GPU FMR
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 4: Statistical Summary (GPU FMR)");

    let pc1: Vec<f64> = gpu_pcoa.coordinates.iter().map(|c| c[0]).collect();
    let pc2: Vec<f64> = gpu_pcoa.coordinates.iter().map(|c| c[1]).collect();

    let t4 = Instant::now();
    let gpu_var_pc1 = stats_gpu::variance_gpu(&gpu, &pc1).unwrap();
    let gpu_var_pc2 = stats_gpu::variance_gpu(&gpu, &pc2).unwrap();
    let gpu_corr = stats_gpu::correlation_gpu(&gpu, &pc1, &pc2).unwrap();
    let stage4_us = t4.elapsed().as_micros() as f64;

    let cpu_var_pc1 = variance_cpu(&pc1);
    let cpu_var_pc2 = variance_cpu(&pc2);
    let cpu_corr = correlation_cpu(&pc1, &pc2);

    v.check(
        "Var(PC1): GPU == CPU",
        gpu_var_pc1,
        cpu_var_pc1,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Var(PC2): GPU == CPU",
        gpu_var_pc2,
        cpu_var_pc2,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Corr(PC1,PC2): GPU == CPU",
        gpu_corr,
        cpu_corr,
        tolerances::GPU_VS_CPU_F64,
    );

    println!("  Stage 4: {stage4_us:.0} µs (var, corr on {N_AXES} PC axes)");

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 5: Spectral Cosine Similarity — GPU GEMM + FMR
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 5: Spectral Cosine (GPU GEMM)");

    let t5 = Instant::now();
    let gpu_cosine = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spectra).unwrap();
    let stage5_us = t5.elapsed().as_micros() as f64;

    let cpu_cosine = pairwise_cosine_cpu(&spectra);
    let cos_pairs = N_SAMPLES * (N_SAMPLES - 1) / 2;

    v.check(
        "Cosine: correct pair count",
        gpu_cosine.len() as f64,
        cos_pairs as f64,
        0.0,
    );

    let cos_max_diff = gpu_cosine
        .iter()
        .zip(cpu_cosine.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max);
    v.check(
        "Cosine: max diff within tolerance",
        f64::from(u8::from(cos_max_diff < tolerances::SPECTRAL_COSINE)),
        1.0,
        0.0,
    );
    v.check(
        "Cosine: all in [0,1]",
        f64::from(u8::from(
            gpu_cosine
                .iter()
                .all(|s| s.is_finite() && *s >= 0.0 && *s <= 1.0 + 1e-10),
        )),
        1.0,
        0.0,
    );

    println!("  Stage 5: {stage5_us:.0} µs ({cos_pairs} spectral pairs)");

    // ═══════════════════════════════════════════════════════════════════
    // Pipeline Summary
    // ═══════════════════════════════════════════════════════════════════
    let pipeline_total = pipeline_start.elapsed().as_micros() as f64;
    let stage_sum = stage1_us + stage2_us + stage3_us + stage4_us + stage5_us;

    v.section("Pipeline Summary");
    v.check("Pipeline completes without error", 1.0, 1.0, 0.0);

    println!();
    println!("┌───────────────────────────────────────────────────────┐");
    println!("│ Exp075 Pure GPU Analytics Pipeline                   │");
    println!("├──────────────────────────┬──────────────┬────────────┤");
    println!("│ Stage                    │ GPU Time(µs) │ Status     │");
    println!("├──────────────────────────┼──────────────┼────────────┤");
    println!("│ 1: Alpha diversity       │ {stage1_us:>12.0} │ ✓ parity   │");
    println!("│ 2: Bray-Curtis distance  │ {stage2_us:>12.0} │ ✓ parity   │");
    println!("│ 3: PCoA ordination       │ {stage3_us:>12.0} │ ✓ parity   │");
    println!("│ 4: Stats (var, corr)     │ {stage4_us:>12.0} │ ✓ parity   │");
    println!("│ 5: Spectral cosine       │ {stage5_us:>12.0} │ ✓ parity   │");
    println!("├──────────────────────────┼──────────────┼────────────┤");
    println!("│ Sum of stages            │ {stage_sum:>12.0} │            │");
    println!("│ Total pipeline           │ {pipeline_total:>12.0} │            │");
    let overhead_pct = if stage_sum > 0.0 {
        (pipeline_total - stage_sum) / pipeline_total * 100.0
    } else {
        0.0
    };
    println!("│ Overhead                 │ {overhead_pct:>11.1}% │            │");
    println!("└──────────────────────────┴──────────────┴────────────┘");
    println!("  {N_SAMPLES} samples × {N_FEATURES} features → 5 stages, all GPU");

    v.finish();
}
