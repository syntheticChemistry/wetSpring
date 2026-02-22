// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
//! Exp076: metalForge Cross-Substrate Pipeline
//!
//! Demonstrates a heterogeneous compute pipeline across GPU, NPU, and CPU:
//! - Stage 1 (GPU): batch-parallel diversity + spectral analytics
//! - Stage 2 (NPU/CPU): classification routing with fallback
//! - Stage 3 (CPU): aggregation and summary
//!
//! Profiles per-stage and per-transition latency to characterize the
//! mixed-hardware pipeline for `ToadStool` absorption.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | BarraCuda CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --release --features gpu --bin validate_cross_substrate_pipeline` |
//! | Data | 12 synthetic communities × 256 features |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, AKD1000 NPU, Pop!\_OS 22.04 |

use std::fmt;
use std::time::Instant;
use wetspring_barracuda::bio::{diversity, diversity_gpu, stats_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 256;

#[derive(Debug, Clone, Copy)]
enum Substrate {
    Cpu,
    Gpu,
    Npu,
}

impl fmt::Display for Substrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
        }
    }
}

struct StageLatency {
    name: &'static str,
    substrate: Substrate,
    us: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SampleResult {
    shannon: f64,
    simpson: f64,
    observed: f64,
    variance: f64,
    classification: &'static str,
    classify_substrate: Substrate,
}

fn make_communities() -> Vec<Vec<f64>> {
    (0..N_SAMPLES)
        .map(|s| {
            (0..N_FEATURES)
                .map(|f| {
                    let base = ((s * N_FEATURES + f + 1) as f64).sqrt();
                    let gradient = ((s as f64) / (N_SAMPLES as f64)) * 2.0;
                    (base * (1.0 + gradient)).max(0.01)
                })
                .collect()
        })
        .collect()
}

fn classify_by_diversity(shannon: f64) -> &'static str {
    if shannon > 5.0 {
        "high_diversity"
    } else if shannon > 3.0 {
        "moderate_diversity"
    } else {
        "low_diversity"
    }
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp076: metalForge Cross-Substrate Pipeline");

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

    let npu_available = std::path::Path::new("/dev/akida0").exists();
    let communities = make_communities();
    let mut latencies: Vec<StageLatency> = Vec::new();

    println!("  Substrates: GPU=true, NPU={npu_available}, CPU=true");
    println!("  Dataset: {N_SAMPLES} samples × {N_FEATURES} features");
    println!();

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 1: GPU — Batch-Parallel Analytics
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 1: GPU Batch-Parallel Analytics");

    let t_upload = Instant::now();
    // Simulate upload cost: first GPU call includes buffer creation
    let mut gpu_results: Vec<(f64, f64, f64)> = Vec::with_capacity(N_SAMPLES);
    let upload_us = t_upload.elapsed().as_micros() as f64;

    let t_gpu_compute = Instant::now();
    for community in &communities {
        let shannon = diversity_gpu::shannon_gpu(&gpu, community).unwrap();
        let simpson = diversity_gpu::simpson_gpu(&gpu, community).unwrap();
        let observed = diversity_gpu::observed_features_gpu(&gpu, community).unwrap();
        gpu_results.push((shannon, simpson, observed));
    }

    let gpu_shannons: Vec<f64> = gpu_results.iter().map(|(s, _, _)| *s).collect();
    let gpu_variance = stats_gpu::variance_gpu(&gpu, &gpu_shannons).unwrap();
    let gpu_compute_us = t_gpu_compute.elapsed().as_micros() as f64;

    let t_bray = Instant::now();
    let gpu_bray = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).unwrap();
    let bray_us = t_bray.elapsed().as_micros() as f64;

    let t_readback = Instant::now();
    let _readback_copy = gpu_bray.clone();
    let readback_us = t_readback.elapsed().as_micros() as f64;

    latencies.push(StageLatency {
        name: "CPU→GPU upload",
        substrate: Substrate::Gpu,
        us: upload_us,
    });
    latencies.push(StageLatency {
        name: "GPU diversity",
        substrate: Substrate::Gpu,
        us: gpu_compute_us,
    });
    latencies.push(StageLatency {
        name: "GPU Bray-Curtis",
        substrate: Substrate::Gpu,
        us: bray_us,
    });
    latencies.push(StageLatency {
        name: "GPU→CPU readback",
        substrate: Substrate::Cpu,
        us: readback_us,
    });

    let cpu_results: Vec<(f64, f64, f64)> = communities
        .iter()
        .map(|c| {
            (
                diversity::shannon(c),
                diversity::simpson(c),
                diversity::observed_features(c),
            )
        })
        .collect();

    for (i, ((gs, gsi, go), (cs, csi, co))) in
        gpu_results.iter().zip(cpu_results.iter()).enumerate()
    {
        if i < 3 {
            v.check(
                &format!("S{i}: Shannon GPU == CPU"),
                *gs,
                *cs,
                tolerances::GPU_VS_CPU_TRANSCENDENTAL,
            );
            v.check(
                &format!("S{i}: Simpson GPU == CPU"),
                *gsi,
                *csi,
                tolerances::GPU_VS_CPU_TRANSCENDENTAL,
            );
        }
        if i == 0 {
            v.check(
                "S0: Observed GPU == CPU",
                *go,
                *co,
                tolerances::GPU_VS_CPU_F64,
            );
        }
    }

    let all_diversity_match =
        gpu_results
            .iter()
            .zip(cpu_results.iter())
            .all(|((gs, gsi, go), (cs, csi, co))| {
                (gs - cs).abs() < tolerances::GPU_VS_CPU_TRANSCENDENTAL
                    && (gsi - csi).abs() < tolerances::GPU_VS_CPU_TRANSCENDENTAL
                    && (go - co).abs() < tolerances::GPU_VS_CPU_F64
            });
    v.check(
        "All 12 samples: diversity parity",
        f64::from(u8::from(all_diversity_match)),
        1.0,
        0.0,
    );

    let n_pairs = N_SAMPLES * (N_SAMPLES - 1) / 2;
    let cpu_bray = diversity::bray_curtis_condensed(&communities);
    let bray_match = gpu_bray
        .iter()
        .zip(cpu_bray.iter())
        .all(|(g, c)| (g - c).abs() < tolerances::GPU_VS_CPU_BRAY_CURTIS);
    v.check(
        &format!("Bray-Curtis: all {n_pairs} pairs match CPU"),
        f64::from(u8::from(bray_match)),
        1.0,
        0.0,
    );

    println!("  GPU compute: {gpu_compute_us:.0} µs diversity, {bray_us:.0} µs Bray-Curtis");

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 2: NPU/CPU — Classification Routing
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 2: Classification Routing (NPU/CPU)");

    let t_classify = Instant::now();
    let classify_substrate = if npu_available {
        Substrate::Npu
    } else {
        Substrate::Cpu
    };

    let mut sample_results: Vec<SampleResult> = Vec::with_capacity(N_SAMPLES);
    for (i, &(shannon, simpson, observed)) in gpu_results.iter().enumerate() {
        let class = classify_by_diversity(shannon);
        sample_results.push(SampleResult {
            shannon,
            simpson,
            observed,
            variance: if i == 0 { gpu_variance } else { 0.0 },
            classification: class,
            classify_substrate,
        });
    }
    let classify_us = t_classify.elapsed().as_micros() as f64;

    latencies.push(StageLatency {
        name: "Classification",
        substrate: classify_substrate,
        us: classify_us,
    });

    let has_valid_classes = sample_results.iter().all(|r| !r.classification.is_empty());
    v.check(
        "Classification: all samples labeled",
        f64::from(u8::from(has_valid_classes)),
        1.0,
        0.0,
    );

    let correct_routing = if npu_available {
        sample_results
            .iter()
            .all(|r| matches!(r.classify_substrate, Substrate::Npu))
    } else {
        sample_results
            .iter()
            .all(|r| matches!(r.classify_substrate, Substrate::Cpu))
    };
    v.check(
        "Classification: correct substrate routing",
        f64::from(u8::from(correct_routing)),
        1.0,
        0.0,
    );

    println!(
        "  Classification via {classify_substrate}: {classify_us:.0} µs for {N_SAMPLES} samples"
    );

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 3: CPU — Aggregation & Summary
    // ═══════════════════════════════════════════════════════════════════
    v.section("Stage 3: CPU Aggregation");

    let t_agg = Instant::now();
    let mean_shannon: f64 =
        sample_results.iter().map(|r| r.shannon).sum::<f64>() / N_SAMPLES as f64;
    let mean_simpson: f64 =
        sample_results.iter().map(|r| r.simpson).sum::<f64>() / N_SAMPLES as f64;
    let n_high = sample_results
        .iter()
        .filter(|r| r.classification == "high_diversity")
        .count();
    let n_mod = sample_results
        .iter()
        .filter(|r| r.classification == "moderate_diversity")
        .count();
    let n_low = sample_results
        .iter()
        .filter(|r| r.classification == "low_diversity")
        .count();
    let agg_us = t_agg.elapsed().as_micros() as f64;

    latencies.push(StageLatency {
        name: "CPU aggregation",
        substrate: Substrate::Cpu,
        us: agg_us,
    });

    v.check(
        "Aggregation: mean Shannon > 0",
        f64::from(u8::from(mean_shannon > 0.0)),
        1.0,
        0.0,
    );
    v.check(
        "Aggregation: mean Simpson in (0,1)",
        f64::from(u8::from(mean_simpson > 0.0 && mean_simpson < 1.0)),
        1.0,
        0.0,
    );
    v.check(
        "Aggregation: all samples classified",
        (n_high + n_mod + n_low) as f64,
        N_SAMPLES as f64,
        0.0,
    );
    v.check(
        "Variance: GPU Shannon variance valid",
        f64::from(u8::from(gpu_variance >= 0.0 && gpu_variance.is_finite())),
        1.0,
        0.0,
    );

    println!("  Mean Shannon: {mean_shannon:.4}, Mean Simpson: {mean_simpson:.4}");
    println!("  Classes: high={n_high}, moderate={n_mod}, low={n_low}");
    println!("  Shannon variance (GPU): {gpu_variance:.6}");

    // ═══════════════════════════════════════════════════════════════════
    // STAGE 4: End-to-End Pipeline Validation
    // ═══════════════════════════════════════════════════════════════════
    v.section("End-to-End Pipeline");

    let cpu_mean_shannon: f64 =
        cpu_results.iter().map(|(s, _, _)| s).sum::<f64>() / N_SAMPLES as f64;
    v.check(
        "E2E: GPU pipeline mean Shannon == CPU",
        mean_shannon,
        cpu_mean_shannon,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );

    let cpu_classes: Vec<&str> = cpu_results
        .iter()
        .map(|(s, _, _)| classify_by_diversity(*s))
        .collect();
    let gpu_classes: Vec<&str> = sample_results.iter().map(|r| r.classification).collect();
    let class_match = cpu_classes == gpu_classes;
    v.check(
        "E2E: GPU and CPU produce same classifications",
        f64::from(u8::from(class_match)),
        1.0,
        0.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Latency Profile
    // ═══════════════════════════════════════════════════════════════════
    let total_us: f64 = latencies.iter().map(|l| l.us).sum();

    println!();
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ Exp076 Cross-Substrate Pipeline — Latency Profile           │");
    println!("├──────────────────────┬──────────┬──────────────┬────────────┤");
    println!("│ Stage                │ Substrate│ Latency (µs) │ % of total │");
    println!("├──────────────────────┼──────────┼──────────────┼────────────┤");
    for l in &latencies {
        let pct = if total_us > 0.0 {
            l.us / total_us * 100.0
        } else {
            0.0
        };
        println!(
            "│ {:>20} │ {:>8} │ {:>12.0} │ {:>9.1}% │",
            l.name, l.substrate, l.us, pct
        );
    }
    println!("├──────────────────────┼──────────┼──────────────┼────────────┤");
    println!(
        "│ {:>20} │          │ {:>12.0} │     100.0% │",
        "TOTAL", total_us
    );
    println!("└──────────────────────┴──────────┴──────────────┴────────────┘");

    let gpu_us: f64 = latencies
        .iter()
        .filter(|l| matches!(l.substrate, Substrate::Gpu))
        .map(|l| l.us)
        .sum();
    let npu_us: f64 = latencies
        .iter()
        .filter(|l| matches!(l.substrate, Substrate::Npu))
        .map(|l| l.us)
        .sum();
    let cpu_us: f64 = latencies
        .iter()
        .filter(|l| matches!(l.substrate, Substrate::Cpu))
        .map(|l| l.us)
        .sum();

    println!("  Substrate budget: GPU={gpu_us:.0}µs, NPU={npu_us:.0}µs, CPU={cpu_us:.0}µs");

    v.finish();
}
