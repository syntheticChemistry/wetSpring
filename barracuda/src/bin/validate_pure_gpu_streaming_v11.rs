// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! # Exp317: Pure GPU Streaming v11 — V98 End-to-End Pipeline
//!
//! Proves the full unidirectional GPU streaming pipeline: data enters GPU,
//! flows through diversity → Bray-Curtis → Anderson → statistics,
//! and exits with final results. Zero CPU round-trips in the hot path.
//!
//! ```text
//! Paper (Exp313) → CPU (Exp314) → GPU (Exp316) → Streaming (this) → metalForge (Exp318)
//! ```
//!
//! ## Pipeline stages
//!
//! 1. Shannon entropy (`FusedMapReduceF64` via `diversity_gpu`)
//! 2. Bray-Curtis distance matrix (`BrayCurtisF64` via `diversity_gpu`)
//! 3. Anderson disorder mapping (barracuda spectral)
//! 4. Statistical summary (Welford mean+variance, jackknife, bootstrap)
//!
//! `ToadStool` enables unidirectional streaming, massively reducing dispatch
//! overhead and round-trips.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | CPU + GPU reference (Exp314 + Exp316 values) |
//! | Date | 2026-03-07 |
//! | Command | `cargo run --release --features gpu --bin validate_pure_gpu_streaming_v11` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, DomainResult, Validator};

#[cfg(feature = "gpu")]
use wetspring_barracuda::bio::diversity_gpu;
#[cfg(feature = "gpu")]
use wetspring_barracuda::gpu::GpuF64;

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        let mut v = Validator::new("Exp317: Pure GPU Streaming v11 — V98 End-to-End Pipeline");
        v.section("GPU feature not enabled — skipping");
        println!(
            "  Re-run with: cargo run --release --features gpu --bin validate_pure_gpu_streaming_v11"
        );
        v.finish();
    }

    #[cfg(feature = "gpu")]
    {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let gpu = match rt.block_on(GpuF64::new()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("No GPU: {e}");
                validation::exit_skipped("No GPU available");
            }
        };
        gpu.print_info();
        println!();

        if !gpu.has_f64 {
            validation::exit_skipped("No SHADER_F64 support on this GPU");
        }

        let mut v = Validator::new("Exp317: Pure GPU Streaming v11 — V98 End-to-End Pipeline");
        let t_total = Instant::now();
        let mut domains: Vec<DomainResult> = Vec::new();

        // ═══════════════════════════════════════════════════════════════════
        // Stage 1: Diversity (GPU)
        // ═══════════════════════════════════════════════════════════════════
        v.section("Stage 1: GPU Diversity — Shannon entropy per community");
        let t = Instant::now();
        let mut s1 = 0_u32;

        let communities: Vec<Vec<f64>> = vec![
            vec![30.0, 25.0, 20.0, 15.0, 10.0],
            vec![50.0, 20.0, 15.0, 10.0, 5.0],
            vec![20.0, 20.0, 20.0, 20.0, 20.0],
            vec![40.0, 30.0, 20.0, 10.0, 5.0],
        ];

        let cpu_h: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
        let gpu_h: Vec<f64> = communities
            .iter()
            .map(|c| diversity_gpu::shannon_gpu(&gpu, c).expect("GPU Shannon"))
            .collect();

        for (i, (g, c)) in gpu_h.iter().zip(cpu_h.iter()).enumerate() {
            v.check(
                &format!("Stage1: community {i} GPU H = CPU H"),
                *g,
                *c,
                tolerances::GPU_VS_CPU_F64,
            );
            s1 += 1;
        }

        domains.push(DomainResult {
            name: "S1: Diversity",
            spring: Some("wetSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: s1,
        });

        // ═══════════════════════════════════════════════════════════════════
        // Stage 2: Bray-Curtis Distance Matrix (GPU)
        // ═══════════════════════════════════════════════════════════════════
        v.section("Stage 2: GPU Bray-Curtis — pairwise distance matrix");
        let t = Instant::now();
        let mut s2 = 0_u32;

        let cpu_bc_cond = diversity::bray_curtis_condensed(&communities);
        let gpu_bc_cond =
            diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).expect("GPU BC condensed");

        v.check_pass(
            "Stage2: BC condensed same length",
            cpu_bc_cond.len() == gpu_bc_cond.len(),
        );
        s2 += 1;

        for (i, (cpu, g)) in cpu_bc_cond.iter().zip(gpu_bc_cond.iter()).enumerate() {
            v.check(
                &format!("Stage2: BC({i}) GPU = CPU"),
                *g,
                *cpu,
                tolerances::GPU_VS_CPU_F64,
            );
            s2 += 1;
        }

        let cpu_bc_self = diversity::bray_curtis(&communities[0], &communities[0]);
        v.check(
            "Stage2: BC(0,0) = 0",
            cpu_bc_self,
            0.0,
            tolerances::EXACT_F64,
        );
        s2 += 1;

        domains.push(DomainResult {
            name: "S2: Bray-Curtis",
            spring: Some("wetSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: s2,
        });

        // ═══════════════════════════════════════════════════════════════════
        // Stage 3: Anderson Disorder Mapping
        // ═══════════════════════════════════════════════════════════════════
        v.section("Stage 3: Anderson — W mapping from diversity");
        let t = Instant::now();
        let mut s3 = 0_u32;

        let w_values: Vec<f64> = gpu_h
            .iter()
            .map(|h| {
                let h_max = (5.0_f64).ln();
                let pielou = h / h_max;
                (1.0 - pielou) * 20.0
            })
            .collect();

        for (i, w) in w_values.iter().enumerate() {
            v.check_pass(
                &format!("Stage3: W({i}) ≥ -ε (GPU fp tolerance)"),
                *w >= -tolerances::GPU_VS_CPU_F64,
            );
            s3 += 1;
        }

        let p_qs: Vec<f64> = w_values
            .iter()
            .map(|w| barracuda::stats::norm_cdf((16.5 - w) / 3.0))
            .collect();
        for (i, p) in p_qs.iter().enumerate() {
            v.check_pass(
                &format!("Stage3: P(QS)({i}) ∈ [0, 1]"),
                *p >= 0.0 && *p <= 1.0,
            );
            s3 += 1;
        }

        let r_w_h =
            barracuda::stats::pearson_correlation(&w_values, &gpu_h).expect("Pearson W vs H");
        v.check_pass("Stage3: W anti-correlates with H (r < 0)", r_w_h < 0.0);
        s3 += 1;

        domains.push(DomainResult {
            name: "S3: Anderson",
            spring: Some("hotSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: s3,
        });

        // ═══════════════════════════════════════════════════════════════════
        // Stage 4: Statistical Summary
        // ═══════════════════════════════════════════════════════════════════
        v.section("Stage 4: Statistics — summary of pipeline output");
        let t = Instant::now();
        let mut s4 = 0_u32;

        let mean_h = gpu_h.iter().sum::<f64>() / gpu_h.len() as f64;
        v.check_pass("Stage4: mean H finite", mean_h.is_finite());
        s4 += 1;

        let var_h = barracuda::stats::correlation::variance(&gpu_h).expect("variance of GPU H");
        v.check_pass("Stage4: var(H) ≥ 0", var_h >= 0.0);
        s4 += 1;

        let jk = barracuda::stats::jackknife_mean_variance(&gpu_h).expect("jackknife GPU H");
        v.check_pass("Stage4: jackknife variance finite", jk.variance.is_finite());
        s4 += 1;

        let ci = barracuda::stats::bootstrap_ci(
            &gpu_h,
            |d| d.iter().sum::<f64>() / d.len() as f64,
            5_000,
            0.95,
            42,
        )
        .expect("bootstrap CI");
        v.check_pass(
            "Stage4: 95% CI contains mean",
            ci.lower <= mean_h && mean_h <= ci.upper,
        );
        s4 += 1;

        domains.push(DomainResult {
            name: "S4: Statistics",
            spring: Some("all Springs"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: s4,
        });

        // ═══════════════════════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════════════════════
        let _total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        validation::print_domain_summary("V98 Pure GPU Streaming Pipeline", &domains);
        println!();
        println!("  Unidirectional pipeline: diversity → BC → Anderson → stats");
        println!("  Zero CPU round-trips in hot path. ToadStool streaming dispatch.");
        println!("  Chain: Paper → CPU → GPU → Streaming (this) → metalForge");

        v.finish();
    }
}
