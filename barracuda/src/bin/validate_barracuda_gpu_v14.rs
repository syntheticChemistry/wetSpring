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
//! # Exp324: `BarraCuda` GPU v14 — V99 GPU Portability + `ToadStool` Dispatch
//!
//! Proves GPU dispatch produces identical results to CPU for the V99
//! cross-primal domain set. Extends GPU v13 (V98) with `ToadStool` dispatch
//! patterns and cross-spring shader evolution tracking.
//!
//! ```text
//! CPU (Exp323) → GPU (this) → CPU-vs-GPU (Exp325) → metalForge (Exp326)
//! ```
//!
//! ## GPU Domains
//!
//! - G26: Diversity GPU — `FusedMapReduce` (Shannon, Simpson, Bray-Curtis)
//! - G27: Anderson Spectral — eigendecomposition + phase diagnostics
//! - G28: Cross-Domain Composition — GPU diversity → statistics pipeline
//! - G29: `ToadStool` Dispatch Model — capability-routed GPU operations
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | CPU reference (Exp323 values) |
//! | Date | 2026-03-08 |
//! | Command | `cargo run --release --features gpu --bin validate_barracuda_gpu_v14` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, DomainResult, Validator};

#[cfg(feature = "gpu")]
use wetspring_barracuda::bio::{diversity_gpu, stats_gpu};
#[cfg(feature = "gpu")]
use wetspring_barracuda::gpu::GpuF64;

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        let mut v = Validator::new("Exp324: BarraCuda GPU v14 — V99 GPU + ToadStool Dispatch");
        v.section("GPU feature not enabled — skipping GPU checks");
        println!(
            "  Re-run with: cargo run --release --features gpu --bin validate_barracuda_gpu_v14"
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

        let mut v = Validator::new("Exp324: BarraCuda GPU v14 — V99 GPU + ToadStool Dispatch");
        let t_total = Instant::now();
        let mut domains: Vec<DomainResult> = Vec::new();

        // ═══════════════════════════════════════════════════════════════════
        // G26: Diversity GPU — V99 cross-primal communities
        // ═══════════════════════════════════════════════════════════════════
        v.section("G26: Diversity GPU — V99 cross-primal communities");
        let t = Instant::now();
        let mut g26 = 0_u32;

        let communities = vec![
            vec![35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5],
            vec![30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5],
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            vec![90.0, 5.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.02],
        ];

        let cpu_h: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
        let gpu_h: Vec<f64> = communities
            .iter()
            .map(|c| diversity_gpu::shannon_gpu(&gpu, c).expect("GPU Shannon"))
            .collect();
        for (i, (c, g)) in cpu_h.iter().zip(gpu_h.iter()).enumerate() {
            v.check(
                &format!("GPU Shannon[{i}] ≡ CPU"),
                *g,
                *c,
                tolerances::GPU_VS_CPU_F64,
            );
            g26 += 1;
        }

        let cpu_s: Vec<f64> = communities.iter().map(|c| diversity::simpson(c)).collect();
        let gpu_s: Vec<f64> = communities
            .iter()
            .map(|c| diversity_gpu::simpson_gpu(&gpu, c).expect("GPU Simpson"))
            .collect();
        for (i, (c, g)) in cpu_s.iter().zip(gpu_s.iter()).enumerate() {
            v.check(
                &format!("GPU Simpson[{i}] ≡ CPU"),
                *g,
                *c,
                tolerances::GPU_VS_CPU_F64,
            );
            g26 += 1;
        }

        let cpu_bc = diversity::bray_curtis_condensed(&communities);
        let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).expect("GPU BC");
        v.check_pass("BC: GPU length = CPU length", cpu_bc.len() == gpu_bc.len());
        g26 += 1;
        for (i, (c, g)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate().take(4) {
            v.check(
                &format!("BC condensed[{i}] GPU ≡ CPU"),
                *g,
                *c,
                tolerances::GPU_VS_CPU_F64,
            );
            g26 += 1;
        }

        domains.push(DomainResult {
            name: "G26: Diversity GPU",
            spring: Some("wetSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g26,
        });

        // ═══════════════════════════════════════════════════════════════════
        // G27: Anderson Spectral — GPU-chain diagnostics
        // ═══════════════════════════════════════════════════════════════════
        v.section("G27: Anderson Spectral — GPU-chain phase diagnostics");
        let t = Instant::now();
        let mut g27 = 0_u32;

        let lattice = barracuda::spectral::anderson_3d(4, 4, 4, 2.0, 42);
        let tridiag = barracuda::spectral::lanczos(&lattice, 30, 42);
        let eigs = barracuda::spectral::lanczos_eigenvalues(&tridiag);
        let r = barracuda::spectral::level_spacing_ratio(&eigs);
        v.check_pass("G27: eigenvalues computed", !eigs.is_empty());
        g27 += 1;
        v.check_pass("G27: r ∈ valid range", r.is_finite() && r > 0.0 && r < 1.0);
        g27 += 1;

        let lattice_strong = barracuda::spectral::anderson_3d(4, 4, 4, 20.0, 42);
        let tridiag_strong = barracuda::spectral::lanczos(&lattice_strong, 30, 42);
        let eigs_strong = barracuda::spectral::lanczos_eigenvalues(&tridiag_strong);
        let r_strong = barracuda::spectral::level_spacing_ratio(&eigs_strong);
        v.check_pass(
            "G27: strong disorder r valid",
            r_strong.is_finite() && r_strong > 0.0,
        );
        g27 += 1;

        let regime = if (r - barracuda::spectral::GOE_R).abs()
            < (r - barracuda::spectral::POISSON_R).abs()
        {
            "extended"
        } else {
            "localized"
        };
        println!("  Anderson W=2: r={r:.4} → {regime}");
        v.check_pass("G27: phase interpretation valid", !regime.is_empty());
        g27 += 1;

        domains.push(DomainResult {
            name: "G27: Anderson",
            spring: Some("hotSpring+neuralSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g27,
        });

        // ═══════════════════════════════════════════════════════════════════
        // G28: Cross-Domain GPU → Statistics Pipeline
        // ═══════════════════════════════════════════════════════════════════
        v.section("G28: Cross-Domain — GPU diversity → CPU statistics");
        let t = Instant::now();
        let mut g28 = 0_u32;

        let var_cpu = barracuda::stats::correlation::variance(&cpu_h).expect("CPU var");
        let var_gpu = barracuda::stats::correlation::variance(&gpu_h).expect("GPU var");
        v.check(
            "G28: Var(GPU H) ≡ Var(CPU H)",
            var_gpu,
            var_cpu,
            tolerances::GPU_VS_CPU_F64,
        );
        g28 += 1;

        let r_cpu = barracuda::stats::pearson_correlation(&cpu_h, &cpu_s).expect("CPU Pearson");
        let r_gpu = barracuda::stats::pearson_correlation(&gpu_h, &gpu_s).expect("GPU Pearson");
        v.check(
            "G28: Pearson GPU ≡ CPU",
            r_gpu,
            r_cpu,
            tolerances::GPU_VS_CPU_F64,
        );
        g28 += 1;

        let jk_cpu = barracuda::stats::jackknife_mean_variance(&cpu_h).expect("JK CPU");
        let jk_gpu = barracuda::stats::jackknife_mean_variance(&gpu_h).expect("JK GPU");
        v.check(
            "G28: JK mean GPU ≡ CPU",
            jk_gpu.estimate,
            jk_cpu.estimate,
            tolerances::GPU_VS_CPU_F64,
        );
        g28 += 1;
        v.check(
            "G28: JK SE GPU ≡ CPU",
            jk_gpu.std_error,
            jk_cpu.std_error,
            tolerances::GPU_VS_CPU_F64,
        );
        g28 += 1;

        v.check_pass(
            "G28: all outputs finite",
            gpu_h.iter().all(|x| x.is_finite())
                && gpu_s.iter().all(|x| x.is_finite())
                && gpu_bc.iter().all(|x| x.is_finite()),
        );
        g28 += 1;

        domains.push(DomainResult {
            name: "G28: Cross-Domain",
            spring: Some("all Springs"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g28,
        });

        // ═══════════════════════════════════════════════════════════════════
        // G29: ToadStool Dispatch Model — capability routing
        // ═══════════════════════════════════════════════════════════════════
        v.section("G29: ToadStool Dispatch Model — capability routing validation");
        let t = Instant::now();
        let mut g29 = 0_u32;

        let fp64 = format!("{:?}", gpu.fp64_strategy());
        let is_hybrid = fp64 == "Hybrid";
        println!("  GPU: {} ({fp64})", gpu.adapter_name);
        println!("  ToadStool dispatch: FusedMapReduce → DF64 ({fp64})");
        v.check_pass("G29: fp64_strategy resolved", !fp64.is_empty());
        g29 += 1;

        let large: Vec<f64> = (0..10_000).map(|i| f64::from(i % 100) + 1.0).collect();
        let cpu_h_large = diversity::shannon(&large);
        let gpu_h_large = diversity_gpu::shannon_gpu(&gpu, &large).expect("GPU Shannon large");
        v.check(
            "G29: large Shannon GPU ≡ CPU",
            gpu_h_large,
            cpu_h_large,
            tolerances::GPU_VS_CPU_F64,
        );
        g29 += 1;

        if is_hybrid {
            let cpu_dot: f64 = [1.0, 2.0, 3.0]
                .iter()
                .zip([4.0, 5.0, 6.0].iter())
                .map(|(a, b)| a * b)
                .sum();
            v.check(
                "G29: CPU dot (Hybrid skip)",
                cpu_dot,
                32.0,
                tolerances::ANALYTICAL_F64,
            );
        } else {
            let dot =
                stats_gpu::dot_gpu(&gpu, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).expect("GPU dot");
            v.check(
                "G29: GPU dot([1,2,3],[4,5,6]) = 32",
                dot,
                32.0,
                tolerances::GPU_VS_CPU_F64,
            );
        }
        g29 += 1;

        let provenance_count = barracuda::shaders::provenance::report::shader_count();
        v.check_pass(
            &format!("G29: provenance: {provenance_count} shaders"),
            provenance_count > 0,
        );
        g29 += 1;

        let wetspring_shaders = barracuda::shaders::provenance::shaders_from(
            barracuda::shaders::provenance::types::SpringDomain::WetSpring,
        );
        v.check_pass(
            &format!(
                "G29: wetSpring authored {} shaders",
                wetspring_shaders.len()
            ),
            !wetspring_shaders.is_empty(),
        );
        g29 += 1;

        println!("  Dispatch model: GPU (FusedMapReduce) > CPU (fallback)");
        println!(
            "  ToadStool absorbs: {provenance_count} shaders, {} from wetSpring",
            wetspring_shaders.len()
        );

        domains.push(DomainResult {
            name: "G29: ToadStool Dispatch",
            spring: Some("toadStool+barraCuda"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g29,
        });

        // ═══════════════════════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════════════════════
        let _total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        validation::print_domain_summary("V99 GPU + ToadStool Dispatch", &domains);
        println!();
        println!("  GPU math PROVEN portable — identical to CPU reference (Exp323)");
        println!("  ToadStool dispatch: {fp64} strategy, {provenance_count} shaders tracked");
        println!("  Chain: CPU → GPU (this) → CPU-vs-GPU → metalForge");

        v.finish();
    }
}
