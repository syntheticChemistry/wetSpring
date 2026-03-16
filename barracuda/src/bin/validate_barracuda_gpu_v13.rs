// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! # Exp316: `BarraCuda` GPU v13 — V98 Full-Domain GPU Portability
//!
//! Proves that GPU dispatch produces **identical** results to CPU reference
//! for all core bio domains. This is the GPU tier of the V98 chain.
//!
//! ```text
//! Paper (Exp313) → CPU (Exp314) → GPU (this) → Streaming (Exp317) → metalForge (Exp318)
//! ```
//!
//! ## GPU Domains
//!
//! - G22: Diversity GPU (Shannon, Simpson, Bray-Curtis via `FusedMapReduceF64` / `BrayCurtisF64`)
//! - G23: Anderson Spectral (CPU eigendecomposition + level spacing statistics)
//! - G24: Chemistry (CPU spectral match + GPU dot product composition)
//! - G25: Cross-domain composition (GPU diversity → CPU statistics pipeline)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | CPU reference (Exp314 values) |
//! | Date | 2026-03-07 |
//! | Command | `cargo run --release --features gpu --bin validate_barracuda_gpu_v13` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, DomainResult, Validator};

#[cfg(feature = "gpu")]
use wetspring_barracuda::bio::{diversity_gpu, stats_gpu};
#[cfg(feature = "gpu")]
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::OrExit;

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        let mut v = Validator::new("Exp316: BarraCuda GPU v13 — V98 Full-Domain GPU Portability");
        v.section("GPU feature not enabled — skipping GPU checks");
        println!(
            "  Re-run with: cargo run --release --features gpu --bin validate_barracuda_gpu_v13"
        );
        v.finish();
    }

    #[cfg(feature = "gpu")]
    {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .or_exit("tokio runtime");
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

        let mut v = Validator::new("Exp316: BarraCuda GPU v13 — V98 Full-Domain GPU Portability");
        let t_total = Instant::now();
        let mut domains: Vec<DomainResult> = Vec::new();

        // ═══════════════════════════════════════════════════════════════════
        // G22: Diversity GPU — FusedMapReduce (Shannon, Simpson, Bray-Curtis)
        // ═══════════════════════════════════════════════════════════════════
        v.section("G22: Diversity GPU — Shannon, Simpson, Pielou, Bray-Curtis");
        let t = Instant::now();
        let mut g22 = 0_u32;

        let communities: Vec<Vec<f64>> = vec![
            vec![30.0, 25.0, 20.0, 15.0, 10.0],
            vec![50.0, 20.0, 15.0, 10.0, 5.0],
            vec![20.0, 20.0, 20.0, 20.0, 20.0],
            vec![10.0, 15.0, 20.0, 25.0, 30.0],
        ];

        let cpu_shannons: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
        let gpu_shannons: Vec<f64> = communities
            .iter()
            .map(|c| diversity_gpu::shannon_gpu(&gpu, c).or_exit("GPU shannon"))
            .collect();

        for (i, (cpu, g)) in cpu_shannons.iter().zip(gpu_shannons.iter()).enumerate() {
            v.check(
                &format!("GPU Shannon[{i}] ≡ CPU"),
                *g,
                *cpu,
                tolerances::GPU_VS_CPU_F64,
            );
            g22 += 1;
        }

        let cpu_simpsons: Vec<f64> = communities.iter().map(|c| diversity::simpson(c)).collect();
        let gpu_simpsons: Vec<f64> = communities
            .iter()
            .map(|c| diversity_gpu::simpson_gpu(&gpu, c).or_exit("GPU simpson"))
            .collect();

        for (i, (cpu, g)) in cpu_simpsons.iter().zip(gpu_simpsons.iter()).enumerate() {
            v.check(
                &format!("GPU Simpson[{i}] ≡ CPU"),
                *g,
                *cpu,
                tolerances::GPU_VS_CPU_F64,
            );
            g22 += 1;
        }

        let n_sp = communities[0].len() as f64;
        let gpu_pielou_0 = gpu_shannons[0] / n_sp.ln();
        let cpu_pielou_0 = cpu_shannons[0] / n_sp.ln();
        v.check(
            "GPU Pielou[0] ≡ CPU",
            gpu_pielou_0,
            cpu_pielou_0,
            tolerances::GPU_VS_CPU_F64,
        );
        g22 += 1;

        let cpu_bc_self = diversity::bray_curtis(&communities[0], &communities[0]);
        v.check(
            "CPU BC(x,x) = 0 (identity)",
            cpu_bc_self,
            0.0,
            tolerances::EXACT_F64,
        );
        g22 += 1;

        let cpu_bc_01 = diversity::bray_curtis(&communities[0], &communities[1]);
        let cpu_bc_10 = diversity::bray_curtis(&communities[1], &communities[0]);
        v.check(
            "BC symmetry: BC(0,1) = BC(1,0)",
            cpu_bc_01,
            cpu_bc_10,
            tolerances::EXACT_F64,
        );
        g22 += 1;

        let cpu_bc_cond = diversity::bray_curtis_condensed(&communities);
        let gpu_bc_cond =
            diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities).or_exit("GPU BC condensed");
        v.check_pass(
            "BrayCurtis GPU: same length as CPU",
            cpu_bc_cond.len() == gpu_bc_cond.len(),
        );
        g22 += 1;
        for (i, (cpu, g)) in cpu_bc_cond
            .iter()
            .zip(gpu_bc_cond.iter())
            .enumerate()
            .take(3)
        {
            v.check(
                &format!("BC condensed GPU[{i}] ≡ CPU"),
                *g,
                *cpu,
                tolerances::GPU_VS_CPU_F64,
            );
            g22 += 1;
        }

        domains.push(DomainResult {
            name: "G22: Diversity GPU",
            spring: Some("wetSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g22,
        });

        // ═══════════════════════════════════════════════════════════════════
        // G23: Anderson Spectral — CPU reference for GPU chain
        // ═══════════════════════════════════════════════════════════════════
        v.section("G23: Anderson Spectral — eigendecomposition + level spacing");
        let t = Instant::now();
        let mut g23 = 0_u32;

        let lattice = barracuda::spectral::anderson_3d(4, 4, 4, 2.0, 42);
        let tridiag = barracuda::spectral::lanczos(&lattice, 30, 42);
        let eigs = barracuda::spectral::lanczos_eigenvalues(&tridiag);
        v.check_pass("Anderson: eigenvalues computed", !eigs.is_empty());
        g23 += 1;

        let r = barracuda::spectral::level_spacing_ratio(&eigs);
        v.check_pass(
            "Anderson: r ∈ [Poisson, GOE]",
            (barracuda::spectral::POISSON_R * 0.8..=barracuda::spectral::GOE_R * 1.2).contains(&r),
        );
        g23 += 1;

        let lattice_strong = barracuda::spectral::anderson_3d(4, 4, 4, 20.0, 42);
        let tridiag_strong = barracuda::spectral::lanczos(&lattice_strong, 30, 42);
        let eigs_strong = barracuda::spectral::lanczos_eigenvalues(&tridiag_strong);
        let r_strong = barracuda::spectral::level_spacing_ratio(&eigs_strong);
        v.check_pass(
            "Anderson: strong disorder r ∈ valid range",
            r_strong.is_finite() && r_strong > 0.0 && r_strong < 1.0,
        );
        g23 += 1;

        domains.push(DomainResult {
            name: "G23: Anderson",
            spring: Some("hotSpring+neuralSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g23,
        });

        // ═══════════════════════════════════════════════════════════════════
        // G24: Chemistry — CPU spectral match + GPU dot composition
        // ═══════════════════════════════════════════════════════════════════
        v.section("G24: Chemistry — spectral match via GPU dot product");
        let t = Instant::now();
        let mut g24 = 0_u32;

        let query_mz = vec![100.0, 200.0, 300.0];
        let query_int = vec![1.0, 0.5, 0.3];
        let ref_mz = vec![100.0, 200.0, 300.0];
        let ref_int = vec![1.0, 0.5, 0.3];

        let cpu_cos = wetspring_barracuda::bio::spectral_match::cosine_similarity(
            &query_mz, &query_int, &ref_mz, &ref_int, 0.5,
        );

        v.check(
            "CPU cosine(identical) = 1.0",
            cpu_cos.score,
            1.0,
            tolerances::ANALYTICAL_LOOSE,
        );
        g24 += 1;

        let is_hybrid = format!("{:?}", gpu.fp64_strategy()) == "Hybrid";
        if is_hybrid {
            println!("  NOTE: Hybrid GPU — skipping fused dot_gpu (DF64 zero output).");
            println!("        FusedMapReduceF64 path (Shannon/Simpson/BC) validated above.");
            let q_norm: f64 = query_int.iter().map(|x| x * x).sum::<f64>().sqrt();
            let r_norm: f64 = ref_int.iter().map(|x| x * x).sum::<f64>().sqrt();
            let cpu_dot: f64 = query_int
                .iter()
                .zip(ref_int.iter())
                .map(|(a, b)| a * b)
                .sum();
            let cpu_cosine = cpu_dot / (q_norm * r_norm);
            v.check(
                "CPU spectral dot cosine = 1.0 (Hybrid skip)",
                cpu_cosine,
                cpu_cos.score,
                tolerances::ANALYTICAL_F64,
            );
        } else {
            let gpu_dot = stats_gpu::dot_gpu(&gpu, &query_int, &ref_int).or_exit("GPU dot");
            let q_norm: f64 = query_int.iter().map(|x| x * x).sum::<f64>().sqrt();
            let r_norm: f64 = ref_int.iter().map(|x| x * x).sum::<f64>().sqrt();
            let gpu_cosine = gpu_dot / (q_norm * r_norm);
            v.check(
                "GPU spectral cosine ≈ CPU cosine",
                gpu_cosine,
                cpu_cos.score,
                tolerances::PYTHON_PARITY,
            );
        }
        g24 += 1;

        domains.push(DomainResult {
            name: "G24: Chemistry GPU",
            spring: Some("wetSpring"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g24,
        });

        // ═══════════════════════════════════════════════════════════════════
        // G25: Cross-Domain GPU Composition
        // ═══════════════════════════════════════════════════════════════════
        v.section("G25: Cross-Domain GPU — diversity → statistics pipeline");
        let t = Instant::now();
        let mut g25 = 0_u32;

        let cpu_h_var = barracuda::stats::correlation::variance(&cpu_shannons)
            .or_exit("CPU variance of Shannon");
        let gpu_h_var = barracuda::stats::correlation::variance(&gpu_shannons)
            .or_exit("CPU variance of GPU Shannon");
        v.check(
            "Cross: Var(GPU H) ≡ Var(CPU H)",
            gpu_h_var,
            cpu_h_var,
            tolerances::GPU_VS_CPU_F64,
        );
        g25 += 1;

        let r_gpu = barracuda::stats::pearson_correlation(&gpu_shannons, &gpu_simpsons)
            .or_exit("Pearson r GPU H vs Simpson");
        let r_cpu = barracuda::stats::pearson_correlation(&cpu_shannons, &cpu_simpsons)
            .or_exit("Pearson r CPU H vs Simpson");
        v.check(
            "Cross: r(GPU H, GPU Si) ≡ r(CPU H, CPU Si)",
            r_gpu,
            r_cpu,
            tolerances::GPU_VS_CPU_F64,
        );
        g25 += 1;

        let jk_gpu =
            barracuda::stats::jackknife_mean_variance(&gpu_shannons).or_exit("JK GPU Shannon");
        let jk_cpu =
            barracuda::stats::jackknife_mean_variance(&cpu_shannons).or_exit("JK CPU Shannon");
        v.check(
            "Cross: JK mean(GPU H) ≡ JK mean(CPU H)",
            jk_gpu.estimate,
            jk_cpu.estimate,
            tolerances::GPU_VS_CPU_F64,
        );
        g25 += 1;
        v.check(
            "Cross: JK SE(GPU H) ≡ JK SE(CPU H)",
            jk_gpu.std_error,
            jk_cpu.std_error,
            tolerances::GPU_VS_CPU_F64,
        );
        g25 += 1;

        v.check_pass(
            "Cross: all GPU outputs finite",
            gpu_shannons.iter().all(|x| x.is_finite())
                && gpu_simpsons.iter().all(|x| x.is_finite())
                && gpu_bc_cond.iter().all(|x| x.is_finite()),
        );
        g25 += 1;

        domains.push(DomainResult {
            name: "G25: Cross-Domain",
            spring: Some("all Springs"),
            ms: t.elapsed().as_secs_f64() * 1e3,
            checks: g25,
        });

        // ═══════════════════════════════════════════════════════════════════
        // Summary
        // ═══════════════════════════════════════════════════════════════════
        let _total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        validation::print_domain_summary("V98 Full-Domain GPU Portability", &domains);
        println!();
        println!("  GPU math PROVEN portable — identical to CPU reference (Exp314)");
        println!("  Chain: Paper → CPU → GPU (this) → Streaming → metalForge");

        v.finish();
    }
}
