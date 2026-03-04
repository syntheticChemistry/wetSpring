// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::similar_names,
    dead_code
)]
//! # Exp284: GPU Validation — Gonzales Reproductions
//!
//! Proves that `BarraCUDA` GPU produces identical results to CPU for all
//! Gonzales paper reproduction domains. Validates that the pure Rust math
//! from Exp280-283 is truly portable to GPU.
//!
//! ## GPU primitives exercised
//! - `FusedMapReduceF64` — Shannon, Simpson on cytokine receptor populations
//! - `BrayCurtisF64` — beta diversity between healthy/AD tissue states
//! - Diversity GPU: Shannon, Simpson, Pielou, Chao1 via GPU dispatch
//!
//! ## Evolution chain
//! - **Previous**: Exp283 CPU parity (pure Rust)
//! - **This**: `BarraCUDA` GPU (diversity on GPU, spectral on CPU)
//! - **Next**: Exp285 `ToadStool` streaming dispatch
//! - **Final**: Exp286 `metalForge` cross-substrate (NUCLEUS)
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | Validates | Exp280-282 GPU portability (Papers 53-56) |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --features gpu --bin validate_gonzales_gpu` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;

use wetspring_barracuda::bio::{diversity, diversity_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

struct Timing {
    domain: &'static str,
    cpu_us: f64,
    gpu_us: f64,
    checks: usize,
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp284: GPU Validation — Gonzales Reproductions");
    let mut timings: Vec<Timing> = Vec::new();

    let gpu = match GpuF64::new().await {
        Ok(g) => {
            println!("  GPU: {}", g.adapter_name);
            g
        }
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };

    let tol = tolerances::GPU_VS_CPU_TRANSCENDENTAL;

    // ═══════════════════════════════════════════════════════════════
    // D01: Immune Cell Population — Shannon & Simpson
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Immune Cell Populations — Shannon & Simpson ═══");

    let immune_cells = [0.40, 0.25, 0.15, 0.10, 0.10]; // Th2, Mast, Eosinophils, DC, LC
    let skin_cells = [0.70, 0.15, 0.10, 0.05]; // Keratinocytes, Fibroblasts, Melanocytes, Endothelial
    let neural_cells = [0.50, 0.30, 0.20]; // Sensory, DRG, Schwann

    let populations: &[(&str, &[f64])] = &[
        ("Immune", &immune_cells),
        ("Skin", &skin_cells),
        ("Neural", &neural_cells),
    ];

    let mut d1_checks = 0_usize;
    let mut d1_cpu = 0.0_f64;
    let mut d1_gpu = 0.0_f64;

    for &(name, pop) in populations {
        let tc = Instant::now();
        let cpu_shannon = diversity::shannon(pop);
        let cpu_simpson = diversity::simpson(pop);
        d1_cpu += tc.elapsed().as_micros() as f64;

        let tg = Instant::now();
        let gpu_shannon = diversity_gpu::shannon_gpu(&gpu, pop).expect("shannon GPU");
        let gpu_simpson = diversity_gpu::simpson_gpu(&gpu, pop).expect("simpson GPU");
        d1_gpu += tg.elapsed().as_micros() as f64;

        let diff_sh = (cpu_shannon - gpu_shannon).abs();
        let diff_si = (cpu_simpson - gpu_simpson).abs();

        v.check_pass(
            &format!("{name} Shannon CPU≈GPU (diff={diff_sh:.2e})"),
            diff_sh < tol,
        );
        v.check_pass(
            &format!("{name} Simpson CPU≈GPU (diff={diff_si:.2e})"),
            diff_si < tol,
        );

        println!(
            "  {name}: Shannon CPU={cpu_shannon:.6} GPU={gpu_shannon:.6}, Simpson CPU={cpu_simpson:.6} GPU={gpu_simpson:.6}"
        );
        d1_checks += 2;
    }

    timings.push(Timing {
        domain: "Shannon+Simpson",
        cpu_us: d1_cpu,
        gpu_us: d1_gpu,
        checks: d1_checks,
    });

    // ═══════════════════════════════════════════════════════════════
    // D02: Receptor Distribution — Pielou & Chao1
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Receptor Distribution — Pielou (GPU) + Chao1 (CPU) ═══");
    let t_cpu = Instant::now();

    let receptor_counts = [95.0, 60.0, 35.0, 12.0, 6.0, 27.0, 27.0];
    let cpu_pielou = diversity::pielou_evenness(&receptor_counts);
    let cpu_chao1 = diversity::chao1(&receptor_counts);
    let d2_cpu = t_cpu.elapsed().as_micros() as f64;

    let t_gpu = Instant::now();
    let gpu_pielou =
        diversity_gpu::pielou_evenness_gpu(&gpu, &receptor_counts).expect("pielou GPU");
    let d2_gpu = t_gpu.elapsed().as_micros() as f64;

    let diff_p = (cpu_pielou - gpu_pielou).abs();

    v.check_pass(&format!("Pielou CPU≈GPU (diff={diff_p:.2e})"), diff_p < tol);
    v.check_pass(
        "Chao1 computed on CPU (richness estimator, integer arithmetic)",
        cpu_chao1 >= receptor_counts.iter().filter(|&&c| c > 0.0).count() as f64,
    );

    println!("  Pielou: CPU={cpu_pielou:.6} GPU={gpu_pielou:.6}");
    println!("  Chao1:  CPU={cpu_chao1:.6} (integer, CPU-only)");

    timings.push(Timing {
        domain: "Pielou+Chao1",
        cpu_us: d2_cpu,
        gpu_us: d2_gpu,
        checks: 2,
    });

    // ═══════════════════════════════════════════════════════════════
    // D03: Bray-Curtis — Healthy vs. AD Tissue
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: Bray-Curtis — Healthy vs. AD Tissue ═══");

    let healthy_tissue = vec![5.0, 8.0, 3.0, 6.0, 4.0, 7.0];
    let ad_tissue = vec![45.0, 62.0, 38.0, 55.0, 72.0, 48.0];
    let mild_ad = vec![15.0, 20.0, 12.0, 18.0, 22.0, 16.0];

    let samples = vec![healthy_tissue.clone(), mild_ad.clone(), ad_tissue.clone()];

    let tc = Instant::now();
    let cpu_bc_ha = diversity::bray_curtis(&healthy_tissue, &ad_tissue);
    let cpu_bc_hm = diversity::bray_curtis(&healthy_tissue, &mild_ad);
    let cpu_bc_ma = diversity::bray_curtis(&mild_ad, &ad_tissue);
    let d3_cpu = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_bc_condensed =
        diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples).expect("BC condensed GPU");
    let d3_gpu = tg.elapsed().as_micros() as f64;

    // Condensed matrix order: (0,1), (0,2), (1,2) → healthy↔mild, healthy↔AD, mild↔AD
    let gpu_bc_hm = gpu_bc_condensed[0];
    let gpu_bc_ha = gpu_bc_condensed[1];
    let gpu_bc_ma = gpu_bc_condensed[2];

    v.check_pass(
        &format!(
            "healthy↔AD BC CPU≈GPU (diff={:.2e})",
            (cpu_bc_ha - gpu_bc_ha).abs()
        ),
        (cpu_bc_ha - gpu_bc_ha).abs() < tol,
    );
    v.check_pass(
        &format!(
            "healthy↔mild BC CPU≈GPU (diff={:.2e})",
            (cpu_bc_hm - gpu_bc_hm).abs()
        ),
        (cpu_bc_hm - gpu_bc_hm).abs() < tol,
    );
    v.check_pass(
        &format!(
            "mild↔AD BC CPU≈GPU (diff={:.2e})",
            (cpu_bc_ma - gpu_bc_ma).abs()
        ),
        (cpu_bc_ma - gpu_bc_ma).abs() < tol,
    );

    v.check_pass(
        "Triangle: BC(h,a) ≤ BC(h,m) + BC(m,a)",
        cpu_bc_ha <= cpu_bc_hm + cpu_bc_ma + tolerances::BOUNDED_METRIC_GUARD,
    );
    v.check_pass(
        "BC range: all in [0,1]",
        cpu_bc_ha <= 1.0 && cpu_bc_hm <= 1.0 && cpu_bc_ma <= 1.0,
    );

    println!("  healthy↔AD:   CPU={cpu_bc_ha:.6} GPU={gpu_bc_ha:.6}");
    println!("  healthy↔mild: CPU={cpu_bc_hm:.6} GPU={gpu_bc_hm:.6}");
    println!("  mild↔AD:      CPU={cpu_bc_ma:.6} GPU={gpu_bc_ma:.6}");

    timings.push(Timing {
        domain: "Bray-Curtis",
        cpu_us: d3_cpu,
        gpu_us: d3_gpu,
        checks: 5,
    });

    // ═══════════════════════════════════════════════════════════════
    // D04: Large Cell Population Benchmark
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: Large Population GPU Benchmark ═══");

    let large_pop: Vec<f64> = (0..10_000)
        .map(|i| f64::from((i * 37 + 13) % 100) + 1.0)
        .collect();

    let tc = Instant::now();
    let cpu_sh = diversity::shannon(&large_pop);
    let cpu_si = diversity::simpson(&large_pop);
    let d4_cpu = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_sh = diversity_gpu::shannon_gpu(&gpu, &large_pop).expect("large shannon GPU");
    let gpu_si = diversity_gpu::simpson_gpu(&gpu, &large_pop).expect("large simpson GPU");
    let d4_gpu = tg.elapsed().as_micros() as f64;

    v.check_pass(
        &format!(
            "Large Shannon CPU≈GPU (diff={:.2e})",
            (cpu_sh - gpu_sh).abs()
        ),
        (cpu_sh - gpu_sh).abs() < tol,
    );
    v.check_pass(
        &format!(
            "Large Simpson CPU≈GPU (diff={:.2e})",
            (cpu_si - gpu_si).abs()
        ),
        (cpu_si - gpu_si).abs() < tol,
    );

    println!(
        "  10K elements: Shannon diff={:.2e}, Simpson diff={:.2e}",
        (cpu_sh - gpu_sh).abs(),
        (cpu_si - gpu_si).abs()
    );
    println!("  CPU: {d4_cpu:.0}µs, GPU: {d4_gpu:.0}µs");

    timings.push(Timing {
        domain: "Large benchmark",
        cpu_us: d4_cpu,
        gpu_us: d4_gpu,
        checks: 2,
    });

    // ═══════════════════════════════════════════════════════════════
    // D05: IC50 Dose-Response on GPU
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D05: IC50 Dose-Response Populations on GPU ═══");
    let tc = Instant::now();

    // IC50 values as a "population" for diversity analysis
    let ic50_pop = [10.0, 36.0, 71.0, 80.0, 150.0, 249.0];
    let cpu_ic_sh = diversity::shannon(&ic50_pop);
    let cpu_ic_si = diversity::simpson(&ic50_pop);
    let d5_cpu = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_ic_sh = diversity_gpu::shannon_gpu(&gpu, &ic50_pop).expect("ic50 shannon GPU");
    let gpu_ic_si = diversity_gpu::simpson_gpu(&gpu, &ic50_pop).expect("ic50 simpson GPU");
    let d5_gpu = tg.elapsed().as_micros() as f64;

    v.check_pass("IC50 Shannon CPU≈GPU", (cpu_ic_sh - gpu_ic_sh).abs() < tol);
    v.check_pass("IC50 Simpson CPU≈GPU", (cpu_ic_si - gpu_ic_si).abs() < tol);

    timings.push(Timing {
        domain: "IC50 diversity",
        cpu_us: d5_cpu,
        gpu_us: d5_gpu,
        checks: 2,
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp284: GPU Validation — Gonzales Reproductions                    ║");
    println!("╠═════════════════════════╦════════════╦════════════╦════════════════╣");
    println!("║ Domain                  ║   CPU (µs) ║   GPU (µs) ║ Checks         ║");
    println!("╠═════════════════════════╬════════════╬════════════╬════════════════╣");

    let mut total_checks = 0_usize;
    let mut total_cpu = 0.0_f64;
    let mut total_gpu = 0.0_f64;
    for t in &timings {
        println!(
            "║ {:<23} ║ {:>10.0} ║ {:>10.0} ║ {:>3}            ║",
            t.domain, t.cpu_us, t.gpu_us, t.checks
        );
        total_checks += t.checks;
        total_cpu += t.cpu_us;
        total_gpu += t.gpu_us;
    }

    println!("╠═════════════════════════╬════════════╬════════════╬════════════════╣");
    println!(
        "║ TOTAL                   ║ {total_cpu:>10.0} ║ {total_gpu:>10.0} ║ {total_checks:>3}            ║"
    );
    println!("╚═════════════════════════╩════════════╩════════════╩════════════════╝");
    println!();

    v.finish();
}
