// SPDX-License-Identifier: AGPL-3.0-or-later
// # Provenance
// Experiment: 167
// Date: February 25, 2026
// Purpose: Validate diversity_fusion_f64.wgsl extension — CPU ↔ GPU parity
// Python baseline: scipy/skbio Shannon/Simpson/Pielou (analytically verified)
// Data: Synthetic abundance vectors (deterministic, no external data)
// Command: cargo run --features gpu --bin validate_gpu_diversity_fusion
//
// This validates the Write-phase diversity fusion WGSL extension following
// hotSpring's absorption pattern: local shader + CPU reference + parity check.

use std::process;
use std::sync::Arc;

fn main() {
    println!("=== Exp167: Diversity Fusion GPU Extension (Write Phase) ===");
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let gpu = rt
        .block_on(wetspring_barracuda::gpu::GpuF64::new())
        .expect("GPU init");
    let device = Arc::new(gpu.to_wgpu_device());

    let fusion = match wetspring_barracuda::bio::diversity_fusion_gpu::DiversityFusionGpu::new(
        Arc::clone(&device),
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("FAIL: shader compilation: {e}");
            process::exit(1);
        }
    };

    let mut checks = 0_u32;
    let mut passed = 0_u32;

    // ── Test 1: Uniform distribution (4 species, 2 samples) ──────────
    {
        let n_species = 4;
        let abundances = vec![
            25.0, 25.0, 25.0, 25.0, // sample 0: uniform
            100.0, 0.0, 0.0, 0.0, // sample 1: single species
        ];
        let n_samples = 2;

        let cpu = wetspring_barracuda::bio::diversity_fusion_gpu::diversity_fusion_cpu(
            &abundances,
            n_species,
        );
        let gpu_result = fusion
            .compute(&abundances, n_samples, n_species)
            .expect("GPU compute");

        // CPU uses hardware f64 ln(); GPU uses log_f64 polyfill (~1e-8 precision).
        // Simpson uses only mul/sub → exact parity (1e-12).
        let cpu_tol = 1e-12;
        let gpu_log_tol = 1e-7;

        // Sample 0: uniform → max diversity
        let expected_shannon = 4.0_f64.ln();
        println!(
            "    CPU Shannon={:.15e}, GPU Shannon={:.15e}, diff={:.3e}",
            cpu[0].shannon,
            gpu_result[0].shannon,
            (cpu[0].shannon - gpu_result[0].shannon).abs()
        );
        check(
            &mut checks,
            &mut passed,
            "uniform Shannon (CPU)",
            (cpu[0].shannon - expected_shannon).abs() < cpu_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "uniform Shannon (GPU, log_f64 polyfill)",
            (gpu_result[0].shannon - expected_shannon).abs() < gpu_log_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "uniform Shannon CPU↔GPU (log_f64 tol)",
            (cpu[0].shannon - gpu_result[0].shannon).abs() < gpu_log_tol,
        );

        check(
            &mut checks,
            &mut passed,
            "uniform Simpson (CPU)",
            (cpu[0].simpson - 0.75).abs() < cpu_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "uniform Simpson CPU↔GPU (exact parity)",
            (cpu[0].simpson - gpu_result[0].simpson).abs() < cpu_tol,
        );

        check(
            &mut checks,
            &mut passed,
            "uniform evenness == 1.0 (CPU)",
            (cpu[0].evenness - 1.0).abs() < cpu_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "uniform evenness CPU↔GPU (log_f64 tol)",
            (cpu[0].evenness - gpu_result[0].evenness).abs() < gpu_log_tol,
        );

        // Sample 1: single species → zero diversity
        check(
            &mut checks,
            &mut passed,
            "single-species Shannon == 0 (CPU)",
            cpu[1].shannon.abs() < cpu_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "single-species Shannon CPU↔GPU",
            (cpu[1].shannon - gpu_result[1].shannon).abs() < cpu_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "single-species Simpson == 0 (CPU)",
            cpu[1].simpson.abs() < cpu_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "single-species Simpson CPU↔GPU",
            (cpu[1].simpson - gpu_result[1].simpson).abs() < cpu_tol,
        );
    }

    // ── Test 2: Skewed distribution (ecological realism) ────────────
    {
        let n_species = 6;
        let abundances = vec![
            50.0, 30.0, 10.0, 5.0, 3.0, 2.0, // skewed community
        ];
        let n_samples = 1;

        let cpu = wetspring_barracuda::bio::diversity_fusion_gpu::diversity_fusion_cpu(
            &abundances,
            n_species,
        );
        let gpu_result = fusion
            .compute(&abundances, n_samples, n_species)
            .expect("GPU compute");

        let gpu_log_tol = 1e-7;
        let cpu_tol = 1e-12;

        check(
            &mut checks,
            &mut passed,
            "skewed Shannon CPU↔GPU (log_f64 tol)",
            (cpu[0].shannon - gpu_result[0].shannon).abs() < gpu_log_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "skewed Simpson CPU↔GPU (exact parity)",
            (cpu[0].simpson - gpu_result[0].simpson).abs() < cpu_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "skewed evenness CPU↔GPU (log_f64 tol)",
            (cpu[0].evenness - gpu_result[0].evenness).abs() < gpu_log_tol,
        );
        check(
            &mut checks,
            &mut passed,
            "skewed Shannon > 0",
            cpu[0].shannon > 0.0,
        );
        check(
            &mut checks,
            &mut passed,
            "skewed evenness < 1.0",
            cpu[0].evenness < 1.0 && cpu[0].evenness > 0.0,
        );
    }

    // ── Test 3: Batch of 128 samples (GPU dispatch sizing) ──────────
    {
        let n_species = 10;
        let n_samples = 128;
        let mut abundances = Vec::with_capacity(n_samples * n_species);
        for i in 0..n_samples {
            for j in 0..n_species {
                abundances.push(((i * n_species + j + 1) as f64).sqrt());
            }
        }

        let cpu = wetspring_barracuda::bio::diversity_fusion_gpu::diversity_fusion_cpu(
            &abundances,
            n_species,
        );
        let gpu_result = fusion
            .compute(&abundances, n_samples, n_species)
            .expect("GPU compute");

        let gpu_log_tol = 1e-7;
        let cpu_tol = 1e-12;
        let mut all_parity = true;
        for i in 0..n_samples {
            if (cpu[i].shannon - gpu_result[i].shannon).abs() > gpu_log_tol
                || (cpu[i].simpson - gpu_result[i].simpson).abs() > cpu_tol
                || (cpu[i].evenness - gpu_result[i].evenness).abs() > gpu_log_tol
            {
                all_parity = false;
                eprintln!("  mismatch at sample {i}");
            }
        }
        check(
            &mut checks,
            &mut passed,
            "128-sample batch CPU↔GPU parity",
            all_parity,
        );
        check(
            &mut checks,
            &mut passed,
            "all 128 Shannon finite",
            gpu_result.iter().all(|r| r.shannon.is_finite()),
        );
    }

    // ── Summary ─────────────────────────────────────────────────────
    println!();
    println!("=== Exp167 Summary: {passed}/{checks} checks passed ===");
    println!();
    println!("Extension pattern (hotSpring absorption model):");
    println!("  WGSL:  barracuda/src/bio/shaders/diversity_fusion_f64.wgsl");
    println!("  Rust:  barracuda/src/bio/diversity_fusion_gpu.rs");
    println!("  CPU:   diversity_fusion_cpu() — analytical reference");
    println!("  GPU:   DiversityFusionGpu — local pipeline (Write phase)");
    println!("  Next:  handoff → ToadStool absorption → lean");

    if passed == checks {
        println!();
        println!("ALL PASS");
        process::exit(0);
    } else {
        eprintln!();
        eprintln!("FAIL: {}/{checks} passed", passed);
        process::exit(1);
    }
}

fn check(checks: &mut u32, passed: &mut u32, name: &str, ok: bool) {
    *checks += 1;
    if ok {
        *passed += 1;
        println!("  PASS: {name}");
    } else {
        eprintln!("  FAIL: {name}");
    }
}
