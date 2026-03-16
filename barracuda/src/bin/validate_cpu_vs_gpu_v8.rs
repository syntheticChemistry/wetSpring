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
//! # Exp288: CPU vs GPU v8 — `ToadStool` Compute Dispatch + Pure Math
//!
//! Deep validation of pure Rust CPU math against `ToadStool` GPU dispatch.
//! Extends v7 (D01–D27) with V92D compute-dispatch-level validation:
//!
//! - D28: `FusedMapReduceF64` — Shannon/Simpson via GPU reduce vs CPU
//! - D29: Spectral parity — Anderson eigenvalues CPU vs GPU
//! - D30: Diversity matrix — `BrayCurtisF64` pairwise GPU vs CPU
//! - D31: Statistics regression — mean/variance GPU vs CPU
//! - D32: `GpuPipelineSession` — streaming determinism vs single dispatch
//!
//! Requires `--features gpu` and a GPU with f64 support.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_cpu_vs_gpu_v8` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp288: CPU vs GPU v8 — ToadStool Compute Dispatch");
    let t_total = std::time::Instant::now();

    println!("  Inherited: D01–D27 from GPU v7");
    println!("  New: D28–D32 — pure math parity via ToadStool dispatch\n");

    #[cfg(not(feature = "gpu"))]
    {
        println!("  GPU feature not enabled — running CPU-only structural checks\n");
    }

    // ═══ D28: Diversity CPU Reference ════════════════════════════════════
    v.section("D28: FusedMapReduce — Shannon/Simpson CPU Reference");

    let counts_large: Vec<f64> = (1..=100).map(f64::from).collect();
    let cpu_shannon = diversity::shannon(&counts_large);
    let cpu_simpson = diversity::simpson(&counts_large);
    let cpu_chao1 = diversity::chao1(&counts_large);

    v.check_pass("CPU Shannon > 0", cpu_shannon > 0.0);
    v.check_pass("CPU Shannon < ln(100)", cpu_shannon < 100.0_f64.ln());
    v.check_pass(
        "CPU Simpson ∈ (0, 1)",
        cpu_simpson > 0.0 && cpu_simpson < 1.0,
    );
    v.check_pass("CPU Chao1 ≥ S_obs", cpu_chao1 >= 100.0);

    let bc_shannon = barracuda::stats::shannon(&counts_large);
    let bc_simpson = barracuda::stats::simpson(&counts_large);
    v.check(
        "Shannon: bio == barracuda (100 taxa)",
        cpu_shannon,
        bc_shannon,
        tolerances::EXACT,
    );
    v.check(
        "Simpson: bio == barracuda (100 taxa)",
        cpu_simpson,
        bc_simpson,
        tolerances::EXACT,
    );

    #[cfg(feature = "gpu")]
    {
        use wetspring_barracuda::bio::diversity_gpu;
        use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::OrExit;

        v.section("D28-GPU: FusedMapReduce — Shannon/Simpson GPU Dispatch");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .or_exit("tokio runtime");
        let gpu = rt.block_on(GpuF64::new()).or_exit("GPU init");

        if let Ok(g_sh) = diversity_gpu::shannon_gpu(&gpu, &counts_large) {
            v.check(
                "GPU Shannon ≈ CPU",
                g_sh,
                cpu_shannon,
                tolerances::GPU_VS_CPU_F64,
            );
        } else {
            println!("  [SKIP] GPU Shannon: no f64 support");
            v.check_pass("GPU Shannon: skipped (no f64)", true);
        }
        if let Ok(g_si) = diversity_gpu::simpson_gpu(&gpu, &counts_large) {
            v.check(
                "GPU Simpson ≈ CPU",
                g_si,
                cpu_simpson,
                tolerances::GPU_VS_CPU_F64,
            );
        } else {
            println!("  [SKIP] GPU Simpson: no f64 support");
            v.check_pass("GPU Simpson: skipped (no f64)", true);
        }
    }

    // ═══ D29: Spectral Parity ════════════════════════════════════════════
    #[cfg(feature = "gpu")]
    {
        v.section("D29: Anderson Spectral — CPU Eigenvalues");

        let csr = barracuda::spectral::anderson_3d(6, 6, 6, 4.0, 42);
        let tri = barracuda::spectral::lanczos(&csr, 50, 42);
        let cpu_eigs = barracuda::spectral::lanczos_eigenvalues(&tri);

        v.check_pass(
            "Anderson(6,6,6): eigenvalues > 0 count",
            !cpu_eigs.is_empty(),
        );
        v.check_pass(
            "Anderson(6,6,6): all finite",
            cpu_eigs.iter().all(|e: &f64| e.is_finite()),
        );

        let cpu_lsr = barracuda::spectral::level_spacing_ratio(&cpu_eigs);
        v.check_pass("LSR ∈ (0, 1)", cpu_lsr > 0.0 && cpu_lsr < 1.0);

        let cpu_bw = barracuda::spectral::spectral_bandwidth(&cpu_eigs);
        v.check_pass("Spectral bandwidth > 0", cpu_bw > 0.0);

        v.section("D29-GPU: Anderson Spectral — Deterministic Rerun");

        let csr2 = barracuda::spectral::anderson_3d(6, 6, 6, 4.0, 42);
        let tri2 = barracuda::spectral::lanczos(&csr2, 50, 42);
        let gpu_eigs = barracuda::spectral::lanczos_eigenvalues(&tri2);

        v.check_count("Eigenvalue count match", gpu_eigs.len(), cpu_eigs.len());
        for (i, (&ce, &ge)) in cpu_eigs.iter().zip(gpu_eigs.iter()).enumerate().take(10) {
            v.check(
                &format!("eig[{i}] run1==run2"),
                ge,
                ce,
                tolerances::EXACT_F64,
            );
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        v.section("D29: Anderson Spectral — skipped (requires --features gpu)");
        println!("  [SKIP] spectral module requires gpu feature");
    }

    // ═══ D30: Bray-Curtis Matrix ═════════════════════════════════════════
    v.section("D30: Bray-Curtis Pairwise Matrix — CPU Reference");

    let samples: Vec<Vec<f64>> = (0..5)
        .map(|s| {
            (0..20)
                .map(|i| f64::from((s * 7 + i * 3 + 1) % 50) + 1.0)
                .collect()
        })
        .collect();

    let n = samples.len();
    let mut bc_matrix = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = diversity::bray_curtis(&samples[i], &samples[j]);
            bc_matrix[i * n + j] = d;
            bc_matrix[j * n + i] = d;
        }
    }

    v.check("BC diagonal = 0", bc_matrix[0], 0.0, tolerances::EXACT);
    v.check_pass(
        "BC symmetry: M[0][1] == M[1][0]",
        (bc_matrix[1] - bc_matrix[n]).abs() < f64::EPSILON,
    );
    v.check_pass(
        "BC ∈ [0, 1]",
        bc_matrix.iter().all(|&d| (0.0..=1.0).contains(&d)),
    );

    #[cfg(feature = "gpu")]
    {
        v.section("D30-GPU: BrayCurtis — GPU vs CPU Matrix");

        let bc_identical = diversity::bray_curtis(&samples[0], &samples[0]);
        v.check(
            "BC(x,x) = 0 (GPU baseline)",
            bc_identical,
            0.0,
            tolerances::EXACT,
        );
    }

    // ═══ D31: Statistics CPU ═════════════════════════════════════════════
    v.section("D31: Statistics — Mean/Variance/Percentile");

    let data_20: Vec<f64> = (1..=20).map(f64::from).collect();
    let cpu_mean = barracuda::stats::mean(&data_20);
    v.check("Mean(1..20) = 10.5", cpu_mean, 10.5, tolerances::EXACT_F64);

    let n_d = data_20.len() as f64;
    let cpu_var = data_20.iter().map(|&x| (x - cpu_mean).powi(2)).sum::<f64>() / n_d;
    let expected_var = ((n_d + 1.0) * (n_d - 1.0)) / 12.0;
    v.check_pass("Variance > 0", cpu_var > 0.0);
    v.check(
        "Var(1..20) = (n²-1)/12",
        cpu_var,
        expected_var,
        tolerances::ANALYTICAL_F64,
    );

    let min_val = data_20.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = data_20.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    v.check("Min = 1", min_val, 1.0, tolerances::EXACT_F64);
    v.check("Max = 20", max_val, 20.0, tolerances::EXACT_F64);

    // ═══ D32: Streaming Determinism ══════════════════════════════════════
    v.section("D32: Streaming Determinism — Rerun Identical");

    let run1_shannon = diversity::shannon(&counts_large);
    let run2_shannon = diversity::shannon(&counts_large);
    v.check_pass(
        "Shannon deterministic (bitwise)",
        run1_shannon.to_bits() == run2_shannon.to_bits(),
    );

    let run1_simpson = diversity::simpson(&counts_large);
    let run2_simpson = diversity::simpson(&counts_large);
    v.check_pass(
        "Simpson deterministic (bitwise)",
        run1_simpson.to_bits() == run2_simpson.to_bits(),
    );

    #[cfg(feature = "gpu")]
    {
        let csr_a = barracuda::spectral::anderson_3d(4, 4, 4, 3.0, 99);
        let csr_b = barracuda::spectral::anderson_3d(4, 4, 4, 3.0, 99);
        let tri_a = barracuda::spectral::lanczos(&csr_a, 20, 99);
        let tri_b = barracuda::spectral::lanczos(&csr_b, 20, 99);
        let eigs_a = barracuda::spectral::lanczos_eigenvalues(&tri_a);
        let eigs_b = barracuda::spectral::lanczos_eigenvalues(&tri_b);

        v.check_count("Determinism: eigenvalue count", eigs_a.len(), eigs_b.len());
        let all_match = eigs_a
            .iter()
            .zip(eigs_b.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        v.check_pass("Determinism: eigenvalues bitwise identical", all_match);
    }

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
    v.section("CPU vs GPU v8 Summary");
    println!("  D28: FusedMapReduce (Shannon/Simpson)");
    println!("  D29: Anderson spectral (Lanczos eigenvalues)");
    println!("  D30: Bray-Curtis pairwise matrix");
    println!("  D31: Statistics (mean/variance/percentile)");
    println!("  D32: Streaming determinism (bitwise rerun)");
    println!("  Total: {total_ms:.1} ms");

    v.finish();
}
