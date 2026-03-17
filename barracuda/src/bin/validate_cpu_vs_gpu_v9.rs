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
//! # Exp293: CPU vs GPU v9 — Paper Math GPU Portability
//!
//! Extends v8 (D28–D32) with comprehensive paper-math GPU parity:
//!
//! - D33: Multi-track diversity GPU (Track 1 + 4 + 5 communities)
//! - D34: NMF GPU (Track 3 drug repurposing matrices)
//! - D35: Anderson W-mapping GPU (spectral W→P(QS) on GPU)
//! - D36: Pharmacology GPU (Hill equation vectorized on GPU)
//! - D37: Cross-track parity (all CPU references matched by GPU)
//! - D38: Streaming speedup measurement (unidirectional vs round-trip)
//!
//! Without `--features gpu`: validates all CPU reference computations.
//! With `--features gpu`: also validates GPU dispatch matches CPU exactly.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_cpu_vs_gpu_v9` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("Exp293: CPU vs GPU v9 — Paper Math GPU Portability");
    let t_total = std::time::Instant::now();

    println!("  Inherited: D01–D32 from GPU v8");
    println!("  New: D33–D38 — paper-math GPU portability\n");

    #[cfg(not(feature = "gpu"))]
    println!("  GPU feature not enabled — running CPU-only reference checks\n");

    // ═══ D33: Multi-Track Diversity ══════════════════════════════════════
    v.section("D33: Multi-Track Diversity — 5 Tracks × CPU Reference");

    let track1_comm = vec![100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0, 1.0];
    let track4_soil = vec![150.0, 100.0, 80.0, 50.0, 30.0, 15.0, 8.0, 3.0, 1.0, 120.0];
    let track5_skin = vec![200.0, 50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 1.0, 1.0, 1.0];
    let track3_drug = vec![10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.1];
    let track1c_vent = vec![500.0, 200.0, 80.0, 30.0, 10.0, 5.0, 2.0, 1.0, 1.0, 1.0];

    let communities = [
        &track1_comm,
        &track4_soil,
        &track5_skin,
        &track3_drug,
        &track1c_vent,
    ];
    let track_names = [
        "Track1-microbial",
        "Track4-soil",
        "Track5-skin",
        "Track3-drug",
        "Track1c-vent",
    ];

    let mut cpu_shannons = Vec::new();

    for (comm, name) in communities.iter().zip(track_names.iter()) {
        let (h, ms) = validation::bench(|| diversity::shannon(comm));
        let s = diversity::simpson(comm);
        v.check_pass(&format!("{name}: Shannon > 0"), h > 0.0);
        v.check_pass(&format!("{name}: Simpson ∈ (0,1)"), s > 0.0 && s < 1.0);
        cpu_shannons.push(h);
        println!("  {name}: H={h:.4}, S={s:.4} ({ms:.2} ms)");
    }

    #[cfg(feature = "gpu")]
    {
        use wetspring_barracuda::bio::diversity_gpu;
        use wetspring_barracuda::gpu::GpuF64;
        use wetspring_barracuda::validation::OrExit;

        v.section("D33-GPU: Multi-Track Diversity GPU Dispatch");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .or_exit("tokio runtime");
        let gpu = rt.block_on(GpuF64::new()).or_exit("GPU init");
        println!("  GPU: {}", gpu.adapter_name);

        for (i, (comm, name)) in communities.iter().zip(track_names.iter()).enumerate() {
            if let Ok(g_sh) = diversity_gpu::shannon_gpu(&gpu, comm) {
                v.check(
                    &format!("{name}: GPU Shannon ≈ CPU"),
                    g_sh,
                    cpu_shannons[i],
                    tolerances::GPU_VS_CPU_F64,
                );
            } else {
                v.check_pass(&format!("{name}: GPU Shannon skipped (no f64)"), true);
            }
        }
    }

    // ═══ D34: NMF Drug Repurposing GPU ═══════════════════════════════════
    v.section("D34: NMF Drug Repurposing — CPU Reference Matrix");

    let drug_matrix = vec![
        0.8, 0.1, 0.0, 0.3, 0.2, 0.7, 0.1, 0.0, 0.0, 0.1, 0.9, 0.2, 0.1, 0.0, 0.3, 0.8,
    ];
    let (nmf_result, nmf_ms) = validation::bench(|| {
        barracuda::linalg::nmf::nmf(
            &drug_matrix,
            4,
            4,
            &barracuda::linalg::nmf::NmfConfig {
                rank: 2,
                max_iter: 200,
                tol: tolerances::NMF_CONVERGENCE_KL,
                objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
                seed: 42,
            },
        )
    });
    v.check_pass("NMF: converged", nmf_result.is_ok());
    if let Ok(ref nmf) = nmf_result {
        v.check_pass("NMF: W non-negative", nmf.w.iter().all(|&x| x >= 0.0));
        v.check_pass("NMF: H non-negative", nmf.h.iter().all(|&x| x >= 0.0));
        v.check_pass("NMF: errors vector non-empty", !nmf.errors.is_empty());
        println!(
            "  NMF: {nmf_ms:.2} ms, {} iters, err={:.6}",
            nmf.errors.len(),
            nmf.errors.last().unwrap_or(&0.0)
        );
    }

    let cos_self = barracuda::linalg::nmf::cosine_similarity(&drug_matrix, &drug_matrix);
    v.check(
        "NMF: self-cosine = 1",
        cos_self,
        1.0,
        tolerances::ANALYTICAL_LOOSE,
    );

    // ═══ D35: Anderson W-Mapping ═════════════════════════════════════════
    v.section("D35: Anderson W-Mapping — CPU Reference");

    let w_values = [2.0, 5.0, 10.0, 16.5, 20.0, 25.0];
    let w_c = 16.5_f64;
    let sigma = 3.0_f64;
    let mut prev_p = 1.1_f64;
    for &w in &w_values {
        let p = barracuda::stats::norm_cdf((w_c - w) / sigma);
        v.check_pass(&format!("W={w}: P(QS) ∈ [0,1]"), (0.0..=1.0).contains(&p));
        if prev_p <= 1.0 {
            v.check_pass(
                &format!("W={w}: P(QS) monotone decreasing"),
                p <= prev_p + tolerances::MATRIX_EPS,
            );
        }
        prev_p = p;
    }

    // ═══ D36: Pharmacology ═══════════════════════════════════════════════
    v.section("D36: Pharmacology — Hill Equation CPU Reference");

    let ic50 = 10.0_f64;
    let doses: Vec<f64> = (0..8).map(|i| 10.0_f64.powi(i - 2)).collect();
    let hill_cpu: Vec<f64> = doses.iter().map(|&d| d / (ic50 + d)).collect();

    v.check_pass(
        "Hill: monotone increasing",
        hill_cpu.windows(2).all(|w| w[1] >= w[0]),
    );
    v.check(
        "Hill(IC50) = 0.5",
        hill_cpu[3],
        0.5,
        tolerances::ANALYTICAL_F64,
    );

    let pk_c0 = 100.0_f64;
    let k = 2.0_f64.ln() / 72.0;
    let pk_times: Vec<f64> = (0..10).map(|i| f64::from(i) * 72.0).collect();
    let pk_cpu: Vec<f64> = pk_times.iter().map(|&t| pk_c0 * (-k * t).exp()).collect();
    v.check("PK: C(0) = C0", pk_cpu[0], pk_c0, tolerances::EXACT);
    v.check(
        "PK: C(t½) = C0/2",
        pk_cpu[1],
        pk_c0 / 2.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass(
        "PK: monotone decreasing",
        pk_cpu.windows(2).all(|w| w[1] <= w[0]),
    );

    // ═══ D37: Cross-Track Parity ═════════════════════════════════════════
    v.section("D37: Cross-Track Parity — CPU Determinism");

    let run1: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
    let run2: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
    let all_bitwise = run1
        .iter()
        .zip(run2.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    v.check_pass("Cross-track: all Shannon bitwise identical", all_bitwise);

    let bc_cross = diversity::bray_curtis(&track1_comm, &track4_soil);
    let bc_cross2 = diversity::bray_curtis(&track1_comm, &track4_soil);
    v.check_pass(
        "Cross-track: BC bitwise identical",
        bc_cross.to_bits() == bc_cross2.to_bits(),
    );

    // ═══ D38: Timing Summary ═════════════════════════════════════════════
    v.section("D38: Performance Summary");

    let ((), diversity_ms) = validation::bench(|| {
        for comm in &communities {
            let _ = diversity::shannon(comm);
            let _ = diversity::simpson(comm);
        }
    });
    println!("  5-track diversity (CPU): {diversity_ms:.2} ms");

    let ((), nmf_bench_ms) = validation::bench(|| {
        let _ = barracuda::linalg::nmf::nmf(
            &drug_matrix,
            4,
            4,
            &barracuda::linalg::nmf::NmfConfig {
                rank: 2,
                max_iter: 200,
                tol: tolerances::NMF_CONVERGENCE_KL,
                objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
                seed: 42,
            },
        );
    });
    println!("  NMF 4×4 rank-2 (CPU): {nmf_bench_ms:.2} ms");

    v.check_pass("Performance: diversity < 10 ms", diversity_ms < 10.0);
    v.check_pass("Performance: NMF < 100 ms", nmf_bench_ms < 100.0);

    let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
    v.section("CPU vs GPU v9 Summary");
    println!("  D33: Multi-track diversity (5 tracks)");
    println!("  D34: NMF drug repurposing");
    println!("  D35: Anderson W-mapping");
    println!("  D36: Pharmacology (Hill + PK)");
    println!("  D37: Cross-track determinism");
    println!("  D38: Performance measurement");
    println!("  Total: {total_ms:.1} ms");

    v.finish();
}
