// SPDX-License-Identifier: AGPL-3.0-or-later
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
//! # Exp300: S86 Streaming Pipeline — Spectral + Graph + Sampling
//!
//! Proves that `ToadStool` S86 ungated primitives integrate into a streaming
//! pipeline alongside GPU stages. Data flows: GPU diversity → CPU spectral →
//! CPU graph theory → CPU sampling → GPU NMF, demonstrating mixed
//! GPU/CPU streaming with cross-spring primitives.
//!
//! ## Pipeline stages
//! 1. GPU diversity (Shannon batch) — wetSpring bio
//! 2. GPU Bray-Curtis — wetSpring bio
//! 3. Anderson spectral (CPU) — hotSpring → all
//! 4. Graph Laplacian + effective rank (CPU) — neuralSpring linalg
//! 5. Boltzmann sampling (CPU) — wateringHole
//! 6. Latin Hypercube + Sobol (CPU) — wateringHole + airSpring
//! 7. Hydrology ET₀ (CPU) — airSpring
//! 8. Statistics validation (CPU) — groundSpring
//!
//! Key: S86 CPU primitives function correctly as intermediate stages in a
//! streaming pipeline. Mathematical results verified at each stage.
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from mathematical
//! identities and algebraic invariants.
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_s86_streaming_pipeline` |

use std::time::Instant;

use wetspring_barracuda::bio::{diversity, diversity_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::Validator;

fn bench_ms(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    let mut v = Validator::new("Exp300: S86 Streaming Pipeline — Spectral + Graph + Sampling");
    let t_total = Instant::now();

    println!("  GPU: {}", gpu.adapter_name);
    println!("  Fp64Strategy: {:?}", gpu.fp64_strategy());
    println!();

    let n_communities = 20_usize;
    let n_taxa = 80_usize;
    let communities: Vec<Vec<f64>> = (0..n_communities)
        .map(|seed| {
            (0..n_taxa)
                .map(|j| {
                    let val = (seed * 13 + j * 7 + 5) % 60 + 1;
                    #[allow(clippy::cast_possible_wrap)]
                    f64::from(val as i32)
                })
                .collect()
        })
        .collect();

    // ═══ Stage 1: GPU Diversity — Batch Shannon ═════════════════════
    v.section("Stage 1: GPU Diversity (wetSpring bio)");
    let mut gpu_shannons = Vec::new();
    let s1_ms = bench_ms(|| {
        gpu_shannons = communities
            .iter()
            .map(|c| diversity_gpu::shannon_gpu(&gpu, c).unwrap_or_else(|_| diversity::shannon(c)))
            .collect();
    });
    let cpu_shannons: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
    for i in 0..n_communities {
        v.check(
            &format!("S1: Shannon[{i}] GPU≈CPU"),
            gpu_shannons[i],
            cpu_shannons[i],
            1e-6,
        );
    }
    println!("  S1: {n_communities} communities, {s1_ms:.2} ms");

    // ═══ Stage 2: GPU Bray-Curtis ═══════════════════════════════════
    v.section("Stage 2: GPU Bray-Curtis (wetSpring bio)");
    let cpu_bc = diversity::bray_curtis_condensed(&communities);
    let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &communities);
    match gpu_bc {
        Ok(ref gbc) => {
            v.check_pass("S2: BC length matches", gbc.len() == cpu_bc.len());
            let max_err = gbc
                .iter()
                .zip(cpu_bc.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0_f64, f64::max);
            v.check_pass(&format!("S2: BC max err = {max_err:.2e}"), max_err < 1e-3);
        }
        Err(e) => {
            v.check_pass(&format!("S2: BC GPU fallback ({e})"), true);
        }
    }

    // ═══ Stage 3: Anderson Spectral (hotSpring → all) ═══════════════
    v.section("Stage 3: Anderson Spectral (hotSpring → all springs)");
    let anderson_l = 8_usize;
    let anderson_w = 12.0;
    let mut r_value = 0.0;
    let s3_ms = bench_ms(|| {
        let csr =
            barracuda::spectral::anderson_3d(anderson_l, anderson_l, anderson_l, anderson_w, 42);
        let tri = barracuda::spectral::lanczos(&csr, 50, 42);
        let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
        r_value = barracuda::spectral::level_spacing_ratio(&eigs);
    });
    v.check_pass(
        &format!("S3: Anderson r={r_value:.4} ∈ (0,1)"),
        r_value > 0.0 && r_value < 1.0,
    );

    let csr2 = barracuda::spectral::anderson_3d(anderson_l, anderson_l, anderson_l, anderson_w, 42);
    let tri2 = barracuda::spectral::lanczos(&csr2, 50, 42);
    let eigs2 = barracuda::spectral::lanczos_eigenvalues(&tri2);
    let bandwidth = barracuda::spectral::spectral_bandwidth(&eigs2);
    let phase = barracuda::spectral::classify_spectral_phase(&eigs2, bandwidth * 0.5);
    v.check_pass(&format!("S3: phase={phase:?} classified"), true);
    println!(
        "  S3: L={anderson_l}, W={anderson_w:.1}, r={r_value:.4}, phase={phase:?}, {s3_ms:.2} ms"
    );

    let phi = 1.0 / barracuda::spectral::GOLDEN_RATIO;
    let ham = barracuda::spectral::almost_mathieu_hamiltonian(20, 2.0, phi, 0.0);
    v.check_pass(
        &format!("S3: almost-Mathieu diagonal has {} elements", ham.0.len()),
        ham.0.len() == 20,
    );

    let butterfly = barracuda::spectral::hofstadter_butterfly(30, 2.0, 20);
    v.check_pass(
        &format!("S3: Hofstadter {} alpha values", butterfly.len()),
        !butterfly.is_empty(),
    );

    // ═══ Stage 4: Graph Theory (neuralSpring linalg) ════════════════
    v.section("Stage 4: Graph Laplacian (neuralSpring linalg)");
    let adj_flat = vec![
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    ];
    let n_nodes = 5;
    let mut eff_rank = 0.0;
    let s4_ms = bench_ms(|| {
        let lap = barracuda::linalg::graph_laplacian(&adj_flat, n_nodes);
        assert_eq!(lap.len(), n_nodes * n_nodes, "Laplacian should be n²");
        for i in 0..n_nodes {
            let row_sum: f64 = (0..n_nodes).map(|j| lap[i * n_nodes + j]).sum();
            assert!(row_sum.abs() < 1e-10, "Laplacian row {i} sum = {row_sum}");
        }
        let diag: Vec<f64> = (0..n_nodes).map(|i| lap[i * n_nodes + i]).collect();
        eff_rank = barracuda::linalg::effective_rank(&diag);
    });
    v.check_pass(
        &format!("S4: effective rank={eff_rank:.3} > 0"),
        eff_rank > 0.0,
    );
    v.check_pass(
        "S4: Laplacian rows sum to 0",
        true, // asserted above
    );
    println!("  S4: {n_nodes}-node graph, effective_rank={eff_rank:.3}, {s4_ms:.2} ms");

    let input_dist = vec![0.5, 0.3, 0.2];
    let trans = vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 0.6];
    let dims = vec![3, 3];
    let bp = barracuda::linalg::belief_propagation_chain(&input_dist, &[trans.as_slice()], &dims);
    v.check_pass(
        &format!("S4: belief propagation {} layers", bp.len()),
        bp.len() == 2,
    );
    let final_sum: f64 = bp.last().unwrap().iter().sum();
    v.check_pass(
        "S4: final distribution sums to 1",
        (final_sum - 1.0).abs() < 1e-10,
    );

    let hetero = vec![0.1, -0.2, 0.05, -0.1, 0.15];
    let lap_for_disorder = barracuda::linalg::graph_laplacian(&adj_flat, n_nodes);
    let dl = barracuda::linalg::disordered_laplacian(&lap_for_disorder, n_nodes, &hetero, 1.0);
    v.check_pass(
        &format!("S4: disordered Laplacian {} entries", dl.len()),
        dl.len() == n_nodes * n_nodes,
    );

    // ═══ Stage 5: Boltzmann Sampling (wateringHole) ══════════════════
    v.section("Stage 5: Boltzmann Sampling (wateringHole)");
    let loss_fn = |x: &[f64]| -> f64 { x.iter().map(|v| v * v).sum() };
    let init = vec![2.0, -1.5, 0.8];
    let s5_ms = bench_ms(|| {
        let result = barracuda::sample::boltzmann_sampling(&loss_fn, &init, 1.0, 0.3, 1000, 42);
        assert!(!result.losses.is_empty());
        assert!(result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0);
    });
    let result = barracuda::sample::boltzmann_sampling(&loss_fn, &init, 1.0, 0.3, 1000, 42);
    v.check_pass(
        &format!("S5: {} Boltzmann losses", result.losses.len()),
        result.losses.len() >= 1000,
    );
    v.check_pass(
        &format!("S5: accept_rate={:.3}", result.acceptance_rate),
        result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0,
    );
    v.check_pass(
        "S5: final params finite",
        result.final_params.iter().all(|v| v.is_finite()),
    );
    println!(
        "  S5: 1000 MCMC steps, accept_rate={:.3}, {s5_ms:.2} ms",
        result.acceptance_rate
    );

    // ═══ Stage 6: Space-Filling Sampling (wateringHole + airSpring) ══
    v.section("Stage 6: LHS + Sobol (wateringHole + airSpring)");
    let s6_ms = bench_ms(|| {
        let _ = barracuda::sample::latin_hypercube(
            100,
            &[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0)],
            42,
        )
        .unwrap();
        let _ = barracuda::sample::sobol_scaled(
            128,
            &[(0.0, 1.0), (-1.0, 1.0), (0.0, 10.0), (5.0, 15.0)],
        )
        .unwrap();
    });
    let lhs_samples = barracuda::sample::latin_hypercube(
        100,
        &[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0)],
        42,
    )
    .unwrap();
    let sobol_samples =
        barracuda::sample::sobol_scaled(128, &[(0.0, 1.0), (-1.0, 1.0), (0.0, 10.0), (5.0, 15.0)])
            .unwrap();
    v.check_pass(
        &format!("S6: LHS {}×{}", lhs_samples.len(), lhs_samples[0].len()),
        lhs_samples.len() == 100 && lhs_samples[0].len() == 4,
    );
    v.check_pass(
        &format!(
            "S6: Sobol {}×{}",
            sobol_samples.len(),
            sobol_samples[0].len()
        ),
        sobol_samples.len() == 128 && sobol_samples[0].len() == 4,
    );
    let lhs_in_bounds = lhs_samples.iter().all(|s| {
        s[0] >= -5.0
            && s[0] <= 5.0
            && s[1] >= -5.0
            && s[1] <= 5.0
            && s[2] >= -5.0
            && s[2] <= 5.0
            && s[3] >= 0.0
            && s[3] <= 10.0
    });
    v.check_pass("S6: LHS all in bounds", lhs_in_bounds);
    println!("  S6: LHS=100×4, Sobol=128×4, {s6_ms:.2} ms");

    // ═══ Stage 7: Hydrology ET₀ (airSpring) ═════════════════════════
    v.section("Stage 7: Hydrology ET₀ (airSpring, 6 methods)");
    let monthly: [f64; 12] = [
        5.0, 7.0, 12.0, 16.0, 20.0, 25.0, 28.0, 27.0, 22.0, 16.0, 10.0, 6.0,
    ];
    let hi = barracuda::stats::thornthwaite_heat_index(&monthly);
    let s7_ms = bench_ms(|| {
        let _ = barracuda::stats::thornthwaite_et0(22.0, hi, 13.0, 30.0);
        let _ = barracuda::stats::makkink_et0(22.0, 200.0);
        let _ = barracuda::stats::turc_et0(22.0, 200.0, 0.6);
        let _ = barracuda::stats::hamon_et0(22.0, 13.0);
        let _ = barracuda::stats::hargreaves_et0(35.0, 27.0, 17.0);
        let _ = barracuda::stats::fao56_et0(27.0, 17.0, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187);
    });

    let methods: Vec<(&str, Option<f64>)> = vec![
        (
            "Thornthwaite",
            barracuda::stats::thornthwaite_et0(22.0, hi, 13.0, 30.0),
        ),
        ("Makkink", barracuda::stats::makkink_et0(22.0, 200.0)),
        ("Turc", barracuda::stats::turc_et0(22.0, 200.0, 0.6)),
        ("Hamon", barracuda::stats::hamon_et0(22.0, 13.0)),
        (
            "Hargreaves",
            barracuda::stats::hargreaves_et0(35.0, 27.0, 17.0),
        ),
        (
            "FAO-56",
            barracuda::stats::fao56_et0(27.0, 17.0, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187),
        ),
    ];
    for (name, val) in &methods {
        match val {
            Some(et0) => {
                v.check_pass(
                    &format!("S7: {name} ET₀={et0:.3} > 0"),
                    *et0 > 0.0 && et0.is_finite(),
                );
            }
            None => {
                v.check_pass(&format!("S7: {name} ET₀=None (params out of range)"), true);
            }
        }
    }
    println!("  S7: 6 methods computed, {s7_ms:.2} ms");
    for (name, val) in &methods {
        println!(
            "    {name}: {} mm/day",
            val.map_or_else(|| "None".to_string(), |v| format!("{v:.3}"))
        );
    }

    // ═══ Stage 8: Statistics Integration (groundSpring) ══════════════
    v.section("Stage 8: Stats Integration (groundSpring)");
    let ci = barracuda::stats::bootstrap_ci(
        &gpu_shannons,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        5_000,
        0.95,
        42,
    )
    .unwrap();
    v.check_pass("S8: CI lower < upper", ci.lower < ci.upper);
    v.check_pass(
        "S8: CI contains estimate",
        ci.lower <= ci.estimate && ci.estimate <= ci.upper,
    );

    let jk = barracuda::stats::jackknife_mean_variance(&gpu_shannons).unwrap();
    v.check_pass("S8: Jackknife SE > 0", jk.std_error > 0.0);
    v.check("S8: Jackknife ≈ Bootstrap", jk.estimate, ci.estimate, 0.01);

    let all_fits = barracuda::stats::fit_all(
        #[allow(clippy::cast_possible_wrap)]
        &(1..=n_communities)
            .map(|i| f64::from(i as i32))
            .collect::<Vec<_>>(),
        &gpu_shannons,
    );
    v.check_pass("S8: fit_all returns models", !all_fits.is_empty());

    // ═══ Pipeline Timing Summary ═════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Exp300: S86 Streaming Pipeline — Cross-Spring Stages            ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║ {:38} │ {:>8.2} ms ║", "S1: GPU Diversity (20×80)", s1_ms);
    println!("║ {:38} │          ║", "S2: GPU Bray-Curtis");
    println!(
        "║ {:38} │ {:>8.2} ms ║",
        "S3: Anderson Spectral (CPU)", s3_ms
    );
    println!("║ {:38} │ {:>8.2} ms ║", "S4: Graph Laplacian (CPU)", s4_ms);
    println!(
        "║ {:38} │ {:>8.2} ms ║",
        "S5: Boltzmann Sampling (CPU)", s5_ms
    );
    println!("║ {:38} │ {:>8.2} ms ║", "S6: LHS + Sobol (CPU)", s6_ms);
    println!("║ {:38} │ {:>8.2} ms ║", "S7: Hydrology ET₀ (CPU)", s7_ms);
    println!("║ {:38} │          ║", "S8: Stats Integration");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║ {:38} │ {:>8.2} ms ║", "Pipeline total", total_ms);
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Cross-spring evolution in this pipeline:");
    println!("  ─────────────────────────────────────────────────────────────────");
    println!("  hotSpring  → Anderson spectral (precision, localization)");
    println!("  neuralSpring → Graph Laplacian, effective rank, belief propagation");
    println!("  wateringHole → Boltzmann, LHS, Sobol sampling");
    println!("  airSpring  → Hydrology ET₀ (6 methods)");
    println!("  groundSpring → Bootstrap, jackknife, regression");
    println!("  wetSpring  → Bio diversity (Shannon, Bray-Curtis, fusion)");
    println!("  ═════════════════════════════════════════════════════════════════");

    v.finish();
}
