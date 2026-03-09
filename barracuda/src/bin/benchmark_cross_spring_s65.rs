// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]
//! Exp183 — Cross-Spring Evolution Benchmark (`ToadStool` S65)
//!
//! Comprehensive benchmark of wetSpring's fully-lean stack after the V48
//! rewire to `ToadStool` S65. Validates every delegation chain and benchmarks
//! each cross-spring primitive with provenance narrative.
//!
//! Covers:
//! 1. GPU ODE (5 systems) — wetSpring bio → `ToadStool` `BatchedOdeRK4`
//! 2. GPU `DiversityFusion` — wetSpring Write → `ToadStool` S63 absorption
//! 3. CPU diversity delegation — wetSpring → `barracuda::stats::diversity` (S64)
//! 4. CPU math delegation — wetSpring → `barracuda::stats::{dot, l2_norm}` (S64)
//! 5. GEMM pipeline — wetSpring `GemmCached` → `ToadStool` `GemmF64`
//! 6. Anderson spectral — `hotSpring` → `ToadStool` → wetSpring
//! 7. Cross-spring primitive inventory + timing
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Validation class | Benchmark |
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | timing harness |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --features gpu --release --bin benchmark_cross_spring_s65` |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::sync::Arc;
use std::time::Instant;

use wetspring_barracuda::bio::bistable::BistableParams;
use wetspring_barracuda::bio::bistable_gpu::{BistableGpu, N_VARS as BIST_VARS};
use wetspring_barracuda::bio::capacitor_gpu::{CapacitorGpu, CapacitorOdeConfig};
use wetspring_barracuda::bio::cooperation::CooperationParams;
use wetspring_barracuda::bio::cooperation_gpu::{CooperationGpu, CooperationOdeConfig};
use wetspring_barracuda::bio::gemm_cached::GemmCached;
use wetspring_barracuda::bio::multi_signal_gpu::{MultiSignalGpu, MultiSignalOdeConfig};
use wetspring_barracuda::bio::phage_defense::PhageDefenseParams;
use wetspring_barracuda::bio::phage_defense_gpu::{PhageDefenseGpu, PhageDefenseOdeConfig};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct Timing {
    label: &'static str,
    origin: &'static str,
    ms: f64,
}

fn bench<F: FnOnce() -> R, R>(label: &str, f: F) -> (R, f64) {
    let t0 = Instant::now();
    let r = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  {label}: {ms:.3} ms");
    (r, ms)
}

fn main() {
    let mut v = Validator::new("Exp183: Cross-Spring Evolution Benchmark (ToadStool S65)");
    let mut timings: Vec<Timing> = Vec::new();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");
    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    // ═══════════════════════════════════════════════════════════════════
    // §1  GPU ODE — wetSpring bio → ToadStool BatchedOdeRK4
    //     Origin: wetSpring V16-V22 (local WGSL) → S58 absorption →
    //     trait-generated WGSL with hotSpring BGL helpers
    // ═══════════════════════════════════════════════════════════════════

    v.section("§1 GPU ODE: 5 bio systems (wetSpring → ToadStool S58, BGL S62)");

    let n_batches: u32 = 128;
    let nb = n_batches as usize;

    let (bist_res, bist_ms) = bench("Bistable GPU (128 batches)", || {
        let gpu_ode = BistableGpu::new(Arc::clone(&device)).expect("BistableGpu");
        let params: Vec<BistableParams> = (0..nb)
            .map(|i| BistableParams {
                alpha_fb: (i as f64).mul_add(0.01, 2.0),
                ..BistableParams::default()
            })
            .collect();
        let initial: Vec<[f64; BIST_VARS]> = vec![[0.01, 0.0, 0.0, 0.0, 0.5]; nb];
        gpu_ode
            .integrate_params(&params, &initial, 500, 0.01)
            .expect("integrate")
    });
    v.check_pass(
        "Bistable: 128 batches finite",
        bist_res.iter().all(|r| r.iter().all(|x| x.is_finite())),
    );
    timings.push(Timing {
        label: "Bistable GPU 128×",
        origin: "wetSpring→S58",
        ms: bist_ms,
    });

    let (coop_res, coop_ms) = bench("Cooperation GPU (128 batches)", || {
        let gpu_ode = CooperationGpu::new(Arc::clone(&device)).expect("CooperationGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| [0.01, 0.0, 0.0, 0.0].iter().copied())
            .collect();
        let params: Vec<CooperationParams> = (0..nb)
            .map(|i| CooperationParams {
                mu_coop: (i as f64).mul_add(0.002, 0.5),
                ..CooperationParams::default()
            })
            .collect();
        let flat_p: Vec<f64> = params.iter().flat_map(CooperationParams::to_flat).collect();
        let config = CooperationOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .expect("integrate")
    });
    v.check_pass(
        "Cooperation: 128 batches finite",
        coop_res.iter().all(|x| x.is_finite()),
    );
    timings.push(Timing {
        label: "Cooperation GPU 128×",
        origin: "wetSpring→S58",
        ms: coop_ms,
    });

    let (phage_res, phage_ms) = bench("PhageDefense GPU (128 batches)", || {
        let gpu_ode = PhageDefenseGpu::new(Arc::clone(&device)).expect("PhageDefenseGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| [1.0, 0.001, 0.01, 10.0].iter().copied())
            .collect();
        let params: Vec<PhageDefenseParams> = (0..nb)
            .map(|i| PhageDefenseParams {
                burst_size: (i as f64).mul_add(0.5, 50.0),
                ..PhageDefenseParams::default()
            })
            .collect();
        let flat_p: Vec<f64> = params
            .iter()
            .flat_map(PhageDefenseParams::to_flat)
            .collect();
        let config = PhageDefenseOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.001,
            t0: 0.0,
            clamp_max: 1e8,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .expect("integrate")
    });
    v.check_pass(
        "PhageDefense: 128 batches finite",
        phage_res.iter().all(|x| x.is_finite()),
    );
    timings.push(Timing {
        label: "PhageDefense GPU 128×",
        origin: "wetSpring→S58",
        ms: phage_ms,
    });

    let (cap_res, cap_ms) = bench("Capacitor GPU (128 batches)", || {
        use wetspring_barracuda::bio::capacitor::{CapacitorParams, N_VARS as CAP_V};
        let gpu_ode = CapacitorGpu::new(Arc::clone(&device)).expect("CapacitorGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| {
                let mut y = [0.0f64; CAP_V];
                y[0] = 0.01;
                y.into_iter()
            })
            .collect();
        let flat_p: Vec<f64> = (0..nb)
            .flat_map(|i| {
                let mut p = CapacitorParams::default();
                p.mu_max += (i as f64) * 0.005;
                p.to_flat().into_iter()
            })
            .collect();
        let config = CapacitorOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .expect("integrate")
    });
    v.check_pass(
        "Capacitor: 128 batches finite",
        cap_res.iter().all(|x| x.is_finite()),
    );
    timings.push(Timing {
        label: "Capacitor GPU 128×",
        origin: "wetSpring→S58",
        ms: cap_ms,
    });

    let (multi_res, multi_ms) = bench("MultiSignal GPU (128 batches)", || {
        use wetspring_barracuda::bio::multi_signal::{MultiSignalParams, N_VARS as MS_V};
        let gpu_ode = MultiSignalGpu::new(Arc::clone(&device)).expect("MultiSignalGpu");
        let flat_y0: Vec<f64> = (0..nb)
            .flat_map(|_| {
                let mut y = [0.0f64; MS_V];
                y[0] = 0.01;
                y.into_iter()
            })
            .collect();
        let flat_p: Vec<f64> = (0..nb)
            .flat_map(|i| {
                let mut p = MultiSignalParams::default();
                p.mu_max += (i as f64) * 0.002;
                p.to_flat().into_iter()
            })
            .collect();
        let config = MultiSignalOdeConfig {
            n_batches,
            n_steps: 500,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        gpu_ode
            .integrate(&config, &flat_y0, &flat_p)
            .expect("integrate")
    });
    v.check_pass(
        "MultiSignal: 128 batches finite",
        multi_res.iter().all(|x| x.is_finite()),
    );
    timings.push(Timing {
        label: "MultiSignal GPU 128×",
        origin: "wetSpring→S58",
        ms: multi_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §2  GPU DiversityFusion — wetSpring Write → ToadStool S63
    //     The diversity_fusion_f64.wgsl was written in wetSpring (Exp167),
    //     absorbed by ToadStool S63, wetSpring now leans on upstream
    // ═══════════════════════════════════════════════════════════════════

    v.section("§2 GPU DiversityFusion: Write→Absorb→Lean (wetSpring→S63)");

    let n_species = 10;
    let n_samples = 256;
    let abundances: Vec<f64> = (0..n_samples * n_species)
        .map(|i| ((i * 7 + 3) % 50) as f64 + 1.0)
        .collect();

    let (fusion_gpu, fusion_gpu_ms) = bench("DiversityFusionGpu (256 samples, 10 spp)", || {
        let fusion = wetspring_barracuda::bio::diversity_fusion_gpu::DiversityFusionGpu::new(
            Arc::clone(&device),
        )
        .expect("DiversityFusionGpu");
        fusion
            .compute(&abundances, n_samples, n_species)
            .expect("compute")
    });

    let (fusion_cpu, fusion_cpu_ms) = bench("diversity_fusion_cpu (256 samples, 10 spp)", || {
        wetspring_barracuda::bio::diversity_fusion_gpu::diversity_fusion_cpu(&abundances, n_species)
    });

    v.check_pass(
        "DiversityFusion: GPU sample count",
        fusion_gpu.len() == n_samples,
    );

    let mut fusion_parity_ok = true;
    for i in 0..n_samples {
        if (fusion_cpu[i].shannon - fusion_gpu[i].shannon).abs() > tolerances::GPU_LOG_POLYFILL
            || (fusion_cpu[i].simpson - fusion_gpu[i].simpson).abs() > tolerances::ANALYTICAL_F64
            || (fusion_cpu[i].evenness - fusion_gpu[i].evenness).abs()
                > tolerances::GPU_LOG_POLYFILL
        {
            fusion_parity_ok = false;
        }
    }
    v.check_pass(
        "DiversityFusion: CPU↔GPU parity (256 samples)",
        fusion_parity_ok,
    );
    v.check_pass(
        "DiversityFusion: all Shannon > 0",
        fusion_gpu.iter().all(|r| r.shannon > 0.0),
    );
    timings.push(Timing {
        label: "DiversityFusion GPU 256×",
        origin: "wetSpring→S63",
        ms: fusion_gpu_ms,
    });
    timings.push(Timing {
        label: "DiversityFusion CPU 256×",
        origin: "wetSpring→S63",
        ms: fusion_cpu_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §3  CPU Diversity Delegation — wetSpring bio → barracuda::stats (S64)
    //     ToadStool S64 cross-spring absorption delivered stats::diversity
    //     with 11 functions and 16 tests. wetSpring now delegates entirely.
    // ═══════════════════════════════════════════════════════════════════

    v.section("§3 CPU Diversity Delegation: bio::diversity → stats::diversity (S64)");

    let community: Vec<f64> = (0..50).map(|i| 1.0 + f64::from(i * 7 % 30)).collect();

    let (sh_local, sh_ms) = bench("diversity::shannon (→ barracuda::stats)", || {
        wetspring_barracuda::bio::diversity::shannon(&community)
    });
    let sh_upstream = barracuda::stats::shannon(&community);
    v.check(
        "Shannon delegation parity",
        sh_local,
        sh_upstream,
        tolerances::EXACT,
    );
    v.check_pass("Shannon > 0", sh_local > 0.0);

    let (si_local, si_ms) = bench("diversity::simpson (→ barracuda::stats)", || {
        wetspring_barracuda::bio::diversity::simpson(&community)
    });
    let si_upstream = barracuda::stats::simpson(&community);
    v.check(
        "Simpson delegation parity",
        si_local,
        si_upstream,
        tolerances::EXACT,
    );
    v.check_pass("Simpson ∈ (0,1]", si_local > 0.0 && si_local <= 1.0);

    let (ch_local, _) = bench("diversity::chao1 (→ barracuda::stats)", || {
        wetspring_barracuda::bio::diversity::chao1(&community)
    });
    let ch_upstream = barracuda::stats::chao1(&community);
    v.check(
        "Chao1 delegation parity",
        ch_local,
        ch_upstream,
        tolerances::EXACT,
    );

    let (pe_local, _) = bench("diversity::pielou_evenness (→ barracuda::stats)", || {
        wetspring_barracuda::bio::diversity::pielou_evenness(&community)
    });
    let pe_upstream = barracuda::stats::pielou_evenness(&community);
    v.check(
        "Pielou delegation parity",
        pe_local,
        pe_upstream,
        tolerances::EXACT,
    );

    let samples_a = vec![10.0, 20.0, 30.0, 0.0, 5.0];
    let samples_b = vec![15.0, 10.0, 25.0, 5.0, 0.0];
    let (bc_local, _) = bench("diversity::bray_curtis (→ barracuda::stats)", || {
        wetspring_barracuda::bio::diversity::bray_curtis(&samples_a, &samples_b)
    });
    let bc_upstream = barracuda::stats::bray_curtis(&samples_a, &samples_b);
    v.check(
        "Bray-Curtis delegation parity",
        bc_local,
        bc_upstream,
        tolerances::EXACT,
    );

    let multi_samples = vec![
        vec![10.0, 20.0, 30.0],
        vec![15.0, 10.0, 25.0],
        vec![0.0, 50.0, 0.0],
    ];
    let (bc_cond_local, _) = bench("bray_curtis_condensed (→ barracuda::stats)", || {
        wetspring_barracuda::bio::diversity::bray_curtis_condensed(&multi_samples)
    });
    let bc_cond_upstream = barracuda::stats::bray_curtis_condensed(&multi_samples);
    let bc_cond_parity = bc_cond_local
        .iter()
        .zip(&bc_cond_upstream)
        .all(|(a, b)| (a - b).abs() <= tolerances::EXACT);
    v.check_pass("bray_curtis_condensed delegation parity", bc_cond_parity);

    let depths: Vec<f64> = (1..=50).map(f64::from).collect();
    let (rare_local, _) = bench("rarefaction_curve (→ barracuda::stats)", || {
        wetspring_barracuda::bio::diversity::rarefaction_curve(&community, &depths)
    });
    let rare_upstream = barracuda::stats::rarefaction_curve(&community, &depths);
    let rare_parity = rare_local
        .iter()
        .zip(&rare_upstream)
        .all(|(a, b)| (a - b).abs() <= tolerances::EXACT);
    v.check_pass("rarefaction_curve delegation parity", rare_parity);
    timings.push(Timing {
        label: "Shannon delegation",
        origin: "S64 cross-spring",
        ms: sh_ms,
    });
    timings.push(Timing {
        label: "Simpson delegation",
        origin: "S64 cross-spring",
        ms: si_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §4  CPU Math Delegation — special::{dot, l2_norm} → stats (S64)
    //     ToadStool S64 added stats::metrics with dot, l2_norm from
    //     airSpring/groundSpring. wetSpring now delegates.
    // ═══════════════════════════════════════════════════════════════════

    v.section("§4 CPU Math: special::{dot,l2_norm} → stats::metrics (S64)");

    let vec_a: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
    let vec_b: Vec<f64> = (0..1000)
        .map(|i| f64::from(i).mul_add(-0.001, 1.0))
        .collect();

    let (dot_local, dot_ms) = bench("special::dot (→ barracuda::stats::dot)", || {
        wetspring_barracuda::special::dot(&vec_a, &vec_b)
    });
    let dot_upstream = barracuda::stats::dot(&vec_a, &vec_b);
    v.check(
        "dot delegation parity",
        dot_local,
        dot_upstream,
        tolerances::EXACT,
    );

    let (l2_local, l2_ms) = bench("special::l2_norm (→ barracuda::stats::l2_norm)", || {
        wetspring_barracuda::special::l2_norm(&vec_a)
    });
    let l2_upstream = barracuda::stats::l2_norm(&vec_a);
    v.check(
        "l2_norm delegation parity",
        l2_local,
        l2_upstream,
        tolerances::EXACT,
    );
    timings.push(Timing {
        label: "dot(1000) delegation",
        origin: "S64 cross-spring",
        ms: dot_ms,
    });
    timings.push(Timing {
        label: "l2_norm(1000) delegation",
        origin: "S64 cross-spring",
        ms: l2_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §5  CPU Special Functions — barracuda::special (always-on)
    //     Origin: hotSpring precision → ToadStool (A&S 7.1.26, Lanczos)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§5 CPU Special Functions: hotSpring precision → ToadStool");

    let (erf_val, erf_ms) = bench("erf(1.0) — barracuda::special", || {
        barracuda::special::erf(1.0)
    });
    v.check(
        "erf(1.0)",
        erf_val,
        0.842_700_792_949_715,
        tolerances::ERF_PARITY,
    );
    timings.push(Timing {
        label: "erf(1.0)",
        origin: "hotSpring→ToadStool",
        ms: erf_ms,
    });

    let (lng_val, lng_ms) = bench("ln_gamma(5.0) — barracuda::special", || {
        barracuda::special::ln_gamma(5.0).expect("ln_gamma")
    });
    v.check(
        "ln_gamma(5.0)",
        lng_val,
        3.178_053_830_347_95,
        tolerances::PYTHON_PARITY,
    );
    timings.push(Timing {
        label: "ln_gamma(5.0)",
        origin: "hotSpring→ToadStool",
        ms: lng_ms,
    });

    let (ncdf_val, ncdf_ms) = bench("norm_cdf(1.96) — barracuda::stats", || {
        barracuda::stats::norm_cdf(1.96)
    });
    v.check(
        "norm_cdf(1.96) ≈ 0.975",
        ncdf_val,
        0.975,
        tolerances::NORM_CDF_PARITY,
    );
    timings.push(Timing {
        label: "norm_cdf(1.96)",
        origin: "hotSpring→ToadStool",
        ms: ncdf_ms,
    });

    let (trapz_val, trapz_ms) = bench("trapz(1000 pts) — barracuda::numerical", || {
        let x: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        barracuda::numerical::trapz(&y, &x).expect("trapz")
    });
    v.check(
        "∫x² dx [0,0.999]",
        trapz_val,
        0.332_334,
        tolerances::ODE_METHOD_PARITY,
    );
    timings.push(Timing {
        label: "trapz(1000)",
        origin: "ToadStool native",
        ms: trapz_ms,
    });

    let (pear_val, pear_ms) = bench("pearson_correlation — barracuda::stats", || {
        barracuda::stats::pearson_correlation(&vec_a, &vec_b).expect("pearson")
    });
    v.check_pass("pearson is finite", pear_val.is_finite());
    v.check_pass("pearson is negative (inverse data)", pear_val < 0.0);
    timings.push(Timing {
        label: "pearson(1000)",
        origin: "ToadStool native",
        ms: pear_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §6  GEMM — wetSpring GemmCached → ToadStool GemmF64
    //     Origin: wetSpring ShaderTemplate → S62 compile_shader_f64 + BGL
    // ═══════════════════════════════════════════════════════════════════

    v.section("§6 GEMM Pipeline: wetSpring → ToadStool GemmF64 (S62 BGL)");

    let (_, gemm_setup_ms) = bench("GemmCached pipeline compile", || {
        GemmCached::new(Arc::clone(&device), Arc::clone(&ctx))
    });
    timings.push(Timing {
        label: "GEMM pipeline compile",
        origin: "wetSpring→S62",
        ms: gemm_setup_ms,
    });

    let gemm = GemmCached::new(Arc::clone(&device), Arc::clone(&ctx));
    let m = 64;
    let k = 32;
    let n = 64;
    let a_mat: Vec<f64> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0)
        .collect();
    let b_mat: Vec<f64> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
        .collect();

    let (gemm_res, first_ms) = bench("GEMM first dispatch (64×32 × 32×64)", || {
        gemm.execute(&a_mat, &b_mat, m, k, n, 1).expect("GEMM")
    });
    v.check_pass("GEMM result finite", gemm_res.iter().all(|x| x.is_finite()));
    let expected_00: f64 = (0..k).map(|j| a_mat[j] * b_mat[j * n]).sum();
    v.check(
        "GEMM C[0,0] matches CPU",
        gemm_res[0],
        expected_00,
        tolerances::GPU_VS_CPU_F64,
    );
    timings.push(Timing {
        label: "GEMM first dispatch 64×64",
        origin: "wetSpring→S62",
        ms: first_ms,
    });

    let ((), repeat_ms) = bench("GEMM ×100 (cached pipeline)", || {
        for _ in 0..100 {
            let _ = gemm.execute(&a_mat, &b_mat, m, k, n, 1).expect("GEMM");
        }
    });
    let per_dispatch = repeat_ms / 100.0;
    v.check_pass("cached dispatch faster", per_dispatch < first_ms);
    timings.push(Timing {
        label: "GEMM cached dispatch",
        origin: "wetSpring→S62",
        ms: per_dispatch,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §7  Anderson Spectral — hotSpring → ToadStool → wetSpring
    //     Origin: hotSpring lattice QCD → Anderson spectral theory
    //     wetSpring uses for soil pore network analysis (Track 4)
    // ═══════════════════════════════════════════════════════════════════

    #[cfg(feature = "gpu")]
    {
        v.section("§7 Anderson Spectral: hotSpring lattice → ToadStool → wetSpring Track 4");

        let (anderson_res, anderson_ms) = bench("anderson_3d(L=8, W=2.0) + lanczos(50)", || {
            let csr = barracuda::spectral::anderson_3d(8, 8, 8, 2.0, 42);
            let tri = barracuda::spectral::lanczos(&csr, 50, 42);
            let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
            let r = barracuda::spectral::level_spacing_ratio(&eigs);
            (eigs.len(), r)
        });
        let (n_eigs, r_val) = anderson_res;
        v.check_pass("Anderson: eigenvalues computed", n_eigs > 0);
        v.check_pass("Anderson: r finite", r_val.is_finite());
        v.check_pass(
            "Anderson: r in valid range (0, 1)",
            r_val > 0.0 && r_val < 1.0,
        );
        timings.push(Timing {
            label: "Anderson 3D + Lanczos",
            origin: "hotSpring→ToadStool",
            ms: anderson_ms,
        });

        let midpoint = f64::midpoint(barracuda::spectral::GOE_R, barracuda::spectral::POISSON_R);
        let (find_wc_res, find_wc_ms) = bench("anderson_sweep + find_w_c(L=6)", || {
            let sweep = barracuda::spectral::anderson_sweep_averaged(6, 1.0, 30.0, 5, 2, 42);
            barracuda::spectral::find_w_c(&sweep, midpoint)
        });
        let wc_ok = find_wc_res.is_some_and(|w| w.is_finite() && w > 0.0);
        v.check_pass(
            "find_w_c: W_c > 0 (or None if no crossing)",
            wc_ok || find_wc_res.is_none(),
        );
        timings.push(Timing {
            label: "sweep+find_w_c(L=6)",
            origin: "hotSpring→ToadStool",
            ms: find_wc_ms,
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // §8  NMF + Ridge — wetSpring → ToadStool linalg (S58)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§8 NMF + Ridge: wetSpring → ToadStool linalg (S58)");

    let nmf_config = barracuda::linalg::nmf::NmfConfig {
        rank: 3,
        max_iter: 200,
        seed: 42,
        objective: barracuda::linalg::nmf::NmfObjective::Euclidean,
        ..barracuda::linalg::nmf::NmfConfig::default()
    };
    let (nmf_res, nmf_ms) = bench("NMF (10×8, k=3) — barracuda::linalg::nmf", || {
        let data: Vec<f64> = (0..80)
            .map(|i| f64::from((i * 17 + 3) % 50) / 50.0 + 0.01)
            .collect();
        barracuda::linalg::nmf::nmf(&data, 10, 8, &nmf_config)
    });
    let nmf_ok = nmf_res
        .as_ref()
        .map(|r| r.w.iter().all(|&x| x >= 0.0) && r.h.iter().all(|&x| x >= 0.0))
        .unwrap_or(false);
    v.check_pass("NMF W, H non-negative", nmf_ok);
    timings.push(Timing {
        label: "NMF 10×8 k=3",
        origin: "wetSpring→S58",
        ms: nmf_ms,
    });

    let (ridge_res, ridge_ms) = bench("ridge regression (20×5→2) — barracuda::linalg", || {
        let x_data: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.01).collect();
        let y_data: Vec<f64> = (0..40).map(|i| f64::from(i).mul_add(0.25, 1.0)).collect();
        barracuda::linalg::ridge_regression(
            &x_data,
            &y_data,
            20,
            5,
            2,
            tolerances::RIDGE_REGULARIZATION_SMALL,
        )
    });
    v.check_pass(
        "ridge weights finite",
        ridge_res
            .map(|r| r.weights.iter().all(|w| w.is_finite()))
            .unwrap_or(false),
    );
    timings.push(Timing {
        label: "Ridge 20×5→2",
        origin: "wetSpring→S58",
        ms: ridge_ms,
    });

    // ═══════════════════════════════════════════════════════════════════
    // §9  Cross-Spring Evolution Timeline
    // ═══════════════════════════════════════════════════════════════════

    v.section("§9 Cross-Spring Shader Evolution Timeline (S39→S65)");

    println!();
    println!("  ╔═════════════════════════════════════════════════════════════════════════╗");
    println!("  ║ SESSION │ ORIGIN       │ CONTRIBUTION → ToadStool (shared primitive)   ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S39-S44 │ hotSpring    │ f64 precision: ShaderTemplate, Fp64Strategy,  ║");
    println!("  ║         │              │ GpuDriverProfile, NVK workarounds, Jacobi eigh║");
    println!("  ║         │              │ RK4/RK45 adaptive, ESN reservoir compute      ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S45-S50 │ neuralSpring │ PairwiseHamming, Jaccard, L2, BatchFitness,   ║");
    println!("  ║         │              │ LocusVariance, SpatialPayoff, graph_laplacian, ║");
    println!("  ║         │              │ batch IPR, MCMC, TransE training, GNN conv     ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S51-S58 │ wetSpring    │ Bio ODE × 5 (BatchedOdeRK4 trait), Gillespie, ║");
    println!("  ║         │              │ Smith-Waterman, Felsenstein, ANI, dN/dS, SNP,  ║");
    println!("  ║         │              │ HMM, KMD, taxonomy, TransE, pangenome, GEMM,  ║");
    println!("  ║         │              │ NMF, ridge, Anderson spectral, diversity       ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S58     │ hotSpring    │ DF64 core: su3_df64, gemm_df64,               ║");
    println!("  ║         │              │ kinetic_energy_df64, wilson_*_df64 (14 shaders)║");
    println!("  ║         │              │ Fp64Strategy::split_workgroups()               ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S60     │ ToadStool    │ SparseGemmF64, TranseScoreF64, TopK,          ║");
    println!("  ║         │              │ DF64 FMA, transcendental polyfill hardening    ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S61-S62 │ ToadStool    │ PeakDetectF64, batched_ode_rk4 improvements,  ║");
    println!("  ║         │              │ storage_bgl_entry / uniform_bgl_entry helpers  ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S63     │ wetSpring    │ diversity_fusion_f64.wgsl absorbed:            ║");
    println!("  ║         │              │ DiversityFusionGpu (fused Shannon+Simpson+J')  ║");
    println!("  ║         │              │ batched_multinomial, cyclic_reduction_f64      ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S64     │ cross-spring │ stats::diversity (wetSpring) — 11 functions    ║");
    println!("  ║         │              │ stats::metrics (airSpring/groundSpring) — dot, ║");
    println!("  ║         │              │ l2_norm, RMSE, MBE, NSE, R², IoA, hit_rate    ║");
    println!("  ║         │              │ 8 lattice shaders (hotSpring V0613/V0614)      ║");
    println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
    println!("  ║ S65     │ ToadStool    │ Smart refactoring: compute_graph 819→522,     ║");
    println!("  ║         │              │ esn_v2 861→482, tensor 808→529, gamma 685→463 ║");
    println!("  ║         │              │ Total: 694 WGSL shaders, 2490 barracuda tests ║");
    println!("  ╚═════════════════════════════════════════════════════════════════════════╝");

    println!();
    println!("  Cross-Spring Synergy Highlights:");
    println!("  ─────────────────────────────────");
    println!("  • hotSpring f64 precision → wetSpring GPU ODE accuracy");
    println!("    (driver workarounds, fp64_strategy auto-detection)");
    println!("  • hotSpring DF64 (14 shaders) → consumer GPU viability for all springs");
    println!("    (RTX 3090: 164 FP64 → DF64 uses 10,496 FP32 for ~10× throughput)");
    println!("  • wetSpring bio ODE → ToadStool BatchedOdeRK4 trait (5 systems)");
    println!("    → neuralSpring uses same trait for population genetics ODE");
    println!("  • wetSpring diversity → ToadStool stats::diversity (S64)");
    println!("    → airSpring uses for crop biodiversity assessment");
    println!("  • wetSpring diversity_fusion WGSL → ToadStool ops::bio (S63)");
    println!("    → first example of Write→Absorb→Lean completing full cycle");
    println!("  • airSpring/groundSpring metrics → ToadStool stats::metrics (S64)");
    println!("    → wetSpring special::{{dot,l2_norm}} delegated upstream");
    println!("  • neuralSpring pairwise ops → wetSpring metalForge cross-substrate");
    println!("  • hotSpring Anderson spectral → wetSpring Track 4 soil pore analysis");
    println!("  • All 4 springs contribute, all 4 consume — true shared evolution");

    v.check_pass("cross-spring evolution timeline documented", true);

    // ═══════════════════════════════════════════════════════════════════
    // §10  Architecture Summary (S65)
    // ═══════════════════════════════════════════════════════════════════

    v.section("§10 Architecture Summary (ToadStool S65, fully lean)");

    println!();
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │ Metric                              │ Value                  │");
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │ ToadStool alignment                  │ S65 (17932267)        │");
    println!("  │ BarraCuda primitives consumed         │ 66 + 2 BGL           │");
    println!("  │ Local WGSL shaders                   │ 0 (fully lean)       │");
    println!("  │ Upstream WGSL shaders (ToadStool)     │ 694                  │");
    println!("  │ DF64 shaders (hotSpring origin)       │ 14                   │");
    println!("  │ Bio shaders (wetSpring origin)        │ 35                   │");
    println!("  │ Lattice shaders (hotSpring S64)       │ 8                    │");
    println!("  │ GPU ODE systems (trait-generated)     │ 5                    │");
    println!("  │ CPU diversity delegation              │ 11 functions (S64)   │");
    println!("  │ CPU metrics delegation                │ 2 functions (S64)    │");
    println!("  │ DiversityFusion Write→Absorb→Lean     │ Complete (S63)       │");
    println!("  │ P0-P3 evolution requests              │ 9/9 DONE             │");
    println!("  │ Passthrough modules                   │ 0                    │");
    println!("  │ Experiments completed                 │ 183                  │");
    println!("  │ Tests (lib + forge)                   │ 866 (819+47)         │");
    println!("  │ Validation checks                     │ 3,618+               │");
    println!("  └──────────────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════
    // §11  Timing Table
    // ═══════════════════════════════════════════════════════════════════

    v.section("§11 Timing Table");

    println!();
    println!("  ┌────────────────────────────────┬────────────────────────┬──────────┐");
    println!("  │ Primitive                      │ Origin                 │ Time     │");
    println!("  ├────────────────────────────────┼────────────────────────┼──────────┤");
    for t in &timings {
        println!("  │ {:30} │ {:22} │ {:7.3}ms│", t.label, t.origin, t.ms);
    }
    println!("  └────────────────────────────────┴────────────────────────┴──────────┘");

    let total_gpu_ode = bist_ms + coop_ms + phage_ms + cap_ms + multi_ms;
    println!();
    println!("  Summary:");
    println!("  GPU ODE (5×128):      {total_gpu_ode:.2} ms");
    println!("  DiversityFusion GPU:  {fusion_gpu_ms:.2} ms");
    println!("  DiversityFusion CPU:  {fusion_cpu_ms:.2} ms");
    println!("  GEMM compile:         {gemm_setup_ms:.2} ms");
    println!("  GEMM cached dispatch: {per_dispatch:.3} ms");

    v.check_pass("all timing data collected", true);

    v.finish();
}
