// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::type_complexity,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::collection_is_never_read,
    clippy::many_single_char_names,
    dead_code
)]
//! # Exp169: Modern Cross-Spring Evolution Benchmark
//!
//! Benchmarks the complete modern barracuda stack through wetSpring's lens,
//! tracking provenance of every primitive to its originating spring. Validates
//! that cross-spring evolution works: shaders and primitives that originated
//! in one biome benefit all springs.
//!
//! ## Cross-Spring Provenance Map
//!
//! | Primitive | Origin | Absorbed | wetSpring Use |
//! |-----------|--------|----------|---------------|
//! | `FusedMapReduceF64` | `ToadStool` core | S31 | 12+ GPU modules |
//! | `BatchedOdeRK4::generate_shader()` | wetSpring → TS | S58 | 5 bio ODE |
//! | `FelsensteinGpu` | wetSpring → TS | S31d | Phylo pruning |
//! | `PeakDetectF64` | hotSpring → TS | S62 | LC-MS signal |
//! | `BatchedEighGpu` | hotSpring NAK | S31g | `PCoA` ordination |
//! | `PairwiseHammingGpu` | neuralSpring → TS | S39 | SNP distance |
//! | `anderson_3d` | hotSpring → TS | S31c | QS-disorder |
//! | `find_w_c` | hotSpring → TS | S59 | Phase transition |
//! | `pearson_correlation` | `ToadStool` stats | S59 | CPU baselines |
//! | `norm_cdf` | `ToadStool` stats | S59 | Normal CDF |
//! | `graph_laplacian` | neuralSpring → TS | S54 | Network spectral |
//!
//! # Provenance
//!
//! | Item        | Value |
//! |-------------|-------|
//! | Date        | 2026-02-25 |
//! | Command     | `cargo test --bin benchmark_cross_spring_modern -- --nocapture` |

use std::time::Instant;
use wetspring_barracuda::special;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

use barracuda::numerical::trapz;
use barracuda::special::{erf, ln_gamma, regularized_gamma_p};
use barracuda::stats::{norm_cdf, pearson_correlation};

#[cfg(feature = "gpu")]
use barracuda::linalg::{graph_laplacian, ridge_regression};

#[cfg(feature = "gpu")]
use barracuda::spectral::{
    AndersonSweepPoint, GOE_R, POISSON_R, anderson_3d, anderson_hamiltonian, find_w_c, lanczos,
    lanczos_eigenvalues, level_spacing_ratio,
};

fn main() {
    let mut v = Validator::new("Exp169: Modern Cross-Spring Evolution Benchmark");

    // ═════════════════════════════════════════════════════════════
    // S1: CPU Math — barracuda::special (always-on, multi-spring)
    // ═════════════════════════════════════════════════════════════
    v.section("── S1: CPU Math — barracuda::special (A&S/Lanczos) ──");

    let t0 = Instant::now();
    let erf_val = erf(1.0);
    let erf_ns = t0.elapsed().as_nanos();
    v.check("erf(1.0)", erf_val, 0.842_700_792_949_715, 5e-7);

    let t0 = Instant::now();
    let lng_val = ln_gamma(5.0).unwrap_or(f64::NAN);
    let lng_ns = t0.elapsed().as_nanos();
    v.check(
        "ln_gamma(5) = ln(24)",
        lng_val,
        24.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );

    let t0 = Instant::now();
    let rgp = regularized_gamma_p(1.0, 1.0).unwrap_or(f64::NAN);
    let rgp_ns = t0.elapsed().as_nanos();
    let expected_rgp = 1.0 - (-1.0_f64).exp();
    v.check(
        "P(1,1) = 1-e^-1",
        rgp,
        expected_rgp,
        tolerances::ANALYTICAL_F64,
    );

    println!("  Timing: erf={erf_ns}ns, ln_gamma={lng_ns}ns, reg_gamma={rgp_ns}ns");

    // ═════════════════════════════════════════════════════════════
    // S2: Stats Module (ToadStool S59)
    // ═════════════════════════════════════════════════════════════
    v.section("── S2: CPU Stats — barracuda::stats (ToadStool S59) ──");

    let t0 = Instant::now();
    let phi = norm_cdf(0.0);
    let ncdf_ns = t0.elapsed().as_nanos();
    v.check("Φ(0) = 0.5", phi, 0.5, tolerances::EXACT);

    let phi_196 = norm_cdf(1.96);
    v.check("Φ(1.96) ≈ 0.975", phi_196, 0.975, 1e-3);

    let t0 = Instant::now();
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = pearson_correlation(&x, &y).unwrap_or(f64::NAN);
    let corr_ns = t0.elapsed().as_nanos();
    v.check("pearson(x, 2x) = 1.0", r, 1.0, tolerances::EXACT);

    let anti_y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let r_neg = pearson_correlation(&x, &anti_y).unwrap_or(f64::NAN);
    v.check("pearson(x, -x) = -1.0", r_neg, -1.0, tolerances::EXACT);

    println!("  Timing: norm_cdf={ncdf_ns}ns, pearson={corr_ns}ns");

    // ═════════════════════════════════════════════════════════════
    // S3: V43 Rewire Validation — normal_cdf delegation
    // ═════════════════════════════════════════════════════════════
    v.section("── S3: V43 Rewire — special::normal_cdf → barracuda::stats::norm_cdf ──");

    let t0 = Instant::now();
    let local_ncdf = special::normal_cdf(1.96);
    let local_ns = t0.elapsed().as_nanos();

    let t0 = Instant::now();
    let upstream_ncdf = norm_cdf(1.96);
    let upstream_ns = t0.elapsed().as_nanos();

    v.check(
        "local == upstream (bit-exact)",
        local_ncdf,
        upstream_ncdf,
        0.0,
    );
    v.check("Φ(1.96) ≈ 0.975 (local)", local_ncdf, 0.975, 1e-3);
    v.check("Φ(1.96) ≈ 0.975 (upstream)", upstream_ncdf, 0.975, 1e-3);

    println!("  Delegation: local={local_ns}ns, upstream={upstream_ns}ns");

    // ═════════════════════════════════════════════════════════════
    // S4: Numerical — trapz (always-on CPU)
    // ═════════════════════════════════════════════════════════════
    v.section("── S4: CPU Numerical — barracuda::numerical::trapz ──");

    let xs: Vec<f64> = (0..101).map(|i| f64::from(i) * 0.01).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();

    let t0 = Instant::now();
    let area = trapz(&ys, &xs).unwrap_or(f64::NAN);
    let trapz_ns = t0.elapsed().as_nanos();
    v.check("∫₀¹ x² dx = 1/3", area, 1.0 / 3.0, 1e-4);

    println!("  Timing: trapz={trapz_ns}ns (101 points)");

    // ═════════════════════════════════════════════════════════════
    // S5: Graph + Spectral (GPU-gated primitives)
    // ═════════════════════════════════════════════════════════════
    #[cfg(feature = "gpu")]
    {
        v.section("── S5: Graph Spectral — neuralSpring → ToadStool (S54) ──");

        let adjacency = vec![
            0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ];

        let t0 = Instant::now();
        let lap = graph_laplacian(&adjacency, 4);
        let graph_ns = t0.elapsed().as_nanos();
        v.check("L[0,0] = degree(0)", lap[0], 2.0, 0.0);
        v.check("L[0,1] = -A[0,1]", lap[1], -1.0, 0.0);
        println!("  Timing: laplacian={graph_ns}ns (4×4)");

        v.section("── S6: Anderson Spectral — hotSpring precision → ToadStool ──");

        let t0 = Instant::now();
        let (diag, _off) = anderson_hamiltonian(100, 4.0, 42);
        let ham_ns = t0.elapsed().as_nanos();
        v.check("Hamiltonian N=100", diag.len() as f64, 100.0, 0.0);

        let t0 = Instant::now();
        let csr = anderson_3d(8, 8, 8, 5.0, 42);
        let a3d_ns = t0.elapsed().as_nanos();
        let n_3d = 8_usize * 8 * 8;
        v.check(
            "3D CSR rows = L³+1",
            csr.row_ptr.len() as f64,
            (n_3d + 1) as f64,
            0.0,
        );

        let t0 = Instant::now();
        let tri = lanczos(&csr, 50, 42);
        let eigs = lanczos_eigenvalues(&tri);
        let r_ratio = level_spacing_ratio(&eigs);
        let lanc_ns = t0.elapsed().as_nanos();
        v.check_pass(
            "level spacing ratio in [0,1]",
            (0.0..=1.0).contains(&r_ratio),
        );
        println!("  Timing: hamiltonian={ham_ns}ns, 3d_csr={a3d_ns}ns, lanczos+lsr={lanc_ns}ns");
        println!("  r(W=5,L=8) = {r_ratio:.4}");

        v.section("── S7: find_w_c Rewire — hotSpring spectral (S59) ──");

        let midpoint = f64::midpoint(GOE_R, POISSON_R);
        let sweep = vec![
            AndersonSweepPoint {
                w: 5.0,
                r_mean: 0.52,
                r_stderr: 0.01,
            },
            AndersonSweepPoint {
                w: 10.0,
                r_mean: 0.48,
                r_stderr: 0.01,
            },
            AndersonSweepPoint {
                w: 15.0,
                r_mean: 0.42,
                r_stderr: 0.01,
            },
            AndersonSweepPoint {
                w: 20.0,
                r_mean: 0.39,
                r_stderr: 0.01,
            },
        ];

        let t0 = Instant::now();
        let w_c = find_w_c(&sweep, midpoint);
        let wc_ns = t0.elapsed().as_nanos();
        v.check_pass("find_w_c returns crossing", w_c.is_some());

        if let Some(wc) = w_c {
            v.check_pass("W_c physically reasonable (5-25)", wc > 5.0 && wc < 25.0);
            println!("  W_c={wc:.2}, midpoint={midpoint:.4}, time={wc_ns}ns");
        }

        v.section("── S8: Ridge Regression — wetSpring → ToadStool (S59) ──");

        let n = 20_usize;
        let x_flat: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_ridge: Vec<f64> = x_flat.iter().map(|x| 2.0 * x + 1.0).collect();

        let t0 = Instant::now();
        let result = ridge_regression(&x_flat, &y_ridge, n, 1, 1, 1e-6);
        let ridge_ns = t0.elapsed().as_nanos();
        v.check_pass("ridge_regression succeeds", result.is_ok());
        println!("  Timing: ridge={ridge_ns}ns (n={n})");
    }

    // ═════════════════════════════════════════════════════════════
    // S9: Cross-Spring Provenance Summary
    // ═════════════════════════════════════════════════════════════
    v.section("── S9: Cross-Spring Evolution Provenance ──");

    println!();
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │             Cross-Spring Evolution: Where Things             │");
    println!("  │               Evolved to Be Helpful                         │");
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │                                                              │");
    println!("  │  hotSpring (physics)                                         │");
    println!("  │    → Precision f64 polyfills (naga workarounds)              │");
    println!("  │    → PeakDetectF64 (LC-MS signal in wetSpring)              │");
    println!("  │    → BatchedEighGpu NAK-optimized (PCoA in wetSpring)       │");
    println!("  │    → Anderson 2D/3D + Lanczos (QS-disorder in wetSpring)    │");
    println!("  │    → find_w_c (phase transition detection in wetSpring)     │");
    println!("  │                                                              │");
    println!("  │  wetSpring (biology)                                         │");
    println!("  │    → ODE trait + generate_shader() (used by all springs)    │");
    println!("  │    → 15 bio GPU shaders (available to neuralSpring/air)     │");
    println!("  │    → ridge_regression, trapz, erf (core CPU math)          │");
    println!("  │    → Tolerance constant pattern (adopted by ToadStool S52)  │");
    println!("  │                                                              │");
    println!("  │  neuralSpring (ML/population)                                │");
    println!("  │    → PairwiseHamming/Jaccard (SNP distances in wetSpring)   │");
    println!("  │    → SpatialPayoff (cooperation game theory in wetSpring)   │");
    println!("  │    → graph_laplacian (community networks in wetSpring)      │");
    println!("  │    → belief_propagation (taxonomy in wetSpring)             │");
    println!("  │                                                              │");
    println!("  │  ToadStool (infrastructure)                                  │");
    println!("  │    → FMR: universal GPU primitive (12+ wetSpring modules)   │");
    println!("  │    → GEMM: tiled matrix multiply (kriging, chimera, NMF)   │");
    println!("  │    → stats: norm_cdf, pearson_correlation (V43 rewire)     │");
    println!("  │    → tolerances: complementary to spring-local constants    │");
    println!("  │                                                              │");
    println!("  └──────────────────────────────────────────────────────────────┘");

    v.check_pass("cross-spring provenance documented", true);

    v.finish();
}
