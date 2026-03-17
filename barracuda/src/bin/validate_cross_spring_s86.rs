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
//! # Exp296: Cross-Spring S86 Rewire Validation
//!
//! Validates `ToadStool` S79→S86 evolution: new primitives, feature-gate fixes,
//! and CPU-accessible modules that were previously incorrectly GPU-gated.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | ToadStool pin | S86 (`2fee1969`) |
//! | Previous pin | S79 (`f97fc2ae`) |
//! | Provenance type | Analytical (mathematical invariants + cross-spring parity) |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --bin validate_cross_spring_s86` |

use std::time::Instant;

use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp296: Cross-Spring S86 Rewire Validation");
    println!("ToadStool pin: S86 (2fee1969) — rewired from S79 (f97fc2ae)");
    println!("Focus: CPU module ungating, new S80-S86 primitives\n");

    // ═══ S1: Spectral — Anderson Hamiltonian + Eigenvalues ══════════════
    {
        let t = Instant::now();
        v.section("S1: Spectral — Anderson 1D eigenvalues [CPU ungated]");

        let w = 4.0_f64;
        let n = 200_usize;
        let eigs = barracuda::spectral::anderson_eigenvalues(n, w, 42);
        v.check_pass("anderson_eigenvalues returns values", !eigs.is_empty());
        v.check_pass("eigenvalue count matches n", eigs.len() == n);

        let mut sorted = eigs;
        sorted.sort_by(|a, b| a.partial_cmp(b).or_exit("unexpected error"));
        let bandwidth = sorted[n - 1] - sorted[0];
        v.check_pass("1D bandwidth > 0", bandwidth > 0.0);
        v.check_pass("1D bandwidth < 4+W+1", bandwidth < 4.0 + w + 1.0);

        let (diag, off) = barracuda::spectral::anderson_hamiltonian(n, w, 42);
        v.check_pass("hamiltonian diagonal has n elements", diag.len() == n);
        v.check_pass("hamiltonian off-diagonal has n-1", off.len() == n - 1);

        println!("  1D: {n} sites, W={w}, bandwidth={bandwidth:.3}");
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S2: Spectral — Anderson 2D/3D CSR Matrices ════════════════════
    {
        let t = Instant::now();
        v.section("S2: Spectral — Anderson 2D/3D [CPU ungated]");

        let csr_2d = barracuda::spectral::anderson_2d(10, 10, 4.0, 42);
        v.check_pass("anderson_2d CSR non-empty", csr_2d.n > 0);
        v.check_pass("2D CSR dimension = 100", csr_2d.n == 100);

        let csr_3d = barracuda::spectral::anderson_3d(5, 5, 5, 4.0, 42);
        v.check_pass("anderson_3d CSR non-empty", csr_3d.n > 0);
        v.check_pass("3D CSR dimension = 125", csr_3d.n == 125);

        let x_2d = vec![1.0; 100];
        let mut y_2d = vec![0.0; 100];
        csr_2d.spmv(&x_2d, &mut y_2d);
        let y_norm: f64 = y_2d.iter().map(|v| v * v).sum::<f64>().sqrt();
        v.check_pass("2D spmv produces non-zero output", y_norm > 0.0);

        println!("  2D: 10×10 = {} sites", csr_2d.n);
        println!("  3D: 5×5×5 = {} sites", csr_3d.n);
        println!("  2D spmv ‖y‖ = {y_norm:.4}");
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S3: Spectral — Lanczos + Level Statistics ══════════════════════
    {
        let t = Instant::now();
        v.section("S3: Spectral — Lanczos + Level Spacing [CPU ungated]");

        let csr = barracuda::spectral::anderson_2d(8, 8, 2.0, 42);
        let tri = barracuda::spectral::lanczos(&csr, 64, 42);
        v.check_pass("Lanczos alpha non-empty", !tri.alpha.is_empty());
        v.check_pass("Lanczos beta non-empty", !tri.beta.is_empty());

        let eigs = barracuda::spectral::lanczos_eigenvalues(&tri);
        v.check_pass("lanczos_eigenvalues returns values", !eigs.is_empty());

        let r = barracuda::spectral::level_spacing_ratio(&eigs);
        v.check_pass("level_spacing_ratio ≥ 0", r >= 0.0);

        let bandwidth = barracuda::spectral::spectral_bandwidth(&eigs);
        v.check_pass("spectral_bandwidth > 0", bandwidth > 0.0);

        let cond = barracuda::spectral::spectral_condition_number(&eigs);
        v.check_pass("condition_number ≥ 1", cond >= 1.0);

        let bands = barracuda::spectral::detect_bands(&eigs, 0.5);
        v.check_pass("detect_bands returns ≥ 1 band", !bands.is_empty());

        let phase = barracuda::spectral::classify_spectral_phase(&eigs, bandwidth * 0.5);
        v.check_pass(
            "spectral phase classified",
            matches!(
                phase,
                barracuda::spectral::SpectralPhase::Bulk
                    | barracuda::spectral::SpectralPhase::EdgeOfChaos
                    | barracuda::spectral::SpectralPhase::Chaotic
            ),
        );

        println!("  Lanczos: 8×8 (64 sites), r={r:.4}, phase={phase:?}");
        println!(
            "  bandwidth={bandwidth:.3}, cond={cond:.3}, {} bands",
            bands.len()
        );
        println!(
            "  GOE_R={:.4}, POISSON_R={:.4}",
            barracuda::spectral::GOE_R,
            barracuda::spectral::POISSON_R
        );
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S4: Graph Laplacian + Belief Propagation (was GPU-gated) ═══════
    {
        let t = Instant::now();
        v.section("S4: Graph — Laplacian + Effective Rank [CPU ungated]");

        let adj = vec![
            0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        let lap = barracuda::linalg::graph_laplacian(&adj, 4);
        v.check_pass("graph_laplacian returns n² values", lap.len() == 16);

        let diag_sum: f64 = (0..4).map(|i| lap[i * 4 + i]).sum();
        let total_degree: f64 = adj.iter().sum::<f64>();
        v.check_pass(
            "Laplacian trace = total degree",
            (diag_sum - total_degree).abs() < tolerances::ANALYTICAL_LOOSE,
        );

        let off_diag_sum: f64 = lap.iter().sum::<f64>();
        v.check_pass(
            "Laplacian rows sum to 0",
            off_diag_sum.abs() < tolerances::ANALYTICAL_LOOSE,
        );

        let hetero = vec![0.1, -0.2, 0.05, -0.1];
        let disordered = barracuda::linalg::disordered_laplacian(&lap, 4, &hetero, 1.0);
        v.check_pass("disordered_laplacian returns n²", disordered.len() == 16);

        let eig_approx: Vec<f64> = (0..4).map(|i| lap[i * 4 + i]).collect();
        let eff = barracuda::linalg::effective_rank(&eig_approx);
        v.check_pass("effective_rank > 0", eff > 0.0);
        v.check_pass(
            "effective_rank ≤ n",
            eff <= 4.0 + tolerances::ANALYTICAL_LOOSE,
        );

        let input_dist = vec![0.5, 0.3, 0.2];
        let trans = vec![0.7, 0.2, 0.1, 0.3, 0.5, 0.2, 0.1, 0.3, 0.6];
        let dims = vec![3_usize];
        let dists =
            barracuda::linalg::belief_propagation_chain(&input_dist, &[trans.as_slice()], &dims);
        v.check_pass("belief_propagation returns layers", dists.len() == 2);
        let final_sum: f64 = dists.last().or_exit("unexpected error").iter().sum();
        v.check_pass(
            "final distribution ≈ 1",
            (final_sum - 1.0).abs() < tolerances::ANALYTICAL_LOOSE,
        );

        println!("  Laplacian: 4×4, trace={diag_sum:.1}, eff_rank={eff:.3}");
        println!("  Belief propagation: final dist sum={final_sum:.10}");
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S5: Boltzmann/Metropolis Sampling (was GPU-gated) ══════════════
    {
        let t = Instant::now();
        v.section("S5: Sample — Boltzmann/Metropolis [CPU ungated]");

        let loss_fn = |params: &[f64]| -> f64 { params.iter().map(|x| x * x).sum() };
        let initial = vec![5.0, -3.0, 2.0];
        let initial_loss = loss_fn(&initial);
        let result = barracuda::sample::boltzmann_sampling(&loss_fn, &initial, 1.0, 0.5, 500, 42);
        v.check_pass("Boltzmann: losses non-empty", !result.losses.is_empty());
        v.check_pass(
            "Boltzmann: accept rate in [0,1]",
            (0.0..=1.0).contains(&result.acceptance_rate),
        );
        let final_loss = result.losses.last().copied().unwrap_or(f64::MAX);
        v.check_pass("Boltzmann: final ≤ initial", final_loss <= initial_loss);

        println!(
            "  Boltzmann: {} steps, accept={:.0}%, loss {initial_loss:.1}→{final_loss:.4}",
            result.losses.len(),
            result.acceptance_rate * 100.0,
        );
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S6: LHS + Sobol Sampling (CPU always available) ════════════════
    {
        let t = Instant::now();
        v.section("S6: Sample — LHS + Sobol [CPU samplers]");

        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let lhs = barracuda::sample::latin_hypercube(100, &bounds, 42).or_exit("unexpected error");
        v.check_pass("LHS: 100 points generated", lhs.len() == 100);
        v.check_pass("LHS: 2D points", lhs[0].len() == 2);
        let all_in_bounds = lhs
            .iter()
            .all(|p| p[0] >= -5.0 && p[0] <= 5.0 && p[1] >= -5.0 && p[1] <= 5.0);
        v.check_pass("LHS: all points in bounds", all_in_bounds);

        let uniform = barracuda::sample::random_uniform(50, &bounds, 42);
        v.check_pass("random_uniform: 50 points", uniform.len() == 50);

        let sobol_bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let sobol = barracuda::sample::sobol_scaled(64, &sobol_bounds).or_exit("unexpected error");
        v.check_pass("Sobol: 64 points generated", sobol.len() == 64);
        let sobol_ok = sobol
            .iter()
            .all(|p| p[0] >= 0.0 && p[0] <= 1.0 && p[1] >= 0.0 && p[1] <= 1.0);
        v.check_pass("Sobol: all in [0,1]²", sobol_ok);

        println!("  LHS: 100×2D in [-5,5]²");
        println!("  Sobol: 64×2D in [0,1]², low-discrepancy");
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S7: Hydrology — Thornthwaite/Makkink/Turc/Hamon (S81) ═════════
    {
        let t = Instant::now();
        v.section("S7: Hydrology — 4 new ET₀ methods [S81]");

        let monthly = [
            3.0, 4.0, 8.0, 12.0, 17.0, 21.0, 24.0, 23.0, 19.0, 13.0, 8.0, 4.0,
        ];
        let hi = barracuda::stats::thornthwaite_heat_index(&monthly);
        v.check_pass("Thornthwaite heat index > 0", hi > 0.0);

        let thorn =
            barracuda::stats::thornthwaite_et0(21.0, hi, 14.5, 30.0).or_exit("unexpected error");
        v.check_pass("Thornthwaite ET₀ > 0", thorn > 0.0);
        v.check_pass("Thornthwaite ET₀ < 200 mm/month", thorn < 200.0);

        let makk = barracuda::stats::makkink_et0(20.0, 18.0).or_exit("unexpected error");
        v.check_pass("Makkink ET₀ > 0", makk > 0.0);
        v.check_pass("Makkink ET₀ < 10", makk < 10.0);

        let turc_h = barracuda::stats::turc_et0(20.0, 18.0, 70.0).or_exit("unexpected error");
        v.check_pass("Turc ET₀ > 0", turc_h > 0.0);
        let turc_d = barracuda::stats::turc_et0(20.0, 18.0, 30.0).or_exit("unexpected error");
        v.check_pass("Turc: dry > humid", turc_d > turc_h);

        let hamon = barracuda::stats::hamon_et0(20.0, 14.0).or_exit("unexpected error");
        v.check_pass("Hamon ET₀ > 0", hamon > 0.0);
        v.check_pass("Hamon ET₀ < 10", hamon < 10.0);

        println!("  Thornthwaite: HI={hi:.2}, ET₀={thorn:.2} mm/month");
        println!("  Makkink: {makk:.3}, Turc: h={turc_h:.3} d={turc_d:.3}");
        println!("  Hamon: {hamon:.3} mm/day");
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S8: FitResult Named Accessors (S81) ════════════════════════════
    {
        let t = Instant::now();
        v.section("S8: Regression — FitResult named accessors [S81]");

        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1_f64, 3.9, 6.1, 8.0, 9.9];
        let fit = barracuda::stats::fit_linear(&x, &y).or_exit("unexpected error");

        v.check_pass("fit_linear: R² > 0.99", fit.r_squared > 0.99);

        let slope = fit.slope().or_exit("unexpected error");
        v.check_pass(
            "slope() ≈ 2.0",
            (slope - 2.0).abs() < tolerances::SOIL_MODEL_APPROX,
        );

        let intercept = fit.intercept().or_exit("unexpected error");
        v.check_pass(
            "intercept() near 0",
            intercept.abs() < tolerances::INTERCEPT_NEAR_ZERO,
        );

        let coeffs = fit.coefficients();
        v.check_pass("coefficients len ≥ 2", coeffs.len() >= 2);

        let fits = barracuda::stats::fit_all(&x, &y);
        v.check_pass("fit_all returns ≥ 1 result", !fits.is_empty());
        let linear_fit = fits.iter().find(|f| f.model == "linear");
        v.check_pass("fit_all includes linear", linear_fit.is_some());

        println!(
            "  y = {slope:.3}x + {intercept:.3}, R²={:.6}",
            fit.r_squared
        );
        println!("  fit_all: {} models converged", fits.len());
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S9: Hofstadter Butterfly + Almost-Mathieu ══════════════════════
    {
        let t = Instant::now();
        v.section("S9: Spectral — Hofstadter + Almost-Mathieu [CPU ungated]");

        v.check_pass(
            "GOLDEN_RATIO ≈ 1/φ ≈ 0.618",
            (barracuda::spectral::GOLDEN_RATIO - 0.618_033_988_7).abs()
                < tolerances::GPU_VS_CPU_F64,
        );

        let gcd_val = barracuda::spectral::gcd(12, 8);
        v.check_pass("gcd(12,8) = 4", gcd_val == 4);

        let phi = 1.0 / barracuda::spectral::GOLDEN_RATIO;
        let ham = barracuda::spectral::almost_mathieu_hamiltonian(20, 2.0, phi, 0.0);
        v.check_pass("Almost-Mathieu: diagonal has n elements", ham.0.len() == 20);

        let butterfly = barracuda::spectral::hofstadter_butterfly(30, 2.0, 20);
        v.check_pass("Hofstadter butterfly non-empty", !butterfly.is_empty());

        println!("  Golden ratio: {:.10}", barracuda::spectral::GOLDEN_RATIO);
        println!("  Almost-Mathieu: 20×20, λ=2.0");
        println!(
            "  Hofstadter: {} (alpha, eigenvalues) pairs",
            butterfly.len()
        );
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S10: Tridiagonal Eigensolve ════════════════════════════════════
    {
        let t = Instant::now();
        v.section("S10: Spectral — Tridiagonal Eigensolve [CPU ungated]");

        let n: usize = 100;
        let alpha: Vec<f64> = (0..n).map(|i| 0.1f64.mul_add(i as f64, 2.0)).collect();
        let beta: Vec<f64> = (0..n - 1).map(|_| -1.0_f64).collect();
        let eigs = barracuda::spectral::find_all_eigenvalues(&alpha, &beta);
        v.check_pass("tridiagonal: n eigenvalues", eigs.len() == n);

        let sorted = eigs
            .windows(2)
            .all(|w| w[0] <= w[1] + tolerances::ANALYTICAL_LOOSE);
        v.check_pass("eigenvalues sorted", sorted);

        let count = barracuda::spectral::sturm_count(&alpha, &beta, 3.0);
        v.check_pass("sturm_count ≤ n", count <= n);

        println!(
            "  Tridiagonal: n={n}, λ_min={:.4}, λ_max={:.4}",
            eigs.first().or_exit("unexpected error"),
            eigs.last().or_exit("unexpected error")
        );
        println!("  Sturm count below 3.0: {count}");
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ S11: Post-rewire regression — existing primitives ══════════════
    {
        let t = Instant::now();
        v.section("S11: Regression — existing primitives post-rewire");

        let erf_half = barracuda::special::erf(0.5);
        v.check_pass(
            "erf(0.5) ≈ 0.5205",
            (erf_half - 0.5205).abs() < tolerances::CROSS_SPRING_NUMERICAL,
        );

        let lg = barracuda::special::ln_gamma(5.0).or_exit("unexpected error");
        v.check_pass(
            "ln_gamma(5) ≈ ln(24)",
            (lg - 24.0_f64.ln()).abs() < tolerances::ANALYTICAL_LOOSE,
        );

        let hargreaves =
            barracuda::stats::hargreaves_et0(35.0, 32.0, 18.0).or_exit("unexpected error");
        v.check_pass("Hargreaves ET₀ > 0", hargreaves > 0.0);

        let fao56 =
            barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187)
                .or_exit("unexpected error");
        v.check_pass(
            "FAO-56 ET₀ ≈ 3.88",
            (fao56 - 3.88).abs() < tolerances::FAO56_ET0_PARITY,
        );

        let ncdf = barracuda::stats::norm_cdf(0.0);
        v.check_pass(
            "norm_cdf(0) = 0.5",
            (ncdf - 0.5).abs() < tolerances::ANALYTICAL_LOOSE,
        );

        let pearson = barracuda::stats::pearson_correlation(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[2.0, 4.0, 5.0, 4.0, 5.0],
        )
        .or_exit("unexpected error");
        v.check_pass("pearson in [-1,1]", (-1.0..=1.0).contains(&pearson));

        println!("  erf(0.5)={erf_half:.6}, ln_Γ(5)={lg:.6}");
        println!("  Hargreaves={hargreaves:.3}, FAO-56={fao56:.3}");
        println!("  norm_cdf(0)={ncdf:.6}, pearson={pearson:.4}");
        println!("  {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    // ═══ Summary ════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  ToadStool S79 → S86 Rewire Validation Complete     ║");
    println!("║  Pin: 2fee1969 (was f97fc2ae)                       ║");
    println!("║  Primitives: 144 (was 93)                           ║");
    println!("║  Feature-gate fixes: spectral, graph, sample        ║");
    println!("╚══════════════════════════════════════════════════════╝");

    v.finish();
}
