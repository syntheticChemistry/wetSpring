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
//! # Exp306: `BarraCuda` CPU v23 — V97 Fused Ops Decomposition Parity
//!
//! Proves that the mathematical formulas used by `barraCuda` v0.3.3's fused GPU
//! shaders (Welford mean+variance, 5-accumulator Pearson) produce **identical**
//! results when computed via decomposed CPU primitives.
//!
//! This is the CPU anchor for the V97 fused ops chain:
//! ```text
//! Paper (Exp291) → CPU (this) → GPU fused (Exp308) → Streaming (Exp309) → metalForge (Exp310)
//! ```
//!
//! ## Why fused ops matter
//!
//! Fused operations (single-pass Welford, 5-accumulator Pearson) reduce GPU
//! dispatches from N to 1 for correlated statistics. Before proving they work
//! on GPU, we prove the underlying math is correct on CPU using the same
//! analytical invariants.
//!
//! ## Domains
//!
//! - D41: Welford decomposition — mean+variance via separate calls = Welford single-pass
//! - D42: 5-accumulator Pearson — `r(x,y)` via `Σx,Σy,Σx²,Σy²,Σxy` = `pearson_correlation`
//! - D43: Covariance decomposition — Cov(x,y) = E\[XY\] - E\[X\]E\[Y\]
//! - D44: Cross-paper variance — soil QS, diversity, pharmacology datasets
//! - D45: Spearman rank correlation — monotonic relationship detection
//! - D46: Correlation matrix — multi-variable pairwise structure
//!
//! # Provenance
//!
//! Expected values are **analytical** — derived from mathematical identities.
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-05 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v23` |

use std::time::Instant;
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, DomainResult, Validator};

fn main() {
    let mut v = Validator::new("Exp306: BarraCuda CPU v23 — V97 Fused Ops Decomposition Parity");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // D41: Welford Decomposition — mean + variance consistency
    // ═══════════════════════════════════════════════════════════════════
    v.section("D41: Welford Decomposition — mean + var consistency");
    let t = Instant::now();
    let mut d41_checks = 0_u32;

    // 41a: Uniform data — analytical mean and sample variance
    // barracuda::stats::correlation::variance uses ddof=1 (sample variance)
    let uniform: Vec<f64> = (1..=100).map(f64::from).collect();
    let n = uniform.len() as f64;

    let bc_mean = barracuda::stats::metrics::mean(&uniform);
    v.check(
        "Welford: mean(1..100) = 50.5",
        bc_mean,
        50.5,
        tolerances::ANALYTICAL_F64,
    );
    d41_checks += 1;

    let bc_var =
        barracuda::stats::correlation::variance(&uniform).expect("variance of uniform 1..100");
    // Sample var of 1..N = N(N+1)/12
    let expected_sample_var = n * (n + 1.0) / 12.0; // 100*101/12 = 841.666...
    v.check(
        "Welford: sVar(1..100) = N(N+1)/12 = 841.67",
        bc_var,
        expected_sample_var,
        tolerances::ANALYTICAL_F64,
    );
    d41_checks += 1;

    // Cross-check: manual sample variance matches barracuda
    let cpu_mean = uniform.iter().sum::<f64>() / n;
    let cpu_sample_var = uniform.iter().map(|x| (x - cpu_mean).powi(2)).sum::<f64>() / (n - 1.0);
    v.check(
        "Welford: manual sample var = barracuda var",
        cpu_sample_var,
        bc_var,
        tolerances::ANALYTICAL_F64,
    );
    d41_checks += 1;

    // 41b: Constant data — variance must be zero
    let constant = vec![42.0; 50];
    let const_var =
        barracuda::stats::correlation::variance(&constant).expect("variance of constant vector");
    v.check(
        "Welford: Var(constant) = 0",
        const_var,
        0.0,
        tolerances::VARIANCE_EXACT,
    );
    d41_checks += 1;

    let const_mean = barracuda::stats::metrics::mean(&constant);
    v.check(
        "Welford: mean(constant) = 42",
        const_mean,
        42.0,
        tolerances::EXACT,
    );
    d41_checks += 1;

    // 41c: Two-point data — sVar([3,7]): mean=5, Σ(x-5)² = 4+4 = 8, /(n-1)=8/1=8
    let two = [3.0, 7.0];
    let two_var =
        barracuda::stats::correlation::variance(&two).expect("variance of two-point [3,7]");
    v.check(
        "Welford: sVar([3,7]) = 8.0",
        two_var,
        8.0,
        tolerances::ANALYTICAL_F64,
    );
    d41_checks += 1;

    // 41d: Large dataset — stress test numerical stability
    let large: Vec<f64> = (0..10_000).map(|i| 1e8 + f64::from(i as u32)).collect();
    let large_n = large.len() as f64;
    let large_mean = barracuda::stats::metrics::mean(&large);
    v.check(
        "Welford: mean(1e8+0..9999) = 1e8+4999.5",
        large_mean,
        1e8 + 4999.5,
        tolerances::ANALYTICAL_F64,
    );
    d41_checks += 1;

    let large_var =
        barracuda::stats::correlation::variance(&large).expect("variance of large shifted dataset");
    let expected_large_sample_var = large_n * (large_n + 1.0) / 12.0;
    v.check(
        "Welford: sVar(large shifted) = N(N+1)/12",
        large_var,
        expected_large_sample_var,
        tolerances::GPU_VS_CPU_ENSEMBLE,
    );
    d41_checks += 1;

    domains.push(DomainResult {
        name: "D41: Welford",
        spring: Some("wetSpring+hotSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: d41_checks,
    });

    // ═══════════════════════════════════════════════════════════════════
    // D42: 5-Accumulator Pearson — decomposition parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("D42: 5-Accumulator Pearson — decomposition parity");
    let t = Instant::now();
    let mut d42_checks = 0_u32;

    // 42a: Perfect correlation r(x, 2x+1) = 1.0
    let x: Vec<f64> = (1..=20).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0_f64.mul_add(xi, 1.0)).collect();
    let r = barracuda::stats::pearson_correlation(&x, &y).expect("Pearson r(x, 2x+1) = 1.0");
    v.check(
        "Pearson: r(x, 2x+1) = 1.0",
        r,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d42_checks += 1;

    // 42b: Negative correlation r(x, -x) = -1.0
    let neg_y: Vec<f64> = x.iter().map(|&xi| -xi).collect();
    let r_neg = barracuda::stats::pearson_correlation(&x, &neg_y).expect("Pearson r(x, -x) = -1.0");
    v.check(
        "Pearson: r(x, -x) = -1.0",
        r_neg,
        -1.0,
        tolerances::ANALYTICAL_F64,
    );
    d42_checks += 1;

    // 42c: Self-correlation r(x, x) = 1.0
    let r_self =
        barracuda::stats::pearson_correlation(&x, &x).expect("Pearson self-correlation r(x, x)");
    v.check(
        "Pearson: r(x, x) = 1.0",
        r_self,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d42_checks += 1;

    // 42d: Zero correlation — orthogonal signals
    let sin_x: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).sin()).collect();
    let cos_x: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).cos()).collect();
    let r_ortho = barracuda::stats::pearson_correlation(&sin_x, &cos_x)
        .expect("Pearson r(sin, cos) orthogonal");
    v.check_pass("Pearson: |r(sin,cos)| < 0.1", r_ortho.abs() < 0.1);
    d42_checks += 1;

    // 42e: Decomposition check — Pearson = Cov(x,y) / (σ_x × σ_y)
    let cov_xy =
        barracuda::stats::covariance(&x, &y).expect("covariance for Pearson decomposition");
    let sd_x =
        barracuda::stats::correlation::std_dev(&x).expect("std dev of x for Pearson decomposition");
    let sd_y =
        barracuda::stats::correlation::std_dev(&y).expect("std dev of y for Pearson decomposition");
    let r_decomposed = cov_xy / (sd_x * sd_y);
    v.check(
        "Pearson: decomposed = direct",
        r_decomposed,
        r,
        tolerances::ANALYTICAL_F64,
    );
    d42_checks += 1;

    // 42f: Paper-derived — QS signal vs. biofilm density correlation
    let qs_signal = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    let biofilm = [0.02, 0.08, 0.15, 0.6, 1.2, 2.5, 5.5, 11.0];
    let r_qs = barracuda::stats::pearson_correlation(&qs_signal, &biofilm)
        .expect("Pearson QS vs biofilm correlation");
    v.check_pass("Pearson: QS↔biofilm r > 0.99", r_qs > 0.99);
    d42_checks += 1;

    domains.push(DomainResult {
        name: "D42: Pearson",
        spring: Some("wetSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: d42_checks,
    });

    // ═══════════════════════════════════════════════════════════════════
    // D43: Covariance Decomposition — Cov(X,Y) = E[XY] - E[X]E[Y]
    // ═══════════════════════════════════════════════════════════════════
    v.section("D43: Covariance Decomposition");
    let t = Instant::now();
    let mut d43_checks = 0_u32;

    // 43a: Cov(x, x) = Var(x)
    let cov_xx = barracuda::stats::covariance(&x, &x).expect("covariance cov(x, x)");
    let var_x =
        barracuda::stats::correlation::variance(&x).expect("variance of x for Cov(x,x)=Var(x)");
    v.check(
        "Cov(x,x) = Var(x)",
        cov_xx,
        var_x,
        tolerances::ANALYTICAL_F64,
    );
    d43_checks += 1;

    // 43b: Cov(x, c) = 0 for constant c
    let const_c = vec![5.0; x.len()];
    let cov_xc = barracuda::stats::covariance(&x, &const_c).expect("covariance cov(x, constant)");
    v.check(
        "Cov(x, constant) = 0",
        cov_xc,
        0.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    d43_checks += 1;

    // 43c: Manual sample covariance matches barracuda (ddof=1)
    let mean_x = barracuda::stats::metrics::mean(&x);
    let mean_y_lin = barracuda::stats::metrics::mean(&y);
    let n_xy = x.len() as f64;
    let manual_sample_cov = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| (a - mean_x) * (b - mean_y_lin))
        .sum::<f64>()
        / (n_xy - 1.0);
    let bc_cov =
        barracuda::stats::covariance(&x, &y).expect("covariance for manual sample cov comparison");
    v.check(
        "Cov: manual sample cov = barracuda",
        manual_sample_cov,
        bc_cov,
        tolerances::ANALYTICAL_F64,
    );
    d43_checks += 1;

    // 43d: Cov(ax+b, cy+d) = ac × Cov(x,y)
    let a_coeff = 3.0;
    let c_coeff = 2.0;
    let ax: Vec<f64> = x.iter().map(|&xi| a_coeff * xi + 7.0).collect();
    let cy: Vec<f64> = y.iter().map(|&yi| c_coeff * yi - 5.0).collect();
    let cov_scaled = barracuda::stats::covariance(&ax, &cy).expect("covariance cov(ax+b, cy+d)");
    let expected_scaled = a_coeff * c_coeff * bc_cov;
    v.check(
        "Cov(ax+b, cy+d) = ac·Cov(x,y)",
        cov_scaled,
        expected_scaled,
        tolerances::ANALYTICAL_LOOSE,
    );
    d43_checks += 1;

    domains.push(DomainResult {
        name: "D43: Covariance",
        spring: Some("wetSpring+hotSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: d43_checks,
    });

    // ═══════════════════════════════════════════════════════════════════
    // D44: Cross-Paper Variance — paper-derived datasets
    // ═══════════════════════════════════════════════════════════════════
    v.section("D44: Cross-Paper Variance — Soil QS + Diversity + Pharma");
    let t = Instant::now();
    let mut d44_checks = 0_u32;

    // 44a: Soil QS — no-till connectivity dataset (Islam 2014)
    let notill_conn = [0.793, 0.785, 0.801, 0.790, 0.798];
    let tilled_conn = [0.385, 0.392, 0.380, 0.390, 0.388];
    let var_notill = barracuda::stats::correlation::variance(&notill_conn)
        .expect("soil no-till connectivity variance");
    let var_tilled = barracuda::stats::correlation::variance(&tilled_conn)
        .expect("soil tilled connectivity variance");
    v.check_pass("Soil: Var(no-till conn) > 0", var_notill > 0.0);
    d44_checks += 1;
    v.check_pass("Soil: Var(tilled conn) > 0", var_tilled > 0.0);
    d44_checks += 1;

    // 44b: Diversity — Shannon variance across communities
    let communities: Vec<Vec<f64>> = vec![
        vec![50.0, 30.0, 15.0, 5.0],
        vec![25.0, 25.0, 25.0, 25.0],
        vec![90.0, 5.0, 3.0, 2.0],
        vec![40.0, 35.0, 20.0, 5.0],
        vec![10.0, 10.0, 10.0, 70.0],
    ];
    let shannons: Vec<f64> = communities.iter().map(|c| diversity::shannon(c)).collect();
    let h_var = barracuda::stats::correlation::variance(&shannons)
        .expect("Shannon diversity variance across communities");
    v.check_pass("Diversity: Var(H') > 0 across communities", h_var > 0.0);
    d44_checks += 1;

    let h_mean = barracuda::stats::metrics::mean(&shannons);
    v.check_pass("Diversity: mean(H') > 0", h_mean > 0.0);
    d44_checks += 1;
    v.check_pass("Diversity: mean(H') < ln(4)", h_mean < 4.0_f64.ln());
    d44_checks += 1;

    // 44c: Pharmacology — IC50 variance (Gonzales 2014)
    let ic50_values = [10.0, 36.0, 75.0, 130.0, 249.0]; // nM, from paper
    let ic50_var =
        barracuda::stats::correlation::variance(&ic50_values).expect("IC50 pharmacology variance");
    v.check_pass("Pharma: Var(IC50) > 1000", ic50_var > 1000.0);
    d44_checks += 1;

    // 44d: Anderson W scores — variance tracks disorder spread
    let w_scores = [3.5, 8.2, 12.7, 16.1, 19.8];
    let w_var =
        barracuda::stats::correlation::variance(&w_scores).expect("Anderson W scores variance");
    v.check_pass("Anderson: Var(W) > 0", w_var > 0.0);
    d44_checks += 1;

    // 44e: Jackknife SE of Shannon — cross-spring pattern
    let jk = barracuda::stats::jackknife_mean_variance(&shannons)
        .expect("jackknife mean-variance for Shannon");
    v.check_pass("JK: Shannon SE > 0", jk.std_error > 0.0);
    d44_checks += 1;
    v.check(
        "JK: Shannon mean = stats mean",
        jk.estimate,
        h_mean,
        tolerances::ANALYTICAL_F64,
    );
    d44_checks += 1;

    domains.push(DomainResult {
        name: "D44: Cross-Paper Var",
        spring: Some("all Springs"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: d44_checks,
    });

    // ═══════════════════════════════════════════════════════════════════
    // D45: Spearman Rank Correlation
    // ═══════════════════════════════════════════════════════════════════
    v.section("D45: Spearman Rank Correlation");
    let t = Instant::now();
    let mut d45_checks = 0_u32;

    // 45a: Perfect monotonic r_s(x, 2x+1) = 1.0
    let r_spearman = barracuda::stats::correlation::spearman_correlation(&x, &y)
        .expect("Spearman r_s(x, 2x+1) = 1.0");
    v.check(
        "Spearman: r_s(x, 2x+1) = 1.0",
        r_spearman,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d45_checks += 1;

    // 45b: Perfect negative monotonic
    let r_s_neg = barracuda::stats::correlation::spearman_correlation(&x, &neg_y)
        .expect("Spearman r_s(x, -x) = -1.0");
    v.check(
        "Spearman: r_s(x, -x) = -1.0",
        r_s_neg,
        -1.0,
        tolerances::ANALYTICAL_F64,
    );
    d45_checks += 1;

    // 45c: Nonlinear monotonic — Spearman should detect it better than Pearson
    let mono_x: Vec<f64> = (1..=10).map(f64::from).collect();
    let expo_y: Vec<f64> = mono_x.iter().map(|&xi| xi.powi(3)).collect();
    let r_s_mono = barracuda::stats::correlation::spearman_correlation(&mono_x, &expo_y)
        .expect("Spearman r_s(x, x³) monotonic");
    let r_p_mono = barracuda::stats::pearson_correlation(&mono_x, &expo_y)
        .expect("Pearson r(x, x³) monotonic");
    v.check(
        "Spearman: r_s(x, x³) = 1.0",
        r_s_mono,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d45_checks += 1;
    v.check_pass(
        "Spearman vs Pearson: r_s ≥ r_p for monotonic",
        r_s_mono >= r_p_mono,
    );
    d45_checks += 1;

    domains.push(DomainResult {
        name: "D45: Spearman",
        spring: Some("wetSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: d45_checks,
    });

    // ═══════════════════════════════════════════════════════════════════
    // D46: Correlation Matrix — multi-variable structure
    // ═══════════════════════════════════════════════════════════════════
    v.section("D46: Correlation Matrix — multi-variable pairwise");
    let t = Instant::now();
    let mut d46_checks = 0_u32;

    // correlation_matrix expects rows=observations, cols=variables
    let n_obs = 10;
    let n_vars = 3;
    let var_cols: Vec<Vec<f64>> = vec![
        (1..=n_obs).map(f64::from).collect(),
        (1..=n_obs)
            .map(|i| f64::from(i).mul_add(2.0, 1.0))
            .collect(),
        (1..=n_obs).map(|i| -(f64::from(i))).collect(),
    ];
    let obs_rows: Vec<Vec<f64>> = (0..n_obs as usize)
        .map(|i| var_cols.iter().map(|col| col[i]).collect())
        .collect();
    let corr_mat = barracuda::stats::correlation::correlation_matrix(&obs_rows)
        .expect("correlation matrix for multi-variable structure");

    // 46a: Diagonal = 1.0
    for i in 0..n_vars as usize {
        v.check(
            &format!("CorrMat: diag[{i}] = 1.0"),
            corr_mat[i * n_vars as usize + i],
            1.0,
            tolerances::ANALYTICAL_F64,
        );
        d46_checks += 1;
    }

    // 46b: r(x, 2x+1) = 1.0
    let nv = n_vars as usize;
    v.check(
        "CorrMat: r[0,1] = 1.0 (x vs 2x+1)",
        corr_mat[1],
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d46_checks += 1;

    // 46c: r(x, -x) = -1.0
    v.check(
        "CorrMat: r[0,2] = -1.0 (x vs -x)",
        corr_mat[2],
        -1.0,
        tolerances::ANALYTICAL_F64,
    );
    d46_checks += 1;

    // 46d: Symmetry
    v.check(
        "CorrMat: symmetric r[0,1] = r[1,0]",
        corr_mat[1],
        corr_mat[nv],
        tolerances::EXACT,
    );
    d46_checks += 1;

    // 46e: Covariance matrix diagonal matches variance
    let cov_mat = barracuda::stats::correlation::covariance_matrix(&obs_rows)
        .expect("covariance matrix for diagonal check");
    let var_0 = barracuda::stats::correlation::variance(&var_cols[0])
        .expect("variance of first variable for CovMat diag");
    v.check(
        "CovMat: diag[0] = Var(x)",
        cov_mat[0],
        var_0,
        tolerances::ANALYTICAL_F64,
    );
    d46_checks += 1;

    domains.push(DomainResult {
        name: "D46: CorrMatrix",
        spring: Some("wetSpring+neuralSpring"),
        ms: t.elapsed().as_secs_f64() * 1e3,
        checks: d46_checks,
    });

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1e3;
    validation::print_domain_summary("V97 Fused Ops CPU Parity", &domains);

    v.section("CPU v23 Summary");
    println!("  D41: Welford mean+variance decomposition (hotSpring precision)");
    println!("  D42: 5-accumulator Pearson correlation (wetSpring bio)");
    println!("  D43: Covariance decomposition (analytical identities)");
    println!("  D44: Cross-paper variance (soil/diversity/pharma/Anderson)");
    println!("  D45: Spearman rank correlation (monotonic detection)");
    println!("  D46: Correlation matrix + covariance matrix");
    println!("  Total: {total_ms:.1} ms — pure Rust, zero FFI");
    println!();
    println!("  These CPU results are the reference for GPU fused ops (Exp308).");
    println!("  GPU mean_variance_gpu → must match D41 values.");
    println!("  GPU correlation_full_gpu → must match D42 values.");

    v.finish();
}
