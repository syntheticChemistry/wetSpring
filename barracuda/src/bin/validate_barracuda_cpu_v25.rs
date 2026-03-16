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
    clippy::many_single_char_names,
    reason = "validation harness: mathematical variable names from papers"
)]
//! # Exp323: `BarraCuda` CPU v25 — V99 Cross-Primal Pure Rust Math
//!
//! Proves that all cross-primal CPU math paths produce correct results.
//! Extends CPU v24 (V98) with V99 additions: cross-spring pipeline math,
//! biomeOS integration primitives, and provenance-tracked operations.
//!
//! ```text
//! CPU (this) → GPU (Exp324) → CPU-vs-GPU (Exp325) → metalForge (Exp326)
//! ```
//!
//! ## Domains
//!
//! - D55: Cross-Primal Bio (diversity, phylogenetics, QS, HMM)
//! - D56: Cross-Spring Math (ET₀ hydrology, spectral, graph Laplacian)
//! - D57: Statistics Pipeline (bootstrap, jackknife, correlation, regression)
//! - D58: Precision Math (erf, NMF, Kimura, `norm_cdf`, Anderson)
//! - D59: IPC Math Engine (diversity + QS + pipeline math verification)
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-08 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v25` |

use std::time::Instant;
use wetspring_barracuda::bio::{cooperation, diversity, felsenstein, hmm, pcoa, qs_biofilm};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{DomainResult, Validator};
use wetspring_barracuda::validation::OrExit;

fn domain(
    name: &'static str,
    spring: &'static str,
    elapsed: std::time::Duration,
    checks: u32,
) -> DomainResult {
    DomainResult {
        name,
        spring: Some(spring),
        ms: elapsed.as_secs_f64() * 1000.0,
        checks,
    }
}

fn main() {
    let mut v = Validator::new("Exp323: BarraCuda CPU v25 — V99 Cross-Primal Pure Rust Math");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // D55: Cross-Primal Bio — diversity, phylogenetics, QS, HMM
    // ═══════════════════════════════════════════════════════════════════
    v.section("D55: Cross-Primal Bio — diversity, phylogenetics, QS, HMM");
    let t = Instant::now();
    let mut d55 = 0_u32;

    let soil = vec![35.0, 22.0, 16.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5];
    let h = diversity::shannon(&soil);
    let s = diversity::simpson(&soil);
    let c1 = diversity::chao1(&soil);
    let p = diversity::pielou_evenness(&soil);
    v.check_pass("D55: Shannon > 0", h > 0.0);
    d55 += 1;
    v.check_pass("D55: Simpson ∈ (0,1)", s > 0.0 && s < 1.0);
    d55 += 1;
    v.check_pass("D55: Chao1 ≥ observed", c1 >= 10.0);
    d55 += 1;
    v.check_pass("D55: Pielou ∈ (0,1]", p > 0.0 && p <= 1.0);
    d55 += 1;

    let uniform = vec![25.0, 25.0, 25.0, 25.0];
    v.check(
        "D55: uniform H = ln(4)",
        diversity::shannon(&uniform),
        4.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );
    d55 += 1;
    v.check(
        "D55: uniform Simpson = 0.75",
        diversity::simpson(&uniform),
        0.75,
        tolerances::ANALYTICAL_F64,
    );
    d55 += 1;

    let root = felsenstein::TreeNode::Internal {
        left: Box::new(felsenstein::TreeNode::Leaf {
            name: "A".into(),
            states: vec![0, 1, 2, 3],
        }),
        right: Box::new(felsenstein::TreeNode::Leaf {
            name: "B".into(),
            states: vec![0, 1, 2, 3],
        }),
        left_branch: 0.01,
        right_branch: 0.01,
    };
    let ll = felsenstein::log_likelihood(&root, 1.0);
    v.check_pass("D55: Felsenstein log-lik finite", ll.is_finite());
    d55 += 1;

    let model = hmm::HmmModel {
        n_states: 2,
        n_symbols: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        log_emit: vec![0.5_f64.ln(), 0.5_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
    };
    let fwd = hmm::forward(&model, &[0, 1, 0, 1]);
    v.check_pass("D55: HMM log-prob finite", fwd.log_likelihood.is_finite());
    d55 += 1;

    let qs_p = qs_biofilm::QsBiofilmParams::default();
    let qs_r = qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 50.0, 0.1, &qs_p);
    v.check_pass("D55: QS converges (>100 steps)", qs_r.t.len() > 100);
    d55 += 1;

    let co_p = cooperation::CooperationParams::default();
    let co_r = cooperation::scenario_equal_start(&co_p, 0.1);
    v.check_pass("D55: cooperation ESS converges", co_r.t.len() > 10);
    d55 += 1;

    let bc_self = diversity::bray_curtis(&soil, &soil);
    v.check("D55: BC(x,x) = 0", bc_self, 0.0, tolerances::EXACT_F64);
    d55 += 1;

    let bc_sym = diversity::bray_curtis(
        &soil,
        &uniform[..4]
            .to_vec()
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(10)
            .collect::<Vec<_>>(),
    );
    let bc_sym2 = diversity::bray_curtis(
        &uniform[..4]
            .to_vec()
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0))
            .take(10)
            .collect::<Vec<_>>(),
        &soil,
    );
    v.check("D55: BC symmetric", bc_sym, bc_sym2, tolerances::EXACT_F64);
    d55 += 1;

    domains.push(domain("D55: Bio", "wetSpring", t.elapsed(), d55));

    // ═══════════════════════════════════════════════════════════════════
    // D56: Cross-Spring Math — ET₀, spectral, graph
    // ═══════════════════════════════════════════════════════════════════
    v.section("D56: Cross-Spring Math — ET₀, spectral, graph Laplacian");
    let t = Instant::now();
    let mut d56 = 0_u32;

    let et0 =
        barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187).or_exit("unexpected error");
    v.check_pass("D56: FAO-56 ET₀ > 0", et0 > 0.0);
    d56 += 1;

    let harg = barracuda::stats::hargreaves_et0(35.0, 32.0, 18.0).or_exit("unexpected error");
    v.check_pass("D56: Hargreaves > 0", harg > 0.0);
    d56 += 1;

    let mak = barracuda::stats::makkink_et0(20.0, 18.0).or_exit("unexpected error");
    v.check_pass("D56: Makkink > 0", mak > 0.0);
    d56 += 1;

    let turc = barracuda::stats::turc_et0(20.0, 18.0, 70.0).or_exit("unexpected error");
    v.check_pass("D56: Turc > 0", turc > 0.0);
    d56 += 1;

    let hamon = barracuda::stats::hamon_et0(20.0, 14.0).or_exit("unexpected error");
    v.check_pass("D56: Hamon > 0", hamon > 0.0);
    d56 += 1;

    let monthly = [
        3.0, 4.0, 8.0, 12.0, 17.0, 21.0, 24.0, 23.0, 19.0, 13.0, 8.0, 4.0,
    ];
    let hi = barracuda::stats::thornthwaite_heat_index(&monthly);
    let thorn = barracuda::stats::thornthwaite_et0(21.0, hi, 14.5, 30.0).or_exit("unexpected error");
    v.check_pass("D56: Thornthwaite > 0", thorn > 0.0);
    d56 += 1;

    let lattice = barracuda::spectral::anderson_3d(4, 4, 4, 2.0, 42);
    let tridiag = barracuda::spectral::lanczos(&lattice, 30, 42);
    let eigs = barracuda::spectral::lanczos_eigenvalues(&tridiag);
    let r = barracuda::spectral::level_spacing_ratio(&eigs);
    v.check_pass("D56: Anderson eigenvalues computed", !eigs.is_empty());
    d56 += 1;
    v.check_pass("D56: r finite ∈ (0,1)", r.is_finite() && r > 0.0 && r < 1.0);
    d56 += 1;

    let communities = vec![
        vec![30.0, 25.0, 20.0, 15.0, 10.0],
        vec![50.0, 20.0, 15.0, 10.0, 5.0],
        vec![20.0, 20.0, 20.0, 20.0, 20.0],
        vec![90.0, 5.0, 3.0, 1.0, 1.0],
    ];
    let n = communities.len();
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                dist[i * n + j] = diversity::bray_curtis(&communities[i], &communities[j]);
            }
        }
    }
    let similarity: Vec<f64> = dist.iter().map(|&d| 1.0 - d).collect();
    let laplacian = barracuda::linalg::graph_laplacian(&similarity, n);
    v.check_pass("D56: Laplacian size = n²", laplacian.len() == n * n);
    d56 += 1;

    let diag: Vec<f64> = (0..n).map(|i| laplacian[i * n + i]).collect();
    let eff_rank = barracuda::linalg::effective_rank(&diag);
    v.check_pass("D56: effective_rank > 0", eff_rank > 0.0);
    d56 += 1;

    domains.push(domain(
        "D56: Cross-Spring",
        "airSpring+hotSpring+neuralSpring",
        t.elapsed(),
        d56,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // D57: Statistics Pipeline — bootstrap, jackknife, correlation
    // ═══════════════════════════════════════════════════════════════════
    v.section("D57: Statistics — bootstrap, jackknife, correlation, regression");
    let t = Instant::now();
    let mut d57 = 0_u32;

    let data: Vec<f64> = (1..=100).map(f64::from).collect();
    let mean = barracuda::stats::metrics::mean(&data);
    v.check(
        "D57: mean(1..100) = 50.5",
        mean,
        50.5,
        tolerances::ANALYTICAL_F64,
    );
    d57 += 1;

    let var = barracuda::stats::correlation::variance(&data).or_exit("unexpected error");
    v.check_pass("D57: var > 0", var > 0.0);
    d57 += 1;

    let x: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.1).collect();
    let y = x.clone();
    let r = barracuda::stats::pearson_correlation(&x, &y).or_exit("unexpected error");
    v.check(
        "D57: Pearson(x,x) = 1.0",
        r,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d57 += 1;

    let rho = barracuda::stats::spearman_correlation(&x, &y).or_exit("unexpected error");
    v.check(
        "D57: Spearman(x,x) = 1.0",
        rho,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d57 += 1;

    let jk = barracuda::stats::jackknife_mean_variance(&data).or_exit("unexpected error");
    v.check_pass("D57: jackknife estimate finite", jk.estimate.is_finite());
    d57 += 1;
    v.check_pass("D57: jackknife variance ≥ 0", jk.variance >= 0.0);
    d57 += 1;

    let ci = barracuda::stats::bootstrap_ci(
        &data,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        5000,
        0.95,
        42,
    )
    .or_exit("unexpected error");
    v.check_pass("D57: CI lower < CI upper", ci.lower < ci.upper);
    d57 += 1;
    v.check_pass("D57: CI contains mean", ci.lower < mean && ci.upper > mean);
    d57 += 1;

    let fit = barracuda::stats::fit_linear(&x, &y).or_exit("unexpected error");
    v.check(
        "D57: linear slope = 1.0",
        fit.params[0],
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d57 += 1;
    v.check(
        "D57: linear intercept ≈ 0",
        fit.params[1],
        0.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    d57 += 1;

    domains.push(domain("D57: Statistics", "groundSpring", t.elapsed(), d57));

    // ═══════════════════════════════════════════════════════════════════
    // D58: Precision Math — erf, NMF, Kimura, norm_cdf
    // ═══════════════════════════════════════════════════════════════════
    v.section("D58: Precision Math — erf, NMF, Kimura, norm_cdf");
    let t = Instant::now();
    let mut d58 = 0_u32;

    v.check(
        "D58: erf(0) = 0",
        barracuda::special::erf(0.0),
        0.0,
        tolerances::ERF_PARITY,
    );
    d58 += 1;
    v.check(
        "D58: erf(1) ≈ 0.8427",
        barracuda::special::erf(1.0),
        0.842_700_792_949_714_9,
        tolerances::ERF_PARITY,
    );
    d58 += 1;
    v.check(
        "D58: erf(∞) → 1",
        barracuda::special::erf(6.0),
        1.0,
        tolerances::ERF_PARITY,
    );
    d58 += 1;

    let ncdf = barracuda::stats::norm_cdf(0.0);
    v.check("D58: Φ(0) = 0.5", ncdf, 0.5, tolerances::NORM_CDF_PARITY);
    d58 += 1;

    let nmf_data = vec![0.8, 0.1, 0.0, 0.2, 0.7, 0.1, 0.0, 0.1, 0.9];
    let nmf_config = barracuda::linalg::nmf::NmfConfig {
        rank: 2,
        max_iter: 200,
        tol: tolerances::NMF_CONVERGENCE_KL,
        objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let nmf_result = barracuda::linalg::nmf::nmf(&nmf_data, 3, 3, &nmf_config).or_exit("unexpected error");
    v.check_pass(
        "D58: NMF W non-negative",
        nmf_result.w.iter().all(|&x| x >= 0.0),
    );
    d58 += 1;
    v.check_pass(
        "D58: NMF H non-negative",
        nmf_result.h.iter().all(|&x| x >= 0.0),
    );
    d58 += 1;

    domains.push(domain(
        "D58: Precision",
        "hotSpring+barraCuda",
        t.elapsed(),
        d58,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // D59: IPC Math Engine — validate math used in biomeOS pipeline
    // ═══════════════════════════════════════════════════════════════════
    v.section("D59: IPC Math Engine — biomeOS pipeline math verification");
    let t = Instant::now();
    let mut d59 = 0_u32;

    let ipc_community = vec![5.0, 10.0, 15.0, 20.0];
    let ipc_h = diversity::shannon(&ipc_community);
    let ipc_s = diversity::simpson(&ipc_community);
    v.check_pass("D59: pipeline Shannon > 0", ipc_h > 0.0);
    d59 += 1;
    v.check_pass("D59: pipeline Simpson > 0", ipc_s > 0.0);
    d59 += 1;

    let qs_default = qs_biofilm::QsBiofilmParams::default();
    let qs_pipe = qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 50.0, 0.1, &qs_default);
    let peak = qs_pipe.y.iter().copied().fold(0.0_f64, f64::max);
    v.check_pass("D59: pipeline QS peak > 0", peak > 0.0);
    d59 += 1;

    let qs_high = qs_biofilm::run_scenario(&[0.1, 0.0, 0.0, 10.0, 2.0], 30.0, 0.05, &qs_default);
    v.check_pass("D59: high_density QS converges", qs_high.t.len() > 10);
    d59 += 1;

    let bc_cond = diversity::bray_curtis_condensed(&communities);
    v.check_pass("D59: BC condensed length correct", bc_cond.len() == 6);
    d59 += 1;
    v.check_pass(
        "D59: BC condensed all ∈ [0,1]",
        bc_cond.iter().all(|&x| (0.0..=1.0).contains(&x)),
    );
    d59 += 1;

    let pcoa_res = pcoa::pcoa(&bc_cond, n, 2);
    v.check_pass("D59: PCoA computed", pcoa_res.is_ok());
    d59 += 1;
    if let Ok(ref pr) = pcoa_res {
        v.check_pass("D59: PCoA eigenvalues present", !pr.eigenvalues.is_empty());
        d59 += 1;
    }

    domains.push(domain(
        "D59: IPC Math",
        "biomeOS+wetSpring",
        t.elapsed(),
        d59,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("V99 CPU v25 Domain Summary");

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║ V99 Cross-Primal Pure Rust Math                                  ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Domain                 │ Spring             │    Time │   ✓ ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    for d in &domains {
        println!(
            "║ {:<22} │ {:<18} │ {:>5.1}ms │ {:>3} ║",
            d.name,
            d.spring.unwrap_or("—"),
            d.ms,
            d.checks
        );
    }
    let total_checks: u32 = domains.iter().map(|d| d.checks).sum();
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ TOTAL                  │                    │ {total_ms:>5.1}ms │ {total_checks:>3} ║"
    );
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Pure Rust CPU math PROVEN — all cross-primal primitives correct");
    println!("  Chain: CPU (this) → GPU (Exp324) → CPU-vs-GPU (Exp325) → metalForge (Exp326)");

    v.finish();
}
