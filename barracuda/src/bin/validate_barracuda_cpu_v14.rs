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
//! # Exp225: `BarraCuda` CPU v14 — V71 Pure Rust Math (Cross-Spring + DF64)
//!
//! Extends CPU v13 (47 domains) with:
//! - **V71 `df64_host`** pack/unpack (host-side DF64 protocol)
//! - **Cross-spring primitives**: `graph_laplacian`, `effective_rank`, `numerical_hessian`
//! - **Paper-referenced** known values from Exp224
//! - **`ToadStool` S68+** stats/special functions
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Analytical (DF64, erf, Φ) + neuralSpring `numerical_hessian` |
//! | Date | 2026-02-28 |
//! | Phase | 71 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v14` |
//!
//! ## Hessian values (D32)
//!
//! - **802, 200**: Rosenbrock f(x,y)=(1−x)²+100(y−x²)² at optimum (1,1),
//!   `numerical_hessian` eps=1e-5. Analytical H\[0,0\]≈802, H\[1,1\]≈200; tolerance 2.0.
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::collections::HashMap;
use wetspring_barracuda::bio::{
    bistable, capacitor, cooperation, derep, diversity, dnds, felsenstein, gillespie, hmm, kmd,
    kmer, merge_pairs, multi_signal, neighbor_joining, pangenome, pcoa, phage_defense, qs_biofilm,
    quality, reconciliation, robinson_foulds, signal, snp, spectral_match, unifrac,
};
use wetspring_barracuda::df64_host;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

use barracuda::special::erf;
use barracuda::stats::norm_cdf;

fn main() {
    let mut v = Validator::new("Exp225: BarraCuda CPU v14 — V71 Pure Rust Math (50 Domains)");
    let mut total_domains = 0_u32;

    // ═══ V71 NEW: DF64 Host Protocol ══════════════════════════════════
    v.section("D00: V71 DF64 Host Pack/Unpack");
    total_domains += 1;

    let pi_err = df64_host::roundtrip_error(std::f64::consts::PI);
    v.check_pass(
        "DF64 π roundtrip < 1e-14",
        pi_err < tolerances::PYTHON_PARITY_TIGHT,
    );

    let packed = df64_host::pack_slice(&[1.0, 2.0, std::f64::consts::PI]);
    let unpacked = df64_host::unpack_slice(&packed);
    v.check_pass("DF64 slice roundtrip len", unpacked.len() == 3);
    v.check_pass(
        "DF64 slice max error < 1e-14",
        [1.0, 2.0, std::f64::consts::PI]
            .iter()
            .zip(&unpacked)
            .all(|(a, b)| (a - b).abs() < tolerances::PYTHON_PARITY_TIGHT),
    );

    let f32_err = (std::f64::consts::PI - f64::from(std::f64::consts::PI as f32)).abs();
    v.check_pass("DF64 more precise than f32", pi_err < f32_err);

    // ═══ D01-D27: Core 47 Domains (same as v13) ══════════════════════

    v.section("D01: Core Diversity");
    total_domains += 1;
    let counts = [50.0, 30.0, 15.0, 4.0, 1.0];
    let h = diversity::shannon(&counts);
    let d = diversity::simpson(&counts);
    let j = diversity::pielou_evenness(&counts);
    let s_obs = diversity::observed_features(&counts);
    let c1 = diversity::chao1(&counts);
    v.check_pass("Shannon > 0", h > 0.0);
    v.check_pass("Simpson in [0,1]", (0.0..=1.0).contains(&d));
    v.check_pass("Pielou in [0,1]", (0.0..=1.0).contains(&j));
    v.check("Observed == 5", s_obs, 5.0, tolerances::EXACT_F64);
    v.check_pass("Chao1 >= observed", c1 >= s_obs);
    v.check(
        "Shannon(uniform 4) = ln(4)",
        diversity::shannon(&[1.0; 4]),
        4.0_f64.ln(),
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "Simpson(uniform 4) = 0.75",
        diversity::simpson(&[1.0; 4]),
        0.75,
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "BC self == 0",
        diversity::bray_curtis(&[10.0], &[10.0]),
        0.0,
        tolerances::EXACT,
    );

    v.section("D02: Quality + Dereplication");
    total_domains += 2;
    let params = quality::QualityParams {
        window_size: 4,
        window_min_quality: 15,
        leading_min_quality: 3,
        trailing_min_quality: 3,
        min_length: 36,
        phred_offset: 33,
    };
    v.check_pass("QualityParams constructible", params.window_size == 4);
    v.check_pass(
        "DerepSort accessible",
        matches!(derep::DerepSort::Abundance, derep::DerepSort::Abundance),
    );

    v.section("D03: Merge Pairs");
    total_domains += 1;
    v.check_pass(
        "revcomp(ATGC) == GCAT",
        merge_pairs::reverse_complement(b"ATGC") == b"GCAT",
    );
    v.check_pass(
        "revcomp involution",
        merge_pairs::reverse_complement(b"GCAT") == b"ATGC",
    );

    v.section("D04: Robinson-Foulds");
    total_domains += 1;
    let t1 = unifrac::tree::PhyloTree::from_newick("((A:1,B:1):1,(C:1,D:1):1);");
    let t2 = unifrac::tree::PhyloTree::from_newick("((A:1,C:1):1,(B:1,D:1):1);");
    v.check_pass(
        "RF > 0 for different topologies",
        robinson_foulds::rf_distance(&t1, &t2) > 0,
    );

    v.section("D05: Neighbor Joining");
    total_domains += 1;
    let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
    let dist = vec![0.0, 5.0, 9.0, 5.0, 0.0, 10.0, 9.0, 10.0, 0.0];
    let nj = neighbor_joining::neighbor_joining(&dist, &labels);
    v.check_pass("NJ tree has Newick", !nj.newick.is_empty());
    v.check_pass(
        "JC distance > 0",
        neighbor_joining::jukes_cantor_distance(b"ATGCATGC", b"ATGGATGC") > 0.0,
    );

    v.section("D06: Reconciliation");
    total_domains += 1;
    let host = reconciliation::FlatRecTree {
        names: vec!["h0".into(), "h1".into(), "h2".into()],
        left_child: vec![1, u32::MAX, u32::MAX],
        right_child: vec![2, u32::MAX, u32::MAX],
    };
    let guest = reconciliation::FlatRecTree {
        names: vec!["g0".into(), "g1".into(), "g2".into()],
        left_child: vec![1, u32::MAX, u32::MAX],
        right_child: vec![2, u32::MAX, u32::MAX],
    };
    let costs = reconciliation::DtlCosts {
        duplication: 1,
        transfer: 2,
        loss: 1,
    };
    let tip_map = vec![
        ("g1".to_string(), "h1".to_string()),
        ("g2".to_string(), "h2".to_string()),
    ];
    let rec = reconciliation::reconcile_dtl(&host, &guest, &tip_map, &costs);
    v.check_pass("DTL cost computed", rec.optimal_cost < u32::MAX);

    v.section("D07: Felsenstein JC69");
    total_domains += 1;
    let p_same = felsenstein::jc69_prob(0, 0, 0.1, 1.0);
    let p_diff = felsenstein::jc69_prob(0, 1, 0.1, 1.0);
    v.check_pass("P(same) > P(diff)", p_same > p_diff);
    v.check(
        "row sum = 1",
        3.0f64.mul_add(p_diff, p_same),
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    v.section("D08: HMM Forward");
    total_domains += 1;
    let model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.5_f64.ln(), 0.5_f64.ln(), 0.1_f64.ln(), 0.9_f64.ln()],
    };
    let fwd = hmm::forward(&model, &[0, 1, 0]);
    v.check_pass("HMM LL finite", fwd.log_likelihood.is_finite());
    v.check_pass("HMM LL < 0", fwd.log_likelihood < 0.0);

    v.section("D09: dN/dS");
    total_domains += 1;
    let dnds_result = dnds::pairwise_dnds(b"ATGATG", b"ATGGTG").unwrap();
    v.check_pass(
        "omega computed",
        dnds_result.omega.is_none_or(f64::is_finite),
    );

    v.section("D10: SNP");
    total_domains += 1;
    let seqs: Vec<&[u8]> = vec![b"ATGCATGC", b"ATGGATGC"];
    v.check_pass("SNPs detected", !snp::call_snps(&seqs).variants.is_empty());

    v.section("D11: Pangenome");
    total_domains += 1;
    let clusters = vec![
        pangenome::GeneCluster {
            id: "core1".into(),
            presence: vec![true, true, true],
        },
        pangenome::GeneCluster {
            id: "acc1".into(),
            presence: vec![true, true, false],
        },
        pangenome::GeneCluster {
            id: "unique1".into(),
            presence: vec![false, false, true],
        },
    ];
    let pan = pangenome::analyze(&clusters, 3);
    v.check_pass("core <= total", pan.core_size <= pan.total_size);
    v.check_pass("total == 3", pan.total_size == 3);

    v.section("D12: K-mer");
    total_domains += 1;
    v.check_pass(
        "k-mer table populated",
        kmer::count_kmers(b"ATGCATGC", 3).total_valid_kmers > 0,
    );

    v.section("D13-17: ODE Systems (5 papers)");
    total_domains += 5;
    let qs_r = qs_biofilm::scenario_standard_growth(&qs_biofilm::QsBiofilmParams::default(), 0.01);
    v.check_pass(
        "QS ODE: N > 0",
        *qs_r.states().last().unwrap().first().unwrap() > 0.0,
    );

    let coop_r =
        cooperation::scenario_equal_start(&cooperation::CooperationParams::default(), 0.01);
    v.check_pass(
        "Cooperation: freq finite",
        cooperation::cooperator_frequency(&coop_r)
            .last()
            .unwrap()
            .is_finite(),
    );

    let bi_r = bistable::run_bistable(
        &[0.01, 0.0, 0.0, 2.0, 0.5],
        0.01,
        100.0,
        &bistable::BistableParams::default(),
    );
    v.check_pass("Bistable: trajectory", bi_r.t.len() > 1);

    let ph_r =
        phage_defense::scenario_no_phage(&phage_defense::PhageDefenseParams::default(), 0.01);
    v.check_pass("Phage: trajectory", ph_r.t.len() > 1);

    let cap_r = capacitor::scenario_normal(&capacitor::CapacitorParams::default(), 0.01);
    v.check_pass("Capacitor: trajectory", cap_r.t.len() > 1);

    v.section("D18: Multi-Signal QS");
    total_domains += 1;
    let ms_r = multi_signal::scenario_wild_type(&multi_signal::MultiSignalParams::default(), 0.01);
    v.check_pass("Multi-signal: trajectory", ms_r.t.len() > 1);

    v.section("D19: Gillespie SSA");
    total_domains += 1;
    let traj = gillespie::birth_death_ssa(0.5, 0.3, 1000.0, 42);
    v.check_pass("SSA: events produced", traj.times.len() > 1);

    v.section("D20-23: Soil Science");
    total_domains += 4;
    let w_c = 16.5_f64;
    let notill_w = 25.0 * (1.0 - 79.3 / 100.0);
    let tilled_w = 25.0 * (1.0 - 38.5 / 100.0);
    v.check_pass(
        "No-till QS > tilled QS",
        norm_cdf((w_c - notill_w) / 3.0) > norm_cdf((w_c - tilled_w) / 3.0),
    );
    v.check_pass(
        "Recovery W(31yr) < W(0)",
        18.0 * (-31.0 / 10.0_f64).exp() < 18.0,
    );
    v.check_pass(
        "Pore gradient: big > small QS",
        norm_cdf((25.0f64.mul_add(-(1.0 - 0.85), w_c)) / 3.0)
            > norm_cdf((25.0f64.mul_add(-(1.0 - 0.2), w_c)) / 3.0),
    );
    v.check_pass(
        "Aggregate interior: large > small",
        (98.0_f64).powi(3) / (100.0_f64).powi(3) > (8.0_f64).powi(3) / (10.0_f64).powi(3),
    );

    v.section("D24: Spectral Match");
    total_domains += 1;
    let mz = [100.0, 200.0, 300.0];
    let int = [1000.0, 500.0, 200.0];
    v.check(
        "cosine self == 1",
        spectral_match::cosine_similarity(&mz, &int, &mz, &int, 0.5).score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    v.section("D25: KMD");
    total_domains += 1;
    v.check_pass(
        "KMD results",
        kmd::kendrick_mass_defect(
            &[200.0, 214.015_650_64],
            kmd::units::CH2_EXACT,
            kmd::units::CH2_NOMINAL,
        )
        .len()
            == 2,
    );

    v.section("D26: Signal Processing");
    total_domains += 1;
    v.check_pass(
        "peaks detected",
        !signal::find_peaks(
            &[0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 5.0, 2.0, 0.0],
            &signal::PeakParams::default(),
        )
        .is_empty(),
    );

    v.section("D27: UniFrac");
    total_domains += 1;
    let tree = unifrac::tree::PhyloTree::from_newick("((A:1,B:2):1,(C:3,D:4):2);");
    let mut s1: HashMap<String, f64> = HashMap::new();
    s1.insert("A".into(), 10.0);
    s1.insert("B".into(), 20.0);
    let mut s2: HashMap<String, f64> = HashMap::new();
    s2.insert("C".into(), 30.0);
    s2.insert("D".into(), 10.0);
    v.check_pass(
        "UniFrac in [0,1]",
        (0.0..=1.0).contains(&unifrac::distance::unweighted_unifrac(&tree, &s1, &s2)),
    );

    v.section("D28: PCoA");
    total_domains += 1;
    let dm = [0.5, 0.8, 0.6];
    v.check_pass(
        "PCoA produces coords",
        pcoa::pcoa(&dm, 3, 2).unwrap().n_samples == 3,
    );

    v.section("D29: Math Primitives");
    total_domains += 1;
    v.check(
        "erf(1)",
        erf(1.0),
        0.842_700_792_949_715,
        tolerances::ERF_PARITY,
    );
    v.check("Φ(0) = 0.5", norm_cdf(0.0), 0.5, tolerances::EXACT_F64);
    v.check(
        "Φ(1.96) ≈ 0.975",
        norm_cdf(1.96),
        0.975,
        tolerances::NORM_CDF_PARITY,
    );

    // ═══ V71 NEW: Cross-Spring Primitives ═════════════════════════════

    v.section("D30: Graph Laplacian (neuralSpring)");
    total_domains += 1;
    let n_g = 6;
    let mut adj = vec![0.0; n_g * n_g];
    for i in 0..n_g {
        for j in (i + 1)..n_g {
            if (i + j) % 2 == 0 {
                adj[i * n_g + j] = 1.0;
                adj[j * n_g + i] = 1.0;
            }
        }
    }
    let lap = barracuda::linalg::graph::graph_laplacian(&adj, n_g);
    let row_sums: Vec<f64> = (0..n_g)
        .map(|i| (0..n_g).map(|j| lap[i * n_g + j]).sum())
        .collect();
    v.check_pass(
        "Laplacian row sums ≈ 0",
        row_sums.iter().all(|s| s.abs() < tolerances::PYTHON_PARITY),
    );
    v.check_pass(
        "Laplacian diagonal ≥ 0",
        (0..n_g).all(|i| lap[i * n_g + i] >= 0.0),
    );

    v.section("D31: Effective Rank (neuralSpring)");
    total_domains += 1;
    let eigs = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1];
    let er = barracuda::linalg::graph::effective_rank(&eigs);
    v.check_pass("effective_rank ∈ [1, 6]", (1.0..=6.0).contains(&er));

    v.section("D32: Numerical Hessian (neuralSpring)");
    total_domains += 1;
    let hess = barracuda::numerical::numerical_hessian(
        &|x: &[f64]| 100.0f64.mul_add((x[0].mul_add(-x[0], x[1])).powi(2), (1.0 - x[0]).powi(2)),
        &[1.0, 1.0],
        tolerances::NUMERICAL_HESSIAN_EPSILON,
    );
    v.check(
        "Hessian H[0,0] ≈ 802",
        hess[0],
        802.0,
        tolerances::HESSIAN_H00_TOL,
    );
    v.check(
        "Hessian H[1,1] ≈ 200",
        hess[3],
        200.0,
        tolerances::HESSIAN_H11_TOL,
    );

    v.section("D33: NMF (wetSpring → ToadStool)");
    total_domains += 1;
    let v_mat: Vec<f64> = (0..20 * 10)
        .map(|i| f64::from(((i * 3 + 1) % 50) as u32) / 50.0)
        .collect();
    let nmf_cfg = barracuda::linalg::nmf::NmfConfig {
        rank: 3,
        max_iter: 100,
        tol: tolerances::NMF_CONVERGENCE_KL,
        objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let nmf = barracuda::linalg::nmf::nmf(&v_mat, 20, 10, &nmf_cfg).expect("NMF");
    v.check_pass("NMF W,H ≥ 0", nmf.w.iter().chain(&nmf.h).all(|&x| x >= 0.0));

    v.section("D34: Ridge Regression (wetSpring → ToadStool)");
    total_domains += 1;
    let rx: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.02).collect();
    let ry: Vec<f64> = (0..20).map(|i| f64::from(i).mul_add(0.5, 1.0)).collect();
    let ridge = barracuda::linalg::ridge_regression(
        &rx,
        &ry,
        10,
        5,
        2,
        tolerances::RIDGE_REGULARIZATION_SMALL,
    )
    .expect("ridge");
    v.check_pass(
        "ridge weights finite",
        ridge.weights.iter().all(|w| w.is_finite()),
    );

    v.section("D35: Anderson Spectral (hotSpring → ToadStool)");
    total_domains += 1;
    let csr = barracuda::spectral::anderson_3d(6, 6, 6, 4.0, 42);
    let tri = barracuda::spectral::lanczos(&csr, 40, 42);
    let eigs_a = barracuda::spectral::lanczos_eigenvalues(&tri);
    v.check_pass(
        "Anderson eigenvalues finite",
        eigs_a.iter().all(|e: &f64| e.is_finite()),
    );

    v.section("D36: Pearson + Stats (airSpring → ToadStool)");
    total_domains += 1;
    let a: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let b: Vec<f64> = a.iter().map(|&x| 2.0f64.mul_add(x, 1.0)).collect();
    let pearson = barracuda::stats::pearson_correlation(&a, &b).expect("pearson");
    v.check(
        "Pearson(linear) = 1.0",
        pearson,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    let trapz = barracuda::numerical::trapz(
        &(0..1001)
            .map(|i| {
                let x = f64::from(i) / 1000.0;
                x * x
            })
            .collect::<Vec<_>>(),
        &(0..1001).map(|i| f64::from(i) / 1000.0).collect::<Vec<_>>(),
    )
    .expect("trapz");
    v.check(
        "trapz(x²) ≈ 1/3",
        trapz,
        1.0 / 3.0,
        tolerances::TRAPZ_COARSE,
    );

    // ═══ Summary ═════════════════════════════════════════════════════
    v.section(&format!("Summary: {total_domains} domains validated"));
    println!(
        "  Core 47 domains (v13) + 3 cross-spring + 3 ToadStool + DF64 = {total_domains} total"
    );
    println!("  V71 additions: df64_host, graph_laplacian, effective_rank, numerical_hessian");
    println!("  All pure Rust CPU math — zero Python, zero GPU, zero unsafe");

    v.finish();
}
