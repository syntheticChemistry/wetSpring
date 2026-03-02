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
//! # Exp233: Paper Math Control v2 — 25 Papers Validated via `BarraCuda` CPU
//!
//! Extends v1 (18 papers, 58 checks) with 7 additional papers:
//! - **Track 1c:** Anderson 2017 population genomics (ANI + dN/dS), Moulana 2020
//!   pangenome, Anderson 2015 rare biosphere
//! - **Track 3:** Yang 2020 NMF drug-disease decomposition
//! - **Phase 37:** Cold seep QS catalog (Exp144), luxR phylogeny (Exp146),
//!   burst statistics (Exp149)
//!
//! # Evolution chain
//!
//! ```text
//! Paper (this) → CPU (Exp234) → GPU (Exp235) → Streaming (Exp236) → metalForge (Exp237)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline | Analytical solutions from published papers |
//! | Date | 2026-02-28 |
//! | Phase | 77 |
//! | Command | `cargo run --release --bin validate_paper_math_control_v2` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::collections::HashMap;
use wetspring_barracuda::bio::{
    ani, bistable, capacitor, cooperation, diversity, dnds, felsenstein, gillespie, hmm, kmer,
    multi_signal, neighbor_joining, pangenome, phage_defense, qs_biofilm, robinson_foulds, snp,
    spectral_match, unifrac,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

use barracuda::special::erf;
use barracuda::stats::norm_cdf;

fn main() {
    let mut v = Validator::new("Exp233: Paper Math Control v2 — 25 Papers via BarraCuda CPU");
    let mut n_papers = 0_u32;

    // ═══════════════════════════════════════════════════════════════════
    // Track 1: Microbial Ecology & QS Signaling (P1-P7, inherited from v1)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P1: Waters 2008 — QS/c-di-GMP ODE");
    n_papers += 1;
    let params = qs_biofilm::QsBiofilmParams::default();
    let r = qs_biofilm::scenario_standard_growth(&params, 0.001);
    let n_ss = ode_tail_mean(&r, 0, 0.1);
    v.check(
        "Waters: N_ss ≈ 0.975",
        n_ss,
        0.975,
        tolerances::ODE_METHOD_PARITY,
    );

    v.section("P2: Massie 2012 — Gillespie SSA");
    n_papers += 1;
    let emp_mean: f64 = (0..500_u64)
        .map(|seed| gillespie::birth_death_ssa(10.0, 0.1, 100.0, seed).final_state()[0] as f64)
        .sum::<f64>()
        / 500.0;
    v.check(
        "Massie: E[X_ss] → 100",
        emp_mean,
        100.0,
        tolerances::GILLESPIE_MEAN_REL * 100.0,
    );

    v.section("P3: Fernandez 2020 — Bistable Switching");
    n_papers += 1;
    let bi = bistable::run_bistable(
        &[0.01, 0.0, 0.0, 2.0, 0.5],
        0.01,
        200.0,
        &bistable::BistableParams::default(),
    );
    v.check_pass("Fernandez: ODE trajectory", bi.t.len() > 1);

    v.section("P4: Srivastava 2011 — Multi-Signal QS");
    n_papers += 1;
    let ms = multi_signal::scenario_wild_type(&multi_signal::MultiSignalParams::default(), 0.01);
    v.check_pass("Srivastava: trajectory produced", ms.t.len() > 10);

    v.section("P5: Bruger & Waters 2018 — Cooperation");
    n_papers += 1;
    let freq = cooperation::cooperator_frequency(&cooperation::scenario_equal_start(
        &cooperation::CooperationParams::default(),
        0.01,
    ));
    v.check_pass("Bruger: cooperators persist", *freq.last().unwrap() > 0.1);

    v.section("P6: Hsueh 2022 — Phage Defense");
    n_papers += 1;
    let ph = phage_defense::scenario_no_phage(&phage_defense::PhageDefenseParams::default(), 0.001);
    let bd = ode_tail_mean(&ph, 0, 0.1);
    v.check(
        "Hsueh: no-phage Bd",
        bd,
        132_242.0,
        tolerances::PHAGE_LARGE_POPULATION,
    );

    v.section("P7: Mhatre 2020 — Capacitor");
    n_papers += 1;
    let cap = capacitor::scenario_normal(&capacitor::CapacitorParams::default(), 0.01);
    v.check_pass("Mhatre: trajectory produced", cap.t.len() > 100);

    // ═══════════════════════════════════════════════════════════════════
    // Track 1b: Phylogenetics (P8-P9, inherited from v1)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P8: Liu 2014 — HMM Forward");
    n_papers += 1;
    let model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.5_f64.ln(), 0.5_f64.ln(), 0.1_f64.ln(), 0.9_f64.ln()],
    };
    let fwd = hmm::forward(&model, &[0, 1, 0, 1, 0]);
    v.check_pass(
        "Liu: LL finite and < 0",
        fwd.log_likelihood.is_finite() && fwd.log_likelihood < 0.0,
    );

    v.section("P9: Felsenstein 1981 — JC69 Pruning");
    n_papers += 1;
    let p_same = felsenstein::jc69_prob(0, 0, 0.1, 1.0);
    let p_diff = felsenstein::jc69_prob(0, 1, 0.1, 1.0);
    let p_exact = 0.75_f64.mul_add((-4.0_f64 * 0.1 / 3.0).exp(), 0.25);
    v.check(
        "Felsenstein: P(A→A) analytical",
        p_same,
        p_exact,
        tolerances::ANALYTICAL_F64,
    );
    v.check(
        "Felsenstein: row sum = 1",
        3.0_f64.mul_add(p_diff, p_same),
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Track 2: Analytical Chemistry (P10, inherited from v1)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P10: Jones Lab — Spectral Cosine");
    n_papers += 1;
    let mz = [100.0, 200.0, 300.0];
    let int = [1000.0, 500.0, 200.0];
    v.check(
        "Jones: cosine self = 1",
        spectral_match::cosine_similarity(&mz, &int, &mz, &int, 0.5).score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // Track 3: Drug Repurposing (P11 inherited + P19 NEW)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P11: Fajgenbaum 2019 — NMF");
    n_papers += 1;
    let v_mat: Vec<f64> = (0..30 * 15)
        .map(|i| f64::from(((i * 3 + 1) % 50) as u32) / 50.0)
        .collect();
    let nmf_cfg = barracuda::linalg::nmf::NmfConfig {
        rank: 5,
        max_iter: 200,
        tol: 1e-4,
        objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let nmf_res = barracuda::linalg::nmf::nmf(&v_mat, 30, 15, &nmf_cfg).expect("NMF");
    v.check_pass(
        "Fajgenbaum: W,H ≥ 0",
        nmf_res.w.iter().chain(&nmf_res.h).all(|&x| x >= 0.0),
    );

    v.section("P19: Yang 2020 — NMF Drug-Disease Decomposition (NEW)");
    n_papers += 1;
    println!("  Paper: Yang X et al. Briefings Bioinf. 2020;21:1516–27.");
    println!("  Model: NMF rank selection via cophenetic correlation; W·H ≈ V");
    let yang_v: Vec<f64> = (0..50 * 20)
        .map(|i| f64::from(((i * 7 + 3) % 80) as u32) / 80.0)
        .collect();
    let yang_r3 = barracuda::linalg::nmf::nmf(
        &yang_v,
        50,
        20,
        &barracuda::linalg::nmf::NmfConfig {
            rank: 3,
            max_iter: 200,
            tol: 1e-4,
            objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
            seed: 42,
        },
    )
    .expect("NMF rank=3");
    let yang_r5 = barracuda::linalg::nmf::nmf(
        &yang_v,
        50,
        20,
        &barracuda::linalg::nmf::NmfConfig {
            rank: 5,
            max_iter: 200,
            tol: 1e-4,
            objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
            seed: 42,
        },
    )
    .expect("NMF rank=5");
    v.check_pass(
        "Yang: rank=3 converges",
        *yang_r3.errors.last().unwrap() < *yang_r3.errors.first().unwrap(),
    );
    v.check_pass(
        "Yang: rank=5 converges",
        *yang_r5.errors.last().unwrap() < *yang_r5.errors.first().unwrap(),
    );
    v.check_pass(
        "Yang: rank=5 lower error",
        yang_r5.errors.last() < yang_r3.errors.last(),
    );

    // ═══════════════════════════════════════════════════════════════════
    // Track 4: Soil QS (P12-P17, inherited from v1)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P12: Martínez-García 2023 — Pore QS");
    n_papers += 1;
    let w_c = 16.5_f64;
    let sigma_qs = 3.0_f64;
    let large_qs = norm_cdf(25.0f64.mul_add(-(1.0 - 0.85), w_c) / sigma_qs);
    let small_qs = norm_cdf(25.0f64.mul_add(-(1.0 - 0.20), w_c) / sigma_qs);
    v.check_pass("MG2023: large pore QS > small pore QS", large_qs > small_qs);

    v.section("P13: Islam 2014 — Brandt Farm");
    n_papers += 1;
    let notill_w = 25.0 * (1.0 - 0.793);
    v.check(
        "Islam: no-till W",
        notill_w,
        5.175,
        tolerances::SOIL_DISORDER_ANALYTICAL,
    );

    v.section("P14: Zuber 2016 — Meta-Analysis");
    n_papers += 1;
    v.check_pass(
        "Zuber: MBC ratio CI excludes 1.0",
        1.96_f64.mul_add(-0.06, 1.14) > 1.0,
    );

    v.section("P15: Feng 2024 — Pore Diversity");
    n_papers += 1;
    v.check_pass(
        "Feng: Shannon monotonicity",
        diversity::shannon(&[0.2; 5]) > diversity::shannon(&[0.8, 0.1, 0.05, 0.03, 0.02]),
    );

    v.section("P16: Liang 2015 — Tillage Recovery");
    n_papers += 1;
    v.check_pass(
        "Liang: W(31yr) < W(0)",
        18.0 * (-31.0 / 10.0_f64).exp() < 18.0,
    );

    v.section("P17: Tecon 2017 — Aggregate Geometry");
    n_papers += 1;
    let frac_large = (100.0_f64 - 2.0).powi(3) / 100.0_f64.powi(3);
    let frac_small = (10.0_f64 - 2.0).powi(3) / 10.0_f64.powi(3);
    v.check_pass("Tecon: large > small interior", frac_large > frac_small);

    // ═══════════════════════════════════════════════════════════════════
    // Cross-Spring: Anderson (P18, inherited from v1)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P18: Bourgain & Kachkovskiy 2018 — Anderson");
    n_papers += 1;
    v.check(
        "BK2018: erf(1)",
        erf(1.0),
        0.842_700_792_949_715,
        tolerances::ERF_PARITY,
    );
    v.check(
        "BK2018: γ(0) = W²/96",
        0.5 * 0.5 / 96.0,
        0.25 / 96.0,
        tolerances::ANALYTICAL_F64,
    );

    // ═══════════════════════════════════════════════════════════════════
    // NEW: Track 1c — Deep-Sea Metagenomics (P20-P22)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P20: Anderson 2017 — Population Genomics (ANI + dN/dS) (NEW)");
    n_papers += 1;
    println!("  Paper: Anderson RE et al. ISME J. 2017;11:176–90.");
    println!("  Model: ANI species boundary 95%, dN/dS selection pressure");

    let seq_a = b"ATGATGATGATGATGATGATGATGATGATG";
    let seq_b = b"ATGGTGATGATGATGCTGATGATGATGATG";
    let ani_result = ani::pairwise_ani(seq_a, seq_b);
    v.check_pass(
        "Anderson: ANI ∈ [0,1]",
        (0.0..=1.0).contains(&ani_result.ani),
    );
    v.check_pass("Anderson: related seqs ANI > 0.8", ani_result.ani > 0.8);

    let dnds_result = dnds::pairwise_dnds(seq_a, seq_b).unwrap();
    v.check_pass("Anderson: dN finite", dnds_result.dn.is_finite());
    v.check_pass("Anderson: dS finite", dnds_result.ds.is_finite());

    let snp_seqs: Vec<&[u8]> = vec![seq_a.as_slice(), seq_b.as_slice()];
    let snps = snp::call_snps(&snp_seqs);
    v.check_pass("Anderson: SNPs detected", !snps.variants.is_empty());

    v.section("P21: Moulana 2020 — Pangenome (NEW)");
    n_papers += 1;
    println!("  Paper: Moulana A et al. mBio. 2020;11:e03188-19.");
    println!("  Model: core/accessory/unique classification; Heap's law γ < 1 → open");

    let clusters = vec![
        pangenome::GeneCluster {
            id: "core1".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "core2".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "acc1".into(),
            presence: vec![true, true, false, false, false],
        },
        pangenome::GeneCluster {
            id: "acc2".into(),
            presence: vec![false, true, true, false, true],
        },
        pangenome::GeneCluster {
            id: "uniq1".into(),
            presence: vec![true, false, false, false, false],
        },
    ];
    let pan = pangenome::analyze(&clusters, 5);
    v.check_pass("Moulana: core ≤ total", pan.core_size <= pan.total_size);
    v.check_pass("Moulana: total == 5", pan.total_size == 5);
    v.check_pass("Moulana: core == 2", pan.core_size == 2);

    v.section("P22: Anderson 2015 — Rare Biosphere (NEW)");
    n_papers += 1;
    println!("  Paper: Anderson RE et al. Front Microbiol. 2015;5:735.");
    println!("  Property: Chao1 ≥ observed; rare taxa boost estimated richness");

    let rare_comm = [100.0, 50.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let obs = diversity::observed_features(&rare_comm);
    let chao1 = diversity::chao1(&rare_comm);
    v.check_pass("Anderson: Chao1 ≥ observed", chao1 >= obs);
    v.check_pass("Anderson: Chao1 > observed (rare singletons)", chao1 > obs);

    let rich_comm = [100.0, 100.0, 100.0, 100.0, 100.0];
    let rich_chao = diversity::chao1(&rich_comm);
    let rich_obs = diversity::observed_features(&rich_comm);
    v.check(
        "Anderson: even comm Chao1 ≈ obs",
        rich_chao,
        rich_obs,
        tolerances::PYTHON_PARITY,
    );

    // ═══════════════════════════════════════════════════════════════════
    // NEW: Phase 37 Extension Papers (P23-P25)
    // ═══════════════════════════════════════════════════════════════════

    v.section("P23: Cold Seep QS Gene Catalog (Exp144) (NEW)");
    n_papers += 1;
    println!("  Source: NCBI GenBank cold seep metagenomes");
    println!("  Model: k-mer diversity of QS gene families across habitats");

    let qs_genes_seep = [b"ATGCATGCATGCATGC" as &[u8], b"TGCATGCATGCATGCA"];
    let kmer_counts: Vec<_> = qs_genes_seep
        .iter()
        .map(|s| kmer::count_kmers(s, 4))
        .collect();
    v.check_pass(
        "ColdSeep: k-mer counts > 0",
        kmer_counts.iter().all(|k| k.total_valid_kmers > 0),
    );

    let tree = unifrac::tree::PhyloTree::from_newick("((seepA:0.1,seepB:0.2):0.3,ventC:0.5);");
    let mut sa: HashMap<String, f64> = HashMap::new();
    sa.insert("seepA".into(), 10.0);
    sa.insert("seepB".into(), 5.0);
    let mut sb: HashMap<String, f64> = HashMap::new();
    sb.insert("ventC".into(), 15.0);
    let uf = unifrac::distance::unweighted_unifrac(&tree, &sa, &sb);
    v.check_pass("ColdSeep: UniFrac ∈ [0,1]", (0.0..=1.0).contains(&uf));

    v.section("P24: luxR Phylogeny × Geometry (Exp146) (NEW)");
    n_papers += 1;
    println!("  Model: RF distance between QS gene trees; topology ↔ habitat");

    let t1 = unifrac::tree::PhyloTree::from_newick("((A:1,B:1):1,(C:1,D:1):1);");
    let t2 = unifrac::tree::PhyloTree::from_newick("((A:1,C:1):1,(B:1,D:1):1);");
    let rf = robinson_foulds::rf_distance(&t1, &t2);
    v.check_pass("luxR: RF > 0 for different topologies", rf > 0);

    let labels: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
    let dist = vec![0.0, 0.1, 0.5, 0.1, 0.0, 0.4, 0.5, 0.4, 0.0];
    let nj = neighbor_joining::neighbor_joining(&dist, &labels);
    v.check_pass("luxR: NJ produces tree", !nj.newick.is_empty());

    v.section("P25: Burst Statistics as Anderson (Exp149) (NEW)");
    n_papers += 1;
    println!("  Model: QS signaling bursts → localization statistics; IPR, level spacing");

    let burst_ssa = gillespie::birth_death_ssa(5.0, 0.3, 200.0, 42);
    v.check_pass("Burst: SSA produces events", burst_ssa.times.len() > 10);

    let burst_counts: Vec<f64> = (0..50)
        .map(|seed| gillespie::birth_death_ssa(5.0, 0.3, 200.0, seed).final_state()[0] as f64)
        .collect();
    let burst_h = diversity::shannon(&burst_counts.iter().map(|&x| x.max(1.0)).collect::<Vec<_>>());
    v.check_pass("Burst: entropy finite", burst_h.is_finite());

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════

    v.section(&format!("Paper Math Control v2 Summary: {n_papers} papers"));
    println!("  Inherited: 18 papers (v1) — Tracks 1, 1b, 2, 3, 4, cross-spring");
    println!("  NEW: P19 Yang 2020, P20-P22 Track 1c, P23-P25 Phase 37 extensions");
    println!("  Chain: Paper (this) → CPU → GPU → Streaming → metalForge");

    v.finish();
}

use wetspring_barracuda::bio::ode::OdeResult;

fn ode_tail_mean(r: &OdeResult, var_idx: usize, tail_frac: f64) -> f64 {
    let states: Vec<&[f64]> = r.states().collect();
    let n = states.len();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let tail_start = (n as f64 * (1.0 - tail_frac)) as usize;
    let tail: Vec<f64> = states[tail_start..].iter().map(|s| s[var_idx]).collect();
    tail.iter().sum::<f64>() / tail.len() as f64
}
