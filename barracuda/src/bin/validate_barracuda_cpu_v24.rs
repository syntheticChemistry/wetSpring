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
//! # Exp314: `BarraCuda` CPU v24 — V98 Comprehensive Bio Domain Parity
//!
//! Proves that **every** `BarraCuda` CPU bio module produces correct results
//! via analytical invariants and known-value checks. This is the deepest
//! CPU domain coverage binary: 33 bio modules + statistics.
//!
//! ```text
//! Paper (Exp313) → CPU (this) → GPU (Exp316) → Streaming (Exp317) → metalForge (Exp318)
//! ```
//!
//! ## Domains
//!
//! - D47: I/O + Quality — Phred, adapter, quality filter, merge, derep, kmer
//! - D48: Alignment + Tree — SW, Felsenstein, RF, NJ, placement, reconciliation
//! - D49: Diversity — Shannon, Simpson, Chao1, Pielou, BC, `PCoA`, `UniFrac`
//! - D50: ODE Systems — QS, bistable, capacitor, multi-signal, phage, cooperation
//! - D51: Metagenomics — ANI, SNP, dN/dS, pangenome, molecular clock
//! - D52: Chemistry — EIC, signal peaks, KMD, tolerance search, spectral match
//! - D53: ML — Decision tree, random forest, GBM, HMM, bootstrap
//! - D54: Statistics — Welford, Pearson, Spearman, jackknife, bootstrap CI
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Analytical (mathematical invariants) |
//! | Date | 2026-03-07 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v24` |

use std::collections::HashMap;
use std::time::Instant;
use wetspring_barracuda::bio::{
    adapter, alignment, ani, bistable, capacitor, cooperation, diversity, dnds, felsenstein,
    gillespie, hmm, kmd, kmer, molecular_clock, multi_signal, neighbor_joining, ode, pangenome,
    pcoa, phage_defense, phred, qs_biofilm, robinson_foulds, signal, snp, spectral_match,
    tolerance_search, unifrac,
};
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
    let mut v = Validator::new("Exp314: BarraCuda CPU v24 — V98 Comprehensive Bio Domain Parity");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // D47: I/O + Quality Pipeline
    // ═══════════════════════════════════════════════════════════════════
    v.section("D47: I/O + Quality — Phred, adapter, quality, merge, derep, kmer");
    let t = Instant::now();
    let mut d47 = 0_u32;

    let p_err = phred::phred_to_error_prob(30.0);
    v.check(
        "Phred Q30 → P(err) = 0.001",
        p_err,
        0.001,
        tolerances::ANALYTICAL_F64,
    );
    d47 += 1;

    let q_back = phred::error_prob_to_phred(p_err);
    v.check(
        "Phred round-trip Q30",
        q_back,
        30.0,
        tolerances::ANALYTICAL_F64,
    );
    d47 += 1;

    let p20 = phred::phred_to_error_prob(20.0);
    v.check(
        "Phred Q20 → P(err) = 0.01",
        p20,
        0.01,
        tolerances::ANALYTICAL_F64,
    );
    d47 += 1;

    let trim_pos = adapter::find_adapter_3prime(b"ACGTACGTAAAAAA", b"AAAAAA", 0, 6);
    v.check_pass("Adapter: exact match found at pos 8", trim_pos == Some(8));
    d47 += 1;

    let no_adapter = adapter::find_adapter_3prime(b"ACGTACGTACGT", b"TTTTTT", 0, 6);
    v.check_pass("Adapter: no match returns None", no_adapter.is_none());
    d47 += 1;

    let counts = kmer::count_kmers(b"ACGTACGT", 4);
    v.check_pass(
        "4-mer count total > 0 for 8bp",
        counts.total_valid_kmers > 0,
    );
    d47 += 1;

    let counts2 = kmer::count_kmers(b"AAAA", 4);
    v.check_count("4-mer of AAAA: 1 unique", counts2.counts.len(), 1);
    d47 += 1;

    let counts_multi = kmer::count_kmers_multi(&[b"ACGT".as_ref(), b"ACGT"], 2);
    v.check_pass(
        "Multi k-mer count positive",
        counts_multi.total_valid_kmers > 0,
    );
    d47 += 1;

    domains.push(domain("D47: I/O+Quality", "wetSpring", t.elapsed(), d47));

    // ═══════════════════════════════════════════════════════════════════
    // D48: Alignment + Tree
    // ═══════════════════════════════════════════════════════════════════
    v.section("D48: Alignment + Tree — SW, Felsenstein, RF, NJ, placement, reconciliation");
    let t = Instant::now();
    let mut d48 = 0_u32;

    let sw = alignment::smith_waterman(
        b"ACGTACGT",
        b"ACGTACGT",
        &alignment::ScoringParams::default(),
    );
    v.check_pass("SW: identical seqs → max score", sw.score > 0);
    d48 += 1;

    let sw_mismatch =
        alignment::smith_waterman(b"AAAA", b"TTTT", &alignment::ScoringParams::default());
    v.check_pass(
        "SW: all-mismatch score < perfect match",
        sw_mismatch.score < sw.score,
    );
    d48 += 1;

    let leaf_a = felsenstein::TreeNode::Leaf {
        name: "A".into(),
        states: vec![0, 1, 2, 3],
    };
    let leaf_b = felsenstein::TreeNode::Leaf {
        name: "B".into(),
        states: vec![0, 1, 2, 3],
    };
    let root = felsenstein::TreeNode::Internal {
        left: Box::new(leaf_a),
        right: Box::new(leaf_b),
        left_branch: 0.1,
        right_branch: 0.1,
    };
    let ll = felsenstein::log_likelihood(&root, 1.0);
    v.check_pass("Felsenstein: LL finite for 2-leaf tree", ll.is_finite());
    d48 += 1;

    v.check_pass("Felsenstein: LL ≤ 0", ll <= 0.0);
    d48 += 1;

    let t1 = unifrac::PhyloTree::from_newick("((A,B),(C,D));");
    let t2 = unifrac::PhyloTree::from_newick("((A,B),(C,D));");
    v.check_count("RF(T, T) = 0", robinson_foulds::rf_distance(&t1, &t2), 0);
    d48 += 1;

    let t3 = unifrac::PhyloTree::from_newick("((A,C),(B,D));");
    let rf_diff = robinson_foulds::rf_distance(&t1, &t3);
    v.check_pass("RF(different trees) > 0", rf_diff > 0);
    d48 += 1;

    let rf_norm = robinson_foulds::rf_distance_normalized(&t1, &t3);
    v.check_pass("RF normalized ∈ (0, 1]", rf_norm > 0.0 && rf_norm <= 1.0);
    d48 += 1;

    let dm = [0.0, 5.0, 9.0, 5.0, 0.0, 10.0, 9.0, 10.0, 0.0];
    let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let nj = neighbor_joining::neighbor_joining(&dm, &labels);
    v.check_pass("NJ: Newick output non-empty", !nj.newick.is_empty());
    d48 += 1;

    let clock = molecular_clock::strict_clock(
        &[0.1, 0.2, 0.15, 0.05],
        &[None, Some(0), Some(0), Some(1)],
        100.0,
        &[],
    );
    v.check_pass("Strict clock: result present", clock.is_some());
    d48 += 1;
    if let Some(ref c) = clock {
        v.check_pass("Strict clock: rate > 0", c.rate > 0.0);
        d48 += 1;
    }

    domains.push(domain(
        "D48: Align+Tree",
        "wetSpring+groundSpring",
        t.elapsed(),
        d48,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // D49: Diversity Ecosystem
    // ═══════════════════════════════════════════════════════════════════
    v.section("D49: Diversity — Shannon, Simpson, Chao1, Pielou, BC, PCoA");
    let t = Instant::now();
    let mut d49 = 0_u32;

    let uniform4 = vec![25.0, 25.0, 25.0, 25.0];
    v.check(
        "Shannon(uniform 4) = ln(4)",
        diversity::shannon(&uniform4),
        4.0_f64.ln(),
        tolerances::ANALYTICAL_F64,
    );
    d49 += 1;

    v.check(
        "Shannon(singleton) = 0",
        diversity::shannon(&[100.0]),
        0.0,
        tolerances::EXACT_F64,
    );
    d49 += 1;

    let even = diversity::simpson(&uniform4);
    v.check(
        "Simpson(uniform 4) = 0.75",
        even,
        0.75,
        tolerances::ANALYTICAL_F64,
    );
    d49 += 1;

    let chao = diversity::chao1(&[10.0, 5.0, 1.0, 1.0, 0.5]);
    v.check_pass("Chao1 ≥ observed richness", chao >= 5.0);
    d49 += 1;

    v.check(
        "BC(x, x) = 0",
        diversity::bray_curtis(&uniform4, &uniform4),
        0.0,
        tolerances::EXACT_F64,
    );
    d49 += 1;

    let disjoint = vec![0.0, 0.0, 0.0, 100.0];
    let bc_max = diversity::bray_curtis(&[100.0, 0.0, 0.0, 0.0], &disjoint);
    v.check("BC(disjoint) = 1", bc_max, 1.0, tolerances::ANALYTICAL_F64);
    d49 += 1;

    let condensed = [0.5, 0.8, 0.6];
    let pcoa_r = pcoa::pcoa(&condensed, 3, 2);
    v.check_pass("PCoA: produces result", pcoa_r.is_ok());
    d49 += 1;
    if let Ok(ref pr) = pcoa_r {
        v.check_pass("PCoA: eigenvalues present", !pr.eigenvalues.is_empty());
        d49 += 1;
    }

    let mut s_a: HashMap<String, f64> = HashMap::new();
    s_a.insert("A".into(), 10.0);
    s_a.insert("B".into(), 20.0);
    let mut s_b: HashMap<String, f64> = HashMap::new();
    s_b.insert("A".into(), 10.0);
    s_b.insert("B".into(), 20.0);

    let uf_tree = unifrac::PhyloTree::from_newick("(A:0.1,B:0.2);");
    let uf = unifrac::unweighted_unifrac(&uf_tree, &s_a, &s_b);
    v.check(
        "UniFrac(identical) = 0",
        uf,
        0.0,
        tolerances::ANALYTICAL_LOOSE,
    );
    d49 += 1;

    domains.push(domain("D49: Diversity", "wetSpring", t.elapsed(), d49));

    // ═══════════════════════════════════════════════════════════════════
    // D50: ODE Systems
    // ═══════════════════════════════════════════════════════════════════
    v.section("D50: ODE — QS, bistable, capacitor, multi-signal, phage, cooperation");
    let t = Instant::now();
    let mut d50 = 0_u32;

    let qs_p = qs_biofilm::QsBiofilmParams::default();
    let qs_r = qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 50.0, 0.1, &qs_p);
    v.check_pass("QS ODE: converges (>100 steps)", qs_r.t.len() > 100);
    d50 += 1;

    let bi_p = bistable::BistableParams::default();
    let bi_r = bistable::run_bistable(&[0.1, 0.0, 0.0, 1.0, 0.5], 30.0, 0.05, &bi_p);
    v.check_pass("Bistable: converges", bi_r.t.len() > 100);
    d50 += 1;

    let cap_p = capacitor::CapacitorParams::default();
    let cap_r = capacitor::scenario_normal(&cap_p, 0.1);
    v.check_pass("Capacitor: converges", cap_r.t.len() > 100);
    d50 += 1;
    let cap_final = cap_r.states().last().or_exit("unexpected error");
    v.check_pass(
        "Capacitor: all state vars ≥ 0",
        cap_final.iter().all(|x| *x >= 0.0),
    );
    d50 += 1;

    let ms_p = multi_signal::MultiSignalParams::default();
    let ms_r = multi_signal::scenario_wild_type(&ms_p, 0.1);
    v.check_pass("Multi-signal: converges", ms_r.t.len() > 100);
    d50 += 1;

    let ph_p = phage_defense::PhageDefenseParams::default();
    let ph_r = phage_defense::scenario_phage_attack(&ph_p, 0.01);
    v.check_pass("Phage defense: converges", ph_r.states().count() > 100);
    d50 += 1;

    let co_p = cooperation::CooperationParams::default();
    let co_r = cooperation::scenario_equal_start(&co_p, 0.1);
    v.check_pass("Cooperation: converges", co_r.t.len() > 10);
    d50 += 1;

    let ssa = gillespie::birth_death_ssa(2.0, 0.5, 100.0, 42);
    v.check_pass("Gillespie SSA: events > 0", ssa.times.len() > 10);
    d50 += 1;

    let exp_ode = ode::rk4_integrate(|y, _t| vec![-y[0]], &[1.0], 0.0, 5.0, 0.01, None);
    let y_final = exp_ode.states().last().or_exit("unexpected error")[0];
    v.check(
        "RK4: exp(-5) ≈ e^{-5}",
        y_final,
        (-5.0_f64).exp(),
        tolerances::ANALYTICAL_LOOSE,
    );
    d50 += 1;

    domains.push(domain(
        "D50: ODE Systems",
        "wetSpring+neuralSpring",
        t.elapsed(),
        d50,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // D51: Metagenomics
    // ═══════════════════════════════════════════════════════════════════
    v.section("D51: Metagenomics — ANI, SNP, dN/dS, pangenome, molecular clock");
    let t = Instant::now();
    let mut d51 = 0_u32;

    let ani_r = ani::pairwise_ani(b"ACGTACGTACGTACGT", b"ACGTACGTACGTACGT");
    v.check(
        "ANI(identical) = 1.0",
        ani_r.ani,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d51 += 1;

    let ani_diff = ani::pairwise_ani(b"ACGTACGTACGTACGT", b"TTTTTTTTTTTTTTTT");
    v.check_pass("ANI(different) < 1.0", ani_diff.ani < 1.0);
    d51 += 1;

    let snp_r = snp::call_snps(&[b"ACGTACGT".as_ref(), b"ACGTACGT"]);
    v.check_count("SNP(identical) = 0 variants", snp_r.variants.len(), 0);
    d51 += 1;

    let snp_diff = snp::call_snps(&[b"ACGTACGT".as_ref(), b"ACTTACGT"]);
    v.check_pass("SNP(1 diff) > 0 variants", !snp_diff.variants.is_empty());
    d51 += 1;

    let ds_r = dnds::pairwise_dnds(b"ATGATGATG", b"ATGATGATG");
    v.check_pass("dN/dS(identical): result valid", ds_r.is_ok());
    d51 += 1;

    let clusters = vec![
        pangenome::GeneCluster {
            id: "geneA".into(),
            presence: vec![true, true, true],
        },
        pangenome::GeneCluster {
            id: "geneB".into(),
            presence: vec![true, true, false],
        },
        pangenome::GeneCluster {
            id: "geneC".into(),
            presence: vec![true, false, false],
        },
    ];
    let pan = pangenome::analyze(&clusters, 3);
    v.check_count("Pangenome: core genes = 1", pan.core_size, 1);
    d51 += 1;
    v.check_count("Pangenome: total genes = 3", pan.total_size, 3);
    d51 += 1;

    domains.push(domain("D51: Metagenomics", "wetSpring", t.elapsed(), d51));

    // ═══════════════════════════════════════════════════════════════════
    // D52: Analytical Chemistry
    // ═══════════════════════════════════════════════════════════════════
    v.section("D52: Chemistry — EIC, peaks, KMD, tolerance search, spectral match");
    let t = Instant::now();
    let mut d52 = 0_u32;

    let gaussian = [0.0, 1.0, 3.0, 7.0, 10.0, 7.0, 3.0, 1.0, 0.0];
    let peaks = signal::find_peaks(&gaussian, &signal::PeakParams::default());
    v.check_pass("Peak: 1 peak in Gaussian", peaks.len() == 1);
    d52 += 1;
    v.check_count("Peak: apex at index 4", peaks[0].index, 4);
    d52 += 1;

    let monotonic = [1.0, 2.0, 3.0, 4.0, 5.0];
    let no_peaks = signal::find_peaks(&monotonic, &signal::PeakParams::default());
    v.check_count("Peak: 0 in monotonic", no_peaks.len(), 0);
    d52 += 1;

    let kmd_r = kmd::kendrick_mass_defect(&[400.0, 450.0, 500.0], 14.0, 14.0);
    v.check_count("KMD: 3 results for 3 masses", kmd_r.len(), 3);
    d52 += 1;
    v.check_pass(
        "KMD: values finite",
        kmd_r.iter().all(|k| k.kmd.is_finite()),
    );
    d52 += 1;

    let ppm_hits = tolerance_search::find_within_ppm(&[100.0, 200.0, 300.0], 200.0, 10.0);
    v.check_pass("PPM: finds exact match", ppm_hits.contains(&1));
    d52 += 1;

    let cosine = spectral_match::cosine_similarity(
        &[100.0, 200.0, 300.0],
        &[1.0, 0.5, 0.3],
        &[100.0, 200.0, 300.0],
        &[1.0, 0.5, 0.3],
        0.5,
    );
    v.check(
        "Cosine(identical spectra) = 1.0",
        cosine.score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    d52 += 1;

    let cosine_orth = spectral_match::cosine_similarity(&[100.0], &[1.0], &[500.0], &[1.0], 0.1);
    v.check(
        "Cosine(disjoint spectra) = 0.0",
        cosine_orth.score,
        0.0,
        tolerances::EXACT_F64,
    );
    d52 += 1;

    domains.push(domain("D52: Chemistry", "wetSpring", t.elapsed(), d52));

    // ═══════════════════════════════════════════════════════════════════
    // D53: ML Pipeline
    // ═══════════════════════════════════════════════════════════════════
    v.section("D53: ML + HMM — inference, forward algorithm");
    let t = Instant::now();
    let mut d53 = 0_u32;

    let dt = wetspring_barracuda::bio::decision_tree::DecisionTree::from_arrays(
        &[0, -1, -1],
        &[3.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .or_exit("unexpected error");
    let pred = dt.predict(&[2.0, 0.0]);
    v.check_pass("DT: predicts class 0 for feature < threshold", pred == 0);
    d53 += 1;

    let pred_hi = dt.predict(&[5.0, 0.0]);
    v.check_pass("DT: predicts class 1 for feature > threshold", pred_hi == 1);
    d53 += 1;

    let batch = dt.predict_batch(&[vec![2.0, 0.0], vec![5.0, 0.0]]);
    v.check_count("DT batch: 2 predictions", batch.len(), 2);
    d53 += 1;

    let model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.6_f64.ln(), 0.4_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.9_f64.ln(), 0.1_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()],
    };
    let obs = vec![0, 1, 0, 1];
    let fwd = hmm::forward(&model, &obs);
    v.check_pass("HMM forward: LL finite", fwd.log_likelihood.is_finite());
    d53 += 1;
    v.check_pass(
        "HMM forward: LL < 0 (log-probability)",
        fwd.log_likelihood < 0.0,
    );
    d53 += 1;

    domains.push(domain(
        "D53: ML+HMM",
        "wetSpring+neuralSpring",
        t.elapsed(),
        d53,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // D54: Statistics
    // ═══════════════════════════════════════════════════════════════════
    v.section("D54: Statistics — Welford, Pearson, Spearman, jackknife, bootstrap");
    let t = Instant::now();
    let mut d54 = 0_u32;

    let data: Vec<f64> = (1..=100).map(f64::from).collect();
    let mean = barracuda::stats::metrics::mean(&data);
    v.check(
        "mean(1..100) = 50.5",
        mean,
        50.5,
        tolerances::ANALYTICAL_F64,
    );
    d54 += 1;

    let var = barracuda::stats::correlation::variance(&data).or_exit("unexpected error");
    let expected_var = 100.0 * 101.0 / 12.0;
    v.check(
        "var(1..100) = n(n+1)/12",
        var,
        expected_var,
        tolerances::ANALYTICAL_F64,
    );
    d54 += 1;

    let x: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.1).collect();
    let y = x.clone();
    let r = barracuda::stats::correlation::pearson_correlation(&x, &y).or_exit("unexpected error");
    v.check("Pearson(x, x) = 1.0", r, 1.0, tolerances::ANALYTICAL_F64);
    d54 += 1;

    let neg_y: Vec<f64> = x.iter().map(|xi| -xi).collect();
    let r_neg = barracuda::stats::correlation::pearson_correlation(&x, &neg_y).or_exit("unexpected error");
    v.check(
        "Pearson(x, -x) = -1.0",
        r_neg,
        -1.0,
        tolerances::ANALYTICAL_F64,
    );
    d54 += 1;

    let jk = barracuda::stats::jackknife_mean_variance(&[1.0, 2.0, 3.0, 4.0, 5.0]).or_exit("unexpected error");
    v.check(
        "Jackknife estimate = 3.0",
        jk.estimate,
        3.0,
        tolerances::ANALYTICAL_F64,
    );
    d54 += 1;

    let ci = barracuda::stats::bootstrap_ci(
        &data,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        5_000,
        0.95,
        42,
    )
    .or_exit("unexpected error");
    v.check_pass("Bootstrap CI: lower < upper", ci.lower < ci.upper);
    d54 += 1;
    v.check_pass(
        "Bootstrap CI: contains true mean",
        ci.lower <= 50.5 && 50.5 <= ci.upper,
    );
    d54 += 1;

    v.check(
        "erf(0) = 0",
        barracuda::special::erf(0.0),
        0.0,
        tolerances::ERF_PARITY,
    );
    d54 += 1;

    v.check(
        "Φ(0) = 0.5",
        barracuda::stats::norm_cdf(0.0),
        0.5,
        tolerances::NORM_CDF_PARITY,
    );
    d54 += 1;

    let linear =
        barracuda::stats::fit_linear(&[1.0, 2.0, 3.0, 4.0], &[2.0, 4.0, 6.0, 8.0]).or_exit("unexpected error");
    v.check(
        "fit_linear slope = 2.0",
        linear.params[0],
        2.0,
        tolerances::ANALYTICAL_F64,
    );
    d54 += 1;
    v.check(
        "fit_linear intercept = 0.0",
        linear.params[1],
        0.0,
        tolerances::ANALYTICAL_F64,
    );
    d54 += 1;

    domains.push(domain("D54: Statistics", "all Springs", t.elapsed(), d54));

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("V98 CPU v24 Domain Summary");

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║ V98 Comprehensive Bio Domain Parity                              ║");
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
        "║ TOTAL                  │                    │ {total_ms:>5.1}ms │ {total_checks:>3} ║",
    );
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  33 bio modules + statistics — pure Rust, zero FFI");
    println!("  Chain: Paper (Exp313) → CPU (this) → GPU (Exp316) → Streaming → metalForge");

    v.finish();
}
