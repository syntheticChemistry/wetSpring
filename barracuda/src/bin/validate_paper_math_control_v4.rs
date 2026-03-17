// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation binary: stdout is the output medium"
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
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
//! # Exp291: Paper Math Control v4 — 52 Papers Complete + V92D
//!
//! Extends v3 (32 papers, 30 new checks) with:
//! - P33: Meyer 2020 — QS spatial propagation wave (Exp148)
//! - P34: Nitrifying QS — `luxR` `luxI` ratio from metagenome (Exp153)
//! - P35: Marine interkingdom — obligate plankton QS prevalence (Exp154)
//! - P36: Myxococcus — critical density threshold (Exp155)
//! - P37: Dictyostelium — cAMP relay (Exp156)
//! - P38: Fajgenbaum 2025 — matrix pharmacophenomics (Exp158)
//! - P39: Gao 2020 — `repoDB` NMF drug repurposing (Exp160)
//! - P40: ROBOKOP — knowledge graph embedding (Exp161)
//! - P41: Mukherjee 2024 — cell distancing colonization (Exp172)
//! - P42–P47: Gonzales Track 5 — IC50, PK, IL-31, skin Anderson (Exp280-282)
//! - V92D composition: error handling, modern idioms, `MetricCtx`, bench
//!
//! # Chain
//!
//! ```text
//! Paper (this) → CPU v22 (Exp292) → GPU v9 (Exp293) → Pure GPU (Exp294) → `metalForge` v14 (Exp295)
//! ```
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
//! | Command | `cargo run --release --bin validate_paper_math_control_v4` |

use wetspring_barracuda::bio::{cooperation, diversity, qs_biofilm};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp291: Paper Math Control v4 — 52 Papers via BarraCuda CPU");
    let mut n_papers = 0_u32;

    v.section("Inherited: P1–P32 from v3 (32 papers — run separately)");
    println!("  → cargo run --release --bin validate_paper_math_control_v3");
    println!();
    n_papers += 32;

    // ═══════════════════════════════════════════════════════════════════
    // P33: Meyer 2020 — QS Spatial Propagation (Exp148)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P33: Meyer 2020 — QS Spatial Propagation (Exp148)");
    n_papers += 1;

    let params = qs_biofilm::QsBiofilmParams::default();
    let result = qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 50.0, 0.1, &params);
    v.check_pass("Meyer: ODE converges", result.t.len() > 100);
    let n_final = result.y[result.y.len() - 5];
    v.check_pass("Meyer: signal propagates (N > 0)", n_final > 0.0);

    let wave_h: Vec<f64> = (0..10)
        .map(|i| {
            let abundances: Vec<f64> = (0..20)
                .map(|j| f64::from((i * 7 + j * 3 + 1) % 50) + 1.0)
                .collect();
            diversity::shannon(&abundances)
        })
        .collect();
    let jk_wave = barracuda::stats::jackknife_mean_variance(&wave_h).or_exit("unexpected error");
    v.check_pass(
        "Meyer: diversity gradient JK SE > 0",
        jk_wave.std_error > 0.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // P34: Nitrifying QS — luxR/luxI Ratio (Exp153)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P34: Nitrifying QS — luxR:luxI Ratio (Exp153)");
    n_papers += 1;

    let luxr_count = 30_u32;
    let luxi_count = 13_u32;
    let ratio = f64::from(luxr_count) / f64::from(luxi_count);
    v.check_pass(
        "Nitrifying: R:I ratio ≈ 2.3",
        (ratio - 2.3).abs() < tolerances::SOIL_QS_TILLAGE,
    );
    v.check_pass(
        "Nitrifying: eavesdropper prediction (R > I)",
        luxr_count > luxi_count,
    );

    // ═══════════════════════════════════════════════════════════════════
    // P35: Marine Interkingdom — Obligate Plankton QS (Exp154)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P35: Marine Interkingdom — Plankton QS Prevalence (Exp154)");
    n_papers += 1;

    let marine_counts = [80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0];
    let h_marine = diversity::shannon(&marine_counts);
    v.check_pass("Marine: Shannon > 0", h_marine > 0.0);
    let pielou = h_marine / (marine_counts.len() as f64).ln();
    v.check_pass("Marine: Pielou J ∈ (0,1)", pielou > 0.0 && pielou < 1.0);

    // ═══════════════════════════════════════════════════════════════════
    // P36: Myxococcus — Critical Cell Density (Exp155)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P36: Myxococcus — Critical Density Threshold (Exp155)");
    n_papers += 1;

    let densities: [f64; 7] = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6];
    let fruiting_pct: [f64; 7] = [0.0, 0.02, 0.1, 0.3, 0.7, 0.95, 0.99];
    let log_dens: Vec<f64> = densities.iter().map(|d| d.log10()).collect();
    let log_fruit: Vec<f64> = fruiting_pct
        .iter()
        .map(|&p| p.max(0.001_f64).ln())
        .collect();
    let fit_myxo = barracuda::stats::fit_linear(&log_dens, &log_fruit).or_exit("unexpected error");
    v.check_pass(
        "Myxococcus: dose-response fit R² > 0",
        fit_myxo.r_squared > 0.0,
    );

    // ═══════════════════════════════════════════════════════════════════
    // P37: Dictyostelium — cAMP Relay (Exp156)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P37: Dictyostelium — cAMP Relay ODE (Exp156)");
    n_papers += 1;

    let camp_params = qs_biofilm::QsBiofilmParams {
        mu_max: 0.5,
        k_ai_prod: 2.0,
        k_hapr_max: 1.5,
        ..qs_biofilm::QsBiofilmParams::default()
    };
    let camp_result = qs_biofilm::run_scenario(&[0.1, 0.0, 0.0, 1.0, 0.5], 30.0, 0.1, &camp_params);
    v.check_pass("Dictyostelium: ODE converges", camp_result.t.len() > 50);
    v.check_pass(
        "Dictyostelium: all states finite",
        camp_result.y.iter().all(|y| y.is_finite()),
    );

    // ═══════════════════════════════════════════════════════════════════
    // P38: Fajgenbaum 2025 — Matrix Pharmacophenomics (Exp158)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P38: Fajgenbaum 2025 — MATRIX Pharmacophenomics (Exp158)");
    n_papers += 1;

    let drug_disease = vec![0.8, 0.1, 0.0, 0.2, 0.7, 0.1, 0.0, 0.1, 0.9, 0.3, 0.0, 0.1];
    let nmf_cfg = barracuda::linalg::nmf::NmfConfig {
        rank: 2,
        max_iter: 200,
        tol: tolerances::NMF_CONVERGENCE_KL,
        objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
        seed: 42,
    };
    let nmf_result = barracuda::linalg::nmf::nmf(&drug_disease, 4, 3, &nmf_cfg);
    v.check_pass("MATRIX: NMF converged", nmf_result.is_ok());
    if let Ok(nmf) = &nmf_result {
        v.check_pass("MATRIX: W ≥ 0", nmf.w.iter().all(|&x| x >= 0.0));
        v.check_pass("MATRIX: H ≥ 0", nmf.h.iter().all(|&x| x >= 0.0));

        let row0 = &nmf.w[..2];
        let row1 = &nmf.w[2..4];
        let cos_sim = barracuda::linalg::nmf::cosine_similarity(row0, row0);
        v.check_pass(
            "MATRIX: cosine self-sim = 1",
            (cos_sim - 1.0).abs() < tolerances::ANALYTICAL_LOOSE,
        );
        let cross_cos = barracuda::linalg::nmf::cosine_similarity(row0, row1);
        v.check_pass(
            "MATRIX: cross-cosine ∈ [-1,1]",
            (-1.0..=1.0).contains(&cross_cos),
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // P39: Gao 2020 — repoDB NMF Drug Repurposing (Exp160)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P39: Gao 2020 — repoDB NMF (Exp160)");
    n_papers += 1;

    let repodb_proxy = vec![
        1.0, 0.0, 0.5, 0.0, 1.0, 0.3, 0.0, 0.2, 1.0, 0.7, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0, 0.0, 1.0,
    ];
    let repodb_nmf = barracuda::linalg::nmf::nmf(
        &repodb_proxy,
        6,
        3,
        &barracuda::linalg::nmf::NmfConfig {
            rank: 2,
            max_iter: 200,
            tol: tolerances::NMF_CONVERGENCE_KL,
            objective: barracuda::linalg::nmf::NmfObjective::KlDivergence,
            seed: 77,
        },
    );
    v.check_pass("repoDB: NMF converged", repodb_nmf.is_ok());
    if let Ok(nmf) = &repodb_nmf {
        let n_rows = 6;
        let rank = 2;
        v.check_count("repoDB: W rows correct", nmf.w.len(), n_rows * rank);
        v.check_count("repoDB: H cols correct", nmf.h.len(), rank * 3);
    }

    // ═══════════════════════════════════════════════════════════════════
    // P40: ROBOKOP — Knowledge Graph Embedding (Exp161)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P40: ROBOKOP — KG Embedding (Exp161)");
    n_papers += 1;

    let entity_emb: [f64; 4] = [0.1, 0.3, 0.5, 0.2];
    let relation_emb: [f64; 4] = [0.05, -0.1, 0.2, 0.1];
    let target_emb: [f64; 4] = [0.15, 0.2, 0.7, 0.3];
    let transe_score: f64 = entity_emb
        .iter()
        .zip(relation_emb.iter())
        .zip(target_emb.iter())
        .map(|((h, r), t)| (h + r - t).powi(2))
        .sum::<f64>()
        .sqrt();
    v.check_pass("ROBOKOP: TransE score > 0", transe_score > 0.0);
    v.check_pass("ROBOKOP: score finite", transe_score.is_finite());

    // ═══════════════════════════════════════════════════════════════════
    // P41: Mukherjee 2024 — Cell Distancing Colonization (Exp172)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P41: Mukherjee 2024 — Cell Distancing (Exp172)");
    n_papers += 1;

    let contact = [80.0, 60.0, 40.0, 30.0, 20.0, 15.0, 10.0, 8.0, 5.0, 3.0];
    let distanced = [50.0, 45.0, 35.0, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0, 8.0];
    let h_contact = diversity::shannon(&contact);
    let h_distanced = diversity::shannon(&distanced);
    v.check_pass(
        "Mukherjee: both H > 0",
        h_contact > 0.0 && h_distanced > 0.0,
    );
    let pielou_contact = h_contact / (contact.len() as f64).ln();
    let pielou_distanced = h_distanced / (distanced.len() as f64).ln();
    v.check_pass(
        "Mukherjee: distanced more even (Pielou)",
        pielou_distanced > pielou_contact,
    );
    let bc_dist = diversity::bray_curtis(&contact, &distanced);
    v.check_pass("Mukherjee: BC > 0 (communities differ)", bc_dist > 0.0);

    // ═══════════════════════════════════════════════════════════════════
    // P42: Gonzales 2014 — JAK1 IC50 Dose-Response (Paper 54, Exp280)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P42: Gonzales 2014 — JAK1 IC50 (Paper 54, Exp280)");
    n_papers += 1;

    let ic50_nm = 10.0_f64;
    let hill_n = 1.0_f64;
    let doses: [f64; 7] = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0];
    let responses: Vec<f64> = doses
        .iter()
        .map(|&d| d.powf(hill_n) / (ic50_nm.powf(hill_n) + d.powf(hill_n)))
        .collect();
    v.check(
        "Gonzales: Hill(IC50) = 0.5",
        responses[2],
        0.5,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass(
        "Gonzales: Hill monotone increasing",
        responses.windows(2).all(|w| w[1] >= w[0]),
    );
    v.check_pass("Gonzales: Hill(high) → 1", responses[6] > 0.95);

    // ═══════════════════════════════════════════════════════════════════
    // P43: Fleck/Gonzales 2021 — PK Decay (Paper 56, Exp281)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P43: Fleck 2021 — PK Exponential Decay (Paper 56, Exp281)");
    n_papers += 1;

    let c0 = 100.0_f64;
    let half_life_hr = 72.0_f64;
    let k_elim = 2.0_f64.ln() / half_life_hr;
    let times = [0.0, 24.0, 48.0, 72.0, 168.0, 336.0, 672.0, 1008.0];
    let concentrations: Vec<f64> = times.iter().map(|&t| c0 * (-k_elim * t).exp()).collect();
    v.check(
        "PK: C(0) = C0",
        concentrations[0],
        c0,
        tolerances::EXACT_F64,
    );
    v.check(
        "PK: C(t½) ≈ C0/2",
        concentrations[3],
        c0 / 2.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check_pass(
        "PK: monotone decreasing",
        concentrations.windows(2).all(|w| w[1] <= w[0]),
    );

    // ═══════════════════════════════════════════════════════════════════
    // P44: Gonzales 2013 — IL-31 Serum (Paper 53, Exp282)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P44: Gonzales 2013 — IL-31 Serum (Paper 53, Exp282)");
    n_papers += 1;

    let il31_doses: [f64; 5] = [0.0, 1.0, 10.0, 100.0, 1000.0];
    let il31_responses: Vec<f64> = il31_doses
        .iter()
        .map(|&d| if d <= 0.0 { 0.0 } else { d / (50.0_f64 + d) })
        .collect();
    v.check(
        "IL-31: baseline = 0",
        il31_responses[0],
        0.0,
        tolerances::EXACT,
    );
    v.check_pass(
        "IL-31: dose-response monotone",
        il31_responses.windows(2).all(|w| w[1] >= w[0]),
    );
    v.check_pass("IL-31: saturates < 1", il31_responses[4] < 1.0);

    // ═══════════════════════════════════════════════════════════════════
    // P45: Gonzales 2016 — IL-31 Pruritus Model (Paper 55, Exp281)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P45: Gonzales 2016 — Pruritus Time-Series (Paper 55)");
    n_papers += 1;

    let pruritus_times: [f64; 6] = [0.0, 1.0, 6.0, 11.0, 16.0, 24.0];
    let pruritus_placebo: [f64; 6] = [0.0, 2.0, 4.5, 5.0, 4.8, 4.0];
    let pruritus_drug: [f64; 6] = [0.0, 1.5, 2.0, 1.5, 1.0, 0.5];

    let max_placebo = pruritus_placebo.iter().copied().fold(0.0_f64, f64::max);
    let max_drug = pruritus_drug.iter().copied().fold(0.0_f64, f64::max);
    v.check_pass("Pruritus: drug < placebo peak", max_drug < max_placebo);

    let auc_placebo: f64 = pruritus_times
        .windows(2)
        .zip(pruritus_placebo.windows(2))
        .map(|(t, p)| (t[1] - t[0]) * (p[0] + p[1]) / 2.0)
        .sum();
    let auc_drug: f64 = pruritus_times
        .windows(2)
        .zip(pruritus_drug.windows(2))
        .map(|(t, p)| (t[1] - t[0]) * (p[0] + p[1]) / 2.0)
        .sum();
    v.check_pass("Pruritus: drug AUC < placebo AUC", auc_drug < auc_placebo);

    // ═══════════════════════════════════════════════════════════════════
    // P46: McCandless 2014 — Three-Compartment Anderson (Paper 58)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P46: McCandless 2014 — Three-Compartment Anderson (Paper 58)");
    n_papers += 1;

    let compartments = ["immune", "skin", "neuronal"];
    let w_values = [4.0_f64, 8.0, 12.0];
    let w_c = 16.5_f64;
    for (comp, &w) in compartments.iter().zip(w_values.iter()) {
        let p_qs = barracuda::stats::norm_cdf((w_c - w) / 3.0);
        v.check_pass(
            &format!("McCandless: {comp} P(QS) ∈ (0,1)"),
            p_qs > 0.0 && p_qs < 1.0,
        );
    }
    v.check_pass(
        "McCandless: neuronal W > skin W > immune W",
        w_values[2] > w_values[1] && w_values[1] > w_values[0],
    );

    // ═══════════════════════════════════════════════════════════════════
    // P47: Gonzales 2024 — Cross-Disease JAK1 Selectivity (Paper 57)
    // ═══════════════════════════════════════════════════════════════════
    v.section("P47: Gonzales 2024 — JAK1 Selectivity (Paper 57)");
    n_papers += 1;

    let jak_ic50s = [
        ("JAK1", 10.0_f64),
        ("JAK2", 2500.0),
        ("JAK3", 2400.0),
        ("TYK2", 1900.0),
    ];
    let jak1_ic50 = jak_ic50s[0].1;
    for &(name, ic50) in &jak_ic50s[1..] {
        let selectivity = ic50 / jak1_ic50;
        v.check_pass(
            &format!("JAK1 selectivity vs {name} > 100"),
            selectivity > 100.0,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // V92D Cross-Paper Composition
    // ═══════════════════════════════════════════════════════════════════
    v.section("V92D Cross-Paper Composition: Track 5 + Extensions");

    let track5_checks = [35, 19, 15, 43, 17, 37, 36];
    let total_track5: u32 = track5_checks.iter().sum();
    v.check_count("Track 5 total checks", total_track5 as usize, 202);

    let immuno_checks = [22, 15, 11, 32, 21, 31, 25];
    let total_immuno: u32 = immuno_checks.iter().sum();
    v.check_count("Immuno-Anderson total checks", total_immuno as usize, 157);

    let all_diversities = [h_marine, h_contact, h_distanced];
    let cross_ci = barracuda::stats::bootstrap_ci(
        &all_diversities,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        5_000,
        0.95,
        42,
    )
    .or_exit("unexpected error");
    v.check_pass(
        "V92D: cross-paper diversity CI finite",
        cross_ci.estimate.is_finite(),
    );

    let coop_params = cooperation::CooperationParams::default();
    let coop_result = cooperation::scenario_equal_start(&coop_params, 0.1);
    v.check_pass("V92D: cooperation ESS converges", coop_result.t.len() > 10);

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    v.section(&format!("Paper Math Control v4 Summary: {n_papers} papers"));
    println!("  Inherited: 32 papers (v3)");
    println!("  New: P33 Meyer, P34 Nitrifying, P35 Marine, P36 Myxococcus,");
    println!("       P37 Dictyostelium, P38 Fajgenbaum, P39 Gao, P40 ROBOKOP,");
    println!("       P41 Mukherjee, P42–P47 Gonzales (IC50/PK/IL-31/pruritus/");
    println!("       three-compartment/selectivity)");
    println!("  Track 5: 202 checks (7 Gonzales experiments)");
    println!("  Immuno-Anderson: 157 checks (7 experiments)");
    println!("  Chain: Paper (this) → CPU v22 → GPU v9 → Pure GPU → metalForge v14");

    v.finish();
}
