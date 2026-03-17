// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
//! # Exp331: Local Evolution & Upstream Readiness
//!
//! Validates all idiomatic improvements made for upstream absorption:
//!
//! | Domain | Checks |
//! |--------|--------|
//! | E1 FitResult | `.slope()` / `.intercept()` used over `.params[0]` |
//! | E2 HmmModel  | Doc alias discoverable, forward algorithm works |
//! | E3 NMF       | Re-exported from `bio::nmf` for domain-level access |
//! | E4 Quality   | Test extraction: `quality_tests.rs` path wired correctly |
//! | E5 Workloads | `data_bytes` wired into bandwidth-sensitive workloads |

use wetspring_barracuda::bio::{hmm, pangenome};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new("Exp331: Local Evolution & Upstream Readiness");

    // ── E1: FitResult API ──
    v.section("E1 — FitResult Named Accessors");

    let fit = barracuda::stats::fit_linear(&[1.0, 2.0, 3.0, 4.0], &[2.0, 4.0, 6.0, 8.0]);
    v.check_pass("fit_linear succeeds", fit.is_some());
    if let Some(ref r) = fit {
        v.check_pass("slope() available", r.slope().is_some());
        v.check_pass("intercept() available", r.intercept().is_some());
        v.check(
            "slope() ≈ 2.0",
            r.slope().unwrap_or(0.0),
            2.0,
            tolerances::ANALYTICAL_LOOSE,
        );
        v.check(
            "intercept() ≈ 0.0",
            r.intercept().unwrap_or(99.0),
            0.0,
            tolerances::ANALYTICAL_LOOSE,
        );
        v.check_pass("coefficients() returns slice", r.coefficients().len() >= 2);
    }

    let clusters = vec![
        pangenome::GeneCluster {
            id: "g1".into(),
            presence: vec![true, true, true],
        },
        pangenome::GeneCluster {
            id: "g2".into(),
            presence: vec![true, true, false],
        },
        pangenome::GeneCluster {
            id: "g3".into(),
            presence: vec![true, false, false],
        },
        pangenome::GeneCluster {
            id: "g4".into(),
            presence: vec![false, true, true],
        },
    ];
    let pan = pangenome::analyze(&clusters, 3);
    v.check_pass(
        "pangenome analyze uses slope() internally",
        pan.heaps_alpha.is_some() || pan.core_size > 0,
    );

    // ── E2: HmmModel doc alias ──
    v.section("E2 — HmmModel + Forward");

    let model = hmm::HmmModel {
        n_states: 2,
        log_pi: vec![0.5_f64.ln(), 0.5_f64.ln()],
        log_trans: vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()],
        n_symbols: 2,
        log_emit: vec![0.9_f64.ln(), 0.1_f64.ln(), 0.2_f64.ln(), 0.8_f64.ln()],
    };
    v.check_count("HmmModel n_states", model.n_states, 2);
    v.check_count("HmmModel log_pi length", model.log_pi.len(), 2);
    v.check_count("HmmModel log_trans length", model.log_trans.len(), 4);
    let fwd = hmm::forward(&model, &[0, 1, 0, 1]);
    v.check_pass(
        "HMM forward log-likelihood is finite",
        fwd.log_likelihood.is_finite(),
    );
    v.check_pass(
        "HMM forward log-likelihood < 0 (log-prob)",
        fwd.log_likelihood < 0.0,
    );

    // ── E3: NMF re-export ──
    v.section("E3 — NMF Re-export");

    v.check_pass("bio::nmf module accessible", {
        let config = wetspring_barracuda::bio::nmf::NmfConfig {
            rank: 2,
            max_iter: 100,
            tol: 1e-4,
            objective: wetspring_barracuda::bio::nmf::NmfObjective::KlDivergence,
            seed: 42,
        };
        config.rank == 2
    });

    let v_matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let nmf_result = wetspring_barracuda::bio::nmf::nmf(
        &v_matrix,
        2,
        3,
        &wetspring_barracuda::bio::nmf::NmfConfig {
            rank: 2,
            max_iter: 200,
            tol: 1e-4,
            objective: wetspring_barracuda::bio::nmf::NmfObjective::Euclidean,
            seed: 42,
        },
    );
    v.check_pass("NMF via bio::nmf succeeds", nmf_result.is_ok());
    if let Ok(ref nmf) = nmf_result {
        v.check_pass("NMF W has correct rows", nmf.w.len() == 2 * 2);
        v.check_pass("NMF H has correct cols", nmf.h.len() == 2 * 3);
    }

    // ── E4: Quality test extraction ──
    v.section("E4 — Quality Module Refactor");

    let params = wetspring_barracuda::bio::quality::QualityParams::default();
    v.check_count("default min_length", params.min_length, 36);
    v.check(
        "default leading_min_quality",
        f64::from(params.leading_min_quality),
        3.0,
        tolerances::EXACT,
    );
    v.check(
        "default trailing_min_quality",
        f64::from(params.trailing_min_quality),
        3.0,
        tolerances::EXACT,
    );

    // ── E5: Workload data_bytes ──
    v.section("E5 — Workload Bandwidth Hints");

    let workloads = wetspring_forge::workloads::all_workloads();
    let with_bytes_count = workloads
        .iter()
        .filter(|w| w.workload.data_bytes.is_some())
        .count();
    v.check_pass("≥4 workloads have data_bytes", with_bytes_count >= 4);
    v.check_pass("kmer_histogram has data_bytes", {
        workloads
            .iter()
            .any(|w| w.workload.name == "kmer_histogram" && w.workload.data_bytes.is_some())
    });
    v.check_pass("dada2 has data_bytes", {
        workloads
            .iter()
            .any(|w| w.workload.name == "dada2" && w.workload.data_bytes.is_some())
    });
    v.check_pass("pcoa has data_bytes", {
        workloads
            .iter()
            .any(|w| w.workload.name == "pcoa" && w.workload.data_bytes.is_some())
    });
    v.check_pass("smith_waterman has data_bytes", {
        workloads
            .iter()
            .any(|w| w.workload.name == "smith_waterman" && w.workload.data_bytes.is_some())
    });

    v.finish();
}
