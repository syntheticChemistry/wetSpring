// SPDX-License-Identifier: AGPL-3.0-or-later
//! `BarraCuda` CPU parity validation — proves pure Rust math matches Python.
//!
//! This is the bridge between Python baselines and GPU promotion. It runs
//! each algorithmic domain through the Rust CPU implementation and validates
//! against hardcoded Python baseline values.
//!
//! # Domains validated
//!
//! 1. **ODE integration** — RK4 vs `scipy.odeint`
//! 2. **Stochastic simulation** — Gillespie SSA statistical properties
//! 3. **HMM** — Forward, Viterbi, posterior decoding
//! 4. **Sequence alignment** — Smith-Waterman DP
//! 5. **Phylogenetics** — Felsenstein pruning likelihood
//! 6. **Diversity metrics** — Shannon, Simpson
//! 7. **Signal processing** — Peak detection
//! 8. **Game theory** — Cooperation ODE
//! 9. **Tree distance** — Robinson-Foulds
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | scipy.odeint, numpy, dendropy, pure Python (per-domain scripts) |
//! | Baseline version | Feb 2026 |
//! | Baseline command | `python3 scripts/benchmark_rust_vs_python.py` (domains 1–9); per-domain: `waters2008_qs_ode.py`, `gillespie_baseline.py`, `liu2014_hmm_baseline.py`, `smith_waterman_baseline.py`, `felsenstein_pruning_baseline.py`, `bruger2018_cooperation.py`, `rf_distance_baseline.py` |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/benchmark_rust_vs_python.py` |
//! | Data | Synthetic test vectors (hardcoded) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! # Evolution path
//!
//! ```text
//! Python baseline → [THIS] BarraCuda CPU → BarraCuda GPU → sovereign pipeline
//! ```

use std::time::Instant;
use wetspring_barracuda::bio::{
    alignment, bistable, capacitor, cooperation, diversity, felsenstein, gillespie, hmm, ode,
    qs_biofilm, robinson_foulds, signal, unifrac::PhyloTree,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut v = Validator::new("BarraCuda CPU Parity — Pure Rust Math vs Python");
    let mut timings: Vec<(&str, f64)> = Vec::new();

    // ════════════════════════════════════════════════════════════════
    //  Domain 1: ODE Integration (RK4)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 1: ODE Integration (RK4 vs scipy) ═══");

    let t0 = Instant::now();
    let qs = qs_biofilm::scenario_standard_growth(&qs_biofilm::QsBiofilmParams::default(), 0.001);
    let ode_us = t0.elapsed().as_micros();
    let b_ss = ode::steady_state_mean(&qs, 4, 0.1);
    v.check(
        "QS-ODE: biofilm dispersed (B≈0.02)",
        b_ss,
        0.020,
        tolerances::ODE_STEADY_STATE,
    );

    let t0 = Instant::now();
    let cap = capacitor::scenario_normal(&capacitor::CapacitorParams::default(), 0.001);
    let cap_us = t0.elapsed().as_micros();
    let vpsr = ode::steady_state_mean(&cap, 2, 0.1);
    v.check(
        "Capacitor: VpsR steady-state",
        vpsr,
        0.766,
        tolerances::ODE_STEADY_STATE,
    );

    let bi = bistable::BistableParams::default();
    let br = bistable::bifurcation_scan(&bi, 0.0, 10.0, 50, 0.001, 48.0);
    v.check(
        "Bistable: hysteresis detected",
        f64::from(u8::from(br.hysteresis_width > 1.0)),
        1.0,
        0.0,
    );

    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("QS ODE (48h)", ode_us as f64));
        timings.push(("Capacitor ODE (48h)", cap_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 2: Stochastic Simulation (Gillespie SSA)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 2: Stochastic Simulation (Gillespie SSA) ═══");

    let t0 = Instant::now();
    let mut total_final = 0_i64;
    let n_reps = 100_u64;
    for seed in 0..n_reps {
        let mut rng = gillespie::Lcg64::new(seed);
        let reactions = vec![
            gillespie::Reaction {
                propensity: Box::new(|state: &[i64]| {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        0.5 * state[0] as f64
                    }
                }),
                stoichiometry: vec![1],
            },
            gillespie::Reaction {
                propensity: Box::new(|state: &[i64]| {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        0.1 * state[0] as f64
                    }
                }),
                stoichiometry: vec![-1],
            },
        ];
        let traj = gillespie::gillespie_ssa(&[100], &reactions, 10.0, &mut rng);
        total_final += traj.final_state()[0];
    }
    let ssa_us = t0.elapsed().as_micros();
    #[allow(clippy::cast_precision_loss)]
    let mean_final = total_final as f64 / n_reps as f64;
    v.check(
        "SSA: mean final > 50 (birth > death)",
        f64::from(u8::from(mean_final > 50.0)),
        1.0,
        0.0,
    );
    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("Gillespie SSA (100 reps)", ssa_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 3: HMM (Forward, Viterbi, Posterior)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 3: Hidden Markov Models ═══");

    let log_pi = vec![0.5_f64.ln(), 0.5_f64.ln()];
    let log_trans = vec![0.7_f64.ln(), 0.3_f64.ln(), 0.4_f64.ln(), 0.6_f64.ln()];
    let log_emit = vec![
        0.5_f64.ln(),
        0.4_f64.ln(),
        0.1_f64.ln(),
        0.1_f64.ln(),
        0.3_f64.ln(),
        0.6_f64.ln(),
    ];
    let model = hmm::HmmModel {
        n_states: 2,
        log_pi,
        log_trans,
        n_symbols: 3,
        log_emit,
    };
    let obs = vec![0_usize, 1, 2, 1, 0];

    let t0 = Instant::now();
    let fwd = hmm::forward(&model, &obs);
    let hmm_us = t0.elapsed().as_micros();
    v.check(
        "HMM forward LL",
        fwd.log_likelihood,
        -5.625_948_481_320_407,
        1e-6,
    );

    let vit = hmm::viterbi(&model, &obs);
    v.check(
        "Viterbi LL ≤ forward LL",
        f64::from(u8::from(vit.log_probability <= fwd.log_likelihood)),
        1.0,
        0.0,
    );
    #[allow(clippy::cast_precision_loss)]
    {
        v.check("Viterbi path length", vit.path.len() as f64, 5.0, 0.0);
    }

    let n_states = model.n_states;
    let posterior = hmm::posterior(&model, &obs);
    for t_idx in 0..obs.len() {
        let sum: f64 = posterior[t_idx * n_states..(t_idx + 1) * n_states]
            .iter()
            .sum();
        v.check(
            &format!("Posterior t={t_idx} sums to 1"),
            sum,
            1.0,
            tolerances::PYTHON_PARITY,
        );
    }
    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("HMM (5 obs)", hmm_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 4: Sequence Alignment (Smith-Waterman)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 4: Sequence Alignment (Smith-Waterman) ═══");

    let params = alignment::ScoringParams::default();
    let q = b"GATCCTGGCTCAGGATGAACGCTGGCGGCGTGCCTAATAC";
    let t_seq = b"GATCCTGGCTCAGAATGAACGCTGGCGGCATGCCTAATAC";

    let t0 = Instant::now();
    let r = alignment::smith_waterman(q, t_seq, &params);
    let sw_us = t0.elapsed().as_micros();
    v.check("SW 16S (40bp): score = 74", f64::from(r.score), 74.0, 0.0);

    let seqs: Vec<&[u8]> = vec![b"ACGT", b"ACTT", b"GGGG", b"ACGA", b"TTTG"];
    let t0 = Instant::now();
    let scores = alignment::pairwise_scores(&seqs, &params);
    let pw_us = t0.elapsed().as_micros();
    #[allow(clippy::cast_precision_loss)]
    {
        v.check(
            "Pairwise: 5 seqs → 10 pairs",
            scores.len() as f64,
            10.0,
            0.0,
        );
    }
    v.check("Pairwise: ACGT-ACTT score", f64::from(scores[0]), 5.0, 0.0);
    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("SW single (40bp)", sw_us as f64));
        timings.push(("SW pairwise (5 seqs)", pw_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 5: Phylogenetics (Felsenstein pruning)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 5: Phylogenetics (Felsenstein) ═══");

    let tree = felsenstein::TreeNode::Internal {
        left: Box::new(felsenstein::TreeNode::Internal {
            left: Box::new(felsenstein::TreeNode::Leaf {
                name: "sp1".into(),
                states: felsenstein::encode_dna("ACGTACGTACGTACGTACGT"),
            }),
            right: Box::new(felsenstein::TreeNode::Leaf {
                name: "sp2".into(),
                states: felsenstein::encode_dna("ACGTACTTACGTACGTACGT"),
            }),
            left_branch: 0.05,
            right_branch: 0.05,
        }),
        right: Box::new(felsenstein::TreeNode::Leaf {
            name: "sp3".into(),
            states: felsenstein::encode_dna("ACGTACGTACTTACGTACGT"),
        }),
        left_branch: 0.1,
        right_branch: 0.15,
    };

    let t0 = Instant::now();
    let ll = felsenstein::log_likelihood(&tree, 1.0);
    let fels_us = t0.elapsed().as_micros();
    v.check(
        "Felsenstein LL (20bp, 3 taxa)",
        ll,
        -40.881_169_027_599_25,
        tolerances::PHYLO_LIKELIHOOD,
    );
    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("Felsenstein (20bp, 3 taxa)", fels_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 6: Diversity Metrics
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 6: Diversity Metrics ═══");

    let counts = &[10.0, 20.0, 30.0, 15.0, 25.0];
    let t0 = Instant::now();
    let sh = diversity::shannon(counts);
    let si = diversity::simpson(counts);
    let div_us = t0.elapsed().as_micros();
    v.check(
        "Shannon (5 OTUs)",
        sh,
        1.544_479_521_096_86,
        tolerances::PYTHON_PARITY,
    );
    v.check(
        "Simpson (5 OTUs, 1-D)",
        si,
        0.775,
        tolerances::ODE_METHOD_PARITY,
    );
    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("Diversity (Shannon+Simpson)", div_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 7: Signal Processing
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 7: Signal Processing ═══");

    let mut test_signal = vec![0.0_f64; 100];
    #[allow(clippy::cast_precision_loss)]
    for (i, val) in test_signal.iter_mut().enumerate() {
        let x = i as f64 / 100.0 * std::f64::consts::TAU * 3.0;
        *val = x.sin().abs();
    }
    let peak_params = signal::PeakParams {
        min_height: Some(0.5),
        ..signal::PeakParams::default()
    };
    let t0 = Instant::now();
    let peaks = signal::find_peaks(&test_signal, &peak_params);
    let sig_us = t0.elapsed().as_micros();
    v.check(
        "Peaks found > 0",
        f64::from(u8::from(!peaks.is_empty())),
        1.0,
        0.0,
    );
    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("Peak detection (100 pts)", sig_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 8: Game Theory (Cooperation ODE)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 8: Evolutionary Game Theory ═══");

    let cp = cooperation::CooperationParams::default();
    let t0 = Instant::now();
    let r_eq = cooperation::scenario_equal_start(&cp, 0.001);
    let coop_us = t0.elapsed().as_micros();
    let freq = cooperation::cooperator_frequency(&r_eq);
    let final_freq = freq.last().copied().unwrap_or(0.0);
    v.check(
        "Cooperation: freq ∈ (0,1)",
        f64::from(u8::from(final_freq > 0.0 && final_freq < 1.0)),
        1.0,
        0.0,
    );
    #[allow(clippy::cast_precision_loss)]
    {
        timings.push(("Cooperation ODE (100h)", coop_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 9: Tree Distance (Robinson-Foulds)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ Domain 9: Robinson-Foulds Tree Distance ═══");

    let tree_a = PhyloTree::from_newick("((A,B),(C,D));");
    let tree_b = PhyloTree::from_newick("((A,C),(B,D));");
    let t0 = Instant::now();
    let rf = robinson_foulds::rf_distance(&tree_a, &tree_b);
    let rf_us = t0.elapsed().as_micros();
    #[allow(clippy::cast_precision_loss)]
    {
        v.check("RF distance (4 taxa)", rf as f64, 2.0, 0.0);
        timings.push(("Robinson-Foulds (4 taxa)", rf_us as f64));
    }

    // ════════════════════════════════════════════════════════════════
    //  Timing Summary
    // ════════════════════════════════════════════════════════════════
    v.section("═══ BarraCuda CPU Timing Summary ═══");
    println!("\n  {:<35} {:>12}", "Domain", "Time (µs)");
    println!("  {}", "-".repeat(50));
    for (name, us) in &timings {
        println!("  {name:<35} {us:>12.0}");
    }
    let total_us: f64 = timings.iter().map(|(_, t)| t).sum();
    println!("  {}", "-".repeat(50));
    println!("  {:<35} {:>12.0}", "TOTAL", total_us);
    println!();

    v.finish();
}
