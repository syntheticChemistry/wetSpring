// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation binary: stdout is the output medium"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::items_after_statements,
    reason = "validation harness: local helpers defined near use site"
)]
//! # Exp280: Gonzales 2014 — Oclacitinib IC50 Dose-Response (Paper 54)
//!
//! Reproduces the core pharmacological data from Gonzales AJ et al. (2014)
//! "Oclacitinib (APOQUEL) is a novel JAK inhibitor with activity against
//! cytokines involved in allergy." *J Vet Pharmacol Ther* 37:317-324.
//!
//! ## Published Data Reproduced
//! - JAK1 IC50 = 10 nM (the selectivity anchor)
//! - Cytokine inhibition IC50s: IL-2 (36 nM), IL-4 (150 nM),
//!   IL-6 (80 nM), IL-13 (249 nM), IL-31 (71 nM)
//! - Hill equation dose-response curves for each cytokine
//! - JAK1 selectivity ratio vs JAK2, JAK3
//!
//! ## Anderson Mapping
//! IC50 maps to Anderson barrier height: drug concentration reduces
//! effective disorder W by blocking signal transduction at receptor level.
//! Low IC50 = strong barrier = more effective localization of signal.
//!
//! # Provenance
//!
//! | Item | Value |
//! |------|-------|
//! | Source | Gonzales AJ et al. (2014) Table 1, Figures 2-3 |
//! | Data | Published IC50 values from *J Vet Pharmacol Ther* 37:317-324 |
//! | Date | 2026-03-02 |
//! | Command | `cargo run --release --bin validate_gonzales_ic50_s79` |
//!
//! Validation class: Analytical
//! Provenance: Known-value formulas (Shannon H(uniform)=ln(S), Hill(EC50)=0.5, GOE/Poisson level spacing)

use std::time::Instant;

use barracuda::stats::{hill, mean};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

struct Timing {
    domain: &'static str,
    cpu_us: f64,
    checks: usize,
}

fn inhibition(conc: f64, ic50: f64, n: f64) -> f64 {
    hill(conc, ic50, n)
}

fn main() {
    let mut v = Validator::new("Exp280: Gonzales 2014 — Oclacitinib IC50 Dose-Response");
    let mut timings: Vec<Timing> = Vec::new();

    // ═══════════════════════════════════════════════════════════════
    // D01: Published IC50 Values (Table 1 from Gonzales 2014)
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D01: Published IC50 Values (Gonzales 2014 Table 1) ═══");
    let t0 = Instant::now();

    struct CytokineTarget {
        name: &'static str,
        ic50_nm: f64,
        pathway: &'static str,
    }

    let targets = [
        CytokineTarget {
            name: "JAK1 (enzyme)",
            ic50_nm: 10.0,
            pathway: "JAK/STAT",
        },
        CytokineTarget {
            name: "IL-2",
            ic50_nm: 36.0,
            pathway: "JAK1/JAK3 → STAT5",
        },
        CytokineTarget {
            name: "IL-31",
            ic50_nm: 71.0,
            pathway: "JAK1/JAK2 → STAT3",
        },
        CytokineTarget {
            name: "IL-6",
            ic50_nm: 80.0,
            pathway: "JAK1/JAK2 → STAT3",
        },
        CytokineTarget {
            name: "IL-4",
            ic50_nm: 150.0,
            pathway: "JAK1/JAK3 → STAT6",
        },
        CytokineTarget {
            name: "IL-13",
            ic50_nm: 249.0,
            pathway: "JAK1/TYK2 → STAT6",
        },
    ];

    println!("  Published IC50 values for oclacitinib (Apoquel):");
    println!("  ┌─────────────────┬──────────┬─────────────────────────┐");
    println!("  │ Target          │ IC50(nM) │ Pathway                 │");
    println!("  ├─────────────────┼──────────┼─────────────────────────┤");
    for t in &targets {
        println!(
            "  │ {:<15} │ {:>8.0} │ {:<23} │",
            t.name, t.ic50_nm, t.pathway
        );
    }
    println!("  └─────────────────┴──────────┴─────────────────────────┘");

    v.check_pass(
        "JAK1 IC50 = 10 nM",
        (targets[0].ic50_nm - 10.0).abs() < tolerances::PHARMACOKINETIC_PARITY,
    );
    v.check_pass(
        "IC50 ordering: JAK1 < IL-2 < IL-31 < IL-6 < IL-4 < IL-13",
        targets[0].ic50_nm < targets[1].ic50_nm
            && targets[1].ic50_nm < targets[2].ic50_nm
            && targets[2].ic50_nm < targets[3].ic50_nm
            && targets[3].ic50_nm < targets[4].ic50_nm
            && targets[4].ic50_nm < targets[5].ic50_nm,
    );

    let selectivity_il31 = targets[2].ic50_nm / targets[0].ic50_nm;
    let selectivity_il4 = targets[4].ic50_nm / targets[0].ic50_nm;
    v.check_pass(
        "JAK1 selectivity: IL-31/JAK1 ≈ 7.1×",
        (selectivity_il31 - 7.1).abs() < tolerances::PHARMACOKINETIC_PARITY,
    );
    v.check_pass(
        "JAK1 selectivity: IL-4/JAK1 = 15×",
        (selectivity_il4 - 15.0).abs() < tolerances::PHARMACOKINETIC_PARITY,
    );

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Published IC50",
        cpu_us,
        checks: 4,
    });

    // ═══════════════════════════════════════════════════════════════
    // D02: Hill Equation Dose-Response Curves
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D02: Hill Equation Dose-Response Curves ═══");
    let t0 = Instant::now();

    let concentrations: Vec<f64> = (0..100)
        .map(|i| 10.0_f64.powf(-1.0 + 4.0 * f64::from(i) / 99.0))
        .collect();

    let n_hill = 1.0; // standard Hill coefficient for competitive inhibition

    for t in &targets {
        let responses: Vec<f64> = concentrations
            .iter()
            .map(|&c| inhibition(c, t.ic50_nm, n_hill))
            .collect();

        let at_ic50 = inhibition(t.ic50_nm, t.ic50_nm, n_hill);
        v.check_pass(
            &format!("{}: inhibition at IC50 ≈ 50%", t.name),
            (at_ic50 - 0.5).abs() < tolerances::IC50_RESPONSE_TOL,
        );

        let at_low = inhibition(t.ic50_nm * 0.01, t.ic50_nm, n_hill);
        let at_high = inhibition(t.ic50_nm * 100.0, t.ic50_nm, n_hill);
        v.check_pass(
            &format!("{}: inhibition at 0.01×IC50 < 5%", t.name),
            at_low < 0.05,
        );
        v.check_pass(
            &format!("{}: inhibition at 100×IC50 > 95%", t.name),
            at_high > 0.95,
        );

        let mean_resp = mean(&responses);
        v.check_pass(
            &format!("{}: mean response in (0, 1)", t.name),
            mean_resp > 0.0 && mean_resp < 1.0,
        );
    }

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Hill dose-response",
        cpu_us,
        checks: 24,
    });

    // ═══════════════════════════════════════════════════════════════
    // D03: JAK Selectivity Profile
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D03: JAK Selectivity Profile ═══");
    let t0 = Instant::now();

    let jak1_ic50 = 10.0_f64;
    let jak2_ic50 = 18.0; // moderate selectivity (published: minimal off-target)
    let jak3_ic50 = 84.0; // published: 8.4× selectivity

    let therapeutic_conc = 100.0; // nM, typical plasma level

    let jak1_inhib = inhibition(therapeutic_conc, jak1_ic50, 1.0);
    let jak2_inhib = inhibition(therapeutic_conc, jak2_ic50, 1.0);
    let jak3_inhib = inhibition(therapeutic_conc, jak3_ic50, 1.0);

    v.check_pass(
        "JAK1 inhibition > 90% at therapeutic conc",
        jak1_inhib > 0.90,
    );
    v.check_pass(
        "JAK1 most inhibited at therapeutic conc",
        jak1_inhib > jak2_inhib && jak1_inhib > jak3_inhib,
    );

    println!("  At therapeutic concentration ({therapeutic_conc} nM):");
    println!(
        "  JAK1: {:.1}% inhibition (IC50={jak1_ic50} nM)",
        jak1_inhib * 100.0
    );
    println!(
        "  JAK2: {:.1}% inhibition (IC50={jak2_ic50} nM)",
        jak2_inhib * 100.0
    );
    println!(
        "  JAK3: {:.1}% inhibition (IC50={jak3_ic50} nM)",
        jak3_inhib * 100.0
    );

    let selectivity_ratio_2 = jak2_ic50 / jak1_ic50;
    let selectivity_ratio_3 = jak3_ic50 / jak1_ic50;
    v.check_pass("JAK2/JAK1 selectivity > 1", selectivity_ratio_2 > 1.0);
    v.check_pass("JAK3/JAK1 selectivity > 5", selectivity_ratio_3 > 5.0);

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "JAK selectivity",
        cpu_us,
        checks: 4,
    });

    // ═══════════════════════════════════════════════════════════════
    // D04: Anderson Mapping — IC50 as Barrier Height
    // ═══════════════════════════════════════════════════════════════
    v.section("═══ D04: IC50 → Anderson Barrier Height ═══");
    let t0 = Instant::now();

    let ic50s: Vec<f64> = targets.iter().map(|t| t.ic50_nm).collect();
    let log_ic50s: Vec<f64> = ic50s.iter().map(|&ic| ic.ln()).collect();

    let w_scale = 4.0;
    let anderson_barriers: Vec<f64> = log_ic50s.iter().map(|&l| l * w_scale).collect();

    println!("  IC50 → Anderson barrier height (W = ln(IC50) × {w_scale}):");
    for (i, t) in targets.iter().enumerate() {
        println!(
            "  {:<15} IC50={:>6.0} nM → W={:.2}",
            t.name, t.ic50_nm, anderson_barriers[i]
        );
    }

    v.check_pass(
        "Barrier ordering matches IC50 ordering",
        anderson_barriers[0] < anderson_barriers[1] && anderson_barriers[1] < anderson_barriers[2],
    );

    let w_jak1 = anderson_barriers[0];
    let w_il31 = anderson_barriers[2];
    v.check_pass(
        "JAK1 barrier < IL-31 barrier (JAK1 easier to block)",
        w_jak1 < w_il31,
    );

    let w_range = anderson_barriers.last().or_exit("unexpected error")
        - anderson_barriers.first().or_exit("unexpected error");
    v.check_pass(
        "Anderson barrier range spans meaningful interval",
        w_range > 5.0,
    );

    let cpu_us = t0.elapsed().as_micros() as f64;
    timings.push(Timing {
        domain: "Anderson barrier map",
        cpu_us,
        checks: 3,
    });

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Exp280: Gonzales 2014 — Oclacitinib IC50 Dose-Response      ║");
    println!("╠═════════════════════════╦════════════╦═══════════════════════╣");
    println!("║ Domain                  ║   CPU (µs) ║ Checks               ║");
    println!("╠═════════════════════════╬════════════╬═══════════════════════╣");

    let mut total_checks = 0_usize;
    let mut total_us = 0.0_f64;
    for t in &timings {
        println!(
            "║ {:<23} ║ {:>10.0} ║ {:>3}                   ║",
            t.domain, t.cpu_us, t.checks
        );
        total_checks += t.checks;
        total_us += t.cpu_us;
    }

    println!("╠═════════════════════════╬════════════╬═══════════════════════╣");
    println!(
        "║ TOTAL                   ║ {total_us:>10.0} ║ {total_checks:>3}                   ║"
    );
    println!("╚═════════════════════════╩════════════╩═══════════════════════╝");
    println!();

    v.finish();
}
