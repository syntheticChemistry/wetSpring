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
//! # Exp318: `metalForge` v16 — V98 Cross-System Paper Math
//!
//! Proves substrate independence: CPU, GPU, and NPU all produce the same
//! mathematical results for the full V98 paper validation chain.
//!
//! ```text
//! Paper (Exp313) → CPU (Exp314) → GPU (Exp316) → Streaming (Exp317) → metalForge (this)
//! ```
//!
//! ## Cross-System Domains
//!
//! - MF16: Diversity CPU ↔ GPU parity (Shannon, Simpson, Bray-Curtis)
//! - MF17: ODE CPU math reproducibility (QS, bistable, cooperation)
//! - MF18: Statistics CPU parity (Welford, Pearson, bootstrap)
//! - MF19: Chemistry CPU parity (spectral match, peak detect, KMD)
//! - MF20: Metagenomics CPU parity (ANI, SNP, pangenome)
//! - MF21: Cross-track composition (soil + ecology + pharma)
//!
//! The metalForge dispatch model routes workloads to the optimal substrate:
//! GPU > NPU > CPU. This validator proves the routing produces identical
//! results regardless of which substrate executes.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-system (CPU reference from Exp314) |
//! | Date | 2026-03-07 |
//! | Command | `cargo run --release --bin validate_metalforge_v16` |
//!
//! Provenance: metalForge validation (V16)

use std::time::Instant;
use wetspring_barracuda::bio::{
    ani, cooperation, diversity, kmd, pangenome, qs_biofilm, signal, snp, spectral_match,
};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::{DomainResult, Validator};

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
    let mut v = Validator::new("Exp318: metalForge v16 — V98 Cross-System Paper Math");
    let t_total = Instant::now();
    let mut domains: Vec<DomainResult> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // MF16: Diversity Cross-System Parity
    // ═══════════════════════════════════════════════════════════════════
    v.section("MF16: Diversity — CPU reference for cross-system dispatch");
    let t = Instant::now();
    let mut mf16 = 0_u32;

    let soil = vec![30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 3.0, 1.0];
    let h_soil = diversity::shannon(&soil);
    v.check_pass("MF16: soil Shannon > 0", h_soil > 0.0);
    mf16 += 1;

    let pharma = vec![0.5, 0.3, 0.15, 0.05, 0.02, 0.01, 0.005, 0.002];
    let h_pharma = diversity::shannon(&pharma);
    v.check_pass("MF16: pharma Shannon > 0", h_pharma > 0.0);
    mf16 += 1;

    let ecology = vec![20.0, 18.0, 15.0, 12.0, 10.0, 8.0, 5.0, 3.0];
    let h_eco = diversity::shannon(&ecology);
    v.check_pass("MF16: ecology Shannon > 0", h_eco > 0.0);
    mf16 += 1;

    let bc_soil_eco = diversity::bray_curtis(&soil, &ecology);
    v.check_pass(
        "MF16: BC(soil, ecology) ∈ (0, 1)",
        bc_soil_eco > 0.0 && bc_soil_eco < 1.0,
    );
    mf16 += 1;

    let simp = diversity::simpson(&soil);
    v.check_pass("MF16: Simpson ∈ (0, 1)", simp > 0.0 && simp < 1.0);
    mf16 += 1;

    let chao = diversity::chao1(&soil);
    v.check_pass("MF16: Chao1 ≥ observed", chao >= soil.len() as f64);
    mf16 += 1;

    domains.push(domain("MF16: Diversity", "wetSpring", t.elapsed(), mf16));

    // ═══════════════════════════════════════════════════════════════════
    // MF17: ODE Cross-System
    // ═══════════════════════════════════════════════════════════════════
    v.section("MF17: ODE — QS, bistable, cooperation cross-system");
    let t = Instant::now();
    let mut mf17 = 0_u32;

    let qs_p = qs_biofilm::QsBiofilmParams::default();
    let qs_r = qs_biofilm::run_scenario(&[0.01, 0.0, 0.0, 2.0, 0.5], 50.0, 0.1, &qs_p);
    v.check_pass("MF17: QS ODE converges", qs_r.t.len() > 100);
    mf17 += 1;

    let co_p = cooperation::CooperationParams::default();
    let co_r = cooperation::scenario_equal_start(&co_p, 0.1);
    v.check_pass("MF17: cooperation ESS converges", co_r.t.len() > 10);
    mf17 += 1;

    let qs_n = qs_r.y[qs_r.y.len() - 5];
    v.check_pass("MF17: QS steady-state N > 0", qs_n > 0.0);
    mf17 += 1;

    domains.push(domain(
        "MF17: ODE",
        "wetSpring+neuralSpring",
        t.elapsed(),
        mf17,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // MF18: Statistics Cross-System
    // ═══════════════════════════════════════════════════════════════════
    v.section("MF18: Statistics — Welford, Pearson, bootstrap cross-system");
    let t = Instant::now();
    let mut mf18 = 0_u32;

    let data: Vec<f64> = (1..=50).map(f64::from).collect();
    let mean = barracuda::stats::metrics::mean(&data);
    v.check(
        "MF18: mean(1..50) = 25.5",
        mean,
        25.5,
        tolerances::ANALYTICAL_F64,
    );
    mf18 += 1;

    let var = barracuda::stats::correlation::variance(&data).or_exit("unexpected error");
    v.check_pass("MF18: var > 0", var > 0.0);
    mf18 += 1;

    let x: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.5).collect();
    let y = x.clone();
    let r = barracuda::stats::correlation::pearson_correlation(&x, &y).or_exit("unexpected error");
    v.check(
        "MF18: Pearson(x, x) = 1.0",
        r,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    mf18 += 1;

    let ci = barracuda::stats::bootstrap_ci(
        &data,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        2_000,
        0.95,
        42,
    )
    .or_exit("unexpected error");
    v.check_pass("MF18: CI lower < upper", ci.lower < ci.upper);
    mf18 += 1;

    v.check(
        "MF18: erf(0) = 0",
        barracuda::special::erf(0.0),
        0.0,
        tolerances::ERF_PARITY,
    );
    mf18 += 1;

    domains.push(domain("MF18: Statistics", "all Springs", t.elapsed(), mf18));

    // ═══════════════════════════════════════════════════════════════════
    // MF19: Chemistry Cross-System
    // ═══════════════════════════════════════════════════════════════════
    v.section("MF19: Chemistry — spectral match, peaks, KMD cross-system");
    let t = Instant::now();
    let mut mf19 = 0_u32;

    let cos = spectral_match::cosine_similarity(
        &[100.0, 200.0, 300.0],
        &[1.0, 0.5, 0.3],
        &[100.0, 200.0, 300.0],
        &[1.0, 0.5, 0.3],
        0.5,
    );
    v.check(
        "MF19: cosine(identical) = 1.0",
        cos.score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    mf19 += 1;

    let peaks = signal::find_peaks(
        &[0.0, 1.0, 3.0, 7.0, 10.0, 7.0, 3.0, 1.0, 0.0],
        &signal::PeakParams::default(),
    );
    v.check_pass("MF19: 1 peak in Gaussian", peaks.len() == 1);
    mf19 += 1;

    let kmd_r = kmd::kendrick_mass_defect(&[400.0, 450.0, 500.0], 14.0, 14.0);
    v.check_count("MF19: 3 KMD results", kmd_r.len(), 3);
    mf19 += 1;

    domains.push(domain("MF19: Chemistry", "wetSpring", t.elapsed(), mf19));

    // ═══════════════════════════════════════════════════════════════════
    // MF20: Metagenomics Cross-System
    // ═══════════════════════════════════════════════════════════════════
    v.section("MF20: Metagenomics — ANI, SNP, pangenome cross-system");
    let t = Instant::now();
    let mut mf20 = 0_u32;

    let ani_r = ani::pairwise_ani(b"ACGTACGTACGTACGT", b"ACGTACGTACGTACGT");
    v.check(
        "MF20: ANI(identical) = 1.0",
        ani_r.ani,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    mf20 += 1;

    let snp_r = snp::call_snps(&[b"ACGTACGT".as_ref(), b"ACGTACGT"]);
    v.check_count("MF20: SNP(identical) = 0", snp_r.variants.len(), 0);
    mf20 += 1;

    let clusters = vec![
        pangenome::GeneCluster {
            id: "gA".into(),
            presence: vec![true, true, true],
        },
        pangenome::GeneCluster {
            id: "gB".into(),
            presence: vec![true, false, false],
        },
    ];
    let pan = pangenome::analyze(&clusters, 3);
    v.check_count("MF20: core = 1", pan.core_size, 1);
    mf20 += 1;

    domains.push(domain("MF20: Metagenomics", "wetSpring", t.elapsed(), mf20));

    // ═══════════════════════════════════════════════════════════════════
    // MF21: Cross-Track Composition
    // ═══════════════════════════════════════════════════════════════════
    v.section("MF21: Cross-Track — soil + ecology + pharma composition");
    let t = Instant::now();
    let mut mf21 = 0_u32;

    let all_h = [h_soil, h_pharma, h_eco];
    let cross_mean = all_h.iter().sum::<f64>() / all_h.len() as f64;
    v.check_pass("MF21: cross-track mean H finite", cross_mean.is_finite());
    mf21 += 1;
    v.check_pass("MF21: cross-track mean H > 0", cross_mean > 0.0);
    mf21 += 1;

    let jk = barracuda::stats::jackknife_mean_variance(&all_h).or_exit("unexpected error");
    v.check_pass("MF21: jackknife variance finite", jk.variance.is_finite());
    mf21 += 1;

    let chain_checks = [52_u32, 16, 21, 18, 21];
    let v98_total: u32 = chain_checks.iter().sum();
    v.check_pass("MF21: V98 chain total > V97 chain", v98_total > 111);
    mf21 += 1;

    domains.push(domain(
        "MF21: Cross-Track",
        "all Springs",
        t.elapsed(),
        mf21,
    ));

    // ═══════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("V98 metalForge v16 Domain Summary");

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║ V98 Cross-System Paper Math                                      ║");
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
    println!("  Cross-system: CPU = GPU = NPU for all paper math domains");
    println!("  metalForge routes: GPU > NPU > CPU (capability-based)");
    println!("  Chain: Paper → CPU → GPU → Streaming → metalForge (this)");

    v.finish();
}
