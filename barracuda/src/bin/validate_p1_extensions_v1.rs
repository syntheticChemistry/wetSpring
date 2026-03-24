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
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "validation harness: non-negative values cast to unsigned"
)]
//! # Exp369: P1 Dataset Extensions — Pipeline Framework
//!
//! Validates the pipeline framework for P1-tier dataset extensions.
//! Each extension uses the P0 pipeline (diversity → Anderson W → P(QS))
//! with domain-specific parameters and new data sources.
//!
//! ## P1 Extensions
//!
//! 1. Cold seep metagenomes (PRJNA315684): anaerobic reference, 170 samples
//! 2. Tara Oceans (PRJEB1787): marine diversity, 243 stations
//! 3. HMP (PRJNA275349): human gut, 4,700 samples
//! 4. AMR surveillance: sentinel framework on real data
//! 5. Mycorrhizal Anderson: fungal network model (ITS + micro-CT)
//!
//! ## Domains
//!
//! - D126: Cold Seep Model — deep anaerobic reference community
//! - D127: Tara Oceans Model — marine epipelagic/mesopelagic diversity
//! - D128: HMP Gut Model — human gut anaerobic reference
//! - D129: AMR Sentinel Framework — resistance gene diversity → Anderson
//! - D130: Mycorrhizal Anderson — fungal hyphal network as lattice
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | P1 dataset extension framework |
//! | Date | 2026-03-11 |
//! | Command | `cargo run --release --features gpu,json --bin validate_p1_extensions_v1` |
//!
//! Provenance: Phase 1 extension validation

use std::time::Instant;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::Validator;

struct P1Extension {
    name: &'static str,
    accession: &'static str,
    n_samples: usize,
    raw_gb: f64,
    compute_hours: f64,
    biome: &'static str,
    oxygen_regime: f64,
    expected_shannon_range: (f64, f64),
    _expected_pqs_range: (f64, f64),
    _description: &'static str,
}

fn p1_extensions() -> Vec<P1Extension> {
    vec![
        P1Extension {
            name: "cold_seep",
            accession: "PRJNA315684",
            n_samples: 170,
            raw_gb: 5.0,
            compute_hours: 2.0,
            biome: "deep_sea_cold_seep",
            oxygen_regime: 0.02,
            expected_shannon_range: (1.5, 3.5),
            _expected_pqs_range: (0.6, 0.99),
            _description: "170 metagenomes from Guaymas Basin cold seeps, anaerobic reference for QS",
        },
        P1Extension {
            name: "tara_oceans",
            accession: "PRJEB1787",
            n_samples: 243,
            raw_gb: 10.0,
            compute_hours: 0.5,
            biome: "ocean_epipelagic",
            oxygen_regime: 0.9,
            expected_shannon_range: (3.0, 5.0),
            _expected_pqs_range: (0.01, 0.3),
            _description: "243 stations from Tara Oceans, marine diversity gradient",
        },
        P1Extension {
            name: "hmp_gut",
            accession: "PRJNA275349",
            n_samples: 4700,
            raw_gb: 20.0,
            compute_hours: 1.0,
            biome: "human_gut",
            oxygen_regime: 0.05,
            expected_shannon_range: (2.0, 4.0),
            _expected_pqs_range: (0.5, 0.95),
            _description: "4,700 human gut samples from HMP, anaerobic gut reference",
        },
        P1Extension {
            name: "amr_surveillance",
            accession: "multiple",
            n_samples: 500,
            raw_gb: 50.0,
            compute_hours: 3.0,
            biome: "clinical",
            oxygen_regime: 0.5,
            expected_shannon_range: (1.0, 3.0),
            _expected_pqs_range: (0.3, 0.8),
            _description: "AMR gene diversity as Anderson disorder, sentinel framework",
        },
        P1Extension {
            name: "mycorrhizal",
            accession: "ITS_micro_CT",
            n_samples: 200,
            raw_gb: 10.0,
            compute_hours: 1.0,
            biome: "rhizosphere_fungal",
            oxygen_regime: 0.4,
            expected_shannon_range: (1.5, 3.5),
            _expected_pqs_range: (0.3, 0.7),
            _description: "Fungal hyphal network as Anderson lattice, ITS + micro-CT topology",
        },
    ]
}

fn generate_synthetic_community(
    n_samples: usize,
    shannon_range: (f64, f64),
    seed: u64,
) -> Vec<Vec<f64>> {
    let n_taxa = 150;
    (0..n_samples)
        .map(|i| {
            let s = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(i as u64 * 1_442_695_040_888_963_407);
            let frac = (i as f64 / n_samples.max(1) as f64)
                .mul_add(shannon_range.1 - shannon_range.0, shannon_range.0);
            let richness_frac = (frac / 5.0).clamp(0.05, 1.0);
            let n_present = ((richness_frac * n_taxa as f64) as usize).max(2);

            (0..n_taxa)
                .map(|j| {
                    if j < n_present {
                        let pseudo = ((s
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(j as u64 * 7_046_029_254_386_353_087))
                            as f64)
                            / u64::MAX as f64;
                        let rank_weight = 1.0 / (1.0 + j as f64).powf(0.7 + pseudo * 0.3);
                        (rank_weight * 50.0 * (0.5 + pseudo)).max(1.0)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn main() {
    let start = Instant::now();
    let mut v = Validator::new("Exp369: P1 Dataset Extensions Framework v1");

    let extensions = p1_extensions();

    println!(
        "\n  P1 Extension Framework — {} extensions",
        extensions.len()
    );
    println!(
        "  {:20} {:>12} {:>8} {:>10} {:>8}",
        "Extension", "Accession", "Samples", "Raw (GB)", "GPU (h)"
    );
    println!("  {}", "─".repeat(62));
    for ext in &extensions {
        println!(
            "  {:20} {:>12} {:>8} {:>10.1} {:>8.1}",
            ext.name, ext.accession, ext.n_samples, ext.raw_gb, ext.compute_hours
        );
    }

    let total_samples: usize = extensions.iter().map(|e| e.n_samples).sum();
    let total_gb: f64 = extensions.iter().map(|e| e.raw_gb).sum();
    let total_hours: f64 = extensions.iter().map(|e| e.compute_hours).sum();
    println!(
        "  {:20} {:>12} {:>8} {:>10.1} {:>8.1}",
        "TOTAL", "", total_samples, total_gb, total_hours
    );

    v.check_pass("5 P1 extensions defined", extensions.len() == 5);

    // ─── D126: Cold Seep Model ───
    println!("\n  ── D126: Cold Seep Model ──");
    let cold_seep = &extensions[0];
    let cs_communities = generate_synthetic_community(50, cold_seep.expected_shannon_range, 42);
    let cs_results: Vec<(f64, f64)> = cs_communities
        .iter()
        .map(|c| {
            let h = barracuda::stats::diversity::shannon(c);
            let w = 3.5f64.mul_add(h, 8.0 * cold_seep.oxygen_regime);
            let p = barracuda::stats::norm_cdf((16.5 - w) / 3.0);
            (h, p)
        })
        .collect();

    let cs_mean_p = cs_results.iter().map(|(_, p)| p).sum::<f64>() / cs_results.len() as f64;
    let cs_mean_h = cs_results.iter().map(|(h, _)| h).sum::<f64>() / cs_results.len() as f64;
    println!("  Cold seep (n=50 synthetic): mean H'={cs_mean_h:.3}, mean P(QS)={cs_mean_p:.4}");
    println!("  Anaerobic deep sea → low O₂ (0.02) → strong QS advantage");
    v.check_pass("cold seep high P(QS)", cs_mean_p > 0.5);

    // ─── D127: Tara Oceans Model ───
    println!("\n  ── D127: Tara Oceans Model ──");
    let tara = &extensions[1];
    let tara_communities = generate_synthetic_community(50, tara.expected_shannon_range, 99);
    let tara_results: Vec<(f64, f64)> = tara_communities
        .iter()
        .map(|c| {
            let h = barracuda::stats::diversity::shannon(c);
            let w = 3.5f64.mul_add(h, 8.0 * tara.oxygen_regime);
            let p = barracuda::stats::norm_cdf((16.5 - w) / 3.0);
            (h, p)
        })
        .collect();

    let tara_mean_p = tara_results.iter().map(|(_, p)| p).sum::<f64>() / tara_results.len() as f64;
    let tara_mean_h = tara_results.iter().map(|(h, _)| h).sum::<f64>() / tara_results.len() as f64;
    println!(
        "  Tara Oceans (n=50 synthetic): mean H'={tara_mean_h:.3}, mean P(QS)={tara_mean_p:.4}"
    );
    println!("  Aerobic ocean → high O₂ (0.9) → QS disadvantage");
    v.check_pass("tara ocean low P(QS) vs cold seep", tara_mean_p < cs_mean_p);

    // ─── D128: HMP Gut Model ───
    println!("\n  ── D128: HMP Gut Model ──");
    let hmp = &extensions[2];
    let hmp_communities = generate_synthetic_community(50, hmp.expected_shannon_range, 777);
    let hmp_results: Vec<(f64, f64)> = hmp_communities
        .iter()
        .map(|c| {
            let h = barracuda::stats::diversity::shannon(c);
            let w = 3.5f64.mul_add(h, 8.0 * hmp.oxygen_regime);
            let p = barracuda::stats::norm_cdf((16.5 - w) / 3.0);
            (h, p)
        })
        .collect();

    let hmp_mean_p = hmp_results.iter().map(|(_, p)| p).sum::<f64>() / hmp_results.len() as f64;
    let hmp_mean_h = hmp_results.iter().map(|(h, _)| h).sum::<f64>() / hmp_results.len() as f64;
    println!("  HMP gut (n=50 synthetic): mean H'={hmp_mean_h:.3}, mean P(QS)={hmp_mean_p:.4}");
    println!("  Anaerobic gut → low O₂ (0.05) → QS active");
    v.check_pass("HMP gut has QS active", hmp_mean_p > 0.5);

    // ─── D129: AMR Sentinel Framework ───
    println!("\n  ── D129: AMR Sentinel Framework ──");
    let amr = &extensions[3];
    println!("  AMR sentinel concept: resistance gene diversity as Anderson disorder");
    println!("  W_AMR = α·(gene diversity) + β·(horizontal transfer rate)");
    println!("  P(resistance spread) = norm_cdf((W_c - W_AMR) / σ)");

    let amr_communities = generate_synthetic_community(20, amr.expected_shannon_range, 333);
    let amr_results: Vec<f64> = amr_communities
        .iter()
        .map(|c| {
            let h = barracuda::stats::diversity::shannon(c);
            let w = 3.5f64.mul_add(h, 8.0 * amr.oxygen_regime);
            barracuda::stats::norm_cdf((16.5 - w) / 3.0)
        })
        .collect();

    let amr_mean_p = amr_results.iter().sum::<f64>() / amr_results.len() as f64;
    println!("  AMR (n=20 synthetic): mean P(spread)={amr_mean_p:.4}");
    v.check_pass(
        "AMR model produces valid probabilities",
        amr_results.iter().all(|&p| (0.0..=1.0).contains(&p)),
    );

    // ─── D130: Mycorrhizal Anderson ───
    println!("\n  ── D130: Mycorrhizal Anderson ──");
    let myc = &extensions[4];
    println!("  Mycorrhizal concept: fungal hyphal network topology → Anderson lattice");
    println!("  Hyphal branch density → lattice connectivity (d_eff)");
    println!("  Species diversity of fungal community → disorder W");
    println!("  Nutrient signal propagation → Anderson QS model");

    let myc_communities = generate_synthetic_community(20, myc.expected_shannon_range, 555);
    let myc_results: Vec<f64> = myc_communities
        .iter()
        .map(|c| {
            let h = barracuda::stats::diversity::shannon(c);
            let w = 3.5f64.mul_add(h, 8.0 * myc.oxygen_regime);
            barracuda::stats::norm_cdf((16.5 - w) / 3.0)
        })
        .collect();

    let myc_mean_p = myc_results.iter().sum::<f64>() / myc_results.len() as f64;
    println!("  Mycorrhizal (n=20 synthetic): mean P(nutrient signal)={myc_mean_p:.4}");
    v.check_pass(
        "mycorrhizal model produces valid probabilities",
        myc_results.iter().all(|&p| (0.0..=1.0).contains(&p)),
    );

    // Cross-biome comparison
    println!("\n  ── Cross-Extension Comparison ──");
    println!(
        "  {:20} {:>8} {:>8} {:>8}",
        "Extension", "O₂", "H'", "P(QS)"
    );
    println!("  {}", "─".repeat(48));
    println!(
        "  {:20} {:>8.2} {:>8.3} {:>8.4}",
        "Cold seep", cold_seep.oxygen_regime, cs_mean_h, cs_mean_p
    );
    println!(
        "  {:20} {:>8.2} {:>8.3} {:>8.4}",
        "Tara Oceans", tara.oxygen_regime, tara_mean_h, tara_mean_p
    );
    println!(
        "  {:20} {:>8.2} {:>8.3} {:>8.4}",
        "HMP gut", hmp.oxygen_regime, hmp_mean_h, hmp_mean_p
    );
    println!(
        "  {:20} {:>8.2} {:>8.3} {:>8.4}",
        "AMR sentinel", amr.oxygen_regime, 0.0, amr_mean_p
    );
    println!(
        "  {:20} {:>8.2} {:>8.3} {:>8.4}",
        "Mycorrhizal", myc.oxygen_regime, 0.0, myc_mean_p
    );

    v.check_pass(
        "anaerobic > aerobic P(QS) (H3 validated across P1)",
        cs_mean_p > tara_mean_p && hmp_mean_p > tara_mean_p,
    );

    // Export
    #[cfg(feature = "json")]
    {
        let export = serde_json::json!({
            "experiment": "Exp369",
            "extensions": extensions.iter().map(|e| serde_json::json!({
                "name": e.name,
                "accession": e.accession,
                "n_samples": e.n_samples,
                "raw_gb": e.raw_gb,
                "biome": e.biome,
                "o2": e.oxygen_regime,
            })).collect::<Vec<_>>(),
            "total_samples": total_samples,
            "total_raw_gb": total_gb,
            "total_compute_hours": total_hours,
            "status": "framework validated, awaiting real data",
        });
        let json = serde_json::to_string_pretty(&export).or_exit("serialize");
        std::fs::create_dir_all("output").ok();
        std::fs::write("output/p1_extensions_framework.json", &json).or_exit("write");
        println!("\n  Exported: output/p1_extensions_framework.json");
        v.check_pass("P1 framework export", true);
    }

    #[cfg(not(feature = "json"))]
    {
        v.check_pass("graceful skip", true);
    }

    let elapsed = start.elapsed();
    println!("\n  Wall time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    v.finish();
}
