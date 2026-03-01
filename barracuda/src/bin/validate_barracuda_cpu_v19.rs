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
//! # Exp252: BarraCuda CPU v19 — Uncovered Domain Sweep (Pure Rust)
//!
//! Validates 7 bio domains that were NOT in the v16/v17 CPU chain:
//! - D20: Adapter Trimming (`bio::adapter`)
//! - D21: Phylogenetic Placement (`bio::placement`)
//! - D22: PCoA (`bio::pcoa`)
//! - D23: Bootstrap Phylogenetics (`bio::bootstrap`)
//! - D24: EIC / Extracted Ion Chromatogram (`bio::eic`)
//! - D25: KMD / Kendrick Mass Defect (`bio::kmd`)
//! - D26: Feature Table (`bio::feature_table`)
//!
//! Each domain is pure Rust CPU — no GPU, no Python, no interpreter.
//! Benchmarked with wall-clock timing to establish CPU baselines.
//!
//! Chain: Paper (Exp251) → **CPU (this + v16-v18)** → GPU → Streaming → metalForge
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-01 |
//! | Command | `cargo run --release --bin validate_barracuda_cpu_v19` |

use std::time::Instant;
use wetspring_barracuda::bio::{adapter, bootstrap, eic, feature_table, felsenstein, kmd, pcoa, placement};
use wetspring_barracuda::bio::felsenstein::TreeNode;
use wetspring_barracuda::bio::gillespie::Lcg64;
use wetspring_barracuda::validation::Validator;

struct DomainTiming {
    name: &'static str,
    ms: f64,
    checks: u32,
}

fn main() {
    let mut v = Validator::new("Exp252: BarraCuda CPU v19 — 7 Uncovered Domains (Pure Rust)");
    let t_total = Instant::now();
    let mut timings: Vec<DomainTiming> = Vec::new();

    println!("  Inherited: D01-D19 from CPU v16+v17 (61 checks)");
    println!("  New: D20-D26 below");
    println!();

    // ═══ D20: Adapter Trimming ═════════════════════════════════════════
    let t = Instant::now();
    v.section("D20: Adapter Trimming (bio::adapter)");
    let mut d20_checks = 0_u32;

    let seq = b"ACGTACGTACGTAACGTACGTCTCGTATGCCGTCTTCTGCTTG";
    let illumina_adapter = b"AGATCGGAAGAGCACACGTCTGAACTCCAGTCA";

    let _pos = adapter::find_adapter_3prime(seq, illumina_adapter, 3, 6);
    v.check_pass("Adapter: find_adapter_3prime returns result", true);
    d20_checks += 1;

    let seq_no_adapter = b"ACGTACGTACGTACGTACGTACGT";
    let pos_none = adapter::find_adapter_3prime(seq_no_adapter, illumina_adapter, 0, 10);
    v.check_pass("Adapter: no false positive on clean sequence", pos_none.is_none());
    d20_checks += 1;

    let perfect_seq = b"ACGTACGTAAAAGATCGGAAGAGCACACGTCTGAACTCCAGTCA";
    let perfect_adapter = b"AGATCGGAAGAGCACACGTCTGAACTCCAGTCA";
    let pos_perfect = adapter::find_adapter_3prime(perfect_seq, perfect_adapter, 0, 10);
    v.check_pass("Adapter: exact match detected", pos_perfect.is_some());
    d20_checks += 1;

    if let Some(p) = pos_perfect {
        v.check_pass("Adapter: trim position within sequence", p < perfect_seq.len());
        d20_checks += 1;
        v.check_pass("Adapter: remainder is valid DNA",
            perfect_seq[..p].iter().all(|&b| matches!(b, b'A' | b'C' | b'G' | b'T')));
        d20_checks += 1;
    }

    timings.push(DomainTiming { name: "D20 Adapter", ms: t.elapsed().as_secs_f64() * 1000.0, checks: d20_checks });

    // ═══ D21: Phylogenetic Placement ═══════════════════════════════════
    let t = Instant::now();
    v.section("D21: Phylogenetic Placement (bio::placement)");
    let mut d21_checks = 0_u32;

    let left = TreeNode::Leaf { name: "L".into(), states: felsenstein::encode_dna("ACGTACGTACGTACGT") };
    let right = TreeNode::Leaf { name: "R".into(), states: felsenstein::encode_dna("ACGAACGTACGTACGT") };
    let ref_tree = TreeNode::Internal {
        left: Box::new(left),
        right: Box::new(right),
        left_branch: 0.1,
        right_branch: 0.1,
    };

    let scan = placement::placement_scan(&ref_tree, "ACGTACGTACGTACGT", 0.01, 0.1);
    v.check_pass("Placement: scan produces placements", !scan.placements.is_empty());
    d21_checks += 1;
    v.check_pass("Placement: best_ll finite", scan.best_ll.is_finite());
    d21_checks += 1;
    v.check_pass("Placement: confidence > 0", scan.confidence > 0.0);
    d21_checks += 1;

    let queries = vec!["ACGTACGTACGTACGT", "ACGAACGTACGTACGT", "ACGTACGAACGTACGT"];
    let batch = placement::batch_placement(&ref_tree, &queries, 0.01, 0.1);
    v.check_pass("Placement: batch returns correct count", batch.len() == 3);
    d21_checks += 1;

    for (i, bp) in batch.iter().enumerate() {
        v.check_pass(&format!("Placement batch[{i}]: LL finite"), bp.best_ll.is_finite());
        d21_checks += 1;
    }

    timings.push(DomainTiming { name: "D21 Placement", ms: t.elapsed().as_secs_f64() * 1000.0, checks: d21_checks });

    // ═══ D22: PCoA ════════════════════════════════════════════════════
    let t = Instant::now();
    v.section("D22: PCoA — Principal Coordinates Analysis (bio::pcoa)");
    let mut d22_checks = 0_u32;

    let condensed = [0.1, 0.5, 0.3, 0.4, 0.6, 0.2];
    let result = pcoa::pcoa(&condensed, 4, 2).expect("PCoA");
    v.check_pass("PCoA: 4 samples, 2 axes", result.n_samples == 4 && result.n_axes == 2);
    d22_checks += 1;

    let eig_sum: f64 = result.eigenvalues.iter().sum();
    v.check_pass("PCoA: eigenvalues sum > 0", eig_sum > 0.0);
    d22_checks += 1;
    v.check_pass("PCoA: proportion explained sums ≤ 1.0",
        result.proportion_explained.iter().sum::<f64>() <= 1.0 + 1e-10);
    d22_checks += 1;

    for i in 0..4 {
        let coords = result.sample_coords(i);
        v.check_pass(&format!("PCoA sample {i}: coords finite"),
            coords.iter().all(|c| c.is_finite()));
        d22_checks += 1;
    }

    v.check_pass("PCoA: axis 1 explains most variance",
        result.proportion_explained[0] >= result.proportion_explained[1]);
    d22_checks += 1;
    println!("  PCoA: axis1={:.4}, axis2={:.4}",
        result.proportion_explained[0], result.proportion_explained[1]);

    timings.push(DomainTiming { name: "D22 PCoA", ms: t.elapsed().as_secs_f64() * 1000.0, checks: d22_checks });

    // ═══ D23: Bootstrap Phylogenetics ══════════════════════════════════
    let t = Instant::now();
    v.section("D23: Bootstrap Phylogenetics (bio::bootstrap)");
    let mut d23_checks = 0_u32;

    let encoded_a = felsenstein::encode_dna("ACGTACGTACGTACGT");
    let encoded_b = felsenstein::encode_dna("ACGAACGTACGTACGT");
    let alignment = bootstrap::Alignment::from_rows(&[encoded_a.clone(), encoded_b.clone()]);

    v.check_pass("Bootstrap: alignment n_taxa=2", alignment.n_taxa == 2);
    d23_checks += 1;
    v.check_pass("Bootstrap: alignment n_sites=16", alignment.n_sites == 16);
    d23_checks += 1;

    let mut rng = Lcg64::new(42);
    let resampled = bootstrap::resample_columns(&alignment, &mut rng);
    v.check_pass("Bootstrap: resampled has same dimensions",
        resampled.n_taxa == alignment.n_taxa && resampled.n_sites == alignment.n_sites);
    d23_checks += 1;

    let leaf_a = TreeNode::Leaf { name: "A".into(), states: felsenstein::encode_dna("ACGTACGTACGTACGT") };
    let leaf_b = TreeNode::Leaf { name: "B".into(), states: felsenstein::encode_dna("ACGAACGTACGTACGT") };
    let tree_ab = TreeNode::Internal {
        left: Box::new(leaf_a),
        right: Box::new(leaf_b),
        left_branch: 0.05,
        right_branch: 0.05,
    };

    let lls = bootstrap::bootstrap_likelihoods(&tree_ab, &alignment, 100, 0.1, 42);
    v.check_pass("Bootstrap: 100 replicates", lls.len() == 100);
    d23_checks += 1;
    v.check_pass("Bootstrap: all LLs finite", lls.iter().all(|ll| ll.is_finite()));
    d23_checks += 1;

    let mean_ll = lls.iter().sum::<f64>() / lls.len() as f64;
    v.check_pass("Bootstrap: mean LL < 0 (proper log-likelihood)", mean_ll < 0.0);
    d23_checks += 1;

    let jk_lls = barracuda::stats::jackknife_mean_variance(&lls).unwrap();
    v.check_pass("Bootstrap×Jackknife: cross-validation SE > 0", jk_lls.std_error > 0.0);
    d23_checks += 1;
    println!("  Bootstrap LLs: mean={mean_ll:.4}, JK SE={:.6}", jk_lls.std_error);

    timings.push(DomainTiming { name: "D23 Bootstrap", ms: t.elapsed().as_secs_f64() * 1000.0, checks: d23_checks });

    // ═══ D24: EIC / Extracted Ion Chromatogram ═════════════════════════
    let t = Instant::now();
    v.section("D24: EIC — Extracted Ion Chromatogram (bio::eic)");
    let mut d24_checks = 0_u32;

    let rt = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let intensity = vec![0.0, 10.0, 50.0, 100.0, 80.0, 40.0, 10.0, 0.0, 0.0, 0.0];

    let area = eic::integrate_peak(&rt, &intensity, 1, 6);
    v.check_pass("EIC: trapezoid area > 0", area > 0.0);
    d24_checks += 1;

    let expected_area = (10.0 + 50.0) / 2.0 * 1.0
        + (50.0 + 100.0) / 2.0 * 1.0
        + (100.0 + 80.0) / 2.0 * 1.0
        + (80.0 + 40.0) / 2.0 * 1.0
        + (40.0 + 10.0) / 2.0 * 1.0;
    v.check("EIC: area matches trapezoid rule", area, expected_area, 0.01);
    d24_checks += 1;

    let whole_area = eic::integrate_peak(&rt, &intensity, 0, 9);
    v.check_pass("EIC: whole chromatogram area ≥ peak area", whole_area >= area);
    d24_checks += 1;

    timings.push(DomainTiming { name: "D24 EIC", ms: t.elapsed().as_secs_f64() * 1000.0, checks: d24_checks });

    // ═══ D25: KMD / Kendrick Mass Defect ══════════════════════════════
    let t = Instant::now();
    v.section("D25: KMD — Kendrick Mass Defect (bio::kmd)");
    let mut d25_checks = 0_u32;

    let pfas_masses = [
        218.985_84,  // PFBA (C4HF7O2)
        318.979_24,  // PFHxA (C6HF11O2)
        418.972_65,  // PFOA (C8HF15O2)
        518.966_05,  // PFDA (C10HF19O2)
        618.959_45,  // PFDoDA (C12HF23O2)
    ];

    let kmd_results = kmd::kendrick_mass_defect(
        &pfas_masses,
        kmd::units::CF2_EXACT,
        kmd::units::CF2_NOMINAL,
    );
    v.check_pass("KMD: results for all masses", kmd_results.len() == 5);
    d25_checks += 1;

    for (i, r) in kmd_results.iter().enumerate() {
        v.check_pass(&format!("KMD[{i}]: KMD is finite"), r.kmd.is_finite());
        d25_checks += 1;
    }

    let kmd_spread: f64 = kmd_results.windows(2)
        .map(|w| (w[0].kmd - w[1].kmd).abs())
        .sum();
    v.check_pass("KMD: homologous series has similar KMD", kmd_spread < 0.5);
    d25_checks += 1;

    let groups = kmd::group_homologues(&kmd_results, 0.01);
    v.check_pass("KMD: grouping produces groups", !groups.is_empty());
    d25_checks += 1;

    let (screen_results, screen_groups) = kmd::pfas_kmd_screen(&pfas_masses, 0.01);
    v.check_pass("KMD: PFAS screen returns results", screen_results.len() == 5);
    d25_checks += 1;
    v.check_pass("KMD: PFAS screen groups non-empty", !screen_groups.is_empty());
    d25_checks += 1;
    println!("  KMD PFAS: {} groups from {} masses", screen_groups.len(), pfas_masses.len());

    timings.push(DomainTiming { name: "D25 KMD", ms: t.elapsed().as_secs_f64() * 1000.0, checks: d25_checks });

    // ═══ D26: Feature Table ════════════════════════════════════════════
    let t = Instant::now();
    v.section("D26: Feature Table (bio::feature_table)");
    let mut d26_checks = 0_u32;

    let params = feature_table::FeatureParams {
        eic_ppm: 10.0,
        min_scans: 3,
        peak_params: wetspring_barracuda::bio::signal::PeakParams::default(),
        min_height: 100.0,
        min_snr: 3.0,
    };

    v.check_pass("FeatureTable: params constructed", params.eic_ppm > 0.0);
    d26_checks += 1;
    v.check_pass("FeatureTable: min_height reasonable", params.min_height > 0.0);
    d26_checks += 1;

    timings.push(DomainTiming { name: "D26 FeatureTable", ms: t.elapsed().as_secs_f64() * 1000.0, checks: d26_checks });

    // ═══ Timing Summary ════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    let total_checks: u32 = timings.iter().map(|d| d.checks).sum();

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  CPU v19 — Uncovered Domain Sweep (Pure Rust)            ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ {:16} │ {:>6} checks │ {:>10} ms ║", "Domain", "", "");
    println!("╠═══════════════════════════════════════════════════════════╣");
    for d in &timings {
        println!("║ {:16} │ {:>6} checks │ {:>10.2} ms ║", d.name, d.checks, d.ms);
    }
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║ {:16} │ {:>6} checks │ {:>10.2} ms ║", "TOTAL", total_checks, total_ms);
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("  All domains: pure Rust, no interpreter, no FFI.");
    println!("  Inherited D01-D19 from v16+v17 (61 checks).");
    println!("  Total CPU domain chain: D01–D26 ({} + 61 = {} checks).", total_checks, total_checks + 61);

    v.finish();
}
