// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate peak detection and retention index matching against Reese 2019 (Exp013).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Paper | Reese et al. 2019. Sci. Rep. 9:13866 (PMC6761164) |
//! | DOI | 10.1038/s41598-019-50125-z |
//! | Baseline | Table 1: 14 VOC compounds (m/z, RI, NIST ID, match %) |
//! | Baseline file | `experiments/results/013_voc_baselines/reese2019_table1.tsv` |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --bin validate_voc_peaks` |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! # Methodology
//!
//! The Reese paper provides retention indices and base peak masses rather than
//! raw chromatograms. We validate:
//!
//! 1. **TSV parsing**: Baseline file loads correctly with all 14 compounds
//! 2. **RI deviation**: Experimental vs theoretical RI within 5% (paper criterion)
//! 3. **Synthetic chromatogram**: Generate GC-MS-like signal with peaks at reported
//!    retention indices; verify `bio::signal::find_peaks` detects all 7 A+R biomarkers
//! 4. **Biomarker classification**: Separate A+R-only from A+R,A compounds
//! 5. **Tolerance matching**: Use `bio::tolerance_search` for RI matching

use std::path::Path;
use wetspring_barracuda::bio::signal::{find_peaks, PeakParams};
use wetspring_barracuda::bio::tolerance_search;
use wetspring_barracuda::validation::{self, Validator};

#[derive(Debug)]
struct VocCompound {
    id: u32,
    base_peak_mz: f64,
    tentative_class: String,
    #[allow(dead_code)]
    nist_id: String,
    nist_match_pct: u32,
    experimental_ri: f64,
    theoretical_ri: f64,
    condition: String,
    n_experiments: u32,
}

fn parse_baseline(path: &Path) -> Vec<VocCompound> {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read {}: {e}", path.display());
            std::process::exit(1);
        }
    };

    let mut compounds = Vec::new();
    for line in contents.lines() {
        if line.starts_with('#') || line.starts_with("compound_id") || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 9 {
            continue;
        }
        compounds.push(VocCompound {
            id: fields[0].trim().parse().unwrap_or(0),
            base_peak_mz: fields[1].trim().parse().unwrap_or(0.0),
            tentative_class: fields[2].trim().to_string(),
            nist_id: fields[3].trim().to_string(),
            nist_match_pct: fields[4].trim().parse().unwrap_or(0),
            experimental_ri: fields[5].trim().parse().unwrap_or(0.0),
            theoretical_ri: fields[6].trim().parse().unwrap_or(0.0),
            condition: fields[7].trim().to_string(),
            n_experiments: fields[8].trim().parse().unwrap_or(0),
        });
    }
    compounds
}

fn main() {
    let mut v = Validator::new("wetSpring VOC Peak Validation (Exp013 — Reese 2019)");

    let baseline_path = validation::data_dir(
        "WETSPRING_VOC_DIR",
        "experiments/results/013_voc_baselines/reese2019_table1.tsv",
    );

    if !baseline_path.exists() {
        println!(
            "  NOTE: VOC baseline not found at {}\n  \
             Extract from PMC6761164 Table 1 first.",
            baseline_path.display()
        );
        v.finish();
    }

    let compounds = parse_baseline(&baseline_path);
    validate_parsing(&mut v, &compounds);
    validate_ri_deviation(&mut v, &compounds);
    validate_synthetic_chromatogram(&mut v, &compounds);
    validate_biomarker_classification(&mut v, &compounds);
    validate_ri_tolerance_matching(&mut v, &compounds);

    v.finish();
}

fn validate_parsing(v: &mut Validator, compounds: &[VocCompound]) {
    v.section("TSV Baseline Parsing");

    v.check_count("Total compounds", compounds.len(), 14);

    v.check_count(
        "A+R only compounds (grazer biomarkers)",
        compounds.iter().filter(|c| c.condition == "A+R").count(),
        7,
    );
    v.check_count(
        "A+R,A compounds (both conditions)",
        compounds.iter().filter(|c| c.condition == "A+R,A").count(),
        7,
    );
    v.check_count(
        "NIST-identified compounds",
        compounds.iter().filter(|c| c.nist_match_pct > 0).count(),
        5,
    );
    v.check_count(
        "Compounds in all 3 experiments",
        compounds.iter().filter(|c| c.n_experiments == 3).count(),
        6,
    );

    let trans_beta_ionone = compounds.iter().find(|c| c.id == 6);
    if let Some(tbi) = trans_beta_ionone {
        v.check("trans-beta-ionone m/z = 177", tbi.base_peak_mz, 177.0, 0.0);
        v.check(
            "trans-beta-ionone RI = 1495",
            tbi.experimental_ri,
            1495.0,
            0.0,
        );
        v.check_count(
            "trans-beta-ionone NIST match = 94%",
            tbi.nist_match_pct as usize,
            94,
        );
    }
}

fn validate_ri_deviation(v: &mut Validator, compounds: &[VocCompound]) {
    v.section("Retention Index Deviation (<5% per paper)");

    let with_theoretical: Vec<_> = compounds
        .iter()
        .filter(|c| c.theoretical_ri > 0.0)
        .collect();

    v.check_count("Compounds with theoretical RI", with_theoretical.len(), 6);

    for c in &with_theoretical {
        let deviation_pct =
            ((c.experimental_ri - c.theoretical_ri) / c.theoretical_ri).abs() * 100.0;
        v.check(
            &format!(
                "Compound {} RI deviation ({:.0} vs {:.0})",
                c.id, c.experimental_ri, c.theoretical_ri
            ),
            deviation_pct,
            0.0,
            5.0, // <5% tolerance as stated in the paper
        );
    }
}

fn validate_synthetic_chromatogram(v: &mut Validator, compounds: &[VocCompound]) {
    v.section("Synthetic GC-MS Chromatogram Peak Detection");

    let ar_only: Vec<_> = compounds.iter().filter(|c| c.condition == "A+R").collect();

    // Generate a synthetic chromatogram with peaks at reported RIs.
    // RI range: ~1000 to ~1700. Map to array indices proportionally.
    let ri_min = 900.0_f64;
    let ri_max = 1800.0_f64;
    let n_points = 2000_usize;

    let mut chromatogram = vec![5.0_f64; n_points]; // baseline noise

    let mut expected_peak_indices = Vec::new();
    for c in &ar_only {
        let frac = (c.experimental_ri - ri_min) / (ri_max - ri_min);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let center = (frac * n_points as f64).round() as usize;
        let center = center.min(n_points - 1);
        expected_peak_indices.push(center);

        // Gaussian peak: height proportional to experiment reproducibility
        let height = 1000.0 * (f64::from(c.n_experiments) / 3.0);
        let sigma = 8.0;
        let lo = center.saturating_sub(40);
        let hi = (center + 40).min(n_points - 1);
        #[allow(clippy::cast_precision_loss)]
        for (i, val) in chromatogram[lo..=hi].iter_mut().enumerate() {
            let x = (lo + i) as f64 - center as f64;
            *val += height * (-0.5 * (x / sigma).powi(2)).exp();
        }
    }

    let params = PeakParams {
        min_height: Some(100.0),
        min_prominence: Some(50.0),
        ..PeakParams::default()
    };

    let peaks = find_peaks(&chromatogram, &params);

    println!(
        "  Synthetic chromatogram: {} points, {} expected peaks, {} found peaks",
        n_points,
        ar_only.len(),
        peaks.len()
    );

    v.check_count("Synthetic peaks detected", peaks.len(), ar_only.len());

    let mut matched = 0_usize;
    for &expected_idx in &expected_peak_indices {
        #[allow(clippy::cast_possible_wrap)]
        let found = peaks
            .iter()
            .any(|p| (p.index as i64 - expected_idx as i64).unsigned_abs() <= 3);
        if found {
            matched += 1;
        }
    }
    v.check_count(
        "Peak indices matched (±3)",
        matched,
        expected_peak_indices.len(),
    );
}

fn validate_biomarker_classification(v: &mut Validator, compounds: &[VocCompound]) {
    v.section("Biomarker Classification");

    let carotenoids: Vec<_> = compounds
        .iter()
        .filter(|c| c.tentative_class.contains("Carotenoid"))
        .collect();
    v.check_count("Carotenoid compounds", carotenoids.len(), 5);

    v.check_count(
        "Grazer-specific carotenoids (A+R)",
        carotenoids.iter().filter(|c| c.condition == "A+R").count(),
        4,
    );

    let all_ar_only: Vec<_> = compounds.iter().filter(|c| c.condition == "A+R").collect();
    let all_ar_in_3 = all_ar_only.iter().all(|c| c.n_experiments >= 2);
    v.check(
        "All A+R markers in >=2 experiments",
        if all_ar_in_3 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
}

fn validate_ri_tolerance_matching(v: &mut Validator, compounds: &[VocCompound]) {
    v.section("RI Tolerance Search Matching");

    let with_theoretical: Vec<_> = compounds
        .iter()
        .filter(|c| c.theoretical_ri > 0.0)
        .collect();

    // Build a sorted array of theoretical RIs
    let mut sorted_theoretical: Vec<f64> =
        with_theoretical.iter().map(|c| c.theoretical_ri).collect();
    sorted_theoretical.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // For each compound, check if its experimental RI matches the correct theoretical
    // value using Da-style absolute tolerance search (RI units, not Da, but same algorithm)
    let mut all_matched = true;
    for c in &with_theoretical {
        let tolerance = c.theoretical_ri * 0.05; // 5% tolerance
        let matches =
            tolerance_search::find_within_da(&sorted_theoretical, c.experimental_ri, tolerance);

        let correct_found = matches
            .iter()
            .any(|&idx| (sorted_theoretical[idx] - c.theoretical_ri).abs() < 0.1);

        if !correct_found {
            all_matched = false;
            println!(
                "  MISMATCH: Compound {} exp RI {:.0} did not match theoretical {:.0}",
                c.id, c.experimental_ri, c.theoretical_ri
            );
        }
    }

    v.check(
        "All theoretical RIs found by tolerance search",
        if all_matched { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Verify that a wildly wrong RI does NOT match
    let bogus_matches = tolerance_search::find_within_da(&sorted_theoretical, 5000.0, 100.0);
    v.check_count("Bogus RI (5000) yields no match", bogus_matches.len(), 0);
}
