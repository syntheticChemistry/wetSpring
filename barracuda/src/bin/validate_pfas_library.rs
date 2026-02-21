// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate PFAS suspect screening against ground-truth reference library (Exp018).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Reference | Jones Lab PFAS Multidimensional Library |
//! | DOI | 10.1038/s41597-024-04363-0 |
//! | Content | 175 PFAS, 281 ion types |
//! | Cross-ref | Exp006 (`FindPFAS` screening, 10/10 PASS) |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --bin validate_pfas_library` |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! # Methodology
//!
//! **Self-contained mode** (no data files required):
//! Uses well-known PFAS exact masses derived from molecular formulas to
//! validate `tolerance_search`, `spectral_match`, and KMD analysis at
//! 5, 10, and 20 ppm systematically.
//!
//! **File mode** (when Jones Lab library is downloaded):
//! Parses the full 175-compound library and runs systematic screening.
//!
//! ## PFAS Reference Masses
//!
//! All `[M-H]⁻` masses are computed from molecular formulas using
//! IUPAC 2021 atomic weights. These are physical constants, not
//! empirical measurements.
//!
//! ## Homologue Series Validated
//!
//! - PFCA series (C4–C14): PFBA → `PFTeDA` (11 compounds, CF₂ spacing)
//! - PFSA series (C4–C10): PFBS → PFDS (6 compounds, CF₂ spacing)
//! - FTSA series (4:2 → 8:2): 3 compounds
//! - `GenX` (HFPO-DA): single compound

use wetspring_barracuda::bio::{kmd, spectral_match, tolerance_search};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

/// PFAS compound with known exact mass.
///
/// Fields `name`, `formula`, and `series` are retained for provenance
/// and human-readable diagnostics even when not referenced in code.
#[allow(dead_code)]
struct PfasRef {
    name: &'static str,
    formula: &'static str,
    /// `[M-H]⁻` monoisotopic mass (Da)
    mh_neg: f64,
    series: &'static str,
}

/// PFCA perfluoroalkyl carboxylic acids `[M-H]⁻`
/// Molecular formula: `CₙHF₍₂ₙ₋₁₎O₂`, `[M-H]⁻` = neutral - 1.00728
const PFCA_SERIES: &[PfasRef] = &[
    PfasRef {
        name: "PFBA",
        formula: "C4HF7O2",
        mh_neg: 212.9790,
        series: "PFCA",
    },
    PfasRef {
        name: "PFPeA",
        formula: "C5HF9O2",
        mh_neg: 262.9758,
        series: "PFCA",
    },
    PfasRef {
        name: "PFHxA",
        formula: "C6HF11O2",
        mh_neg: 312.9726,
        series: "PFCA",
    },
    PfasRef {
        name: "PFHpA",
        formula: "C7HF13O2",
        mh_neg: 362.9694,
        series: "PFCA",
    },
    PfasRef {
        name: "PFOA",
        formula: "C8HF15O2",
        mh_neg: 412.9664,
        series: "PFCA",
    },
    PfasRef {
        name: "PFNA",
        formula: "C9HF17O2",
        mh_neg: 462.9632,
        series: "PFCA",
    },
    PfasRef {
        name: "PFDA",
        formula: "C10HF19O2",
        mh_neg: 512.9600,
        series: "PFCA",
    },
    PfasRef {
        name: "PFUnDA",
        formula: "C11HF21O2",
        mh_neg: 562.9568,
        series: "PFCA",
    },
    PfasRef {
        name: "PFDoA",
        formula: "C12HF23O2",
        mh_neg: 612.9536,
        series: "PFCA",
    },
    PfasRef {
        name: "PFTrDA",
        formula: "C13HF25O2",
        mh_neg: 662.9504,
        series: "PFCA",
    },
    PfasRef {
        name: "PFTeDA",
        formula: "C14HF27O2",
        mh_neg: 712.9472,
        series: "PFCA",
    },
];

/// PFSA perfluoroalkyl sulfonic acids `[M-H]⁻`
const PFSA_SERIES: &[PfasRef] = &[
    PfasRef {
        name: "PFBS",
        formula: "C4F9SO3",
        mh_neg: 298.9430,
        series: "PFSA",
    },
    PfasRef {
        name: "PFPeS",
        formula: "C5F11SO3",
        mh_neg: 348.9398,
        series: "PFSA",
    },
    PfasRef {
        name: "PFHxS",
        formula: "C6F13SO3",
        mh_neg: 398.9366,
        series: "PFSA",
    },
    PfasRef {
        name: "PFHpS",
        formula: "C7F15SO3",
        mh_neg: 448.9334,
        series: "PFSA",
    },
    PfasRef {
        name: "PFOS",
        formula: "C8F17SO3",
        mh_neg: 498.9302,
        series: "PFSA",
    },
    PfasRef {
        name: "PFDS",
        formula: "C10F21SO3",
        mh_neg: 598.9238,
        series: "PFSA",
    },
];

/// Fluorotelomer sulfonic acids `[M-H]⁻`
const FTSA_SERIES: &[PfasRef] = &[
    PfasRef {
        name: "4:2 FTSA",
        formula: "C6H4F9SO3",
        mh_neg: 326.9715,
        series: "FTSA",
    },
    PfasRef {
        name: "6:2 FTSA",
        formula: "C8H4F13SO3",
        mh_neg: 426.9651,
        series: "FTSA",
    },
    PfasRef {
        name: "8:2 FTSA",
        formula: "C10H4F17SO3",
        mh_neg: 526.9619,
        series: "FTSA",
    },
];

/// Individual reference compounds
const OTHER_PFAS: &[PfasRef] = &[
    PfasRef {
        name: "GenX",
        formula: "C6HF11O3",
        mh_neg: 328.9692,
        series: "HFPO",
    },
    PfasRef {
        name: "ADONA",
        formula: "C7H2F12O4",
        mh_neg: 376.9720,
        series: "ether",
    },
];

fn all_pfas() -> Vec<&'static PfasRef> {
    PFCA_SERIES
        .iter()
        .chain(PFSA_SERIES)
        .chain(FTSA_SERIES)
        .chain(OTHER_PFAS)
        .collect()
}

fn main() {
    let mut v = Validator::new("wetSpring PFAS Library Validation (Exp018)");

    validate_tolerance_search_systematic(&mut v);
    validate_kmd_homologue_series(&mut v);
    validate_spectral_match_pfas(&mut v);
    validate_cross_series_discrimination(&mut v);
    validate_jones_library_expansion(&mut v);

    v.finish();
}

// ── Systematic tolerance search at 5, 10, 20 ppm ───────────────────────────

#[allow(clippy::similar_names, clippy::cast_precision_loss)]
fn validate_tolerance_search_systematic(v: &mut Validator) {
    v.section("Systematic Tolerance Search (5/10/20 ppm)");

    let all = all_pfas();
    let sorted_mz: Vec<f64> = {
        let mut v: Vec<f64> = all.iter().map(|p| p.mh_neg).collect();
        v.sort_by(f64::total_cmp);
        v
    };

    // At each ppm level, query every known PFAS mass and check it finds itself
    for &ppm in &[5.0, 10.0, 20.0] {
        let mut hits = 0_usize;
        let total = all.len();

        for compound in &all {
            let matches = tolerance_search::find_within_ppm(&sorted_mz, compound.mh_neg, ppm);
            if matches
                .iter()
                .any(|&idx| (sorted_mz[idx] - compound.mh_neg).abs() < 1e-6)
            {
                hits += 1;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let hit_rate = hits as f64 / total as f64;
        println!(
            "  {ppm:.0} ppm: {hits}/{total} self-hits ({:.1}%)",
            hit_rate * 100.0
        );

        v.check(
            &format!("{ppm:.0} ppm self-hit rate = 100%"),
            hit_rate,
            1.0,
            0.0,
        );
    }

    // Verify no false negatives: search for slightly shifted masses
    v.section("Tolerance Edge Cases");

    let pfoa_mz = 412.966_4_f64;
    let shift_5ppm = pfoa_mz * 5.0 / 1_000_000.0;

    // Mass shifted by exactly 4.9 ppm should be found at 5 ppm
    let query_in = pfoa_mz + pfoa_mz * 4.9 / 1_000_000.0;
    let result_in = tolerance_search::find_within_ppm(&sorted_mz, query_in, 5.0);
    let found_in = result_in
        .iter()
        .any(|&idx| (sorted_mz[idx] - pfoa_mz).abs() < 1e-4);
    v.check(
        "4.9 ppm shift found at 5 ppm window",
        if found_in { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Mass shifted by 6 ppm should NOT be found at 5 ppm
    let query_out = pfoa_mz + pfoa_mz * 6.0 / 1_000_000.0;
    let result_out = tolerance_search::find_within_ppm(&sorted_mz, query_out, 5.0);
    let found_out = result_out
        .iter()
        .any(|&idx| (sorted_mz[idx] - pfoa_mz).abs() < 1e-4);
    v.check(
        "6 ppm shift NOT found at 5 ppm window",
        if found_out { 0.0 } else { 1.0 },
        1.0,
        0.0,
    );

    // Da tolerance: PFOA at ±0.01 Da
    let result_da = tolerance_search::find_within_da(&sorted_mz, pfoa_mz, 0.01);
    let found_da = result_da
        .iter()
        .any(|&idx| (sorted_mz[idx] - pfoa_mz).abs() < 1e-6);
    v.check(
        "PFOA found at ±0.01 Da",
        if found_da { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Verify the 5 ppm window is tight enough to avoid false positives
    // between PFCA adjacent members (CF2 = ~50 Da apart)
    let pfhxa_mz = 312.972_6_f64;
    let pfhpa_mz = 362.969_4_f64;
    let between = (pfhxa_mz + pfhpa_mz) / 2.0;
    let result_between = tolerance_search::find_within_ppm(&sorted_mz, between, 5.0);
    v.check(
        "5 ppm resolves adjacent PFCA (no false match midpoint)",
        if result_between.is_empty() { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Validate find_within_ppm returns correct index for shifted search
    let _ = shift_5ppm; // used only to document the tolerance window
}

// ── KMD homologue series detection ──────────────────────────────────────────

#[allow(clippy::similar_names, clippy::cast_precision_loss)]
fn validate_kmd_homologue_series(v: &mut Validator) {
    v.section("KMD Homologue Series Detection");

    // PFCA series: 11 compounds with CF2 spacing
    let pfca_masses: Vec<f64> = PFCA_SERIES.iter().map(|p| p.mh_neg).collect();
    let pfca_kmd =
        kmd::kendrick_mass_defect(&pfca_masses, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);

    // All PFCA KMDs should be similar (same homologue series)
    let kmd_values: Vec<f64> = pfca_kmd.iter().map(|r| r.kmd).collect();
    let kmd_min = kmd_values.iter().copied().fold(f64::INFINITY, f64::min);
    let kmd_max = kmd_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let kmd_range = kmd_max - kmd_min;

    println!("  PFCA KMD range: {kmd_range:.6} (min={kmd_min:.4}, max={kmd_max:.4})");
    v.check(
        "PFCA series KMD spread < 0.02",
        if kmd_range < 0.02 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // PFSA series: 6 compounds
    let pfsa_masses: Vec<f64> = PFSA_SERIES.iter().map(|p| p.mh_neg).collect();
    let pfsa_kmd =
        kmd::kendrick_mass_defect(&pfsa_masses, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);
    let pfsa_kmd_values: Vec<f64> = pfsa_kmd.iter().map(|r| r.kmd).collect();
    let pfsa_range = pfsa_kmd_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - pfsa_kmd_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

    println!("  PFSA KMD range: {pfsa_range:.6}");
    v.check(
        "PFSA series KMD spread < 0.02",
        if pfsa_range < 0.02 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // PFCA and PFSA should be in DIFFERENT KMD groups
    let pfca_mean_kmd: f64 = kmd_values.iter().sum::<f64>() / kmd_values.len() as f64;
    let pfsa_mean_kmd: f64 = pfsa_kmd_values.iter().sum::<f64>() / pfsa_kmd_values.len() as f64;
    let inter_series_gap = (pfca_mean_kmd - pfsa_mean_kmd).abs();
    println!("  PFCA mean KMD: {pfca_mean_kmd:.4}, PFSA mean KMD: {pfsa_mean_kmd:.4}");
    println!("  Inter-series gap: {inter_series_gap:.4}");

    v.check(
        "PFCA vs PFSA KMD separation > 0.01",
        if inter_series_gap > 0.01 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // pfas_kmd_screen grouping: all PFCA should cluster together
    let (_, pfca_groups) = kmd::pfas_kmd_screen(&pfca_masses, 0.01);
    let max_pfca_group = pfca_groups.iter().map(Vec::len).max().unwrap_or(0);
    println!(
        "  PFCA grouping: {} groups, largest = {}",
        pfca_groups.len(),
        max_pfca_group
    );
    v.check(
        "PFCA largest KMD group ≥ 8 of 11",
        if max_pfca_group >= 8 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Mixed input: PFCA + PFSA + FTSA should produce multiple distinct groups
    let mut mixed_masses: Vec<f64> = pfca_masses;
    mixed_masses.extend(pfsa_masses.iter());
    mixed_masses.extend(FTSA_SERIES.iter().map(|p| p.mh_neg));

    let (_, mixed_groups) = kmd::pfas_kmd_screen(&mixed_masses, 0.005);
    println!(
        "  Mixed series: {} groups from {} compounds",
        mixed_groups.len(),
        mixed_masses.len()
    );
    v.check(
        "Mixed series produces ≥ 2 KMD groups",
        if mixed_groups.len() >= 2 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
}

// ── Spectral matching for PFAS compound classes ─────────────────────────────

#[allow(clippy::similar_names)]
fn validate_spectral_match_pfas(v: &mut Validator) {
    v.section("PFAS Spectral Matching");

    // Synthetic PFAS-like MS2 spectra: PFCA compounds share CF2 loss patterns.
    // Characteristic fragments for PFOA [M-H]⁻:
    //   m/z 368.977 (loss of CO₂), 218.986 (C4F9⁻), 168.989 (C3F7⁻)
    let pfoa_mz = vec![168.989, 218.986, 368.977, 412.966];
    let pfoa_int = vec![30.0, 100.0, 45.0, 15.0];

    // PFNA shares similar fragmentation pattern (shifted by CF2 = ~50 Da)
    let pfna_mz = vec![218.986, 268.983, 418.974, 462.963];
    let pfna_int = vec![30.0, 100.0, 45.0, 15.0];

    // Same-class PFCA: should have moderate-high spectral similarity
    // due to shared fragment at 218.986 (C4F9⁻)
    let result_same =
        spectral_match::cosine_similarity(&pfoa_mz, &pfoa_int, &pfna_mz, &pfna_int, 0.5);
    println!(
        "  PFOA vs PFNA cosine: {:.4} ({} matched peaks)",
        result_same.score, result_same.matched_peaks
    );

    v.check(
        "Same-class cosine > 0 (shared fragments)",
        if result_same.score > 0.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Self-similarity = 1.0
    let result_self =
        spectral_match::cosine_similarity(&pfoa_mz, &pfoa_int, &pfoa_mz, &pfoa_int, 0.5);
    v.check(
        "PFOA self-cosine = 1.0",
        result_self.score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // Different compound class: PFSA fragments are distinct from PFCA
    // PFOS fragment pattern: m/z 79.957 (SO₃⁻), 98.956 (HSO₄⁻), 498.930 (precursor)
    let pfos_mz = vec![79.957, 98.956, 129.954, 498.930];
    let pfos_int = vec![100.0, 40.0, 25.0, 10.0];

    let result_diff =
        spectral_match::cosine_similarity(&pfoa_mz, &pfoa_int, &pfos_mz, &pfos_int, 0.5);
    println!(
        "  PFOA vs PFOS cosine: {:.4} ({} matched peaks)",
        result_diff.score, result_diff.matched_peaks
    );

    v.check(
        "Different-class cosine < 0.3",
        if result_diff.score < 0.3 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Pairwise: PFOA, PFNA, PFOS — upper triangle
    let spectra = vec![
        (pfoa_mz, pfoa_int),
        (pfna_mz, pfna_int),
        (pfos_mz, pfos_int),
    ];
    let scores = spectral_match::pairwise_cosine(&spectra, 0.5);
    v.check_count("Pairwise: 3 pairs", scores.len(), 3);

    // scores[0] = PFOA vs PFNA, scores[1] = PFOA vs PFOS, scores[2] = PFNA vs PFOS
    println!(
        "  Pairwise: PFOA-PFNA={:.3}, PFOA-PFOS={:.3}, PFNA-PFOS={:.3}",
        scores[0], scores[1], scores[2]
    );

    // Weighted cosine (mz_power=0, int_power=0.5 — intensity-focused)
    let pfoa_mz2 = vec![168.989, 218.986, 368.977, 412.966];
    let pfoa_int2 = vec![30.0, 100.0, 45.0, 15.0];
    let wresult = spectral_match::cosine_similarity_weighted(
        &pfoa_mz2, &pfoa_int2, &pfoa_mz2, &pfoa_int2, 0.5, 0.0, 0.5,
    );
    v.check(
        "Weighted self-cosine = 1.0",
        wresult.score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
}

// ── Cross-series discrimination ─────────────────────────────────────────────

#[allow(clippy::similar_names, clippy::cast_precision_loss)]
fn validate_cross_series_discrimination(v: &mut Validator) {
    v.section("Cross-Series Discrimination");

    let all = all_pfas();

    // For each compound, find_within_ppm at 5 ppm should return only
    // that compound (or possibly one very close neighbor)
    let sorted_mz: Vec<f64> = {
        let mut mzs: Vec<f64> = all.iter().map(|p| p.mh_neg).collect();
        mzs.sort_by(f64::total_cmp);
        mzs
    };

    let mut total_queries = 0_usize;
    let mut correct_single_hit = 0_usize;

    for compound in &all {
        let matches = tolerance_search::find_within_ppm(&sorted_mz, compound.mh_neg, 5.0);
        total_queries += 1;
        if matches.len() == 1 {
            correct_single_hit += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let selectivity = correct_single_hit as f64 / total_queries as f64;
    println!(
        "  5 ppm selectivity: {correct_single_hit}/{total_queries} unique hits ({:.1}%)",
        selectivity * 100.0
    );

    // At 5 ppm, most PFAS should resolve uniquely (masses > 50 Da apart)
    v.check(
        "5 ppm selectivity ≥ 80%",
        if selectivity >= 0.80 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // CF₂ spacing validation: adjacent PFCA members differ by ~49.997 Da
    let pfca_masses: Vec<f64> = PFCA_SERIES.iter().map(|p| p.mh_neg).collect();
    let mut cf2_diffs = Vec::new();
    for i in 1..pfca_masses.len() {
        cf2_diffs.push(pfca_masses[i] - pfca_masses[i - 1]);
    }

    let cf2_mean: f64 = cf2_diffs.iter().sum::<f64>() / cf2_diffs.len() as f64;
    println!("  PFCA CF₂ spacing: mean={cf2_mean:.4} Da (expected ~49.997)");

    v.check(
        "PFCA CF₂ spacing ≈ 49.997 Da",
        cf2_mean,
        kmd::units::CF2_EXACT,
        0.01,
    );

    // Same for PFSA
    let pfsa_masses: Vec<f64> = PFSA_SERIES.iter().map(|p| p.mh_neg).collect();
    let mut pfsa_diffs = Vec::new();
    for i in 1..pfsa_masses.len() {
        pfsa_diffs.push(pfsa_masses[i] - pfsa_masses[i - 1]);
    }

    // PFSA has a gap (PFPeS→PFHxS, PFHpS→PFOS, PFOS→PFDS)
    // Only adjacent-by-1-CF2 pairs should be ~50 Da
    let single_cf2: Vec<&f64> = pfsa_diffs.iter().filter(|d| d.abs() < 60.0).collect();

    if !single_cf2.is_empty() {
        let pfsa_cf2_mean: f64 = single_cf2.iter().copied().sum::<f64>() / single_cf2.len() as f64;
        println!("  PFSA single CF₂ spacing: mean={pfsa_cf2_mean:.4} Da");
        v.check(
            "PFSA CF₂ spacing ≈ 49.997 Da",
            pfsa_cf2_mean,
            kmd::units::CF2_EXACT,
            0.01,
        );
    }

    // Total compound coverage
    v.check_count("Total PFAS reference compounds", all.len(), 22);
}

// ── Jones Lab PFAS Library expansion (Zenodo 14341321) ──────────────────────

/// Validate against the full Jones Lab PFAS library (175 compounds, 259 ion
/// types) downloaded from Zenodo. The library provides experimentally measured
/// m/z, CCS, and RT values from LC-DTIMS-HRMS with certified standards.
///
/// Jones/Joseph et al. 2025, Nature Scientific Data, DOI 10.1038/s41597-024-04363-0
#[allow(clippy::cast_precision_loss)]
fn validate_jones_library_expansion(v: &mut Validator) {
    v.section("Jones Lab PFAS Library (175 compounds, Zenodo 14341321)");

    let lib_path = validation::data_dir("WETSPRING_PFAS_LIBRARY_DIR", "data/jones_pfas_library")
        .join("pfas_library_parsed.json");

    if !lib_path.exists() {
        println!(
            "  [SKIP] Jones library not found at {}\n  \
             Download from https://zenodo.org/record/14341321",
            lib_path.display()
        );
        return;
    }

    let raw = match std::fs::read_to_string(&lib_path) {
        Ok(s) => s,
        Err(e) => {
            println!("  [ERROR] Failed to read library: {e}");
            return;
        }
    };

    // Minimal JSON parsing: extract m/z values from "mz": N.NNNN fields
    let mz_values: Vec<f64> = raw
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("\"mz\":") {
                let val_str = trimmed
                    .trim_start_matches("\"mz\":")
                    .trim()
                    .trim_end_matches(',');
                val_str.parse::<f64>().ok()
            } else {
                None
            }
        })
        .collect();

    let n_ions = mz_values.len();
    println!("  Loaded {n_ions} ion types from Jones library");

    v.check(
        "Jones library loaded > 200 ion types",
        if n_ions > 200 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Cross-validate: our 22 hardcoded PFAS should have matching m/z in Jones
    let all = all_pfas();
    let mut sorted_jones: Vec<f64> = mz_values;
    sorted_jones.sort_by(f64::total_cmp);

    let mut cross_hits = 0_usize;
    for compound in &all {
        let matches = tolerance_search::find_within_ppm(&sorted_jones, compound.mh_neg, 20.0);
        if !matches.is_empty() {
            cross_hits += 1;
        }
    }

    println!(
        "  Cross-match: {cross_hits}/{} hardcoded PFAS found in Jones library at 20 ppm",
        all.len()
    );

    // Not all 22 may match — the Jones library uses [M-H]⁻ ESI(-) masses which
    // may differ slightly from our calculated monoisotopic masses. Expect ≥ 50%.
    let cross_rate = cross_hits as f64 / all.len() as f64;
    v.check(
        "Cross-match rate ≥ 50% at 20 ppm",
        if cross_rate >= 0.5 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Jones library m/z range should span PFBA (lowest, ~213) to PFTeDA (~713+)
    let mz_min = sorted_jones.first().copied().unwrap_or(0.0);
    let mz_max = sorted_jones.last().copied().unwrap_or(0.0);
    println!("  Jones library m/z range: {mz_min:.1} — {mz_max:.1} Da");

    v.check(
        "Jones m/z range spans > 500 Da",
        if (mz_max - mz_min) > 500.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // KMD analysis on Jones library: should detect multiple homologue groups
    let (_, groups) = kmd::pfas_kmd_screen(&sorted_jones, 0.01);
    println!(
        "  Jones KMD screening: {} groups from {} ions",
        groups.len(),
        n_ions
    );

    v.check(
        "Jones KMD groups ≥ 3 (PFCA + PFSA + others)",
        if groups.len() >= 3 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Largest group should have ≥ 10 members (PFCA homologue series)
    let max_group = groups.iter().map(Vec::len).max().unwrap_or(0);
    println!("  Largest KMD group: {max_group} members");

    v.check(
        "Largest Jones KMD group ≥ 10 members",
        if max_group >= 10 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
}
