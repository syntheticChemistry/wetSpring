// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate PFAS screening against `FindPFAS` baseline (Exp006).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline script | `scripts/validate_track2.py` |
//! | Baseline commit | `eb99b12` (Track 2 validation — 8/8 PASS) |
//! | Baseline date | 2026-02-16 |
//! | Exact command | `python3 scripts/validate_track2.py` |
//! | Dataset | `FindPFAS` test data (PFAS Standard Mix, ddMS2, 20 eV) |
//! | `FindPFAS` version | JonZwe/FindPFAS (GitHub) + pyOpenMS 3.5.0 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, Pop!\_OS 22.04) |
//!
//! # Expected values (from Python `FindPFAS`)
//!
//! - 738 MS2 spectra (exact — deterministic parser)
//! - 62 PFAS candidate spectra (CF2/C2F4/HF fragment screening)
//! - 25 unique PFAS precursor m/z values

use std::collections::HashSet;
use std::path::Path;
use wetspring_barracuda::bio::{kmd, spectral_match, tolerance_search};
use wetspring_barracuda::io::ms2;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("wetSpring PFAS Screening Validation");

    // Self-contained analytical validations (no external data needed)
    validate_spectral_match(&mut v);
    validate_kmd(&mut v);

    // External data-dependent validations
    let ms2_path = validation::data_dir(
        "WETSPRING_PFAS_MS2",
        "data/exp006_pfascreen/TestSample_PFAS_Standard_MIX_ddMS2_20eV_Inj5.ms2",
    );

    if ms2_path.exists() {
        validate_ms2_parsing(&ms2_path, &mut v);
    } else {
        println!(
            "\n  NOTE: MS2 test data not found at {}\n  Set WETSPRING_PFAS_MS2 to enable Track 2 validation.",
            ms2_path.display()
        );
    }

    v.finish();
}

fn validate_ms2_parsing(ms2_path: &Path, v: &mut Validator) {
    v.section("── MS2 parsing ──");
    let spectra = match ms2::parse_ms2(ms2_path) {
        Ok(s) => s,
        Err(e) => {
            println!("  FAILED: {e}");
            std::process::exit(1);
        }
    };
    let stats = ms2::compute_stats(&spectra);
    println!(
        "  Parsed {} spectra, {} total peaks",
        stats.num_spectra, stats.total_peaks
    );
    if let (Some(lo), Some(hi)) = (stats.min_precursor_mz, stats.max_precursor_mz) {
        println!("  Precursor m/z: {lo:.2} - {hi:.2}");
    }
    if let (Some(lo), Some(hi)) = (stats.min_rt, stats.max_rt) {
        println!("  RT: {lo:.2} - {hi:.2} min");
    }

    v.check_count("Total spectra", stats.num_spectra, 738);

    // ── PFAS fragment screening
    v.section("── PFAS fragment difference screening ──");
    let tol_da = tolerances::MZ_FRAGMENT;
    let min_intensity_pct = 5.0;

    let mut pfas_hits = Vec::new();
    for spec in &spectra {
        if let Some(result) = tolerance_search::screen_pfas_fragments(
            &spec.mz_array,
            &spec.intensity_array,
            spec.precursor_mz,
            spec.rt_minutes,
            tol_da,
            min_intensity_pct,
        ) {
            pfas_hits.push(result);
        }
    }

    // Unique precursors (rounded to 2 decimal places, same as Python)
    #[allow(clippy::cast_possible_truncation)] // intentional rounding
    let unique_mzs: HashSet<i64> = pfas_hits
        .iter()
        .map(|h| (h.precursor_mz * 100.0).round() as i64)
        .collect();

    println!("  PFAS candidates: {} spectra", pfas_hits.len());
    println!("  Unique precursor m/z: {}", unique_mzs.len());

    if let Some(top) = pfas_hits.iter().max_by_key(|h| h.total_diffs) {
        println!(
            "  Top hit: m/z {:.4}, RT {:.2} min, {} diffs (CF2:{}, C2F4:{}, HF:{})",
            top.precursor_mz, top.rt, top.total_diffs, top.cf2_count, top.c2f4_count, top.hf_count
        );
    }

    v.check_count("PFAS candidate spectra", pfas_hits.len(), 62);
    v.check_count("Unique PFAS precursors", unique_mzs.len(), 25);

    // ── Tolerance search unit tests
    validate_tolerance_search(v);
}

fn validate_tolerance_search(v: &mut Validator) {
    v.section("── Tolerance search unit tests ──");

    let mz_list = vec![
        100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0,
    ];

    let matches = tolerance_search::find_within_ppm(&mz_list, 500.0, 10.0);
    v.check_count("ppm search (exact)", matches.len(), 1);

    let matches_da = tolerance_search::find_within_da(&mz_list, 500.0, 1.0);
    v.check_count("Da search (±1 Da)", matches_da.len(), 1);

    let close_mz = vec![499.5, 499.9, 500.0, 500.1, 500.5, 501.0];
    let matches_close = tolerance_search::find_within_da(&close_mz, 500.0, 0.2);
    v.check_count("Da search (close values)", matches_close.len(), 3);
}

fn validate_spectral_match(v: &mut Validator) {
    v.section("── MS2 cosine similarity ──");

    // Identical spectra should score 1.0
    let mz = vec![100.0, 200.0, 300.0, 400.0, 500.0];
    let int = vec![50.0, 100.0, 75.0, 25.0, 10.0];
    let result = spectral_match::cosine_similarity(&mz, &int, &mz, &int, 0.5);
    v.check(
        "Cosine(identical) = 1.0",
        result.score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );
    v.check_count("Matched peaks (identical)", result.matched_peaks, 5);

    // Orthogonal spectra should score 0.0
    let r_mz = vec![600.0, 700.0, 800.0];
    let r_int = vec![50.0, 100.0, 75.0];
    let result2 = spectral_match::cosine_similarity(&mz, &int, &r_mz, &r_int, 0.5);
    v.check(
        "Cosine(disjoint) = 0.0",
        result2.score,
        0.0,
        tolerances::ANALYTICAL_F64,
    );

    // Weighted cosine (sqrt intensity scaling)
    let wresult = spectral_match::cosine_similarity_weighted(&mz, &int, &mz, &int, 0.5, 0.0, 0.5);
    v.check(
        "Weighted cosine(identical) = 1.0",
        wresult.score,
        1.0,
        tolerances::ANALYTICAL_F64,
    );

    // Pairwise cosine matrix
    let spectra = vec![(mz.clone(), int.clone()), (mz, int), (r_mz, r_int)];
    let scores = spectral_match::pairwise_cosine(&spectra, 0.5);
    v.check_count("Pairwise cosine pairs", scores.len(), 3);
    v.check(
        "Pairwise(0,1) identical",
        scores[0],
        1.0,
        tolerances::ANALYTICAL_F64,
    );
}

fn validate_kmd(v: &mut Validator) {
    v.section("── Kendrick Mass Defect (PFAS) ──");

    // PFOS/PFHxS/PFBS homologous series (CF2 repeating unit)
    // Exact masses for sulfonate series [M-H]-
    let pfos = 498.930; // C8F17SO3-
    let pfhxs = 398.936; // C6F13SO3-
    let pfbs_mass = 298.943; // C4F9SO3-

    let masses = vec![pfos, pfhxs, pfbs_mass];
    let results =
        kmd::kendrick_mass_defect(&masses, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);

    // KMDs of homologues should be similar (within ~0.01)
    let kmd_spread_01 = (results[0].kmd - results[1].kmd).abs();
    let kmd_spread_02 = (results[0].kmd - results[2].kmd).abs();
    println!("  PFOS KMD: {:.4}", results[0].kmd);
    println!("  PFHxS KMD: {:.4}", results[1].kmd);
    println!("  PFBS KMD: {:.4}", results[2].kmd);
    println!("  KMD spread (PFOS-PFHxS): {kmd_spread_01:.6}");
    println!("  KMD spread (PFOS-PFBS): {kmd_spread_02:.6}");

    v.check(
        "KMD homologue spread (01)",
        kmd_spread_01,
        0.0,
        tolerances::KMD_GROUPING,
    );
    v.check("KMD homologue spread (02)", kmd_spread_02, 0.0, 0.02);

    // Grouping should put homologues together
    let (_, groups) = kmd::pfas_kmd_screen(&masses, tolerances::KMD_GROUPING);
    let max_group = groups.iter().map(Vec::len).max().unwrap_or(0);
    v.check_count("Largest homologue group", max_group, 3);

    // Non-homologues should be separated
    let mixed = vec![pfos, pfhxs, pfbs_mass, 600.0, 312.5];
    let (_, mixed_groups) = kmd::pfas_kmd_screen(&mixed, 0.005);
    let has_separation = mixed_groups.len() >= 2;
    v.check(
        "Non-homologues separated",
        f64::from(u8::from(has_separation)),
        1.0,
        0.0,
    );
}
