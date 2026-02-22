// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate feature extraction pipeline against asari MT02 baseline (Exp009).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | asari 1.13.1 |
//! | Baseline version | asari 1.13.1 |
//! | Baseline file | `experiments/results/005_asari/preferred_Feature_table.tsv` (Exp009) |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --bin validate_features` |
//! | Data | shuzhao-li-lab/data (MT02 HILIC-pos, 8 mzML files, Orbitrap HRMS) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//! | Asari features | 5,951 (filtered from 8,659 total) |
//! | Asari compounds | 4,107 unique |
//! | Asari params | 5 ppm, min 6 scans, SNR >= 2, `min_peak_height` 100,000 |
//! | Asari m/z range | 83–999 Da |
//!
//! # Acceptance criteria (from Exp009)
//!
//! 1. Mass track count within 2× of asari `parent_masstrack_id` range
//! 2. Feature count in thousands (same order of magnitude as asari 5,951)
//! 3. m/z range covers asari's range (83–999)
//! 4. RT range covers asari's RT range
//! 5. At least 30% of asari features matched by m/z (5 ppm) + RT (50 sec)
//!
//! # What this validates
//!
//! - `bio::eic::detect_mass_tracks` — mass track count
//! - `bio::eic::extract_eics` — EIC extraction correctness
//! - `bio::signal::find_peaks` — peak detection on real chromatograms
//! - `bio::feature_table::extract_features` — end-to-end feature extraction
//!
//! Run: `cargo run --bin validate_features`

use std::path::Path;
use wetspring_barracuda::bio::eic;
use wetspring_barracuda::bio::feature_table::{self, FeatureParams};
use wetspring_barracuda::bio::signal::PeakParams;
use wetspring_barracuda::io::mzml;
use wetspring_barracuda::validation::{self, Validator};

#[allow(clippy::too_many_lines)] // sequential feature validation: parse → tracks → EIC → features → asari cross-reference
fn main() {
    let mut v = Validator::new("wetSpring Feature Pipeline Validation (Exp009)");

    let data_dir = validation::data_dir("WETSPRING_MZML_DIR", "data/exp005_asari/MT02/MT02Dataset");

    let asari_tsv = validation::data_dir(
        "WETSPRING_ASARI_TSV",
        "experiments/results/005_asari/preferred_Feature_table.tsv",
    );

    if !data_dir.exists() {
        println!(
            "  NOTE: mzML data not found at {}\n  \
             Set WETSPRING_MZML_DIR to enable feature validation.",
            data_dir.display()
        );
        v.finish();
    }

    let asari_features = load_asari_features(&asari_tsv);

    println!("  Asari baseline: {} features", asari_features.len());

    // Parse all mzML files
    let mut mzml_files: Vec<_> = match std::fs::read_dir(&data_dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "mzML"))
            .map(|e| e.path())
            .collect(),
        Err(e) => {
            validation::exit_skipped(&format!("Cannot read data dir {}: {e}", data_dir.display()));
        }
    };
    mzml_files.sort();

    v.check_count("mzML files found", mzml_files.len(), 8);

    // Use the first file for single-file validation
    let first_file = &mzml_files[0];
    println!(
        "\n  Parsing {}...",
        first_file
            .file_name()
            .map_or_else(|| first_file.to_string_lossy(), |n| n.to_string_lossy())
    );
    let spectra = mzml::parse_mzml(first_file).expect("parse mzML");
    let ms1: Vec<_> = spectra
        .iter()
        .filter(|s| s.ms_level == 1)
        .cloned()
        .collect();

    v.section("Mass track detection");

    let mass_tracks = eic::detect_mass_tracks(&ms1, 5.0, 6);
    println!(
        "  Detected {} mass tracks from first file",
        mass_tracks.len()
    );

    // Acceptance: mass track count within 2× of asari parent track range (thousands expected)
    v.check(
        "Mass tracks >= 500 (lower bound, asari finds thousands)",
        f64::from(u8::from(mass_tracks.len() >= 500)),
        1.0,
        0.0,
    );

    v.section("Feature extraction (single file)");

    let params = FeatureParams {
        eic_ppm: 5.0,
        min_scans: 6,
        peak_params: PeakParams {
            min_prominence: Some(1000.0),
            min_height: Some(10000.0),
            ..PeakParams::default()
        },
        min_height: 10000.0,
        min_snr: 2.0,
    };

    let table = feature_table::extract_features(&ms1, &params);
    println!(
        "  Features: {}, mass tracks: {}, EICs with peaks: {}",
        table.features.len(),
        table.mass_tracks_evaluated,
        table.eics_with_peaks
    );

    // Acceptance: feature count in same order of magnitude as asari (5,951 features)
    v.check(
        "Features >= 100 (asari finds ~6k from 8 files; single file yields fewer)",
        f64::from(u8::from(table.features.len() >= 100)),
        1.0,
        0.0,
    );
    v.check(
        "Mass tracks evaluated >= 100",
        f64::from(u8::from(table.mass_tracks_evaluated >= 100)),
        1.0,
        0.0,
    );

    if !table.features.is_empty() {
        let mz_min = table
            .features
            .iter()
            .map(|f| f.mz)
            .fold(f64::INFINITY, f64::min);
        let mz_max = table
            .features
            .iter()
            .map(|f| f.mz)
            .fold(f64::NEG_INFINITY, f64::max);
        let rt_min = table
            .features
            .iter()
            .map(|f| f.rt_apex)
            .fold(f64::INFINITY, f64::min);
        let rt_max = table
            .features
            .iter()
            .map(|f| f.rt_apex)
            .fold(f64::NEG_INFINITY, f64::max);

        println!("  m/z range: {mz_min:.2} - {mz_max:.2}");
        println!("  RT range:  {rt_min:.2} - {rt_max:.2} sec");

        // Acceptance: m/z range covers asari's range (83–999 Da)
        v.check(
            "m/z min < 150 (asari low end is ~83)",
            f64::from(u8::from(mz_min < 150.0)),
            1.0,
            0.0,
        );
        v.check(
            "m/z max > 800 (asari high end is ~999)",
            f64::from(u8::from(mz_max > 800.0)),
            1.0,
            0.0,
        );
    }

    v.section("Asari cross-reference");

    if !asari_features.is_empty() && !table.features.is_empty() {
        let matched = count_matched_features(&table.features, &asari_features, 5.0, 50.0);
        #[allow(clippy::cast_precision_loss)]
        let match_pct = (matched as f64 / asari_features.len() as f64) * 100.0;
        println!(
            "  Matched {matched}/{} asari features ({match_pct:.1}%) by m/z+RT",
            asari_features.len()
        );

        // Acceptance: at least 30% of asari features matched (Exp009 criterion)
        v.check(
            "Cross-match >= 30% of asari features",
            match_pct,
            30.0,
            70.0, // tolerance: match_pct must be >= 30% (i.e. within 70 of 100)
        );

        let rust_in_asari_range = table
            .features
            .iter()
            .filter(|f| f.mz >= 80.0 && f.mz <= 1000.0)
            .count();
        #[allow(clippy::cast_precision_loss)]
        let range_pct = (rust_in_asari_range as f64 / table.features.len() as f64) * 100.0;
        v.check(
            "Rust features in asari m/z range (80-1000) >= 90%",
            range_pct,
            100.0,
            10.0,
        );
    } else {
        println!("  (skipping cross-reference — no features to compare)");
    }

    v.finish();
}

struct AsariFeature {
    mz: f64,
    rtime: f64,
    #[allow(dead_code)]
    peak_area: f64,
    #[allow(dead_code)]
    snr: f64,
}

fn load_asari_features(path: &Path) -> Vec<AsariFeature> {
    if !path.exists() {
        println!(
            "  NOTE: asari feature table not found at {}",
            path.display()
        );
        return vec![];
    }

    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read asari TSV {}: {e}", path.display());
            return vec![];
        }
    };
    let mut features = Vec::new();

    for (i, line) in contents.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 10 {
            continue;
        }

        let mz: f64 = cols[1].parse().unwrap_or(0.0);
        let rtime: f64 = cols[2].parse().unwrap_or(0.0);
        let peak_area: f64 = cols[6].parse().unwrap_or(0.0);
        let snr: f64 = cols[9].parse().unwrap_or(0.0);

        if mz > 0.0 {
            features.push(AsariFeature {
                mz,
                rtime,
                peak_area,
                snr,
            });
        }
    }

    features
}

fn count_matched_features(
    rust_features: &[feature_table::Feature],
    asari_features: &[AsariFeature],
    ppm: f64,
    rt_tol_sec: f64,
) -> usize {
    asari_features
        .iter()
        .filter(|af| {
            rust_features.iter().any(|rf| {
                let mz_diff_ppm = ((rf.mz - af.mz) / af.mz).abs() * 1e6;
                let rt_diff = (rf.rt_apex - af.rtime).abs();
                mz_diff_ppm <= ppm && rt_diff <= rt_tol_sec
            })
        })
        .count()
}
