// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate mzML parser against MT02 demo dataset (Exp005 baseline).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline script | `scripts/validate_track2.py` |
//! | Baseline commit | `eb99b12` (Track 2 validation — 8/8 PASS) |
//! | Baseline date | 2026-02-16 |
//! | Dataset | `shuzhao-li-lab/data` (MT02 demo, 8 mzML files) |
//! | asari version | 1.13.1 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, Pop!\_OS 22.04) |
//!
//! # Expected values
//!
//! - 8 mzML files, Orbitrap HRMS, HILIC-pos
//! - ~6,256 MS1 spectra total (all centroid)
//! - m/z range ~80–1000, 64-bit float + zlib compressed
//! - ~6M total decoded peaks

use std::path::PathBuf;
use wetspring_barracuda::io::mzml;
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("wetSpring mzML Parser Validation");

    let data_dir = validation::data_dir("WETSPRING_MZML_DIR", "data/exp005_asari/MT02/MT02Dataset");

    // Collect all mzML files
    let mut mzml_files: Vec<_> = match std::fs::read_dir(&data_dir) {
        Ok(entries) => entries
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "mzML"))
            .map(|e| e.path())
            .collect(),
        Err(e) => {
            println!("  FAILED: Cannot read data dir: {e}");
            std::process::exit(1);
        }
    };
    mzml_files.sort();

    v.section("── File count ──");
    v.check_count("mzML files", mzml_files.len(), 8);

    let aggregates = parse_all_files(&mzml_files, &v);

    v.section("── Aggregate statistics ──");
    println!(
        "  Total: {} spectra, {} MS1, {} peaks",
        aggregates.spectra, aggregates.ms1, aggregates.peaks
    );
    if let (Some(lo), Some(hi)) = (aggregates.min_mz, aggregates.max_mz) {
        println!("  m/z range: {lo:.2} - {hi:.2}");
    }

    v.check_count("First file spectra", aggregates.first_file_spectra, 777);
    v.check_count("Total spectra", aggregates.spectra, 6256);
    v.check_count("MS1 spectra", aggregates.ms1, 6256);
    v.check_count("Total decoded peaks", aggregates.peaks, 6_066_434);
    v.check("Min m/z", aggregates.min_mz.unwrap_or(0.0), 80.001_03, 0.01);
    v.check(
        "Max m/z",
        aggregates.max_mz.unwrap_or(0.0),
        999.992_92,
        0.01,
    );

    v.finish();
}

struct Aggregates {
    spectra: usize,
    ms1: usize,
    peaks: usize,
    min_mz: Option<f64>,
    max_mz: Option<f64>,
    first_file_spectra: usize,
}

fn parse_all_files(files: &[PathBuf], v: &Validator) -> Aggregates {
    let _ = v; // used for section context only
    let mut agg = Aggregates {
        spectra: 0,
        ms1: 0,
        peaks: 0,
        min_mz: None,
        max_mz: None,
        first_file_spectra: 0,
    };

    for (i, path) in files.iter().enumerate() {
        let fname = path
            .file_name()
            .map_or_else(|| path.to_string_lossy(), |n| n.to_string_lossy());
        print!("  Parsing {fname}... ");

        let t0 = std::time::Instant::now();
        match mzml::parse_mzml(path) {
            Ok(spectra) => {
                let stats = mzml::compute_stats(&spectra);
                let elapsed = t0.elapsed();
                println!(
                    "{} spectra ({} MS1, {} MS2), {:.1}s",
                    stats.num_spectra,
                    stats.num_ms1,
                    stats.num_ms2,
                    elapsed.as_secs_f64()
                );

                if i == 0 {
                    agg.first_file_spectra = stats.num_spectra;
                }
                agg.spectra += stats.num_spectra;
                agg.ms1 += stats.num_ms1;
                agg.peaks += stats.total_peaks;

                if let Some(lo) = stats.min_mz {
                    agg.min_mz = Some(agg.min_mz.map_or(lo, |v: f64| v.min(lo)));
                }
                if let Some(hi) = stats.max_mz {
                    agg.max_mz = Some(agg.max_mz.map_or(hi, |v: f64| v.max(hi)));
                }

                if !spectra.is_empty() && spectra[0].mz_array.is_empty() {
                    println!("    WARNING: First spectrum has empty m/z array!");
                }
            }
            Err(e) => {
                println!("FAILED: {e}");
            }
        }
    }

    agg
}
