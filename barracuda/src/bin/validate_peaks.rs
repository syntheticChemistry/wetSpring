// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate `bio::signal::find_peaks` against `scipy.signal.find_peaks` (Exp010).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | `scipy.signal.find_peaks` (scipy 1.14+) |
//! | Baseline version | scipy 1.14+ |
//! | Baseline command | `scripts/generate_peak_baselines.py` (Exp010) |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `python3 scripts/generate_peak_baselines.py` |
//! | Data | Synthetic (`experiments/results/010_peak_baselines/*.dat`) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! # Format of .dat files
//!
//! Line 1: space-separated data values (f64)
//! Line 2: space-separated peak indices (usize) — may be empty
//! Line 3: space-separated peak heights (f64) — may be empty
//! Line 4: space-separated prominences (f64) — may be empty
//!
//! Run: `cargo run --bin validate_peaks`

use std::path::Path;
use wetspring_barracuda::bio::signal::{find_peaks, PeakParams};
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("wetSpring Peak Detection Validation (Exp010)");

    let baseline_dir = validation::data_dir(
        "WETSPRING_PEAKS_DIR",
        "experiments/results/010_peak_baselines",
    );

    if !baseline_dir.exists() {
        println!(
            "  NOTE: peak baselines not found at {}\n  \
             Run scripts/generate_peak_baselines.py first.",
            baseline_dir.display()
        );
        v.finish();
    }

    validate_case(
        &mut v,
        &baseline_dir,
        "single_gaussian",
        &PeakParams {
            min_height: Some(50.0),
            min_prominence: Some(50.0),
            min_width: Some(2.0),
            ..PeakParams::default()
        },
    );

    validate_case(
        &mut v,
        &baseline_dir,
        "three_chromatographic",
        &PeakParams {
            min_height: Some(100.0),
            min_prominence: Some(200.0),
            min_width: Some(2.0),
            ..PeakParams::default()
        },
    );

    validate_case(
        &mut v,
        &baseline_dir,
        "noisy_with_spikes",
        &PeakParams {
            min_height: Some(300.0),
            min_prominence: Some(200.0),
            distance: 20,
            ..PeakParams::default()
        },
    );

    validate_case(
        &mut v,
        &baseline_dir,
        "overlapping_peaks",
        &PeakParams {
            min_prominence: Some(100.0),
            ..PeakParams::default()
        },
    );

    validate_case(
        &mut v,
        &baseline_dir,
        "monotonic_no_peaks",
        &PeakParams::default(),
    );

    v.finish();
}

fn validate_case(v: &mut Validator, dir: &Path, name: &str, params: &PeakParams) {
    v.section(name);

    let path = dir.join(format!("{name}.dat"));
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            validation::exit_skipped(&format!("Cannot read {}: {e}", path.display()));
        }
    };
    let lines: Vec<&str> = contents.lines().collect();

    let data: Vec<f64> = lines[0]
        .split_whitespace()
        .map(|s| s.parse::<f64>().expect("parse data value"))
        .collect();

    let scipy_indices: Vec<usize> = if lines[1].trim().is_empty() {
        vec![]
    } else {
        lines[1]
            .split_whitespace()
            .map(|s| s.parse::<usize>().expect("parse index"))
            .collect()
    };

    let scipy_heights: Vec<f64> = if lines[2].trim().is_empty() {
        vec![]
    } else {
        lines[2]
            .split_whitespace()
            .map(|s| s.parse::<f64>().expect("parse height"))
            .collect()
    };

    let rust_peaks = find_peaks(&data, params);

    println!(
        "  {name}: {} data points, scipy found {} peaks, Rust found {} peaks",
        data.len(),
        scipy_indices.len(),
        rust_peaks.len()
    );

    v.check_count(
        &format!("{name} peak count"),
        rust_peaks.len(),
        scipy_indices.len(),
    );

    // Index matching: allow ±1 for boundary rounding
    let mut indices_matched = 0_usize;
    for (i, &sci_idx) in scipy_indices.iter().enumerate() {
        #[allow(clippy::cast_possible_wrap)]
        let found = rust_peaks
            .iter()
            .any(|p| (p.index as i64 - sci_idx as i64).unsigned_abs() <= 1);
        if found {
            indices_matched += 1;
        } else {
            println!("  MISMATCH: scipy peak {i} at index {sci_idx} not found in Rust (±1)");
        }
    }

    if !scipy_indices.is_empty() {
        v.check_count(
            &format!("{name} indices matched (±1)"),
            indices_matched,
            scipy_indices.len(),
        );
    }

    // Height matching: Rust and scipy process identical data, heights should be very close
    for (i, &sci_height) in scipy_heights.iter().enumerate() {
        #[allow(clippy::cast_possible_wrap)]
        if let Some(rp) = rust_peaks.iter().find(|p| {
            let sci_idx = scipy_indices[i];
            (p.index as i64 - sci_idx as i64).unsigned_abs() <= 1
        }) {
            let rel_err = if sci_height.abs() > 1e-12 {
                (rp.height - sci_height).abs() / sci_height
            } else {
                (rp.height - sci_height).abs()
            };
            v.check(
                &format!("{name} peak {i} height"),
                rp.height,
                sci_height,
                sci_height * tolerances::PEAK_HEIGHT_REL,
            );
            if rel_err > tolerances::PEAK_HEIGHT_REL {
                println!(
                    "  Height drift peak {i}: rust={:.4} scipy={sci_height:.4} err={rel_err:.6}",
                    rp.height
                );
            }
        }
    }
}
