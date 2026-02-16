//! Validate PFAS screening against FindPFAS baseline (Exp006).
//!
//! Parses PFAS Standard Mix MS2 data and screens for CF2/C2F4/HF
//! fragment mass differences. Expected (from Python FindPFAS):
//!   738 spectra, 62 candidates, 25 unique precursors.

use std::collections::HashSet;
use std::path::Path;
use wetspring_barracuda::bio::tolerance_search;
use wetspring_barracuda::io::ms2;

fn check(label: &str, actual: f64, expected: f64, tolerance: f64) -> bool {
    let pass = (actual - expected).abs() <= tolerance;
    let tag = if pass { "OK" } else { "FAIL" };
    println!(
        "  [{}]  {}: {:.4} (expected {:.4}, tol {:.4})",
        tag, label, actual, expected, tolerance
    );
    pass
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  wetSpring PFAS Screening Validation");
    println!("  Reference: FindPFAS (Exp006 — PFAS Standard Mix)");
    println!("═══════════════════════════════════════════════════════════\n");

    let ms2_path = Path::new("/tmp/FindPFAS/TestSample_PFAS_Standard_MIX_ddMS2_20eV_Inj5.ms2");

    if !ms2_path.exists() {
        println!("  SKIP: MS2 test data not found at {}", ms2_path.display());
        println!("  Install with: pip install git+https://github.com/JonZwe/FindPFAS.git");
        std::process::exit(0);
    }

    let mut total = 0u32;
    let mut passed = 0u32;

    // ── Parse MS2 file ──────────────────────────────────────────
    println!("── MS2 parsing ──");
    let spectra = match ms2::parse_ms2(ms2_path) {
        Ok(s) => s,
        Err(e) => {
            println!("  FAILED: {}", e);
            std::process::exit(1);
        }
    };
    let stats = ms2::compute_stats(&spectra);
    println!(
        "  Parsed {} spectra, {} total peaks",
        stats.num_spectra, stats.total_peaks
    );
    println!(
        "  Precursor m/z: {:.2} - {:.2}",
        stats.min_precursor_mz, stats.max_precursor_mz
    );
    println!(
        "  RT: {:.2} - {:.2} min",
        stats.min_rt, stats.max_rt
    );

    total += 1;
    if check("Total spectra", stats.num_spectra as f64, 738.0, 0.0) {
        passed += 1;
    }

    // ── PFAS fragment screening ─────────────────────────────────
    println!("\n── PFAS fragment difference screening ──");
    let tol_da = 0.001; // 1 mDa tolerance (same as FindPFAS)
    let min_intensity_pct = 5.0; // 5% relative intensity threshold

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
    let unique_mzs: HashSet<i64> = pfas_hits
        .iter()
        .map(|h| (h.precursor_mz * 100.0).round() as i64)
        .collect();

    println!("  PFAS candidates: {} spectra", pfas_hits.len());
    println!("  Unique precursor m/z: {}", unique_mzs.len());

    if !pfas_hits.is_empty() {
        let top = &pfas_hits
            .iter()
            .max_by_key(|h| h.total_diffs)
            .unwrap();
        println!(
            "  Top hit: m/z {:.4}, RT {:.2} min, {} diffs (CF2:{}, C2F4:{}, HF:{})",
            top.precursor_mz, top.rt, top.total_diffs,
            top.cf2_count, top.c2f4_count, top.hf_count
        );
    }

    // Exp006 baseline: 62 candidate spectra, 25 unique precursors
    total += 1;
    if check("PFAS candidate spectra", pfas_hits.len() as f64, 62.0, 5.0) {
        passed += 1;
    }

    total += 1;
    if check("Unique PFAS precursors", unique_mzs.len() as f64, 25.0, 0.0) {
        passed += 1;
    }

    // ── Tolerance search unit tests ─────────────────────────────
    println!("\n── Tolerance search unit tests ──");
    {
        let mz_list = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0];

        // ppm search
        let matches = tolerance_search::find_within_ppm(&mz_list, 500.0, 10.0);
        total += 1;
        if check("ppm search (exact)", matches.len() as f64, 1.0, 0.0) {
            passed += 1;
        }

        // Da search (wider)
        let matches_da = tolerance_search::find_within_da(&mz_list, 500.0, 1.0);
        total += 1;
        if check("Da search (±1 Da)", matches_da.len() as f64, 1.0, 0.0) {
            passed += 1;
        }

        // Da search that catches neighbors
        let close_mz = vec![499.5, 499.9, 500.0, 500.1, 500.5, 501.0];
        let matches_close = tolerance_search::find_within_da(&close_mz, 500.0, 0.2);
        total += 1;
        if check("Da search (close values)", matches_close.len() as f64, 3.0, 0.0) {
            passed += 1;
        }
    }

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  PFAS Validation: {}/{} checks passed",
        passed, total
    );
    if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");

    std::process::exit(if passed == total { 0 } else { 1 });
}
