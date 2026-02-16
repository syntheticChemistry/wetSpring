//! Validate mzML parser against MT02 demo dataset (Exp005 baseline).
//!
//! Expected: 8 mzML files, Orbitrap HRMS data, m/z range ~80-1000,
//! centroid spectra, 64-bit float + zlib compressed.

use std::path::Path;
use wetspring_barracuda::io::mzml;

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
    println!("  wetSpring mzML Parser Validation");
    println!("  Reference: asari / pyteomics on MT02 dataset");
    println!("═══════════════════════════════════════════════════════════\n");

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/exp005_asari/MT02/MT02Dataset");

    let mut total = 0u32;
    let mut passed = 0u32;

    // Collect all mzML files
    let mut mzml_files: Vec<_> = match std::fs::read_dir(&data_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext == "mzML")
            })
            .map(|e| e.path())
            .collect(),
        Err(e) => {
            println!("  FAILED: Cannot read data dir: {}", e);
            std::process::exit(1);
        }
    };
    mzml_files.sort();

    println!("── File count ──");
    total += 1;
    if check("mzML files", mzml_files.len() as f64, 8.0, 0.0) {
        passed += 1;
    }

    // Parse each file and accumulate stats
    let mut all_spectra_count = 0usize;
    let mut all_ms1 = 0usize;
    let mut all_peaks = 0usize;
    let mut global_min_mz = f64::MAX;
    let mut global_max_mz = f64::MIN;
    let mut first_file_spectra = 0usize;

    for (i, path) in mzml_files.iter().enumerate() {
        let fname = path.file_name().unwrap().to_string_lossy();
        print!("  Parsing {}... ", fname);

        let t0 = std::time::Instant::now();
        match mzml::parse_mzml(path) {
            Ok(spectra) => {
                let stats = mzml::compute_stats(&spectra);
                let elapsed = t0.elapsed();
                println!(
                    "{} spectra ({} MS1, {} MS2), {:.1}s",
                    stats.num_spectra, stats.num_ms1, stats.num_ms2,
                    elapsed.as_secs_f64()
                );

                if i == 0 {
                    first_file_spectra = stats.num_spectra;
                }
                all_spectra_count += stats.num_spectra;
                all_ms1 += stats.num_ms1;
                all_peaks += stats.total_peaks;
                if stats.min_mz < global_min_mz {
                    global_min_mz = stats.min_mz;
                }
                if stats.max_mz > global_max_mz {
                    global_max_mz = stats.max_mz;
                }

                // Verify binary arrays decoded correctly (spot check first spectrum)
                if !spectra.is_empty() {
                    let s = &spectra[0];
                    if s.mz_array.is_empty() {
                        println!("    WARNING: First spectrum has empty m/z array!");
                    }
                }
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }

    println!("\n── Aggregate statistics ──");
    println!(
        "  Total: {} spectra, {} MS1, {} peaks",
        all_spectra_count, all_ms1, all_peaks
    );
    println!(
        "  m/z range: {:.2} - {:.2}",
        global_min_mz, global_max_mz
    );

    // First file should have ~777 spectra (from header: spectrumList count="777")
    total += 1;
    if check("First file spectra", first_file_spectra as f64, 777.0, 100.0) {
        passed += 1;
    }

    // 8 files × ~780 spectra each = ~6,256 total
    total += 1;
    if check("Total spectra", all_spectra_count as f64, 6256.0, 200.0) {
        passed += 1;
    }

    // All spectra are MS1 in this LC-MS dataset (no MS2)
    total += 1;
    if check("MS1 spectra", all_ms1 as f64, 6256.0, 200.0) {
        passed += 1;
    }

    // Binary arrays decoded: ~6M peaks across all spectra
    total += 1;
    if check("Total decoded peaks", all_peaks as f64, 6000000.0, 500000.0) {
        passed += 1;
    }

    // m/z range should cover ~80-1000 (Orbitrap full scan)
    total += 1;
    if check("Min m/z < 100", global_min_mz, 85.0, 15.0) {
        passed += 1;
    }

    total += 1;
    if check("Max m/z > 900", global_max_mz, 1000.0, 100.0) {
        passed += 1;
    }

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  mzML Validation: {}/{} checks passed", passed, total);
    if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");

    std::process::exit(if passed == total { 0 } else { 1 });
}
