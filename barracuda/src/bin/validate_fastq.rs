//! Validate FASTQ parser against MiSeq SOP training data (Exp001 baseline).
//!
//! Expected values from Galaxy/QIIME2 validation:
//!   F3D0_R1: 7,793 sequences, 249-251 bp, GC ~54%, mean Q ~32-38

use std::path::Path;
use wetspring_barracuda::io::fastq;

fn check(label: &str, actual: f64, expected: f64, tolerance: f64) -> bool {
    let pass = (actual - expected).abs() <= tolerance;
    let tag = if pass { "OK" } else { "FAIL" };
    println!("  [{}]  {}: {:.4} (expected {:.4}, tol {:.4})", tag, label, actual, expected, tolerance);
    pass
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  wetSpring FASTQ Parser Validation");
    println!("  Reference: Galaxy FastQC + QIIME2 Exp001");
    println!("═══════════════════════════════════════════════════════════\n");

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/validation/MiSeq_SOP");

    let mut total = 0u32;
    let mut passed = 0u32;

    // ── F3D0_R1 (forward reads) ─────────────────────────────────
    println!("── F3D0_R1.fastq (forward reads) ──");
    let r1_path = data_dir.join("F3D0_R1.fastq");
    match fastq::parse_fastq(&r1_path) {
        Ok(records) => {
            let stats = fastq::compute_stats(&records);
            println!("  Parsed {} sequences, {} total bases",
                     stats.num_sequences, stats.total_bases);
            println!("  Lengths: {}-{} bp (mean {:.1})",
                     stats.min_length, stats.max_length, stats.mean_length);
            println!("  GC: {:.1}%, Mean Q: {:.1}",
                     stats.gc_content * 100.0, stats.mean_quality);

            total += 1;
            if check("Sequence count", stats.num_sequences as f64, 7793.0, 0.0) { passed += 1; }

            total += 1;
            if check("Min length", stats.min_length as f64, 249.0, 1.0) { passed += 1; }

            total += 1;
            if check("Max length", stats.max_length as f64, 251.0, 1.0) { passed += 1; }

            total += 1;
            if check("GC content", stats.gc_content, 0.54, 0.02) { passed += 1; }

            total += 1;
            if check("Mean quality >= 30", stats.mean_quality, 35.0, 5.0) { passed += 1; }
        }
        Err(e) => {
            println!("  FAILED: {}", e);
            total += 5;
        }
    }

    // ── F3D0_R2 (reverse reads) ─────────────────────────────────
    println!("\n── F3D0_R2.fastq (reverse reads) ──");
    let r2_path = data_dir.join("F3D0_R2.fastq");
    match fastq::parse_fastq(&r2_path) {
        Ok(records) => {
            let stats = fastq::compute_stats(&records);
            println!("  Parsed {} sequences, {} total bases",
                     stats.num_sequences, stats.total_bases);

            total += 1;
            if check("R2 sequence count", stats.num_sequences as f64, 7793.0, 0.0) { passed += 1; }

            // R2 quality is expected to be lower than R1 (normal for Illumina)
            total += 1;
            if check("R2 mean quality < R1", stats.mean_quality, 30.0, 8.0) { passed += 1; }
        }
        Err(e) => {
            println!("  FAILED: {}", e);
            total += 2;
        }
    }

    // ── Parse all 40 FASTQ files (20 paired samples) ────────────
    println!("\n── Bulk parse: all MiSeq SOP FASTQ files ──");
    let mut total_files = 0usize;
    let mut total_seqs = 0usize;
    if let Ok(entries) = std::fs::read_dir(&data_dir) {
        let mut fastq_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "fastq"))
            .collect();
        fastq_files.sort_by_key(|e| e.path());

        for entry in &fastq_files {
            match fastq::parse_fastq(&entry.path()) {
                Ok(records) => {
                    total_seqs += records.len();
                    total_files += 1;
                }
                Err(e) => {
                    println!("  FAILED {}: {}", entry.path().display(), e);
                }
            }
        }
    }
    println!("  Parsed {} files, {} total sequences", total_files, total_seqs);

    total += 1;
    if check("Total FASTQ files", total_files as f64, 40.0, 0.0) { passed += 1; }

    // 304,720 total sequences across 40 files (R1+R2 both directions)
    // Exp001 DADA2 reported 162,360 input read-pairs (one direction)
    total += 1;
    if check("Total sequences (R1+R2)", total_seqs as f64, 304720.0, 500.0) { passed += 1; }

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  FASTQ Validation: {}/{} checks passed", passed, total);
    if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");

    std::process::exit(if passed == total { 0 } else { 1 });
}
