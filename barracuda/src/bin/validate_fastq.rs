// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate FASTQ parser against `MiSeq` SOP training data (Exp001 baseline).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Kozich et al. 2013. Appl. Environ. Microbiol. (`MiSeq` SOP) |
//! | DOI | 10.1128/AEM.01043-13 |
//! | Baseline tool | `FastQC` + Galaxy 24.1 |
//! | Baseline version | Galaxy 24.1, QIIME2 2026.1.0 |
//! | Baseline command | `scripts/validate_exp001.py` (Exp001) |
//! | Baseline date | 2026-02-19 |
//! | Data | Zenodo 800651 (`MiSeq` SOP, mouse gut 16S) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! # Expected values (from Galaxy `FastQC` + QIIME2 Exp001)
//!
//! - `F3D0_R1`: 7,793 sequences, 249–251 bp, GC ~54.7%, mean Q ~35.8
//! - All 40 files: 304,720 total sequences (20 paired samples)

use std::path::Path;
use wetspring_barracuda::bio::{derep, merge_pairs, quality};
use wetspring_barracuda::io::fastq;
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("wetSpring FASTQ Parser Validation");

    // Self-contained validations (no external data)
    validate_quality_filtering(&mut v);
    validate_merge_pairs(&mut v);
    validate_dereplication(&mut v);

    // External data-dependent validation
    let data_dir = validation::data_dir("WETSPRING_FASTQ_DIR", "data/validation/MiSeq_SOP");

    if data_dir.exists() {
        validate_r1(&data_dir, &mut v);
        validate_r2(&data_dir, &mut v);
        validate_bulk(&data_dir, &mut v);
    } else {
        println!(
            "\n  NOTE: MiSeq SOP data not found at {}\n  Set WETSPRING_FASTQ_DIR to enable Track 1 validation.",
            data_dir.display()
        );
    }

    v.finish();
}

fn validate_r1(data_dir: &Path, v: &mut Validator) {
    v.section("── F3D0_R1.fastq (forward reads) ──");
    let r1_path = data_dir.join("F3D0_R1.fastq");
    match fastq::parse_fastq(&r1_path) {
        Ok(records) => {
            let stats = fastq::compute_stats(&records);
            println!(
                "  Parsed {} sequences, {} total bases",
                stats.num_sequences, stats.total_bases
            );
            println!(
                "  Lengths: {}-{} bp (mean {:.1})",
                stats.min_length, stats.max_length, stats.mean_length
            );
            println!(
                "  GC: {:.1}%, Mean Q: {:.1}",
                stats.gc_content * 100.0,
                stats.mean_quality
            );

            v.check_count("Sequence count", stats.num_sequences, 7793);
            v.check_count("Min length", stats.min_length, 249);
            v.check_count("Max length", stats.max_length, 251);
            v.check("GC content", stats.gc_content, 0.547, 0.005);
            v.check("Mean quality", stats.mean_quality, 35.8, 0.5);
        }
        Err(e) => {
            println!("  FAILED: {e}");
            // Register 5 failed checks for the missing section
            for _ in 0..5 {
                v.check("(skipped — parse error)", 0.0, 1.0, 0.0);
            }
        }
    }
}

fn validate_r2(data_dir: &Path, v: &mut Validator) {
    v.section("── F3D0_R2.fastq (reverse reads) ──");
    let r2_path = data_dir.join("F3D0_R2.fastq");
    match fastq::parse_fastq(&r2_path) {
        Ok(records) => {
            let stats = fastq::compute_stats(&records);
            println!(
                "  Parsed {} sequences, {} total bases",
                stats.num_sequences, stats.total_bases
            );

            v.check_count("R2 sequence count", stats.num_sequences, 7793);
            // R2 mean quality baseline: 33.67 from Rust deterministic parse of
            // F3D0_R2.fastq (Zenodo 800651).  Phred33 mean across 7,793 reads.
            // Cross-validated against Galaxy FastQC report (Exp001, commit d71227d,
            // 2026-02-16) which reports per-base quality ~32-34 for R2 reverse reads.
            // Tolerance ±0.5 accounts for read-level Phred33 rounding.
            v.check("R2 mean quality", stats.mean_quality, 33.67, 0.5);
        }
        Err(e) => {
            println!("  FAILED: {e}");
            for _ in 0..2 {
                v.check("(skipped — parse error)", 0.0, 1.0, 0.0);
            }
        }
    }
}

fn validate_bulk(data_dir: &Path, v: &mut Validator) {
    v.section("── Bulk parse: all MiSeq SOP FASTQ files ──");
    let mut total_files = 0_usize;
    let mut total_seqs = 0_usize;
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        let mut fastq_files: Vec<_> = entries
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "fastq"))
            .collect();
        fastq_files.sort_by_key(std::fs::DirEntry::path);

        for entry in &fastq_files {
            match fastq::parse_fastq(&entry.path()) {
                Ok(records) => {
                    total_seqs += records.len();
                    total_files += 1;
                }
                Err(e) => {
                    println!("  FAILED {}: {e}", entry.path().display());
                }
            }
        }
    }
    println!("  Parsed {total_files} files, {total_seqs} total sequences");

    v.check_count("Total FASTQ files", total_files, 40);
    v.check_count("Total sequences (R1+R2)", total_seqs, 304_720);
}

fn validate_quality_filtering(v: &mut Validator) {
    v.section("── Quality filtering (Trimmomatic equivalent) ──");

    // Simulate reads with known quality profiles
    let good_read = fastq::FastqRecord {
        id: "good".to_string(),
        sequence: vec![b'A'; 100],
        quality: vec![33 + 30; 100], // all Q30
    };

    let bad_trailing = fastq::FastqRecord {
        id: "bad_trailing".to_string(),
        sequence: vec![b'A'; 100],
        quality: {
            let mut q = vec![33 + 30; 80]; // 80 good
            q.extend(vec![33 + 2; 20]); // 20 bad trailing
            q
        },
    };

    let all_bad = fastq::FastqRecord {
        id: "all_bad".to_string(),
        sequence: vec![b'A'; 100],
        quality: vec![33 + 2; 100], // all Q2
    };

    let short_good = fastq::FastqRecord {
        id: "short".to_string(),
        sequence: vec![b'A'; 30],
        quality: vec![33 + 30; 30], // good but short
    };

    let records = vec![good_read, bad_trailing, all_bad, short_good];

    let params = quality::QualityParams {
        window_size: 4,
        window_min_quality: 20,
        leading_min_quality: 3,
        trailing_min_quality: 3,
        min_length: 36,
        phred_offset: 33,
    };

    let (filtered, stats) = quality::filter_reads(&records, &params);

    println!(
        "  Input: {} reads → Output: {} reads ({} discarded)",
        stats.input_reads, stats.output_reads, stats.discarded_reads
    );

    v.check_count("Input reads", stats.input_reads, 4);
    v.check(
        "Filtered output >= 1",
        f64::from(u8::from(stats.output_reads >= 1)),
        1.0,
        0.0,
    );
    v.check(
        "Discarded >= 2",
        f64::from(u8::from(stats.discarded_reads >= 2)),
        1.0,
        0.0,
    );
    v.check(
        "All outputs >= min_length",
        f64::from(u8::from(filtered.iter().all(|r| r.sequence.len() >= 36))),
        1.0,
        0.0,
    );

    // Adapter trimming
    let with_adapter = fastq::FastqRecord {
        id: "adapter".to_string(),
        sequence: b"ACGTACGTACGTACGTACGTACGTACGTACGTAGATCGGAAGAG".to_vec(),
        quality: vec![33 + 30; 44],
    };
    let adapter = b"AGATCGGAAGAG";
    let (trimmed, found) = quality::trim_adapter_3prime(&with_adapter, adapter, 1, 8);
    v.check("Adapter found", f64::from(u8::from(found)), 1.0, 0.0);
    v.check(
        "Adapter trimmed correctly",
        f64::from(u8::from(
            trimmed.sequence.len() < with_adapter.sequence.len(),
        )),
        1.0,
        0.0,
    );
}

fn validate_merge_pairs(v: &mut Validator) {
    v.section("── Paired-end merging (VSEARCH equivalent) ──");

    // Simulate a 16S V4 amplicon: 253bp, 250bp reads, ~247bp overlap
    let amplicon: Vec<u8> = (0..253)
        .map(|i| match i % 4 {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            _ => b'T',
        })
        .collect();

    let fwd_seq = amplicon[..250].to_vec();
    let rev_region = &amplicon[3..]; // 250bp from position 3
    let rev_seq = merge_pairs::reverse_complement(rev_region);

    let fwd = fastq::FastqRecord {
        id: "test_pair".to_string(),
        sequence: fwd_seq,
        quality: vec![33 + 30; 250],
    };
    let rev = fastq::FastqRecord {
        id: "test_pair".to_string(),
        sequence: rev_seq,
        quality: vec![33 + 30; 250],
    };

    let result = merge_pairs::merge_pair(&fwd, &rev, &merge_pairs::MergeParams::default());
    let merged_ok = result.merged.is_some();
    let merged_len = result.merged.as_ref().map_or(0, |r| r.sequence.len());

    println!(
        "  Merged: {} ({} bp, overlap {} bp, {} mismatches)",
        if merged_ok { "YES" } else { "NO" },
        merged_len,
        result.overlap,
        result.mismatches,
    );

    v.check("Merge successful", f64::from(u8::from(merged_ok)), 1.0, 0.0);
    v.check_count("Merged length", merged_len, 253);
    v.check_count("Overlap", result.overlap, 247);
    v.check_count("Mismatches", result.mismatches, 0);

    // Batch test: 10 identical pairs
    let fwd_batch: Vec<_> = (0..10).map(|_| fwd.clone()).collect();
    let rev_batch: Vec<_> = (0..10).map(|_| rev.clone()).collect();
    let (_merged_batch, stats) =
        merge_pairs::merge_pairs(&fwd_batch, &rev_batch, &merge_pairs::MergeParams::default());

    v.check_count("Batch merged count", stats.merged_count, 10);
    v.check("Batch mean overlap", stats.mean_overlap, 247.0, 1.0);
}

fn validate_dereplication(v: &mut Validator) {
    v.section("── Dereplication (VSEARCH equivalent) ──");

    // Simulate community: 5 ASVs with known abundances
    let asvs = vec![
        (b"ACGTACGTACGTACGT" as &[u8], 50_usize),
        (b"GCTAGCTAGCTAGCTA", 30),
        (b"TTTTAAAACCCCGGGG", 15),
        (b"AACCGGTTAACCGGTT", 4),
        (b"GGGGCCCCAAAATTTT", 1),
    ];

    let mut records = Vec::new();
    for (seq, count) in &asvs {
        for _ in 0..*count {
            records.push(fastq::FastqRecord {
                id: format!("r{}", records.len()),
                sequence: seq.to_vec(),
                quality: vec![33 + 30; seq.len()],
            });
        }
    }

    let (uniques, stats) = derep::dereplicate(&records, derep::DerepSort::Abundance, 0);

    println!(
        "  {} input → {} unique ({} singletons, max abundance {})",
        stats.input_sequences, stats.unique_sequences, stats.singletons, stats.max_abundance,
    );

    v.check_count("Input sequences", stats.input_sequences, 100);
    v.check_count("Unique sequences", stats.unique_sequences, 5);
    v.check_count("Max abundance", stats.max_abundance, 50);
    v.check_count("Singletons", stats.singletons, 1);

    // Abundance sort: first should be most abundant
    v.check_count("Top ASV abundance", uniques[0].abundance, 50);

    // Singleton filtering
    let (filtered, fstats) = derep::dereplicate(&records, derep::DerepSort::Abundance, 2);
    v.check_count("After singleton removal", fstats.unique_sequences, 4);

    // FASTA output
    let fasta = derep::to_fasta_with_abundance(&filtered);
    v.check(
        "FASTA has size annotations",
        f64::from(u8::from(fasta.contains(";size="))),
        1.0,
        0.0,
    );
}
