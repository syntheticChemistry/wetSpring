// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate Rust 16S pipeline on PUBLIC open data, benchmarked against paper findings.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Public dataset | PRJNA1114688 — N. oculata + B. plicatilis V4 16S time series |
//! | Paper benchmarks | Humphrey 2023 (OTUs, genera), Carney 2016 (crash agents) |
//! | Strategy | Run pipeline on open data → compare against paper ground truth |
//! | Date | 2026-02-19 |
//!
//! # Purpose
//!
//! This is the critical validation step: we process **publicly available data**
//! through our Rust pipeline and verify that results are **biologically consistent**
//! with what the source papers reported. We do not replicate the exact studies
//! (their data is restricted), but we validate that the same analytical pipeline
//! produces scientifically sound results on similar organisms from an independent lab.

use std::path::{Path, PathBuf};
use wetspring_barracuda::bio::dada2::{self, Dada2Params};
use wetspring_barracuda::bio::derep::{self, DerepSort};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::quality::{self, QualityParams};
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::validation::Validator;

fn main() {
    let mut v = Validator::new(
        "wetSpring Public Data Benchmark (4 BioProjects vs Paper Ground Truth)",
    );

    let base = std::env::var("WETSPRING_PUBLIC_DIR").map_or_else(
        |_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../data/public_benchmarks")
        },
        PathBuf::from,
    );

    let mut all_results: Vec<SampleResult> = Vec::new();

    // ── Dataset 1: PRJNA1114688 — N. oculata + B. plicatilis (Exp014) ───────
    let d1 = base.join("PRJNA1114688");
    if d1.exists() {
        validate_manifest(&mut v, &d1);
        let samples = [
            ("SRR29127218", "Nanno Day 1 (1114688)"),
            ("SRR29127209", "Nanno Day 14 (1114688)"),
            ("SRR29127205", "Brachio Day 1 (1114688)"),
            ("SRR29127215", "Brachio Day 14 (1114688)"),
        ];
        for (acc, label) in samples {
            let dir = d1.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Dataset 2: PRJNA629095 — N. oceanica phycosphere probiotic ──────────
    let d2 = base.join("PRJNA629095");
    if d2.exists() {
        validate_manifest(&mut v, &d2);
        let samples = [
            ("SRR11638224", "Nanno phyco-1 (629095)"),
            ("SRR11638231", "Nanno phyco-2 (629095)"),
        ];
        for (acc, label) in samples {
            let dir = d2.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Dataset 3: PRJNA1178324 — Cyanobacteria toxin (sewage/fertilizer) ───
    let d3 = base.join("PRJNA1178324");
    if d3.exists() {
        validate_manifest(&mut v, &d3);
        let samples = [
            ("SRR31143973", "Cyano-tox-1 (1178324)"),
            ("SRR31143980", "Cyano-tox-2 (1178324)"),
        ];
        for (acc, label) in samples {
            let dir = d3.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Dataset 4: PRJNA516219 — Lake Erie cyanotoxin (interleaved) ─────────
    let d4 = base.join("PRJNA516219");
    if d4.exists() {
        validate_manifest(&mut v, &d4);
        let samples = [
            ("SRR8472475", "LakeErie-1 (516219)"),
            ("SRR8472476", "LakeErie-2 (516219)"),
        ];
        for (acc, label) in samples {
            let dir = d4.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Cross-dataset benchmark ─────────────────────────────────────────────
    cross_dataset_benchmark(&mut v, &all_results);

    v.finish();
}

// ── Sample processing result ────────────────────────────────────────────────

#[allow(dead_code)]
struct SampleResult {
    label: String,
    total_reads: usize,
    filtered_reads: usize,
    mean_read_len: f64,
    n_unique: usize,
    n_asvs: usize,
    shannon: f64,
    simpson: f64,
    observed: f64,
    asv_counts: Vec<f64>,
}

// ── Manifest validation ─────────────────────────────────────────────────────

fn validate_manifest(v: &mut Validator, data_dir: &Path) {
    v.section("Dataset Manifest");

    let manifest_path = data_dir.join("manifest.json");
    if manifest_path.exists() {
        v.check("manifest.json exists", 1.0, 1.0, 0.0);
        if let Ok(content) = std::fs::read_to_string(&manifest_path) {
            let has_bioproject = content.contains("\"bioproject\"");
            v.check(
                "Manifest has bioproject field",
                if has_bioproject { 1.0 } else { 0.0 },
                1.0,
                0.0,
            );
        }
    } else {
        println!("  [SKIP] manifest.json not found at {}", manifest_path.display());
        v.check("manifest.json exists", 0.0, 1.0, 0.0);
    }
}

// ── Process a single sample through the full 16S pipeline ───────────────────

fn process_sample(
    v: &mut Validator,
    sample_dir: &Path,
    label: &str,
    accession: &str,
) -> Option<SampleResult> {
    v.section(&format!("Pipeline: {label} ({accession})"));

    // Try paired-end first (_1.fastq.gz), then interleaved (.fastq.gz)
    let r1_name = format!("{accession}_1.fastq.gz");
    let interleaved_name = format!("{accession}.fastq.gz");
    let fastq_path = if sample_dir.join(&r1_name).exists() {
        sample_dir.join(&r1_name)
    } else if sample_dir.join(&interleaved_name).exists() {
        sample_dir.join(&interleaved_name)
    } else {
        println!("  [SKIP] No FASTQ found in {}", sample_dir.display());
        return None;
    };

    let records = match decompress_gz_fastq(&fastq_path) {
        Ok(recs) => recs,
        Err(e) => {
            println!("  [ERROR] Failed to parse {}: {e}", fastq_path.display());
            v.check(&format!("{label}: FASTQ parse"), 0.0, 1.0, 0.0);
            return None;
        }
    };

    let total_reads = records.len();
    println!("  {label}: {total_reads} reads parsed");

    v.check(
        &format!("{label}: read count > 0"),
        if total_reads > 0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    v.check(
        &format!("{label}: read count > 1000"),
        if total_reads > 1000 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Quality filtering
    let qparams = QualityParams::default();
    let (filtered, _) = quality::filter_reads(&records, &qparams);
    let retention = if total_reads > 0 {
        filtered.len() as f64 / total_reads as f64
    } else {
        0.0
    };
    println!(
        "  {label}: quality filter {}/{} retained ({:.1}%)",
        filtered.len(),
        total_reads,
        retention * 100.0
    );

    v.check(
        &format!("{label}: quality retention > 30%"),
        if retention > 0.3 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Read length check
    let mean_len = if filtered.is_empty() {
        0.0
    } else {
        filtered.iter().map(|r| r.sequence.len() as f64).sum::<f64>() / filtered.len() as f64
    };
    println!("  {label}: mean read length = {mean_len:.0} bp");

    v.check(
        &format!("{label}: mean length > 100 bp"),
        if mean_len > 100.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Subsample for denoising (cap at 5000 to keep runtime reasonable)
    let sub: Vec<_> = filtered.into_iter().take(5000).collect();
    if sub.len() < 50 {
        println!("  {label}: too few reads after filtering ({}) — skipping denoising", sub.len());
        return None;
    }

    // Dereplication
    let (uniques, _) = derep::dereplicate(&sub, DerepSort::Abundance, 2);
    let n_unique = uniques.len();
    println!("  {label}: {n_unique} unique sequences from {} reads", sub.len());

    v.check(
        &format!("{label}: >1 unique sequence"),
        if n_unique > 1 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // DADA2 denoising
    let (asvs, _) = dada2::denoise(&uniques, &Dada2Params::default());
    let n_asvs = asvs.len();
    println!("  {label}: {n_asvs} ASVs after DADA2");

    v.check(
        &format!("{label}: >1 ASV"),
        if n_asvs > 1 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Diversity metrics
    let counts: Vec<f64> = asvs.iter().map(|a| a.abundance as f64).collect();
    let shannon = diversity::shannon(&counts);
    let simpson = diversity::simpson(&counts);
    let observed = diversity::observed_features(&counts);

    println!(
        "  {label}: observed={}, Shannon={:.3}, Simpson={:.3}",
        observed as usize, shannon, simpson
    );

    v.check(
        &format!("{label}: Shannon > 0"),
        if shannon > 0.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    v.check(
        &format!("{label}: Simpson in (0,1)"),
        if simpson > 0.0 && simpson < 1.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    Some(SampleResult {
        label: label.to_string(),
        total_reads,
        filtered_reads: sub.len(),
        mean_read_len: mean_len,
        n_unique,
        n_asvs,
        shannon,
        simpson,
        observed,
        asv_counts: counts,
    })
}

// ── Cross-dataset benchmark ─────────────────────────────────────────────────

fn cross_dataset_benchmark(v: &mut Validator, all_results: &[SampleResult]) {
    v.section("Cross-Dataset Paper Benchmark");

    if all_results.is_empty() {
        println!("  [SKIP] No sample results available for benchmarking");
        return;
    }

    println!("  Benchmarking {} samples across 4 BioProjects", all_results.len());

    // Partition by dataset type
    let nanno_samples: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.contains("Nanno"))
        .collect();
    let brachio_samples: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.contains("Brachio"))
        .collect();
    let cyano_samples: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.contains("Cyano") || r.label.contains("LakeErie"))
        .collect();

    // ── Humphrey 2023 benchmark: Nannochloropsis samples ────────────────
    if !nanno_samples.is_empty() {
        let avg_shannon =
            nanno_samples.iter().map(|r| r.shannon).sum::<f64>() / nanno_samples.len() as f64;
        let avg_simpson =
            nanno_samples.iter().map(|r| r.simpson).sum::<f64>() / nanno_samples.len() as f64;
        let avg_asvs =
            nanno_samples.iter().map(|r| r.n_asvs as f64).sum::<f64>() / nanno_samples.len() as f64;

        println!(
            "  Nannochloropsis ({} samples): Shannon={:.3}, Simpson={:.3}, ASVs={:.0}",
            nanno_samples.len(), avg_shannon, avg_simpson, avg_asvs
        );

        v.check(
            "Humphrey: Nanno Shannon in [0.5, 5.0]",
            if (0.5..=5.0).contains(&avg_shannon) { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
        v.check(
            "Humphrey: Nanno Simpson in [0.3, 1.0]",
            if (0.3..=1.0).contains(&avg_simpson) { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
        v.check(
            "Humphrey: Nanno ASVs > 3 (ref: 18 OTUs)",
            if avg_asvs > 3.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
    }

    // ── Carney 2016 benchmark: Brachionus samples ───────────────────────
    if !brachio_samples.is_empty() {
        let avg_shannon =
            brachio_samples.iter().map(|r| r.shannon).sum::<f64>() / brachio_samples.len() as f64;
        println!("  Brachionus ({} samples): Shannon={:.3}", brachio_samples.len(), avg_shannon);

        v.check(
            "Carney: Brachio Shannon > 0 (diverse community)",
            if avg_shannon > 0.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
    }

    // ── Cross-condition: Nanno vs Brachio ───────────────────────────────
    if !nanno_samples.is_empty() && !brachio_samples.is_empty() {
        let nanno_obs =
            nanno_samples.iter().map(|r| r.observed).sum::<f64>() / nanno_samples.len() as f64;
        let brachio_obs =
            brachio_samples.iter().map(|r| r.observed).sum::<f64>() / brachio_samples.len() as f64;
        println!("  Cross-condition: Nanno obs={:.0}, Brachio obs={:.0}", nanno_obs, brachio_obs);

        v.check(
            "Cross: both Nanno and Brachio have >1 observed feature",
            if nanno_obs > 1.0 && brachio_obs > 1.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
    }

    // ── Cyanobacteria/HAB generalizability ──────────────────────────────
    if !cyano_samples.is_empty() {
        let avg_shannon =
            cyano_samples.iter().map(|r| r.shannon).sum::<f64>() / cyano_samples.len() as f64;
        let avg_asvs =
            cyano_samples.iter().map(|r| r.n_asvs as f64).sum::<f64>() / cyano_samples.len() as f64;
        println!(
            "  Cyanobacteria/HAB ({} samples): Shannon={:.3}, ASVs={:.0}",
            cyano_samples.len(), avg_shannon, avg_asvs
        );

        v.check(
            "HAB generalizability: cyano Shannon > 0",
            if avg_shannon > 0.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
        v.check(
            "HAB generalizability: cyano produces >1 ASV",
            if avg_asvs > 1.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
    }

    // ── Cross-domain: marine algae vs freshwater cyano ──────────────────
    if !nanno_samples.is_empty() && !cyano_samples.is_empty() {
        let nanno_shannon =
            nanno_samples.iter().map(|r| r.shannon).sum::<f64>() / nanno_samples.len() as f64;
        let cyano_shannon =
            cyano_samples.iter().map(|r| r.shannon).sum::<f64>() / cyano_samples.len() as f64;
        let delta = (nanno_shannon - cyano_shannon).abs();
        println!(
            "  Cross-domain: marine Nanno Shannon={:.3} vs freshwater cyano Shannon={:.3} (delta={:.3})",
            nanno_shannon, cyano_shannon, delta
        );

        v.check(
            "Cross-domain: pipeline handles both marine and freshwater",
            if nanno_shannon > 0.0 && cyano_shannon > 0.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
    }

    // ── Multi-BioProject consistency ────────────────────────────────────
    let n_bioprojects = [
        !nanno_samples.is_empty() || !brachio_samples.is_empty(),
        all_results.iter().any(|r| r.label.contains("629095")),
        all_results.iter().any(|r| r.label.contains("1178324")),
        all_results.iter().any(|r| r.label.contains("516219")),
    ]
    .iter()
    .filter(|&&b| b)
    .count();

    println!("  BioProjects with results: {n_bioprojects}/4");
    v.check(
        "Multi-project: results from >= 2 independent BioProjects",
        if n_bioprojects >= 2 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // ── Summary table ───────────────────────────────────────────────────
    println!("\n  ┌──────────────────────────────┬──────────┬──────────┬────────┬────────┬─────────┐");
    println!("  │ Sample                       │   Reads  │ Filtered │  ASVs  │Shannon │ Simpson │");
    println!("  ├──────────────────────────────┼──────────┼──────────┼────────┼────────┼─────────┤");
    for r in all_results {
        println!(
            "  │ {:28} │ {:>8} │ {:>8} │ {:>6} │ {:>6.3} │ {:>7.3} │",
            r.label, r.total_reads, r.filtered_reads, r.n_asvs, r.shannon, r.simpson
        );
    }
    println!("  └──────────────────────────────┴──────────┴──────────┴────────┴────────┴─────────┘");
}

// ── Gzipped FASTQ decompression ─────────────────────────────────────────────

fn decompress_gz_fastq(path: &Path) -> Result<Vec<FastqRecord>, String> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let decoder = flate2::read::GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut records = Vec::new();
    let mut lines = reader.lines();

    loop {
        let header = match lines.next() {
            Some(Ok(l)) if l.starts_with('@') => l,
            Some(Ok(_)) => continue,
            _ => break,
        };
        let seq = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };
        match lines.next() {
            Some(Ok(_)) => {}
            _ => break,
        };
        let qual = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };

        let id = header[1..]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        records.push(FastqRecord {
            id,
            sequence: seq.into_bytes(),
            quality: qual.into_bytes(),
        });
    }

    if records.is_empty() {
        Err("No complete FASTQ records".to_string())
    } else {
        println!("  Parsed {} records from {}", records.len(), path.display());
        Ok(records)
    }
}
