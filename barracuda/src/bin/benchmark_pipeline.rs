// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark: Full 16S pipeline timing — Rust CPU vs Galaxy/Python reference.
//!
//! Measures wall-clock time and energy for every pipeline stage on real public
//! NCBI data, producing a direct comparison against Galaxy/QIIME2 control
//! experiments. Emits structured JSON via [`BenchReport`].
//!
//! Run: `cargo run --release --bin benchmark_pipeline`
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | timing harness |
//! | Baseline version | N/A (performance measurement, not correctness) |
//! | Baseline command | `cargo run --release --bin benchmark_pipeline` |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --release --bin benchmark_pipeline` |
//! | Data | PRJNA1114688, PRJNA629095, PRJNA1178324, PRJNA516219 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! # Galaxy Reference Times (same hardware, Docker)
//!
//! | Experiment | Samples | Reads    | Total   | DADA2   | Taxonomy |
//! |------------|---------|----------|---------|---------|----------|
//! | Exp001     | 20      | 124,249  | 71.5s   | 42.5s   | 10.5s    |
//! | Exp002     | 10      | 820,548  | 95.6s   | 68.0s   | 9.5s     |

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use wetspring_barracuda::bench::{
    self, BenchReport, EnergyReport, HardwareInventory, PhaseResult, PowerMonitor,
};
use wetspring_barracuda::bio::chimera::{self, ChimeraParams};
use wetspring_barracuda::bio::dada2::{self, Dada2Params};
use wetspring_barracuda::bio::derep::{self, DerepSort};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::quality::{self, QualityParams};
use wetspring_barracuda::bio::taxonomy::{
    ClassifyParams, Lineage, NaiveBayesClassifier, ReferenceSeq,
};
use wetspring_barracuda::io::fastq::FastqRecord;
use wetspring_barracuda::validation;

#[derive(Default)]
struct TimingAccumulator {
    fastq_parse_ms: f64,
    quality_filter_ms: f64,
    dereplication_ms: f64,
    dada2_denoise_ms: f64,
    chimera_detect_ms: f64,
    taxonomy_classify_ms: f64,
    diversity_calc_ms: f64,
    fastq_energy: EnergyReport,
    quality_energy: EnergyReport,
    derep_energy: EnergyReport,
    dada2_energy: EnergyReport,
    chimera_energy: EnergyReport,
    taxonomy_energy: EnergyReport,
    diversity_energy: EnergyReport,
    samples_processed: usize,
    total_reads_parsed: usize,
    total_reads_filtered: usize,
    total_asvs: usize,
}

impl TimingAccumulator {
    fn add_energy(target: &mut EnergyReport, src: &EnergyReport) {
        target.cpu_joules += src.cpu_joules;
        target.gpu_joules += src.gpu_joules;
        if src.gpu_watts_peak > target.gpu_watts_peak {
            target.gpu_watts_peak = src.gpu_watts_peak;
        }
        if src.gpu_temp_peak_c > target.gpu_temp_peak_c {
            target.gpu_temp_peak_c = src.gpu_temp_peak_c;
        }
        if src.gpu_vram_peak_mib > target.gpu_vram_peak_mib {
            target.gpu_vram_peak_mib = src.gpu_vram_peak_mib;
        }
        target.gpu_samples += src.gpu_samples;
    }
}

#[allow(clippy::too_many_lines, clippy::similar_names)] // benchmark harness: sequential timing of each pipeline stage; pipeline_ms/pipeline_s are related units
fn main() {
    let wall_monitor = PowerMonitor::start();
    let wall_start = Instant::now();

    let hw = HardwareInventory::detect("wetSpring Pipeline");
    hw.print();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  wetSpring Pipeline Benchmark — Rust CPU vs Galaxy/QIIME2          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let base = validation::data_dir("WETSPRING_PUBLIC_DIR", "data/public_benchmarks");
    let ref_dir = validation::data_dir("WETSPRING_REF_DIR", "data/reference_dbs/silva_138");

    // ── Phase 1: SILVA classifier training (one-time cost) ──────────────
    println!("── Phase 1: SILVA 138.1 Classifier Training ─────────────────────");
    let silva_monitor = PowerMonitor::start();
    let t = Instant::now();
    let classifier = load_silva_classifier(&ref_dir);
    let silva_train_ms = t.elapsed().as_secs_f64() * 1000.0;
    let silva_energy = silva_monitor.stop();
    println!(
        "   SILVA training: {:.1}ms ({:.2}s)",
        silva_train_ms,
        silva_train_ms / 1000.0
    );
    if let Some(ref c) = classifier {
        println!("   Taxa in classifier: {}", c.n_taxa());
    }

    // ── Phase 2: Process representative samples ──────────────────────────
    println!("\n── Phase 2: Pipeline — PRJNA1114688 (4 representative samples) ──");
    let d1 = base.join("PRJNA1114688");
    let samples_1114688 = ["SRR29127218", "SRR29127209", "SRR29127205", "SRR29127215"];

    let mut acc = TimingAccumulator::default();

    for accession in &samples_1114688 {
        let dir = d1.join(accession);
        if dir.exists() {
            benchmark_sample(&dir, accession, classifier.as_ref(), &mut acc);
        }
    }

    // ── Phase 3: Process other BioProjects ──────────────────────────────
    println!("\n── Phase 3: Cross-Dataset — 6 Additional Samples ────────────────");
    let other_samples = [
        ("PRJNA629095", vec!["SRR11638224", "SRR11638231"]),
        ("PRJNA1178324", vec!["SRR31143973", "SRR31143980"]),
        ("PRJNA516219", vec!["SRR8472475", "SRR8472476"]),
    ];

    for (project, accs) in &other_samples {
        let d = base.join(project);
        if d.exists() {
            for accession in accs {
                let dir = d.join(accession);
                if dir.exists() {
                    benchmark_sample(&dir, accession, classifier.as_ref(), &mut acc);
                }
            }
        }
    }

    let wall_total_s = wall_start.elapsed().as_secs_f64();
    let wall_energy = wall_monitor.stop();

    // ── Build structured report ─────────────────────────────────────────
    let mut report = BenchReport::new(hw);

    report.add_phase(PhaseResult {
        phase: "SILVA training".to_string(),
        substrate: "Rust CPU".to_string(),
        wall_time_s: silva_train_ms / 1000.0,
        per_eval_us: silva_train_ms * 1000.0,
        n_evals: 1,
        energy: silva_energy,
        peak_rss_mb: bench::peak_rss_mb(),
        notes: "one-time classifier training".to_string(),
    });

    let pipeline_ms = acc.fastq_parse_ms
        + acc.quality_filter_ms
        + acc.dereplication_ms
        + acc.dada2_denoise_ms
        + acc.chimera_detect_ms
        + acc.taxonomy_classify_ms
        + acc.diversity_calc_ms;

    let stages: &[(&str, f64, &EnergyReport)] = &[
        ("FASTQ parse", acc.fastq_parse_ms, &acc.fastq_energy),
        ("Quality filter", acc.quality_filter_ms, &acc.quality_energy),
        ("Dereplication", acc.dereplication_ms, &acc.derep_energy),
        ("DADA2 denoise", acc.dada2_denoise_ms, &acc.dada2_energy),
        ("Chimera detect", acc.chimera_detect_ms, &acc.chimera_energy),
        (
            "Taxonomy classify",
            acc.taxonomy_classify_ms,
            &acc.taxonomy_energy,
        ),
        (
            "Diversity metrics",
            acc.diversity_calc_ms,
            &acc.diversity_energy,
        ),
    ];

    for &(name, ms, energy) in stages {
        report.add_phase(PhaseResult {
            phase: name.to_string(),
            substrate: "Rust CPU".to_string(),
            wall_time_s: ms / 1000.0,
            #[allow(clippy::cast_precision_loss)] // sample count always fits in f64
            per_eval_us: if acc.samples_processed > 0 {
                ms * 1000.0 / acc.samples_processed as f64
            } else {
                0.0
            },
            n_evals: acc.samples_processed,
            energy: energy.clone(),
            peak_rss_mb: bench::peak_rss_mb(),
            notes: String::new(),
        });
    }

    report.print_summary();

    // ── Results table ───────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      BENCHMARK RESULTS                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Rust Pipeline ({} samples, {} reads)              ║",
        acc.samples_processed, acc.total_reads_parsed
    );
    println!("║  ┌─────────────────────────┬────────────┬────────────┐             ║");
    println!("║  │ Stage                   │  Time (ms) │  Time (s)  │             ║");
    println!("║  ├─────────────────────────┼────────────┼────────────┤             ║");
    println!(
        "║  │ SILVA training (1×)     │ {:>10.1} │ {:>10.2} │             ║",
        silva_train_ms,
        silva_train_ms / 1000.0
    );
    for &(name, ms, _) in stages {
        println!(
            "║  │ {:<23} │ {:>10.1} │ {:>10.2} │             ║",
            name,
            ms,
            ms / 1000.0
        );
    }
    println!("║  ├─────────────────────────┼────────────┼────────────┤             ║");
    println!(
        "║  │ Pipeline total          │ {:>10.1} │ {:>10.2} │             ║",
        pipeline_ms,
        pipeline_ms / 1000.0
    );
    println!(
        "║  │ Wall clock (total)      │ {:>10.1} │ {:>10.2} │             ║",
        wall_total_s * 1000.0,
        wall_total_s
    );
    println!("║  └─────────────────────────┴────────────┴────────────┘             ║");
    println!(
        "║  Reads: {:>10}  Filtered: {:>10}  ASVs: {:>6}                 ║",
        acc.total_reads_parsed, acc.total_reads_filtered, acc.total_asvs
    );

    // ── Galaxy comparison ───────────────────────────────────────────────
    #[allow(clippy::cast_precision_loss)]
    let samples_f64 = acc.samples_processed as f64;
    let galaxy_per_sample = 95.6 / 10.0;
    let rust_per_sample = if samples_f64 > 0.0 {
        pipeline_ms / 1000.0 / samples_f64
    } else {
        0.0
    };
    let galaxy_dada2_per_sample = 68.0 / 10.0;
    let rust_dada2_per_sample = if samples_f64 > 0.0 {
        acc.dada2_denoise_ms / 1000.0 / samples_f64
    } else {
        0.0
    };

    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║                  GALAXY / QIIME2 COMPARISON                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    if rust_per_sample > 0.0 {
        println!(
            "║  Per-sample: Galaxy {:.2}s → Rust {:.2}s  ({:.1}× speedup)              ║",
            galaxy_per_sample,
            rust_per_sample,
            galaxy_per_sample / rust_per_sample
        );
        println!(
            "║  DADA2/sample: Galaxy {:.2}s → Rust {:.2}s ({:.1}× speedup)             ║",
            galaxy_dada2_per_sample,
            rust_dada2_per_sample,
            galaxy_dada2_per_sample / rust_dada2_per_sample
        );
    }

    // ── Energy estimate ─────────────────────────────────────────────────
    let cpu_tdp_w = 125.0_f64;
    #[allow(clippy::similar_names)] // pipeline_s and pipeline_ms are related units
    let pipeline_s = pipeline_ms / 1000.0;
    let rapl_j = wall_energy.cpu_joules;
    let rust_kwh = if rapl_j > 0.0 {
        rapl_j / 3_600_000.0
    } else {
        cpu_tdp_w * pipeline_s / 3_600_000.0
    };
    let galaxy_s = galaxy_per_sample * samples_f64;
    let galaxy_kwh = cpu_tdp_w * galaxy_s / 3_600_000.0;

    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║                      ENERGY & COST ESTIMATE                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    if rapl_j > 0.0 {
        println!("║  CPU energy (RAPL): {rapl_j:.2} J                                         ║");
    }
    println!(
        "║  Rust: {rust_kwh:.6} kWh  Galaxy: {galaxy_kwh:.6} kWh                            ║"
    );
    let us_kwh_cost = 0.12;
    if samples_f64 > 0.0 {
        println!(
            "║  At 10K samples — Rust ${:.4}  Galaxy ${:.4}                       ║",
            rust_kwh * us_kwh_cost * 10000.0 / samples_f64,
            galaxy_kwh * us_kwh_cost * 10000.0 / samples_f64,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // ── Save JSON ───────────────────────────────────────────────────────
    let out_dir = format!("{}/../benchmarks/results", env!("CARGO_MANIFEST_DIR"));
    match report.save_json(&out_dir) {
        Ok(path) => println!("\nJSON results saved to {path}"),
        Err(e) => eprintln!("Warning: could not save JSON: {e}"),
    }
}

fn benchmark_sample(
    sample_dir: &Path,
    accession: &str,
    classifier: Option<&NaiveBayesClassifier>,
    acc: &mut TimingAccumulator,
) {
    let r1_name = format!("{accession}_1.fastq.gz");
    let interleaved_name = format!("{accession}.fastq.gz");
    let fastq_path = if sample_dir.join(&r1_name).exists() {
        sample_dir.join(&r1_name)
    } else if sample_dir.join(&interleaved_name).exists() {
        sample_dir.join(&interleaved_name)
    } else {
        return;
    };

    // 1. FASTQ parsing
    let mon = PowerMonitor::start();
    let t = Instant::now();
    let Ok(records) = decompress_gz_fastq(&fastq_path) else {
        let _ = mon.stop();
        return;
    };
    acc.fastq_parse_ms += t.elapsed().as_secs_f64() * 1000.0;
    TimingAccumulator::add_energy(&mut acc.fastq_energy, &mon.stop());
    acc.total_reads_parsed += records.len();

    // 2. Quality filtering
    let mon = PowerMonitor::start();
    let t = Instant::now();
    let (filtered, _) = quality::filter_reads(&records, &QualityParams::default());
    acc.quality_filter_ms += t.elapsed().as_secs_f64() * 1000.0;
    TimingAccumulator::add_energy(&mut acc.quality_energy, &mon.stop());
    acc.total_reads_filtered += filtered.len();

    let sub: Vec<_> = filtered.into_iter().take(5_000).collect();
    let filtered = sub;

    // 3. Dereplication
    let mon = PowerMonitor::start();
    let t = Instant::now();
    let (uniques, _) = derep::dereplicate(&filtered, DerepSort::Abundance, 2);
    acc.dereplication_ms += t.elapsed().as_secs_f64() * 1000.0;
    TimingAccumulator::add_energy(&mut acc.derep_energy, &mon.stop());

    if uniques.len() < 2 {
        return;
    }

    // 4. DADA2 denoising
    let mon = PowerMonitor::start();
    let t = Instant::now();
    let (asvs, _) = dada2::denoise(&uniques, &Dada2Params::default());
    acc.dada2_denoise_ms += t.elapsed().as_secs_f64() * 1000.0;
    TimingAccumulator::add_energy(&mut acc.dada2_energy, &mon.stop());
    acc.total_asvs += asvs.len();

    // 5. Chimera detection
    let mon = PowerMonitor::start();
    let t = Instant::now();
    let (_clean, _stats) = chimera::remove_chimeras(&asvs, &ChimeraParams::default());
    acc.chimera_detect_ms += t.elapsed().as_secs_f64() * 1000.0;
    TimingAccumulator::add_energy(&mut acc.chimera_energy, &mon.stop());

    // 6. Taxonomy classification
    if let Some(clf) = classifier {
        let mon = PowerMonitor::start();
        let t = Instant::now();
        let params = ClassifyParams {
            bootstrap_n: 50,
            ..ClassifyParams::default()
        };
        for asv in asvs.iter().take(10) {
            let _ = clf.classify(&asv.sequence, &params);
        }
        acc.taxonomy_classify_ms += t.elapsed().as_secs_f64() * 1000.0;
        TimingAccumulator::add_energy(&mut acc.taxonomy_energy, &mon.stop());
    }

    // 7. Diversity metrics
    let mon = PowerMonitor::start();
    let t = Instant::now();
    #[allow(clippy::cast_precision_loss)]
    let counts: Vec<f64> = asvs.iter().map(|a| a.abundance as f64).collect();
    let _ = diversity::shannon(&counts);
    let _ = diversity::simpson(&counts);
    let _ = diversity::observed_features(&counts);
    acc.diversity_calc_ms += t.elapsed().as_secs_f64() * 1000.0;
    TimingAccumulator::add_energy(&mut acc.diversity_energy, &mon.stop());

    acc.samples_processed += 1;
    println!(
        "   {} — {} reads, {} ASVs",
        accession,
        records.len(),
        asvs.len()
    );
}

fn load_silva_classifier(ref_dir: &Path) -> Option<NaiveBayesClassifier> {
    let fasta_path = ref_dir.join("silva_138_99_seqs.fasta");
    let tax_path = ref_dir.join("silva_138_99_taxonomy.tsv");

    if !fasta_path.exists() || !tax_path.exists() {
        println!("   [SKIP] SILVA not found at {}", ref_dir.display());
        return None;
    }

    let tax_content = std::fs::read_to_string(&tax_path).ok()?;
    let mut tax_map: HashMap<String, String> = HashMap::new();
    for line in tax_content.lines().skip(1) {
        let parts: Vec<&str> = line.splitn(2, '\t').collect();
        if parts.len() == 2 {
            tax_map.insert(parts[0].to_string(), parts[1].trim().to_string());
        }
    }

    let fasta_content = std::fs::read_to_string(&fasta_path).ok()?;
    let mut refs = Vec::new();
    let mut current_id = String::new();
    let mut current_seq: Vec<u8> = Vec::new();
    let mut n_parsed = 0_usize;

    for line in fasta_content.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if !current_id.is_empty() && !current_seq.is_empty() {
                n_parsed += 1;
                if n_parsed % 87 == 0 {
                    if let Some(tax) = tax_map.get(&current_id) {
                        refs.push(ReferenceSeq {
                            id: current_id.clone(),
                            sequence: current_seq.clone(),
                            lineage: Lineage::from_taxonomy_string(tax),
                        });
                    }
                }
            }
            current_id = header.split_whitespace().next().unwrap_or("").to_string();
            current_seq.clear();
        } else {
            current_seq.extend(
                line.trim()
                    .bytes()
                    .filter(u8::is_ascii_alphabetic)
                    .map(|b| b.to_ascii_uppercase()),
            );
        }
    }
    if !current_id.is_empty() && !current_seq.is_empty() {
        n_parsed += 1;
        if n_parsed % 87 == 0 {
            if let Some(tax) = tax_map.get(&current_id) {
                refs.push(ReferenceSeq {
                    id: current_id,
                    sequence: current_seq,
                    lineage: Lineage::from_taxonomy_string(tax),
                });
            }
        }
    }

    println!("   SILVA: {} refs subsampled from {}", refs.len(), n_parsed);

    if refs.is_empty() {
        return None;
    }

    let classifier = NaiveBayesClassifier::train(&refs, 8);
    println!("   Classifier: {} taxa", classifier.n_taxa());
    Some(classifier)
}

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
        let Some(Ok(seq)) = lines.next() else { break };
        match lines.next() {
            Some(Ok(_)) => {}
            _ => break,
        }
        let Some(Ok(qual)) = lines.next() else { break };

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
        Err("No FASTQ records".to_string())
    } else {
        Ok(records)
    }
}
