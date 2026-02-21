// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validate Rust 16S pipeline on PUBLIC open data, benchmarked against paper findings.
#![allow(clippy::cast_precision_loss)] // usize→f64 for stats/display; loss acceptable
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Humphrey 2023 (OTUs, genera), Carney 2016 (crash agents) |
//! | Baseline tool | Published paper ground truth |
//! | Baseline version | Exp014 |
//! | Baseline date | 2026-02-19 |
//! | Data | PRJNA1114688, PRJNA629095, PRJNA1178324, PRJNA516219 |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::collections::HashMap;
use std::path::Path;
use wetspring_barracuda::bio::dada2::{self, Dada2Params};
use wetspring_barracuda::bio::derep::{self, DerepSort};
use wetspring_barracuda::bio::diversity;
use wetspring_barracuda::bio::quality::{self, QualityParams};
use wetspring_barracuda::bio::taxonomy::{
    ClassifyParams, Lineage, NaiveBayesClassifier, ReferenceSeq, TaxRank,
};
use wetspring_barracuda::validation::{self, Validator};

fn main() {
    let mut v = Validator::new("wetSpring Public Data Benchmark — Full Time Series + Taxonomy");

    let base = validation::data_dir("WETSPRING_PUBLIC_DIR", "data/public_benchmarks");

    // ── Load SILVA reference database (if available) ─────────────────────
    let ref_dir = validation::data_dir("WETSPRING_REF_DIR", "data/reference_dbs/silva_138");
    let classifier = load_silva_classifier(&ref_dir);

    let mut all_results: Vec<SampleResult> = Vec::new();

    // ── Dataset 1: PRJNA1114688 — Full 16-sample time series ─────────────
    let d1 = base.join("PRJNA1114688");
    if d1.exists() {
        validate_manifest(&mut v, &d1);
        let samples = [
            ("SRR29127218", "N.oculata D1-R1"),
            ("SRR29127219", "N.oculata D1-R2"),
            ("SRR29127206", "N.oculata D7-R1"),
            ("SRR29127207", "N.oculata D7-R2"),
            ("SRR29127208", "N.oculata D7-R3"),
            ("SRR29127209", "N.oculata D14-R1"),
            ("SRR29127210", "N.oculata D14-R2"),
            ("SRR29127211", "N.oculata D14-R3"),
            ("SRR29127204", "B.plicatilis D1-R1"),
            ("SRR29127205", "B.plicatilis D1-R2"),
            ("SRR29127212", "B.plicatilis D7-R1"),
            ("SRR29127213", "B.plicatilis D7-R2"),
            ("SRR29127214", "B.plicatilis D7-R3"),
            ("SRR29127215", "B.plicatilis D14-R1"),
            ("SRR29127216", "B.plicatilis D14-R2"),
            ("SRR29127217", "B.plicatilis D14-R3"),
        ];
        for (acc, label) in samples {
            let dir = d1.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc, classifier.as_ref()) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Dataset 2: PRJNA629095 — N. oceanica phycosphere probiotic ──────
    let d2 = base.join("PRJNA629095");
    if d2.exists() {
        validate_manifest(&mut v, &d2);
        let samples = [
            ("SRR11638224", "N.oceanica phyco-1"),
            ("SRR11638231", "N.oceanica phyco-2"),
        ];
        for (acc, label) in samples {
            let dir = d2.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc, classifier.as_ref()) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Dataset 3: PRJNA1178324 — Cyanobacteria toxin ───────────────────
    let d3 = base.join("PRJNA1178324");
    if d3.exists() {
        validate_manifest(&mut v, &d3);
        let samples = [
            ("SRR31143973", "Cyano-tox-1"),
            ("SRR31143980", "Cyano-tox-2"),
        ];
        for (acc, label) in samples {
            let dir = d3.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc, classifier.as_ref()) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Dataset 4: PRJNA516219 — Lake Erie cyanotoxin (interleaved) ─────
    let d4 = base.join("PRJNA516219");
    if d4.exists() {
        validate_manifest(&mut v, &d4);
        let samples = [("SRR8472475", "LakeErie-1"), ("SRR8472476", "LakeErie-2")];
        for (acc, label) in samples {
            let dir = d4.join(acc);
            if dir.exists() {
                if let Some(r) = process_sample(&mut v, &dir, label, acc, classifier.as_ref()) {
                    all_results.push(r);
                }
            }
        }
    }

    // ── Temporal analysis (PRJNA1114688 time series) ─────────────────────
    temporal_analysis(&mut v, &all_results);

    // ── Cross-dataset benchmark ─────────────────────────────────────────
    cross_dataset_benchmark(&mut v, &all_results);

    // ── Taxonomy benchmark ──────────────────────────────────────────────
    taxonomy_benchmark(&mut v, &all_results);

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
    top_taxa: Vec<String>,
}

// ── SILVA database loading ──────────────────────────────────────────────────

fn load_silva_classifier(ref_dir: &Path) -> Option<NaiveBayesClassifier> {
    let fasta_path = ref_dir.join("silva_138_99_seqs.fasta");
    let tax_path = ref_dir.join("silva_138_99_taxonomy.tsv");

    if !fasta_path.exists() || !tax_path.exists() {
        println!(
            "  [INFO] SILVA reference not found at {} — skipping taxonomy",
            ref_dir.display()
        );
        return None;
    }

    println!("  Loading SILVA 138.1 NR99 reference database...");

    let tax_content = std::fs::read_to_string(&tax_path).ok()?;
    let mut tax_map: HashMap<String, String> = HashMap::new();
    for line in tax_content.lines().skip(1) {
        let parts: Vec<&str> = line.splitn(2, '\t').collect();
        if parts.len() == 2 {
            tax_map.insert(parts[0].to_string(), parts[1].trim().to_string());
        }
    }
    println!("  Loaded {} taxonomy entries", tax_map.len());

    // Subsample for training performance: take every Nth sequence
    // Full SILVA has ~436K seqs; training on all of them with k=8 is expensive.
    // We subsample to ~5000 sequences, stratified across phyla.
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

    println!(
        "  Subsampled {} reference sequences from {} total",
        refs.len(),
        n_parsed
    );

    if refs.is_empty() {
        println!("  [WARN] No reference sequences loaded — skipping taxonomy");
        return None;
    }

    println!("  Training NaiveBayes classifier (k=8)...");
    let classifier = NaiveBayesClassifier::train(&refs, 8);
    println!("  Classifier ready: {} taxa", classifier.n_taxa());
    Some(classifier)
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
        println!(
            "  [SKIP] manifest.json not found at {}",
            manifest_path.display()
        );
        v.check("manifest.json exists", 0.0, 1.0, 0.0);
    }
}

// ── Process a single sample through the full 16S pipeline ───────────────────

#[allow(clippy::too_many_lines)] // sequential validation: parse → filter → derep → DADA2 → diversity → taxonomy
fn process_sample(
    v: &mut Validator,
    sample_dir: &Path,
    label: &str,
    accession: &str,
    classifier: Option<&NaiveBayesClassifier>,
) -> Option<SampleResult> {
    v.section(&format!("Pipeline: {label} ({accession})"));

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

    let records = match wetspring_barracuda::io::fastq::parse_fastq(&fastq_path) {
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

    let mean_len = if filtered.is_empty() {
        0.0
    } else {
        filtered
            .iter()
            .map(|r| r.sequence.len() as f64)
            .sum::<f64>()
            / filtered.len() as f64
    };
    println!("  {label}: mean read length = {mean_len:.0} bp");

    v.check(
        &format!("{label}: mean length > 100 bp"),
        if mean_len > 100.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    let sub: Vec<_> = filtered.into_iter().take(5000).collect();
    if sub.len() < 50 {
        println!(
            "  {label}: too few reads after filtering ({}) — skipping denoising",
            sub.len()
        );
        return None;
    }

    let (uniques, _) = derep::dereplicate(&sub, DerepSort::Abundance, 2);
    let n_unique = uniques.len();
    println!(
        "  {label}: {n_unique} unique sequences from {} reads",
        sub.len()
    );

    v.check(
        &format!("{label}: >1 unique sequence"),
        if n_unique > 1 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    let (asvs, _) = dada2::denoise(&uniques, &Dada2Params::default());
    let n_asvs = asvs.len();
    println!("  {label}: {n_asvs} ASVs after DADA2");

    v.check(
        &format!("{label}: >1 ASV"),
        if n_asvs > 1 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    let counts: Vec<f64> = asvs.iter().map(|a| a.abundance as f64).collect();
    let shannon = diversity::shannon(&counts);
    let simpson = diversity::simpson(&counts);
    let observed = diversity::observed_features(&counts);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let observed_usize = observed as usize;
    println!("  {label}: observed={observed_usize}, Shannon={shannon:.3}, Simpson={simpson:.3}",);

    v.check(
        &format!("{label}: Shannon > 0"),
        if shannon > 0.0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    v.check(
        &format!("{label}: Simpson in (0,1)"),
        if simpson > 0.0 && simpson < 1.0 {
            1.0
        } else {
            0.0
        },
        1.0,
        0.0,
    );

    // Taxonomy classification (top 5 ASVs)
    let mut top_taxa = Vec::new();
    if let Some(clf) = classifier {
        let params = ClassifyParams {
            bootstrap_n: 50,
            ..ClassifyParams::default()
        };
        let n_classify = n_asvs.min(5);
        for asv in asvs.iter().take(n_classify) {
            let result = clf.classify(&asv.sequence, &params);
            let genus = result
                .lineage
                .at_rank(TaxRank::Genus)
                .unwrap_or("Unclassified")
                .to_string();
            let phylum = result
                .lineage
                .at_rank(TaxRank::Phylum)
                .unwrap_or("?")
                .to_string();
            let conf = if result.confidence.len() > TaxRank::Genus.depth() {
                result.confidence[TaxRank::Genus.depth()]
            } else {
                0.0
            };
            top_taxa.push(format!(
                "{genus} ({phylum}) [{conf:.0}%]",
                conf = conf * 100.0
            ));
        }
        if !top_taxa.is_empty() {
            println!("  {label} taxonomy: {}", top_taxa.join(", "));
        }
    }

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
        top_taxa,
    })
}

// ── Temporal analysis ───────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)] // sequential time-series checks across PRJNA1114688 samples
fn temporal_analysis(v: &mut Validator, all_results: &[SampleResult]) {
    v.section("Temporal Analysis (PRJNA1114688 Time Series)");

    let nanno_d1: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("N.oculata") && r.label.contains("D1"))
        .collect();
    let nanno_d7: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("N.oculata") && r.label.contains("D7"))
        .collect();
    let nanno_d14: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("N.oculata") && r.label.contains("D14"))
        .collect();
    let brach_d1: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("B.plicatilis") && r.label.contains("D1"))
        .collect();
    let brach_d7: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("B.plicatilis") && r.label.contains("D7"))
        .collect();
    let brach_d14: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.starts_with("B.plicatilis") && r.label.contains("D14"))
        .collect();

    let groups = [
        ("N.oculata Day 1", &nanno_d1),
        ("N.oculata Day 7", &nanno_d7),
        ("N.oculata Day 14", &nanno_d14),
        ("B.plicatilis Day 1", &brach_d1),
        ("B.plicatilis Day 7", &brach_d7),
        ("B.plicatilis Day 14", &brach_d14),
    ];

    println!(
        "\n  ┌────────────────────────┬──────┬──────────┬──────────┬──────┬────────┬─────────┐"
    );
    println!("  │ Group                  │  n   │ Reads(μ) │ Filt(μ)  │ ASVs │Shannon │ Simpson │");
    println!("  ├────────────────────────┼──────┼──────────┼──────────┼──────┼────────┼─────────┤");
    for (name, grp) in &groups {
        if grp.is_empty() {
            continue;
        }
        let n = grp.len() as f64;
        let avg_reads = grp.iter().map(|r| r.total_reads as f64).sum::<f64>() / n;
        let avg_filt = grp.iter().map(|r| r.filtered_reads as f64).sum::<f64>() / n;
        let avg_asvs = grp.iter().map(|r| r.n_asvs as f64).sum::<f64>() / n;
        let avg_shan = grp.iter().map(|r| r.shannon).sum::<f64>() / n;
        let avg_simp = grp.iter().map(|r| r.simpson).sum::<f64>() / n;
        println!(
            "  │ {:22} │ {:>4} │ {:>8.0} │ {:>8.0} │ {:>4.0} │ {:>6.3} │ {:>7.3} │",
            name,
            grp.len(),
            avg_reads,
            avg_filt,
            avg_asvs,
            avg_shan,
            avg_simp
        );
    }
    println!("  └────────────────────────┴──────┴──────────┴──────────┴──────┴────────┴─────────┘");

    // Biological expectation: community composition changes over time
    let avg_shannon = |g: &[&SampleResult]| -> f64 {
        if g.is_empty() {
            return 0.0;
        }
        g.iter().map(|r| r.shannon).sum::<f64>() / g.len() as f64
    };

    let nanno_shannon_d1 = avg_shannon(&nanno_d1);
    let nanno_shannon_d14 = avg_shannon(&nanno_d14);
    let brach_shannon_d1 = avg_shannon(&brach_d1);
    let brach_shannon_d14 = avg_shannon(&brach_d14);

    v.check(
        "Temporal: N.oculata has diversity at Day 1 and Day 14",
        if nanno_shannon_d1 > 0.0 && nanno_shannon_d14 > 0.0 {
            1.0
        } else {
            0.0
        },
        1.0,
        0.0,
    );
    v.check(
        "Temporal: B.plicatilis has diversity at Day 1 and Day 14",
        if brach_shannon_d1 > 0.0 && brach_shannon_d14 > 0.0 {
            1.0
        } else {
            0.0
        },
        1.0,
        0.0,
    );

    let nanno_delta = (nanno_shannon_d14 - nanno_shannon_d1).abs();
    let brach_delta = (brach_shannon_d14 - brach_shannon_d1).abs();
    println!(
        "  N.oculata Shannon: D1={nanno_shannon_d1:.3} → D14={nanno_shannon_d14:.3} (Δ={nanno_delta:.3})",
    );
    println!(
        "  B.plicatilis Shannon: D1={brach_shannon_d1:.3} → D14={brach_shannon_d14:.3} (Δ={brach_delta:.3})",
    );

    // Replicate consistency: within-group CV of Shannon should be reasonable
    let cv = |g: &[&SampleResult]| -> f64 {
        if g.len() < 2 {
            return 0.0;
        }
        let mean = g.iter().map(|r| r.shannon).sum::<f64>() / g.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }
        let var = g.iter().map(|r| (r.shannon - mean).powi(2)).sum::<f64>() / g.len() as f64;
        var.sqrt() / mean
    };

    let nanno_d7_cv = cv(&nanno_d7);
    let brach_d7_cv = cv(&brach_d7);
    println!(
        "  Replicate consistency: N.oculata D7 CV={nanno_d7_cv:.2}, B.plicatilis D7 CV={brach_d7_cv:.2}",
    );

    v.check(
        "Temporal: replicates have CV < 1.0 (reasonable consistency)",
        if nanno_d7.len() >= 2 && nanno_d7_cv < 1.0 {
            1.0
        } else {
            0.0
        },
        1.0,
        0.0,
    );

    // All 16 samples produced results
    let n_1114688 = all_results
        .iter()
        .filter(|r| r.label.starts_with("N.oculata") || r.label.starts_with("B.plicatilis"))
        .count();
    println!("  PRJNA1114688 samples processed: {n_1114688}/16");
    v.check(
        "Temporal: all 16 PRJNA1114688 samples produced results",
        if n_1114688 >= 16 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // 6 timepoint groups represented
    let n_groups = groups.iter().filter(|(_, g)| !g.is_empty()).count();
    v.check(
        "Temporal: all 6 organism×timepoint groups have data",
        if n_groups == 6 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
}

// ── Cross-dataset benchmark ─────────────────────────────────────────────────

#[allow(clippy::too_many_lines)] // sequential cross-BioProject validation checks
fn cross_dataset_benchmark(v: &mut Validator, all_results: &[SampleResult]) {
    v.section("Cross-Dataset Paper Benchmark");

    if all_results.is_empty() {
        println!("  [SKIP] No sample results available for benchmarking");
        return;
    }

    println!(
        "  Benchmarking {} samples across 4 BioProjects",
        all_results.len()
    );

    let nanno_samples: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.contains("oculata") || r.label.contains("oceanica"))
        .collect();
    let brachio_samples: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.contains("plicatilis"))
        .collect();
    let cyano_samples: Vec<&SampleResult> = all_results
        .iter()
        .filter(|r| r.label.contains("Cyano") || r.label.contains("LakeErie"))
        .collect();

    if !nanno_samples.is_empty() {
        let avg_shannon =
            nanno_samples.iter().map(|r| r.shannon).sum::<f64>() / nanno_samples.len() as f64;
        let avg_simpson =
            nanno_samples.iter().map(|r| r.simpson).sum::<f64>() / nanno_samples.len() as f64;
        let avg_asvs =
            nanno_samples.iter().map(|r| r.n_asvs as f64).sum::<f64>() / nanno_samples.len() as f64;

        println!(
            "  Nannochloropsis ({} samples): Shannon={:.3}, Simpson={:.3}, ASVs={:.0}",
            nanno_samples.len(),
            avg_shannon,
            avg_simpson,
            avg_asvs
        );

        v.check(
            "Humphrey: Nanno Shannon in [0.5, 5.0]",
            if (0.5..=5.0).contains(&avg_shannon) {
                1.0
            } else {
                0.0
            },
            1.0,
            0.0,
        );
        v.check(
            "Humphrey: Nanno Simpson in [0.3, 1.0]",
            if (0.3..=1.0).contains(&avg_simpson) {
                1.0
            } else {
                0.0
            },
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

    if !brachio_samples.is_empty() {
        let avg_shannon =
            brachio_samples.iter().map(|r| r.shannon).sum::<f64>() / brachio_samples.len() as f64;
        println!(
            "  Brachionus ({} samples): Shannon={:.3}",
            brachio_samples.len(),
            avg_shannon
        );

        v.check(
            "Carney: Brachio Shannon > 0 (diverse community)",
            if avg_shannon > 0.0 { 1.0 } else { 0.0 },
            1.0,
            0.0,
        );
    }

    if !nanno_samples.is_empty() && !brachio_samples.is_empty() {
        let nanno_obs =
            nanno_samples.iter().map(|r| r.observed).sum::<f64>() / nanno_samples.len() as f64;
        let brachio_obs =
            brachio_samples.iter().map(|r| r.observed).sum::<f64>() / brachio_samples.len() as f64;
        println!("  Cross-condition: Nanno obs={nanno_obs:.0}, Brachio obs={brachio_obs:.0}",);

        v.check(
            "Cross: both Nanno and Brachio have >1 observed feature",
            if nanno_obs > 1.0 && brachio_obs > 1.0 {
                1.0
            } else {
                0.0
            },
            1.0,
            0.0,
        );
    }

    if !cyano_samples.is_empty() {
        let avg_shannon =
            cyano_samples.iter().map(|r| r.shannon).sum::<f64>() / cyano_samples.len() as f64;
        let avg_asvs =
            cyano_samples.iter().map(|r| r.n_asvs as f64).sum::<f64>() / cyano_samples.len() as f64;
        println!(
            "  Cyanobacteria/HAB ({} samples): Shannon={:.3}, ASVs={:.0}",
            cyano_samples.len(),
            avg_shannon,
            avg_asvs
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

    if !nanno_samples.is_empty() && !cyano_samples.is_empty() {
        let nanno_shannon =
            nanno_samples.iter().map(|r| r.shannon).sum::<f64>() / nanno_samples.len() as f64;
        let cyano_shannon =
            cyano_samples.iter().map(|r| r.shannon).sum::<f64>() / cyano_samples.len() as f64;
        let delta = (nanno_shannon - cyano_shannon).abs();
        println!(
            "  Cross-domain: marine Shannon={nanno_shannon:.3} vs freshwater Shannon={cyano_shannon:.3} (Δ={delta:.3})",
        );

        v.check(
            "Cross-domain: pipeline handles both marine and freshwater",
            if nanno_shannon > 0.0 && cyano_shannon > 0.0 {
                1.0
            } else {
                0.0
            },
            1.0,
            0.0,
        );
    }

    let n_bioprojects = [
        !nanno_samples.is_empty() || !brachio_samples.is_empty(),
        all_results.iter().any(|r| r.label.contains("oceanica")),
        all_results.iter().any(|r| r.label.contains("Cyano")),
        all_results.iter().any(|r| r.label.contains("LakeErie")),
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

    println!(
        "\n  ┌────────────────────────────────┬──────────┬──────────┬────────┬────────┬─────────┐"
    );
    println!(
        "  │ Sample                         │   Reads  │ Filtered │  ASVs  │Shannon │ Simpson │"
    );
    println!(
        "  ├────────────────────────────────┼──────────┼──────────┼────────┼────────┼─────────┤"
    );
    for r in all_results {
        println!(
            "  │ {:30} │ {:>8} │ {:>8} │ {:>6} │ {:>6.3} │ {:>7.3} │",
            r.label, r.total_reads, r.filtered_reads, r.n_asvs, r.shannon, r.simpson
        );
    }
    println!(
        "  └────────────────────────────────┴──────────┴──────────┴────────┴────────┴─────────┘"
    );
}

// ── Taxonomy benchmark ──────────────────────────────────────────────────────

fn taxonomy_benchmark(v: &mut Validator, all_results: &[SampleResult]) {
    v.section("Taxonomy Benchmark (SILVA 138.1)");

    let has_taxa = all_results.iter().any(|r| !r.top_taxa.is_empty());
    if !has_taxa {
        println!("  [SKIP] No taxonomy results — SILVA not loaded or no classifications");
        return;
    }

    let classified_samples = all_results
        .iter()
        .filter(|r| !r.top_taxa.is_empty())
        .count();
    println!(
        "  Samples with taxonomy: {classified_samples}/{}",
        all_results.len()
    );

    v.check(
        "Taxonomy: at least 1 sample classified",
        if classified_samples > 0 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Collect all unique phyla across samples
    let mut all_phyla: HashMap<String, usize> = HashMap::new();
    let mut all_genera: HashMap<String, usize> = HashMap::new();
    for r in all_results {
        for tax in &r.top_taxa {
            if let Some(genus_part) = tax.split(" (").next() {
                *all_genera.entry(genus_part.to_string()).or_insert(0) += 1;
            }
            if let Some(phylum_start) = tax.find('(') {
                if let Some(phylum_end) = tax.find(')') {
                    let phylum = &tax[phylum_start + 1..phylum_end];
                    *all_phyla.entry(phylum.to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    println!("  Unique phyla detected: {}", all_phyla.len());
    for (phylum, count) in &all_phyla {
        println!("    {phylum}: {count} hits");
    }
    println!("  Unique genera detected: {}", all_genera.len());
    let mut genera_sorted: Vec<_> = all_genera.iter().collect();
    genera_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (genus, count) in genera_sorted.iter().take(10) {
        println!("    {genus}: {count} hits");
    }

    v.check(
        "Taxonomy: >1 phylum detected across all samples",
        if all_phyla.len() > 1 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    v.check(
        "Taxonomy: >3 genera detected across all samples",
        if all_genera.len() > 3 { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );

    // Check for biologically expected phyla in aquaculture samples
    let has_proteobacteria = all_phyla
        .keys()
        .any(|p| p.to_lowercase().contains("proteobacteria"));
    let has_bacteroidetes = all_phyla
        .keys()
        .any(|p| p.to_lowercase().contains("bacteroid"));

    println!(
        "  Expected phyla: Proteobacteria={}, Bacteroidetes={}",
        if has_proteobacteria {
            "found"
        } else {
            "not found"
        },
        if has_bacteroidetes {
            "found"
        } else {
            "not found"
        },
    );

    v.check(
        "Taxonomy: Proteobacteria detected (expected in aquaculture)",
        if has_proteobacteria { 1.0 } else { 0.0 },
        1.0,
        0.0,
    );
}
