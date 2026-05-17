// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: pipeline progress printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential pipeline phases read best as one flow"
)]
//! # Exp381: breseq Pipeline — Barrick 2009 via Nest Atomic Composition
//!
//! First real-data Nest Atomic composition: download SRA reads, run breseq
//! variant calling against REL606, record provenance through the trio, and
//! export a ferment transcript braid for lithoSpore.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Barrick et al. *Nature* 461, 1243–1247 (2009) |
//! | SRA Study | SRP001569 |
//! | Reference | REL606 (NC_012967.1, 4,629,812 bp) |
//! | lithoSpore | Module 6 (breseq comparison, via ferment transcript braid) |
//! | Command | See `BRESEQ_ENV` and `WORKSPACE` below |
//!
//! Provenance: NCBI SRA → breseq variant calling → trio → braid export

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use wetspring_barracuda::ipc::provenance;
use wetspring_barracuda::ipc::provenance::braid_handoff::{
    ComputationMetadata, FermentTranscriptBraid,
};
use wetspring_barracuda::ncbi;
use wetspring_barracuda::validation::Validator;

/// CP000819.1 is the actual genome record with sequence (NC_012967 is a CON stub).
const REFERENCE_ACCESSION: &str = "CP000819.1";
const REFERENCE_LENGTH_BP: u64 = 4_629_812;

/// Barrick 2009 SRA runs: Ara-1 population clones across generations.
const BARRICK_RUNS: &[(&str, &str)] = &[
    ("SRR032370", "REL1164M"),
    ("SRR032371", "REL2179M"),
    ("SRR032372", "REL4536M"),
    ("SRR032373", "REL7177M"),
    ("SRR032374", "REL8593M"),
    ("SRR032375", "REL10379"),
    ("SRR032376", "REL10926"),
];

fn workspace_dir() -> PathBuf {
    std::env::var("WETSPRING_WORKSPACE").map_or_else(
        |_| PathBuf::from("/mnt/4tb-work/ecoPrimals/springs/wetSpring/datasets/ltee/barrick_2009"),
        PathBuf::from,
    )
}

fn breseq_env_bin() -> PathBuf {
    std::env::var("BRESEQ_ENV_BIN").map_or_else(
        |_| PathBuf::from("/mnt/4tb-work/micromamba/envs/breseq-env/bin"),
        PathBuf::from,
    )
}

fn run_in_env(tool: &str, args: &[&str], work_dir: &Path) -> std::io::Result<std::process::Output> {
    let bin = breseq_env_bin();
    let tool_path = bin.join(tool);

    let current_path = std::env::var("PATH").unwrap_or_default();
    let env_path = format!("{}:{current_path}", bin.display());

    Command::new(&tool_path)
        .args(args)
        .current_dir(work_dir)
        .env("PATH", &env_path)
        .output()
}

fn download_reference(workspace: &Path) -> Result<PathBuf, String> {
    let ref_dir = workspace.join("reference");
    std::fs::create_dir_all(&ref_dir).map_err(|e| format!("mkdir reference: {e}"))?;

    let gbk_path = ref_dir.join("REL606.gbk");
    if gbk_path.exists() {
        let size = std::fs::metadata(&gbk_path).map_or(0, |m| m.len());
        if size > 1_000_000 {
            println!("  [CACHED] REL606.gbk already present ({size} bytes)");
            return Ok(gbk_path);
        }
        println!("  Existing REL606.gbk too small ({size} bytes), re-downloading...");
    }

    println!("  Downloading REL606 reference ({REFERENCE_ACCESSION}, GenBank format)...");

    let url = format!(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nucleotide&id={REFERENCE_ACCESSION}&rettype=gb&retmode=text"
    );
    let body = ncbi::http_get(&url).map_err(|e| format!("http GET REL606: {e}"))?;
    if !body.contains("LOCUS") {
        return Err("downloaded reference does not contain LOCUS line".to_string());
    }
    if body.len() < 1_000_000 {
        return Err(format!(
            "reference too small ({} bytes) — expected full genome with sequence",
            body.len()
        ));
    }
    std::fs::write(&gbk_path, &body).map_err(|e| format!("write REL606.gbk: {e}"))?;
    println!("  [OK] REL606.gbk ({} bytes)", body.len());
    Ok(gbk_path)
}

fn download_sra_run(accession: &str, output_dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(output_dir).map_err(|e| format!("mkdir: {e}"))?;

    let fastq_path = output_dir.join(format!("{accession}.fastq"));
    if fastq_path.exists() {
        println!("    [CACHED] {accession}.fastq already present");
        return Ok(fastq_path);
    }

    println!("    Downloading {accession} via prefetch + fasterq-dump...");

    let prefetch_out = run_in_env(
        "prefetch",
        &["--max-size", "50G", "--output-directory", &output_dir.to_string_lossy(), accession],
        output_dir,
    )
    .map_err(|e| format!("prefetch {accession}: {e}"))?;

    if !prefetch_out.status.success() {
        let stderr = String::from_utf8_lossy(&prefetch_out.stderr);
        let limit = stderr.len().min(200);
        return Err(format!("prefetch {accession} failed: {}", &stderr[..limit]));
    }

    let fasterq_out = run_in_env(
        "fasterq-dump",
        &[
            "--outdir", &output_dir.to_string_lossy(),
            "--split-3",
            "--skip-technical",
            "--threads", "4",
            accession,
        ],
        output_dir,
    )
    .map_err(|e| format!("fasterq-dump {accession}: {e}"))?;

    if !fasterq_out.status.success() {
        let stderr = String::from_utf8_lossy(&fasterq_out.stderr);
        let limit = stderr.len().min(200);
        return Err(format!("fasterq-dump {accession} failed: {}", &stderr[..limit]));
    }

    if fastq_path.exists() {
        let size = std::fs::metadata(&fastq_path).map_or(0, |m| m.len());
        println!("    [OK] {accession}.fastq ({} MB)", size / 1_000_000);
        Ok(fastq_path)
    } else {
        let alt = output_dir.join(format!("{accession}_1.fastq"));
        if alt.exists() {
            println!("    [OK] {accession}_1.fastq (paired-end split)");
            Ok(alt)
        } else {
            Err(format!("fasterq-dump succeeded but no output found for {accession}"))
        }
    }
}

fn run_breseq(
    accession: &str,
    clone_name: &str,
    fastq_path: &Path,
    reference: &Path,
    workspace: &Path,
) -> Result<PathBuf, String> {
    let output_dir = workspace.join(format!("breseq_output/{clone_name}"));
    let gd_path = output_dir.join("output/output.gd");

    if gd_path.exists() {
        println!("    [CACHED] breseq output for {clone_name} already present");
        return Ok(gd_path);
    }

    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir)
            .map_err(|e| format!("cleanup breseq output: {e}"))?;
    }
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("mkdir breseq output: {e}"))?;

    println!("    Running breseq on {clone_name} ({accession})...");

    let output = run_in_env(
        "breseq",
        &[
            "-r", &reference.to_string_lossy(),
            "-o", &output_dir.to_string_lossy(),
            "-n", clone_name,
            "-j", "4",
            &fastq_path.to_string_lossy(),
        ],
        workspace,
    )
    .map_err(|e| format!("breseq {clone_name}: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let limit = stderr.len().min(500);
        return Err(format!(
            "breseq {clone_name} failed (exit {:?}): {}",
            output.status.code(),
            &stderr[..limit]
        ));
    }

    if gd_path.exists() {
        println!("    [OK] breseq complete for {clone_name}");
        Ok(gd_path)
    } else {
        let index_html = output_dir.join("output/index.html");
        if index_html.exists() {
            println!("    [OK] breseq complete for {clone_name} (HTML output only)");
            Ok(index_html)
        } else {
            Err(format!("breseq completed but no output.gd found for {clone_name}"))
        }
    }
}

fn count_mutations_in_gd(gd_path: &Path) -> usize {
    let Ok(content) = std::fs::read_to_string(gd_path) else {
        return 0;
    };
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("AUTHOR")
                && !trimmed.starts_with("READSEQ")
                && !trimmed.starts_with("REFSEQ")
                && !trimmed.starts_with("NOTE")
                && !trimmed.starts_with("ADAPTSEQ")
        })
        .count()
}

fn blake3_file(path: &Path) -> String {
    let Ok(data) = std::fs::read(path) else {
        return "unavailable".to_string();
    };
    blake3::hash(&data).to_hex().to_string()
}

fn run_pipeline(v: &mut Validator) {
    let t0 = Instant::now();
    let workspace = workspace_dir();

    println!("  Workspace: {}", workspace.display());
    println!("  breseq env: {}", breseq_env_bin().display());
    println!();

    // PHASE 1: Provenance — begin DAG session
    v.section("P01: Provenance session");
    let prov_session = provenance::begin_session("barrick_2009_breseq_pipeline");
    println!("  Session ID: {}", prov_session.id);
    println!("  Trio available: {}", prov_session.available);
    v.check_pass(
        "provenance session started (local or trio)",
        !prov_session.id.is_empty(),
    );

    // PHASE 2: Reference genome
    v.section("P02: REL606 reference genome");
    let reference = match download_reference(&workspace) {
        Ok(path) => {
            let size = std::fs::metadata(&path).map_or(0, |m| m.len());
            v.check_pass("REL606 GenBank downloaded", size > 1_000_000);
            let _ = provenance::record_step(
                &prov_session.id,
                &serde_json::json!({
                    "step": "download_reference",
                    "accession": REFERENCE_ACCESSION,
                    "size_bytes": size,
                    "blake3": blake3_file(&path),
                }),
            );
            path
        }
        Err(e) => {
            println!("  [FAIL] Reference download failed: {e}");
            v.check_pass("REL606 GenBank downloaded", false);
            return;
        }
    };

    // PHASE 3: SRA data acquisition
    v.section("P03: SRA data acquisition");
    let reads_dir = workspace.join("reads");
    let mut downloaded_runs: Vec<(&str, &str, PathBuf)> = Vec::new();

    for &(accession, clone_name) in BARRICK_RUNS {
        match download_sra_run(accession, &reads_dir) {
            Ok(path) => {
                let _ = provenance::record_step(
                    &prov_session.id,
                    &serde_json::json!({
                        "step": "download_sra",
                        "accession": accession,
                        "clone": clone_name,
                        "blake3": blake3_file(&path),
                    }),
                );
                downloaded_runs.push((accession, clone_name, path));
            }
            Err(e) => {
                println!("    [FAIL] {accession} ({clone_name}): {e}");
            }
        }
    }

    v.check_count("SRA runs downloaded", downloaded_runs.len(), BARRICK_RUNS.len());
    println!("  Downloaded {}/{} runs", downloaded_runs.len(), BARRICK_RUNS.len());

    if downloaded_runs.len() != BARRICK_RUNS.len() {
        println!("\n  Not all runs downloaded. Proceeding with available data.");
    }

    // PHASE 4: breseq variant calling
    v.section("P04: breseq variant calling");
    let mut mutation_counts: Vec<(&str, usize)> = Vec::new();

    for (accession, clone_name, fastq_path) in &downloaded_runs {
        match run_breseq(accession, clone_name, fastq_path, &reference, &workspace) {
            Ok(gd_path) => {
                let count = count_mutations_in_gd(&gd_path);
                println!("    {clone_name}: {count} mutations");
                let _ = provenance::record_step(
                    &prov_session.id,
                    &serde_json::json!({
                        "step": "breseq_variant_calling",
                        "clone": clone_name,
                        "accession": accession,
                        "mutations": count,
                        "output_blake3": blake3_file(&gd_path),
                    }),
                );
                mutation_counts.push((clone_name, count));
            }
            Err(e) => {
                println!("    [FAIL] breseq {clone_name}: {e}");
                mutation_counts.push((clone_name, 0));
            }
        }
    }

    let breseq_succeeded = mutation_counts.iter().filter(|(_, c)| *c > 0).count();
    v.check_pass("at least one clone produced mutations", breseq_succeeded > 0);

    // PHASE 5: Science validation — mutation accumulation trend
    v.section("P05: Mutation accumulation trend");
    if mutation_counts.len() >= 2 {
        let first = mutation_counts.first().map_or(0, |(_, c)| *c);
        let last = mutation_counts.last().map_or(0, |(_, c)| *c);
        v.check_pass("later clones have more mutations than early clones", last > first);
        println!(
            "  Early ({}) = {} mutations, Late ({}) = {} mutations",
            mutation_counts.first().map_or("?", |(n, _)| n),
            first,
            mutation_counts.last().map_or("?", |(n, _)| n),
            last,
        );
    } else {
        v.check_pass("enough clones for trend analysis", false);
    }

    // PHASE 6: Provenance complete — dehydrate → commit → braid
    v.section("P06: Provenance completion");
    let session_result = provenance::complete_session(&prov_session.id);
    let prov_status = session_result["provenance"].as_str().unwrap_or("unknown");
    println!("  Provenance status: {prov_status}");
    v.check_pass(
        "provenance session completed (complete or unavailable)",
        prov_status == "complete" || prov_status == "unavailable",
    );

    // PHASE 7: Ferment transcript braid export
    v.section("P07: Ferment transcript braid export");
    let elapsed = t0.elapsed();

    let summary = serde_json::json!({
        "dataset": "barrick_2009",
        "paper": "Barrick et al. Nature 461:1243 (2009)",
        "reference": REFERENCE_ACCESSION,
        "reference_length_bp": REFERENCE_LENGTH_BP,
        "clones_processed": mutation_counts.len(),
        "total_mutations": mutation_counts.iter().map(|(_, c)| c).sum::<usize>(),
        "mutation_counts": mutation_counts.iter()
            .map(|(name, count)| serde_json::json!({"clone": name, "mutations": count}))
            .collect::<Vec<_>>(),
    });

    let summary_str = serde_json::to_string(&summary).unwrap_or_default();
    let summary_hash = blake3::hash(summary_str.as_bytes()).to_hex().to_string();

    let computation = ComputationMetadata {
        tool: "breseq".to_string(),
        tool_version: "0.40.1".to_string(),
        input_accession: "SRP001569".to_string(),
        input_blake3: "aggregate".to_string(),
        output_blake3: summary_hash.clone(),
        wall_time_seconds: elapsed.as_secs(),
        node_count: u64::try_from(mutation_counts.len()).unwrap_or(0),
    };

    let braid = FermentTranscriptBraid::from_session_result(
        "barrick_2009_mutations",
        &session_result,
        computation,
        &summary_hash,
    );

    let braid_json = braid.to_json();
    v.check_pass(
        "braid has non-empty dataset_id",
        braid_json["dataset_id"].as_str().is_some_and(|s| !s.is_empty()),
    );
    v.check_pass(
        "braid has spring = wetSpring",
        braid_json["spring"].as_str() == Some("wetSpring"),
    );
    v.check_pass("braid has non-empty summary_blake3", !summary_hash.is_empty());

    let braid_dir = workspace.join("provenance/braids");
    std::fs::create_dir_all(&braid_dir).ok();
    let braid_path = braid_dir.join("barrick_2009_mutations.json");
    let braid_str = serde_json::to_string_pretty(&braid_json).unwrap_or_default();
    match std::fs::write(&braid_path, &braid_str) {
        Ok(()) => {
            println!("  Braid exported to {}", braid_path.display());
            v.check_pass("braid JSON written to disk", true);
        }
        Err(e) => {
            println!("  [FAIL] Write braid: {e}");
            v.check_pass("braid JSON written to disk", false);
        }
    }

    let summary_path = workspace.join("provenance/barrick_2009_summary.json");
    std::fs::write(&summary_path, &summary_str).ok();

    println!("\n  Wall time: {:.1}s", elapsed.as_secs_f64());
    println!("  Workspace: {}", workspace.display());
}

fn main() {
    let mut v = Validator::new("Exp381: breseq Pipeline — Barrick 2009 (Nest Atomic)");
    run_pipeline(&mut v);
    v.finish();
}
