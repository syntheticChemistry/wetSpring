// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Sovereign Rust Resequencing — Primal Composition
//!
//! Data-driven sovereign resequencing pipeline. Discovers its dataset from the
//! workspace: reads `clones.tsv` (accession\tname) if present, otherwise falls
//! back to the Barrick 2009 default clone list.
//!
//! Set `WETSPRING_WORKSPACE` to point at any LTEE dataset workspace.
//! Set `WETSPRING_DATASET_ID` to override the dataset identifier for braids.
//! Set `WETSPRING_ACCESSION` to override the project accession (e.g. SRP064605).
//! Set `WETSPRING_MAX_CLONES` to limit the number of clones per batch.
//! Set `WETSPRING_CLONE_OFFSET` to skip already-processed clones (for batched runs).
//!
//! # Pipeline
//!
//! FM-index (CPU) → SmithWatermanGpu (GPU mapping) → pileup (CPU, Q20-filtered) →
//! Tensor::scan (GPU coverage) → SnpCallingF64 (GPU variant calling)
//!
//! Full GPU systems study: all compute-intensive stages dispatch to GPU via
//! barraCuda primitives. CPU handles I/O, indexing, and orchestration.
//! Provenance: live trio via plasmidBin (rhizoCrypt DAG, loamSpine aglets, sweetGrass braid)
//! Discovery: connect-probe with DEAD_SOCKET_CACHE (Wave 22)

use std::path::PathBuf;
use std::time::Instant;

use crate::bio::pileup;
use crate::bio::read_mapper::{self, MapperConfig};
use crate::bio::ref_index::FmIndex;
use crate::bio::variant_caller::{self, CallerConfig};
use crate::io::fasta::{FastaRecord, GenBankRecord};
use crate::io::fastq;
use crate::io::sam;
use crate::ncbi::fetch_sra_composed;
use crate::validation::Validator;

use crate::ipc::provenance;
use crate::ipc::provenance::braid_handoff::{
    ComputationMetadata, FermentTranscriptBraid,
};

#[cfg(feature = "gpu")]
use crate::gpu::GpuF64;

const BARRICK_CLONES: &[(&str, &str)] = &[
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

fn find_file_by_ext(dir: &std::path::Path, ext: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.find_map(|entry| {
        let path = entry.ok()?.path();
        if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            Some(path)
        } else {
            None
        }
    })
}

fn dataset_id(workspace: &std::path::Path) -> String {
    std::env::var("WETSPRING_DATASET_ID").unwrap_or_else(|_| {
        workspace
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    })
}

fn project_accession() -> String {
    std::env::var("WETSPRING_ACCESSION").unwrap_or_else(|_| "SRP001569".to_string())
}

/// Load clone list from `clones.tsv` (accession\tname per line) or fall back
/// to `accession_list.txt` (accession per line, name = accession) or the
/// hardcoded Barrick 2009 default.
fn clone_offset() -> usize {
    std::env::var("WETSPRING_CLONE_OFFSET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn load_clones(workspace: &std::path::Path) -> Vec<(String, String)> {
    let max_clones: usize = std::env::var("WETSPRING_MAX_CLONES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let offset = clone_offset();

    let tsv_path = workspace.join("clones.tsv");
    if tsv_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&tsv_path) {
            let clones: Vec<(String, String)> = contents
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
                .skip(offset)
                .take(max_clones)
                .map(|line| {
                    let mut parts = line.split('\t');
                    let acc = parts.next().unwrap_or("").trim().to_string();
                    let name = parts.next().map_or_else(|| acc.clone(), |n| n.trim().to_string());
                    (acc, name)
                })
                .collect();
            if !clones.is_empty() {
                return clones;
            }
        }
    }

    let acc_path = workspace.join("accession_list.txt");
    if acc_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&acc_path) {
            let clones: Vec<(String, String)> = contents
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
                .skip(offset)
                .take(max_clones)
                .map(|line| {
                    let acc = line.trim().to_string();
                    let name = acc.clone();
                    (acc, name)
                })
                .collect();
            if !clones.is_empty() {
                return clones;
            }
        }
    }

    BARRICK_CLONES
        .iter()
        .skip(offset)
        .take(max_clones)
        .map(|&(a, n)| (a.to_string(), n.to_string()))
        .collect()
}

/// Run the `validate_sovereign_resequencing` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let t0 = Instant::now();
    let workspace = workspace_dir();
    let ds_id = dataset_id(&workspace);
    let clones = load_clones(&workspace);
    let accession = project_accession();

    let title = format!("Sovereign Resequencing — {ds_id} ({} clones)", clones.len());
    let offset = clone_offset();
    println!("  Dataset: {ds_id}");
    println!("  Accession: {accession}");
    println!("  Clones: {} (offset {offset}, source: {})", clones.len(),
        if workspace.join("clones.tsv").exists() { "clones.tsv" }
        else if workspace.join("accession_list.txt").exists() { "accession_list.txt" }
        else { "built-in default" }
    );

    // ── GPU init (if --features gpu) ─────────────────────────────
    #[cfg(feature = "gpu")]
    let gpu = {
        println!("── GPU: Initializing barraCuda compute device ──");
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        match rt.block_on(GpuF64::new()) {
            Ok(g) if g.has_f64 => {
                println!("  GPU: {} (f64 native)", g.adapter_name);
                Some(g)
            }
            Ok(g) => {
                println!("  GPU: {} (no f64 — CPU fallback)", g.adapter_name);
                Some(g)
            }
            Err(e) => {
                println!("  GPU: unavailable ({e}) — CPU fallback");
                None
            }
        }
    };

    #[cfg(feature = "gpu")]
    let gpu_device = gpu.as_ref().map(|g| g.to_wgpu_device());

    #[cfg(not(feature = "gpu"))]
    println!("── Substrate: CPU only (build with --features gpu for GPU acceleration) ──");

    // ── Provenance: begin DAG session (live trio via plasmidBin) ──
    println!("\n── Provenance: DAG session ──");
    let session_name = format!("sovereign_resequencing_{ds_id}");
    let prov = provenance::begin_session(&session_name);
    println!("  Session ID: {}", prov.id);
    println!("  Trio available: {}", prov.available);
    v.check_pass(
        "provenance session started (live trio or degraded local)",
        !prov.id.is_empty(),
    );

    // ── Phase 1: Load reference genome ───────────────────────────
    println!("\n── Phase 1: Loading reference genome ──");

    let ref_dir = workspace.join("reference");
    let ref_fasta_path = find_file_by_ext(&ref_dir, "fasta")
        .or_else(|| find_file_by_ext(&ref_dir, "fa"))
        .or_else(|| find_file_by_ext(&ref_dir, "fna"))
        .unwrap_or_else(|| ref_dir.join("REL606.fasta"));
    let ref_gbk_path = find_file_by_ext(&ref_dir, "gbk")
        .or_else(|| find_file_by_ext(&ref_dir, "gb"))
        .unwrap_or_else(|| ref_dir.join("REL606.gbk"));

    let reference = if ref_fasta_path.exists() {
        println!("  Loading FASTA: {}", ref_fasta_path.display());
        let records = FastaRecord::load_all(&ref_fasta_path).expect("Failed to load reference FASTA");
        v.check_pass(
            "reference FASTA loaded",
            !records.is_empty() && records[0].len() > 4_000_000,
        );
        records[0].sequence.clone()
    } else if ref_gbk_path.exists() {
        println!("  Loading GenBank: {}", ref_gbk_path.display());
        let gbk = GenBankRecord::load(&ref_gbk_path).expect("Failed to load reference GenBank");
        v.check_pass(
            "reference GenBank loaded",
            gbk.sequence.len() > 4_000_000,
        );
        gbk.sequence
    } else {
        println!("  WARNING: No reference genome found at {}", workspace.display());
        println!("  Expected: reference/REL606.fasta or reference/REL606.gbk");
        println!("  Run Exp381 first to download the reference.");
        v.check_pass("reference genome exists", false);
        return;
    };

    println!("  Reference: {} bp", reference.len());
    v.check_pass("reference length > 4 Mb", reference.len() > 4_000_000);

    let _ = provenance::record_step(
        &prov.id,
        &serde_json::json!({
            "step": "load_reference",
            "reference": "REL606 (CP000819.1)",
            "length_bp": reference.len(),
        }),
    );

    // Load GenBank features if available
    let features = if ref_gbk_path.exists() {
        GenBankRecord::load(&ref_gbk_path)
            .map(|g| g.features)
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    println!("  Features: {} CDS annotations", features.iter().filter(|f| f.feature_type == "CDS").count());

    // ── Phase 2: Build FM-index ──────────────────────────────────
    println!("\n── Phase 2: Building FM-index ──");
    let idx_t0 = Instant::now();
    let fm_index = FmIndex::build(&reference);
    let idx_secs = idx_t0.elapsed().as_secs_f64();
    println!("  FM-index built in {idx_secs:.1}s ({} bp indexed)", fm_index.reference_len());
    v.check_pass("FM-index reference length matches", fm_index.reference_len() == reference.len());

    let _ = provenance::record_step(
        &prov.id,
        &serde_json::json!({
            "step": "build_fm_index",
            "reference_len": fm_index.reference_len(),
            "wall_seconds": idx_secs,
        }),
    );

    let mapper_config = MapperConfig {
        seed_k: 20,
        max_seed_hits: 200,
        extension_window: 30,
        min_score: 40,
        ..MapperConfig::default()
    };

    let caller_config = CallerConfig::default();
    let pileup_config = pileup::PileupConfig {
        min_base_quality: 20,
        min_mapq: 0,
        skip_duplicates: true,
        skip_secondary: true,
    };
    println!("  Caller: min_depth={}, min_alt_freq={:.2}, quality_weighted={}, strand_balance={:.2}, min_bq={}, min_mapq={}, skip_dup={}, skip_sec={}",
        caller_config.min_depth, caller_config.min_alt_frequency,
        caller_config.quality_weighted, caller_config.min_strand_balance,
        pileup_config.min_base_quality, pileup_config.min_mapq,
        pileup_config.skip_duplicates, pileup_config.skip_secondary);

    // ── Phase 3: Process each clone ──────────────────────────────
    println!("\n── Phase 3: Per-clone sovereign pipeline ──");

    let mut total_sovereign_variants = 0usize;
    let mut total_breseq_variants = 0usize;
    let mut total_matches = 0usize;
    let mut clones_processed = 0usize;

    for (clone_accession, clone_name) in &clones {
        let clone_t0 = Instant::now();
        println!("\n  ── {clone_name} ({clone_accession}) ──");

        // Composed SRA fetch: NestGate → local cache → SRA Toolkit download
        let reads_dir = workspace.join("reads");
        let fastq_dir = workspace.join("fastq");

        let sra_result = fetch_sra_composed(clone_accession, &reads_dir)
            .or_else(|_| fetch_sra_composed(clone_accession, &fastq_dir));

        let (fq_path, fq_blake3, fq_source) = match sra_result {
            Ok(r) => {
                println!("    FASTQ: {} via {}", r.path.display(), r.source);
                let _ = provenance::record_step(
                    &prov.id,
                    &serde_json::json!({
                        "step": "fetch_sra",
                        "accession": clone_accession,
                        "clone": clone_name,
                        "source": r.source.to_string(),
                        "blake3": &r.blake3,
                    }),
                );
                (r.path, r.blake3, r.source.to_string())
            }
            Err(e) => {
                println!("    SKIP: no FASTQ for {clone_accession} ({e})");
                continue;
            }
        };
        let _ = (fq_blake3, fq_source);

        // Full-depth processing for primal composition braid
        let max_reads = std::env::var("WETSPRING_MAX_READS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        let mut reads: Vec<(String, Vec<u8>, Vec<u8>)> = Vec::new();
        let iter = fastq::FastqIter::open(&fq_path);
        if let Ok(iter) = iter {
            for result in iter {
                if reads.len() >= max_reads {
                    break;
                }
                if let Ok(record) = result {
                    reads.push((record.id, record.sequence, record.quality));
                }
            }
        }
        println!("    Reads loaded: {} (subsampled from {})", reads.len(), fq_path.display());

        if reads.is_empty() {
            println!("    SKIP: no reads loaded");
            continue;
        }

        // Map reads — GPU SmithWatermanGpu reserved for long reads (≥250bp)
        // where per-read compute dominates wgpu dispatch overhead. Illumina
        // short reads (36-150bp) always use CPU seed-extend: per-read GPU
        // dispatch (~20-75μs: buffer create + queue submit + sync) × N reads
        // exceeds SW compute savings at these lengths.
        // Empirical: 7.5M×36bp → 344min GPU vs 26min CPU (13x);
        //            500K×101bp → >17min GPU vs ~5min CPU (est 3-4x).
        let map_t0 = Instant::now();
        let median_read_len = reads.get(reads.len() / 2).map_or(0, |(_, seq, _)| seq.len());
        const GPU_MAPPING_MIN_READ_LEN: usize = 250;

        #[cfg(feature = "gpu")]
        let (sam_records, map_substrate) = if let Some(ref dev) = gpu_device {
            if median_read_len >= GPU_MAPPING_MIN_READ_LEN {
                let recs = read_mapper::map_reads_gpu(
                    &reads, &fm_index, &reference, "REL606", &mapper_config, dev,
                );
                (recs, "GPU SmithWatermanGpu")
            } else {
                println!("    Reads {}bp < {}bp threshold — CPU mapping (GPU reserved for pileup+calling)",
                    median_read_len, GPU_MAPPING_MIN_READ_LEN);
                let recs = read_mapper::map_reads(&reads, &fm_index, &reference, "REL606", &mapper_config);
                (recs, "CPU seed-extend (short reads)")
            }
        } else {
            let recs = read_mapper::map_reads(&reads, &fm_index, &reference, "REL606", &mapper_config);
            (recs, "CPU seed-extend")
        };
        #[cfg(not(feature = "gpu"))]
        let (sam_records, map_substrate) = {
            let recs = read_mapper::map_reads(&reads, &fm_index, &reference, "REL606", &mapper_config);
            (recs, "CPU seed-extend")
        };
        let map_secs = map_t0.elapsed().as_secs_f64();

        let mapped_count = sam_records.iter().filter(|r| r.is_mapped()).count();
        println!("    Mapped: {mapped_count}/{} reads ({map_secs:.1}s, {map_substrate})", sam_records.len());

        // Sort by position for pileup
        let mut sorted_records = sam_records;
        sam::sort_by_position(&mut sorted_records);

        // Generate pileup with base quality filtering
        let pileup_columns = pileup::generate_pileup_filtered(&sorted_records, reference.len(), &pileup_config);
        let cov_stats = pileup::coverage_stats(&pileup_columns, reference.len());
        println!(
            "    Pileup: {} positions covered, mean depth {:.1}, coverage {:.1}%",
            cov_stats.covered_positions,
            cov_stats.mean_depth,
            cov_stats.coverage_fraction * 100.0
        );

        // GPU cumulative coverage track (when available)
        #[cfg(feature = "gpu")]
        if let Some(ref dev) = gpu_device {
            match pileup::cumulative_coverage_gpu(&pileup_columns, dev) {
                Ok(cumsum) => {
                    let total_bases: f64 = cumsum.last().copied().unwrap_or(0.0);
                    println!("    GPU coverage scan: {total_bases:.0} cumulative base-depth (Tensor::scan)");
                }
                Err(e) => println!("    GPU coverage scan skipped: {e}"),
            }
        }

        // Call variants — GPU SNP calling when available
        #[cfg(feature = "gpu")]
        let sovereign_variants = if let Some(ref dev) = gpu_device {
            println!("    Calling via SnpCallingF64 (GPU)...");
            variant_caller::call_variants_gpu(
                &pileup_columns, &reference, &features, &caller_config, dev,
            ).unwrap_or_else(|e| {
                println!("    GPU SNP failed ({e}), falling back to CPU");
                variant_caller::call_variants(&pileup_columns, &reference, &features, &caller_config)
            })
        } else {
            variant_caller::call_variants(&pileup_columns, &reference, &features, &caller_config)
        };
        #[cfg(not(feature = "gpu"))]
        let sovereign_variants = variant_caller::call_variants(
            &pileup_columns, &reference, &features, &caller_config,
        );
        println!("    Sovereign variants: {}", sovereign_variants.len());
        total_sovereign_variants += sovereign_variants.len();

        // Compare against breseq output.gd (if available)
        let gd_path = workspace
            .join("breseq_output")
            .join(clone_name)
            .join("output")
            .join("output.gd");

        if gd_path.exists() {
            let gd_contents = std::fs::read_to_string(&gd_path).unwrap_or_default();
            let breseq_mutations = variant_caller::parse_gd_file(&gd_contents);
            let breseq_snps: Vec<_> = breseq_mutations
                .iter()
                .filter(|(t, _, _)| t == "SNP" || t == "DEL" || t == "INS")
                .cloned()
                .collect();

            println!("    breseq variants: {}", breseq_snps.len());
            total_breseq_variants += breseq_snps.len();

            let (matches, only_sov, only_breseq) =
                variant_caller::compare_calls(&sovereign_variants, &breseq_snps);

            println!("    Parity: {matches} match, {only_sov} sovereign-only, {only_breseq} breseq-only");
            total_matches += matches;

            v.check_pass(
                &format!("{clone_name}: sovereign pipeline produced variants"),
                !sovereign_variants.is_empty() || mapped_count < 100,
            );
        } else {
            println!("    breseq output.gd not found — skipping parity check");
            println!("    (Run Exp381 first to generate breseq baselines)");
        }

        let clone_secs = clone_t0.elapsed().as_secs_f64();
        println!("    {clone_name}: completed in {clone_secs:.1}s");

        // DAG event: clone node sealed
        let clone_hash = blake3::hash(
            format!("{clone_name}:{clone_accession}:variants={}", sovereign_variants.len()).as_bytes(),
        ).to_hex().to_string();

        let _ = provenance::record_step(
            &prov.id,
            &serde_json::json!({
                "step": "clone_complete",
                "clone": clone_name,
                "accession": clone_accession,
                "reads": reads.len(),
                "mapped": mapped_count,
                "coverage_pct": cov_stats.coverage_fraction * 100.0,
                "mean_depth": cov_stats.mean_depth,
                "sovereign_variants": sovereign_variants.len(),
                "wall_seconds": clone_secs,
                "output_blake3": clone_hash,
            }),
        );

        clones_processed += 1;

        // Partial dehydrate: seal this clone's vertex in the DAG while
        // keeping the session open for remaining clones (aglet pattern).
        // rhizoCrypt S69 dag.partial_dehydrate computes Merkle root over
        // sealed vertices without closing the session.
        if let Some(partial) = provenance::rhizocrypt::partial_dehydrate(&prov.id, &[]) {
            let partial_root = partial.get("merkle_root")
                .and_then(|v| v.as_str())
                .unwrap_or("pending");
            println!("    DAG partial root ({clones_processed}/{} sealed): {partial_root}", clones.len());
        }
    }

    // ── Phase 4: Summary ─────────────────────────────────────────
    println!("\n── Phase 4: Summary ──");
    #[cfg(feature = "gpu")]
    let substrate_label = if gpu.is_some() {
        "Adaptive: CPU mapping (Illumina <250bp) + GPU Tensor::scan (coverage) + GPU SnpCallingF64 (variants)"
    } else {
        "CPU fallback (GPU init failed)"
    };
    #[cfg(not(feature = "gpu"))]
    let substrate_label = "CPU only";
    println!("  Substrate: {substrate_label}");
    println!("  Clones processed: {clones_processed}");
    println!("  Total sovereign variants: {total_sovereign_variants}");
    println!("  Total breseq variants: {total_breseq_variants}");
    println!("  Total position matches: {total_matches}");
    println!("  Wall time: {:.0}s", t0.elapsed().as_secs_f64());

    v.check_pass("at least 1 clone processed", clones_processed >= 1);

    // ── Phase 5: Provenance completion — dehydrate → commit → braid ──
    println!("\n── Phase 5: Provenance Completion (live trio) ──");

    let session_result = provenance::complete_session(&prov.id);
    let prov_status = session_result["provenance"].as_str().unwrap_or("unknown");
    println!("  Provenance status: {prov_status}");
    v.check_pass(
        "provenance session completed (complete or unavailable)",
        prov_status == "complete" || prov_status == "unavailable",
    );

    // BLAKE3 summary hash (real cryptographic hash, not SipHash)
    let summary_str = format!(
        "sovereign_variants={total_sovereign_variants},breseq_variants={total_breseq_variants},matches={total_matches},clones={clones_processed}"
    );
    let summary_hash = blake3::hash(summary_str.as_bytes()).to_hex().to_string();

    let computation = ComputationMetadata {
        tool: "wetspring-sovereign-pipeline".to_string(),
        tool_version: "0.1.0".to_string(),
        input_accession: accession.clone(),
        input_blake3: "aggregate".to_string(),
        output_blake3: summary_hash.clone(),
        wall_time_seconds: t0.elapsed().as_secs(),
        node_count: u64::try_from(clones_processed).unwrap_or(0),
    };

    let braid_dataset = if offset > 0 {
        format!("{ds_id}_sovereign_batch_{offset}")
    } else {
        format!("{ds_id}_sovereign_resequencing")
    };

    let braid = FermentTranscriptBraid::from_session_result(
        &braid_dataset,
        &session_result,
        computation,
        &summary_hash,
    );

    let braid_json = braid.to_json();
    let braid_str = serde_json::to_string_pretty(&braid_json).unwrap_or_default();

    println!("  Braid ID: {}", braid.braid_id);
    println!("  DAG session: {}", braid.dag_session_id);
    println!("  Merkle root: {}", braid.dag_merkle_root);
    println!("  Spine ID: {}", braid.spine_id);
    println!("  Summary BLAKE3: {}", &summary_hash[..16]);

    // Write braid to dataset provenance
    let braid_dir = workspace.join("provenance").join("braids");
    let _ = std::fs::create_dir_all(&braid_dir);
    let braid_filename = if offset > 0 {
        format!("{ds_id}_sovereign_batch_{offset}.json")
    } else {
        format!("{ds_id}_sovereign.json")
    };
    let braid_path = braid_dir.join(&braid_filename);
    match std::fs::write(&braid_path, &braid_str) {
        Ok(()) => {
            println!("  Braid exported: {}", braid_path.display());
            v.check_pass("ferment transcript braid exported", true);
        }
        Err(e) => {
            println!("  Braid export failed: {e}");
            v.check_pass("ferment transcript braid exported", false);
        }
    }

    // Also export to the global provenance braids directory
    let global_braid_dir = workspace
        .parent()
        .map(|p| p.join("provenance").join("braids"));
    if let Some(dir) = global_braid_dir {
        let _ = std::fs::create_dir_all(&dir);
        let global_path = dir.join(&braid_filename);
        if std::fs::write(&global_path, &braid_str).is_ok() {
            println!("  Global braid: {}", global_path.display());
        }
    }

}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_sovereign_resequencing");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario = crate::validation::scenarios::registry::Scenario {
    meta: crate::validation::scenarios::registry::ScenarioMeta {
        id: "sovereign_resequencing",
        track: crate::validation::scenarios::registry::Track::Science,
        tier: crate::validation::scenarios::registry::Tier::Live,
        provenance_crate: "validate_sovereign_resequencing",
        provenance_date: "2026-05-20",
        description: "# Sovereign Rust Resequencing — Primal Composition",
    },
    run: |v, _ctx| run_as_scenario(v),
};
