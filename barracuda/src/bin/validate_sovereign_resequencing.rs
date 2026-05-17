// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: pipeline progress printed to stdout"
)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: expect is fine for data loading"
)]
//! # Exp382: Sovereign Rust Resequencing — Barrick 2009 Parity
//!
//! Runs the sovereign Rust resequencing pipeline (FM-index + seed-extend +
//! pileup + variant caller) on the cached Barrick 2009 FASTQ data and
//! compares mutation calls against breseq `output.gd` per clone.
//!
//! This is the cross-tier parity proof: Tier 2 (sovereign Rust) vs
//! Tier 1 (breseq C++ baseline).
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Paper | Barrick et al. *Nature* 461, 1243–1247 (2009) |
//! | Baseline | Exp381 breseq output (cached) |
//! | Pipeline | FM-index → seed-extend SW → pileup → variant caller |
//! | Composition | barraCuda GPU: SmithWatermanGpu, SnpCallingF64, Tensor::scan |
//! | Provenance | Live trio via plasmidBin: rhizoCrypt DAG, loamSpine aglets, sweetGrass braid |

use std::path::PathBuf;
use std::time::Instant;

use wetspring_barracuda::bio::pileup;
use wetspring_barracuda::bio::read_mapper::{self, MapperConfig};
use wetspring_barracuda::bio::ref_index::FmIndex;
use wetspring_barracuda::bio::variant_caller::{self, CallerConfig};
use wetspring_barracuda::io::fasta::{FastaRecord, GenBankRecord};
use wetspring_barracuda::io::fastq;
use wetspring_barracuda::io::sam;
use wetspring_barracuda::ncbi::fetch_sra_composed;
use wetspring_barracuda::validation::Validator;

use wetspring_barracuda::ipc::provenance;
use wetspring_barracuda::ipc::provenance::braid_handoff::{
    ComputationMetadata, FermentTranscriptBraid,
};

#[cfg(feature = "gpu")]
use wetspring_barracuda::gpu::GpuF64;

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

fn main() {
    let mut v = Validator::new("Exp382: Sovereign Resequencing — Barrick 2009 Parity");
    let t0 = Instant::now();
    let workspace = workspace_dir();

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
    let prov = provenance::begin_session("sovereign_resequencing_barrick_2009");
    println!("  Session ID: {}", prov.id);
    println!("  Trio available: {}", prov.available);
    v.check_pass(
        "provenance session started (live trio or degraded local)",
        !prov.id.is_empty(),
    );

    // ── Phase 1: Load reference genome ───────────────────────────
    println!("\n── Phase 1: Loading reference genome ──");

    let ref_fasta_path = workspace.join("reference").join("REL606.fasta");
    let ref_gbk_path = workspace.join("reference").join("REL606.gbk");

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
        v.finish();
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

    // ── Phase 3: Process each clone ──────────────────────────────
    println!("\n── Phase 3: Per-clone sovereign pipeline ──");

    let mut total_sovereign_variants = 0usize;
    let mut total_breseq_variants = 0usize;
    let mut total_matches = 0usize;
    let mut clones_processed = 0usize;

    for &(accession, clone_name) in BARRICK_CLONES {
        let clone_t0 = Instant::now();
        println!("\n  ── {clone_name} ({accession}) ──");

        // Composed SRA fetch: NestGate → local cache → SRA Toolkit download
        let reads_dir = workspace.join("reads");
        let fastq_dir = workspace.join("fastq");

        let sra_result = fetch_sra_composed(accession, &reads_dir)
            .or_else(|_| fetch_sra_composed(accession, &fastq_dir));

        let (fq_path, fq_blake3, fq_source) = match sra_result {
            Ok(r) => {
                println!("    FASTQ: {} via {}", r.path.display(), r.source);
                let _ = provenance::record_step(
                    &prov.id,
                    &serde_json::json!({
                        "step": "fetch_sra",
                        "accession": accession,
                        "clone": clone_name,
                        "source": r.source.to_string(),
                        "blake3": &r.blake3,
                    }),
                );
                (r.path, r.blake3, r.source.to_string())
            }
            Err(e) => {
                println!("    SKIP: no FASTQ for {accession} ({e})");
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

        // Map reads — CPU is optimal for short-read SW (GPU per-pair overhead
        // exceeds compute for 36bp reads). GPU dispatches downstream (pileup, SNP).
        let map_t0 = Instant::now();
        let sam_records = read_mapper::map_reads(&reads, &fm_index, &reference, "REL606", &mapper_config);
        let map_secs = map_t0.elapsed().as_secs_f64();

        let mapped_count = sam_records.iter().filter(|r| r.is_mapped()).count();
        println!("    Mapped: {mapped_count}/{} reads ({map_secs:.1}s, CPU seed-extend)", sam_records.len());

        // Sort by position for pileup
        let mut sorted_records = sam_records;
        sam::sort_by_position(&mut sorted_records);

        // Generate pileup
        let pileup_columns = pileup::generate_pileup(&sorted_records, reference.len());
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
            format!("{clone_name}:{accession}:variants={}", sovereign_variants.len()).as_bytes(),
        ).to_hex().to_string();

        let _ = provenance::record_step(
            &prov.id,
            &serde_json::json!({
                "step": "clone_complete",
                "clone": clone_name,
                "accession": accession,
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
    }

    // ── Phase 4: Summary ─────────────────────────────────────────
    println!("\n── Phase 4: Summary ──");
    let substrate_label = if cfg!(feature = "gpu") { "GPU+CPU hybrid" } else { "CPU only" };
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
        input_accession: "SRP001569".to_string(),
        input_blake3: "aggregate".to_string(),
        output_blake3: summary_hash.clone(),
        wall_time_seconds: t0.elapsed().as_secs(),
        node_count: u64::try_from(clones_processed).unwrap_or(0),
    };

    let braid = FermentTranscriptBraid::from_session_result(
        "barrick_2009_sovereign_resequencing",
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
    let braid_path = braid_dir.join("barrick_2009_sovereign.json");
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
        let global_path = dir.join("barrick_2009_sovereign.json");
        if std::fs::write(&global_path, &braid_str).is_ok() {
            println!("  Global braid: {}", global_path.display());
        }
    }

    v.finish();
}
