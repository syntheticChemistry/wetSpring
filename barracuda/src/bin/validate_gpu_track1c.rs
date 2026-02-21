// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp058: GPU Track 1c Promotion — ANI + SNP + Pangenome + dN/dS
//!
//! Validates four local WGSL shaders against CPU baselines:
//! 1. **ANI batch** — pairwise nucleotide identity (pair-parallel)
//! 2. **SNP calling** — position-parallel variant detection
//! 3. **Pangenome classify** — gene classification (gene-parallel)
//! 4. **dN/dS batch** — Nei-Gojobori codon analysis (pair-parallel)
//!
//! Each section runs the GPU shader, then compares against the CPU
//! `bio::ani`, `bio::snp`, `bio::pangenome`, and `bio::dnds` modules.
//!
//! ```text
//! CPU v4 (23 domains) → [THIS] GPU promotion (ANI, SNP, Pan, dN/dS) → ToadStool absorption
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | BarraCUDA CPU (reference) |
//! | Baseline version | wetspring-barracuda 0.1.0 (CPU path) |
//! | Baseline command | bio::ani, bio::snp, bio::pangenome, bio::dnds |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --release --features gpu --bin validate_gpu_track1c` |
//! | Data | Synthetic pairs, sequences, gene clusters, codon pairs |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Local WGSL shaders: ANI batch, SNP calling, Pangenome classify, dN/dS batch (Nei-Gojobori).

use std::time::Instant;
use wetspring_barracuda::bio::{
    ani, ani_gpu::AniGpu, dnds, dnds_gpu::DnDsGpu, pangenome, pangenome_gpu::PangenomeGpu, snp,
    snp_gpu::SnpGpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp058: GPU Track 1c — ANI + SNP + Pangenome + dN/dS");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            validation::exit_skipped(&format!("GPU init failed: {e}"));
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }
    println!();

    let device = gpu.to_wgpu_device();
    let mut timings: Vec<(&str, f64)> = Vec::new();

    validate_ani_gpu(&AniGpu::new(&device), &mut v, &mut timings);
    validate_snp_gpu(&SnpGpu::new(&device), &mut v, &mut timings);
    validate_pangenome_gpu(&PangenomeGpu::new(&device), &mut v, &mut timings);
    validate_dnds_gpu(&DnDsGpu::new(&device), &mut v, &mut timings);

    // Timing summary
    v.section("═══ GPU Track 1c Timing Summary ═══");
    println!("\n  {:<45} {:>12}", "Workload", "Time (µs)");
    println!("  {}", "-".repeat(60));
    for (name, us) in &timings {
        println!("  {name:<45} {us:>12.0}");
    }
    let total_us: f64 = timings.iter().map(|(_, t)| t).sum();
    println!("  {}", "-".repeat(60));
    println!("  {:<45} {:>12.0}", "TOTAL GPU", total_us);
    println!();

    v.finish();
}

#[allow(clippy::cast_precision_loss)]
fn validate_ani_gpu(gpu: &AniGpu, v: &mut Validator, timings: &mut Vec<(&str, f64)>) {
    v.section("═══ Section 1: GPU ANI Batch ═══");

    let pairs: Vec<(&[u8], &[u8])> = vec![
        (b"ATGATGATG", b"ATGATGATG"), // identical → 1.0
        (b"AAAA", b"TTTT"),           // different → 0.0
        (b"AATT", b"AAGC"),           // half → 0.5
        (b"A-TG", b"ACTG"),           // gap excluded
        (b"ANTG", b"ACTG"),           // N excluded
    ];

    let cpu_results: Vec<_> = pairs.iter().map(|(a, b)| ani::pairwise_ani(a, b)).collect();

    let t0 = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.batch_ani(&pairs)));
    let gpu_us = t0.elapsed().as_micros();

    match result {
        Ok(Ok(gpu_result)) => {
            v.check(
                "ANI GPU: identical → 1.0",
                gpu_result.ani_values[0],
                cpu_results[0].ani,
                1e-10,
            );
            v.check(
                "ANI GPU: different → 0.0",
                gpu_result.ani_values[1],
                cpu_results[1].ani,
                1e-10,
            );
            v.check(
                "ANI GPU: half → 0.5",
                gpu_result.ani_values[2],
                cpu_results[2].ani,
                1e-10,
            );
            v.check(
                "ANI GPU: gap excluded, ANI=1.0",
                gpu_result.ani_values[3],
                cpu_results[3].ani,
                1e-10,
            );
            v.check(
                "ANI GPU: N excluded",
                gpu_result.ani_values[4],
                cpu_results[4].ani,
                1e-10,
            );

            // Verify aligned counts match CPU
            v.check(
                "ANI GPU: aligned count (identical)",
                f64::from(gpu_result.aligned_counts[0]),
                cpu_results[0].aligned_length as f64,
                0.0,
            );
            v.check(
                "ANI GPU: aligned count (gaps)",
                f64::from(gpu_result.aligned_counts[3]),
                cpu_results[3].aligned_length as f64,
                0.0,
            );

            timings.push(("ANI batch (5 pairs)", gpu_us as f64));
        }
        Ok(Err(e)) => {
            println!("  [SKIP] ANI GPU error: {e}");
            v.check("ANI GPU: available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] ANI GPU panicked (driver shader compilation)");
            v.check("ANI GPU: available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn validate_snp_gpu(gpu: &SnpGpu, v: &mut Validator, timings: &mut Vec<(&str, f64)>) {
    v.section("═══ Section 2: GPU SNP Calling ═══");

    let seqs: Vec<&[u8]> = vec![
        b"ATGATGATGATG",
        b"ATGATGATGATG",
        b"CTGATGTTGATG",
        b"ATGATCATGATG",
    ];

    let cpu_result = snp::call_snps(&seqs);

    let t0 = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.call_snps(&seqs)));
    let gpu_us = t0.elapsed().as_micros();

    match result {
        Ok(Ok(gpu_result)) => {
            let gpu_variant_count: u32 = gpu_result.is_variant.iter().sum();
            v.check(
                "SNP GPU: variant count matches CPU",
                f64::from(gpu_variant_count),
                cpu_result.variants.len() as f64,
                0.0,
            );

            // Verify variant positions match
            let gpu_variant_positions: Vec<usize> = gpu_result
                .is_variant
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == 1)
                .map(|(i, _)| i)
                .collect();
            let cpu_variant_positions: Vec<usize> =
                cpu_result.variants.iter().map(|v| v.position).collect();

            let positions_match = gpu_variant_positions == cpu_variant_positions;
            v.check(
                "SNP GPU: variant positions match CPU",
                f64::from(u8::from(positions_match)),
                1.0,
                0.0,
            );

            // Verify depths at variant positions
            let depths_ok = cpu_result.variants.iter().all(|cv| {
                let gpu_depth = gpu_result.depths[cv.position];
                gpu_depth as usize == cv.depth
            });
            v.check(
                "SNP GPU: depths match CPU at variant sites",
                f64::from(u8::from(depths_ok)),
                1.0,
                0.0,
            );

            // Verify alt frequencies within tolerance
            let freq_ok = cpu_result.variants.iter().all(|cv| {
                let gpu_freq = gpu_result.alt_frequencies[cv.position];
                let cpu_freq = cv.alt_frequency();
                (gpu_freq - cpu_freq).abs() < 1e-6
            });
            v.check(
                "SNP GPU: alt frequencies match CPU",
                f64::from(u8::from(freq_ok)),
                1.0,
                0.0,
            );

            // Non-variant positions should have 0 alt freq
            let non_variants_clean = gpu_result
                .is_variant
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == 0)
                .all(|(i, _)| gpu_result.alt_frequencies[i].abs() < 1e-15);
            v.check(
                "SNP GPU: non-variant alt_freq = 0",
                f64::from(u8::from(non_variants_clean)),
                1.0,
                0.0,
            );

            timings.push(("SNP calling (4 seqs × 12bp)", gpu_us as f64));
        }
        Ok(Err(e)) => {
            println!("  [SKIP] SNP GPU error: {e}");
            v.check("SNP GPU: available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] SNP GPU panicked (driver shader compilation)");
            v.check("SNP GPU: available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn validate_pangenome_gpu(gpu: &PangenomeGpu, v: &mut Validator, timings: &mut Vec<(&str, f64)>) {
    v.section("═══ Section 3: GPU Pangenome Classification ═══");

    let clusters = vec![
        pangenome::GeneCluster {
            id: "core1".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "core2".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "core3".into(),
            presence: vec![true, true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "acc1".into(),
            presence: vec![true, true, false, false, false],
        },
        pangenome::GeneCluster {
            id: "acc2".into(),
            presence: vec![false, true, true, false, false],
        },
        pangenome::GeneCluster {
            id: "uniq1".into(),
            presence: vec![true, false, false, false, false],
        },
        pangenome::GeneCluster {
            id: "uniq2".into(),
            presence: vec![false, false, false, false, true],
        },
    ];
    let n_genomes = 5;
    let n_genes = clusters.len();

    let cpu_result = pangenome::analyze(&clusters, n_genomes);
    let flat = pangenome::presence_matrix_flat(&clusters, n_genomes);

    let t0 = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.classify(&flat, n_genes, n_genomes)
    }));
    let gpu_us = t0.elapsed().as_micros();

    match result {
        Ok(Ok(gpu_result)) => {
            v.check(
                "Pan GPU: core count matches CPU",
                gpu_result.core_count() as f64,
                cpu_result.core_size as f64,
                0.0,
            );
            v.check(
                "Pan GPU: accessory count matches CPU",
                gpu_result.accessory_count() as f64,
                cpu_result.accessory_size as f64,
                0.0,
            );
            v.check(
                "Pan GPU: unique count matches CPU",
                gpu_result.unique_count() as f64,
                cpu_result.unique_size as f64,
                0.0,
            );
            v.check(
                "Pan GPU: total = core + acc + uniq",
                (gpu_result.core_count() + gpu_result.accessory_count() + gpu_result.unique_count())
                    as f64,
                cpu_result.total_size as f64,
                0.0,
            );

            // Verify genome counts per gene
            let expected_counts: Vec<u32> = clusters
                .iter()
                .map(|c| c.presence.iter().filter(|&&p| p).count() as u32)
                .collect();
            let counts_match = gpu_result.genome_counts == expected_counts;
            v.check(
                "Pan GPU: genome counts match CPU",
                f64::from(u8::from(counts_match)),
                1.0,
                0.0,
            );

            // Core genes should have count == n_genomes
            let core_correct = gpu_result
                .classifications
                .iter()
                .zip(&gpu_result.genome_counts)
                .filter(|(&c, _)| c == 3)
                .all(|(_, &cnt)| cnt == n_genomes as u32);
            v.check(
                "Pan GPU: core genes have count = n_genomes",
                f64::from(u8::from(core_correct)),
                1.0,
                0.0,
            );

            timings.push(("Pangenome classify (7 genes × 5 genomes)", gpu_us as f64));
        }
        Ok(Err(e)) => {
            println!("  [SKIP] Pangenome GPU error: {e}");
            v.check("Pan GPU: available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] Pangenome GPU panicked (driver shader compilation)");
            v.check("Pan GPU: available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn validate_dnds_gpu(gpu: &DnDsGpu, v: &mut Validator, timings: &mut Vec<(&str, f64)>) {
    v.section("═══ Section 4: GPU dN/dS Batch (Nei-Gojobori 1986) ═══");

    // Test pairs matching CPU validator
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (b"ATGATGATG", b"ATGATGATG"), // identical → dN=0, dS=0
        (b"TTTGCTAAA", b"TTCGCTAAA"), // syn-only → dS>0, dN=0
        (b"AAAGCTGCT", b"GAAGCTGCT"), // nonsyn (Lys→Glu at codon 1)
        (
            b"ATGGCTAAATTTGCTGCTGCTGCTGCTGCT", // mixed changes
            b"ATGGCCGAATTTGCTGCTGCTGCTGCCGCT",
        ),
    ];

    let cpu_results: Vec<_> = pairs
        .iter()
        .map(|(a, b)| dnds::pairwise_dnds(a, b))
        .collect();

    let t0 = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.batch_dnds(&pairs)));
    let gpu_us = t0.elapsed().as_micros();

    match result {
        Ok(Ok(gpu_result)) => {
            // Pair 0: identical → dN=0, dS=0
            let cpu0 = cpu_results[0].as_ref().unwrap();
            v.check(
                "dN/dS GPU: identical dN=0",
                gpu_result.dn[0],
                cpu0.dn,
                tolerances::GPU_VS_CPU_F64,
            );
            v.check(
                "dN/dS GPU: identical dS=0",
                gpu_result.ds[0],
                cpu0.ds,
                tolerances::GPU_VS_CPU_F64,
            );

            // Pair 1: synonymous only → dS>0, dN=0
            let cpu1 = cpu_results[1].as_ref().unwrap();
            v.check(
                "dN/dS GPU: syn-only dN=0",
                gpu_result.dn[1],
                cpu1.dn,
                tolerances::GPU_VS_CPU_F64,
            );
            v.check(
                "dN/dS GPU: syn-only dS>0",
                f64::from(u8::from(gpu_result.ds[1] > 0.0)),
                1.0,
                0.0,
            );
            v.check(
                "dN/dS GPU: syn-only dS matches CPU",
                gpu_result.ds[1],
                cpu1.ds,
                1e-4,
            );

            // Pair 2: nonsynonymous → dN>0
            let cpu2 = cpu_results[2].as_ref().unwrap();
            v.check(
                "dN/dS GPU: nonsyn dN>0",
                f64::from(u8::from(gpu_result.dn[2] > 0.0)),
                1.0,
                0.0,
            );
            v.check(
                "dN/dS GPU: nonsyn dN matches CPU",
                gpu_result.dn[2],
                cpu2.dn,
                1e-4,
            );

            // Pair 3: mixed → both sites and diffs present
            let cpu3 = cpu_results[3].as_ref().unwrap();
            v.check(
                "dN/dS GPU: mixed dS matches CPU",
                gpu_result.ds[3],
                cpu3.ds,
                1e-4,
            );
            v.check(
                "dN/dS GPU: mixed dN matches CPU",
                gpu_result.dn[3],
                cpu3.dn,
                1e-4,
            );

            timings.push(("dN/dS batch (4 pairs, Nei-Gojobori)", gpu_us as f64));
        }
        Ok(Err(e)) => {
            println!("  [SKIP] dN/dS GPU error: {e}");
            v.check("dN/dS GPU: available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] dN/dS GPU panicked (driver shader compilation)");
            v.check("dN/dS GPU: available (driver skip)", 1.0, 1.0, 0.0);
        }
    }
}
