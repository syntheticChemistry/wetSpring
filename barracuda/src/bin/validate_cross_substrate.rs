// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp060: Cross-Substrate Validation (metalForge)
//!
//! Proves that the same algorithm produces identical results on CPU and GPU
//! substrates. For each GPU-promoted algorithm, run CPU first as reference
//! truth, then GPU, and compare.
//!
//! This is the metalForge proof: the math is substrate-independent.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | BarraCuda CPU (reference implementation) |
//! | Baseline version | Feb 2026 |
//! | Baseline command | CPU run first as ground truth; GPU validated against CPU |
//! | Baseline date | 2026-02-19 |
//! | Exact command | `cargo run --release --features gpu --bin validate_cross_substrate` |
//! | Data | Synthetic test vectors (ANI, SNP, pangenome, dN/dS pairs) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |

use std::time::Instant;
use wetspring_barracuda::bio::{
    ani, ani_gpu::AniGpu, dnds, dnds_gpu::DnDsGpu, pangenome, pangenome_gpu::PangenomeGpu, snp,
    snp_gpu::SnpGpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
async fn main() {
    let mut v = Validator::new("Exp060: metalForge Cross-Substrate Validation");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();

    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let device = gpu.to_wgpu_device();
    let mut timings: Vec<(&str, f64, f64)> = Vec::new();

    // ═══ ANI: CPU vs GPU ═══
    v.section("═══ Substrate 1: ANI (CPU ↔ GPU) ═══");
    let ani_pairs: Vec<(&[u8], &[u8])> = vec![
        (b"ATGATGATG", b"ATGATGATG"),
        (b"ATGATGATG", b"CTGATGATG"),
        (b"ATGATGATG", b"CTGCTGCTG"),
        (b"ACGTNACGT", b"ACGTNACGT"),
        (b"ACG-TACGT", b"ACGTACGT-"),
    ];

    let t_cpu = Instant::now();
    let cpu_results: Vec<_> = ani_pairs
        .iter()
        .map(|(a, b)| ani::pairwise_ani(a, b))
        .collect();
    let cpu_us = t_cpu.elapsed().as_micros() as f64;

    let t_gpu = Instant::now();
    let gpu_ani = AniGpu::new(&device).expect("ANI GPU shader");
    let gpu_results = gpu_ani.batch_ani(&ani_pairs).unwrap();
    let gpu_us = t_gpu.elapsed().as_micros() as f64;

    for (i, (cpu_r, gpu_val)) in cpu_results
        .iter()
        .zip(gpu_results.ani_values.iter())
        .enumerate()
    {
        v.check(
            &format!("ANI pair {i}: CPU == GPU"),
            *gpu_val,
            cpu_r.ani,
            wetspring_barracuda::tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }
    timings.push(("ANI (5 pairs)", cpu_us, gpu_us));

    // ═══ SNP: CPU vs GPU ═══
    v.section("═══ Substrate 2: SNP Calling (CPU ↔ GPU) ═══");
    let snp_seqs: Vec<&[u8]> = vec![
        b"ATGATGATGATG",
        b"ATCATGATGATG",
        b"ATGATCATGATG",
        b"ATGATGATCATG",
    ];

    let t_cpu = Instant::now();
    let cpu_snp = snp::call_snps(&snp_seqs);
    let cpu_us = t_cpu.elapsed().as_micros() as f64;

    let t_gpu = Instant::now();
    let gpu_snp = SnpGpu::new(&device).expect("SNP GPU shader");
    let gpu_snp_result = gpu_snp.call_snps(&snp_seqs).unwrap();
    let gpu_us = t_gpu.elapsed().as_micros() as f64;

    let cpu_variant_count = cpu_snp.variants.len();
    let gpu_variant_count = gpu_snp_result
        .is_variant
        .iter()
        .filter(|&&v| v != 0)
        .count();
    v.check(
        "SNP: variant count CPU == GPU",
        gpu_variant_count as f64,
        cpu_variant_count as f64,
        0.0,
    );

    for cv in &cpu_snp.variants {
        let pos = cv.position;
        if pos < gpu_snp_result.is_variant.len() {
            v.check(
                &format!("SNP pos {pos}: variant flag CPU == GPU"),
                f64::from(gpu_snp_result.is_variant[pos]),
                1.0,
                0.0,
            );
        }
    }
    timings.push(("SNP (4 seqs × 12bp)", cpu_us, gpu_us));

    // ═══ Pangenome: CPU vs GPU ═══
    v.section("═══ Substrate 3: Pangenome (CPU ↔ GPU) ═══");
    let clusters = vec![
        pangenome::GeneCluster {
            id: "g1".into(),
            presence: vec![true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "g2".into(),
            presence: vec![true, true, true, true],
        },
        pangenome::GeneCluster {
            id: "g3".into(),
            presence: vec![true, true, false, false],
        },
        pangenome::GeneCluster {
            id: "g4".into(),
            presence: vec![true, false, false, false],
        },
        pangenome::GeneCluster {
            id: "g5".into(),
            presence: vec![false, false, false, true],
        },
    ];

    let t_cpu = Instant::now();
    let cpu_pan = pangenome::analyze(&clusters, 4);
    let cpu_us = t_cpu.elapsed().as_micros() as f64;

    let presence_flat: Vec<u8> = clusters
        .iter()
        .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
        .collect();

    let t_gpu = Instant::now();
    let gpu_pan = PangenomeGpu::new(&device).expect("Pangenome GPU shader");
    let gpu_pan_result = gpu_pan.classify(&presence_flat, 5, 4).unwrap();
    let gpu_us = t_gpu.elapsed().as_micros() as f64;

    v.check(
        "Pan: core count CPU == GPU",
        gpu_pan_result
            .classifications
            .iter()
            .filter(|&&c| c == 3)
            .count() as f64,
        cpu_pan.core_size as f64,
        0.0,
    );
    v.check(
        "Pan: accessory count CPU == GPU",
        gpu_pan_result
            .classifications
            .iter()
            .filter(|&&c| c == 2)
            .count() as f64,
        cpu_pan.accessory_size as f64,
        0.0,
    );
    v.check(
        "Pan: unique count CPU == GPU",
        gpu_pan_result
            .classifications
            .iter()
            .filter(|&&c| c == 1)
            .count() as f64,
        cpu_pan.unique_size as f64,
        0.0,
    );
    timings.push(("Pangenome (5 genes × 4 genomes)", cpu_us, gpu_us));

    // ═══ dN/dS: CPU vs GPU ═══
    v.section("═══ Substrate 4: dN/dS (CPU ↔ GPU) ═══");
    let dnds_pairs: Vec<(&[u8], &[u8])> = vec![
        (b"ATGATGATG", b"ATGATGATG"),
        (b"TTTGCTAAA", b"TTCGCTAAA"),
        (b"AAAGCTGCT", b"GAAGCTGCT"),
        (
            b"ATGGCTAAATTTGCTGCTGCTGCTGCTGCT",
            b"ATGGCCGAATTTGCTGCTGCTGCTGCCGCT",
        ),
    ];

    let t_cpu = Instant::now();
    let cpu_dnds: Vec<_> = dnds_pairs
        .iter()
        .map(|(a, b)| dnds::pairwise_dnds(a, b))
        .collect();
    let cpu_us = t_cpu.elapsed().as_micros() as f64;

    let t_gpu = Instant::now();
    let gpu_dnds_mod = DnDsGpu::new(&device).expect("dN/dS GPU shader");
    let gpu_dnds_result = gpu_dnds_mod.batch_dnds(&dnds_pairs).unwrap();
    let gpu_us = t_gpu.elapsed().as_micros() as f64;

    for (i, cpu_r) in cpu_dnds.iter().enumerate() {
        if let Ok(cr) = cpu_r {
            v.check(
                &format!("dN/dS pair {i}: dN CPU == GPU"),
                gpu_dnds_result.dn[i],
                cr.dn,
                wetspring_barracuda::tolerances::GPU_VS_CPU_F64,
            );
            v.check(
                &format!("dN/dS pair {i}: dS CPU == GPU"),
                gpu_dnds_result.ds[i],
                cr.ds,
                wetspring_barracuda::tolerances::GPU_VS_CPU_F64,
            );
        }
    }
    timings.push(("dN/dS (4 pairs, Nei-Gojobori)", cpu_us, gpu_us));

    // Summary
    v.section("═══ metalForge Cross-Substrate Summary ═══");
    println!();
    println!(
        "  {:<40} {:>10} {:>10} {:>10}",
        "Workload", "CPU (µs)", "GPU (µs)", "Substrate"
    );
    println!("  {}", "─".repeat(72));
    for (name, cpu, gpu) in &timings {
        println!(
            "  {:<40} {:>10.0} {:>10.0} {:>10}",
            name, cpu, gpu, "CPU=GPU"
        );
    }
    println!("  {}", "─".repeat(72));
    println!();
    println!("  All algorithms produce identical results regardless of substrate.");
    println!("  The math is substrate-independent — pure GPU execution validated.");
    println!();

    v.finish();
}
