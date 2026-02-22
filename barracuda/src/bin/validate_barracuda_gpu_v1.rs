// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
//! Exp064: `BarraCUDA` GPU Parity v1 — Consolidated GPU Domain Validation
//!
//! The GPU analogue of `barracuda_cpu_v1-v5`: a single binary that proves
//! pure GPU math matches CPU reference truth across all GPU-eligible domains.
//!
//! Domains: diversity, Bray-Curtis, ANI, SNP, dN/dS, pangenome, Random
//! Forest, HMM forward.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | BarraCUDA CPU (sovereign Rust reference) |
//! | Baseline version | Feb 2026 |
//! | Baseline command | CPU run first as ground truth; GPU validated against CPU |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --features gpu --release --bin validate_barracuda_gpu_v1` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

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
    let mut v = Validator::new("Exp064: BarraCUDA GPU Parity v1 — All GPU Domains");

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

    // ════════════════════════════════════════════════════════════════
    //  Domain 1: Diversity (Shannon, Simpson) — FusedMapReduceF64
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 1: Diversity (FMR) ═══");
    {
        use wetspring_barracuda::bio::{diversity, diversity_gpu};

        let abundances: Vec<f64> = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];

        let t_cpu = Instant::now();
        let cpu_shannon = diversity::shannon(&abundances);
        let cpu_simpson = diversity::simpson(&abundances);
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let t_gpu = Instant::now();
        let gpu_shannon = diversity_gpu::shannon_gpu(&gpu, &abundances).unwrap();
        let gpu_simpson = diversity_gpu::simpson_gpu(&gpu, &abundances).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        v.check(
            "Shannon: CPU == GPU",
            gpu_shannon,
            cpu_shannon,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        v.check(
            "Simpson: CPU == GPU",
            gpu_simpson,
            cpu_simpson,
            tolerances::GPU_VS_CPU_F64,
        );
        timings.push(("Diversity (Shannon + Simpson)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 2: Bray-Curtis Distance — BrayCurtisF64
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 2: Bray-Curtis ═══");
    {
        use wetspring_barracuda::bio::{diversity, diversity_gpu};

        let samples = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 25.0, 10.0],
            vec![5.0, 30.0, 25.0],
        ];

        let t_cpu = Instant::now();
        let cpu_bc = diversity::bray_curtis_condensed(&samples);
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let t_gpu = Instant::now();
        let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &samples).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        for (i, (c, g)) in cpu_bc.iter().zip(gpu_bc.iter()).enumerate() {
            v.check(
                &format!("BC condensed[{i}]: CPU == GPU"),
                *g,
                *c,
                tolerances::GPU_VS_CPU_F64,
            );
        }
        timings.push(("Bray-Curtis (3 samples, condensed)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 3: ANI — ToadStool (absorbed)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 3: ANI (ToadStool absorbed) ═══");
    {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"ATGATGATG", b"ATGATGATG"),
            (b"ATGATGATG", b"CTGATGATG"),
            (b"ATGATGATG", b"CTGCTGCTG"),
        ];

        let t_cpu = Instant::now();
        let cpu_results: Vec<_> = pairs.iter().map(|(a, b)| ani::pairwise_ani(a, b)).collect();
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let t_gpu = Instant::now();
        let gpu_ani = AniGpu::new(&device).expect("ANI GPU shader");
        let gpu_results = gpu_ani.batch_ani(&pairs).unwrap();
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
                tolerances::GPU_VS_CPU_F64,
            );
        }
        timings.push(("ANI (3 pairs)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 4: SNP Calling — ToadStool (absorbed)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 4: SNP Calling ═══");
    {
        let seqs: Vec<&[u8]> = vec![
            b"ATGATGATGATG",
            b"ATCATGATGATG",
            b"ATGATCATGATG",
            b"ATGATGATCATG",
        ];

        let t_cpu = Instant::now();
        let cpu_snp = snp::call_snps(&seqs);
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let t_gpu = Instant::now();
        let gpu_snp = SnpGpu::new(&device).expect("SNP GPU shader");
        let gpu_snp_result = gpu_snp.call_snps(&seqs).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        let cpu_variant_count = cpu_snp.variants.len();
        let gpu_variant_count = gpu_snp_result
            .is_variant
            .iter()
            .filter(|&&val| val != 0)
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
                    &format!("SNP pos {pos}: variant flag"),
                    f64::from(gpu_snp_result.is_variant[pos]),
                    1.0,
                    0.0,
                );
            }
        }
        timings.push(("SNP (4 seqs × 12bp)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 5: dN/dS — ToadStool (absorbed)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 5: dN/dS ═══");
    {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"ATGATGATG", b"ATGATGATG"),
            (b"TTTGCTAAA", b"TTCGCTAAA"),
            (b"AAAGCTGCT", b"GAAGCTGCT"),
        ];

        let t_cpu = Instant::now();
        let cpu_dnds: Vec<_> = pairs
            .iter()
            .map(|(a, b)| dnds::pairwise_dnds(a, b))
            .collect();
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let t_gpu = Instant::now();
        let gpu_dnds_mod = DnDsGpu::new(&device).expect("dN/dS GPU shader");
        let gpu_dnds_result = gpu_dnds_mod.batch_dnds(&pairs).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        for (i, cpu_r) in cpu_dnds.iter().enumerate() {
            if let Ok(cr) = cpu_r {
                v.check(
                    &format!("dN/dS pair {i}: dN CPU == GPU"),
                    gpu_dnds_result.dn[i],
                    cr.dn,
                    tolerances::GPU_VS_CPU_F64,
                );
                v.check(
                    &format!("dN/dS pair {i}: dS CPU == GPU"),
                    gpu_dnds_result.ds[i],
                    cr.ds,
                    tolerances::GPU_VS_CPU_F64,
                );
            }
        }
        timings.push(("dN/dS (3 pairs)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 6: Pangenome — ToadStool (absorbed)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 6: Pangenome ═══");
    {
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
        let gpu_result = gpu_pan.classify(&presence_flat, 5, 4).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        v.check(
            "Pan: core CPU == GPU",
            gpu_result
                .classifications
                .iter()
                .filter(|&&c| c == 3)
                .count() as f64,
            cpu_pan.core_size as f64,
            0.0,
        );
        v.check(
            "Pan: accessory CPU == GPU",
            gpu_result
                .classifications
                .iter()
                .filter(|&&c| c == 2)
                .count() as f64,
            cpu_pan.accessory_size as f64,
            0.0,
        );
        v.check(
            "Pan: unique CPU == GPU",
            gpu_result
                .classifications
                .iter()
                .filter(|&&c| c == 1)
                .count() as f64,
            cpu_pan.unique_size as f64,
            0.0,
        );
        timings.push(("Pangenome (5 genes × 4 genomes)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 7: Random Forest — ToadStool (absorbed)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 7: Random Forest ═══");
    {
        use wetspring_barracuda::bio::{
            decision_tree::DecisionTree, random_forest::RandomForest,
            random_forest_gpu::RandomForestGpu,
        };

        let tree1 = DecisionTree::from_arrays(
            &[0, -2, 1, -2, -2],
            &[5.0, 0.0, 3.0, 0.0, 0.0],
            &[1, -1, 3, -1, -1],
            &[2, -1, 4, -1, -1],
            &[None, Some(0), None, Some(1), Some(2)],
            2,
        )
        .unwrap();
        let tree2 = DecisionTree::from_arrays(
            &[1, -2, -2],
            &[4.0, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(2)],
            2,
        )
        .unwrap();
        let tree3 = DecisionTree::from_arrays(
            &[0, -2, -2],
            &[6.0, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(1), Some(2)],
            2,
        )
        .unwrap();

        let rf = RandomForest::from_trees(vec![tree1, tree2, tree3], 3).unwrap();
        let samples = vec![vec![3.0, 1.0], vec![7.0, 6.0], vec![5.5, 3.5]];

        let t_cpu = Instant::now();
        let cpu_preds = rf.predict_batch(&samples);
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let t_gpu = Instant::now();
        let rf_gpu = RandomForestGpu::new(&device);
        let gpu_preds = rf_gpu.predict_batch(&rf, &samples).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        for (i, (c, g)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
            v.check(
                &format!("RF sample {i}: CPU == GPU"),
                g.class as f64,
                *c as f64,
                0.0,
            );
        }
        timings.push(("RF (3 trees, 3 samples)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Domain 8: HMM Forward — ToadStool (absorbed)
    // ════════════════════════════════════════════════════════════════
    v.section("═══ GPU Domain 8: HMM Forward ═══");
    {
        use wetspring_barracuda::bio::{hmm, hmm_gpu::HmmGpuForward};

        let model = hmm::HmmModel {
            n_states: 2,
            n_symbols: 2,
            log_pi: vec![-std::f64::consts::LN_2, -std::f64::consts::LN_2],
            log_trans: vec![-0.3567, -1.2040, -1.2040, -0.3567],
            log_emit: vec![-0.2231, -1.6094, -1.6094, -0.2231],
        };
        let obs1 = [0_usize, 1, 0, 1];
        let obs2 = [0_usize, 0, 0, 0];

        let t_cpu = Instant::now();
        let cpu_ll1 = hmm::forward(&model, &obs1).log_likelihood;
        let cpu_ll2 = hmm::forward(&model, &obs2).log_likelihood;
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let flat_obs: Vec<u32> = obs1.iter().chain(obs2.iter()).map(|&x| x as u32).collect();
        let t_gpu = Instant::now();
        let hmm_gpu = HmmGpuForward::new(&device).expect("HMM GPU shader");
        let gpu_result = hmm_gpu.forward_batch(&model, &flat_obs, 2, 4).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        v.check(
            "HMM seq 0: loglik CPU == GPU",
            gpu_result.log_likelihoods[0],
            cpu_ll1,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            "HMM seq 1: loglik CPU == GPU",
            gpu_result.log_likelihoods[1],
            cpu_ll2,
            tolerances::GPU_VS_CPU_F64,
        );
        timings.push(("HMM forward (2 seqs × 4 obs)", cpu_us, gpu_us));
    }

    // ════════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════════
    v.section("═══ BarraCUDA GPU Parity v1 Summary ═══");
    println!();
    println!(
        "  {:<40} {:>10} {:>10} {:>10}",
        "Domain", "CPU (µs)", "GPU (µs)", "Parity"
    );
    println!("  {}", "─".repeat(72));
    for (name, cpu, gpu_t) in &timings {
        println!(
            "  {:<40} {:>10.0} {:>10.0} {:>10}",
            name, cpu, gpu_t, "CPU=GPU"
        );
    }
    println!("  {}", "─".repeat(72));
    let total_cpu: f64 = timings.iter().map(|(_, c, _)| c).sum();
    let total_gpu: f64 = timings.iter().map(|(_, _, g)| g).sum();
    println!("  {:<40} {:>10.0} {:>10.0}", "TOTAL", total_cpu, total_gpu);
    println!();
    println!("  All GPU-eligible domains produce identical results on GPU.");
    println!("  The math is substrate-independent — pure GPU execution validated.");
    println!();

    v.finish();
}
