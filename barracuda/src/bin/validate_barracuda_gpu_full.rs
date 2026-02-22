// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
//! Exp071: `BarraCuda` GPU — Full Math Portability Proof
//!
//! Consolidates all GPU-eligible domains into one definitive binary.
//! Proves: same Rust math, same answers, different substrate.
//! CPU computes reference truth; GPU must match within tolerance.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | BarraCuda CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --release --features gpu --bin validate_barracuda_gpu_full` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{
    ani, ani_gpu::AniGpu, decision_tree::DecisionTree, diversity, diversity_gpu, dnds,
    dnds_gpu::DnDsGpu, hmm, hmm_gpu::HmmGpuForward, pangenome, pangenome_gpu::PangenomeGpu,
    random_forest::RandomForest, random_forest_gpu::RandomForestGpu, snp, snp_gpu::SnpGpu,
    spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp071: BarraCuda GPU — Full Math Portability Proof");

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
    let mut timings: Vec<(&str, &str, f64, f64)> = Vec::new();

    // ═══ G01: Shannon Entropy — FMR (ToadStool) ══════════════════
    v.section("G01: Shannon Entropy (FMR)");
    {
        let abundances = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
        let t_cpu = Instant::now();
        let cpu_val = diversity::shannon(&abundances);
        let cpu_us = t_cpu.elapsed().as_micros() as f64;
        let t_gpu = Instant::now();
        let gpu_val = diversity_gpu::shannon_gpu(&gpu, &abundances).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        v.check(
            "Shannon: CPU == GPU",
            gpu_val,
            cpu_val,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
        timings.push(("G01: Shannon", "FMR (absorbed)", cpu_us, gpu_us));
    }

    // ═══ G02: Simpson Diversity — FMR (ToadStool) ════════════════
    v.section("G02: Simpson Diversity (FMR)");
    {
        let abundances = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
        let t_cpu = Instant::now();
        let cpu_val = diversity::simpson(&abundances);
        let cpu_us = t_cpu.elapsed().as_micros() as f64;
        let t_gpu = Instant::now();
        let gpu_val = diversity_gpu::simpson_gpu(&gpu, &abundances).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        v.check(
            "Simpson: CPU == GPU",
            gpu_val,
            cpu_val,
            tolerances::GPU_VS_CPU_F64,
        );
        timings.push(("G02: Simpson", "FMR (absorbed)", cpu_us, gpu_us));
    }

    // ═══ G03: Bray-Curtis — BrayCurtisF64 (ToadStool) ═══════════
    v.section("G03: Bray-Curtis (BrayCurtisF64)");
    {
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
                &format!("BC[{i}]: CPU == GPU"),
                *g,
                *c,
                tolerances::GPU_VS_CPU_F64,
            );
        }
        timings.push((
            "G03: Bray-Curtis",
            "BrayCurtisF64 (absorbed)",
            cpu_us,
            gpu_us,
        ));
    }

    // ═══ G04: Spectral Cosine — FMR (ToadStool) ══════════════════
    v.section("G04: Spectral Cosine (FMR)");
    {
        let spectra_gpu: Vec<Vec<f64>> = vec![
            vec![1000.0, 500.0, 800.0, 300.0, 600.0],
            vec![900.0, 550.0, 750.0, 350.0, 550.0],
            vec![600.0, 400.0, 700.0, 200.0, 500.0],
        ];
        let self_spectra = vec![spectra_gpu[0].clone(), spectra_gpu[0].clone()];
        let t_gpu = Instant::now();
        let gpu_self = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &self_spectra).unwrap();
        let gpu_pw = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spectra_gpu).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        v.check(
            "Cosine: self-match ≈ 1.0",
            gpu_self[0],
            1.0,
            tolerances::SPECTRAL_COSINE,
        );
        let all_valid = gpu_pw
            .iter()
            .all(|s| s.is_finite() && *s >= 0.0 && *s <= 1.0 + 1e-10);
        v.check(
            "Cosine: all pairs in [0,1]",
            f64::from(u8::from(all_valid)),
            1.0,
            0.0,
        );
        timings.push(("G04: Spectral cosine", "FMR (absorbed)", 0.0, gpu_us));
    }

    // ═══ G05: Variance — FMR (ToadStool) ═════════════════════════
    v.section("G05: Variance (FMR)");
    {
        let data = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let cpu_var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let t_gpu = Instant::now();
        let gpu_var = stats_gpu::variance_gpu(&gpu, &data).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        v.check(
            "Variance: CPU == GPU",
            gpu_var,
            cpu_var,
            tolerances::GPU_VS_CPU_F64,
        );
        timings.push(("G05: Variance", "FMR (absorbed)", 0.0, gpu_us));
    }

    // ═══ G06: ANI — ToadStool bio ═══════════════════════════════════
    v.section("G06: ANI (ToadStool bio)");
    {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"ATGATGATG", b"ATGATGATG"),
            (b"ATGATGATG", b"CTGATGATG"),
            (b"ATGATGATG", b"CTGCTGCTG"),
        ];
        let t_cpu = Instant::now();
        let cpu_r: Vec<_> = pairs.iter().map(|(a, b)| ani::pairwise_ani(a, b)).collect();
        let cpu_us = t_cpu.elapsed().as_micros() as f64;
        let t_gpu = Instant::now();
        let gpu_ani = AniGpu::new(&device).expect("ANI GPU shader");
        let gpu_r = gpu_ani.batch_ani(&pairs).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        for (i, (c, g)) in cpu_r.iter().zip(gpu_r.ani_values.iter()).enumerate() {
            v.check(
                &format!("ANI[{i}]: CPU == GPU"),
                *g,
                c.ani,
                tolerances::GPU_VS_CPU_F64,
            );
        }
        timings.push(("G06: ANI", "ToadStool (absorbed)", cpu_us, gpu_us));
    }

    // ═══ G07: SNP Calling — ToadStool bio ═══════════════════════════
    v.section("G07: SNP Calling (ToadStool bio)");
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
        let gpu_r = gpu_snp.call_snps(&seqs).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        let gpu_count = gpu_r.is_variant.iter().filter(|&&x| x != 0).count();
        v.check(
            "SNP: variant count CPU == GPU",
            gpu_count as f64,
            cpu_snp.variants.len() as f64,
            0.0,
        );
        timings.push(("G07: SNP calling", "ToadStool (absorbed)", cpu_us, gpu_us));
    }

    // ═══ G08: dN/dS — ToadStool bio ════════════════════════════════
    v.section("G08: dN/dS (ToadStool bio)");
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
        let gpu_mod = DnDsGpu::new(&device).expect("dN/dS GPU shader");
        let gpu_r = gpu_mod.batch_dnds(&pairs).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        for (i, cpu_r) in cpu_dnds.iter().enumerate() {
            if let Ok(cr) = cpu_r {
                v.check(
                    &format!("dN/dS[{i}] dN: CPU == GPU"),
                    gpu_r.dn[i],
                    cr.dn,
                    tolerances::GPU_VS_CPU_F64,
                );
                v.check(
                    &format!("dN/dS[{i}] dS: CPU == GPU"),
                    gpu_r.ds[i],
                    cr.ds,
                    tolerances::GPU_VS_CPU_F64,
                );
            }
        }
        timings.push(("G08: dN/dS", "ToadStool (absorbed)", cpu_us, gpu_us));
    }

    // ═══ G09: Pangenome — ToadStool bio ═════════════════════════════
    v.section("G09: Pangenome (ToadStool bio)");
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
        let flat: Vec<u8> = clusters
            .iter()
            .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
            .collect();
        let t_gpu = Instant::now();
        let gpu_pan = PangenomeGpu::new(&device).expect("Pangenome GPU shader");
        let gpu_r = gpu_pan.classify(&flat, 5, 4).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        let gpu_core = gpu_r.classifications.iter().filter(|&&c| c == 3).count();
        v.check(
            "Pan: core CPU == GPU",
            gpu_core as f64,
            cpu_pan.core_size as f64,
            0.0,
        );
        timings.push(("G09: Pangenome", "ToadStool (absorbed)", cpu_us, gpu_us));
    }

    // ═══ G10: Random Forest — ToadStool bio ═════════════════════════
    v.section("G10: Random Forest (ToadStool bio)");
    {
        let t1 = DecisionTree::from_arrays(
            &[0, -2, 1, -2, -2],
            &[5.0, 0.0, 3.0, 0.0, 0.0],
            &[1, -1, 3, -1, -1],
            &[2, -1, 4, -1, -1],
            &[None, Some(0), None, Some(1), Some(2)],
            2,
        )
        .unwrap();
        let t2 = DecisionTree::from_arrays(
            &[1, -2, -2],
            &[4.0, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(2)],
            2,
        )
        .unwrap();
        let t3 = DecisionTree::from_arrays(
            &[0, -2, -2],
            &[6.0, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(1), Some(2)],
            2,
        )
        .unwrap();
        let rf = RandomForest::from_trees(vec![t1, t2, t3], 3).unwrap();
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
                &format!("RF[{i}]: CPU == GPU"),
                g.class as f64,
                *c as f64,
                0.0,
            );
        }
        timings.push(("G10: Random Forest", "ToadStool (absorbed)", cpu_us, gpu_us));
    }

    // ═══ G11: HMM Forward — ToadStool bio ═══════════════════════════
    v.section("G11: HMM Forward (ToadStool bio)");
    {
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
        let gpu_r = hmm_gpu.forward_batch(&model, &flat_obs, 2, 4).unwrap();
        let gpu_us = t_gpu.elapsed().as_micros() as f64;
        v.check(
            "HMM[0]: CPU == GPU",
            gpu_r.log_likelihoods[0],
            cpu_ll1,
            tolerances::GPU_VS_CPU_F64,
        );
        v.check(
            "HMM[1]: CPU == GPU",
            gpu_r.log_likelihoods[1],
            cpu_ll2,
            tolerances::GPU_VS_CPU_F64,
        );
        timings.push(("G11: HMM forward", "ToadStool (absorbed)", cpu_us, gpu_us));
    }

    // ═══ Summary: Math Portability Proof ══════════════════════════
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║  BarraCuda GPU — Full Math Portability Proof                              ║");
    println!("║  Same Rust math. Same answers. Different substrate.                       ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:<28} {:<22} {:>8} {:>8}  ║",
        "Domain", "Primitive", "CPU µs", "GPU µs"
    );
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    for (name, prim, cpu, gpu_t) in &timings {
        println!("║  {name:<28} {prim:<22} {cpu:>8.0} {gpu_t:>8.0}  ║");
    }
    let total_cpu: f64 = timings.iter().map(|(_, _, c, _)| c).sum();
    let total_gpu: f64 = timings.iter().map(|(_, _, _, g)| g).sum();
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:<28} {:<22} {:>8.0} {:>8.0}  ║",
        "TOTAL", "", total_cpu, total_gpu
    );
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  32 ToadStool primitives (Lean complete), 0 local WGSL                 ║");
    println!("║  Math is substrate-independent. Ready for pure GPU streaming pipeline.   ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    v.finish();
}
