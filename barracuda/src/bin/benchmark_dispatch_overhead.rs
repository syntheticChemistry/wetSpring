// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::approx_constant
)]
//! Exp067: `ToadStool` Dispatch Overhead Profiling
//!
//! Measures actual dispatch overhead (buffer upload + shader dispatch +
//! readback) for each GPU domain at minimal input size. This separates
//! "fixed cost of going to GPU" from "variable cost of computation."
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | wgpu/ToadStool dispatch timing |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --features gpu --release --bin benchmark_dispatch_overhead` |
//! | Data | Minimal test vectors |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;

use wetspring_barracuda::bio::{
    ani, ani_gpu::AniGpu, diversity, diversity_gpu, dnds, dnds_gpu::DnDsGpu, hmm,
    hmm_gpu::HmmGpuForward, pangenome, pangenome_gpu::PangenomeGpu,
    random_forest_gpu::RandomForestGpu, snp, snp_gpu::SnpGpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation;

const WARMUP: usize = 3;
const ITERS: usize = 10;

#[tokio::main]
async fn main() {
    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let device = gpu.to_wgpu_device();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp067: ToadStool Dispatch Overhead Profiling                     ║");
    println!("║  GPU: {:<60} ║", gpu.adapter_name);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:<30} {:>12} {:>12} {:>12}",
        "Domain", "CPU (µs)", "GPU (µs)", "Overhead"
    );
    println!("  {}", "─".repeat(68));

    let mut results: Vec<(&str, f64, f64)> = Vec::new();

    // Shannon (minimal: 8 elements)
    {
        let data = vec![10.0, 20.0, 30.0, 15.0, 25.0, 5.0, 12.0, 8.0];
        let cpu = bench(|| {
            let _ = diversity::shannon(&data);
        });
        let gpu_t = bench(|| {
            let _ = diversity_gpu::shannon_gpu(&gpu, &data);
        });
        row("Shannon (N=8)", cpu, gpu_t);
        results.push(("Shannon", cpu, gpu_t));
    }

    // Bray-Curtis (minimal: 3 samples)
    {
        let s = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 25.0, 10.0],
            vec![5.0, 30.0, 25.0],
        ];
        let cpu = bench(|| {
            let _ = diversity::bray_curtis_condensed(&s);
        });
        let gpu_t = bench(|| {
            let _ = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &s);
        });
        row("Bray-Curtis (3×3)", cpu, gpu_t);
        results.push(("Bray-Curtis", cpu, gpu_t));
    }

    // ANI (minimal: 2 pairs)
    {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"ATGATGATG" as &[u8], b"ATGATGATG" as &[u8]),
            (b"ATGATGATG", b"CTGATGATG"),
        ];
        let cpu = bench(|| {
            for (a, b) in &pairs {
                let _ = ani::pairwise_ani(a, b);
            }
        });
        let gpu_ani = AniGpu::new(&device);
        let gpu_t = bench(|| {
            let _ = gpu_ani.batch_ani(&pairs);
        });
        row("ANI (2 pairs)", cpu, gpu_t);
        results.push(("ANI", cpu, gpu_t));
    }

    // SNP (minimal: 3 seqs × 12bp)
    {
        let seqs: Vec<&[u8]> = vec![b"ATGATGATGATG", b"ATCATGATGATG", b"ATGATCATGATG"];
        let cpu = bench(|| {
            let _ = snp::call_snps(&seqs);
        });
        let gpu_snp = SnpGpu::new(&device);
        let gpu_t = bench(|| {
            let _ = gpu_snp.call_snps(&seqs);
        });
        row("SNP (3×12bp)", cpu, gpu_t);
        results.push(("SNP", cpu, gpu_t));
    }

    // dN/dS (minimal: 2 pairs)
    {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b"ATGATGATG" as &[u8], b"ATGATGATG" as &[u8]),
            (b"TTTGCTAAA", b"TTCGCTAAA"),
        ];
        let cpu = bench(|| {
            for (a, b) in &pairs {
                let _ = dnds::pairwise_dnds(a, b);
            }
        });
        let gpu_mod = DnDsGpu::new(&device);
        let gpu_t = bench(|| {
            let _ = gpu_mod.batch_dnds(&pairs);
        });
        row("dN/dS (2 pairs)", cpu, gpu_t);
        results.push(("dN/dS", cpu, gpu_t));
    }

    // Pangenome (minimal: 5 genes × 4 genomes)
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
                presence: vec![true, false, false, false],
            },
            pangenome::GeneCluster {
                id: "g4".into(),
                presence: vec![false, true, false, false],
            },
            pangenome::GeneCluster {
                id: "g5".into(),
                presence: vec![false, false, false, true],
            },
        ];
        let flat: Vec<u8> = clusters
            .iter()
            .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
            .collect();

        let cpu = bench(|| {
            let _ = pangenome::analyze(&clusters, 4);
        });
        let gpu_pan = PangenomeGpu::new(&device);
        let gpu_t = bench(|| {
            let _ = gpu_pan.classify(&flat, 5, 4);
        });
        row("Pangenome (5×4)", cpu, gpu_t);
        results.push(("Pangenome", cpu, gpu_t));
    }

    // RF (minimal: 3 trees, 2 samples)
    {
        use wetspring_barracuda::bio::{decision_tree::DecisionTree, random_forest::RandomForest};

        let t1 = DecisionTree::from_arrays(
            &[0, -2, -2],
            &[5.0, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(1)],
            2,
        )
        .unwrap();
        let t2 = DecisionTree::from_arrays(
            &[1, -2, -2],
            &[3.0, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(1)],
            2,
        )
        .unwrap();
        let t3 = DecisionTree::from_arrays(
            &[0, -2, -2],
            &[6.0, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(0), Some(1)],
            2,
        )
        .unwrap();
        let rf = RandomForest::from_trees(vec![t1, t2, t3], 2).unwrap();
        let samples = vec![vec![3.0, 1.0], vec![7.0, 6.0]];

        let cpu = bench(|| {
            let _ = rf.predict_batch(&samples);
        });
        let rf_gpu = RandomForestGpu::new(&device);
        let gpu_t = bench(|| {
            let _ = rf_gpu.predict_batch(&rf, &samples);
        });
        row("RF (3t×2s)", cpu, gpu_t);
        results.push(("RF", cpu, gpu_t));
    }

    // HMM (minimal: 2 seqs × 4 obs)
    {
        let model = hmm::HmmModel {
            n_states: 2,
            n_symbols: 2,
            log_pi: vec![-std::f64::consts::LN_2, -std::f64::consts::LN_2],
            log_trans: vec![-0.3567, -1.2040, -1.2040, -0.3567],
            log_emit: vec![-0.2231, -1.6094, -1.6094, -0.2231],
        };
        let obs = [0_usize, 1, 0, 1];
        let flat: Vec<u32> = [0_u32, 1, 0, 1, 0, 0, 0, 0].to_vec();

        let cpu = bench(|| {
            let _ = hmm::forward(&model, &obs);
            let _ = hmm::forward(&model, &obs);
        });
        let hmm_gpu = HmmGpuForward::new(&device);
        let gpu_t = bench(|| {
            let _ = hmm_gpu.forward_batch(&model, &flat, 2, 4);
        });
        row("HMM (2×4obs)", cpu, gpu_t);
        results.push(("HMM", cpu, gpu_t));
    }

    println!("  {}", "─".repeat(68));

    let avg_cpu: f64 = results.iter().map(|(_, c, _)| c).sum::<f64>() / results.len() as f64;
    let avg_gpu: f64 = results.iter().map(|(_, _, g)| g).sum::<f64>() / results.len() as f64;
    let avg_overhead = avg_gpu - avg_cpu;
    println!(
        "  {:<30} {:>12.0} {:>12.0} {:>10.0}µs",
        "AVERAGE", avg_cpu, avg_gpu, avg_overhead
    );

    println!();
    println!("  Dispatch overhead = GPU_time - CPU_time at minimal input.");
    println!("  This is the FIXED COST of using the GPU for each domain.");
    println!("  GPU wins when: batch_compute_savings > dispatch_overhead.");
    println!();
    println!("  metalForge routing: if estimated GPU speedup × batch_size");
    println!("  exceeds dispatch overhead, route to GPU. Otherwise CPU.");
}

fn bench<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    start.elapsed().as_secs_f64() * 1_000_000.0 / ITERS as f64
}

fn row(label: &str, cpu_us: f64, gpu_us: f64) {
    let overhead = gpu_us - cpu_us;
    let oh_str = if overhead > 0.0 {
        format!("+{overhead:.0}µs")
    } else {
        format!("{overhead:.0}µs")
    };
    println!("  {label:<30} {cpu_us:>12.0} {gpu_us:>12.0} {oh_str:>12}");
}
