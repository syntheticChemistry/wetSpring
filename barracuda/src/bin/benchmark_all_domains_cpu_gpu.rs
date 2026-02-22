// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::approx_constant,
    clippy::unnecessary_cast
)]
//! Exp066: CPU vs GPU Scaling Benchmark — All GPU Domains
//!
//! Extends the existing CPU vs GPU benchmark to cover ANI, SNP, dN/dS,
//! pangenome, Random Forest, and HMM. Measures wall-clock time at multiple
//! data sizes to characterize the GPU crossover point.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | BarraCuda CPU (timing harness) |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --features gpu --release --bin benchmark_all_domains_cpu_gpu` |
//! | Data | Synthetic sequences at increasing sizes |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;
use wetspring_barracuda::bio::{
    ani, ani_gpu::AniGpu, decision_tree::DecisionTree, dnds, dnds_gpu::DnDsGpu, hmm,
    hmm_gpu::HmmGpuForward, pangenome, pangenome_gpu::PangenomeGpu, random_forest::RandomForest,
    random_forest_gpu::RandomForestGpu, snp, snp_gpu::SnpGpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation;

const WARMUP: usize = 2;
const MIN_ITERS: u64 = 3;

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
    println!("║  Exp066: CPU vs GPU Scaling — All Domains                          ║");
    println!("║  GPU: {:<60} ║", gpu.adapter_name);
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    bench_ani(&device);
    bench_snp(&device);
    bench_dnds(&device);
    bench_pangenome(&device);
    bench_random_forest(&device);
    bench_hmm(&device);

    println!();
    println!("  Benchmark complete. GPU dispatch overhead is ~0.5-2ms.");
    println!("  GPU wins when batch size exceeds the dispatch breakeven.");
}

fn section(title: &str) {
    println!();
    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│ {title:<66} │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ {:<30} {:>8} {:>11} {:>11} {:>6}│",
        "Workload", "N", "CPU", "GPU", "Ratio"
    );
    println!("├────────────────────────────────────────────────────────────────────┤");
}

fn row(label: &str, n: usize, cpu_us: f64, gpu_us: f64) {
    let ratio = if gpu_us > 0.01 { cpu_us / gpu_us } else { 0.0 };
    let arrow = if ratio >= 1.0 { "▲" } else { "▼" };
    println!(
        "│ {label:<30} {n:>8} {:>11} {:>11} {ratio:>4.1}x{arrow}│",
        format_us(cpu_us),
        format_us(gpu_us),
    );
}

fn format_us(us: f64) -> String {
    if us < 1.0 {
        format!("{:.0}ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{us:.1}µs")
    } else {
        format!("{:.2}ms", us / 1000.0)
    }
}

fn time_fn<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut iters = MIN_ITERS;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = start.elapsed();
        if elapsed.as_micros() > 5_000 || iters >= 500 {
            return elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
        }
        iters = (iters * 3).min(500);
    }
}

fn gen_seq(len: usize, seed: u64) -> Vec<u8> {
    let bases = [b'A', b'T', b'G', b'C'];
    let mut rng = seed;
    (0..len)
        .map(|_| {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bases[((rng >> 33) % 4) as usize]
        })
        .collect()
}

fn gen_codon_seq(n_codons: usize, seed: u64) -> Vec<u8> {
    gen_seq(n_codons * 3, seed)
}

// ── ANI ──────────────────────────────────────────────────────────────

fn bench_ani(device: &Arc<WgpuDevice>) {
    section("ANI PAIRWISE IDENTITY (GPU batch vs CPU sequential)");
    for &n_pairs in &[4, 16, 64, 256] {
        let seq_len = 100;
        let seqs: Vec<Vec<u8>> = (0..n_pairs * 2)
            .map(|i| gen_seq(seq_len, 42 + i as u64))
            .collect();
        let pairs: Vec<(&[u8], &[u8])> = (0..n_pairs)
            .map(|i| (seqs[i * 2].as_slice(), seqs[i * 2 + 1].as_slice()))
            .collect();

        let cpu = time_fn(|| {
            for (a, b) in &pairs {
                let _ = ani::pairwise_ani(a, b);
            }
        });
        let gpu_ani = AniGpu::new(device).expect("ANI GPU shader");
        let gpu = time_fn(|| {
            let _ = gpu_ani.batch_ani(&pairs);
        });
        row("ANI", n_pairs, cpu, gpu);
    }
}

// ── SNP ──────────────────────────────────────────────────────────────

fn bench_snp(device: &Arc<WgpuDevice>) {
    section("SNP CALLING (GPU parallel-position vs CPU sequential)");
    for &(n_seqs, seq_len) in &[(4, 100), (8, 200), (16, 500), (32, 1000)] {
        let seqs: Vec<Vec<u8>> = (0..n_seqs)
            .map(|i| gen_seq(seq_len, 100 + i as u64))
            .collect();
        let refs: Vec<&[u8]> = seqs.iter().map(Vec::as_slice).collect();

        let cpu = time_fn(|| {
            let _ = snp::call_snps(&refs);
        });
        let gpu_snp = SnpGpu::new(device).expect("SNP GPU shader");
        let gpu = time_fn(|| {
            let _ = gpu_snp.call_snps(&refs);
        });
        row(
            &format!("SNP {n_seqs}×{seq_len}bp"),
            n_seqs * seq_len,
            cpu,
            gpu,
        );
    }
}

// ── dN/dS ────────────────────────────────────────────────────────────

fn bench_dnds(device: &Arc<WgpuDevice>) {
    section("dN/dS NEI-GOJOBORI (GPU batch codon vs CPU sequential)");
    for &n_pairs in &[4, 16, 64, 256] {
        let n_codons = 50;
        let seqs: Vec<Vec<u8>> = (0..n_pairs * 2)
            .map(|i| gen_codon_seq(n_codons, 200 + i as u64))
            .collect();
        let pairs: Vec<(&[u8], &[u8])> = (0..n_pairs)
            .map(|i| (seqs[i * 2].as_slice(), seqs[i * 2 + 1].as_slice()))
            .collect();

        let cpu = time_fn(|| {
            for (a, b) in &pairs {
                let _ = dnds::pairwise_dnds(a, b);
            }
        });
        let gpu_mod = DnDsGpu::new(device).expect("dN/dS GPU shader");
        let gpu = time_fn(|| {
            let _ = gpu_mod.batch_dnds(&pairs);
        });
        row("dN/dS", n_pairs, cpu, gpu);
    }
}

// ── Pangenome ────────────────────────────────────────────────────────

fn bench_pangenome(device: &Arc<WgpuDevice>) {
    section("PANGENOME CLASSIFY (GPU reduce vs CPU scan)");
    for &(n_genes, n_genomes) in &[(10, 5), (50, 10), (200, 20), (500, 50)] {
        let mut rng = 300_u64;
        let clusters: Vec<pangenome::GeneCluster> = (0..n_genes)
            .map(|g| {
                let presence = (0..n_genomes)
                    .map(|_| {
                        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        (rng >> 33) % 3 != 0
                    })
                    .collect();
                pangenome::GeneCluster {
                    id: format!("g{g}"),
                    presence,
                }
            })
            .collect();

        let flat: Vec<u8> = clusters
            .iter()
            .flat_map(|c| c.presence.iter().map(|&p| u8::from(p)))
            .collect();

        let cpu = time_fn(|| {
            let _ = pangenome::analyze(&clusters, n_genomes);
        });
        let gpu_pan = PangenomeGpu::new(device).expect("Pangenome GPU shader");
        let gpu = time_fn(|| {
            let _ = gpu_pan.classify(&flat, n_genes, n_genomes);
        });
        row(
            &format!("Pan {n_genes}g×{n_genomes}G"),
            n_genes * n_genomes,
            cpu,
            gpu,
        );
    }
}

// ── Random Forest ────────────────────────────────────────────────────

fn bench_random_forest(device: &Arc<WgpuDevice>) {
    section("RANDOM FOREST (GPU SoA batch vs CPU sequential)");

    let mk_stump = |feat: i32, thresh: f64, left_class: usize, right_class: usize| {
        DecisionTree::from_arrays(
            &[feat, -2, -2],
            &[thresh, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[None, Some(left_class), Some(right_class)],
            2,
        )
        .unwrap()
    };

    let trees: Vec<DecisionTree> = (0..10)
        .map(|i| mk_stump((i % 2) as i32, f64::from(i).mul_add(0.3, 5.0), 0, 1))
        .collect();
    let rf = RandomForest::from_trees(trees, 2).unwrap();

    for &n_samples in &[10, 50, 200, 1000] {
        let mut rng = 400_u64;
        let samples: Vec<Vec<f64>> = (0..n_samples)
            .map(|_| {
                (0..2)
                    .map(|_| {
                        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        ((rng >> 33) % 100) as f64 / 10.0
                    })
                    .collect()
            })
            .collect();

        let cpu = time_fn(|| {
            let _ = rf.predict_batch(&samples);
        });
        let rf_gpu = RandomForestGpu::new(device);
        let gpu = time_fn(|| {
            let _ = rf_gpu.predict_batch(&rf, &samples);
        });
        row(&format!("RF 10t×{n_samples}s"), n_samples, cpu, gpu);
    }
}

// ── HMM Forward ──────────────────────────────────────────────────────

fn bench_hmm(device: &Arc<WgpuDevice>) {
    section("HMM FORWARD LOG-SPACE (GPU batch vs CPU sequential)");

    let model = hmm::HmmModel {
        n_states: 3,
        n_symbols: 4,
        log_pi: vec![-1.0986, -1.0986, -1.0986],
        log_trans: vec![
            -0.3567, -1.2040, -2.3026, -1.2040, -0.3567, -1.2040, -2.3026, -1.2040, -0.3567,
        ],
        log_emit: vec![
            -0.2231, -1.6094, -2.3026, -2.3026, -2.3026, -0.2231, -1.6094, -2.3026, -2.3026,
            -2.3026, -0.2231, -1.6094,
        ],
    };

    for &(n_seqs, n_steps) in &[(4, 20), (16, 50), (64, 100), (256, 200)] {
        let mut rng = 500_u64;
        let obs_cpu: Vec<Vec<usize>> = (0..n_seqs)
            .map(|_| {
                (0..n_steps)
                    .map(|_| {
                        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        ((rng >> 33) % 4) as usize
                    })
                    .collect()
            })
            .collect();

        let flat_obs: Vec<u32> = obs_cpu
            .iter()
            .flat_map(|seq| seq.iter().map(|&x| x as u32))
            .collect();

        let cpu = time_fn(|| {
            for seq in &obs_cpu {
                let _ = hmm::forward(&model, seq);
            }
        });
        let hmm_gpu = HmmGpuForward::new(device).expect("HMM GPU shader");
        let gpu = time_fn(|| {
            let _ = hmm_gpu.forward_batch(&model, &flat_obs, n_seqs, n_steps);
        });
        row(
            &format!("HMM {n_seqs}×{n_steps}obs"),
            n_seqs * n_steps,
            cpu,
            gpu,
        );
    }
}
