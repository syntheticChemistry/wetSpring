// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
//! Exp095: Cross-Spring Scaling Benchmark
//!
//! # Provenance
//!
//! | Script  | `benchmark_cross_spring_scaling` |
//! | Command | `cargo run --release --features gpu --bin benchmark_cross_spring_scaling` |
//! | Hardware| RTX 4070 (Ada), Titan V (Volta GV100, NVK) |
//!
//! # Purpose
//!
//! Benchmarks cross-spring evolved primitives at realistic bioinformatics
//! problem sizes. Shows GPU scaling advantage and traces each shader's origin
//! through the ecoPrimals biome.
//!
//! ## Provenance
//!
//! | Primitive | Evolved By | Absorbed | Consumed By |
//! |-----------|-----------|----------|-------------|
//! | `PairwiseHammingGpu` | neuralSpring | Session 31f | wetSpring |
//! | `PairwiseJaccardGpu` | neuralSpring | Session 31f | wetSpring |
//! | `SpatialPayoffGpu` | neuralSpring | Session 31f | wetSpring |
//! | `BatchFitnessGpu` | neuralSpring | Session 31f | wetSpring |
//! | `LocusVarianceGpu` | neuralSpring | Session 31f | wetSpring |
//! | `FusedMapReduceF64` | hotSpring | Session 18 | wetSpring, neuralSpring |
//! | `GemmF64` | wetSpring (60×) | Session 18 | hotSpring HFB |

use barracuda::ops::bio::batch_fitness::BatchFitnessGpu;
use barracuda::ops::bio::locus_variance::LocusVarianceGpu;
use barracuda::ops::bio::pairwise_hamming::PairwiseHammingGpu;
use barracuda::ops::bio::pairwise_jaccard::PairwiseJaccardGpu;
use barracuda::ops::bio::spatial_payoff::SpatialPayoffGpu;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::linalg::gemm_f64::GemmF64;
use std::time::Instant;
use wgpu::util::DeviceExt;

use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation;

struct BenchResult {
    primitive: &'static str,
    evolved_by: &'static str,
    problem_size: String,
    cpu_us: f64,
    gpu_us: f64,
    speedup: f64,
}

#[tokio::main]
async fn main() {
    println!("════════════════════════════════════════════════════════════════════");
    println!("  Exp095: Cross-Spring Scaling Benchmark");
    println!("  GPU advantage at realistic bioinformatics sizes");
    println!("════════════════════════════════════════════════════════════════════\n");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    let device = gpu.to_wgpu_device();
    let d = device.device();

    let mut results: Vec<BenchResult> = Vec::new();

    // ═══ neuralSpring-evolved primitives (buffer-based) ═════════════════

    // PairwiseHamming: 500 sequences × 1000 bases
    println!("\n── PairwiseHamming: 500 seqs × 1000 bp (neuralSpring) ──");
    {
        let n_seqs: u32 = 500;
        let seq_len: u32 = 1000;
        let sequences: Vec<u32> = (0..(n_seqs * seq_len) as usize)
            .map(|i| (i % 4) as u32)
            .collect();

        let tc = Instant::now();
        cpu_pairwise_hamming(&sequences, n_seqs as usize, seq_len as usize);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let seq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&sequences),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let n_pairs = n_seqs * (n_seqs - 1) / 2;
        let dist_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_pairs as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hamming = PairwiseHammingGpu::new(device.clone());
        hamming.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);
        d.poll(wgpu::Maintain::Wait);

        let tg = Instant::now();
        hamming.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);
        d.poll(wgpu::Maintain::Wait);
        let gpu_us = tg.elapsed().as_micros() as f64;

        let speedup = if gpu_us > 0.0 { cpu_us / gpu_us } else { 0.0 };
        println!("  CPU: {cpu_us:.0} µs | GPU: {gpu_us:.0} µs | Speedup: {speedup:.1}×");
        results.push(BenchResult {
            primitive: "PairwiseHamming",
            evolved_by: "neuralSpring",
            problem_size: format!("500×1000 ({n_pairs} pairs)"),
            cpu_us,
            gpu_us,
            speedup,
        });
    }

    // PairwiseJaccard: 200 genomes × 2000 genes
    println!("\n── PairwiseJaccard: 200 genomes × 2000 genes (neuralSpring) ──");
    {
        let n_genomes: u32 = 200;
        let n_genes: u32 = 2000;
        let pa: Vec<f32> = (0..(n_genes * n_genomes) as usize)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();

        let tc = Instant::now();
        cpu_pairwise_jaccard(&pa, n_genomes as usize, n_genes as usize);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let pa_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pa),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let n_pairs = n_genomes * (n_genomes - 1) / 2;
        let dist_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_pairs as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let jaccard = PairwiseJaccardGpu::new(device.clone());
        jaccard.dispatch(&pa_buf, &dist_buf, n_genomes, n_genes);
        d.poll(wgpu::Maintain::Wait);

        let tg = Instant::now();
        jaccard.dispatch(&pa_buf, &dist_buf, n_genomes, n_genes);
        d.poll(wgpu::Maintain::Wait);
        let gpu_us = tg.elapsed().as_micros() as f64;

        let speedup = if gpu_us > 0.0 { cpu_us / gpu_us } else { 0.0 };
        println!("  CPU: {cpu_us:.0} µs | GPU: {gpu_us:.0} µs | Speedup: {speedup:.1}×");
        results.push(BenchResult {
            primitive: "PairwiseJaccard",
            evolved_by: "neuralSpring",
            problem_size: format!("200×2000 ({n_pairs} pairs)"),
            cpu_us,
            gpu_us,
            speedup,
        });
    }

    // SpatialPayoff: 256×256 grid
    println!("\n── SpatialPayoff: 256×256 grid (neuralSpring) ──");
    {
        let grid_size: u32 = 256;
        let benefit: f32 = 3.0;
        let cost: f32 = 1.0;
        let n = (grid_size * grid_size) as usize;
        let grid: Vec<u32> = (0..n).map(|i| (i % 3 == 0) as u32).collect();

        let tc = Instant::now();
        cpu_spatial_payoff(&grid, grid_size as usize, benefit, cost);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let grid_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fit_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spatial = SpatialPayoffGpu::new(device.clone());
        spatial.dispatch(&grid_buf, &fit_buf, grid_size, benefit, cost);
        d.poll(wgpu::Maintain::Wait);

        let tg = Instant::now();
        spatial.dispatch(&grid_buf, &fit_buf, grid_size, benefit, cost);
        d.poll(wgpu::Maintain::Wait);
        let gpu_us = tg.elapsed().as_micros() as f64;

        let speedup = if gpu_us > 0.0 { cpu_us / gpu_us } else { 0.0 };
        println!("  CPU: {cpu_us:.0} µs | GPU: {gpu_us:.0} µs | Speedup: {speedup:.1}×");
        results.push(BenchResult {
            primitive: "SpatialPayoff",
            evolved_by: "neuralSpring",
            problem_size: "256×256 (65K cells)".to_string(),
            cpu_us,
            gpu_us,
            speedup,
        });
    }

    // BatchFitness: 4096 pop × 256 genome
    println!("\n── BatchFitness: 4096 × 256 genome (neuralSpring) ──");
    {
        let pop_size: u32 = 4096;
        let genome_len: u32 = 256;
        let population: Vec<f32> = (0..(pop_size * genome_len) as usize)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let weights: Vec<f32> = (0..genome_len)
            .map(|i| (i as f32 + 1.0) / genome_len as f32)
            .collect();

        let tc = Instant::now();
        cpu_batch_fitness(
            &population,
            &weights,
            pop_size as usize,
            genome_len as usize,
        );
        let cpu_us = tc.elapsed().as_micros() as f64;

        let pop_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&population),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fit_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (pop_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bf = BatchFitnessGpu::new(device.clone());
        bf.dispatch(&pop_buf, &w_buf, &fit_buf, pop_size, genome_len);
        d.poll(wgpu::Maintain::Wait);

        let tg = Instant::now();
        bf.dispatch(&pop_buf, &w_buf, &fit_buf, pop_size, genome_len);
        d.poll(wgpu::Maintain::Wait);
        let gpu_us = tg.elapsed().as_micros() as f64;

        let speedup = if gpu_us > 0.0 { cpu_us / gpu_us } else { 0.0 };
        println!("  CPU: {cpu_us:.0} µs | GPU: {gpu_us:.0} µs | Speedup: {speedup:.1}×");
        results.push(BenchResult {
            primitive: "BatchFitness",
            evolved_by: "neuralSpring",
            problem_size: "4096×256 (1M elems)".to_string(),
            cpu_us,
            gpu_us,
            speedup,
        });
    }

    // LocusVariance: 100 pops × 10,000 loci
    println!("\n── LocusVariance: 100 pops × 10K loci (neuralSpring) ──");
    {
        let n_pops: u32 = 100;
        let n_loci: u32 = 10_000;
        let freqs: Vec<f32> = (0..(n_pops * n_loci) as usize)
            .map(|i| (i as f32 * 0.001) % 1.0)
            .collect();

        let tc = Instant::now();
        cpu_locus_variance_rowmajor(&freqs, n_pops as usize, n_loci as usize);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let freq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&freqs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let var_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n_loci * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let lv = LocusVarianceGpu::new(device.clone());
        lv.dispatch(&freq_buf, &var_buf, n_pops, n_loci);
        d.poll(wgpu::Maintain::Wait);

        let tg = Instant::now();
        lv.dispatch(&freq_buf, &var_buf, n_pops, n_loci);
        d.poll(wgpu::Maintain::Wait);
        let gpu_us = tg.elapsed().as_micros() as f64;

        let speedup = if gpu_us > 0.0 { cpu_us / gpu_us } else { 0.0 };
        println!("  CPU: {cpu_us:.0} µs | GPU: {gpu_us:.0} µs | Speedup: {speedup:.1}×");
        results.push(BenchResult {
            primitive: "LocusVariance",
            evolved_by: "neuralSpring",
            problem_size: "100×10K (1M elems)".to_string(),
            cpu_us,
            gpu_us,
            speedup,
        });
    }

    // ═══ hotSpring-evolved: FusedMapReduceF64 ═══════════════════════════

    println!("\n── FusedMapReduceF64 (Shannon): 100K f64 (hotSpring) ──");
    {
        let n: usize = 100_000;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();

        let tc = Instant::now();
        let total: f64 = data.iter().sum();
        let mut _cpu_h = 0.0f64;
        for &x in &data {
            let p = x / total;
            if p > 0.0 {
                _cpu_h -= p * p.ln();
            }
        }
        let cpu_us = tc.elapsed().as_micros() as f64;

        let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let fmr = FusedMapReduceF64::new(device.clone()).expect("FMR");
            fmr.shannon_entropy(&data).expect("shannon warm");

            let tg = Instant::now();
            let _result = fmr.shannon_entropy(&data).expect("shannon bench");
            tg.elapsed().as_micros() as f64
        }));

        match gpu_result {
            Ok(gpu_us) => {
                let speedup = if gpu_us > 0.0 { cpu_us / gpu_us } else { 0.0 };
                println!("  CPU: {cpu_us:.0} µs | GPU: {gpu_us:.0} µs | Speedup: {speedup:.1}×");
                results.push(BenchResult {
                    primitive: "FusedMapReduce(Shannon)",
                    evolved_by: "hotSpring",
                    problem_size: "100K f64".to_string(),
                    cpu_us,
                    gpu_us,
                    speedup,
                });
            }
            Err(_) => {
                println!("  SKIP: driver issue");
                results.push(BenchResult {
                    primitive: "FusedMapReduce(Shannon)",
                    evolved_by: "hotSpring",
                    problem_size: "100K f64".to_string(),
                    cpu_us,
                    gpu_us: 0.0,
                    speedup: 0.0,
                });
            }
        }
    }

    // ═══ wetSpring-evolved: GemmF64 ═════════════════════════════════════

    println!("\n── GemmF64: 256×256 matmul (wetSpring → hotSpring HFB) ──");
    {
        let n: usize = 256;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
        let b: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.001).collect();

        let tc = Instant::now();
        cpu_matmul(&a, &b, n);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GemmF64::execute(device.clone(), &a, &b, n, n, n, 1).expect("gemm warm");

            let tg = Instant::now();
            let _c = GemmF64::execute(device.clone(), &a, &b, n, n, n, 1).expect("gemm bench");
            tg.elapsed().as_micros() as f64
        }));

        match gpu_result {
            Ok(gpu_us) => {
                let speedup = if gpu_us > 0.0 { cpu_us / gpu_us } else { 0.0 };
                println!("  CPU: {cpu_us:.0} µs | GPU: {gpu_us:.0} µs | Speedup: {speedup:.1}×");
                results.push(BenchResult {
                    primitive: "GemmF64 (256×256)",
                    evolved_by: "wetSpring",
                    problem_size: "256×256 f64 matmul".to_string(),
                    cpu_us,
                    gpu_us,
                    speedup,
                });
            }
            Err(_) => {
                println!("  SKIP: driver issue");
                results.push(BenchResult {
                    primitive: "GemmF64 (256×256)",
                    evolved_by: "wetSpring",
                    problem_size: "256×256 f64".to_string(),
                    cpu_us,
                    gpu_us: 0.0,
                    speedup: 0.0,
                });
            }
        }
    }

    // ═══ Summary Table ═══════════════════════════════════════════════════
    println!("\n═══ Cross-Spring Scaling Results ════════════════════════════════\n");
    let hdr_sp = "Speedup";
    println!(
        "  {prim:25} {evol:15} {size:25} {cpu:>10} {gpu:>10} {hdr_sp:>8}",
        prim = "Primitive",
        evol = "Evolved By",
        size = "Problem Size",
        cpu = "CPU (µs)",
        gpu = "GPU (µs)",
    );
    let sep = "─";
    println!(
        "  {a:25} {b:15} {c:25} {d:>10} {e:>10} {f:>8}",
        a = sep.repeat(25),
        b = sep.repeat(15),
        c = sep.repeat(25),
        d = sep.repeat(10),
        e = sep.repeat(10),
        f = sep.repeat(8),
    );
    for r in &results {
        let sp = if r.speedup > 0.0 {
            format!("{:.1}×", r.speedup)
        } else {
            "SKIP".to_string()
        };
        println!(
            "  {prim:25} {evol:15} {size:25} {cpu:>10.0} {gpu:>10.0} {sp:>8}",
            prim = r.primitive,
            evol = r.evolved_by,
            size = r.problem_size,
            cpu = r.cpu_us,
            gpu = r.gpu_us,
        );
    }

    println!("\n═══ Cross-Spring Evolution = Biome Acceleration ═══════════════");
    println!("  hotSpring  → precision shaders, driver workarounds, eigensolvers");
    println!("  wetSpring  → bio shaders (12), GEMM 60× (powers HFB), f64 fixes");
    println!("  neuralSpring → distance metrics, fitness eval, locus variance");
    println!("  ToadStool  → absorbs all → every Spring benefits\n");
}

fn cpu_pairwise_hamming(seqs: &[u32], n_seqs: usize, seq_len: usize) -> Vec<f32> {
    let mut dists = Vec::with_capacity(n_seqs * (n_seqs - 1) / 2);
    for i in 0..n_seqs {
        for j in (i + 1)..n_seqs {
            let mut diff = 0u32;
            for k in 0..seq_len {
                if seqs[i * seq_len + k] != seqs[j * seq_len + k] {
                    diff += 1;
                }
            }
            dists.push(diff as f32 / seq_len as f32);
        }
    }
    dists
}

fn cpu_pairwise_jaccard(pa: &[f32], n_genomes: usize, n_genes: usize) -> Vec<f32> {
    let mut dists = Vec::with_capacity(n_genomes * (n_genomes - 1) / 2);
    for i in 0..n_genomes {
        for j in (i + 1)..n_genomes {
            let mut intersection = 0.0f32;
            let mut union_count = 0.0f32;
            for g in 0..n_genes {
                let a = pa[g * n_genomes + i];
                let b = pa[g * n_genomes + j];
                if a > 0.5 || b > 0.5 {
                    union_count += 1.0;
                }
                if a > 0.5 && b > 0.5 {
                    intersection += 1.0;
                }
            }
            let jaccard = if union_count > 0.0 {
                intersection / union_count
            } else {
                1.0
            };
            dists.push(1.0 - jaccard);
        }
    }
    dists
}

fn cpu_spatial_payoff(grid: &[u32], size: usize, benefit: f32, cost: f32) -> Vec<f32> {
    let mut fitness = vec![0.0f32; size * size];
    for r in 0..size {
        for c in 0..size {
            let me = grid[r * size + c];
            let mut payoff = 0.0f32;
            for dr in [-1i32, 0, 1] {
                for dc in [-1i32, 0, 1] {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = ((r as i32 + dr).rem_euclid(size as i32)) as usize;
                    let nc = ((c as i32 + dc).rem_euclid(size as i32)) as usize;
                    let neighbor = grid[nr * size + nc];
                    if me == 1 && neighbor == 1 {
                        payoff += benefit - cost;
                    } else if me == 1 && neighbor == 0 {
                        payoff -= cost;
                    } else if me == 0 && neighbor == 1 {
                        payoff += benefit;
                    }
                }
            }
            fitness[r * size + c] = payoff;
        }
    }
    fitness
}

fn cpu_batch_fitness(pop: &[f32], weights: &[f32], pop_size: usize, genome_len: usize) -> Vec<f32> {
    (0..pop_size)
        .map(|i| {
            (0..genome_len)
                .map(|g| pop[i * genome_len + g] * weights[g])
                .sum()
        })
        .collect()
}

fn cpu_locus_variance_rowmajor(freqs: &[f32], n_pops: usize, n_loci: usize) -> Vec<f32> {
    (0..n_loci)
        .map(|l| {
            let mean: f32 = (0..n_pops).map(|p| freqs[p * n_loci + l]).sum::<f32>() / n_pops as f32;
            (0..n_pops)
                .map(|p| {
                    let diff = freqs[p * n_loci + l] - mean;
                    diff * diff
                })
                .sum::<f32>()
                / n_pops as f32
        })
        .collect()
}

fn cpu_matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}
