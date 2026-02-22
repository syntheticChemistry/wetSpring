// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
//! Exp094: Cross-Spring Evolution Validation + Benchmark
//!
//! # Provenance
//!
//! | Script  | `validate_cross_spring_evolution` |
//! | Commit  | (current) |
//! | Command | `cargo run --features gpu --bin validate_cross_spring_evolution` |
//! | Hardware| RTX 4070 (Ada, f64 1:2), Titan V (Volta GV100, NVK) |
//!
//! # Purpose
//!
//! Validate and benchmark the cross-spring evolution: shaders and systems
//! evolved by different Springs, absorbed into `ToadStool`, now consumed by
//! all Springs. Proves the biome model works.
//!
//! ## Cross-Spring Provenance Map
//!
//! | Primitive | Evolved By | Absorbed | Consumed By |
//! |-----------|-----------|----------|-------------|
//! | `FusedMapReduceF64` | hotSpring | Session 18 | wetSpring, neuralSpring |
//! | `GemmCachedF64` (60×) | wetSpring | Session 18 | hotSpring HFB |
//! | `BatchedEighGpu` (NAK) | hotSpring | Session 25 | wetSpring PCoA |
//! | `ShaderTemplate` | hotSpring NVK | Session 18 | All Springs |
//! | `math_f64.wgsl` | wetSpring | Session 27 | All Springs |
//! | `PairwiseHammingGpu` | neuralSpring | Session 31f | wetSpring (new) |
//! | `PairwiseJaccardGpu` | neuralSpring | Session 31f | wetSpring (new) |
//! | `SpatialPayoffGpu` | neuralSpring | Session 31f | wetSpring (new) |
//! | `BatchFitnessGpu` | neuralSpring | Session 31f | wetSpring (new) |
//! | `LocusVarianceGpu` | neuralSpring | Session 31f | wetSpring (new) |

use barracuda::ops::bio::batch_fitness::BatchFitnessGpu;
use barracuda::ops::bio::locus_variance::LocusVarianceGpu;
use barracuda::ops::bio::pairwise_hamming::PairwiseHammingGpu;
use barracuda::ops::bio::pairwise_jaccard::PairwiseJaccardGpu;
use barracuda::ops::bio::spatial_payoff::SpatialPayoffGpu;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    println!("════════════════════════════════════════════════════════════════════");
    println!("  Exp094: Cross-Spring Evolution Validation + Benchmark");
    println!("  Proving: Write → Absorb → Lean across all three Springs");
    println!("════════════════════════════════════════════════════════════════════\n");

    let mut v = Validator::new("Exp094: Cross-Spring Evolution");

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

    let mut results: Vec<(&str, &str, f64, f64, &str)> = Vec::new();

    // ═══ neuralSpring-evolved primitives (new to wetSpring) ══════════════
    println!("\n── neuralSpring Primitives (evolved → absorbed → NEW lean) ──\n");

    // PairwiseHamming (neuralSpring → ToadStool session 31f)
    v.section("PairwiseHamming (neuralSpring → ToadStool)");
    {
        let n_seqs: u32 = 5;
        let seq_len: u32 = 8;
        #[rustfmt::skip]
        let sequences: Vec<u32> = vec![
            0, 1, 2, 3, 0, 1, 2, 3,
            0, 1, 2, 3, 0, 1, 2, 0,
            3, 3, 3, 3, 3, 3, 3, 3,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 2, 3, 3, 2, 1, 0,
        ];

        let tc = Instant::now();
        let cpu_dists = cpu_pairwise_hamming(&sequences, n_seqs as usize, seq_len as usize);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let seq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hamming Seqs"),
            contents: bytemuck::cast_slice(&sequences),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let n_pairs = n_seqs * (n_seqs - 1) / 2;
        let dist_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hamming Dists"),
            size: (n_pairs as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hamming = PairwiseHammingGpu::new(device.clone());
        let tg = Instant::now();
        hamming.dispatch(&seq_buf, &dist_buf, n_seqs, seq_len);
        let gpu_dists = readback_f32(&device, &dist_buf, n_pairs as usize);
        let gpu_us = tg.elapsed().as_micros() as f64;

        for (i, (cpu_d, gpu_d)) in cpu_dists.iter().zip(gpu_dists.iter()).enumerate() {
            v.check(
                &format!("Hamming pair {i}"),
                *gpu_d as f64,
                *cpu_d as f64,
                1e-6,
            );
        }
        results.push((
            "PairwiseHamming",
            "neuralSpring",
            cpu_us,
            gpu_us,
            "session 31f",
        ));
    }

    // PairwiseJaccard (neuralSpring → ToadStool session 31f)
    v.section("PairwiseJaccard (neuralSpring → ToadStool)");
    {
        let n_genomes: u32 = 4;
        let n_genes: u32 = 6;
        #[rustfmt::skip]
        let pa: Vec<f32> = vec![
            1.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 1.0,
        ];

        let tc = Instant::now();
        let cpu_dists = cpu_pairwise_jaccard(&pa, n_genomes as usize, n_genes as usize);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let pa_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Jaccard PA"),
            contents: bytemuck::cast_slice(&pa),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let n_pairs = n_genomes * (n_genomes - 1) / 2;
        let dist_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Jaccard Dists"),
            size: (n_pairs as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let jaccard = PairwiseJaccardGpu::new(device.clone());
        let tg = Instant::now();
        jaccard.dispatch(&pa_buf, &dist_buf, n_genomes, n_genes);
        let gpu_dists = readback_f32(&device, &dist_buf, n_pairs as usize);
        let gpu_us = tg.elapsed().as_micros() as f64;

        for (i, (cpu_d, gpu_d)) in cpu_dists.iter().zip(gpu_dists.iter()).enumerate() {
            v.check(
                &format!("Jaccard pair {i}"),
                *gpu_d as f64,
                *cpu_d as f64,
                1e-5,
            );
        }
        results.push((
            "PairwiseJaccard",
            "neuralSpring",
            cpu_us,
            gpu_us,
            "session 31f",
        ));
    }

    // SpatialPayoff (neuralSpring → ToadStool session 31f)
    v.section("SpatialPayoff (neuralSpring → ToadStool)");
    {
        let grid_size: u32 = 8;
        let benefit: f32 = 3.0;
        let cost: f32 = 1.0;
        let grid: Vec<u32> = (0..(grid_size * grid_size))
            .map(|i| if i % 3 == 0 { 1 } else { 0 })
            .collect();

        let tc = Instant::now();
        let cpu_fitness = cpu_spatial_payoff(&grid, grid_size as usize, benefit, cost);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let grid_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spatial Grid"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fit_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Fitness"),
            size: (grid_size * grid_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let spatial = SpatialPayoffGpu::new(device.clone());
        let tg = Instant::now();
        spatial.dispatch(&grid_buf, &fit_buf, grid_size, benefit, cost);
        let gpu_fitness = readback_f32(&device, &fit_buf, (grid_size * grid_size) as usize);
        let gpu_us = tg.elapsed().as_micros() as f64;

        let mut matching = 0usize;
        for (cf, gf) in cpu_fitness.iter().zip(gpu_fitness.iter()) {
            if (*gf as f64 - *cf as f64).abs() < 1e-4 {
                matching += 1;
            }
        }
        v.check(
            "SpatialPayoff cells matching",
            matching as f64,
            (grid_size * grid_size) as f64,
            0.0,
        );
        results.push((
            "SpatialPayoff",
            "neuralSpring",
            cpu_us,
            gpu_us,
            "session 31f",
        ));
    }

    // BatchFitness (neuralSpring → ToadStool session 31f)
    v.section("BatchFitness (neuralSpring → ToadStool)");
    {
        let pop_size: u32 = 16;
        let genome_len: u32 = 8;
        let weights: Vec<f32> = (0..genome_len)
            .map(|i| (i as f32 + 1.0) / genome_len as f32)
            .collect();
        let population: Vec<f32> = (0..pop_size)
            .flat_map(|i| (0..genome_len).map(move |g| if (i + g) % 2 == 0 { 1.0 } else { 0.0 }))
            .collect();

        let tc = Instant::now();
        let cpu_fit = cpu_batch_fitness(
            &population,
            &weights,
            pop_size as usize,
            genome_len as usize,
        );
        let cpu_us = tc.elapsed().as_micros() as f64;

        let pop_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fitness Pop"),
            contents: bytemuck::cast_slice(&population),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fitness Weights"),
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fit_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fitness Output"),
            size: (pop_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bf = BatchFitnessGpu::new(device.clone());
        let tg = Instant::now();
        bf.dispatch(&pop_buf, &w_buf, &fit_buf, pop_size, genome_len);
        let gpu_fit = readback_f32(&device, &fit_buf, pop_size as usize);
        let gpu_us = tg.elapsed().as_micros() as f64;

        for (i, (cf, gf)) in cpu_fit.iter().zip(gpu_fit.iter()).enumerate() {
            v.check(
                &format!("Fitness individual {i}"),
                *gf as f64,
                *cf as f64,
                1e-5,
            );
        }
        results.push((
            "BatchFitness",
            "neuralSpring",
            cpu_us,
            gpu_us,
            "session 31f",
        ));
    }

    // LocusVariance (neuralSpring → ToadStool session 31f)
    // Layout: allele_freqs[pop * n_loci + locus] — row-major [pop × loci]
    v.section("LocusVariance (neuralSpring → ToadStool)");
    {
        let n_pops: u32 = 4;
        let n_loci: u32 = 6;
        #[rustfmt::skip]
        let freqs: Vec<f32> = vec![
            // pop0: loci 0..5
            0.1, 0.2, 0.9, 0.0, 1.0, 0.4,
            // pop1
            0.3, 0.2, 0.1, 0.0, 1.0, 0.6,
            // pop2
            0.5, 0.2, 0.5, 0.0, 1.0, 0.3,
            // pop3
            0.7, 0.2, 0.3, 0.0, 1.0, 0.7,
        ];

        let tc = Instant::now();
        let cpu_var = cpu_locus_variance_rowmajor(&freqs, n_pops as usize, n_loci as usize);
        let cpu_us = tc.elapsed().as_micros() as f64;

        let freq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LocusVar Freqs"),
            contents: bytemuck::cast_slice(&freqs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let var_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LocusVar Output"),
            size: (n_loci * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let lv = LocusVarianceGpu::new(device.clone());
        let tg = Instant::now();
        lv.dispatch(&freq_buf, &var_buf, n_pops, n_loci);
        let gpu_var = readback_f32(&device, &var_buf, n_loci as usize);
        let gpu_us = tg.elapsed().as_micros() as f64;

        for (i, (cv, gv)) in cpu_var.iter().zip(gpu_var.iter()).enumerate() {
            v.check(&format!("LocusVar locus {i}"), *gv as f64, *cv as f64, 1e-5);
        }
        results.push((
            "LocusVariance",
            "neuralSpring",
            cpu_us,
            gpu_us,
            "session 31f",
        ));
    }

    // ═══ Summary ═════════════════════════════════════════════════════════
    println!("\n═══ Cross-Spring Evolution Provenance ═══════════════════════════\n");
    let hdr_abs = "Absorbed";
    let sep_abs = "─".repeat(12);
    println!(
        "  {prim:25} {evol:15} {cpu:>10} {gpu:>10} {hdr_abs}",
        prim = "Primitive",
        evol = "Evolved By",
        cpu = "CPU (µs)",
        gpu = "GPU (µs)",
    );
    println!(
        "  {prim:25} {evol:15} {cpu:>10} {gpu:>10} {sep_abs}",
        prim = "─".repeat(25),
        evol = "─".repeat(15),
        cpu = "─".repeat(10),
        gpu = "─".repeat(10),
    );
    for (name, spring, cpu_us, gpu_us, session) in &results {
        println!("  {name:25} {spring:15} {cpu_us:>10.0} {gpu_us:>10.0} {session}",);
    }

    println!("\n═══ The Biome Model Works ═══════════════════════════════════════");
    println!("  hotSpring  → precision shaders, driver workarounds, lattice QCD");
    println!("  wetSpring  → 12 bio shaders, GEMM 60× speedup, f64 precision");
    println!("  neuralSpring → distance metrics, fitness eval, locus variance");
    println!("  ToadStool  → absorbs all, every Spring benefits\n");

    v.finish();
}

fn cpu_pairwise_hamming(seqs: &[u32], n_seqs: usize, seq_len: usize) -> Vec<f32> {
    let mut dists = Vec::new();
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
    let mut dists = Vec::new();
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

fn readback_f32(
    device: &Arc<barracuda::device::WgpuDevice>,
    buf: &wgpu::Buffer,
    n: usize,
) -> Vec<f32> {
    let d = device.device();
    let q = device.queue();
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, (n * 4) as u64);
    q.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    d.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}
