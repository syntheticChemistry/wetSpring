// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
//! Exp165: metalForge Drug Repurposing — CPU vs GPU Cross-Substrate Parity
//!
//! Proves that Track 3 drug repurposing computations produce identical
//! results on CPU and GPU substrates. This is the metalForge three-tier
//! proof: same math, different hardware, same answers.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | 1f9f80e |
//! | Baseline tool | CPU ↔ GPU parity |
//! | Baseline date | 2026-02-25 |
//! | Exact command | `cargo run --features gpu --release --bin validate_metalforge_drug_repurposing` |
//! | Data | Synthetic test vectors (self-contained) |

use barracuda::linalg::nmf::{self, NmfConfig, NmfObjective};
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::transe_score_f64::TranseScoreF64;
use std::time::Instant;
use wetspring_barracuda::bio::gemm_cached::GemmCached;
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct LcgRng(u64);

impl LcgRng {
    const fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.0 >> 11) | 0x3FF0_0000_0000_0000;
        f64::from_bits(bits) - 1.0
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_f64() * max as f64) as usize % max
    }
}

// ── MF01: NMF CPU vs GPU Reconstruction Parity ─────────────────────────────

fn validate_nmf_reconstruction_parity(
    v: &mut Validator,
    gemm: &GemmCached,
    timings: &mut Vec<(&'static str, f64, f64, f64)>,
) {
    v.section("═══ MF01: NMF Reconstruction — CPU vs GPU ═══");

    let n_drugs = 200;
    let n_diseases = 150;
    let total = n_drugs * n_diseases;

    let mut rng = LcgRng::new(42);
    let mut matrix = vec![0.0_f64; total];
    for c in 0..10 {
        let mut planted = 0;
        while planted < 80 {
            let d = c * 20 + rng.next_usize(20);
            let dis = c * 15 + rng.next_usize(15);
            if matrix[d * n_diseases + dis] == 0.0 {
                matrix[d * n_diseases + dis] = 1.0;
                planted += 1;
            }
        }
    }

    let config = NmfConfig {
        rank: 10,
        max_iter: 200,
        tol: 1e-6,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };
    let result = nmf::nmf(&matrix, n_drugs, n_diseases, &config).expect("NMF failed");
    let rank = result.k;

    let t_cpu = Instant::now();
    let mut cpu_wh = vec![0.0_f64; total];
    for i in 0..n_drugs {
        for kk in 0..rank {
            let w_ik = result.w[i * rank + kk];
            for j in 0..n_diseases {
                cpu_wh[i * n_diseases + j] += w_ik * result.h[kk * n_diseases + j];
            }
        }
    }
    let cpu_us = t_cpu.elapsed().as_micros() as f64;

    let t_gpu = Instant::now();
    let gpu_wh = gemm
        .execute(&result.w, &result.h, n_drugs, rank, n_diseases, 1)
        .expect("GEMM failed");
    let gpu_us = t_gpu.elapsed().as_micros() as f64;

    let max_diff = cpu_wh
        .iter()
        .zip(gpu_wh.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0_f64, f64::max);

    let mean_diff = cpu_wh
        .iter()
        .zip(gpu_wh.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f64>()
        / total as f64;

    println!("  NMF {n_drugs}×{n_diseases}, rank={rank}");
    println!("  Max  CPU↔GPU diff: {max_diff:.2e}");
    println!("  Mean CPU↔GPU diff: {mean_diff:.2e}");

    v.check(
        "max reconstruction diff < f64 tolerance",
        max_diff,
        0.0,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check_pass("matching element count", cpu_wh.len() == gpu_wh.len());

    timings.push(("NMF reconstruct", cpu_us, gpu_us, max_diff));
}

// ── MF02: Cosine Scoring CPU vs GPU ─────────────────────────────────────────

fn validate_cosine_scoring_parity(
    v: &mut Validator,
    fmr: &FusedMapReduceF64,
    timings: &mut Vec<(&'static str, f64, f64, f64)>,
) {
    v.section("═══ MF02: Cosine Scoring — CPU vs GPU ═══");

    let mut rng = LcgRng::new(77);

    let dims = [50, 200, 1000];
    for dim in dims {
        let a: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
        let b: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();

        let t_cpu = Instant::now();
        let cpu_cos = nmf::cosine_similarity(&a, &b);
        let cpu_us = t_cpu.elapsed().as_micros() as f64;

        let t_gpu = Instant::now();
        let gpu_dot = fmr.dot(&a, &b).expect("FMR dot");
        let gpu_na = fmr.sum_of_squares(&a).expect("FMR sos").sqrt();
        let gpu_nb = fmr.sum_of_squares(&b).expect("FMR sos").sqrt();
        let gpu_cos = gpu_dot / (gpu_na * gpu_nb);
        let gpu_us = t_gpu.elapsed().as_micros() as f64;

        let diff = (cpu_cos - gpu_cos).abs();
        println!("  dim={dim}: CPU={cpu_cos:.10} GPU={gpu_cos:.10} diff={diff:.2e}");

        v.check(
            &format!("cosine dim={dim}"),
            gpu_cos,
            cpu_cos,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );

        if dim == 1000 {
            timings.push(("cosine 1000-d", cpu_us, gpu_us, diff));
        }
    }
}

// ── MF03: TransE CPU vs GPU ─────────────────────────────────────────────────

fn validate_transe_parity(
    v: &mut Validator,
    gpu: &GpuF64,
    timings: &mut Vec<(&'static str, f64, f64, f64)>,
) {
    v.section("═══ MF03: TransE Scoring — CPU vs GPU ═══");

    const EMBED_DIM: usize = 32;
    const N_ENTITIES: usize = 310;
    const N_RELATIONS: usize = 4;

    let mut rng = LcgRng::new(42);
    let mut entity_emb = vec![0.0_f64; N_ENTITIES * EMBED_DIM];
    for i in 0..N_ENTITIES {
        let row = &mut entity_emb[i * EMBED_DIM..(i + 1) * EMBED_DIM];
        for val in row.iter_mut() {
            *val = rng.next_f64().mul_add(0.2, -0.1);
        }
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for val in row.iter_mut() {
                *val /= norm;
            }
        }
    }
    let mut relation_emb = vec![0.0_f64; N_RELATIONS * EMBED_DIM];
    for val in &mut relation_emb {
        *val = rng.next_f64().mul_add(0.2, -0.1);
    }

    let n_triples = 500;
    let mut triples = Vec::new();
    for c in 0..10 {
        for i in 0..10 {
            triples.push((c * 10 + i, 0, 100 + c * 10 + i));
            triples.push((c * 10 + i, 1, 200 + c * 10 + i));
            triples.push((200 + c * 10 + i, 2, 100 + c * 10 + i));
        }
    }
    triples.truncate(n_triples);

    let heads: Vec<u32> = triples.iter().map(|t| t.0 as u32).collect();
    let rels: Vec<u32> = triples.iter().map(|t| t.1 as u32).collect();
    let tails: Vec<u32> = triples.iter().map(|t| t.2 as u32).collect();

    let t_cpu = Instant::now();
    let cpu_scores: Vec<f64> = triples
        .iter()
        .map(|&(h, r, t)| {
            let mut ss = 0.0;
            for d in 0..EMBED_DIM {
                let diff = entity_emb[h * EMBED_DIM + d] + relation_emb[r * EMBED_DIM + d]
                    - entity_emb[t * EMBED_DIM + d];
                ss += diff * diff;
            }
            -ss.sqrt()
        })
        .collect();
    let cpu_us = t_cpu.elapsed().as_micros() as f64;

    let device = gpu.to_wgpu_device();
    let scorer = TranseScoreF64 {
        entities: &entity_emb,
        relations: &relation_emb,
        n_entities: N_ENTITIES,
        n_relations: N_RELATIONS,
        dim: EMBED_DIM,
        heads: &heads,
        rels: &rels,
        tails: &tails,
    };

    let t_gpu = Instant::now();
    let gpu_scores = scorer.execute(&device).expect("GPU TransE");
    let gpu_us = t_gpu.elapsed().as_micros() as f64;

    let max_diff = gpu_scores
        .iter()
        .zip(cpu_scores.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0_f64, f64::max);

    println!("  {} triples, embed_dim={EMBED_DIM}", triples.len());
    println!("  Max CPU↔GPU diff: {max_diff:.2e}");
    println!("  CPU: {cpu_us:.0}µs, GPU: {gpu_us:.0}µs");

    v.check(
        "TransE max diff",
        max_diff,
        0.0,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check_pass("score count matches", gpu_scores.len() == cpu_scores.len());

    timings.push(("TransE 300-triple", cpu_us, gpu_us, max_diff));
}

// ── MF04: Drug Ranking Parity ───────────────────────────────────────────────

fn validate_drug_ranking_parity(v: &mut Validator, gemm: &GemmCached) {
    v.section("═══ MF04: Drug Ranking Parity ═══");

    let n_drugs = 100;
    let n_diseases = 80;
    let total = n_drugs * n_diseases;
    let rank = 8;

    let mut rng = LcgRng::new(55);
    let mut matrix = vec![0.0_f64; total];
    for c in 0..5 {
        for _ in 0..50 {
            let d = c * 20 + rng.next_usize(20);
            let dis = c * 16 + rng.next_usize(16);
            matrix[d * n_diseases + dis] = 1.0;
        }
    }

    let config = NmfConfig {
        rank,
        max_iter: 200,
        tol: 1e-6,
        objective: NmfObjective::Euclidean,
        seed: 42,
    };
    let result = nmf::nmf(&matrix, n_drugs, n_diseases, &config).expect("NMF failed");

    let mut cpu_wh = vec![0.0_f64; total];
    for i in 0..n_drugs {
        for kk in 0..rank {
            let w_ik = result.w[i * rank + kk];
            for j in 0..n_diseases {
                cpu_wh[i * n_diseases + j] += w_ik * result.h[kk * n_diseases + j];
            }
        }
    }

    let gpu_wh = gemm
        .execute(&result.w, &result.h, n_drugs, rank, n_diseases, 1)
        .expect("GEMM failed");

    let top_k = 20;

    let mut cpu_scores: Vec<(usize, f64)> = cpu_wh.iter().copied().enumerate().collect();
    cpu_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let cpu_top: Vec<usize> = cpu_scores.iter().take(top_k).map(|(i, _)| *i).collect();

    let mut gpu_scores: Vec<(usize, f64)> = gpu_wh.iter().copied().enumerate().collect();
    gpu_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let gpu_top: Vec<usize> = gpu_scores.iter().take(top_k).map(|(i, _)| *i).collect();

    let overlap = cpu_top.iter().filter(|i| gpu_top.contains(i)).count();

    println!("  Top-{top_k} drug-disease predictions:");
    println!("  CPU top-{top_k} = {:?}", &cpu_top[..5]);
    println!("  GPU top-{top_k} = {:?}", &gpu_top[..5]);
    println!("  Overlap: {overlap}/{top_k}");

    v.check_pass(
        "top-K predictions identical on CPU and GPU",
        cpu_top == gpu_top,
    );
    v.check_pass("100% overlap", overlap == top_k);
}

// ── MF05: Timing Comparison ─────────────────────────────────────────────────

fn print_timing_summary(timings: &[(&str, f64, f64, f64)]) {
    println!(
        "\n  {:>22} {:>10} {:>10} {:>10} {:>8}",
        "Domain", "CPU µs", "GPU µs", "Max Diff", "Speedup"
    );
    println!(
        "  {:->22} {:->10} {:->10} {:->10} {:->8}",
        "", "", "", "", ""
    );
    for (name, cpu, gpu_t, diff) in timings {
        let speedup = if *gpu_t > 0.0 { cpu / gpu_t } else { f64::NAN };
        println!("  {name:>22} {cpu:>10.0} {gpu_t:>10.0} {diff:>10.2e} {speedup:>7.2}×");
    }
}

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("GPU init — requires SHADER_F64");
    let device = gpu.to_wgpu_device();
    let ctx = gpu.tensor_context().clone();

    let mut v =
        Validator::new("Exp165: metalForge Drug Repurposing — CPU vs GPU Cross-Substrate Parity");
    let t_total = Instant::now();
    let mut timings: Vec<(&str, f64, f64, f64)> = Vec::new();

    let gemm = GemmCached::new(device.clone(), ctx);
    let fmr = FusedMapReduceF64::new(device.clone()).expect("FMR init");

    validate_nmf_reconstruction_parity(&mut v, &gemm, &mut timings);
    validate_cosine_scoring_parity(&mut v, &fmr, &mut timings);
    validate_transe_parity(&mut v, &gpu, &mut timings);
    validate_drug_ranking_parity(&mut v, &gemm);

    v.section("═══ MF05: Timing Summary ═══");
    print_timing_summary(&timings);

    let total_ms = t_total.elapsed().as_millis();
    println!("\n  Total wall-clock: {total_ms} ms");

    v.finish();
}
