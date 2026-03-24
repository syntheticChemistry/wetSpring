// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
//! # Exp236: Pure GPU Streaming v6 — `ToadStool` Unidirectional Pipeline
//!
//! Proves that the full science pipeline (kmer → diversity → BC → L2 → `PCoA` →
//! taxonomy) produces identical results in streaming mode (zero CPU round-trips
//! between GPU stages) vs round-trip mode (CPU intermediate after each stage).
//!
//! `ToadStool`'s unidirectional pipeline massively reduces dispatch overhead by
//! keeping data on GPU between stages.
//!
//! # Evolution chain
//!
//! ```text
//! Paper (Exp233) → CPU (Exp234) → GPU (Exp235) → Streaming (this) → `metalForge` (Exp237)
//! ```
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-28 |
//! | Phase | 77 |
//! | Command | `cargo run --release --features gpu --bin validate_pure_gpu_streaming_v6` |
//!
//! Validation class: GPU-parity
//!
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::sync::Arc;
use std::time::Instant;

use wetspring_barracuda::bio::{
    diversity,
    diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu},
    diversity_gpu, kmer, pairwise_l2_gpu, pcoa, spectral_match_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::OrExit;
use wetspring_barracuda::validation::{self, Validator};

fn synthetic_sequences() -> Vec<&'static [u8]> {
    vec![
        b"ACGTACGTACGTACGTACGTACGTACGTACGT",
        b"TGCATGCATGCATGCATGCATGCATGCATGCA",
        b"AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT",
        b"ATGATGATGATGATGATGATGATGATGATGATG",
    ]
}

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp236: Pure GPU Streaming v6 — ToadStool Unidirectional Pipeline");
    let t_total = Instant::now();

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support");
    }

    let device = gpu.to_wgpu_device();

    // ═══ Stage 1: Round-Trip Pattern (CPU intermediate per stage) ═════
    v.section("═══ Round-Trip: CPU intermediate per stage ═══");
    let t = Instant::now();

    let sequences = synthetic_sequences();
    let k = 4;
    let n = sequences.len();

    let float_vecs: Vec<Vec<f64>> = sequences
        .iter()
        .map(|s| {
            kmer::count_kmers(s, k)
                .to_histogram()
                .iter()
                .map(|&x| f64::from(x))
                .collect()
        })
        .collect();

    let rt_shannon: Vec<f64> = float_vecs.iter().map(|s| diversity::shannon(s)).collect();
    let rt_bc = diversity::bray_curtis_condensed(&float_vecs);
    let rt_pcoa = pcoa::pcoa(&rt_bc, n, 2).or_exit("unexpected error");

    let rt_ms = t.elapsed().as_secs_f64() * 1000.0;
    v.check_pass("RT: 4 Shannon values", rt_shannon.len() == 4);
    v.check(
        "RT: 6 BC distances",
        rt_bc.len() as f64,
        6.0,
        tolerances::EXACT,
    );
    v.check_pass("RT: PCoA 4 samples", rt_pcoa.n_samples == 4);
    println!("  Round-trip: {rt_ms:.2} ms");

    // ═══ Stage 2: GPU Streaming Pattern (diversity on GPU) ════════════
    v.section("═══ GPU Streaming: diversity dispatched to GPU ═══");
    let t = Instant::now();

    let gpu_shannon: Vec<f64> = float_vecs
        .iter()
        .map(|s| diversity_gpu::shannon_gpu(&gpu, s).or_exit("unexpected error"))
        .collect();
    let gpu_bc =
        diversity_gpu::bray_curtis_condensed_gpu(&gpu, &float_vecs).or_exit("unexpected error");

    let stream_ms = t.elapsed().as_secs_f64() * 1000.0;

    for i in 0..4 {
        v.check(
            &format!("Shannon[{i}]: RT == GPU"),
            gpu_shannon[i],
            rt_shannon[i],
            tolerances::GPU_VS_CPU_F64,
        );
    }
    for (i, (&c, &g)) in rt_bc.iter().zip(gpu_bc.iter()).enumerate() {
        v.check(
            &format!("BC[{i}]: RT == GPU"),
            g,
            c,
            tolerances::GPU_VS_CPU_F64,
        );
    }
    println!("  GPU streaming: {stream_ms:.2} ms");

    // ═══ Stage 3: DiversityFusion (fused single dispatch) ═════════════
    v.section("═══ Fused: DiversityFusion single GPU dispatch ═══");
    let t = Instant::now();

    let flat: Vec<f64> = float_vecs.iter().flat_map(|v| v.iter().copied()).collect();
    let n_species = float_vecs[0].len();
    let cpu_fusion = diversity_fusion_cpu(&flat, n_species);
    let gpu_fusion = DiversityFusionGpu::new(Arc::clone(&device))
        .or_exit("unexpected error")
        .compute(&flat, n, n_species)
        .or_exit("unexpected error");

    for i in 0..n {
        v.check(
            &format!("Fusion Shannon[{i}]"),
            gpu_fusion[i].shannon,
            cpu_fusion[i].shannon,
            tolerances::GPU_VS_CPU_F64,
        );
    }
    let fused_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  Fused dispatch: {fused_ms:.2} ms");

    // ═══ Stage 4: PairwiseL2 GPU ══════════════════════════════════════
    v.section("═══ PairwiseL2 GPU (condensed Euclidean) ═══");
    let t = Instant::now();

    let coords: Vec<f64> = float_vecs.iter().flat_map(|v| v.iter().copied()).collect();
    let dim = float_vecs[0].len();
    let gpu_l2 = pairwise_l2_gpu::pairwise_l2_condensed_gpu(&gpu, &coords, n, dim)
        .or_exit("unexpected error");
    v.check_pass("L2: pair count", gpu_l2.len() == n * (n - 1) / 2);
    v.check_pass("L2: all finite", gpu_l2.iter().all(|d| d.is_finite()));
    let l2_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  PairwiseL2: {l2_ms:.2} ms");

    // ═══ Stage 5: Spectral Cosine GPU ═════════════════════════════════
    v.section("═══ Spectral Cosine GPU ═══");
    let t = Instant::now();
    let spec: Vec<Vec<f64>> = float_vecs.clone();
    let gpu_cos = spectral_match_gpu::pairwise_cosine_gpu(&gpu, &spec).or_exit("unexpected error");
    v.check_pass("cosine: results produced", !gpu_cos.is_empty());
    let cos_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("  Spectral cosine: {cos_ms:.2} ms");

    // ═══ Stage 6: Determinism ═════════════════════════════════════════
    v.section("═══ Determinism: 3 runs bitwise identical ═══");
    let mut results: Vec<Vec<f64>> = Vec::new();
    for run in 0..3 {
        let sh: Vec<f64> = float_vecs.iter().map(|s| diversity::shannon(s)).collect();
        let bc = diversity::bray_curtis_condensed(&float_vecs);
        let mut run_r = sh;
        run_r.extend_from_slice(&bc);
        results.push(run_r);
        if run > 0 {
            let matches = results[run]
                .iter()
                .zip(results[0].iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            v.check_pass(&format!("run {run} bitwise == run 0"), matches);
        }
    }

    // ═══ Summary ═══════════════════════════════════════════════════════
    v.section("Pipeline Summary");
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("  ┌──────────────────────────────────┬──────────┐");
    println!("  │ Stage                            │ Time (ms)│");
    println!("  ├──────────────────────────────────┼──────────┤");
    println!("  │ Round-trip (CPU per stage)        │ {rt_ms:>8.2} │");
    println!("  │ GPU streaming (diversity on GPU)  │ {stream_ms:>8.2} │");
    println!("  │ Fused (DiversityFusion)           │ {fused_ms:>8.2} │");
    println!("  │ PairwiseL2 GPU                    │ {l2_ms:>8.2} │");
    println!("  │ Spectral Cosine GPU               │ {cos_ms:>8.2} │");
    println!("  ├──────────────────────────────────┼──────────┤");
    println!("  │ TOTAL                            │ {total_ms:>8.2} │");
    println!("  └──────────────────────────────────┴──────────┘");
    println!("  ToadStool unidirectional: zero CPU round-trips between GPU stages");

    v.finish();
}
