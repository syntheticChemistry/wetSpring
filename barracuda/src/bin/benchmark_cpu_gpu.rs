// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark: CPU Rust vs GPU (ToadStool/BarraCUDA) across scientific workloads.
//!
//! Measures wall-clock time for each workload at multiple data sizes to
//! demonstrate GPU parallelism advantage. Reports speedup factors.
//!
//! GPU excels at **batch parallelism** — many independent computations in
//! parallel. Single-vector reductions (sum, dot) are memory-bandwidth-bound
//! and CPU wins at small N due to GPU dispatch overhead (~0.5–2ms).
//!
//! Run: `cargo run --release --features gpu --bin benchmark_cpu_gpu`
//!
//! # Workloads (grouped by parallelism type)
//!
//! ## Single-vector (GPU overhead vs data size)
//! Shannon, Simpson, Variance, Correlation, Dot product
//!
//! ## Pairwise N×N (GPU parallelism scales with N²)
//! Bray-Curtis matrix, Spectral cosine matrix
//!
//! ## Matrix algebra (GPU wins at matrix size)
//! PCoA eigendecomposition

use std::time::Instant;
use wetspring_barracuda::bio::{
    diversity, diversity_gpu, pcoa, pcoa_gpu, spectral_match, spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  wetSpring CPU vs GPU Benchmark                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    gpu.print_info();
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // ── Single-vector reductions ──────────────────────────────────────
    section("SINGLE-VECTOR REDUCTIONS (GPU overhead vs data size)");
    bench_shannon(&gpu);
    bench_simpson(&gpu);
    bench_variance(&gpu);
    bench_dot_product(&gpu);

    // ── Pairwise N×N workloads ────────────────────────────────────────
    section("PAIRWISE N×N WORKLOADS (GPU parallelism scales with N²)");
    bench_bray_curtis(&gpu);
    bench_spectral_cosine(&gpu);

    // ── Matrix algebra ────────────────────────────────────────────────
    section("MATRIX ALGEBRA (eigendecomposition, GEMM)");
    bench_pcoa(&gpu);

    println!("\n══════════════════════════════════════════════════════════════════════");
    println!("Notes:");
    println!("  ▲ = GPU faster   ▼ = CPU faster");
    println!("  GPU dispatch overhead is ~0.5-2ms per call (buffer upload + readback).");
    println!("  GPU advantage emerges at O(N²) or batch workloads where parallelism");
    println!("  amortizes dispatch cost across many independent computations.");
}

// ── Helpers ──────────────────────────────────────────────────────────

fn section(title: &str) {
    println!("\n┌────────────────────────────────────────────────────────────────────┐");
    println!("│ {title:<66} │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ {:<26} {:>8} {:>11} {:>11} {:>9}│",
        "Workload", "N", "CPU", "GPU", "Speedup"
    );
    println!("├────────────────────────────────────────────────────────────────────┤");
}

fn generate_counts(n: usize, seed: u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut rng = seed;
    for _ in 0..n {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        v.push(((rng >> 33) % 1000 + 1) as f64);
    }
    v
}

fn generate_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut rng = seed;
    for _ in 0..n {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        v.push((rng >> 11) as f64 / (1_u64 << 53) as f64);
    }
    v
}

#[allow(clippy::cast_precision_loss)]
fn report(label: &str, n: usize, cpu_us: f64, gpu_us: f64) {
    let speedup = if gpu_us > 0.01 {
        cpu_us / gpu_us
    } else {
        0.0
    };
    let arrow = if speedup >= 1.0 { "▲" } else { "▼" };
    let cpu_str = format_time(cpu_us);
    let gpu_str = format_time(gpu_us);
    println!(
        "│ {:<26} {:>8} {:>11} {:>11} {:>6.2}x {arrow}│",
        label, n, cpu_str, gpu_str, speedup,
    );
}

fn format_time(us: f64) -> String {
    if us < 1.0 {
        format!("{:.1}ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{:.1}µs", us)
    } else {
        format!("{:.2}ms", us / 1000.0)
    }
}

const WARMUP: usize = 3;

/// Time a closure, returning microseconds per iteration.
/// Adapts iteration count to get reliable timing.
#[allow(clippy::cast_precision_loss)]
fn bench<F: FnMut()>(mut f: F) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        f();
    }

    // Calibrate: run until total > 10ms or 1000 iters
    let mut iters = 5_u64;
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = start.elapsed();
        if elapsed.as_micros() > 10_000 || iters >= 1000 {
            return elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
        }
        iters = (iters * 3).min(1000);
    }
}

// ── Shannon ──────────────────────────────────────────────────────────

fn bench_shannon(gpu: &GpuF64) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_counts(n, 42);
        let cpu = bench(|| {
            let _ = diversity::shannon(&data);
        });
        let gpu_t = bench(|| {
            let _ = diversity_gpu::shannon_gpu(gpu, &data);
        });
        report("Shannon entropy", n, cpu, gpu_t);
    }
}

// ── Simpson ──────────────────────────────────────────────────────────

fn bench_simpson(gpu: &GpuF64) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_counts(n, 123);
        let cpu = bench(|| {
            let _ = diversity::simpson(&data);
        });
        let gpu_t = bench(|| {
            let _ = diversity_gpu::simpson_gpu(gpu, &data);
        });
        report("Simpson diversity", n, cpu, gpu_t);
    }
}

// ── Variance ─────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
fn bench_variance(gpu: &GpuF64) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64(n, 7);
        let cpu = bench(|| {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let _var: f64 =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        });
        let gpu_t = bench(|| {
            let _ = stats_gpu::variance_gpu(gpu, &data);
        });
        report("Variance", n, cpu, gpu_t);
    }
}

// ── Dot product ──────────────────────────────────────────────────────

fn bench_dot_product(gpu: &GpuF64) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let a = generate_f64(n, 11);
        let b = generate_f64(n, 22);
        let cpu = bench(|| {
            let _dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        });
        let gpu_t = bench(|| {
            let _ = stats_gpu::dot_gpu(gpu, &a, &b);
        });
        report("Dot product", n, cpu, gpu_t);
    }
}

// ── Bray-Curtis ──────────────────────────────────────────────────────

fn bench_bray_curtis(gpu: &GpuF64) {
    for &(n_samples, n_species) in &[(10, 500), (20, 500), (50, 500), (100, 500)] {
        let samples: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| generate_counts(n_species, 42 + i as u64))
            .collect();

        let cpu = bench(|| {
            let _ = diversity::bray_curtis_condensed(&samples);
        });
        let gpu_t = bench(|| {
            let _ = diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples);
        });
        let n_pairs = n_samples * (n_samples - 1) / 2;
        report(
            &format!("Bray-Curtis {n_samples}×{n_samples}"),
            n_pairs,
            cpu,
            gpu_t,
        );
    }
}

// ── Spectral cosine ──────────────────────────────────────────────────

fn bench_spectral_cosine(gpu: &GpuF64) {
    for &(n_spectra, dim) in &[(10, 500), (50, 500), (100, 500), (200, 500)] {
        let gpu_spectra: Vec<Vec<f64>> = (0..n_spectra)
            .map(|i| generate_f64(dim, 300 + i as u64))
            .collect();

        let cpu_spectra: Vec<(Vec<f64>, Vec<f64>)> = gpu_spectra
            .iter()
            .map(|intensities| {
                let mzs: Vec<f64> = (0..dim).map(|j| 100.0 + j as f64 * 0.5).collect();
                (mzs, intensities.clone())
            })
            .collect();

        let cpu = bench(|| {
            let _ = spectral_match::pairwise_cosine(&cpu_spectra, 0.5);
        });
        let gpu_t = bench(|| {
            let _ = spectral_match_gpu::pairwise_cosine_gpu(gpu, &gpu_spectra);
        });
        let n_pairs = n_spectra * (n_spectra - 1) / 2;
        report(
            &format!("Cosine {n_spectra}×{n_spectra}"),
            n_pairs,
            cpu,
            gpu_t,
        );
    }
}

// ── PCoA ─────────────────────────────────────────────────────────────

fn bench_pcoa(gpu: &GpuF64) {
    for &n_samples in &[10, 20, 30] {
        let n_species = 200;
        let samples: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| generate_counts(n_species, 100 + i as u64))
            .collect();
        let distances = diversity::bray_curtis_condensed(&samples);

        let cpu = bench(|| {
            let _ = pcoa::pcoa(&distances, n_samples, 3);
        });
        let gpu_t = bench(|| {
            let _ = pcoa_gpu::pcoa_gpu(gpu, &distances, n_samples, 3);
        });
        report(&format!("PCoA {n_samples}×{n_samples}"), n_samples, cpu, gpu_t);
    }
}
