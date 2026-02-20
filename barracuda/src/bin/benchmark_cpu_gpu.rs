// SPDX-License-Identifier: AGPL-3.0-or-later
//! Benchmark: CPU Rust vs GPU (ToadStool/BarraCUDA) across scientific workloads.
//!
//! Measures wall-clock time and energy for each workload at multiple data sizes
//! to demonstrate GPU parallelism advantage. Emits structured JSON via
//! [`BenchReport`] and a human-readable summary table.
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
//! `PCoA` eigendecomposition

use std::time::Instant;
use wetspring_barracuda::bench::{
    self, BenchReport, EnergyReport, HardwareInventory, PhaseResult, PowerMonitor,
};
use wetspring_barracuda::bio::{
    diversity, diversity_gpu, pcoa, pcoa_gpu, spectral_match, spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    let hw = HardwareInventory::detect("wetSpring CPU vs GPU");
    hw.print();

    let mut report = BenchReport::new(hw);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  wetSpring CPU vs GPU Benchmark (with energy profiling)             ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // ── Single-vector reductions ──────────────────────────────────────
    section("SINGLE-VECTOR REDUCTIONS (GPU overhead vs data size)");
    bench_shannon(&gpu, &mut report);
    bench_simpson(&gpu, &mut report);
    bench_variance(&gpu, &mut report);
    bench_dot_product(&gpu, &mut report);

    // ── Pairwise N×N workloads ────────────────────────────────────────
    section("PAIRWISE N×N WORKLOADS (GPU parallelism scales with N²)");
    bench_bray_curtis(&gpu, &mut report);
    bench_spectral_cosine(&gpu, &mut report);

    // ── Matrix algebra ────────────────────────────────────────────────
    section("MATRIX ALGEBRA (eigendecomposition, GEMM)");
    bench_pcoa(&gpu, &mut report);

    report.print_summary();

    let out_dir = format!("{}/../benchmarks/results", env!("CARGO_MANIFEST_DIR"));
    match report.save_json(&out_dir) {
        Ok(path) => println!("JSON results saved to {path}"),
        Err(e) => eprintln!("Warning: could not save JSON: {e}"),
    }
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
fn display_and_record(
    label: &str,
    n: usize,
    cpu_us: f64,
    gpu_us: f64,
    cpu_energy: EnergyReport,
    gpu_energy: EnergyReport,
    report: &mut BenchReport,
) {
    let speedup = if gpu_us > 0.01 { cpu_us / gpu_us } else { 0.0 };
    let arrow = if speedup >= 1.0 { "▲" } else { "▼" };
    let cpu_str = format_time(cpu_us);
    let gpu_str = format_time(gpu_us);
    println!("│ {label:<26} {n:>8} {cpu_str:>11} {gpu_str:>11} {speedup:>6.2}x {arrow}│",);

    let phase_name = format!("{label} N={n}");
    let cpu_iters = (cpu_us.recip() * 1e6 * 0.1).max(1.0) as usize; // estimate
    let gpu_iters = (gpu_us.recip() * 1e6 * 0.1).max(1.0) as usize;

    report.add_phase(PhaseResult {
        phase: phase_name.clone(),
        substrate: "Rust CPU".to_string(),
        wall_time_s: cpu_us / 1e6,
        per_eval_us: cpu_us,
        n_evals: cpu_iters,
        energy: cpu_energy,
        peak_rss_mb: bench::peak_rss_mb(),
        notes: String::new(),
    });
    report.add_phase(PhaseResult {
        phase: phase_name,
        substrate: "BarraCUDA GPU".to_string(),
        wall_time_s: gpu_us / 1e6,
        per_eval_us: gpu_us,
        n_evals: gpu_iters,
        energy: gpu_energy,
        peak_rss_mb: bench::peak_rss_mb(),
        notes: String::new(),
    });
}

fn format_time(us: f64) -> String {
    if us < 1.0 {
        format!("{:.1}ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{us:.1}µs")
    } else {
        format!("{:.2}ms", us / 1000.0)
    }
}

const WARMUP: usize = 3;

/// Time a closure with energy monitoring, returning (microseconds/iter, energy report).
#[allow(clippy::cast_precision_loss)]
fn bench_with_energy<F: FnMut()>(mut f: F) -> (f64, EnergyReport) {
    for _ in 0..WARMUP {
        f();
    }

    let mut iters = 5_u64;
    loop {
        let monitor = PowerMonitor::start();
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        let elapsed = start.elapsed();
        let energy = monitor.stop();
        if elapsed.as_micros() > 10_000 || iters >= 1000 {
            return (elapsed.as_secs_f64() * 1_000_000.0 / iters as f64, energy);
        }
        iters = (iters * 3).min(1000);
    }
}

// ── Shannon ──────────────────────────────────────────────────────────

fn bench_shannon(gpu: &GpuF64, report: &mut BenchReport) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_counts(n, 42);
        let (cpu, cpu_e) = bench_with_energy(|| {
            let _ = diversity::shannon(&data);
        });
        let (gpu_t, gpu_e) = bench_with_energy(|| {
            let _ = diversity_gpu::shannon_gpu(gpu, &data);
        });
        display_and_record("Shannon entropy", n, cpu, gpu_t, cpu_e, gpu_e, report);
    }
}

// ── Simpson ──────────────────────────────────────────────────────────

fn bench_simpson(gpu: &GpuF64, report: &mut BenchReport) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_counts(n, 123);
        let (cpu, cpu_e) = bench_with_energy(|| {
            let _ = diversity::simpson(&data);
        });
        let (gpu_t, gpu_e) = bench_with_energy(|| {
            let _ = diversity_gpu::simpson_gpu(gpu, &data);
        });
        display_and_record("Simpson diversity", n, cpu, gpu_t, cpu_e, gpu_e, report);
    }
}

// ── Variance ─────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
fn bench_variance(gpu: &GpuF64, report: &mut BenchReport) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_f64(n, 7);
        let (cpu, cpu_e) = bench_with_energy(|| {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let _var: f64 =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        });
        let (gpu_t, gpu_e) = bench_with_energy(|| {
            let _ = stats_gpu::variance_gpu(gpu, &data);
        });
        display_and_record("Variance", n, cpu, gpu_t, cpu_e, gpu_e, report);
    }
}

// ── Dot product ──────────────────────────────────────────────────────

fn bench_dot_product(gpu: &GpuF64, report: &mut BenchReport) {
    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let a = generate_f64(n, 11);
        let b = generate_f64(n, 22);
        let (cpu, cpu_e) = bench_with_energy(|| {
            let _dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        });
        let (gpu_t, gpu_e) = bench_with_energy(|| {
            let _ = stats_gpu::dot_gpu(gpu, &a, &b);
        });
        display_and_record("Dot product", n, cpu, gpu_t, cpu_e, gpu_e, report);
    }
}

// ── Bray-Curtis ──────────────────────────────────────────────────────

fn bench_bray_curtis(gpu: &GpuF64, report: &mut BenchReport) {
    for &(n_samples, n_species) in &[(10, 500), (20, 500), (50, 500), (100, 500)] {
        let samples: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| generate_counts(n_species, 42 + i as u64))
            .collect();

        let (cpu, cpu_e) = bench_with_energy(|| {
            let _ = diversity::bray_curtis_condensed(&samples);
        });
        let (gpu_t, gpu_e) = bench_with_energy(|| {
            let _ = diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples);
        });
        let n_pairs = n_samples * (n_samples - 1) / 2;
        display_and_record(
            &format!("Bray-Curtis {n_samples}x{n_samples}"),
            n_pairs,
            cpu,
            gpu_t,
            cpu_e,
            gpu_e,
            report,
        );
    }
}

// ── Spectral cosine ──────────────────────────────────────────────────

fn bench_spectral_cosine(gpu: &GpuF64, report: &mut BenchReport) {
    for &(n_spectra, dim) in &[(10, 500), (50, 500), (100, 500), (200, 500)] {
        let gpu_spectra: Vec<Vec<f64>> = (0..n_spectra)
            .map(|i| generate_f64(dim, 300 + i as u64))
            .collect();

        let cpu_spectra: Vec<(Vec<f64>, Vec<f64>)> = gpu_spectra
            .iter()
            .map(|intensities| {
                let mzs: Vec<f64> = (0..dim).map(|j| (j as f64).mul_add(0.5, 100.0)).collect();
                (mzs, intensities.clone())
            })
            .collect();

        let (cpu, cpu_e) = bench_with_energy(|| {
            let _ = spectral_match::pairwise_cosine(&cpu_spectra, 0.5);
        });
        let (gpu_t, gpu_e) = bench_with_energy(|| {
            let _ = spectral_match_gpu::pairwise_cosine_gpu(gpu, &gpu_spectra);
        });
        let n_pairs = n_spectra * (n_spectra - 1) / 2;
        display_and_record(
            &format!("Cosine {n_spectra}x{n_spectra}"),
            n_pairs,
            cpu,
            gpu_t,
            cpu_e,
            gpu_e,
            report,
        );
    }
}

// ── PCoA ─────────────────────────────────────────────────────────────

fn bench_pcoa(gpu: &GpuF64, report: &mut BenchReport) {
    for &n_samples in &[10, 20, 30] {
        let n_species = 200;
        let samples: Vec<Vec<f64>> = (0..n_samples)
            .map(|i| generate_counts(n_species, 100 + i as u64))
            .collect();
        let distances = diversity::bray_curtis_condensed(&samples);

        let (cpu, cpu_e) = bench_with_energy(|| {
            let _ = pcoa::pcoa(&distances, n_samples, 3);
        });
        let (gpu_t, gpu_e) = bench_with_energy(|| {
            let _ = pcoa_gpu::pcoa_gpu(gpu, &distances, n_samples, 3);
        });
        display_and_record(
            &format!("PCoA {n_samples}x{n_samples}"),
            n_samples,
            cpu,
            gpu_t,
            cpu_e,
            gpu_e,
            report,
        );
    }
}
