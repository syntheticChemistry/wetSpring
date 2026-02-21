// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::too_many_lines, clippy::cast_precision_loss)]
//! Exp069: Python → Rust CPU → GPU Three-Tier Benchmark
//!
//! Formalizes the full value chain: Python (numpy/scipy) → Rust CPU
//! (BarraCUDA) → Rust GPU (ToadStool + local WGSL). Reads Python baseline
//! JSON results and benchmarks the same workloads on Rust CPU and GPU.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | `e4358c5` |
//! | Baseline tool | Python numpy/scipy + Rust CPU + GPU |
//! | Baseline date | 2026-02-21 |
//! | Exact command | `cargo run --features gpu --release --bin benchmark_three_tier` |
//! | Data | Synthetic vectors matching Python baseline |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;

use wetspring_barracuda::bio::{
    diversity, diversity_gpu, pcoa, pcoa_gpu, spectral_match, spectral_match_gpu, stats_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation;

const WARMUP: usize = 3;

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

    let py_json = load_python_baseline();

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp069: Python → Rust CPU → GPU  Three-Tier Benchmark                    ║");
    println!("║  GPU: {:<68} ║", gpu.adapter_name);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    if py_json.is_some() {
        println!("  Python baseline loaded from benchmarks/results/python_baseline_latest.json");
    } else {
        println!("  No Python baseline — run scripts/benchmark_python_baseline.py first.");
        println!("  Showing Rust CPU vs GPU only.");
    }
    println!();

    let mut rows: Vec<Row> = Vec::new();

    bench_section(
        "SINGLE-VECTOR REDUCTIONS",
        &mut rows,
        &gpu,
        &py_json,
        |rows, gpu, py| {
            for &n in &[1_000, 10_000, 100_000, 1_000_000] {
                let data = gen_counts(n, 42);
                let cpu = time(|| {
                    let _ = diversity::shannon(&data);
                });
                let gpu_t = time(|| {
                    let _ = diversity_gpu::shannon_gpu(gpu, &data);
                });
                let py_t = py_lookup(py, &format!("Shannon entropy N={n}"));
                rows.push(Row::new("Shannon", n, py_t, cpu, gpu_t));
            }
            for &n in &[1_000, 10_000, 100_000, 1_000_000] {
                let data = gen_counts(n, 123);
                let cpu = time(|| {
                    let _ = diversity::simpson(&data);
                });
                let gpu_t = time(|| {
                    let _ = diversity_gpu::simpson_gpu(gpu, &data);
                });
                let py_t = py_lookup(py, &format!("Simpson diversity N={n}"));
                rows.push(Row::new("Simpson", n, py_t, cpu, gpu_t));
            }
            for &n in &[1_000, 10_000, 100_000, 1_000_000] {
                let data = gen_f64(n, 7);
                let cpu = time(|| {
                    let mean = data.iter().sum::<f64>() / data.len() as f64;
                    let _: f64 =
                        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                });
                let gpu_t = time(|| {
                    let _ = stats_gpu::variance_gpu(gpu, &data);
                });
                let py_t = py_lookup(py, &format!("Variance N={n}"));
                rows.push(Row::new("Variance", n, py_t, cpu, gpu_t));
            }
        },
    );

    bench_section(
        "PAIRWISE N×N WORKLOADS",
        &mut rows,
        &gpu,
        &py_json,
        |rows, gpu, py| {
            for &(ns, nsp) in &[(10, 500), (20, 500), (50, 500), (100, 500)] {
                let samples: Vec<Vec<f64>> =
                    (0..ns).map(|i| gen_counts(nsp, 42 + i as u64)).collect();
                let np = ns * (ns - 1) / 2;
                let cpu = time(|| {
                    let _ = diversity::bray_curtis_condensed(&samples);
                });
                let gpu_t = time(|| {
                    let _ = diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples);
                });
                let py_t = py_lookup(py, &format!("Bray-Curtis {ns}x{ns} N={np}"));
                rows.push(Row::new(&format!("B-C {ns}×{ns}"), np, py_t, cpu, gpu_t));
            }
            for &(nsp, dim) in &[(10, 500), (50, 500), (100, 500), (200, 500)] {
                let gpu_spectra: Vec<Vec<f64>> =
                    (0..nsp).map(|i| gen_f64(dim, 300 + i as u64)).collect();
                let cpu_spectra: Vec<(Vec<f64>, Vec<f64>)> = gpu_spectra
                    .iter()
                    .map(|int| {
                        let mzs: Vec<f64> =
                            (0..dim).map(|j| (j as f64).mul_add(0.5, 100.0)).collect();
                        (mzs, int.clone())
                    })
                    .collect();
                let np = nsp * (nsp - 1) / 2;
                let cpu = time(|| {
                    let _ = spectral_match::pairwise_cosine(&cpu_spectra, 0.5);
                });
                let gpu_t = time(|| {
                    let _ = spectral_match_gpu::pairwise_cosine_gpu(gpu, &gpu_spectra);
                });
                let py_t = py_lookup(py, &format!("Cosine {nsp}x{nsp} N={np}"));
                rows.push(Row::new(&format!("Cos {nsp}×{nsp}"), np, py_t, cpu, gpu_t));
            }
        },
    );

    bench_section(
        "MATRIX ALGEBRA",
        &mut rows,
        &gpu,
        &py_json,
        |rows, gpu, py| {
            for &ns in &[10, 20, 30] {
                let samples: Vec<Vec<f64>> =
                    (0..ns).map(|i| gen_counts(200, 100 + i as u64)).collect();
                let distances = diversity::bray_curtis_condensed(&samples);
                let cpu = time(|| {
                    let _ = pcoa::pcoa(&distances, ns, 3);
                });
                let gpu_t = time(|| {
                    let _ = pcoa_gpu::pcoa_gpu(gpu, &distances, ns, 3);
                });
                let py_t = py_lookup(py, &format!("PCoA {ns}x{ns} N={ns}"));
                rows.push(Row::new(&format!("PCoA {ns}×{ns}"), ns, py_t, cpu, gpu_t));
            }
        },
    );

    // Summary
    println!();
    println!("  ═══════════════════════════════════════════════════════════════════════");
    println!("  SUMMARY: Value Chain Evidence");
    println!("  ═══════════════════════════════════════════════════════════════════════");

    let (mut py_wins, mut rust_wins, mut gpu_wins) = (0, 0, 0);
    for r in &rows {
        if let Some(py) = r.python_us {
            if r.cpu_us < py {
                rust_wins += 1;
            } else {
                py_wins += 1;
            }
        }
        if r.gpu_us < r.cpu_us {
            gpu_wins += 1;
        }
    }
    let total = rows.len();
    let with_py = rows.iter().filter(|r| r.python_us.is_some()).count();

    if with_py > 0 {
        println!("  Python vs Rust CPU: Rust faster in {rust_wins}/{with_py} workloads");
    }
    println!("  Rust CPU vs GPU:    GPU faster in {gpu_wins}/{total} workloads");
    println!();
    println!("  Write → Absorb → Lean: Each tier builds on the previous.");
    println!("  Python validates the math. Rust CPU makes it fast. GPU makes it parallel.");
}

struct Row {
    label: String,
    n: usize,
    python_us: Option<f64>,
    cpu_us: f64,
    gpu_us: f64,
}

impl Row {
    fn new(label: &str, n: usize, python_us: Option<f64>, cpu_us: f64, gpu_us: f64) -> Self {
        Self {
            label: label.to_string(),
            n,
            python_us,
            cpu_us,
            gpu_us,
        }
    }
}

fn bench_section<F>(
    title: &str,
    rows: &mut Vec<Row>,
    gpu: &GpuF64,
    py: &Option<serde_json::Value>,
    f: F,
) where
    F: FnOnce(&mut Vec<Row>, &GpuF64, &Option<serde_json::Value>),
{
    let before = rows.len();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ {title:<75} │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ {:<18} {:>7} {:>11} {:>11} {:>11} {:>7} {:>7}│",
        "Workload", "N", "Python", "CPU", "GPU", "Py→CPU", "CPU→GPU"
    );
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    f(rows, gpu, py);

    for r in &rows[before..] {
        let py_str = r.python_us.map_or("—".to_string(), |v| fmt_us(v));
        let py_cpu = r.python_us.map_or("—".into(), |py_v| {
            if r.cpu_us > 0.01 {
                format!("{:.0}×", py_v / r.cpu_us)
            } else {
                "—".into()
            }
        });
        let cpu_gpu = if r.gpu_us > 0.01 {
            let ratio = r.cpu_us / r.gpu_us;
            if ratio >= 1.0 {
                format!("{ratio:.1}×▲")
            } else {
                format!("{ratio:.2}×▼")
            }
        } else {
            "—".into()
        };
        println!(
            "│ {:<18} {:>7} {:>11} {:>11} {:>11} {:>7} {:>7}│",
            r.label,
            r.n,
            py_str,
            fmt_us(r.cpu_us),
            fmt_us(r.gpu_us),
            py_cpu,
            cpu_gpu
        );
    }
}

fn fmt_us(us: f64) -> String {
    if us < 1.0 {
        format!("{:.0}ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{us:.1}µs")
    } else {
        format!("{:.2}ms", us / 1000.0)
    }
}

fn time<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
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

fn gen_counts(n: usize, seed: u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut rng = seed;
    for _ in 0..n {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        v.push(((rng >> 33) % 1000 + 1) as f64);
    }
    v
}

fn gen_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut rng = seed;
    for _ in 0..n {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        v.push((rng >> 11) as f64 / (1_u64 << 53) as f64);
    }
    v
}

fn load_python_baseline() -> Option<serde_json::Value> {
    let path = format!(
        "{}/../benchmarks/results/python_baseline_latest.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

fn py_lookup(py: &Option<serde_json::Value>, phase_name: &str) -> Option<f64> {
    let phases = py.as_ref()?.get("phases")?.as_array()?;
    for p in phases {
        if let Some(name) = p.get("phase").and_then(|v| v.as_str()) {
            if name == phase_name {
                return p.get("per_eval_us").and_then(|v| v.as_f64());
            }
        }
    }
    None
}
