// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::items_after_statements,
    clippy::float_cmp
)]
//! Cross-spring modern benchmark — `ToadStool` primitive throughput.
//!
//! Benchmarks `BarraCuda` primitives evolved from all five Springs,
//! tracking cross-spring lineage and proving shared evolution benefits.

use std::time::Instant;

struct BenchRow {
    primitive: &'static str,
    origin: &'static str,
    n: usize,
    us_per_iter: f64,
    ops_per_sec: f64,
}

fn bench_n<F, T>(n: usize, mut f: F) -> (T, f64)
where
    F: FnMut() -> T,
{
    let t = Instant::now();
    let mut result = None;
    for _ in 0..n {
        result = Some(f());
    }
    let elapsed_ns = t.elapsed().as_nanos() as f64;
    let us_per_iter = elapsed_ns / 1000.0 / n as f64;
    (result.unwrap(), us_per_iter)
}

fn main() {
    let mut rows: Vec<BenchRow> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════
    // Bio primitives (wetSpring → ToadStool)
    // ═══════════════════════════════════════════════════════════════════
    let abundance_1k: Vec<f64> = (1..=1000).map(|i| f64::from(i % 50 + 1)).collect();
    let abundance_500: Vec<f64> = (1..=500).map(|i| f64::from(i % 30 + 1)).collect();
    let vec_a: Vec<f64> = (0..1000)
        .map(|i| (f64::from(i) * 0.3).sin().abs().mul_add(50.0, 1.0))
        .collect();
    let vec_b: Vec<f64> = (0..1000)
        .map(|i| (f64::from(i) * 0.31).sin().abs().mul_add(50.0, 1.0))
        .collect();

    let (_, us) = bench_n(1000, || barracuda::stats::shannon(&abundance_1k));
    rows.push(BenchRow {
        primitive: "Shannon entropy",
        origin: "wetSpring",
        n: 1000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(1000, || barracuda::stats::simpson(&abundance_1k));
    rows.push(BenchRow {
        primitive: "Simpson diversity",
        origin: "wetSpring",
        n: 1000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(1000, || barracuda::stats::bray_curtis(&vec_a, &vec_b));
    rows.push(BenchRow {
        primitive: "Bray-Curtis",
        origin: "wetSpring",
        n: 1000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(1000, || barracuda::stats::chao1(&abundance_500));
    rows.push(BenchRow {
        primitive: "Chao1",
        origin: "wetSpring",
        n: 1000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══════════════════════════════════════════════════════════════════
    // Precision math (hotSpring → ToadStool)
    // ═══════════════════════════════════════════════════════════════════
    let n_rows = 50_usize;
    let n_cols = 10_usize;
    let x_ridge: Vec<f64> = (0..n_rows * n_cols)
        .map(|i| {
            ((i / n_cols) as f64)
                .mul_add(0.1, (i % n_cols) as f64 * 0.05)
                .sin()
        })
        .collect();
    let y_ridge: Vec<f64> = (0..n_rows).map(|i| (i as f64 * 0.2).cos()).collect();

    let (_, us) = bench_n(100, || {
        barracuda::linalg::ridge_regression(&x_ridge, &y_ridge, n_rows, n_cols, 1, 0.1)
    });
    rows.push(BenchRow {
        primitive: "Ridge regression",
        origin: "hotSpring",
        n: 100,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let vec_10k: Vec<f64> = (0..10_000).map(|i| (f64::from(i) * 0.001).sin()).collect();
    let (_, us) = bench_n(10_000, || barracuda::stats::mean(&vec_10k));
    rows.push(BenchRow {
        primitive: "Mean",
        origin: "hotSpring",
        n: 10_000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══════════════════════════════════════════════════════════════════
    // ML/Stats (neuralSpring → ToadStool)
    // ═══════════════════════════════════════════════════════════════════
    let x_1k: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.1).collect();
    let y_1k: Vec<f64> = x_1k
        .iter()
        .map(|&xi| 0.01f64.mul_add(xi.sin(), 2.0f64.mul_add(xi, 3.0)))
        .collect();

    let (_, us) = bench_n(1000, || barracuda::stats::pearson_correlation(&x_1k, &y_1k));
    rows.push(BenchRow {
        primitive: "Pearson correlation",
        origin: "neuralSpring",
        n: 1000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let x_500: Vec<f64> = (0..500).map(f64::from).collect();
    let y_500: Vec<f64> = x_500.iter().map(|&xi| 3.0f64.mul_add(xi, 7.0)).collect();
    let (_, us) = bench_n(1000, || barracuda::stats::fit_linear(&x_500, &y_500));
    rows.push(BenchRow {
        primitive: "Linear fit",
        origin: "neuralSpring",
        n: 1000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══════════════════════════════════════════════════════════════════
    // Evolution (groundSpring → ToadStool)
    // ═══════════════════════════════════════════════════════════════════
    let jk_data: Vec<f64> = (0..200)
        .map(|i| 2.0 + (f64::from(i) * 0.01).sin())
        .collect();
    let (_, us) = bench_n(500, || barracuda::stats::jackknife_mean_variance(&jk_data));
    rows.push(BenchRow {
        primitive: "Jackknife",
        origin: "groundSpring",
        n: 500,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(10_000, || {
        barracuda::stats::kimura_fixation_prob(1000, 0.0, 0.01)
    });
    rows.push(BenchRow {
        primitive: "Kimura fixation",
        origin: "groundSpring",
        n: 10_000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══════════════════════════════════════════════════════════════════
    // General math
    // ═══════════════════════════════════════════════════════════════════
    let trap_x: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
    let trap_y: Vec<f64> = trap_x.iter().map(|x| x * x).collect();
    let (_, us) = bench_n(5000, || barracuda::numerical::trapz(&trap_y, &trap_x));
    rows.push(BenchRow {
        primitive: "Trapezoidal integration",
        origin: "hotSpring",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let erf_pts: Vec<f64> = (0..1000).map(|i| (f64::from(i) - 500.0) / 500.0).collect();
    let (_, us) = bench_n(5000, || {
        let mut acc = 0.0;
        for &x in &erf_pts {
            acc += barracuda::special::erf(x);
        }
        acc
    });
    rows.push(BenchRow {
        primitive: "Error function",
        origin: "hotSpring",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══════════════════════════════════════════════════════════════════
    // Output table
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       Cross-Spring Modern BarraCuda Benchmark — wetSpring          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<24} │ {:<11} │ {:>5} │ {:>9} │ {:>11} ║",
        "Primitive", "Origin", "N", "µs/iter", "ops/sec"
    );
    println!("╠════════════════════════╪═════════════╪═════╪═══════════╪═════════════╣");

    for r in &rows {
        let ops_fmt = format!("{:>11.0}", r.ops_per_sec);
        let us_fmt = if r.us_per_iter < 0.001 && r.us_per_iter > 0.0 {
            format!("{:>9.6}", r.us_per_iter)
        } else if r.us_per_iter < 0.01 {
            format!("{:>9.4}", r.us_per_iter)
        } else {
            format!("{:>9.2}", r.us_per_iter)
        };
        println!(
            "║ {:<24} │ {:<11} │ {:>5} │ {:>9} │ {:>11} ║",
            r.primitive, r.origin, r.n, us_fmt, ops_fmt
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
}
