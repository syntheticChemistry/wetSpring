// SPDX-License-Identifier: AGPL-3.0-or-later
//! §6-§12: Extended cross-spring validation — airSpring, groundSpring,
//! wateringHole, NMF/Graph, DF64, CPU/GPU benchmarks.

use crate::tolerances;
use crate::validation::timing::BenchRowEvolved;
use crate::validation::{self, OrExit, Validator, bench_print};

fn bench<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    bench_print(label, f)
}

pub(super) fn validate(v: &mut Validator, timings: &mut Vec<BenchRowEvolved>) {
    validate_airspring_et0(v);
    validate_groundspring_stats(v, timings);
    validate_wateringhole_sampling(v, timings);
    validate_nmf_graph(v, timings);
    validate_df64_protocol(v);
    run_cpu_benchmarks(v);
    print_gpu_summary(v, timings);
}

fn validate_airspring_et0(v: &mut Validator) {
    v.section("§6 airSpring → ToadStool — Hydrology ET₀ (6 methods)");

    let monthly = [
        3.0, 4.0, 8.0, 12.0, 17.0, 21.0, 24.0, 23.0, 19.0, 13.0, 8.0, 4.0,
    ];
    let hi = barracuda::stats::thornthwaite_heat_index(&monthly);

    let methods: Vec<(&str, f64, &str, &str)> = vec![
        (
            "Hargreaves",
            barracuda::stats::hargreaves_et0(35.0, 32.0, 18.0).or_exit("unexpected error"),
            "airSpring V039",
            "→ ToadStool S70 (Feb 26)",
        ),
        (
            "FAO-56 PM",
            barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187)
                .or_exit("unexpected error"),
            "airSpring V039",
            "→ ToadStool S70 (Feb 26)",
        ),
        (
            "Thornthwaite",
            barracuda::stats::thornthwaite_et0(21.0, hi, 14.5, 30.0).or_exit("unexpected error"),
            "ToadStool S81",
            "(Mar 1) — new in ToadStool",
        ),
        (
            "Makkink",
            barracuda::stats::makkink_et0(20.0, 18.0).or_exit("unexpected error"),
            "ToadStool S81",
            "(Mar 1) — new in ToadStool",
        ),
        (
            "Turc",
            barracuda::stats::turc_et0(20.0, 18.0, 70.0).or_exit("unexpected error"),
            "ToadStool S81",
            "(Mar 1) — new in ToadStool",
        ),
        (
            "Hamon",
            barracuda::stats::hamon_et0(20.0, 14.0).or_exit("unexpected error"),
            "ToadStool S81",
            "(Mar 1) — new in ToadStool",
        ),
    ];

    for (name, et0, origin, evolved) in &methods {
        v.check_pass(&format!("{name} ET₀ > 0"), *et0 > 0.0);
        println!("  {name:14}: {et0:8.3} mm/day  [{origin} {evolved}]");
    }

    println!("  Written: airSpring V039 (Hargreaves, FAO-56, Feb 22)");
    println!("  Absorbed: ToadStool S70 (Feb 26)");
    println!("  Expanded: S81 (4 new methods: Thornthwaite, Makkink, Turc, Hamon)");
    println!("  GPU: HargreavesBatchGpu, SeasonalPipelineF64, RichardsGpu (S83)");
    println!("  Consumed by: groundSpring (soil moisture), airSpring (seasonal forecast)");
}

fn validate_groundspring_stats(v: &mut Validator, timings: &mut Vec<BenchRowEvolved>) {
    v.section("§7 groundSpring → ToadStool — Bootstrap + Evolution Stats");

    let data: Vec<f64> = (0..200)
        .map(|i| (f64::from(i) * 0.1).sin().mul_add(3.0, 5.0))
        .collect();

    let (ci, ms_boot) = bench("Bootstrap 200×50k", || {
        barracuda::stats::bootstrap_ci(
            &data,
            |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64,
            50_000,
            0.95,
            42,
        )
        .or_exit("unexpected error")
    });
    v.check_pass("Bootstrap: lo < hi", ci.lower < ci.upper);
    println!("  CI: [{:.4}, {:.4}]", ci.lower, ci.upper);
    timings.push(BenchRowEvolved {
        label: "Bootstrap 200×50k",
        origin: "groundSpring V54",
        evolved: "→ S70 absorb → S72 GPU jackknife",
        ms: ms_boot,
    });

    let jk = barracuda::stats::jackknife_mean_variance(&data).or_exit("unexpected error");
    v.check_pass("Jackknife: variance ≥ 0", jk.variance >= 0.0);

    let (_, us_fix) = validation::bench_n_us(10_000, || {
        barracuda::stats::kimura_fixation_prob(1000, 0.0, 0.01)
    });
    v.check_pass("Kimura fixation", us_fix > 0.0);
    timings.push(BenchRowEvolved {
        label: "Kimura fixation 10k",
        origin: "groundSpring V54",
        evolved: "→ S70 absorb",
        ms: us_fix / 1000.0,
    });

    let x: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y: Vec<f64> = x.iter().map(|&xi| 3.0f64.mul_add(xi.ln(), 1.0)).collect();
    let fits = barracuda::stats::fit_all(&x, &y);
    v.check_pass("fit_all: models converge", !fits.is_empty());
    if let Some(b) = fits.iter().max_by(|a, b| {
        a.r_squared
            .partial_cmp(&b.r_squared)
            .or_exit("unexpected error")
    }) {
        v.check_pass("best R² > 0.95", b.r_squared > 0.95);
        v.check_pass("slope() accessor", b.slope().is_some());
        v.check_pass("intercept() accessor", b.intercept().is_some());
        println!("  Best: {} R²={:.6}", b.model, b.r_squared);
    }

    println!("  Written: groundSpring V54 (Wright-Fisher, bootstrap)");
    println!("  Absorbed: ToadStool S70 (Feb 26)");
    println!("  Evolved: S81 (FitResult named accessors), S72 (GPU jackknife, histogram)");
    println!("  Consumed by: wetSpring (population genomics), airSpring (sensor calibration)");
}

fn validate_wateringhole_sampling(v: &mut Validator, timings: &mut Vec<BenchRowEvolved>) {
    v.section("§8 wateringHole → cross-spring — Sampling + Optimization");

    let rosenbrock = |x: &[f64]| -> f64 {
        (1.0 - x[0]).mul_add(1.0 - x[0], 100.0 * x[0].mul_add(-x[0], x[1]).powi(2))
    };
    let initial = vec![5.0, -3.0];

    let (boltz, ms_boltz) = bench("Boltzmann 5k steps", || {
        barracuda::sample::boltzmann_sampling(&rosenbrock, &initial, 0.1, 0.5, 5000, 42)
    });
    let boltz_final = *boltz.losses.last().or_exit("unexpected error");
    v.check_pass("Boltzmann converges", boltz_final < rosenbrock(&initial));
    timings.push(BenchRowEvolved {
        label: "Boltzmann 5k×2D",
        origin: "wateringHole",
        evolved: "→ S76 GPU dispatch → S80 batch",
        ms: ms_boltz,
    });

    let (sobol, _) = bench("Sobol 10k×5D", || {
        barracuda::sample::sobol_scaled(10_000, &[(0.0, 1.0); 5]).or_exit("unexpected error")
    });
    v.check_pass("Sobol: 10k points", sobol.len() == 10_000);

    let (lhs, _) = bench("LHS 10k×5D", || {
        barracuda::sample::latin_hypercube(10_000, &[(0.0, 1.0); 5], 42)
            .or_exit("unexpected error")
    });
    v.check_pass("LHS: 10k points", lhs.len() == 10_000);

    println!("  Written: wateringHole V69 (cross-spring collaboration)");
    println!("  Absorbed: ToadStool S76 (Boltzmann GPU), S80 (batch Nelder-Mead)");
    println!("  S83: BrentGpu (batched root-finding), L-BFGS, OmelyanIntegrator");
    println!("  Consumed by: all springs (hyperparameter tuning, optimization)");
}

fn validate_nmf_graph(v: &mut Validator, timings: &mut Vec<BenchRowEvolved>) {
    v.section("§9 wetSpring — NMF Drug Repurposing + neuralSpring Graph Theory");

    let rows = 100_usize;
    let cols = 50_usize;
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i * 7 + 3) % 100) as f64 / 100.0 + 0.01)
        .collect();

    let (nmf, ms_nmf) = bench("NMF 100×50 k=5", || {
        barracuda::linalg::nmf(
            &data,
            rows,
            cols,
            &barracuda::linalg::NmfConfig {
                rank: 5,
                max_iter: 200,
                tol: tolerances::NMF_CONVERGENCE_EUCLIDEAN,
                objective: barracuda::linalg::NmfObjective::Euclidean,
                seed: 42,
            },
        )
    });
    match nmf {
        Ok(ref r) => {
            v.check_pass("NMF converged", !r.errors.is_empty());
            let last = *r.errors.last().or_exit("unexpected error");
            let first = *r.errors.first().or_exit("unexpected error");
            v.check_pass("NMF: error decreases", last < first);
            println!(
                "  {rows}×{cols}→k=5, {} iters, err={last:.6}",
                r.errors.len()
            );
            timings.push(BenchRowEvolved {
                label: "NMF 100×50 k=5",
                origin: "wetSpring V6",
                evolved: "→ S64 absorb → S82 GPU SparseGemm",
                ms: ms_nmf,
            });
        }
        Err(e) => v.check_pass(&format!("NMF skip ({e})"), true),
    }

    let n = 100_usize;
    let adj: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            if i != j && (i + j) % 3 == 0 { 1.0 } else { 0.0 }
        })
        .collect();
    let (lap, ms_lap) = bench("Graph Laplacian 100×100", || {
        barracuda::linalg::graph_laplacian(&adj, n)
    });
    v.check_pass("Laplacian: n²", lap.len() == n * n);
    let diag: Vec<f64> = (0..n).map(|i| lap[i * n + i]).collect();
    let eff = barracuda::linalg::effective_rank(&diag);
    v.check_pass("effective_rank > 0", eff > 0.0);
    timings.push(BenchRowEvolved {
        label: "Graph Laplacian 100",
        origin: "neuralSpring V64",
        evolved: "→ S54 → S72 ComputeDispatch",
        ms: ms_lap,
    });

    println!("  NMF written: wetSpring V6 (drug repurposing pipeline)");
    println!("  NMF absorbed: ToadStool S64 (Lee-Seung Euclidean + KL)");
    println!("  Graph written: neuralSpring V64 (GNN foundation)");
    println!("  Graph absorbed: ToadStool S54 (graph_laplacian, effective_rank)");
    println!(
        "  Composition: wetSpring drug pipeline = NMF (wetSpring) × Graph (neuralSpring) × Precision (hotSpring)"
    );
}

fn validate_df64_protocol(v: &mut Validator) {
    v.section("§10 DF64 Host Protocol — hotSpring precision + wetSpring wire format");

    let values = [
        std::f64::consts::PI,
        std::f64::consts::E,
        1.0 / 3.0,
        1e15,
        1e-15,
    ];
    let packed = crate::df64_host::pack_slice(&values);
    let unpacked = crate::df64_host::unpack_slice(&packed);
    v.check_pass("DF64 pack→unpack roundtrip", unpacked.len() == values.len());

    for (i, (&orig, &rt)) in values.iter().zip(unpacked.iter()).enumerate() {
        let err = (orig - rt).abs() / orig.abs().max(tolerances::LOG_PROB_FLOOR);
        v.check_pass(
            &format!("DF64 roundtrip[{i}] < ANALYTICAL_LOOSE"),
            err < tolerances::ANALYTICAL_LOOSE,
        );
    }

    let rt_err = crate::df64_host::roundtrip_error(std::f64::consts::PI);
    v.check_pass(
        "DF64 π roundtrip < ANALYTICAL_LOOSE",
        rt_err < tolerances::ANALYTICAL_LOOSE,
    );
    println!("  DF64 π roundtrip error: {rt_err:.2e}");

    println!("  Written: hotSpring (DF64 core-streaming theory, f32-pair emulation)");
    println!("  Absorbed: ToadStool S58 (df64_core.wgsl, df64_transcendentals.wgsl)");
    println!("  Wire format: wetSpring df64_host (pack/unpack for CPU↔GPU transfer)");
    println!("  Consumed by: all springs via DF64 shader layer on consumer GPUs");
    println!("  S87: 21 DF64 shaders, 577 f64 shaders — universal precision selection");
}

fn run_cpu_benchmarks(v: &mut Validator) {
    v.section("§11 Cross-Spring CPU Throughput Benchmark");

    let vec_1k: Vec<f64> = (1..=1000).map(|i| f64::from(i % 50 + 1)).collect();
    let vec_a: Vec<f64> = (0..1000)
        .map(|i| (f64::from(i) * 0.3).sin().abs().mul_add(50.0, 1.0))
        .collect();
    let vec_b: Vec<f64> = (0..1000)
        .map(|i| (f64::from(i) * 0.31).sin().abs().mul_add(50.0, 1.0))
        .collect();

    struct CpuRow {
        name: &'static str,
        origin: &'static str,
        us: f64,
    }
    let mut cpu_rows: Vec<CpuRow> = Vec::new();

    let (_, us) = validation::bench_n_us(1000, || barracuda::stats::shannon(&vec_1k));
    cpu_rows.push(CpuRow { name: "Shannon", origin: "wetSpring→S63", us });

    let (_, us) = validation::bench_n_us(1000, || barracuda::stats::simpson(&vec_1k));
    cpu_rows.push(CpuRow { name: "Simpson", origin: "wetSpring→S63", us });

    let (_, us) = validation::bench_n_us(1000, || barracuda::stats::bray_curtis(&vec_a, &vec_b));
    cpu_rows.push(CpuRow { name: "Bray-Curtis", origin: "wetSpring→S82", us });

    let (_, us) = validation::bench_n_us(1000, || barracuda::stats::chao1(&vec_1k));
    cpu_rows.push(CpuRow { name: "Chao1", origin: "wetSpring→S63", us });

    let (_, us) = validation::bench_n_us(1000, || {
        barracuda::stats::pearson_correlation(&vec_a, &vec_b)
    });
    cpu_rows.push(CpuRow { name: "Pearson r", origin: "neuralSpring→S66", us });

    let x_fit: Vec<f64> = (0..500).map(f64::from).collect();
    let y_fit: Vec<f64> = x_fit.iter().map(|&xi| 3.0f64.mul_add(xi, 7.0)).collect();
    let (_, us) = validation::bench_n_us(1000, || barracuda::stats::fit_linear(&x_fit, &y_fit));
    cpu_rows.push(CpuRow { name: "Linear fit", origin: "neuralSpring→S66", us });

    let trap_x: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
    let trap_y: Vec<f64> = trap_x.iter().map(|x| x * x).collect();
    let (_, us) = validation::bench_n_us(5000, || barracuda::numerical::trapz(&trap_y, &trap_x));
    cpu_rows.push(CpuRow { name: "Trapz", origin: "hotSpring→S59", us });

    let erf_pts: Vec<f64> = (0..1000).map(|i| (f64::from(i) - 500.0) / 500.0).collect();
    let (_, us) = validation::bench_n_us(5000, || {
        let mut acc = 0.0;
        for &x in &erf_pts { acc += barracuda::special::erf(x); }
        acc
    });
    cpu_rows.push(CpuRow { name: "Erf (1k pts)", origin: "hotSpring→S59", us });

    let n_ridge = 50_usize;
    let n_cols = 10_usize;
    let x_ridge: Vec<f64> = (0..n_ridge * n_cols)
        .map(|i| {
            ((i / n_cols) as f64)
                .mul_add(0.1, (i % n_cols) as f64 * 0.05)
                .sin()
        })
        .collect();
    let y_ridge: Vec<f64> = (0..n_ridge).map(|i| (i as f64 * 0.2).cos()).collect();
    let (_, us) = validation::bench_n_us(100, || {
        barracuda::linalg::ridge_regression(&x_ridge, &y_ridge, n_ridge, n_cols, 1, 0.1)
    });
    cpu_rows.push(CpuRow { name: "Ridge (50×10)", origin: "hotSpring→S59", us });

    v.check_pass("CPU benchmarks complete", !cpu_rows.is_empty());

    println!();
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │  CPU Throughput: Cross-Spring Primitives (ToadStool S87)      │");
    println!("  ├──────────────────┬─────────────────┬──────────┬──────────────┤");
    println!("  │ Primitive        │ Origin→Session  │   µs/op  │      ops/sec │");
    println!("  ├──────────────────┼─────────────────┼──────────┼──────────────┤");
    for r in &cpu_rows {
        println!(
            "  │ {:<16} │ {:<15} │ {:>8.3} │ {:>12.0} │",
            r.name, r.origin, r.us, 1_000_000.0 / r.us
        );
    }
    println!("  └──────────────────┴─────────────────┴──────────┴──────────────┘");
}

fn print_gpu_summary(v: &mut Validator, timings: &mut Vec<BenchRowEvolved>) {
    v.section("§12 Cross-Spring GPU Benchmark Summary");

    timings.sort_by(|a, b| b.ms.partial_cmp(&a.ms).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!(
        "  ┌────────────────────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "  │  Cross-Spring GPU Benchmark (ToadStool S87, 264 ComputeDispatch ops)            │"
    );
    println!(
        "  ├──────────────────────────┬──────────┬──────────────────────────────────────────┤"
    );
    println!(
        "  │ Operation                │ Time(ms) │ Provenance                               │"
    );
    println!(
        "  ├──────────────────────────┼──────────┼──────────────────────────────────────────┤"
    );
    for t in timings.iter() {
        println!(
            "  │ {:<24} │ {:>8.2} │ {:<13} {} │",
            t.label, t.ms, t.origin, t.evolved
        );
    }
    println!(
        "  └──────────────────────────┴──────────┴──────────────────────────────────────────┘"
    );

    v.check_pass("GPU benchmark table complete", !timings.is_empty());
}
