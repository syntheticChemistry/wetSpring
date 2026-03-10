// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
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
    clippy::float_cmp,
    clippy::needless_raw_string_hashes
)]
//! # Exp320: Cross-Spring Modern Evolution Benchmark (V98+)
//!
//! Throughput benchmark across all 5 springs' contributions to barraCuda,
//! measuring CPU ops/sec and GPU dispatch latency. Tracks provenance
//! showing when and where each primitive evolved across springs.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-03-08 |
//! | barraCuda | v0.3.3 `a898dee` |
//! | toadStool | S130+ `bfe7977b` |
//! | coralReef | Iteration 10 `d29a734` |
//! | Command | `cargo run --features gpu --release --bin benchmark_cross_spring_evolution_v98` |

use std::time::Instant;

use barracuda::shaders::provenance::report::shader_count;
use barracuda::shaders::provenance::types::SpringDomain;
use barracuda::shaders::provenance::{cross_spring_shaders, shaders_consumed_by, shaders_from};
use wetspring_barracuda::tolerances;

struct BenchRow {
    primitive: &'static str,
    origin: &'static str,
    evolution: &'static str,
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

    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  Exp320: Cross-Spring Modern Evolution Benchmark (V98+)              ║");
    println!("║  barraCuda a898dee · toadStool S130+ · coralReef Iteration 10        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Provenance summary
    let total = shader_count();
    let cross = cross_spring_shaders().len();
    println!("  Shader registry: {total} total, {cross} cross-spring");
    for (name, domain) in [
        ("hotSpring", SpringDomain::HOT_SPRING),
        ("wetSpring", SpringDomain::WET_SPRING),
        ("neuralSpring", SpringDomain::NEURAL_SPRING),
        ("airSpring", SpringDomain::AIR_SPRING),
        ("groundSpring", SpringDomain::GROUND_SPRING),
    ] {
        let a = shaders_from(domain).len();
        let c = shaders_consumed_by(domain).len();
        println!("    {name:14}: authored {a:2}, consumes {c:2}");
    }
    println!();

    // ═══ wetSpring Bio Diversity (1k elements) ══════════════════════════
    println!("  ── wetSpring Bio Diversity (n=1000) ──");
    let abundance: Vec<f64> = (1..=1000).map(|i| f64::from(i % 50 + 1)).collect();
    let vec_a: Vec<f64> = (0..1000)
        .map(|i| (f64::from(i) * 0.3).sin().abs().mul_add(50.0, 1.0))
        .collect();
    let vec_b: Vec<f64> = (0..1000)
        .map(|i| (f64::from(i) * 0.31).sin().abs().mul_add(50.0, 1.0))
        .collect();

    let (_, us) = bench_n(5000, || barracuda::stats::shannon(&abundance));
    rows.push(BenchRow {
        primitive: "Shannon H'",
        origin: "wetSpring",
        evolution: "V6→S63 Write→S64 Absorb",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(5000, || barracuda::stats::simpson(&abundance));
    rows.push(BenchRow {
        primitive: "Simpson 1-D",
        origin: "wetSpring",
        evolution: "V6→S63 Write→S64 Absorb",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(5000, || barracuda::stats::bray_curtis(&vec_a, &vec_b));
    rows.push(BenchRow {
        primitive: "Bray-Curtis",
        origin: "wetSpring",
        evolution: "V6→S64→BrayCurtisF64",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(5000, || barracuda::stats::chao1(&abundance));
    rows.push(BenchRow {
        primitive: "Chao1",
        origin: "wetSpring",
        evolution: "V6→S64",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(5000, || barracuda::stats::pielou_evenness(&abundance));
    rows.push(BenchRow {
        primitive: "Pielou J'",
        origin: "wetSpring",
        evolution: "V6→S64",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══ hotSpring Precision + Spectral ═════════════════════════════════
    println!("  ── hotSpring Precision + Spectral ──");
    let erf_pts: Vec<f64> = (0..1000).map(|i| (f64::from(i) - 500.0) / 500.0).collect();

    let (_, us) = bench_n(5000, || {
        let mut acc = 0.0;
        for &x in &erf_pts {
            acc += barracuda::special::erf(x);
        }
        acc
    });
    rows.push(BenchRow {
        primitive: "erf (1k pts)",
        origin: "hotSpring",
        evolution: "S58→df64_core→all springs",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(10_000, || barracuda::stats::norm_cdf(1.96));
    rows.push(BenchRow {
        primitive: "norm_cdf",
        origin: "hotSpring",
        evolution: "S58→stats→all springs",
        n: 10_000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(100, || barracuda::spectral::anderson_3d(4, 4, 4, 2.0, 42));
    rows.push(BenchRow {
        primitive: "Anderson 3D 4³",
        origin: "hotSpring",
        evolution: "v0.6→spectral→groundSpring",
        n: 100,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let lattice = barracuda::spectral::anderson_3d(6, 6, 6, 2.0, 42);
    let (_, us) = bench_n(100, || barracuda::spectral::lanczos(&lattice, 50, 42));
    rows.push(BenchRow {
        primitive: "Lanczos 216×50",
        origin: "hotSpring",
        evolution: "v0.6→spectral→wetSpring",
        n: 100,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══ neuralSpring ML + Graph ════════════════════════════════════════
    println!("  ── neuralSpring ML + Graph ──");
    let n = 100_usize;
    let adj: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            if i != j && (i + j) % 3 == 0 { 1.0 } else { 0.0 }
        })
        .collect();

    let (_, us) = bench_n(100, || barracuda::linalg::graph_laplacian(&adj, n));
    rows.push(BenchRow {
        primitive: "Graph Laplacian 100²",
        origin: "neuralSpring",
        evolution: "V64→S72 ComputeDispatch",
        n: 100,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let x_1k: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.1).collect();
    let y_1k: Vec<f64> = x_1k.iter().map(|&xi| 2.0f64.mul_add(xi, 3.0)).collect();

    let (_, us) = bench_n(5000, || barracuda::stats::pearson_correlation(&x_1k, &y_1k));
    rows.push(BenchRow {
        primitive: "Pearson r (n=1k)",
        origin: "neuralSpring",
        evolution: "V64→stats→groundSpring",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(1000, || barracuda::stats::fit_linear(&x_1k, &y_1k));
    rows.push(BenchRow {
        primitive: "Linear fit (n=1k)",
        origin: "neuralSpring",
        evolution: "V64→stats",
        n: 1000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══ airSpring Hydrology ════════════════════════════════════════════
    println!("  ── airSpring Hydrology ──");

    let (_, us) = bench_n(10_000, || {
        barracuda::stats::hargreaves_et0(35.0, 32.0, 18.0)
    });
    rows.push(BenchRow {
        primitive: "Hargreaves ET₀",
        origin: "airSpring",
        evolution: "V039→S70→wetSpring soil",
        n: 10_000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(10_000, || {
        barracuda::stats::fao56_et0(21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 187)
    });
    rows.push(BenchRow {
        primitive: "FAO-56 PM ET₀",
        origin: "airSpring",
        evolution: "V039→S70→Richards PDE",
        n: 10_000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══ groundSpring Stats ═════════════════════════════════════════════
    println!("  ── groundSpring Stats ──");
    let data: Vec<f64> = (1..=1000).map(f64::from).collect();

    let (_, us) = bench_n(10_000, || barracuda::stats::mean(&data));
    rows.push(BenchRow {
        primitive: "Mean (n=1k)",
        origin: "groundSpring",
        evolution: "V73→stats→all springs",
        n: 10_000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let jk_data: Vec<f64> = (0..200)
        .map(|i| 2.0 + (f64::from(i) * 0.01).sin())
        .collect();
    let (_, us) = bench_n(500, || barracuda::stats::jackknife_mean_variance(&jk_data));
    rows.push(BenchRow {
        primitive: "Jackknife (n=200)",
        origin: "groundSpring",
        evolution: "V73→stats→wetSpring CI",
        n: 500,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(100, || {
        barracuda::stats::bootstrap_ci(
            &data,
            |d| d.iter().sum::<f64>() / d.len() as f64,
            1000,
            0.95,
            42,
        )
    });
    rows.push(BenchRow {
        primitive: "Bootstrap CI (B=1k)",
        origin: "groundSpring",
        evolution: "V73→stats→airSpring err",
        n: 100,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    let (_, us) = bench_n(10_000, || {
        barracuda::stats::kimura_fixation_prob(1000, 0.0, 0.01)
    });
    rows.push(BenchRow {
        primitive: "Kimura fixation",
        origin: "groundSpring",
        evolution: "V73→pop-gen→neuralSpring",
        n: 10_000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══ wetSpring NMF ══════════════════════════════════════════════════
    println!("  ── wetSpring NMF + Drug Repurposing ──");
    let nmf_data: Vec<f64> = (0..50 * 30)
        .map(|i| f64::from((i * 7 + 3) % 100) / 100.0 + 0.01)
        .collect();
    let nmf_config = barracuda::linalg::NmfConfig {
        rank: 5,
        max_iter: 200,
        tol: tolerances::NMF_CONVERGENCE_KL,
        objective: barracuda::linalg::NmfObjective::Euclidean,
        seed: 42,
    };

    let (_, us) = bench_n(10, || {
        barracuda::linalg::nmf(&nmf_data, 50, 30, &nmf_config)
    });
    rows.push(BenchRow {
        primitive: "NMF 50×30 k=5",
        origin: "wetSpring",
        evolution: "V6→S64→SparseGemmF64 S82",
        n: 10,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══ General math ═══════════════════════════════════════════════════
    println!("  ── General math (cross-spring) ──");
    let trap_x: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
    let trap_y: Vec<f64> = trap_x.iter().map(|x| x * x).collect();

    let (_, us) = bench_n(5000, || barracuda::numerical::trapz(&trap_y, &trap_x));
    rows.push(BenchRow {
        primitive: "Trapz (n=1k)",
        origin: "hotSpring",
        evolution: "S58→numerical→airSpring",
        n: 5000,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

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
        primitive: "Ridge 50×10",
        origin: "hotSpring",
        evolution: "S58→linalg→neuralSpring",
        n: 100,
        us_per_iter: us,
        ops_per_sec: 1_000_000.0 / us,
    });

    // ═══ GPU benchmarks (if available) ══════════════════════════════════
    #[cfg(feature = "gpu")]
    {
        println!("  ── GPU Diversity (wetSpring→FusedMapReduce) ──");

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        match rt.block_on(wetspring_barracuda::gpu::GpuF64::new()) {
            Ok(gpu) => {
                use wetspring_barracuda::bio::diversity_gpu;

                println!("    GPU: {} ({:?})", gpu.adapter_name, gpu.fp64_strategy());

                let community_1k: Vec<f64> = (1..=1000).map(|i| f64::from(i % 50 + 1)).collect();

                let t = Instant::now();
                let _ = diversity_gpu::shannon_gpu(&gpu, &community_1k);
                let warmup_ms = t.elapsed().as_secs_f64() * 1000.0;
                println!("    GPU warmup (first dispatch): {warmup_ms:.1} ms");

                let (_, us) = bench_n(100, || diversity_gpu::shannon_gpu(&gpu, &community_1k));
                rows.push(BenchRow {
                    primitive: "GPU Shannon (1k)",
                    origin: "wetSpring+hot",
                    evolution: "FusedMapReduceF64 → DF64",
                    n: 100,
                    us_per_iter: us,
                    ops_per_sec: 1_000_000.0 / us,
                });

                let (_, us) = bench_n(100, || diversity_gpu::simpson_gpu(&gpu, &community_1k));
                rows.push(BenchRow {
                    primitive: "GPU Simpson (1k)",
                    origin: "wetSpring+hot",
                    evolution: "FusedMapReduceF64 → DF64",
                    n: 100,
                    us_per_iter: us,
                    ops_per_sec: 1_000_000.0 / us,
                });

                let b_1k: Vec<f64> = (1..=1000).map(|i| f64::from((i + 7) % 50 + 1)).collect();
                let (_, us) = bench_n(50, || {
                    diversity_gpu::bray_curtis_condensed_gpu(
                        &gpu,
                        &[community_1k.clone(), b_1k.clone()],
                    )
                });
                rows.push(BenchRow {
                    primitive: "GPU BC (1k)",
                    origin: "wetSpring",
                    evolution: "BrayCurtisF64 → DF64",
                    n: 50,
                    us_per_iter: us,
                    ops_per_sec: 1_000_000.0 / us,
                });
            }
            Err(e) => {
                println!("    GPU not available: {e}");
            }
        }
    }

    // ═══ Output table ═══════════════════════════════════════════════════
    println!();
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║  Cross-Spring Evolution Benchmark V98+ — Pure Rust vs GPU Throughput                           ║"
    );
    println!(
        "║  barraCuda a898dee · toadStool S130+ · coralReef Iteration 10                                  ║"
    );
    println!(
        "╠══════════════════════════╤════════════════╤════════════════════════════╤═══════╤═════════╤══════╣"
    );
    println!(
        "║ Primitive                │ Origin         │ Evolution                  │     N │  µs/iter│M ops ║"
    );
    println!(
        "╠══════════════════════════╪════════════════╪════════════════════════════╪═══════╪═════════╪══════╣"
    );

    for r in &rows {
        let m_ops = r.ops_per_sec / 1_000_000.0;
        let us_fmt = if r.us_per_iter < 0.01 {
            format!("{:>8.4}", r.us_per_iter)
        } else if r.us_per_iter < 100.0 {
            format!("{:>8.2}", r.us_per_iter)
        } else {
            format!("{:>8.0}", r.us_per_iter)
        };
        let m_fmt = if m_ops < 0.001 {
            format!("{m_ops:>5.4}")
        } else if m_ops < 1.0 {
            format!("{m_ops:>5.3}")
        } else {
            format!("{m_ops:>5.1}")
        };
        println!(
            "║ {:<24} │ {:<14} │ {:<26} │ {:>5} │ {:>7} │{:>5} ║",
            r.primitive, r.origin, r.evolution, r.n, us_fmt, m_fmt
        );
    }

    println!(
        "╚══════════════════════════╧════════════════╧════════════════════════════╧═══════╧═════════╧══════╝"
    );

    // ═══ Cross-spring evolution narrative ════════════════════════════════
    println!();
    println!("Cross-spring shader evolution highlights:");
    println!("  hotSpring precision shaders → used by ALL springs:");
    println!("    df64_core.wgsl, df64_transcendental.wgsl → consumer GPU f64");
    println!("    Anderson spectral → wetSpring QS-disorder, groundSpring noise");
    println!();
    println!("  wetSpring bio shaders → used by neuralSpring + groundSpring:");
    println!("    fused_map_reduce_f64.wgsl → Shannon/Simpson GPU (n=1k in ~1ms)");
    println!("    smith_waterman_banded_f64.wgsl → neuralSpring sequence alignment");
    println!("    hmm_forward_f64.wgsl → neuralSpring HMM batch inference");
    println!();
    println!("  neuralSpring → used by wetSpring + hotSpring:");
    println!("    GemmF64, SparseGemmF64 → drug repurposing, lattice QCD");
    println!("    graph_laplacian → wetSpring community network analysis");
    println!();
    println!("  airSpring → used by wetSpring + groundSpring:");
    println!("    6 ET₀ methods → soil QS models, Richards PDE");
    println!();
    println!("  groundSpring → used by ALL springs:");
    println!("    13-tier tolerance architecture → validation backbone");
    println!("    Bootstrap, Jackknife → diversity CI, error metrics");
    println!();
}
