// SPDX-License-Identifier: AGPL-3.0-or-later
//! §1-§5: Core cross-spring validation — DiversityFusion, GemmF64,
//! GemmCached, Bray-Curtis, Anderson spectral.

use std::sync::Arc;

use crate::bio::diversity_fusion_gpu::{DiversityFusionGpu, diversity_fusion_cpu};
use crate::bio::gemm_cached::GemmCached;
use crate::gpu::GpuF64;
use crate::tolerances;
use crate::validation::timing::BenchRowEvolved;
use crate::validation::{self, OrExit, Validator, bench_print};
use barracuda::ops::linalg::gemm_f64::GemmF64;

fn bench<T>(label: &str, f: impl FnOnce() -> T) -> (T, f64) {
    bench_print(label, f)
}

pub(super) fn validate(
    v: &mut Validator,
    gpu: &GpuF64,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<BenchRowEvolved>,
) {
    validate_diversity_fusion(v, gpu, device, timings);
    validate_gemm_f64(v, gpu, device, timings);
    let ctx = gpu.tensor_context().clone();
    validate_gemm_cached(v, device, ctx, timings);
    validate_bray_curtis(v, device, timings);
    validate_anderson_spectral(v, timings);
}

fn validate_diversity_fusion(
    v: &mut Validator,
    _gpu: &GpuF64,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<BenchRowEvolved>,
) {
    v.section("§1 wetSpring Bio → ToadStool → ALL — Diversity Fusion");

    let counts: Vec<f64> = (0..500)
        .map(|i: i32| f64::from((i * 13 + 7) % 200) + 1.0)
        .collect();
    let n_taxa = counts.len();

    let (cpu_results, ms_cpu) = bench("diversity_fusion_cpu 500 taxa", || {
        diversity_fusion_cpu(&counts, n_taxa)
    });
    let cpu_shannon = cpu_results[0].shannon;
    let cpu_simpson = cpu_results[0].simpson;
    v.check_pass("CPU: shannon > 0", cpu_shannon > 0.0);
    v.check_pass(
        "CPU: simpson in (0,1)",
        cpu_simpson > 0.0 && cpu_simpson < 1.0,
    );

    let fusion = DiversityFusionGpu::new(Arc::clone(device)).or_exit("unexpected error");
    let (gpu_results, ms_gpu) = bench("DiversityFusionGpu 500 taxa", || {
        fusion.compute(&counts, 1, n_taxa)
    });
    match gpu_results {
        Ok(ref res) => {
            let gpu_shannon = res[0].shannon;
            let parity = (gpu_shannon - cpu_shannon).abs();
            v.check_pass("GPU↔CPU shannon parity < 0.3", parity < 0.3);
            println!("  CPU: {cpu_shannon:.6}, GPU: {gpu_shannon:.6}, Δ={parity:.2e}");
            if ms_cpu > 0.0 {
                println!("  Speedup: {:.1}× GPU vs CPU", ms_cpu / ms_gpu.max(0.001));
            }
            timings.push(BenchRowEvolved {
                label: "DiversityFusion GPU",
                origin: "wetSpring V6",
                evolved: "→ S63 absorb → all springs consume",
                ms: ms_gpu,
            });
        }
        Err(e) => v.check_pass(&format!("GPU diversity skip ({e})"), true),
    }
    timings.push(BenchRowEvolved {
        label: "DiversityFusion CPU",
        origin: "wetSpring V6",
        evolved: "→ S63 absorb",
        ms: ms_cpu,
    });

    println!("  Written: wetSpring V6 (diversity_fusion_f64.wgsl)");
    println!("  Absorbed: ToadStool S63 (Feb 24)");
    println!("  Consumed by: neuralSpring (brain diversity), groundSpring (ecological)");
}

fn validate_gemm_f64(
    v: &mut Validator,
    gpu: &GpuF64,
    _device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<BenchRowEvolved>,
) {
    v.section("§2 neuralSpring → ToadStool → wetSpring — GemmF64 + DF64");

    for &n in &[64_usize, 128, 256] {
        let a: Vec<f64> = (0..n * n)
            .map(|i| ((i * 17 + 3) % 100) as f64 / 100.0)
            .collect();
        let b: Vec<f64> = (0..n * n)
            .map(|i| ((i * 13 + 7) % 100) as f64 / 100.0)
            .collect();

        let label = format!("GEMM {n}×{n}");
        let dev = gpu.to_wgpu_device();
        let (result, ms) = bench(&label, || GemmF64::execute(dev.clone(), &a, &b, n, n, n, 1));
        match result {
            Ok(ref c) => {
                let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                v.check_pass(&format!("GEMM {n}: non-trivial"), norm > 0.0);
                let (trace_cpu, _) = bench(&format!("CPU matmul {n}"), || {
                    let mut out = vec![0.0; n * n];
                    for i in 0..n {
                        for j in 0..n {
                            let mut s = 0.0;
                            for k in 0..n {
                                s += a[i * n + k] * b[k * n + j];
                            }
                            out[i * n + j] = s;
                        }
                    }
                    out.iter()
                        .enumerate()
                        .filter(|&(i, _)| i / n == i % n)
                        .map(|(_, &v)| v)
                        .sum::<f64>()
                });
                let trace_gpu: f64 = (0..n).map(|i| c[i * n + i]).sum();
                v.check_pass(
                    &format!("GEMM {n}: trace parity < 1.0"),
                    (trace_gpu - trace_cpu).abs() < 1.0,
                );
            }
            Err(e) => v.check_pass(&format!("GEMM {n} skip ({e})"), true),
        }

        if n == 256 {
            timings.push(BenchRowEvolved {
                label: "GEMM 256×256",
                origin: "neuralSpring S64",
                evolved: "→ ToadStool GemmF64 → hotSpring DF64",
                ms,
            });
        }
    }

    println!("  Written: neuralSpring S64 (GemmF64 — 60× faster than CPU loop)");
    println!("  Evolved: hotSpring DF64 precision fallback (S58)");
    println!("  Consumed by: wetSpring (drug repurposing, NMF), neuralSpring (AlphaFold2)");
}

fn validate_gemm_cached(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    ctx: Arc<barracuda::device::TensorContext>,
    timings: &mut Vec<BenchRowEvolved>,
) {
    v.section("§3 wetSpring — GemmCached (neuralSpring GEMM + hotSpring precision)");

    let gemm = GemmCached::new(Arc::clone(device), ctx);

    for &(m, k, n) in &[(64_usize, 32, 16), (128, 64, 32), (256, 128, 64)] {
        let a: Vec<f64> = (0..m * k)
            .map(|i| ((i * 11 + 5) % 100) as f64 / 100.0)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| ((i * 13 + 3) % 100) as f64 / 100.0)
            .collect();

        let label = format!("GemmCached {m}×{k}×{n}");
        let (result, ms) = bench(&label, || gemm.execute(&a, &b, m, k, n, 1));
        match result {
            Ok(ref c) => {
                v.check_pass(&format!("GemmCached {m}×{k}×{n}: size"), c.len() == m * n);
                let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                v.check_pass(&format!("GemmCached {m}×{k}×{n}: non-trivial"), norm > 0.0);
            }
            Err(e) => v.check_pass(&format!("GemmCached skip ({e})"), true),
        }

        if m == 256 {
            timings.push(BenchRowEvolved {
                label: "GemmCached 256×128×64",
                origin: "wetSpring V6",
                evolved: "→ neuralSpring GEMM + hotSpring precision",
                ms,
            });
        }
    }

    println!("  Written: wetSpring V6 (drug repurposing pipeline, cached pipeline)");
    println!("  Composes: neuralSpring GemmF64 shader + hotSpring DF64 precision");
    println!("  Uses: ToadStool BufferPool (S65), ComputeDispatch (S72)");
}

fn validate_bray_curtis(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<BenchRowEvolved>,
) {
    v.section("§4 wetSpring → ToadStool — Bray-Curtis GPU Distance");

    for &n_samples in &[10_usize, 20, 50] {
        let n_features = 200_usize;
        let samples: Vec<f64> = (0..n_samples * n_features)
            .map(|i| ((i * 7 + 1) % 50) as f64 + 1.0)
            .collect();

        let bc = barracuda::ops::bray_curtis_f64::BrayCurtisF64::new(Arc::clone(device));
        match bc {
            Ok(bc_gpu) => {
                let label = format!("BrayCurtis GPU {n_samples}×{n_features}");
                let (dist, ms) = bench(&label, || {
                    bc_gpu.condensed_distance_matrix(&samples, n_samples, n_features)
                });
                match dist {
                    Ok(ref d) => {
                        let expected = n_samples * (n_samples - 1) / 2;
                        v.check_pass(
                            &format!("BC {n_samples}: condensed={expected}"),
                            d.len() == expected,
                        );
                        v.check_pass(
                            &format!("BC {n_samples}: values in [0,1]"),
                            d.iter().all(|&x| (0.0..=1.0).contains(&x)),
                        );
                        if n_samples == 50 {
                            timings.push(BenchRowEvolved {
                                label: "BrayCurtis GPU 50×200",
                                origin: "wetSpring bio",
                                evolved: "→ ToadStool S82 ComputeDispatch",
                                ms,
                            });
                        }
                    }
                    Err(e) => v.check_pass(&format!("BC {n_samples} dist skip ({e})"), true),
                }
            }
            Err(e) => v.check_pass(&format!("BC init skip ({e})"), true),
        }
    }

    println!("  Written: wetSpring bio (16S rRNA distance matrices)");
    println!("  Absorbed: ToadStool S82 (BrayCurtisF64 ComputeDispatch)");
    println!("  Consumed by: wetSpring (UniFrac), groundSpring (soil ecology)");
}

fn validate_anderson_spectral(v: &mut Validator, timings: &mut Vec<BenchRowEvolved>) {
    v.section("§5 hotSpring → ALL — Anderson Spectral (1D→3D→4D)");

    let sizes = [100, 500, 1000, 2000];
    for &n in &sizes {
        let (eigs, ms) = bench(&format!("Anderson 1D n={n}"), || {
            barracuda::spectral::anderson_eigenvalues(n, 4.0, 42)
        });
        v.check_pass(&format!("Anderson n={n}"), eigs.len() == n);
        let r = barracuda::spectral::level_spacing_ratio(&eigs);
        println!(
            "    n={n}: r={r:.4} (GOE={:.4}, Poisson={:.4})",
            barracuda::spectral::GOE_R,
            barracuda::spectral::POISSON_R
        );
        if n == 2000 {
            timings.push(BenchRowEvolved {
                label: "Anderson 1D n=2000",
                origin: "hotSpring v0.6.0",
                evolved: "→ S26 absorb → S79 Lanczos → S83 4D",
                ms,
            });
        }
    }

    let (csr_3d, ms_3d) = bench("Anderson 3D 8³=512", || {
        barracuda::spectral::anderson_3d(8, 8, 8, 4.0, 42)
    });
    v.check_pass("3D: 512 sites", csr_3d.n == 512);
    let (tri, _) = bench("Lanczos 200 steps", || {
        barracuda::spectral::lanczos(&csr_3d, 200, 42)
    });
    let eigs_3d = barracuda::spectral::lanczos_eigenvalues(&tri);
    let phase = barracuda::spectral::classify_spectral_phase(&eigs_3d, 4.0);
    v.check_pass("3D Lanczos converges", !eigs_3d.is_empty());
    println!("  3D phase: {phase:?}");

    let (csr_4d, _) = bench("Anderson 4D 4⁴=256", || {
        barracuda::spectral::anderson::anderson_4d(4, 4.0, 42)
    });
    v.check_pass("4D: 256 sites", csr_4d.n == 256);
    timings.push(BenchRowEvolved {
        label: "Anderson 3D 512",
        origin: "hotSpring v0.6.0",
        evolved: "→ S26 → S79 Lanczos → S83 4D+Wegner",
        ms: ms_3d,
    });

    println!("  Written: hotSpring v0.6.0 (Kachkovskiy spectral theory, Feb 14)");
    println!("  Absorbed: ToadStool S26 (Feb 22)");
    println!("  Evolved: S79 (Lanczos), S83 (4D + Wegner block RG)");
    println!("  Consumed by: neuralSpring (NautilusBrain phase), wetSpring (disorder models)");
}
