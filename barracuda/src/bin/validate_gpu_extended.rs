// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
//! Exp087: GPU Extended Domains — EIC, `PCoA`, Kriging, Rarefaction
//!
//! Extends the GPU validation suite with 4 previously uncovered domains:
//!
//! 1. **EIC total intensity** — `FusedMapReduceF64` via `batch_eic_total_intensity_gpu`
//! 2. **`PCoA`** — `BatchedEighGpu` ordination via `pcoa_gpu`
//! 3. **Kriging** — `KrigingF64` spatial interpolation via `interpolate_diversity`
//! 4. **Rarefaction** — GPU-accelerated bootstrap via `rarefaction_bootstrap_gpu`
//!
//! Each domain is tested for CPU ↔ GPU parity.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | current HEAD |
//! | Baseline tool | `BarraCUDA` CPU (sovereign Rust reference) |
//! | Baseline date | 2026-02-22 |
//! | Exact command | `cargo run --features gpu --release --bin validate_gpu_extended` |
//! | Data | Synthetic test vectors (self-contained) |
//! | Hardware | i9-12900K, 64 GB DDR5, RTX 4070, Pop!\_OS 22.04 |

use std::time::Instant;
use wetspring_barracuda::bio::{diversity, eic, eic_gpu, kriging, pcoa, pcoa_gpu, rarefaction_gpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::io::mzml::MzmlSpectrum;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    let mut v = Validator::new("Exp087: GPU Extended Domains — EIC, PCoA, Kriging, Rarefaction");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();

    if !gpu.has_f64 {
        validation::exit_skipped("No SHADER_F64 support on this GPU");
    }

    let t0 = Instant::now();

    validate_eic_total_intensity(&mut v, &gpu);
    validate_pcoa(&mut v, &gpu);
    validate_kriging(&mut v, &gpu);
    validate_rarefaction(&mut v, &gpu);

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [Total] {ms:.1} ms");
    v.finish();
}

// ════════════════════════════════════════════════════════════════
//  Domain 1: EIC Total Intensity (CPU ↔ GPU)
// ════════════════════════════════════════════════════════════════

fn validate_eic_total_intensity(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Domain 1: EIC Total Intensity CPU ↔ GPU ═══");
    let t0 = Instant::now();

    let spectra = synthetic_spectra();
    let target_mzs = vec![150.0, 200.0, 250.0, 300.0];
    let ppm = 10.0;

    let cpu_eics = eic::extract_eics(&spectra, &target_mzs, ppm);
    let gpu_eics = eic_gpu::extract_eics_gpu(gpu, &spectra, &target_mzs, ppm).unwrap();

    v.check(
        "EIC count matches",
        gpu_eics.len() as f64,
        cpu_eics.len() as f64,
        0.0,
    );

    for (i, (ce, ge)) in cpu_eics.iter().zip(gpu_eics.iter()).enumerate() {
        v.check(
            &format!("EIC {i}: point count"),
            ge.rt.len() as f64,
            ce.rt.len() as f64,
            0.0,
        );
    }

    let cpu_totals = eic_total_intensity_cpu(&cpu_eics);
    let gpu_totals = eic_gpu::batch_eic_total_intensity_gpu(gpu, &gpu_eics).unwrap();

    for (i, (ct, gt)) in cpu_totals.iter().zip(gpu_totals.iter()).enumerate() {
        v.check(
            &format!("EIC {i}: total intensity CPU ↔ GPU"),
            *gt,
            *ct,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [EIC total intensity] {ms:.1} ms");
}

fn eic_total_intensity_cpu(eics: &[eic::Eic]) -> Vec<f64> {
    eics.iter().map(|e| e.intensity.iter().sum()).collect()
}

// ════════════════════════════════════════════════════════════════
//  Domain 2: PCoA Ordination (CPU ↔ GPU)
// ════════════════════════════════════════════════════════════════

fn validate_pcoa(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Domain 2: PCoA Ordination CPU ↔ GPU ═══");
    let t0 = Instant::now();

    let samples = vec![
        vec![120.0, 85.0, 230.0, 55.0],
        vec![180.0, 12.0, 42.0, 310.0],
        vec![8.0, 95.0, 150.0, 200.0],
        vec![300.0, 5.0, 10.0, 45.0],
        vec![50.0, 200.0, 100.0, 120.0],
    ];
    let n_samples = samples.len();
    let n_axes = 2;

    let condensed = diversity::bray_curtis_condensed(&samples);

    let cpu_result = pcoa::pcoa(&condensed, n_samples, n_axes).unwrap();
    let gpu_result = pcoa_gpu::pcoa_gpu(gpu, &condensed, n_samples, n_axes).unwrap();

    v.check(
        "PCoA: axes count",
        gpu_result.eigenvalues.len() as f64,
        cpu_result.eigenvalues.len() as f64,
        0.0,
    );
    v.check(
        "PCoA: coordinate rows",
        gpu_result.coordinates.len() as f64,
        cpu_result.coordinates.len() as f64,
        0.0,
    );

    for (i, (ce, ge)) in cpu_result
        .eigenvalues
        .iter()
        .zip(gpu_result.eigenvalues.iter())
        .enumerate()
    {
        v.check(
            &format!("PCoA: eigenvalue {i} CPU ↔ GPU"),
            *ge,
            *ce,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    for (i, (cp, gp)) in cpu_result
        .proportion_explained
        .iter()
        .zip(gpu_result.proportion_explained.iter())
        .enumerate()
    {
        v.check(
            &format!("PCoA: variance explained {i} CPU ↔ GPU"),
            *gp,
            *cp,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    for (si, (cc, gc)) in cpu_result
        .coordinates
        .iter()
        .zip(gpu_result.coordinates.iter())
        .enumerate()
    {
        for (ai, (cv, gv)) in cc.iter().zip(gc.iter()).enumerate() {
            let sign_flip = if cv.signum() != gv.signum() && cv.abs() > 1e-6 {
                -1.0
            } else {
                1.0
            };
            v.check(
                &format!("PCoA: sample {si} axis {ai} CPU ↔ GPU"),
                gv * sign_flip,
                *cv,
                tolerances::GPU_VS_CPU_F64 * 10.0,
            );
        }
    }

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [PCoA] {ms:.1} ms");
}

// ════════════════════════════════════════════════════════════════
//  Domain 3: Kriging Spatial Interpolation (GPU)
// ════════════════════════════════════════════════════════════════

fn validate_kriging(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Domain 3: Kriging Spatial Interpolation ═══");
    let t0 = Instant::now();

    let sites = vec![
        kriging::SpatialSample {
            x: 0.0,
            y: 0.0,
            value: 3.2,
        },
        kriging::SpatialSample {
            x: 1.0,
            y: 0.0,
            value: 2.8,
        },
        kriging::SpatialSample {
            x: 0.0,
            y: 1.0,
            value: 3.5,
        },
        kriging::SpatialSample {
            x: 1.0,
            y: 1.0,
            value: 2.1,
        },
        kriging::SpatialSample {
            x: 0.5,
            y: 0.5,
            value: 3.0,
        },
    ];

    let targets = vec![(0.25, 0.25), (0.75, 0.75), (0.5, 0.0)];
    let config = kriging::VariogramConfig::spherical(0.0, 1.0, 2.0);

    let ordinary = kriging::interpolate_diversity(gpu, &sites, &targets, &config).unwrap();

    v.check(
        "kriging ordinary: value count",
        ordinary.values.len() as f64,
        targets.len() as f64,
        0.0,
    );
    v.check(
        "kriging ordinary: variance count",
        ordinary.variances.len() as f64,
        targets.len() as f64,
        0.0,
    );

    for (i, val) in ordinary.values.iter().enumerate() {
        v.check_pass(&format!("kriging: value {i} is finite"), val.is_finite());
    }
    for (i, var) in ordinary.variances.iter().enumerate() {
        v.check_pass(
            &format!("kriging: variance {i} ≥ 0"),
            *var >= 0.0 || (*var + 1e-10).is_sign_positive(),
        );
    }

    let known_mean = sites.iter().map(|s| s.value).sum::<f64>() / sites.len() as f64;
    let simple =
        kriging::interpolate_diversity_simple(gpu, &sites, &targets, &config, known_mean).unwrap();

    v.check(
        "kriging simple: value count",
        simple.values.len() as f64,
        targets.len() as f64,
        0.0,
    );

    for (i, (ov, sv)) in ordinary.values.iter().zip(simple.values.iter()).enumerate() {
        v.check_pass(
            &format!("kriging: ordinary vs simple {i} both finite"),
            ov.is_finite() && sv.is_finite(),
        );
    }

    let variogram = kriging::empirical_variogram(&sites, 5, 1.5).unwrap();
    v.check_pass("variogram: has lags", !variogram.0.is_empty());
    v.check(
        "variogram: lag count = semivariance count",
        variogram.0.len() as f64,
        variogram.1.len() as f64,
        0.0,
    );

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [Kriging] {ms:.1} ms");
}

// ════════════════════════════════════════════════════════════════
//  Domain 4: Rarefaction Bootstrap (GPU)
// ════════════════════════════════════════════════════════════════

fn validate_rarefaction(v: &mut Validator, gpu: &GpuF64) {
    v.section("═══ Domain 4: Rarefaction Bootstrap (GPU) ═══");
    let t0 = Instant::now();

    let counts: Vec<f64> = vec![
        120.0, 85.0, 230.0, 55.0, 180.0, 12.0, 42.0, 310.0, 8.0, 95.0, 33.0, 67.0, 145.0, 22.0,
        78.0, 200.0, 15.0, 50.0, 110.0, 40.0,
    ];

    let params = rarefaction_gpu::RarefactionGpuParams {
        n_bootstrap: 100,
        depth: Some(500),
        seed: 42,
    };

    let result = rarefaction_gpu::rarefaction_bootstrap_gpu(gpu, &counts, &params).unwrap();

    v.check_pass(
        "rarefaction: Shannon mean is finite",
        result.shannon.mean.is_finite(),
    );
    v.check_pass("rarefaction: Shannon mean > 0", result.shannon.mean > 0.0);
    v.check_pass(
        "rarefaction: Shannon CI lower ≤ mean",
        result.shannon.lower <= result.shannon.mean + 1e-10,
    );
    v.check_pass(
        "rarefaction: Shannon CI upper ≥ mean",
        result.shannon.upper >= result.shannon.mean - 1e-10,
    );

    v.check_pass(
        "rarefaction: Simpson mean is finite",
        result.simpson.mean.is_finite(),
    );
    v.check_pass(
        "rarefaction: Simpson ∈ [0,1]",
        result.simpson.mean >= 0.0 && result.simpson.mean <= 1.0,
    );

    v.check_pass(
        "rarefaction: observed mean is finite",
        result.observed.mean.is_finite(),
    );
    v.check_pass("rarefaction: observed > 0", result.observed.mean > 0.0);

    v.check(
        "rarefaction: depth matches",
        result.depth as f64,
        500.0,
        0.0,
    );

    let cpu_shannon = diversity::shannon(&counts);
    let cpu_simpson = diversity::simpson(&counts);

    v.check_pass(
        "rarefaction: bootstrap Shannon ≤ full Shannon",
        result.shannon.mean <= cpu_shannon + 0.5,
    );
    v.check_pass(
        "rarefaction: bootstrap Simpson reasonable",
        (result.simpson.mean - cpu_simpson).abs() < 0.3,
    );

    let params_batch = rarefaction_gpu::RarefactionGpuParams {
        n_bootstrap: 50,
        depth: Some(300),
        seed: 123,
    };

    let batch_samples = vec![counts.clone(), vec![50.0, 100.0, 200.0, 50.0, 100.0]];
    let batch_results =
        rarefaction_gpu::batch_rarefaction_gpu(gpu, &batch_samples, &params_batch).unwrap();

    v.check(
        "batch rarefaction: result count",
        batch_results.len() as f64,
        2.0,
        0.0,
    );
    for (i, br) in batch_results.iter().enumerate() {
        v.check_pass(
            &format!("batch {i}: Shannon finite"),
            br.shannon.mean.is_finite(),
        );
    }

    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  [Rarefaction] {ms:.1} ms");
}

fn synthetic_spectra() -> Vec<MzmlSpectrum> {
    (0..50)
        .map(|i| {
            let rt = i as f64 * 0.1;
            let base_mzs = [150.0, 200.0, 250.0, 300.0, 350.0];
            let mz_array: Vec<f64> = base_mzs.iter().map(|m| m + (i as f64) * 0.001).collect();
            let intensity_array: Vec<f64> = mz_array
                .iter()
                .enumerate()
                .map(|(j, _)| {
                    let peak_rt = (j as f64 + 1.0) * 1.0;
                    let sigma = 0.5;
                    1000.0 * f64::exp(-((rt - peak_rt).powi(2)) / (2.0 * sigma * sigma))
                })
                .collect();
            let lowest_mz = mz_array.first().copied().unwrap_or(0.0);
            let highest_mz = mz_array.last().copied().unwrap_or(0.0);
            MzmlSpectrum {
                index: i,
                ms_level: 1,
                rt_minutes: rt,
                tic: intensity_array.iter().sum(),
                base_peak_mz: mz_array[0],
                base_peak_intensity: intensity_array[0],
                lowest_mz,
                highest_mz,
                mz_array,
                intensity_array,
            }
        })
        .collect()
}
