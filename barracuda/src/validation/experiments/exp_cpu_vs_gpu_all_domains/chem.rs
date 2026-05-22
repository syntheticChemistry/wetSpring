// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chemistry/spatial domain CPU↔GPU parity: Spectral Cosine, EIC, PCoA,
//! Kriging (D12-D15).

use std::time::Instant;

use crate::bio::{eic, eic_gpu, kriging, pcoa, pcoa_gpu, spectral_match_gpu};
use crate::gpu::GpuF64;
use crate::io::mzml::MzmlSpectrum;
use crate::special;
use crate::tolerances;
use crate::validation::{CpuGpuRow, OrExit, Validator};

pub(super) fn validate_spectral_cosine(
    v: &mut Validator,
    gpu: &GpuF64,
    timings: &mut Vec<CpuGpuRow>,
) {
    v.section("D12: Spectral Cosine");
    let spectra: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.5, 0.2, 0.0, 0.8, 0.0, 0.3],
        vec![0.9, 0.1, 0.4, 0.3, 0.0, 0.7, 0.0, 0.2],
        vec![0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.6, 0.0],
    ];
    let n = spectra.len();
    let tc = Instant::now();
    let mut cpu_cos = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = special::dot(&spectra[i], &spectra[j]);
            let na: f64 = special::l2_norm(&spectra[i]);
            let nb: f64 = special::l2_norm(&spectra[j]);
            cpu_cos.push(if na > 0.0 && nb > 0.0 {
                dot / (na * nb)
            } else {
                0.0
            });
        }
    }
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_cos =
        spectral_match_gpu::batch_cosine_gpu(gpu, &spectra).or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    for (i, (c, g)) in cpu_cos.iter().zip(gpu_cos.iter()).enumerate() {
        v.check(
            &format!("cosine pair {i}"),
            *g,
            *c,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }
    timings.push(CpuGpuRow {
        name: "Spectral Cosine",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_eic(v: &mut Validator, gpu: &GpuF64, timings: &mut Vec<CpuGpuRow>) {
    v.section("D13: EIC Extraction");
    let spectra = super::synthetic_spectra();
    let target_mz = 200.0;
    let ppm = 10.0;
    let tc = Instant::now();
    let cpu_eic = eic::extract_eic(&spectra, target_mz, ppm);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_eic = eic_gpu::extract_eic_gpu(gpu, &spectra, target_mz, ppm)
        .or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check(
        "EIC length",
        gpu_eic.len() as f64,
        cpu_eic.len() as f64,
        tolerances::EXACT,
    );
    for (i, (c, g)) in cpu_eic.iter().zip(gpu_eic.iter()).enumerate() {
        v.check(
            &format!("EIC[{i}] intensity"),
            g.intensity,
            c.intensity,
            tolerances::GPU_VS_CPU_TRANSCENDENTAL,
        );
    }
    timings.push(CpuGpuRow {
        name: "EIC Extraction",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_pcoa(v: &mut Validator, gpu: &GpuF64, timings: &mut Vec<CpuGpuRow>) {
    v.section("D14: PCoA");
    let dist = vec![
        0.0, 0.5, 0.8, 0.3, 0.5, 0.0, 0.6, 0.7, 0.8, 0.6, 0.0, 0.4, 0.3, 0.7, 0.4, 0.0,
    ];
    let tc = Instant::now();
    let cpu_pcoa = pcoa::pcoa(&dist, 4, 2);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_pcoa = pcoa_gpu::pcoa_gpu(gpu, &dist, 4, 2).or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check(
        "PCoA eigenvalue 1",
        gpu_pcoa.eigenvalues[0].abs(),
        cpu_pcoa.eigenvalues[0].abs(),
        tolerances::GPU_VS_CPU_F64,
    );
    v.check_pass("PCoA coords populated", !gpu_pcoa.coordinates.is_empty());
    timings.push(CpuGpuRow {
        name: "PCoA (4×4)",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

pub(super) fn validate_kriging(v: &mut Validator, gpu: &GpuF64, timings: &mut Vec<CpuGpuRow>) {
    v.section("D15: Ordinary Kriging");
    let points: Vec<kriging::KrigingPoint> = vec![
        kriging::KrigingPoint {
            x: 0.0,
            y: 0.0,
            value: 10.0,
        },
        kriging::KrigingPoint {
            x: 1.0,
            y: 0.0,
            value: 12.0,
        },
        kriging::KrigingPoint {
            x: 0.0,
            y: 1.0,
            value: 11.0,
        },
        kriging::KrigingPoint {
            x: 1.0,
            y: 1.0,
            value: 13.0,
        },
        kriging::KrigingPoint {
            x: 0.5,
            y: 0.5,
            value: 11.5,
        },
    ];
    let targets: Vec<(f64, f64)> = vec![(0.25, 0.25), (0.75, 0.75), (0.5, 0.0)];
    let variogram = kriging::Variogram {
        nugget: 0.0,
        sill: 1.0,
        range: 2.0,
    };
    let tc = Instant::now();
    let cpu_est = kriging::ordinary_kriging(&points, &targets, &variogram);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_est = kriging::ordinary_kriging_gpu(gpu, &points, &targets, &variogram)
        .or_exit("GPU/CPU validation");
    let gpu_us = tg.elapsed().as_micros() as f64;
    for (i, (c, g)) in cpu_est.iter().zip(gpu_est.iter()).enumerate() {
        v.check(
            &format!("kriging[{i}]"),
            *g,
            *c,
            tolerances::GPU_VS_CPU_F64,
        );
    }
    timings.push(CpuGpuRow {
        name: "Kriging (5→3)",
        cpu_us,
        gpu_us,
        status: "PASS",
    });
}

/// Synthetic mzML spectra for EIC/spectral tests.
pub(super) fn synthetic_spectra() -> Vec<MzmlSpectrum> {
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
                    1000.0 * f64::exp(-((rt - peak_rt).powi(2)) / (2.0 * 0.5_f64.powi(2)))
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
