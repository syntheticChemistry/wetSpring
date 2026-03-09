// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp328: CPU vs GPU pure math parity for visualization-exercised domains.
//!
//! Runs the same scientific computations on CPU and GPU, compares results
//! within documented tolerances. Validates that `barraCuda` CPU math
//! matches `barraCuda` GPU math for all domains visualized by `petalTongue`.
//!
//! | Domain   | CPU Function              | GPU Function               |
//! |----------|---------------------------|----------------------------|
//! | G1 Div   | `diversity::shannon`      | `diversity_gpu::shannon_gpu` |
//! | G2 Div   | `diversity::simpson`      | `diversity_gpu::simpson_gpu` |
//! | G3 Div   | `diversity::observed`     | `diversity_gpu::observed_features_gpu` |
//! | G4 Div   | `diversity::pielou`       | `diversity_gpu::pielou_evenness_gpu` |
//! | G5 Beta  | `diversity::bray_curtis`  | `diversity_gpu::bray_curtis_condensed_gpu` |
//! | G6 Ord   | `pcoa::pcoa`              | `pcoa_gpu::pcoa_gpu`       |
//! | G7 KMD   | `kmd::kendrick_mass_defect` | `kmd_gpu::kendrick_mass_defect_gpu` |
//! | G8 ODE   | `ode::rk4_integrate`      | CPU reference (GPU ODE sweep for full parity) |
//!
//! Reference: Exp323 (CPU v25), Exp324 (GPU v14), Exp327 (Viz V1).
//! Tolerances from `tolerances::GPU_VS_CPU_F64` (1e-6).

use wetspring_barracuda::bio::{
    diversity, diversity_gpu, kmd, kmd_gpu, pcoa, pcoa_gpu, qs_biofilm,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::{self, Validator};

fn validate_alpha_diversity(v: &mut Validator, gpu: &GpuF64, counts: &[f64]) {
    v.section("G1 — Shannon H'");
    let cpu_val = diversity::shannon(counts);
    let gpu_val = diversity_gpu::shannon_gpu(gpu, counts).unwrap_or(f64::NAN);
    v.check(
        "Shannon CPU vs GPU",
        gpu_val,
        cpu_val,
        tolerances::GPU_VS_CPU_F64,
    );

    v.section("G2 — Simpson D");
    let cpu_val = diversity::simpson(counts);
    let gpu_val = diversity_gpu::simpson_gpu(gpu, counts).unwrap_or(f64::NAN);
    v.check(
        "Simpson CPU vs GPU",
        gpu_val,
        cpu_val,
        tolerances::GPU_VS_CPU_F64,
    );

    v.section("G3 — Observed Features");
    let cpu_val = diversity::observed_features(counts);
    let gpu_val = diversity_gpu::observed_features_gpu(gpu, counts).unwrap_or(f64::NAN);
    v.check(
        "Observed CPU vs GPU",
        gpu_val,
        cpu_val,
        tolerances::GPU_VS_CPU_F64,
    );

    v.section("G4 — Pielou Evenness");
    let cpu_val = diversity::pielou_evenness(counts);
    let gpu_val = diversity_gpu::pielou_evenness_gpu(gpu, counts).unwrap_or(f64::NAN);
    v.check(
        "Pielou CPU vs GPU",
        gpu_val,
        cpu_val,
        tolerances::GPU_VS_CPU_F64,
    );
}

fn validate_beta_and_ordination(v: &mut Validator, gpu: &GpuF64) {
    v.section("G5 — Bray-Curtis Matrix");
    let samples = vec![
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![15.0, 25.0, 5.0, 35.0, 45.0],
        vec![8.0, 12.0, 40.0, 20.0, 60.0],
    ];
    let bc_full = diversity::bray_curtis_matrix(&samples);
    let n = samples.len();
    let mut bc_cpu_condensed = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            bc_cpu_condensed.push(bc_full[i * n + j]);
        }
    }
    let bc_gpu = diversity_gpu::bray_curtis_condensed_gpu(gpu, &samples).unwrap_or_default();
    v.check_count("BC condensed length", bc_gpu.len(), bc_cpu_condensed.len());
    for (i, (cpu_v, gpu_v)) in bc_cpu_condensed.iter().zip(bc_gpu.iter()).enumerate() {
        v.check(
            &format!("BC[{i}] CPU vs GPU"),
            *gpu_v,
            *cpu_v,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    v.section("G6 — PCoA Ordination");
    let dm_condensed = vec![0.5, 0.8, 0.6];
    let pcoa_cpu = pcoa::pcoa(&dm_condensed, 3, 2);
    let pcoa_gpu_result = pcoa_gpu::pcoa_gpu(gpu, &dm_condensed, 3, 2);
    match (&pcoa_cpu, &pcoa_gpu_result) {
        (Ok(cpu_r), Ok(gpu_r)) => {
            for ax in 0..2 {
                v.check(
                    &format!("PCoA eigenvalue[{ax}] CPU vs GPU"),
                    gpu_r.eigenvalues[ax].abs(),
                    cpu_r.eigenvalues[ax].abs(),
                    tolerances::GPU_VS_CPU_F64 * 10.0,
                );
            }
            for i in 0..3 {
                for ax in 0..2 {
                    let cpu_c = cpu_r.coord(i, ax).abs();
                    let gpu_c = gpu_r.coord(i, ax).abs();
                    v.check(
                        &format!("PCoA coord[{i},{ax}] |CPU| vs |GPU|"),
                        gpu_c,
                        cpu_c,
                        tolerances::GPU_VS_CPU_F64 * 10.0,
                    );
                }
            }
        }
        _ => {
            v.check_pass("PCoA: both succeed", false);
        }
    }
}

fn validate_kmd(v: &mut Validator, gpu: &GpuF64) {
    v.section("G7 — Kendrick Mass Defect");
    let test_masses = vec![200.0, 250.0, 300.0, 350.0, 400.0];
    let kmd_cpu =
        kmd::kendrick_mass_defect(&test_masses, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);
    let kmd_gpu_result = kmd_gpu::kendrick_mass_defect_gpu(
        gpu,
        &test_masses,
        kmd::units::CF2_EXACT,
        kmd::units::CF2_NOMINAL,
    );
    match kmd_gpu_result {
        Ok(kmd_gpu_vals) => {
            for (i, (cpu_r, gpu_r)) in kmd_cpu.iter().zip(kmd_gpu_vals.iter()).enumerate() {
                v.check(
                    &format!("KMD[{i}] CPU vs GPU"),
                    gpu_r.kmd,
                    cpu_r.kmd,
                    tolerances::GPU_VS_CPU_F64,
                );
            }
        }
        Err(e) => {
            eprintln!("  KMD GPU failed: {e}");
            v.check_pass("KMD GPU succeeds", false);
        }
    }
}

fn validate_ode_determinism(v: &mut Validator) {
    v.section("G8 — ODE Determinism (CPU)");
    let params = qs_biofilm::QsBiofilmParams::default();
    let y0 = [0.01, 0.0, 0.0, 0.0, 0.0];
    let r1 = qs_biofilm::run_scenario(&y0, 10.0, tolerances::ODE_DEFAULT_DT, &params);
    let r2 = qs_biofilm::run_scenario(&y0, 10.0, tolerances::ODE_DEFAULT_DT, &params);
    v.check_count("ODE: same step count", r1.steps, r2.steps);
    for var in 0..5 {
        v.check(
            &format!("ODE var[{var}] determinism"),
            r1.y_final[var],
            r2.y_final[var],
            tolerances::EXACT,
        );
    }
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            validation::exit_skipped("no GPU device");
        }
    };

    let mut v = Validator::new("Exp328: CPU vs GPU Viz Math Parity");
    let counts = vec![10.0, 20.0, 30.0, 40.0, 50.0, 5.0, 15.0, 25.0, 35.0, 45.0];

    validate_alpha_diversity(&mut v, &gpu, &counts);
    validate_beta_and_ordination(&mut v, &gpu);
    validate_kmd(&mut v, &gpu);
    validate_ode_determinism(&mut v);

    v.finish();
}
