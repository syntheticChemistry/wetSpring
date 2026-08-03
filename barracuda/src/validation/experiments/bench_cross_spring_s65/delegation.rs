// SPDX-License-Identifier: AGPL-3.0-or-later
//! §2-§5: GPU DiversityFusion, CPU diversity/math delegation, special functions.

use std::sync::Arc;

use crate::tolerances;
use crate::validation::{BenchRow, OrExit, Validator, bench_print};

pub(super) fn validate(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<BenchRow>,
) {
    validate_diversity_fusion(v, device, timings);
    validate_cpu_diversity(v, timings);
    validate_cpu_math(v, timings);
    validate_special_functions(v, timings);
}

fn validate_diversity_fusion(
    v: &mut Validator,
    device: &Arc<barracuda::device::WgpuDevice>,
    timings: &mut Vec<BenchRow>,
) {
    v.section("§2 GPU DiversityFusion: Write→Absorb→Lean (wetSpring→S63)");

    let n_species = 10;
    let n_samples = 256;
    let abundances: Vec<f64> = (0..n_samples * n_species)
        .map(|i| ((i * 7 + 3) % 50) as f64 + 1.0)
        .collect();

    let (fusion_gpu, fusion_gpu_ms) =
        bench_print("DiversityFusionGpu (256 samples, 10 spp)", || {
            let fusion =
                crate::bio::diversity_fusion_gpu::DiversityFusionGpu::new(Arc::clone(device))
                    .or_exit("DiversityFusionGpu");
            fusion
                .compute(&abundances, n_samples, n_species)
                .or_exit("compute")
        });

    let (fusion_cpu, fusion_cpu_ms) =
        bench_print("diversity_fusion_cpu (256 samples, 10 spp)", || {
            crate::bio::diversity_fusion_gpu::diversity_fusion_cpu(&abundances, n_species)
        });

    v.check_pass(
        "DiversityFusion: GPU sample count",
        fusion_gpu.len() == n_samples,
    );

    let mut fusion_parity_ok = true;
    for i in 0..n_samples {
        if (fusion_cpu[i].shannon - fusion_gpu[i].shannon).abs() > tolerances::GPU_LOG_POLYFILL
            || (fusion_cpu[i].simpson - fusion_gpu[i].simpson).abs() > tolerances::ANALYTICAL_F64
            || (fusion_cpu[i].evenness - fusion_gpu[i].evenness).abs()
                > tolerances::GPU_LOG_POLYFILL
        {
            fusion_parity_ok = false;
        }
    }
    v.check_pass(
        "DiversityFusion: CPU↔GPU parity (256 samples)",
        fusion_parity_ok,
    );
    v.check_pass(
        "DiversityFusion: all Shannon > 0",
        fusion_gpu.iter().all(|r| r.shannon > 0.0),
    );
    timings.push(BenchRow {
        label: "DiversityFusion GPU 256×",
        origin: "wetSpring→S63",
        ms: fusion_gpu_ms,
    });
    timings.push(BenchRow {
        label: "DiversityFusion CPU 256×",
        origin: "wetSpring→S63",
        ms: fusion_cpu_ms,
    });
}

fn validate_cpu_diversity(v: &mut Validator, timings: &mut Vec<BenchRow>) {
    v.section("§3 CPU Diversity Delegation: bio::diversity → stats::diversity (S64)");

    let community: Vec<f64> = (0..50).map(|i| 1.0 + f64::from(i * 7 % 30)).collect();

    let (sh_local, sh_ms) = bench_print("diversity::shannon (→ barracuda::stats)", || {
        crate::bio::diversity::shannon(&community)
    });
    let sh_upstream = barracuda::stats::shannon(&community);
    v.check(
        "Shannon delegation parity",
        sh_local,
        sh_upstream,
        tolerances::EXACT,
    );
    v.check_pass("Shannon > 0", sh_local > 0.0);

    let (si_local, si_ms) = bench_print("diversity::simpson (→ barracuda::stats)", || {
        crate::bio::diversity::simpson(&community)
    });
    let si_upstream = barracuda::stats::simpson(&community);
    v.check(
        "Simpson delegation parity",
        si_local,
        si_upstream,
        tolerances::EXACT,
    );
    v.check_pass("Simpson ∈ (0,1]", si_local > 0.0 && si_local <= 1.0);

    let (ch_local, _) = bench_print("diversity::chao1 (→ barracuda::stats)", || {
        crate::bio::diversity::chao1(&community)
    });
    v.check(
        "Chao1 delegation parity",
        ch_local,
        barracuda::stats::chao1(&community),
        tolerances::EXACT,
    );

    let (pe_local, _) = bench_print("diversity::pielou_evenness (→ barracuda::stats)", || {
        crate::bio::diversity::pielou_evenness(&community)
    });
    v.check(
        "Pielou delegation parity",
        pe_local,
        barracuda::stats::pielou_evenness(&community),
        tolerances::EXACT,
    );

    let samples_a = vec![10.0, 20.0, 30.0, 0.0, 5.0];
    let samples_b = vec![15.0, 10.0, 25.0, 5.0, 0.0];
    let (bc_local, _) = bench_print("diversity::bray_curtis (→ barracuda::stats)", || {
        crate::bio::diversity::bray_curtis(&samples_a, &samples_b)
    });
    v.check(
        "Bray-Curtis delegation parity",
        bc_local,
        barracuda::stats::bray_curtis(&samples_a, &samples_b),
        tolerances::EXACT,
    );

    let multi_samples = vec![
        vec![10.0, 20.0, 30.0],
        vec![15.0, 10.0, 25.0],
        vec![0.0, 50.0, 0.0],
    ];
    let (bc_cond_local, _) = bench_print("bray_curtis_condensed (→ barracuda::stats)", || {
        crate::bio::diversity::bray_curtis_condensed(&multi_samples)
    });
    let bc_cond_upstream = barracuda::stats::bray_curtis_condensed(&multi_samples);
    let bc_cond_parity = bc_cond_local
        .iter()
        .zip(&bc_cond_upstream)
        .all(|(a, b)| (a - b).abs() <= tolerances::EXACT);
    v.check_pass("bray_curtis_condensed delegation parity", bc_cond_parity);

    let depths: Vec<f64> = (1..=50).map(f64::from).collect();
    let (rare_local, _) = bench_print("rarefaction_curve (→ barracuda::stats)", || {
        crate::bio::diversity::rarefaction_curve(&community, &depths)
    });
    let rare_upstream = barracuda::stats::rarefaction_curve(&community, &depths);
    let rare_parity = rare_local
        .iter()
        .zip(&rare_upstream)
        .all(|(a, b)| (a - b).abs() <= tolerances::EXACT);
    v.check_pass("rarefaction_curve delegation parity", rare_parity);
    timings.push(BenchRow {
        label: "Shannon delegation",
        origin: "S64 cross-spring",
        ms: sh_ms,
    });
    timings.push(BenchRow {
        label: "Simpson delegation",
        origin: "S64 cross-spring",
        ms: si_ms,
    });
}

fn validate_cpu_math(v: &mut Validator, timings: &mut Vec<BenchRow>) {
    v.section("§4 CPU Math: special::{dot,l2_norm} → stats::metrics (S64)");

    let vec_a: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
    let vec_b: Vec<f64> = (0..1000)
        .map(|i| f64::from(i).mul_add(-0.001, 1.0))
        .collect();

    let (dot_local, dot_ms) = bench_print("special::dot (→ barracuda::stats::dot)", || {
        crate::special::dot(&vec_a, &vec_b)
    });
    v.check(
        "dot delegation parity",
        dot_local,
        barracuda::stats::dot(&vec_a, &vec_b),
        tolerances::EXACT,
    );

    let (l2_local, l2_ms) = bench_print("special::l2_norm (→ barracuda::stats::l2_norm)", || {
        crate::special::l2_norm(&vec_a)
    });
    v.check(
        "l2_norm delegation parity",
        l2_local,
        barracuda::stats::l2_norm(&vec_a),
        tolerances::EXACT,
    );
    timings.push(BenchRow {
        label: "dot(1000) delegation",
        origin: "S64 cross-spring",
        ms: dot_ms,
    });
    timings.push(BenchRow {
        label: "l2_norm(1000) delegation",
        origin: "S64 cross-spring",
        ms: l2_ms,
    });
}

fn validate_special_functions(v: &mut Validator, timings: &mut Vec<BenchRow>) {
    v.section("§5 CPU Special Functions: hotSpring precision → ToadStool");

    let (erf_val, erf_ms) = bench_print("erf(1.0) — barracuda::special", || {
        barracuda::special::erf(1.0)
    });
    v.check(
        "erf(1.0)",
        erf_val,
        0.842_700_792_949_715,
        tolerances::ERF_PARITY,
    );
    timings.push(BenchRow {
        label: "erf(1.0)",
        origin: "hotSpring→ToadStool",
        ms: erf_ms,
    });

    let (lng_val, lng_ms) = bench_print("ln_gamma(5.0) — barracuda::special", || {
        barracuda::special::ln_gamma(5.0).or_exit("ln_gamma")
    });
    v.check(
        "ln_gamma(5.0)",
        lng_val,
        3.178_053_830_347_95,
        tolerances::PYTHON_PARITY,
    );
    timings.push(BenchRow {
        label: "ln_gamma(5.0)",
        origin: "hotSpring→ToadStool",
        ms: lng_ms,
    });

    let (ncdf_val, ncdf_ms) = bench_print("norm_cdf(1.96) — barracuda::stats", || {
        barracuda::stats::norm_cdf(1.96)
    });
    v.check(
        "norm_cdf(1.96) ≈ 0.975",
        ncdf_val,
        0.975,
        tolerances::NORM_CDF_PARITY,
    );
    timings.push(BenchRow {
        label: "norm_cdf(1.96)",
        origin: "hotSpring→ToadStool",
        ms: ncdf_ms,
    });

    let (trapz_val, trapz_ms) = bench_print("trapz(1000 pts) — barracuda::numerical", || {
        let x: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        barracuda::numerical::trapz(&y, &x).or_exit("trapz")
    });
    v.check(
        "∫x² dx [0,0.999]",
        trapz_val,
        0.332_334,
        tolerances::ODE_METHOD_PARITY,
    );
    timings.push(BenchRow {
        label: "trapz(1000)",
        origin: "ToadStool native",
        ms: trapz_ms,
    });

    let vec_a: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.001).collect();
    let vec_b: Vec<f64> = (0..1000)
        .map(|i| f64::from(i).mul_add(-0.001, 1.0))
        .collect();
    let (pear_val, pear_ms) = bench_print("pearson_correlation — barracuda::stats", || {
        barracuda::stats::pearson_correlation(&vec_a, &vec_b).or_exit("pearson")
    });
    v.check_pass("pearson is finite", pear_val.is_finite());
    v.check_pass("pearson is negative (inverse data)", pear_val < 0.0);
    timings.push(BenchRow {
        label: "pearson(1000)",
        origin: "ToadStool native",
        ms: pear_ms,
    });
}
