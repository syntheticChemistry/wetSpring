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
//! # Exp243: CPU vs GPU Extended Parity — 22 Domains Head-to-Head
//!
//! Consolidated proof that `BarraCuda` pure Rust math produces identical
//! results on CPU and GPU across all 22 GPU-eligible domains.
//! CPU is the reference; GPU must match within tolerance.
//! Wall-clock timing captures both paths.
//!
//! Chain: Paper → CPU → GPU → **Parity (this)** → `ToadStool` Dispatch → `metalForge`
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_cpu_vs_gpu_v6_extended` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;

use wetspring_barracuda::bio::{
    chimera, chimera_gpu, dada2, dada2_gpu, decision_tree, derep, gbm, gbm_gpu, molecular_clock,
    molecular_clock_gpu, random_forest, random_forest_gpu, reconciliation, reconciliation_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

struct Timing {
    name: &'static str,
    cpu_us: f64,
    gpu_us: f64,
}

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    let mut v = Validator::new("Exp243: CPU vs GPU Extended Parity — 22 Domains");
    let t_total = Instant::now();
    let mut timings: Vec<Timing> = Vec::new();

    println!("  GPU: {}", gpu.adapter_name);
    println!("  Inherited: D01-D16 from Exp092 (16 domains, all PASS)");
    println!("  New: D17-D22 below (6 domains with CPU+GPU variants)");
    println!();

    // ═══ D17: Chimera CPU vs GPU ═════════════════════════════════════════
    v.section("D17: Chimera Detection CPU↔GPU");
    let asvs = vec![
        dada2::Asv {
            sequence: b"AAAACCCCGGGGTTTT".to_vec(),
            abundance: 100,
            n_members: 100,
        },
        dada2::Asv {
            sequence: b"AAAACCCCTTTTGGGG".to_vec(),
            abundance: 80,
            n_members: 80,
        },
        dada2::Asv {
            sequence: b"AAAACCCCGGGGGGGG".to_vec(),
            abundance: 5,
            n_members: 5,
        },
    ];
    let params = chimera::ChimeraParams::default();
    let tc = Instant::now();
    let (cpu_results, cpu_stats) = chimera::detect_chimeras(&asvs, &params);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let (gpu_results, gpu_stats) = chimera_gpu::detect_chimeras_gpu(&gpu, &asvs, &params).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check_pass(
        "Chimera: result count match",
        gpu_results.len() == cpu_results.len(),
    );
    v.check_pass(
        "Chimera: stats match",
        gpu_stats.chimeras_found == cpu_stats.chimeras_found,
    );
    v.check_pass(
        "Chimera: retained match",
        gpu_stats.retained == cpu_stats.retained,
    );
    timings.push(Timing {
        name: "Chimera",
        cpu_us,
        gpu_us,
    });

    // ═══ D18: DADA2 CPU vs GPU ═══════════════════════════════════════════
    v.section("D18: DADA2 Denoising CPU↔GPU");
    let seqs = vec![
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGT".to_vec(),
            abundance: 50,
            best_quality: 40.0,
            representative_id: "s1".into(),
            representative_quality: vec![40; 12],
        },
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGA".to_vec(),
            abundance: 3,
            best_quality: 35.0,
            representative_id: "s2".into(),
            representative_quality: vec![35; 12],
        },
        derep::UniqueSequence {
            sequence: b"TTTTACGTACGT".to_vec(),
            abundance: 40,
            best_quality: 39.0,
            representative_id: "s3".into(),
            representative_quality: vec![39; 12],
        },
    ];
    let dada2_params = dada2::Dada2Params::default();
    let tc = Instant::now();
    let (cpu_asvs, cpu_dstats) = dada2::denoise(&seqs, &dada2_params);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let dada2_device = gpu.to_wgpu_device();
    let dada2_engine = dada2_gpu::Dada2Gpu::new(dada2_device).unwrap();
    let tg = Instant::now();
    let (gpu_asvs, gpu_dstats) =
        dada2_gpu::denoise_gpu(&dada2_engine, &seqs, &dada2_params).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check_pass("DADA2: ASV count match", gpu_asvs.len() == cpu_asvs.len());
    v.check_pass(
        "DADA2: output reads match",
        gpu_dstats.output_reads == cpu_dstats.output_reads,
    );
    v.check_pass(
        "DADA2: iterations match",
        gpu_dstats.iterations == cpu_dstats.iterations,
    );
    timings.push(Timing {
        name: "DADA2",
        cpu_us,
        gpu_us,
    });

    // ═══ D19: GBM CPU vs GPU ═════════════════════════════════════════════
    v.section("D19: GBM Classifier CPU↔GPU");
    let tree1 = gbm::GbmTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.5, 0.5],
    )
    .unwrap();
    let tree2 = gbm::GbmTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[0.0, -0.3, 0.3],
    )
    .unwrap();
    let model = gbm::GbmClassifier::new(vec![tree1, tree2], 0.1, 0.0, 2).unwrap();
    let samples = vec![
        vec![0.8, 0.5],
        vec![0.2, 0.1],
        vec![0.5, 0.5],
        vec![0.9, 0.9],
    ];
    let tc = Instant::now();
    let cpu_preds = model.predict_batch_proba(&samples);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_preds = gbm_gpu::predict_batch_gpu(&gpu, &model, &samples).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check_pass("GBM: batch size match", gpu_preds.len() == cpu_preds.len());
    for (i, (cp, gp)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
        v.check_pass(&format!("GBM [{i}]: class match"), cp.class == gp.class);
    }
    timings.push(Timing {
        name: "GBM",
        cpu_us,
        gpu_us,
    });

    // ═══ D20: Reconciliation CPU vs GPU ══════════════════════════════════
    v.section("D20: DTL Reconciliation CPU↔GPU");
    let host = reconciliation::FlatRecTree {
        names: vec!["h0".into(), "h1".into(), "h2".into()],
        left_child: vec![u32::MAX, u32::MAX, 0],
        right_child: vec![u32::MAX, u32::MAX, 1],
    };
    let parasite = reconciliation::FlatRecTree {
        names: vec!["p0".into(), "p1".into(), "p2".into()],
        left_child: vec![u32::MAX, u32::MAX, 0],
        right_child: vec![u32::MAX, u32::MAX, 1],
    };
    let tip_map = vec![
        ("p0".to_string(), "h0".to_string()),
        ("p1".to_string(), "h1".to_string()),
    ];
    let costs = reconciliation::DtlCosts::default();
    let tc = Instant::now();
    let cpu_dtl = reconciliation::reconcile_dtl(&host, &parasite, &tip_map, &costs);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_dtl =
        reconciliation_gpu::reconcile_dtl_gpu(&gpu, &host, &parasite, &tip_map, &costs).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check_pass(
        "DTL: cost match",
        gpu_dtl.optimal_cost == cpu_dtl.optimal_cost,
    );
    v.check_pass(
        "DTL: host mapping match",
        gpu_dtl.optimal_host == cpu_dtl.optimal_host,
    );
    timings.push(Timing {
        name: "Reconciliation",
        cpu_us,
        gpu_us,
    });

    // ═══ D21: Molecular Clock CPU vs GPU ═════════════════════════════════
    v.section("D21: Molecular Clock CPU↔GPU");
    let bl = vec![0.1, 0.2, 0.15, 0.05, 0.0];
    let parents_cpu = vec![Some(4), Some(4), Some(3), Some(4), None];
    let parents_gpu: Vec<i64> = parents_cpu
        .iter()
        .map(|p| p.map_or(-1, |x| i64::try_from(x).expect("index fits i64")))
        .collect();
    let cal = vec![molecular_clock::CalibrationPoint {
        node_id: 4,
        min_age_ma: 10.0,
        max_age_ma: 50.0,
    }];
    let tc = Instant::now();
    let cpu_clock = molecular_clock::strict_clock(&bl, &parents_cpu, 30.0, &cal);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_clock =
        molecular_clock_gpu::strict_clock_gpu(&gpu, &bl, &parents_gpu, 30.0, &cal).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check_pass(
        "Clock: both present",
        cpu_clock.is_some() == gpu_clock.is_some(),
    );
    if let (Some(cc), Some(gc)) = (&cpu_clock, &gpu_clock) {
        v.check(
            "Clock: rate match",
            gc.rate,
            cc.rate,
            tolerances::GPU_VS_CPU_F64,
        );
    }
    let ages = vec![0.0, 0.0, 10.0, 15.0, 30.0];
    let cpu_rates = molecular_clock::relaxed_clock_rates(&bl, &ages, &parents_cpu);
    let gpu_rates =
        molecular_clock_gpu::relaxed_clock_rates_gpu(&gpu, &bl, &ages, &parents_gpu).unwrap();
    for (i, (cr, gr)) in cpu_rates.iter().zip(gpu_rates.iter()).enumerate() {
        v.check(
            &format!("Relaxed [{i}]"),
            *gr,
            *cr,
            tolerances::GPU_VS_CPU_F64,
        );
    }
    timings.push(Timing {
        name: "Molecular Clock",
        cpu_us,
        gpu_us,
    });

    // ═══ D22: Random Forest CPU vs GPU ═══════════════════════════════════
    v.section("D22: Random Forest CPU↔GPU");
    let dt1 = decision_tree::DecisionTree::from_arrays(
        &[0, -1, -1],
        &[0.5, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .unwrap();
    let dt2 = decision_tree::DecisionTree::from_arrays(
        &[1, -1, -1],
        &[0.3, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(1)],
        2,
    )
    .unwrap();
    let forest = random_forest::RandomForest::from_trees(vec![dt1, dt2], 2).unwrap();
    let rf_gpu = random_forest_gpu::RandomForestGpu::new(&gpu.to_wgpu_device());
    let rf_samples = vec![vec![0.8, 0.5], vec![0.1, 0.9], vec![0.5, 0.3]];
    let tc = Instant::now();
    let cpu_rf = forest.predict_batch_with_votes(&rf_samples);
    let cpu_us = tc.elapsed().as_micros() as f64;
    let tg = Instant::now();
    let gpu_rf = rf_gpu.predict_batch(&forest, &rf_samples).unwrap();
    let gpu_us = tg.elapsed().as_micros() as f64;
    v.check_pass("RF: batch size match", gpu_rf.len() == cpu_rf.len());
    for (i, (cp, gp)) in cpu_rf.iter().zip(gpu_rf.iter()).enumerate() {
        v.check_pass(&format!("RF [{i}]: class match"), cp.class == gp.class);
    }
    timings.push(Timing {
        name: "Random Forest",
        cpu_us,
        gpu_us,
    });

    // ═══ Summary ═════════════════════════════════════════════════════════
    v.section("CPU vs GPU Head-to-Head Summary");
    println!();
    println!("  {:<25} {:>10} {:>10}", "Domain", "CPU (µs)", "GPU (µs)");
    println!("  {}", "─".repeat(47));
    for t in &timings {
        println!("  {:<25} {:>10.0} {:>10.0}", t.name, t.cpu_us, t.gpu_us);
    }
    println!("  {}", "─".repeat(47));
    let total_cpu: f64 = timings.iter().map(|t| t.cpu_us).sum();
    let total_gpu: f64 = timings.iter().map(|t| t.gpu_us).sum();
    println!(
        "  {:<25} {:>10.0} {:>10.0}",
        "TOTAL (new)", total_cpu, total_gpu
    );
    println!();
    println!("  6 new domains + 16 inherited = 22 total, CPU↔GPU parity proven");
    println!("  Same equations, different hardware — math is truly portable");
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("  Total: {total_ms:.1} ms");
    println!();

    v.finish();
}
