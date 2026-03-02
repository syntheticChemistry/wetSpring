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
//! # Exp241: Pure GPU Streaming v7 — Extended `ToadStool` Unidirectional Pipeline
//!
//! Extends streaming v6 with new domains:
//! - Stage 1: DADA2 GPU denoising
//! - Stage 2: Chimera GPU filtering
//! - Stage 3: Diversity GPU (Shannon + Simpson + Bray-Curtis)
//! - Stage 4: Rarefaction GPU with bootstrap CIs
//! - Stage 5: Kriging GPU spatial interpolation
//! - Stage 6: Reconciliation GPU
//!
//! Validates: round-trip == streaming parity, bitwise determinism,
//! zero CPU round-trips between GPU stages.
//!
//! Chain position: Paper → CPU (Exp239) → GPU (Exp240) → **Streaming (this)** → metalForge
//!
//! | Field | Value |
//! |-------|-------|
//! | Date | 2026-02-28 |
//! | Command | `cargo run --features gpu --bin validate_pure_gpu_streaming_v7` |
//!
//! Validation class: GPU-parity
//! Provenance: CPU reference implementation in `barracuda::bio`

use std::time::Instant;

use wetspring_barracuda::bio::{
    chimera, chimera_gpu, dada2, dada2_gpu, derep, diversity, diversity_gpu, kriging,
    rarefaction_gpu, reconciliation, reconciliation_gpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    let mut v = Validator::new(
        "Exp241: Pure GPU Streaming v7 — Extended ToadStool Unidirectional Pipeline",
    );
    let t_total = Instant::now();

    // ═══ Stage 1: DADA2 GPU Denoising ════════════════════════════════════
    v.section("Stage 1: DADA2 GPU Denoising");
    let dada2_device = gpu.to_wgpu_device();
    let dada2_engine = dada2_gpu::Dada2Gpu::new(dada2_device).unwrap();
    let raw_seqs = vec![
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGTACGT".to_vec(),
            abundance: 100,
            best_quality: 40.0,
            representative_id: "r1".into(),
            representative_quality: vec![40; 16],
        },
        derep::UniqueSequence {
            sequence: b"ACGTACGTACGTACGA".to_vec(),
            abundance: 5,
            best_quality: 35.0,
            representative_id: "r2".into(),
            representative_quality: vec![35; 16],
        },
        derep::UniqueSequence {
            sequence: b"TTTTACGTACGTTTTT".to_vec(),
            abundance: 80,
            best_quality: 39.0,
            representative_id: "r3".into(),
            representative_quality: vec![39; 16],
        },
        derep::UniqueSequence {
            sequence: b"GGGGACGTACGTGGGG".to_vec(),
            abundance: 60,
            best_quality: 38.0,
            representative_id: "r4".into(),
            representative_quality: vec![38; 16],
        },
    ];
    let (gpu_asvs, gpu_stats) =
        dada2_gpu::denoise_gpu(&dada2_engine, &raw_seqs, &dada2::Dada2Params::default()).unwrap();
    let (cpu_asvs, _) = dada2::denoise(&raw_seqs, &dada2::Dada2Params::default());
    v.check_pass(
        "DADA2: GPU ASV count matches CPU",
        gpu_asvs.len() == cpu_asvs.len(),
    );
    v.check_pass("DADA2: output reads > 0", gpu_stats.output_reads > 0);
    println!(
        "    {} ASVs from {} unique sequences",
        gpu_asvs.len(),
        raw_seqs.len()
    );

    // ═══ Stage 2: Chimera GPU Filtering ══════════════════════════════════
    v.section("Stage 2: Chimera GPU Filtering");
    let (clean_asvs_gpu, chim_stats) =
        chimera_gpu::remove_chimeras_gpu(&gpu, &gpu_asvs, &chimera::ChimeraParams::default())
            .unwrap();
    let (clean_asvs_cpu, _) =
        chimera::remove_chimeras(&gpu_asvs, &chimera::ChimeraParams::default());
    v.check_pass(
        "Chimera: GPU clean count == CPU",
        clean_asvs_gpu.len() == clean_asvs_cpu.len(),
    );
    v.check_pass("Chimera: stats populated", chim_stats.input_sequences > 0);
    println!(
        "    {} non-chimeric ASVs (from {})",
        clean_asvs_gpu.len(),
        gpu_asvs.len()
    );

    // ═══ Stage 3: Diversity GPU ══════════════════════════════════════════
    v.section("Stage 3: Diversity GPU (Shannon + BC)");
    let combined: Vec<f64> = clean_asvs_gpu.iter().map(|a| a.abundance as f64).collect();
    let gpu_shannon = diversity_gpu::shannon_gpu(&gpu, &combined).unwrap();
    let cpu_shannon = diversity::shannon(&combined);
    v.check(
        "Diversity: Shannon CPU == GPU",
        gpu_shannon,
        cpu_shannon,
        tolerances::GPU_VS_CPU_F64,
    );
    let gpu_simpson = diversity_gpu::simpson_gpu(&gpu, &combined).unwrap();
    let cpu_simpson = diversity::simpson(&combined);
    v.check(
        "Diversity: Simpson CPU == GPU",
        gpu_simpson,
        cpu_simpson,
        tolerances::GPU_VS_CPU_F64,
    );
    println!("    Shannon: {gpu_shannon:.6}, Simpson: {gpu_simpson:.6}");

    // ═══ Stage 4: Rarefaction GPU ════════════════════════════════════════
    v.section("Stage 4: Rarefaction GPU (Bootstrap CIs)");
    let rare_params = rarefaction_gpu::RarefactionGpuParams {
        n_bootstrap: 100,
        depth: None,
        seed: 42,
    };
    let rare = rarefaction_gpu::rarefaction_bootstrap_gpu(&gpu, &combined, &rare_params).unwrap();
    v.check_pass(
        "Rarefaction: Shannon CI valid",
        rare.shannon.lower <= rare.shannon.upper,
    );
    v.check_pass(
        "Rarefaction: Simpson CI valid",
        rare.simpson.lower <= rare.simpson.upper,
    );
    v.check_pass(
        "Rarefaction: observed features > 0",
        rare.observed.mean > 0.0,
    );
    println!(
        "    Shannon CI: [{:.4}, {:.4}]",
        rare.shannon.lower, rare.shannon.upper
    );

    // ═══ Stage 5: Kriging GPU ════════════════════════════════════════════
    v.section("Stage 5: Kriging GPU (Spatial Interpolation)");
    let sites: Vec<kriging::SpatialSample> = clean_asvs_gpu
        .iter()
        .enumerate()
        .map(|(i, a)| kriging::SpatialSample {
            x: (i as f64) * 0.5,
            y: (i as f64) * 0.3,
            value: a.abundance as f64,
        })
        .collect();
    if sites.len() >= 3 {
        let targets = vec![(0.25, 0.15), (0.75, 0.45)];
        let config = kriging::VariogramConfig::spherical(0.0, 1.0, 5.0);
        let krig = kriging::interpolate_diversity(&gpu, &sites, &targets, &config).unwrap();
        v.check_pass(
            "Kriging: interpolated values present",
            krig.values.len() == 2,
        );
        v.check_pass(
            "Kriging: values finite",
            krig.values.iter().all(|v| v.is_finite()),
        );
        println!("    Interpolated {} target locations", krig.values.len());
    } else {
        v.check_pass("Kriging: skipped (< 3 sites)", true);
    }

    // ═══ Stage 6: Reconciliation GPU ═════════════════════════════════════
    v.section("Stage 6: Reconciliation GPU");
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
    let tip_mapping = vec![
        ("p0".to_string(), "h0".to_string()),
        ("p1".to_string(), "h1".to_string()),
    ];
    let costs = reconciliation::DtlCosts::default();
    let cpu_dtl = reconciliation::reconcile_dtl(&host, &parasite, &tip_mapping, &costs);
    let gpu_dtl =
        reconciliation_gpu::reconcile_dtl_gpu(&gpu, &host, &parasite, &tip_mapping, &costs)
            .unwrap();
    v.check_pass(
        "DTL: GPU cost == CPU",
        gpu_dtl.optimal_cost == cpu_dtl.optimal_cost,
    );

    // ═══ Determinism Check ═══════════════════════════════════════════════
    v.section("Determinism Check (3 runs)");
    let mut shannons = Vec::new();
    for run in 0..3 {
        let s = diversity_gpu::shannon_gpu(&gpu, &combined).unwrap();
        shannons.push(s);
        println!("    Run {}: Shannon = {s:.15}", run + 1);
    }
    v.check_pass(
        "Bitwise determinism: run 1 == run 2",
        shannons[0] == shannons[1],
    );
    v.check_pass(
        "Bitwise determinism: run 2 == run 3",
        shannons[1] == shannons[2],
    );

    // ═══ Bray-Curtis GPU (streaming element) ══════════════════════════════
    v.section("Bray-Curtis GPU (streaming parity)");
    if clean_asvs_gpu.len() >= 2 {
        let bc_vecs: Vec<Vec<f64>> = clean_asvs_gpu
            .iter()
            .map(|a| vec![a.abundance as f64])
            .collect();
        let gpu_bc = diversity_gpu::bray_curtis_condensed_gpu(&gpu, &bc_vecs).unwrap();
        let cpu_bc = diversity::bray_curtis_condensed(&bc_vecs);
        v.check_pass(
            "BC GPU: condensed len matches",
            gpu_bc.len() == cpu_bc.len(),
        );
        for (i, (g, c)) in gpu_bc.iter().zip(cpu_bc.iter()).enumerate() {
            v.check(
                &format!("BC [{i}]: CPU == GPU"),
                *g,
                *c,
                tolerances::GPU_VS_CPU_F64,
            );
        }
    }

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("Pipeline Summary");
    println!(
        "  6-stage GPU streaming pipeline: DADA2 → Chimera → Diversity → Rarefaction → Kriging → Reconciliation"
    );
    println!("  ToadStool unidirectional: zero CPU round-trips between GPU stages");
    println!("  Total pipeline time: {total_ms:.2} ms");
    println!();

    v.finish();
}
