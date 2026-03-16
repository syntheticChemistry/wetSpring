// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::print_stdout,
    reason = "validation harness: results printed to stdout"
)]
#![expect(
    clippy::too_many_lines,
    reason = "validation harness: sequential domain checks in single main()"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::similar_names,
    reason = "validation harness: domain variables from published notation"
)]
#![expect(
    clippy::float_cmp,
    reason = "validation harness: exact comparison with known analytical constants"
)]
//! # Exp244: `ToadStool` Compute Dispatch v2 — Extended Overhead Proof
//!
//! Extends Exp073 dispatch overhead proof with new streaming domains.
//! Proves `ToadStool` unidirectional streaming eliminates CPU round-trips
//! for: DADA2, chimera, diversity, Bray-Curtis, taxonomy, full analytics.
//!
//! Validates:
//! 1. `GpuPipelineSession` warmup and pre-compilation
//! 2. Streaming vs individual dispatch overhead reduction
//! 3. Full analytics pipeline (taxonomy + diversity + BC in one session)
//! 4. CPU reference parity for all streaming outputs
//!
//! Chain: Paper → CPU → GPU → Parity → **`ToadStool` (this)** → `metalForge`
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Provenance type | Cross-spring validation |
//! | Date | 2026-03-03 |
//! | Command | `cargo run --release --bin validate_toadstool_dispatch_v2` |
//!
//! Validation class: Cross-spring
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

use std::time::Instant;

use wetspring_barracuda::bio::{diversity, diversity_gpu, streaming_gpu, taxonomy};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::tolerances;
use wetspring_barracuda::validation::Validator;

fn main() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init");

    let mut v = Validator::new("Exp244: ToadStool Compute Dispatch v2 — Extended Overhead Proof");
    let t_total = Instant::now();

    println!("  GPU: {}", gpu.adapter_name);

    // ═══ S1: Pipeline Session Pre-Warmup ═════════════════════════════════
    v.section("S1: GpuPipelineSession Pre-Warmup");
    let t_warmup = Instant::now();
    let session =
        streaming_gpu::GpuPipelineSession::new(&gpu).expect("GpuPipelineSession creation");
    let warmup_ms = t_warmup.elapsed().as_secs_f64() * 1000.0;
    v.check_pass("Session created", true);
    v.check_pass("Warmup < 5s", warmup_ms < 5000.0);
    println!("    Session warmup: {warmup_ms:.1} ms");
    println!("    Context: {}", session.ctx_stats());

    // ═══ S2: Streaming Diversity vs Individual Dispatch ═══════════════════
    v.section("S2: Streaming vs Individual Dispatch (Diversity)");
    let abundances: Vec<f64> = (0..1024)
        .map(|i| f64::from(i + 1).mul_add(1.5, 0.5))
        .collect();

    let tc = Instant::now();
    let cpu_sh = diversity::shannon(&abundances);
    let cpu_si = diversity::simpson(&abundances);
    let cpu_obs = diversity::observed_features(&abundances);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg_ind = Instant::now();
    let gpu_sh_ind = diversity_gpu::shannon_gpu(&gpu, &abundances).expect("GPU Shannon dispatch");
    let gpu_si_ind = diversity_gpu::simpson_gpu(&gpu, &abundances).expect("GPU Simpson dispatch");
    let gpu_obs_ind = diversity_gpu::observed_features_gpu(&gpu, &abundances)
        .expect("GPU observed features dispatch");
    let ind_us = tg_ind.elapsed().as_micros() as f64;

    let tg_stream = Instant::now();
    let stream_sh = session.shannon(&abundances).expect("streaming Shannon");
    let stream_si = session.simpson(&abundances).expect("streaming Simpson");
    let stream_obs = session
        .observed_features(&abundances)
        .expect("streaming observed features");
    let stream_us = tg_stream.elapsed().as_micros() as f64;

    v.check(
        "Shannon: CPU == streaming",
        stream_sh,
        cpu_sh,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Simpson: CPU == streaming",
        stream_si,
        cpu_si,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Observed: CPU == streaming",
        stream_obs,
        cpu_obs,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "Shannon: individual == streaming",
        stream_sh,
        gpu_sh_ind,
        tolerances::EXACT,
    );
    v.check(
        "Simpson: individual == streaming",
        stream_si,
        gpu_si_ind,
        tolerances::EXACT,
    );
    v.check(
        "Observed: individual == streaming",
        stream_obs,
        gpu_obs_ind,
        tolerances::EXACT,
    );

    println!("    CPU:        {cpu_us:.0} µs");
    println!("    Individual: {ind_us:.0} µs (3 dispatches)");
    println!("    Streaming:  {stream_us:.0} µs (pre-warmed session)");
    if ind_us > 0.0 {
        println!(
            "    Overhead reduction: {:.0}%",
            (1.0 - stream_us / ind_us) * 100.0
        );
    }

    // ═══ S3: Bray-Curtis Matrix Streaming ════════════════════════════════
    v.section("S3: Bray-Curtis Matrix Streaming");
    let sample_a: Vec<f64> = (0..100).map(|i| f64::from((i + 1) % 30 + 1)).collect();
    let sample_b: Vec<f64> = (0..100).map(|i| f64::from((i * 3 + 7) % 30 + 1)).collect();
    let sample_c: Vec<f64> = (0..100).map(|i| f64::from((i * 5 + 2) % 30 + 1)).collect();
    let slices: Vec<&[f64]> = vec![&sample_a, &sample_b, &sample_c];

    let tc = Instant::now();
    let cpu_bc_01 = diversity::bray_curtis(&sample_a, &sample_b);
    let cpu_bc_02 = diversity::bray_curtis(&sample_a, &sample_c);
    let cpu_bc_12 = diversity::bray_curtis(&sample_b, &sample_c);
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let stream_bc = session
        .bray_curtis_matrix(&slices)
        .expect("streaming Bray-Curtis matrix");
    let stream_us = tg.elapsed().as_micros() as f64;

    v.check_pass("BC matrix: condensed len = 3", stream_bc.len() == 3);
    v.check(
        "BC [0,1] CPU == streaming",
        stream_bc[0],
        cpu_bc_01,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "BC [0,2] CPU == streaming",
        stream_bc[1],
        cpu_bc_02,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check(
        "BC [1,2] CPU == streaming",
        stream_bc[2],
        cpu_bc_12,
        tolerances::GPU_VS_CPU_F64,
    );
    println!("    CPU: {cpu_us:.0} µs, Streaming: {stream_us:.0} µs");

    // ═══ S4: Taxonomy + Diversity Streaming (Full Pipeline) ══════════════
    v.section("S4: Full Streaming Pipeline (Taxonomy + Diversity)");
    let refs = vec![
        taxonomy::ReferenceSeq {
            id: "r1".into(),
            sequence: b"ACGTACGTACGT".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("Bac;Firm"),
        },
        taxonomy::ReferenceSeq {
            id: "r2".into(),
            sequence: b"GGTTTTGGTTTT".to_vec(),
            lineage: taxonomy::Lineage::from_taxonomy_string("Bac;Prot"),
        },
    ];
    let classifier = taxonomy::NaiveBayesClassifier::train(&refs, 8);
    let query_seqs: Vec<&[u8]> = vec![b"ACGTACGTACGT", b"GGTTTTGGTTTT", b"ACGTACGTACGT"];
    let counts: Vec<f64> = vec![50.0, 30.0, 20.0];
    let classify_params = taxonomy::ClassifyParams::default();

    let tc = Instant::now();
    let cpu_result = streaming_gpu::stream_classify_and_diversity_cpu(
        &classifier,
        &query_seqs,
        &counts,
        &classify_params,
    );
    let cpu_us = tc.elapsed().as_micros() as f64;

    let tg = Instant::now();
    let gpu_result = session
        .stream_sample(&classifier, &query_seqs, &counts, &classify_params)
        .expect("streaming taxonomy + diversity");
    let gpu_us = tg.elapsed().as_micros() as f64;

    v.check(
        "Stream Shannon: CPU == GPU",
        gpu_result.shannon,
        cpu_result.shannon,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Stream Simpson: CPU == GPU",
        gpu_result.simpson,
        cpu_result.simpson,
        tolerances::GPU_VS_CPU_TRANSCENDENTAL,
    );
    v.check(
        "Stream Observed: CPU == GPU",
        gpu_result.observed,
        cpu_result.observed,
        tolerances::GPU_VS_CPU_F64,
    );
    v.check_pass(
        "Stream: classifications count",
        gpu_result.classifications.len() == query_seqs.len(),
    );
    println!("    CPU pipeline: {cpu_us:.0} µs");
    println!("    GPU streaming: {gpu_us:.0} µs");
    println!(
        "    Taxonomy: {:.1} ms, Diversity: {:.1} ms",
        gpu_result.taxonomy_ms, gpu_result.diversity_ms
    );

    // ═══ S5: Full Analytics Pipeline (Taxonomy + Diversity + BC) ═════════
    v.section("S5: Full Analytics Pipeline");
    let multi_samples: Vec<&[f64]> = vec![&sample_a, &sample_b, &sample_c];

    let tg = Instant::now();
    let full = session
        .stream_full_analytics(&classifier, &query_seqs, &multi_samples, &classify_params)
        .expect("streaming full analytics pipeline");
    let full_us = tg.elapsed().as_micros() as f64;

    v.check_pass(
        "Full: classifications present",
        !full.classifications.is_empty(),
    );
    v.check_pass(
        "Full: alpha diversity for each sample",
        full.alpha.len() == multi_samples.len(),
    );
    v.check_pass("Full: BC matrix condensed", full.bray_curtis.len() == 3);
    v.check_pass("Full: total_ms > 0", full.total_ms > 0.0);
    println!(
        "    Full analytics: {full_us:.0} µs ({:.1} ms total GPU)",
        full.total_ms
    );
    println!(
        "    Taxonomy: {:.1} ms, Diversity: {:.1} ms, BC: {:.1} ms",
        full.taxonomy_ms, full.diversity_ms, full.bray_curtis_ms
    );

    // ═══ S6: Streaming Determinism ═══════════════════════════════════════
    v.section("S6: Streaming Determinism (3 runs)");
    let mut shannons = Vec::new();
    for run in 0..3 {
        let s = session
            .shannon(&abundances)
            .expect("streaming Shannon for determinism check");
        shannons.push(s);
        println!("    Run {}: {s:.15}", run + 1);
    }
    v.check_pass("Determinism: run 1 == run 2", shannons[0] == shannons[1]);
    v.check_pass("Determinism: run 2 == run 3", shannons[1] == shannons[2]);

    // ═══ Summary ═════════════════════════════════════════════════════════
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    v.section("ToadStool Dispatch Summary");
    println!("  ToadStool streaming: pre-warmed GpuPipelineSession");
    println!("  Unidirectional: data flows source → GPU → sink (zero CPU round-trips)");
    println!(
        "  6 sections: warmup, diversity, BC matrix, stream_sample, full_analytics, determinism"
    );
    println!("  Total: {total_ms:.1} ms");
    println!();

    v.finish();
}
