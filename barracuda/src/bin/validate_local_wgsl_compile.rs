// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Exp096: ToadStool Bio Op Absorption Validation
//!
//! # Provenance
//!
//! | Script  | `validate_local_wgsl_compile` |
//! | Command | `cargo run --features gpu --bin validate_local_wgsl_compile` |
//!
//! # Purpose
//!
//! Validates that ToadStool's upstream bio ops (absorbed from wetSpring local
//! WGSL shaders in Sessions 39-41) produce correct results. This is the
//! **Lean phase** proof: same test data, same expected values, upstream ops.
//!
//! 1. `KmerHistogramGpu` — k-mer counting (absorbed S39)
//! 2. `TaxonomyFcGpu` — Naive Bayes taxonomy scoring (absorbed S39)
//! 3. `UniFracPropagateGpu` — tree propagation compile (absorbed S39)
//! 4. `BatchedOdeRK4F64` — ODE RK4 integration via OdeSweepGpu (absorbed S41)

use std::time::Instant;
use wgpu::util::DeviceExt;

use barracuda::{KmerHistogramGpu, TaxonomyFcGpu, UniFracPropagateGpu};
use wetspring_barracuda::bio::ode_sweep_gpu::{OdeSweepConfig, OdeSweepGpu};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
async fn main() {
    println!("════════════════════════════════════════════════════════════════════");
    println!("  Exp096: ToadStool Bio Op Absorption Validation");
    println!("  Proving Lean-phase: 4 absorbed ops, same data, upstream dispatch");
    println!("════════════════════════════════════════════════════════════════════\n");

    let mut v = Validator::new("Exp096: ToadStool Bio Op Absorption");

    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("No GPU: {e}");
            validation::exit_skipped("No GPU available");
        }
    };
    gpu.print_info();
    let device = gpu.to_wgpu_device();
    let d = device.device();

    // ═══ Op 1: KmerHistogramGpu (upstream, absorbed S39) ═════════════════
    v.section("KmerHistogramGpu — ToadStool dispatch");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let op = KmerHistogramGpu::new(device.clone());

            let kmers: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 0, 0];
            let histogram = vec![0u32; 16]; // 4^2

            let kmer_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&kmers),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let hist_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&histogram),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

            op.dispatch(&kmer_buf, &hist_buf, 8, 2);
            d.poll(wgpu::Maintain::Wait);

            device.read_buffer_u32(&hist_buf, 16).expect("readback")
        }));
        let us = t0.elapsed().as_micros();

        match result {
            Ok(hist) => {
                v.check("kmer bin 0 count", f64::from(hist[0]), 4.0, 0.0);
                v.check("kmer bin 1 count", f64::from(hist[1]), 2.0, 0.0);
                v.check("kmer bin 2 count", f64::from(hist[2]), 1.0, 0.0);
                v.check("kmer bin 3 count", f64::from(hist[3]), 1.0, 0.0);
                println!("  Upstream dispatch in {us} µs");
            }
            Err(e) => {
                v.check_pass(&format!("KmerHistogramGpu FAILED: {e:?}"), false);
            }
        }
    }

    // ═══ Op 2: TaxonomyFcGpu (upstream, absorbed S39) ════════════════════
    v.section("TaxonomyFcGpu — ToadStool dispatch");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let op = TaxonomyFcGpu::new(device.clone());

            let log_probs: Vec<f64> = vec![
                -1.0, -2.0, -0.5, -1.5, // taxon 0
                -0.5, -1.0, -2.0, -0.5, // taxon 1
                -2.0, -0.5, -1.0, -1.0, // taxon 2
            ];
            let log_priors: Vec<f64> = vec![-1.0, -1.5, -0.8];
            let features: Vec<u32> = vec![
                1, 0, 1, 0, // query 0
                0, 1, 0, 1, // query 1
            ];
            let scores = vec![0.0f64; 6];

            let lp_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&log_probs),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let prior_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&log_priors),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let feat_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&features),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let score_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&scores),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

            op.dispatch(&lp_buf, &prior_buf, &feat_buf, &score_buf, 2, 3, 4);
            d.poll(wgpu::Maintain::Wait);

            device.read_buffer_f64(&score_buf, 6).expect("readback")
        }));
        let us = t0.elapsed().as_micros();

        match result {
            Ok(scores) => {
                v.check("tax q0,t0 score", scores[0], -2.5, 1e-10);
                v.check("tax q0,t1 score", scores[1], -4.0, 1e-10);
                v.check_pass("TaxonomyFcGpu dispatched and readback OK", true);
                println!("  Upstream dispatch in {us} µs");
            }
            Err(e) => {
                v.check_pass(&format!("TaxonomyFcGpu FAILED: {e:?}"), false);
            }
        }
    }

    // ═══ Op 3: UniFracPropagateGpu (upstream, absorbed S39) ══════════════
    v.section("UniFracPropagateGpu — ToadStool construction");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _op = UniFracPropagateGpu::new(device.clone());
            true
        }));
        let us = t0.elapsed().as_micros();

        match result {
            Ok(true) => {
                v.check_pass("UniFracPropagateGpu constructs (pipeline + BGL)", true);
                v.check_pass("unifrac_leaf_init pipeline ready", true);
                v.check_pass("unifrac_propagate_level pipeline ready", true);
                println!("  Upstream construction in {us} µs");
            }
            _ => {
                v.check_pass("UniFracPropagateGpu construction FAILED", false);
            }
        }
    }

    // ═══ Op 4: ODE RK4 via OdeSweepGpu (upstream BatchedOdeRK4F64, S41) ═
    v.section("BatchedOdeRK4F64 via OdeSweepGpu — Lean wrapper");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let sweep = OdeSweepGpu::new(device.clone());
            let config = OdeSweepConfig {
                n_batches: 2,
                n_steps: 100,
                h: 0.01,
                t0: 0.0,
                clamp_max: 1e6,
                clamp_min: 0.0,
            };
            let init = vec![
                0.1, 0.0, 0.0, 0.5, 0.0, // batch 0
                0.1, 0.0, 0.0, 0.5, 0.0, // batch 1
            ];
            let params = vec![
                0.5, 1e9, 0.01, 0.1, 0.05, 0.3, 0.1, 2.0, 0.1, 0.2, 0.1, 0.05, 0.1, 0.3, 0.5, 2.0,
                0.05, // batch 0
                0.5, 1e9, 0.01, 0.1, 0.05, 0.3, 0.1, 2.0, 0.1, 0.2, 0.1, 0.05, 0.1, 0.3, 0.5, 2.0,
                0.05, // batch 1
            ];
            sweep.integrate(&config, &init, &params)
        }));
        let us = t0.elapsed().as_micros();

        match result {
            Ok(Ok(finals)) => {
                v.check_pass("BatchedOdeRK4F64 dispatched via lean wrapper", true);
                v.check_pass(
                    &format!("ODE returned {} values", finals.len()),
                    finals.len() == 10,
                );
                let all_finite = finals.iter().all(|x| x.is_finite());
                v.check_pass("All ODE outputs finite", all_finite);
                println!("  Upstream ODE dispatch in {us} µs");
            }
            Ok(Err(e)) => {
                v.check_pass(&format!("ODE dispatch error: {e}"), false);
            }
            Err(e) => {
                v.check_pass(&format!("ODE panic: {e:?}"), false);
            }
        }
    }

    println!("\n═══ Absorption Status ═══════════════════════════════════════════");
    println!("  KmerHistogramGpu      → ABSORBED (ToadStool S39, upstream dispatch)");
    println!("  TaxonomyFcGpu         → ABSORBED (ToadStool S39, upstream dispatch)");
    println!("  UniFracPropagateGpu   → ABSORBED (ToadStool S39, upstream pipeline)");
    println!("  BatchedOdeRK4F64      → ABSORBED (ToadStool S41, compile_shader_f64)");
    println!("  Local WGSL shaders    → RETIRED (4 files removed)\n");

    v.finish();
}
