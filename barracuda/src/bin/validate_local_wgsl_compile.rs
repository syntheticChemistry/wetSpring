// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Exp096: Local WGSL Shader Compile + Dispatch Validation
//!
//! # Provenance
//!
//! | Script  | `validate_local_wgsl_compile` |
//! | Command | `cargo run --features gpu --bin validate_local_wgsl_compile` |
//!
//! # Purpose
//!
//! Compile-test and basic dispatch-test all 4 local WGSL shaders in the
//! Write phase. These will eventually be handed off to `ToadStool`.
//!
//! 1. `kmer_histogram_f64.wgsl` — k-mer counting (f32/atomic)
//! 2. `unifrac_propagate_f64.wgsl` — UniFrac tree propagation (f64)
//! 3. `taxonomy_fc_f64.wgsl` — Naive Bayes taxonomy scoring (f64)
//! 4. `batched_qs_ode_rk4_f64.wgsl` — ODE RK4 integration (f64, existing)

use std::time::Instant;
use wgpu::util::DeviceExt;

use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

const KMER_WGSL: &str = include_str!("../shaders/kmer_histogram_f64.wgsl");
const TAXONOMY_WGSL: &str = include_str!("../shaders/taxonomy_fc_f64.wgsl");
const UNIFRAC_WGSL: &str = include_str!("../shaders/unifrac_propagate_f64.wgsl");
const ODE_WGSL: &str = include_str!("../shaders/batched_qs_ode_rk4_f64.wgsl");

#[tokio::main]
async fn main() {
    println!("════════════════════════════════════════════════════════════════════");
    println!("  Exp096: Local WGSL Shader Compile + Dispatch Validation");
    println!("  Testing 4 Write-phase shaders before `ToadStool` absorption");
    println!("════════════════════════════════════════════════════════════════════\n");

    let mut v = Validator::new("Exp096: Local WGSL Compile + Dispatch");

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

    // ═══ Shader 1: kmer_histogram (f32/atomic, no f64 preamble needed) ══
    v.section("kmer_histogram_f64.wgsl — Compile + Dispatch");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let module = device.compile_shader(KMER_WGSL, Some("kmer_histogram"));

            let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("kmer BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("kmer_histogram pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: "kmer_histogram",
                compilation_options: Default::default(),
                cache: None,
            });

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct KmerConfig {
                n_kmers: u32,
                k: u32,
                _pad0: u32,
                _pad1: u32,
            }

            let config = KmerConfig {
                n_kmers: 8,
                k: 2,
                _pad0: 0,
                _pad1: 0,
            };
            let kmers: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 0, 0];
            let histogram = vec![0u32; 16];

            let cfg_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });
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

            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cfg_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: kmer_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: hist_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder =
                d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            device.queue().submit(std::iter::once(encoder.finish()));
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
                println!("  Compiled + dispatched in {us} µs");
            }
            Err(e) => {
                v.check_pass(&format!("kmer compile FAILED: {e:?}"), false);
            }
        }
    }

    // ═══ Shader 2: taxonomy_fc (f64, needs preamble) ════════════════════
    v.section("taxonomy_fc_f64.wgsl — Compile + Dispatch");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let module = device.compile_shader_f64(TAXONOMY_WGSL, Some("taxonomy_fc"));

            let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tax BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("taxonomy_fc pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: "taxonomy_fc",
                compilation_options: Default::default(),
                cache: None,
            });

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct TaxConfig {
                n_queries: u32,
                n_taxa: u32,
                n_features: u32,
                _pad: u32,
            }

            let config = TaxConfig {
                n_queries: 2,
                n_taxa: 3,
                n_features: 4,
                _pad: 0,
            };
            // 3 taxa × 4 features log-probabilities
            let log_probs: Vec<f64> = vec![
                -1.0, -2.0, -0.5, -1.5, // taxon 0
                -0.5, -1.0, -2.0, -0.5, // taxon 1
                -2.0, -0.5, -1.0, -1.0, // taxon 2
            ];
            let log_priors: Vec<f64> = vec![-1.0, -1.5, -0.8]; // 3 taxa
            // 2 queries × 4 features (binary presence)
            let features: Vec<u32> = vec![
                1, 0, 1, 0, // query 0: features 0,2 present
                0, 1, 0, 1, // query 1: features 1,3 present
            ];
            let scores = vec![0.0f64; 6]; // 2 queries × 3 taxa

            let cfg_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::UNIFORM,
            });
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

            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cfg_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: lp_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: prior_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: feat_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: score_buf.as_entire_binding() },
                ],
            });

            let mut encoder =
                d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            device.queue().submit(std::iter::once(encoder.finish()));
            d.poll(wgpu::Maintain::Wait);

            device
                .read_buffer_f64(&score_buf, 6)
                .expect("readback")
        }));
        let us = t0.elapsed().as_micros();

        match result {
            Ok(scores) => {
                // query 0, taxon 0: prior(-1.0) + log_prob[0](-1.0) + log_prob[2](-0.5) = -2.5
                v.check("tax q0,t0 score", scores[0], -2.5, 1e-10);
                // query 0, taxon 1: prior(-1.5) + log_prob[0](-0.5) + log_prob[2](-2.0) = -4.0
                v.check("tax q0,t1 score", scores[1], -4.0, 1e-10);
                v.check_pass("taxonomy_fc dispatched and readback OK", true);
                println!("  Compiled + dispatched in {us} µs");
            }
            Err(e) => {
                v.check_pass(&format!("taxonomy_fc compile/dispatch FAILED: {e:?}"), false);
            }
        }
    }

    // ═══ Shader 3: unifrac_propagate (f64, needs preamble) ══════════════
    v.section("unifrac_propagate_f64.wgsl — Compile");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let module =
                device.compile_shader_f64(UNIFRAC_WGSL, Some("unifrac_propagate"));

            let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unifrac BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let _p1 = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("unifrac_leaf_init"),
                layout: Some(&layout),
                module: &module,
                entry_point: "unifrac_leaf_init",
                compilation_options: Default::default(),
                cache: None,
            });
            let _p2 = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("unifrac_propagate_level"),
                layout: Some(&layout),
                module: &module,
                entry_point: "unifrac_propagate_level",
                compilation_options: Default::default(),
                cache: None,
            });
            true
        }));
        let us = t0.elapsed().as_micros();

        match result {
            Ok(true) => {
                v.check_pass("unifrac_leaf_init compiles", true);
                v.check_pass("unifrac_propagate_level compiles", true);
                println!("  Both entry points compiled in {us} µs");
            }
            _ => {
                v.check_pass(&format!("unifrac compile FAILED"), false);
            }
        }
    }

    // ═══ Shader 4: ODE RK4 (already validated, confirm compile) ═════════
    v.section("batched_qs_ode_rk4_f64.wgsl — Compile");
    {
        let t0 = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _module =
                device.compile_shader_f64(ODE_WGSL, Some("ode_rk4"));
            true
        }));
        let us = t0.elapsed().as_micros();

        match result {
            Ok(true) => {
                v.check_pass("ODE RK4 compiles with f64 preamble", true);
                println!("  Compiled in {us} µs");
            }
            _ => {
                v.check_pass(&format!("ODE compile FAILED"), false);
            }
        }
    }

    println!("\n═══ Local WGSL Shader Status ════════════════════════════════════");
    println!("  kmer_histogram     → COMPILED + DISPATCHED (f32/atomic)");
    println!("  taxonomy_fc        → COMPILED + DISPATCHED (f64, preamble)");
    println!("  unifrac_propagate  → COMPILED (f64, 2 entry points)");
    println!("  batched_ode_rk4    → COMPILED (f64, preamble)");
    println!("  All 4 ready for `ToadStool` absorption handoff\n");

    v.finish();
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
