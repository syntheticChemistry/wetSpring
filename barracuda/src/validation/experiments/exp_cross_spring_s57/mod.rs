// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exp162: Cross-Spring S57 Evolution — Rewire + Validate + Benchmark
//!
//! # Provenance
//!
//! | Script  | `validate_cross_spring_s57` |
//! | Commit  | 1f9f80e |
//! | Command | `cargo run --release --features gpu --bin validate_cross_spring_s57` |
//! | Hardware| RTX 4070 (Ada, f64 1:2), Titan V (Volta GV100, NVK) |
//!
//! # Purpose
//!
//! Validates and benchmarks `ToadStool` S54-S57 primitives evolved by
//! neuralSpring and airSpring, now available to wetSpring. Demonstrates
//! the cross-spring evolution model: each Spring writes domain-specific
//! code, `ToadStool` absorbs it, and all Springs benefit.
//!
//! Validation class: Cross-spring
//!
//! Provenance: Validates across multiple primals/springs (hotSpring, wetSpring, neuralSpring, etc.)

mod gpu_regression;
mod s54_primitives;
mod s56_primitives;

use std::sync::Arc;

use crate::validation::{self, CrossSpringEntry, OrExit, dense_to_csr};
use barracuda::spectral::{lanczos, lanczos_eigenvalues};

fn eigenvalues_from_dense(matrix: &[f64], n: usize) -> Vec<f64> {
    let csr = dense_to_csr(matrix, n);
    lanczos_eigenvalues(&lanczos(&csr, n, 42))
}

fn readback_f32(
    device: &Arc<barracuda::device::WgpuDevice>,
    buf: &wgpu::Buffer,
    n: usize,
) -> Vec<f32> {
    let d = device.device();
    let q = device.queue();
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, (n * 4) as u64);
    q.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).or_exit("channel send");
    });
    let _ = d.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().or_exit("channel recv").or_exit("GPU buffer map");

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

/// Run the `validate_cross_spring_s57` experiment, recording checks into `v`.
pub fn run(v: &mut crate::validation::Validator) {
    let __rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    __rt.block_on(async {
        println!("════════════════════════════════════════════════════════════════════");
        println!("  Exp162: Cross-Spring S57 Evolution — Rewire + Validate + Benchmark");
        println!("  Proving: neuralSpring graph/MCMC primitives benefit wetSpring bio");
        println!("════════════════════════════════════════════════════════════════════\n");

        let gpu = validation::gpu_or_skip().await;
        let device = gpu.to_wgpu_device();

        let mut bench_results: Vec<CrossSpringEntry> = Vec::new();

        s54_primitives::validate(v, &mut bench_results);
        s56_primitives::validate(v, &mut bench_results);
        gpu_regression::validate_compound(v, &mut bench_results);
        gpu_regression::validate_gpu_primitives(v, &device);
        gpu_regression::print_summary(&bench_results);
    });
}

/// Bridge into [`primalspring::validation::ValidationResult`] for UniBin dispatch.
pub fn run_as_scenario(result: &mut primalspring::validation::ValidationResult) {
    let mut v = crate::validation::Validator::silent("validate_cross_spring_s57");
    run(&mut v);
    v.bridge_into(result);
}

/// Scenario registration for the UniBin registry.
pub const SCENARIO: crate::validation::scenarios::registry::Scenario =
    crate::validation::scenarios::registry::Scenario {
        meta: crate::validation::scenarios::registry::ScenarioMeta {
            id: "cross_spring_s57",
            track: crate::validation::scenarios::registry::Track::Science,
            tier: crate::validation::scenarios::registry::Tier::Both,
            provenance_crate: "validate_cross_spring_s57",
            provenance_date: "2026-05-20",
            description: "Exp162: Cross-Spring S57 Evolution — Rewire + Validate + Benchmark",
        },
        run: |v, _ctx| run_as_scenario(v),
    };
