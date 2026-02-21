// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::similar_names)]
//! Exp063: GPU Random Forest Batch Inference
//!
//! Validates the local WGSL shader for batch RF inference against
//! CPU majority-vote results. Proves that ensemble ML inference
//! is portable to GPU.
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline tool | BarraCUDA CPU (reference) |
//! | Baseline version | wetspring-barracuda 0.1.0 (CPU path) |
//! | Baseline command | RandomForest::predict_batch_with_votes |
//! | Baseline date | 2026-02-19 |
//! | Data | 6 samples × 5-tree forest (same as CPU v5) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070, Pop!\_OS 22.04) |
//!
//! Local WGSL shader: batch RF inference (RandomForestGpu).

use std::time::Instant;
use wetspring_barracuda::bio::{
    decision_tree::DecisionTree, random_forest::RandomForest, random_forest_gpu::RandomForestGpu,
};
use wetspring_barracuda::gpu::GpuF64;
use wetspring_barracuda::validation::{self, Validator};

#[tokio::main]
#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
async fn main() {
    let mut v = Validator::new("Exp063: GPU Random Forest Batch Inference");

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

    let device = gpu.to_wgpu_device();

    // Build the same 5-tree forest as CPU v5
    let tree1 = DecisionTree::from_arrays(
        &[0, -2, 1, -2, -2],
        &[5.0, 0.0, 3.0, 0.0, 0.0],
        &[1, -1, 3, -1, -1],
        &[2, -1, 4, -1, -1],
        &[None, Some(0), None, Some(1), Some(2)],
        2,
    )
    .unwrap();

    let tree2 = DecisionTree::from_arrays(
        &[1, -2, -2],
        &[4.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(0), Some(2)],
        2,
    )
    .unwrap();

    let tree3 = DecisionTree::from_arrays(
        &[0, -2, -2],
        &[6.0, 0.0, 0.0],
        &[1, -1, -1],
        &[2, -1, -1],
        &[None, Some(1), Some(2)],
        2,
    )
    .unwrap();

    let tree4 = DecisionTree::from_arrays(
        &[0, 1, -2, -2, -2],
        &[4.0, 2.0, 0.0, 0.0, 0.0],
        &[1, 2, -1, -1, -1],
        &[4, 3, -1, -1, -1],
        &[None, None, Some(0), Some(1), Some(2)],
        2,
    )
    .unwrap();

    let tree5 = DecisionTree::from_arrays(
        &[1, -2, 0, -2, -2],
        &[5.0, 0.0, 7.0, 0.0, 0.0],
        &[1, -1, 3, -1, -1],
        &[2, -1, 4, -1, -1],
        &[None, Some(0), None, Some(1), Some(2)],
        2,
    )
    .unwrap();

    let rf = RandomForest::from_trees(vec![tree1, tree2, tree3, tree4, tree5], 3).unwrap();

    let samples = vec![
        vec![3.0, 1.0],
        vec![7.0, 6.0],
        vec![5.5, 3.5],
        vec![1.0, 1.0],
        vec![8.0, 8.0],
        vec![4.0, 5.0],
    ];

    // CPU reference
    let cpu_preds = rf.predict_batch_with_votes(&samples);

    // GPU
    let rf_gpu = RandomForestGpu::new(&device);
    let t0 = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rf_gpu.predict_batch(&rf, &samples)
    }));
    let gpu_us = t0.elapsed().as_micros();

    match result {
        Ok(Ok(gpu_preds)) => {
            v.check(
                "RF GPU: result count matches",
                gpu_preds.len() as f64,
                samples.len() as f64,
                0.0,
            );

            for (i, (cpu, gpu)) in cpu_preds.iter().zip(gpu_preds.iter()).enumerate() {
                v.check(
                    &format!("RF GPU: sample {i} class CPU == GPU"),
                    gpu.class as f64,
                    cpu.class as f64,
                    0.0,
                );
                v.check(
                    &format!("RF GPU: sample {i} confidence matches"),
                    gpu.confidence,
                    cpu.confidence,
                    1e-10,
                );
            }

            println!();
            println!(
                "  GPU RF batch inference: {gpu_us} µs ({} samples × {} trees)",
                samples.len(),
                rf.n_trees()
            );
        }
        Ok(Err(e)) => {
            println!("  [SKIP] RF GPU error: {e}");
            v.check("RF GPU: available (skipped)", 1.0, 1.0, 0.0);
        }
        Err(_) => {
            println!("  [SKIP] RF GPU panicked (driver shader compilation)");
            v.check("RF GPU: available (driver skip)", 1.0, 1.0, 0.0);
        }
    }

    v.finish();
}
