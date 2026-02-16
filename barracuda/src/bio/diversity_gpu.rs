// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated diversity metrics via `BarraCUDA` / `ToadStool`.
//!
//! Each function computes the same metric as its CPU counterpart in
//! [`super::diversity`], but dispatches to GPU. Results should match
//! within [`crate::tolerances::GPU_VS_CPU_F64`].
//!
//! # Shader architecture
//!
//! - **Shannon / Simpson**: Fused map-reduce via `ToadStool`'s
//!   `FusedMapReduceF64` — single GPU dispatch with workgroup reduction.
//!   Replaces the previous map-shader + CPU-sum pattern.
//!
//! - **Bray-Curtis pairs**: Custom `bray_curtis_pairs_f64.wgsl` shader,
//!   one thread per pair. The GPU computes all N*(N-1)/2 pairwise
//!   distances in parallel — the primary GPU acceleration target.

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

/// Uniform params for the Bray-Curtis pairs shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BcParams {
    n_samples: u32,
    n_features: u32,
    n_pairs: u32,
    _pad: u32,
}

/// Shannon entropy: H = -sum(p\_i \* ln(p\_i)), computed on GPU.
///
/// Uses `ToadStool`'s `FusedMapReduceF64` for single-dispatch map-reduce.
/// Returns same result as [`super::diversity::shannon`] within GPU f64
/// tolerance.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn shannon_gpu(gpu: &GpuF64, counts: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    if counts.is_empty() {
        return Ok(0.0);
    }
    let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;
    fmr.shannon_entropy(counts)
        .map_err(|e| Error::Gpu(format!("Shannon GPU: {e}")))
}

/// Simpson diversity: 1 - sum(p\_i^2), computed on GPU.
///
/// Uses `ToadStool`'s `FusedMapReduceF64` for single-dispatch map-reduce.
/// `ToadStool`'s `simpson_index` returns `Σ p²` (dominance); we subtract
/// from 1 to match the diversity convention used by our CPU
/// [`super::diversity::simpson`].
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn simpson_gpu(gpu: &GpuF64, counts: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    if counts.is_empty() {
        return Ok(0.0);
    }
    let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;
    let dominance = fmr
        .simpson_index(counts)
        .map_err(|e| Error::Gpu(format!("Simpson GPU: {e}")))?;
    // ToadStool returns Σ p² (dominance); convert to diversity = 1 - Σ p²
    Ok(1.0 - dominance)
}

/// All-pairs Bray-Curtis condensed distance matrix, computed on GPU.
///
/// Dispatches `bray_curtis_pairs_f64.wgsl` with one thread per pair.
/// Returns N*(N-1)/2 distances in condensed order:
/// (1,0), (2,0), (2,1), (3,0), ...
///
/// This is the primary GPU acceleration target. For N=1000 samples
/// of D=2000 features: 500K pairs × 2000 features = 1B operations,
/// embarrassingly parallel.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64`, dispatch fails,
/// or samples have inconsistent dimensions.
#[allow(clippy::cast_possible_truncation)]
pub fn bray_curtis_condensed_gpu(gpu: &GpuF64, samples: &[Vec<f64>]) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let n = samples.len();
    if n < 2 {
        return Ok(vec![]);
    }

    let d = samples[0].len();
    for s in samples {
        if s.len() != d {
            return Err(Error::Gpu(
                "all samples must have the same number of features".into(),
            ));
        }
    }

    let n_pairs = n * (n - 1) / 2;

    // Flatten samples to contiguous f64 array [N*D], row-major
    let flat: Vec<f64> = samples.iter().flat_map(|s| s.iter().copied()).collect();

    let samples_buf = gpu.create_f64_buffer(&flat, "bc_samples");
    let output_buf = gpu.create_f64_output_buffer(n_pairs, "bc_output");

    let params = BcParams {
        n_samples: n as u32,
        n_features: d as u32,
        n_pairs: n_pairs as u32,
        _pad: 0,
    };
    let params_buf = gpu.create_uniform_buffer(&params, "bc_params");

    let shader_source = include_str!("../shaders/bray_curtis_pairs_f64.wgsl");
    let pipeline = gpu.create_pipeline(shader_source, "bray_curtis_pairs");

    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bc_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: samples_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let workgroups = n_pairs.div_ceil(256) as u32;
    gpu.dispatch_and_read(&pipeline, &bind_group, workgroups, &output_buf, n_pairs)
}

/// Observed features (count of non-zero entries), computed on GPU.
///
/// Uses `FusedMapReduceF64` with identity map + sum reduce on a
/// binarized version of the input (CPU preprocessing, GPU reduction).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn observed_features_gpu(gpu: &GpuF64, counts: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    if counts.is_empty() {
        return Ok(0.0);
    }
    let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;
    // Binarize: 1.0 for non-zero, 0.0 for zero
    let binary: Vec<f64> = counts
        .iter()
        .map(|&c| if c > 0.0 { 1.0 } else { 0.0 })
        .collect();
    fmr.sum(&binary)
        .map_err(|e| Error::Gpu(format!("observed_features GPU: {e}")))
}

/// Pielou's evenness: J' = H / ln(S), computed on GPU.
///
/// Combines GPU Shannon entropy with GPU observed features.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn pielou_evenness_gpu(gpu: &GpuF64, counts: &[f64]) -> Result<f64> {
    let s = observed_features_gpu(gpu, counts)?;
    if s <= 1.0 {
        return Ok(0.0);
    }
    let h = shannon_gpu(gpu, counts)?;
    Ok(h / s.ln())
}

/// Full alpha diversity computed on GPU.
///
/// Returns observed features, Shannon, Simpson, and Pielou evenness
/// all via GPU dispatch. Chao1 remains CPU since it requires counting
/// exact singletons/doubletons (integer comparison, not GPU-friendly).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn alpha_diversity_gpu(
    gpu: &GpuF64,
    counts: &[f64],
) -> Result<super::diversity::AlphaDiversity> {
    let observed = observed_features_gpu(gpu, counts)?;
    let shannon = shannon_gpu(gpu, counts)?;
    let simpson = simpson_gpu(gpu, counts)?;
    let chao1 = super::diversity::chao1(counts); // CPU — integer counting
    let evenness = if observed > 1.0 {
        shannon / observed.ln()
    } else {
        0.0
    };

    Ok(super::diversity::AlphaDiversity {
        observed,
        shannon,
        simpson,
        chao1,
        evenness,
    })
}

// ───────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if gpu.has_f64 {
        Ok(())
    } else {
        Err(Error::Gpu("SHADER_F64 not supported on this GPU".into()))
    }
}
