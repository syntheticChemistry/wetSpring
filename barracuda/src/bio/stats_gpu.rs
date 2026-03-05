// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated statistical functions via barraCuda primitives.
//!
//! Thin wrappers around barraCuda v0.3.3 GPU ops. Fused ops
//! (`mean_variance_gpu`, `correlation_full_gpu`) use single-pass Welford
//! and 5-accumulator Pearson shaders respectively — one dispatch instead
//! of multiple. On consumer GPUs with `Fp64Strategy::Hybrid`, these
//! automatically route through DF64 core-streaming (~10x throughput).
//!
//! Cross-spring evolution: variance/correlation shaders absorbed from
//! wetSpring + hotSpring precision patterns. DF64 variants use hotSpring's
//! `df64_core.wgsl`. `FusedMapReduceF64` (Shannon, Simpson) originated
//! in wetSpring, now consumed by all springs.

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::correlation_f64_wgsl::CorrelationF64;
pub use barracuda::ops::correlation_f64_wgsl::CorrelationResult;
use barracuda::ops::covariance_f64_wgsl::CovarianceF64;
use barracuda::ops::variance_f64_wgsl::VarianceF64;
use barracuda::ops::weighted_dot_f64::WeightedDotF64;

/// GPU population variance: Var(x) = E[(x - μ)²].
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn variance_gpu(gpu: &GpuF64, data: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    let v = VarianceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("VarianceF64: {e}")))?;
    v.variance(data)
        .map_err(|e| Error::Gpu(format!("variance GPU: {e}")))
}

/// GPU sample variance: s² = Σ(x - x̄)² / (n - 1).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn sample_variance_gpu(gpu: &GpuF64, data: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    let v = VarianceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("VarianceF64: {e}")))?;
    v.sample_variance(data)
        .map_err(|e| Error::Gpu(format!("sample variance GPU: {e}")))
}

/// GPU sample standard deviation: s = √(s²).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn std_dev_gpu(gpu: &GpuF64, data: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    let v = VarianceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("VarianceF64: {e}")))?;
    v.sample_std_dev(data)
        .map_err(|e| Error::Gpu(format!("std dev GPU: {e}")))
}

/// Fused GPU mean + population variance in a single Welford pass.
///
/// Returns `[mean, variance]`. One dispatch instead of two — the Welford
/// shader accumulates both statistics in the same grid-stride + tree
/// reduction. On `Fp64Strategy::Hybrid` GPUs, this routes through the
/// DF64 fused shader (~10x throughput on consumer FP32 cores).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn mean_variance_gpu(gpu: &GpuF64, data: &[f64]) -> Result<[f64; 2]> {
    require_f64(gpu)?;
    let v = VarianceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("VarianceF64: {e}")))?;
    v.mean_variance(data, 0)
        .map_err(|e| Error::Gpu(format!("mean_variance GPU: {e}")))
}

/// Fused GPU mean + sample variance in a single Welford pass.
///
/// Returns `[mean, sample_variance]` where `sample_variance` uses `ddof=1`.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn mean_sample_variance_gpu(gpu: &GpuF64, data: &[f64]) -> Result<[f64; 2]> {
    require_f64(gpu)?;
    let v = VarianceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("VarianceF64: {e}")))?;
    v.mean_variance(data, 1)
        .map_err(|e| Error::Gpu(format!("mean_sample_variance GPU: {e}")))
}

/// GPU Pearson correlation: r(x, y) ∈ [-1, 1].
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails, vectors differ in length,
/// or device lacks `SHADER_F64`.
pub fn correlation_gpu(gpu: &GpuF64, x: &[f64], y: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    let c = CorrelationF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("CorrelationF64: {e}")))?;
    c.correlation(x, y)
        .map_err(|e| Error::Gpu(format!("correlation GPU: {e}")))
}

/// Fused GPU correlation: means, variances, and Pearson r in one dispatch.
///
/// Uses the 5-accumulator single-pass shader (Σx, Σy, Σx², Σy², Σxy).
/// Returns [`CorrelationResult`] with `mean_x`, `mean_y`, `var_x`, `var_y`,
/// and `pearson_r` — all from a single kernel launch. Replaces the pattern
/// of calling `correlation_gpu` + `covariance_gpu` + `variance_gpu`
/// separately on the same `(x, y)` pair.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails, vectors differ in length,
/// or device lacks `SHADER_F64`.
pub fn correlation_full_gpu(gpu: &GpuF64, x: &[f64], y: &[f64]) -> Result<CorrelationResult> {
    require_f64(gpu)?;
    let c = CorrelationF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("CorrelationF64: {e}")))?;
    c.correlation_full(x, y)
        .map_err(|e| Error::Gpu(format!("correlation_full GPU: {e}")))
}

/// GPU sample covariance: Cov(x, y) = Σ(x - x̄)(y - ȳ) / (n - 1).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails, vectors differ in length,
/// or device lacks `SHADER_F64`.
pub fn covariance_gpu(gpu: &GpuF64, x: &[f64], y: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    let c = CovarianceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("CovarianceF64: {e}")))?;
    c.sample_covariance(x, y)
        .map_err(|e| Error::Gpu(format!("covariance GPU: {e}")))
}

/// GPU weighted dot product: Σ wᵢ · aᵢ · bᵢ.
///
/// Useful for weighted spectral similarity, quadrature, and inner products.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn weighted_dot_gpu(gpu: &GpuF64, weights: &[f64], a: &[f64], b: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    let wd = WeightedDotF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("WeightedDotF64: {e}")))?;
    wd.weighted_dot(weights, a, b)
        .map_err(|e| Error::Gpu(format!("weighted dot GPU: {e}")))
}

/// GPU dot product: Σ aᵢ · bᵢ.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn dot_gpu(gpu: &GpuF64, a: &[f64], b: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    let wd = WeightedDotF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("WeightedDotF64: {e}")))?;
    wd.dot(a, b)
        .map_err(|e| Error::Gpu(format!("dot GPU: {e}")))
}

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if gpu.has_f64 {
        Ok(())
    } else {
        Err(Error::Gpu("SHADER_F64 not supported on this GPU".into()))
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::used_underscore_items
)]
mod tests {
    use super::*;

    #[test]
    fn api_surface_compiles() {
        fn _assert_variance(_: fn(&GpuF64, &[f64]) -> Result<f64>) {}
        fn _assert_correlation(_: fn(&GpuF64, &[f64], &[f64]) -> Result<f64>) {}
        _assert_variance(variance_gpu);
        _assert_correlation(correlation_gpu);
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn variance_gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = variance_gpu(&gpu, &data);
        assert!(result.is_ok(), "variance_gpu should succeed");
    }
}
