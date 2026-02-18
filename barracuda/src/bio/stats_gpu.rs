// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated statistical functions via `ToadStool` primitives.
//!
//! Wraps newly wired `ToadStool` orchestrators for variance, correlation,
//! covariance, and weighted dot products — useful for feature normalization,
//! QC metrics, multivariate analysis, and weighted spectral scoring.

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::correlation_f64_wgsl::CorrelationF64;
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
