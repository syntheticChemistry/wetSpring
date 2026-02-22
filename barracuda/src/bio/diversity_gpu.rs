// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated diversity metrics via `BarraCUDA` / `ToadStool`.
//!
//! Each function computes the same metric as its CPU counterpart in
//! [`super::diversity`], but dispatches to GPU. Results should match
//! within [`crate::tolerances::GPU_VS_CPU_F64`].
//!
//! # Shader architecture
//!
//! - **Shannon / Simpson / Observed / Evenness / Alpha**: Fused map-reduce
//!   via `ToadStool`'s `FusedMapReduceF64` — single GPU dispatch with
//!   workgroup reduction.
//!
//! - **Bray-Curtis pairs**: `ToadStool`'s `BrayCurtisF64` — absorbed from
//!   wetSpring's custom shader. One thread per pair, with CPU fallback for
//!   N < 32 samples.

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::bray_curtis_f64::BrayCurtisF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

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
/// Uses `ToadStool`'s `BrayCurtisF64` (absorbed from wetSpring's custom
/// shader). Returns N*(N-1)/2 distances in condensed order:
/// (1,0), (2,0), (2,1), (3,0), ...
///
/// `ToadStool` automatically falls back to CPU for N < 32 samples.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64`, dispatch fails,
/// or samples have inconsistent dimensions.
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

    // Flatten samples to contiguous f64 array [N*D], row-major
    let flat: Vec<f64> = samples.iter().flat_map(|s| s.iter().copied()).collect();

    let bc = BrayCurtisF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("BrayCurtisF64: {e}")))?;
    bc.condensed_distance_matrix(&flat, n, d)
        .map_err(|e| Error::Gpu(format!("Bray-Curtis GPU: {e}")))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances::GPU_VS_CPU_TRANSCENDENTAL;

    type ScalarGpuFn = fn(&GpuF64, &[f64]) -> Result<f64>;
    type MatrixGpuFn = fn(&GpuF64, &[Vec<f64>]) -> Result<Vec<f64>>;

    #[test]
    fn api_surface_compiles() {
        fn _assert_send<T: Send>() {}
        fn _assert_sync<T: Send>() {}
        let _: ScalarGpuFn = shannon_gpu;
        let _: ScalarGpuFn = simpson_gpu;
        let _: ScalarGpuFn = observed_features_gpu;
        let _: ScalarGpuFn = pielou_evenness_gpu;
        let _: MatrixGpuFn = bray_curtis_condensed_gpu;
    }

    #[test]
    fn empty_inputs_early_return() {
        // Empty-input paths don't need GPU — they return Ok(0.0) immediately.
        // These exercise the guard clauses without a device.
        // (We cannot call them directly since GpuF64 requires async init,
        //  but the logic is validated via validate_diversity_gpu binary.)
    }

    /// Shannon entropy of uniform distribution: H = ln(n).
    /// For n equal-probability species, p_i = 1/n, so -sum(p_i ln(p_i)) = ln(n).
    #[tokio::test]
    #[ignore] // requires GPU hardware
    async fn known_value_shannon_uniform() {
        let gpu = GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return; // skip if no f64 support
        }
        let n = 4_usize;
        let counts = vec![1.0; n];
        let result = shannon_gpu(&gpu, &counts).unwrap();
        let expected = (n as f64).ln();
        assert!(
            (result - expected).abs() < GPU_VS_CPU_TRANSCENDENTAL,
            "Shannon uniform: got {result}, expected {expected}"
        );
    }

    /// Simpson diversity of uniform distribution: 1 - sum(p_i^2) = (n-1)/n.
    /// For p_i = 1/n, sum(p_i^2) = n * (1/n)^2 = 1/n, so diversity = 1 - 1/n.
    #[tokio::test]
    #[ignore] // requires GPU hardware
    async fn known_value_simpson_uniform() {
        let gpu = GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return; // skip if no f64 support
        }
        let n = 4_usize;
        let counts = vec![1.0; n];
        let result = simpson_gpu(&gpu, &counts).unwrap();
        let expected = (n - 1) as f64 / n as f64;
        assert!(
            (result - expected).abs() < 1e-10,
            "Simpson uniform: got {result}, expected {expected}"
        );
    }

    /// Shannon entropy of single-species community: H = 0 (no uncertainty).
    #[tokio::test]
    #[ignore] // requires GPU hardware
    async fn known_value_shannon_single_species() {
        let gpu = GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return; // skip if no f64 support
        }
        let counts = vec![100.0, 0.0, 0.0, 0.0];
        let result = shannon_gpu(&gpu, &counts).unwrap();
        assert!(
            result.abs() < GPU_VS_CPU_TRANSCENDENTAL,
            "Shannon single-species: got {result}, expected 0"
        );
    }
}
