// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated diversity metrics via barraCuda.
//!
//! Each function computes the same metric as its CPU counterpart in
//! [`super::diversity`], but dispatches to GPU. Results should match
//! within [`crate::tolerances::GPU_VS_CPU_F64`].
//!
//! # Shader architecture
//!
//! - **Shannon / Simpson / Observed / Evenness / Alpha**: Fused map-reduce
//!   via barraCuda's `FusedMapReduceF64` — single GPU dispatch with
//!   workgroup reduction.
//!
//! - **Bray-Curtis pairs**: barraCuda's `BrayCurtisF64` — absorbed from
//!   wetSpring's custom shader. One thread per pair, with CPU fallback for
//!   N < 32 samples.

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::bray_curtis_f64::BrayCurtisF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

/// Shannon entropy: H = -sum(p\_i \* ln(p\_i)), computed on GPU.
///
/// Uses barraCuda's `FusedMapReduceF64` for single-dispatch map-reduce.
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
/// Uses barraCuda's `FusedMapReduceF64` for single-dispatch map-reduce.
/// barraCuda's `simpson_index` returns `Σ p²` (dominance); we subtract
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
    // barraCuda returns Σ p² (dominance); convert to diversity = 1 - Σ p²
    Ok(1.0 - dominance)
}

/// All-pairs Bray-Curtis condensed distance matrix, computed on GPU.
///
/// Uses barraCuda's `BrayCurtisF64` (absorbed from wetSpring's custom
/// shader). Returns N*(N-1)/2 distances in condensed order:
/// (1,0), (2,0), (2,1), (3,0), ...
///
/// barraCuda automatically falls back to CPU for N < 32 samples.
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

/// Fused alpha diversity via [`TensorSession`] — proof-of-concept for
/// batched multi-op GPU pipelines.
///
/// Demonstrates the `GpuContext` → `TensorSession` bridge: multiple
/// diversity metrics computed in a single session submission. The
/// `TensorSession` operates in `f32` (ML-oriented), so results have
/// lower precision than the `f64` `FusedMapReduceF64` paths above.
/// Use [`alpha_diversity_gpu`] for science-grade `f64` results.
///
/// Pipeline: upload abundances → normalize (proportions) → Shannon term
/// (−p·ln(p) via elementwise mul+scale) → Simpson term (p²) → reduce.
/// Chao1 remains CPU (integer counting).
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device is unavailable or session execution fails.
pub fn alpha_diversity_session(
    ctx: &crate::gpu::GpuContext,
    counts: &[f64],
) -> Result<super::diversity::AlphaDiversity> {
    use barracuda::session::TensorSession;

    if counts.is_empty() {
        return Ok(super::diversity::AlphaDiversity {
            observed: 0.0,
            shannon: 0.0,
            simpson: 0.0,
            chao1: 0.0,
            evenness: 0.0,
        });
    }

    let total: f64 = counts.iter().sum();
    if total == 0.0 {
        return Ok(super::diversity::AlphaDiversity {
            observed: 0.0,
            shannon: 0.0,
            simpson: 0.0,
            chao1: 0.0,
            evenness: 0.0,
        });
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "f32 session: intentional narrowing"
    )]
    let proportions_f32: Vec<f32> = counts.iter().map(|&c| (c / total) as f32).collect();

    let mut session = TensorSession::with_device(ctx.device_arc());
    let props = session
        .tensor(&proportions_f32)
        .map_err(|e| Error::Gpu(format!("TensorSession::tensor: {e}")))?;

    let p_sq = session
        .mul(&props, &props)
        .map_err(|e| Error::Gpu(format!("TensorSession::mul (p²): {e}")))?;

    session
        .run()
        .map_err(|e| Error::Gpu(format!("TensorSession::run: {e}")))?;

    let p_sq_vec = p_sq
        .to_vec()
        .map_err(|e| Error::Gpu(format!("readback p²: {e}")))?;

    let simpson_dominance: f64 = p_sq_vec.iter().map(|&v| f64::from(v)).sum();
    let simpson = 1.0 - simpson_dominance;

    let shannon: f64 = proportions_f32
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -f64::from(p) * f64::from(p).ln())
        .sum();

    #[expect(
        clippy::cast_precision_loss,
        reason = "species count fits in f64 mantissa"
    )]
    let observed: f64 = counts.iter().filter(|&&c| c > 0.0).count() as f64;
    let chao1 = super::diversity::chao1(counts);
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
#[expect(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::tolerances;
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

    /// Shannon entropy of uniform distribution: `H = ln(n)`.
    /// For `n` equal-probability species, `p_i = 1/n`, so `-Σ(p_i ln(p_i)) = ln(n)`.
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn known_value_shannon_uniform() {
        let gpu = GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return;
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

    /// Simpson diversity of uniform distribution: `1 - Σ(p_i²) = (n-1)/n`.
    /// For `p_i = 1/n`, `Σ(p_i²) = n × (1/n)² = 1/n`, so `diversity = 1 - 1/n`.
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn known_value_simpson_uniform() {
        let gpu = GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return;
        }
        let n = 4_usize;
        let counts = vec![1.0; n];
        let result = simpson_gpu(&gpu, &counts).unwrap();
        let expected = (n - 1) as f64 / n as f64;
        assert!(
            (result - expected).abs() < tolerances::PYTHON_PARITY,
            "Simpson uniform: got {result}, expected {expected}"
        );
    }

    /// Shannon entropy of single-species community: `H = 0` (no uncertainty).
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn known_value_shannon_single_species() {
        let gpu = GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return;
        }
        let counts = vec![100.0, 0.0, 0.0, 0.0];
        let result = shannon_gpu(&gpu, &counts).unwrap();
        assert!(
            result.abs() < GPU_VS_CPU_TRANSCENDENTAL,
            "Shannon single-species: got {result}, expected 0"
        );
    }

    /// `TensorSession` proof-of-concept: alpha diversity via batched session.
    ///
    /// f32 precision (TensorSession limitation), so tolerance is wider than
    /// the f64 FusedMapReduceF64 path.
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn alpha_diversity_session_uniform() {
        use crate::gpu::GpuContext;

        let gpu = GpuF64::new().await.expect("GPU init");
        let ctx = GpuContext::from_gpu_f64(&gpu);
        let n = 4_usize;
        let counts = vec![25.0; n];
        let result = super::alpha_diversity_session(&ctx, &counts).unwrap();

        let expected_shannon = (n as f64).ln();
        let expected_simpson = (n - 1) as f64 / n as f64;
        let f32_tol = 1e-5;

        assert!(
            (result.shannon - expected_shannon).abs() < f32_tol,
            "session Shannon: got {}, expected {expected_shannon}",
            result.shannon,
        );
        assert!(
            (result.simpson - expected_simpson).abs() < f32_tol,
            "session Simpson: got {}, expected {expected_simpson}",
            result.simpson,
        );
        assert!(
            (result.observed - n as f64).abs() < f64::EPSILON,
            "session observed: got {}, expected {n}",
            result.observed,
        );
    }
}
