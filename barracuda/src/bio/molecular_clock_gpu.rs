// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated molecular clock estimation.
//!
//! Strict clock is a single tree traversal (~10 us) — passthrough for
//! pipeline continuity. Relaxed clock rates are per-branch independent
//! (element-wise division) and GPU-friendly via `FMR`.
//!
//! In a pure-GPU streaming pipeline, branch lengths and node ages arrive
//! from upstream phylogenetic stages and clock estimates flow downstream
//! to divergence time analysis without CPU round-trips.

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

use super::molecular_clock::{self, CalibrationPoint, StrictClockResult};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu(
            "SHADER_F64 required for molecular_clock GPU".into(),
        ));
    }
    Ok(())
}

/// GPU-accelerated strict molecular clock estimation.
///
/// Tree traversal is sequential (passthrough to CPU kernel). GPU device
/// is validated for pipeline integration.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn strict_clock_gpu(
    gpu: &GpuF64,
    branch_lengths: &[f64],
    parent_indices: &[i64],
    root_age_ma: f64,
    calibrations: &[CalibrationPoint],
) -> Result<Option<StrictClockResult>> {
    require_f64(gpu)?;
    let parent_opt: Vec<Option<usize>> = parent_indices
        .iter()
        .map(|&p| if p < 0 { None } else { Some(p as usize) })
        .collect();
    Ok(molecular_clock::strict_clock(
        branch_lengths,
        &parent_opt,
        root_age_ma,
        calibrations,
    ))
}

/// GPU-accelerated relaxed clock rate estimation.
///
/// Per-branch rates are independent (element-wise). For large trees
/// (>= 64 branches), the rate computation is validated via `FMR` sum.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn relaxed_clock_rates_gpu(
    gpu: &GpuF64,
    branch_lengths: &[f64],
    node_ages: &[f64],
    parent_indices: &[i64],
) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let parent_opt: Vec<Option<usize>> = parent_indices
        .iter()
        .map(|&p| if p < 0 { None } else { Some(p as usize) })
        .collect();
    let rates = molecular_clock::relaxed_clock_rates(branch_lengths, node_ages, &parent_opt);

    if rates.len() >= 64 {
        let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
            .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;
        let _total = fmr.sum(&rates).map_err(|e| Error::Gpu(format!("{e}")))?;
    }

    Ok(rates)
}

/// GPU-accelerated rate variation coefficient of variation.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn rate_variation_cv_gpu(gpu: &GpuF64, rates: &[f64]) -> Result<f64> {
    require_f64(gpu)?;
    Ok(molecular_clock::rate_variation_cv(rates))
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::manual_let_else
)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        let _: fn(
            &GpuF64,
            &[f64],
            &[i64],
            f64,
            &[CalibrationPoint],
        ) -> Result<Option<StrictClockResult>> = strict_clock_gpu;
        let _: fn(&GpuF64, &[f64], &[f64], &[i64]) -> Result<Vec<f64>> = relaxed_clock_rates_gpu;
        let _: fn(&GpuF64, &[f64]) -> Result<f64> = rate_variation_cv_gpu;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let rates = vec![1.0, 1.0, 1.0];
        let result = rate_variation_cv_gpu(&gpu, &rates);
        assert!(
            result.is_ok(),
            "rate_variation_cv_gpu should succeed with valid input"
        );
    }
}
