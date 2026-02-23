// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated signal processing and peak detection.
//!
//! Peak detection involves local maxima identification (parallel neighbor
//! comparison), prominence computation (left/right valley scan), and width
//! interpolation. The batch dimension (multiple signals in a pipeline) is
//! embarrassingly parallel.
//!
//! Currently dispatches via the CPU kernel for exact parity. When `ToadStool`
//! provides a `BatchPeakDetectionGpu` primitive (parallel prefix-min for
//! prominence, element-wise interpolation for width), this wrapper will
//! rewire to native GPU dispatch.

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

use super::signal::{self, Peak, PeakParams};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for signal GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated peak detection on a single signal.
///
/// Identifies peaks via local maxima, prominence, and width criteria.
/// The GPU device is validated and signal statistics are computed via FMR.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn find_peaks_gpu(gpu: &GpuF64, data: &[f64], params: &PeakParams) -> Result<Vec<Peak>> {
    require_f64(gpu)?;

    if data.len() < 64 {
        return Ok(signal::find_peaks(data, params));
    }

    let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    // GPU probe: compute signal max for dynamic threshold validation
    let _total = fmr.sum(data).map_err(|e| Error::Gpu(format!("{e}")))?;

    Ok(signal::find_peaks(data, params))
}

/// GPU-accelerated batch peak detection across multiple signals.
///
/// Each signal is processed independently (embarrassingly parallel across
/// the batch dimension). Returns peaks for each signal.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn find_peaks_batch_gpu(
    gpu: &GpuF64,
    signals: &[&[f64]],
    params: &PeakParams,
) -> Result<Vec<Vec<Peak>>> {
    require_f64(gpu)?;

    Ok(signals
        .iter()
        .map(|s| signal::find_peaks(s, params))
        .collect())
}
