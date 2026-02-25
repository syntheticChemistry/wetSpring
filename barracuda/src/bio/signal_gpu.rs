// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated signal processing and peak detection.
//!
//! Peak detection involves local maxima identification (parallel neighbor
//! comparison), prominence computation (left/right valley scan), and width
//! interpolation. The batch dimension (multiple signals in a pipeline) is
//! embarrassingly parallel.
//!
//! Delegates to `ToadStool`'s `PeakDetectF64` (S62) for native GPU dispatch
//! with WGSL parallel local-maxima + prominence shader. Falls back to CPU
//! `find_peaks` for signals shorter than 64 elements.

use barracuda::ops::peak_detect_f64::PeakDetectF64;

use super::signal::{self, Peak, PeakParams};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for signal GPU".into()));
    }
    Ok(())
}

const fn upstream_to_local(det: &barracuda::ops::peak_detect_f64::DetectedPeak) -> Peak {
    Peak {
        index: det.index,
        height: det.height,
        prominence: det.prominence,
        left_base: 0,
        right_base: 0,
        width: det.width,
        left_ips: 0.0,
        right_ips: 0.0,
    }
}

/// GPU-accelerated peak detection on a single signal.
///
/// Uses `ToadStool`'s `PeakDetectF64` WGSL shader for parallel local-maxima
/// detection and prominence computation. Falls back to CPU for short signals.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support or GPU dispatch fails.
pub fn find_peaks_gpu(gpu: &GpuF64, data: &[f64], params: &PeakParams) -> Result<Vec<Peak>> {
    require_f64(gpu)?;

    if data.len() < 64 {
        return Ok(signal::find_peaks(data, params));
    }

    let mut builder = PeakDetectF64::new(data, params.distance);
    if let Some(h) = params.min_height {
        builder = builder.height(h);
    }
    if let Some(p) = params.min_prominence {
        builder = builder.prominence(p);
    }
    if let Some(w) = params.min_width {
        builder = builder.width(w);
    }

    Ok(builder.execute(&gpu.to_wgpu_device()).map_or_else(
        |_| signal::find_peaks(data, params),
        |detected| detected.iter().map(upstream_to_local).collect(),
    ))
}

/// GPU-accelerated batch peak detection across multiple signals.
///
/// Each signal dispatches to `PeakDetectF64` independently.
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

    signals
        .iter()
        .map(|s| find_peaks_gpu(gpu, s, params))
        .collect()
}
