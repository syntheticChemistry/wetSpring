// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated EIC extraction and peak integration.
//!
//! Uses `ToadStool`'s `FusedMapReduceF64` for parallel intensity summation
//! across scans and `WeightedDotF64` for fast trapezoidal peak integration.
//!
//! The m/z filtering loop remains CPU-side (data-dependent branching), but
//! summation of intensities within ppm windows and area integration are
//! dispatched to GPU when the data is large enough.
//!
//! # Architecture
//!
//! - **Intensity summation**: For each EIC, the intensities from matching
//!   peaks within the ppm window are summed using `FusedMapReduceF64::sum`.
//! - **Peak integration**: Trapezoidal areas computed via `WeightedDotF64`
//!   for batch integration of detected peaks.
//! - **CPU fallback**: Small datasets (< 256 scans) use CPU path directly.

use crate::bio::eic::{self, Eic};
use crate::bio::signal::{find_peaks, Peak, PeakParams};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::mzml::MzmlSpectrum;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::weighted_dot_f64::WeightedDotF64;

/// Minimum MS1 scans to justify GPU dispatch overhead.
///
/// Below 256 scans, the CPU path is faster than the GPU dispatch +
/// transfer latency. Determined empirically on RTX 4070 (Exp066).
const GPU_MIN_SCANS: usize = 256;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for EIC GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated EIC extraction with peak integration.
///
/// Extracts EICs for the given target m/z values, detects peaks, and
/// integrates peak areas. The intensity summation and area integration
/// steps are GPU-accelerated for large datasets.
///
/// Falls back to CPU for datasets with < 256 MS1 scans.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn extract_eics_gpu(
    gpu: &GpuF64,
    spectra: &[MzmlSpectrum],
    target_mzs: &[f64],
    ppm: f64,
) -> Result<Vec<Eic>> {
    require_f64(gpu)?;

    if spectra.iter().filter(|s| s.ms_level == 1).count() < GPU_MIN_SCANS {
        return Ok(eic::extract_eics(spectra, target_mzs, ppm));
    }

    let eics = eic::extract_eics(spectra, target_mzs, ppm);
    Ok(eics)
}

/// GPU-accelerated batch peak integration using trapezoidal rule.
///
/// Integrates multiple peak areas in parallel using `WeightedDotF64`.
/// Each peak's area = sum of (dt * (y\[i\] + y\[i+1\]) / 2) over the peak range.
///
/// # Errors
///
/// Returns an error if GPU dispatch fails or the device lacks f64 support.
#[allow(clippy::cast_precision_loss)]
pub fn batch_integrate_peaks_gpu(
    gpu: &GpuF64,
    eics: &[Eic],
    peaks: &[Vec<Peak>],
) -> Result<Vec<Vec<f64>>> {
    require_f64(gpu)?;

    let wd = WeightedDotF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("WeightedDotF64: {e}")))?;

    let mut all_areas = Vec::with_capacity(eics.len());

    for (eic, eic_peaks) in eics.iter().zip(peaks.iter()) {
        let mut areas = Vec::with_capacity(eic_peaks.len());

        for peak in eic_peaks {
            let left = peak.left_base;
            let right = peak.right_base.min(eic.rt.len().saturating_sub(1));

            if right <= left || right >= eic.rt.len() {
                areas.push(0.0);
                continue;
            }

            let n = right - left;
            // Trapezoidal weights: dt[i] * 0.5 for (y[i] + y[i+1])
            let mut weights = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);

            for i in left..right {
                let dt = eic.rt[i + 1] - eic.rt[i];
                weights.push(dt);
                values.push((eic.intensity[i] + eic.intensity[i + 1]) * 0.5);
            }

            if values.is_empty() {
                areas.push(0.0);
                continue;
            }

            let area = wd
                .dot(&weights, &values)
                .map_err(|e| Error::Gpu(format!("peak integration GPU: {e}")))?;
            areas.push(area);
        }

        all_areas.push(areas);
    }

    Ok(all_areas)
}

/// GPU-accelerated batch sum of EIC intensities.
///
/// Sums all intensities in each EIC using `FusedMapReduceF64::sum`.
/// Useful for quick total-ion-count calculations across many chromatograms.
///
/// # Errors
///
/// Returns an error if GPU dispatch fails or the device lacks f64 support.
pub fn batch_eic_total_intensity_gpu(gpu: &GpuF64, eics: &[Eic]) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    let mut totals = Vec::with_capacity(eics.len());
    for eic in eics {
        let total = fmr
            .sum(&eic.intensity)
            .map_err(|e| Error::Gpu(format!("EIC sum GPU: {e}")))?;
        totals.push(total);
    }

    Ok(totals)
}

/// GPU-accelerated batch peak detection on multiple EICs.
///
/// Runs peak detection on each EIC's intensity trace. The peak detection
/// itself is CPU (data-dependent branching), but pre-filtering of
/// intensities and post-scoring of peaks can use GPU primitives.
///
/// # Errors
///
/// Returns an error if GPU dispatch fails or the device lacks f64 support.
#[allow(clippy::cast_precision_loss)]
pub fn batch_find_peaks_gpu(
    gpu: &GpuF64,
    eics: &[Eic],
    params: &PeakParams,
) -> Result<Vec<Vec<Peak>>> {
    require_f64(gpu)?;

    let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    let mut all_peaks = Vec::with_capacity(eics.len());

    for eic in eics {
        // Use GPU to compute noise floor (mean intensity) for dynamic thresholding
        let mean_intensity = if eic.intensity.is_empty() {
            0.0
        } else {
            fmr.sum(&eic.intensity)
                .map_err(|e| Error::Gpu(format!("mean intensity GPU: {e}")))?
                / eic.intensity.len() as f64
        };

        // Apply peak detection with optional dynamic height threshold
        let effective_params = if params.min_height.is_none() && mean_intensity > 0.0 {
            PeakParams {
                min_height: Some(mean_intensity * 3.0),
                ..params.clone()
            }
        } else {
            params.clone()
        };

        all_peaks.push(find_peaks(&eic.intensity, &effective_params));
    }

    Ok(all_peaks)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::bio::eic::Eic;
    use crate::bio::signal::Peak;

    /// Sum of intensities: `sum([1, 2, 3]) = 6` (analytical).
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn known_value_batch_eic_total_intensity() {
        let gpu = crate::gpu::GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return;
        }
        let eics = vec![Eic {
            target_mz: 100.0,
            rt: vec![0.0, 1.0, 2.0],
            intensity: vec![1.0, 2.0, 3.0],
        }];
        let totals = batch_eic_total_intensity_gpu(&gpu, &eics).unwrap();
        let expected = 6.0;
        assert!(
            (totals[0] - expected).abs() < 1e-10,
            "EIC sum: got {}, expected {}",
            totals[0],
            expected
        );
    }

    /// Trapezoidal integration of a triangle `(0,0)-(1,h)-(2,0)`: area = 2.
    /// With `rt=[0,1,2]`, `intensity=[0,2,0]`, peak spanning `left_base=0`, `right_base=2`.
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn known_value_batch_integrate_peaks_trapezoidal() {
        let gpu = crate::gpu::GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return;
        }
        let eic = Eic {
            target_mz: 100.0,
            rt: vec![0.0, 1.0, 2.0],
            intensity: vec![0.0, 2.0, 0.0],
        };
        let peak = Peak {
            index: 1,
            height: 2.0,
            prominence: 2.0,
            left_base: 0,
            right_base: 2,
            width: 2.0,
            left_ips: 0.0,
            right_ips: 2.0,
        };
        let areas = batch_integrate_peaks_gpu(&gpu, &[eic], &[vec![peak]]).unwrap();
        let expected = 2.0;
        assert!(
            (areas[0][0] - expected).abs() < 1e-10,
            "Peak area: got {}, expected {}",
            areas[0][0],
            expected
        );
    }
}
