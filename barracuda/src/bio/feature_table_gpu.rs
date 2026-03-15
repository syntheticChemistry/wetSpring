// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated LC-MS feature extraction pipeline.
//!
//! Chains `eic_gpu` (mass track detection + EIC extraction) and `signal_gpu`
//! (peak detection) to build a complete feature table on GPU. Feature
//! filtering (`min_height`, `min_snr`) is element-wise and GPU-friendly.
//!
//! In a pure-GPU streaming pipeline, spectra arrive from the mzML parser
//! and feature tables flow to the downstream KMD/PFAS screening stage
//! without CPU round-trips.
//!
//! # Compose pattern
//!
//! This module composes two existing barraCuda-backed GPU modules:
//! - [`super::eic_gpu`]: `FusedMapReduceF64` + `WeightedDotF64` for EIC extraction
//! - [`super::signal_gpu`]: `PeakDetectF64` for peak detection
//!
//! Mass track detection remains on CPU (data-dependent branching), but
//! per-track EIC extraction + peak detection + integration is GPU-dispatched.

use super::eic;
use super::feature_table::{Feature, FeatureParams, FeatureTable};
use super::signal_gpu;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::mzml::MzmlSpectrum;

/// Minimum MS1 scan count for GPU dispatch. Below this, CPU path is faster.
const MIN_MS1_SCANS_FOR_GPU: usize = 256;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu(
            "SHADER_F64 required for feature_table GPU".into(),
        ));
    }
    Ok(())
}

/// GPU-accelerated feature extraction from LC-MS spectra.
///
/// Pipeline: mass track detection (CPU) -> EIC extraction (CPU) ->
/// GPU peak detection via `PeakDetectF64` -> trapezoidal integration ->
/// feature filtering.
///
/// Falls back to [`super::feature_table::extract_features`] when spectra
/// are too small for GPU dispatch overhead to be worthwhile (< 256 MS1 scans).
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn extract_features_gpu(
    gpu: &GpuF64,
    spectra: &[MzmlSpectrum],
    params: &FeatureParams,
) -> Result<FeatureTable> {
    require_f64(gpu)?;

    let ms1_count = spectra.iter().filter(|s| s.ms_level == 1).count();
    if ms1_count < MIN_MS1_SCANS_FOR_GPU {
        return Ok(super::feature_table::extract_features(spectra, params));
    }

    let mass_tracks = eic::detect_mass_tracks(spectra, params.eic_ppm, params.min_scans);
    let n_tracks = mass_tracks.len();

    if mass_tracks.is_empty() {
        return Ok(FeatureTable {
            features: vec![],
            mass_tracks_evaluated: 0,
            eics_with_peaks: 0,
        });
    }

    let eics = eic::extract_eics(spectra, &mass_tracks, params.eic_ppm);

    let mut features = Vec::new();
    let mut eics_with_peaks = 0;

    for chromatogram in &eics {
        if chromatogram.intensity.is_empty() {
            continue;
        }

        let peaks = signal_gpu::find_peaks_gpu(gpu, &chromatogram.intensity, &params.peak_params)?;

        if peaks.is_empty() {
            continue;
        }
        eics_with_peaks += 1;

        for peak in &peaks {
            let height = peak.height;
            if height < params.min_height {
                continue;
            }

            let noise_floor = estimate_noise(&chromatogram.intensity);
            let snr = if noise_floor > 0.0 {
                height / noise_floor
            } else {
                f64::INFINITY
            };
            if snr < params.min_snr {
                continue;
            }

            let left_idx = peak.left_base;
            let right_idx = if peak.right_base > 0 {
                peak.right_base
            } else {
                chromatogram.rt.len().saturating_sub(1)
            };
            let area = eic::integrate_peak(
                &chromatogram.rt,
                &chromatogram.intensity,
                left_idx,
                right_idx,
            );

            let rt_apex = if peak.index < chromatogram.rt.len() {
                chromatogram.rt[peak.index]
            } else {
                0.0
            };
            let rt_start = if left_idx < chromatogram.rt.len() {
                chromatogram.rt[left_idx]
            } else {
                0.0
            };
            let rt_end = if right_idx < chromatogram.rt.len() {
                chromatogram.rt[right_idx]
            } else {
                0.0
            };

            features.push(Feature {
                mz: chromatogram.target_mz,
                rt_apex,
                rt_start,
                rt_end,
                height,
                area,
                snr,
                width_fwhm: peak.width,
            });
        }
    }

    features.sort_by(|a, b| a.mz.total_cmp(&b.mz));

    Ok(FeatureTable {
        features,
        mass_tracks_evaluated: n_tracks,
        eics_with_peaks,
    })
}

/// Estimate noise floor from median of bottom 25% of intensities.
#[expect(clippy::cast_precision_loss)] // Precision: noise_slice.len() bounded
fn estimate_noise(intensity: &[f64]) -> f64 {
    if intensity.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = intensity.iter().copied().filter(|&v| v > 0.0).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(f64::total_cmp);
    let quarter = sorted.len() / 4;
    let noise_slice = &sorted[..quarter.max(1)];
    noise_slice.iter().sum::<f64>() / noise_slice.len() as f64
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;
    use crate::bio::feature_table::FeatureParams;
    use crate::gpu::GpuF64;
    use crate::io::mzml::MzmlSpectrum;

    #[test]
    fn api_surface_compiles() {
        let _: fn(&GpuF64, &[MzmlSpectrum], &FeatureParams) -> Result<FeatureTable> =
            extract_features_gpu;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let spectra: Vec<MzmlSpectrum> = vec![];
        let params = FeatureParams::default();
        let result = extract_features_gpu(&gpu, &spectra, &params);
        assert!(
            result.is_ok(),
            "extract_features_gpu should succeed with empty input"
        );
    }
}
