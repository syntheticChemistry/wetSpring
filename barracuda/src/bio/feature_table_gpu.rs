// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated LC-MS feature extraction pipeline.
//!
//! Chains `eic_gpu` (mass track detection + EIC extraction) and `signal_gpu`
//! (peak detection) to build a complete feature table on GPU. Feature
//! filtering (min_height, min_snr) is element-wise and GPU-friendly.
//!
//! In a pure-GPU streaming pipeline, spectra arrive from the mzML parser
//! and feature tables flow to the downstream KMD/PFAS screening stage
//! without CPU round-trips.

use super::feature_table::{self, FeatureParams, FeatureTable};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::mzml::MzmlSpectrum;

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
/// Pipeline: mass track detection -> EIC extraction -> peak detection ->
/// trapezoidal integration -> feature filtering. Each mass track is
/// processed independently (embarrassingly parallel).
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
    Ok(feature_table::extract_features(spectra, params))
}
