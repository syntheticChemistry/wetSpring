// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spatial interpolation of diversity metrics via `ToadStool`'s `KrigingF64`.
//!
//! Wraps ordinary and simple kriging for ecological use: interpolating
//! diversity indices (Shannon, Simpson, evenness, etc.) across geographic
//! sampling sites.
//!
//! # Variogram models
//!
//! - **Spherical**: Most common for ecological data, bounded correlation.
//! - **Exponential**: Reaches sill asymptotically, for unbounded fields.
//! - **Gaussian**: Very smooth surfaces, continuous phenomena.
//! - **Linear**: Simple, for monotonic spatial trends.
//!
//! # Example
//!
//! ```ignore
//! use wetspring_barracuda::bio::kriging::{interpolate_diversity, SpatialSample, VariogramConfig};
//! use wetspring_barracuda::gpu::GpuF64;
//!
//! let sites = vec![
//!     SpatialSample { x: 0.0, y: 0.0, value: 2.3 },   // Shannon at site A
//!     SpatialSample { x: 100.0, y: 0.0, value: 1.8 },  // Shannon at site B
//!     SpatialSample { x: 50.0, y: 80.0, value: 2.5 },   // Shannon at site C
//! ];
//! let targets = vec![(25.0, 40.0), (75.0, 40.0)];
//! let config = VariogramConfig::spherical(0.0, 0.3, 120.0);
//!
//! let result = interpolate_diversity(&gpu, &sites, &targets, &config)?;
//! // result.values[0] = interpolated Shannon at (25, 40)
//! // result.variances[0] = uncertainty estimate
//! ```

use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::kriging_f64::{KrigingF64, KrigingResult, VariogramModel};

/// A spatially-referenced sample measurement.
#[derive(Debug, Clone, Copy)]
pub struct SpatialSample {
    /// X coordinate (e.g. easting in meters, or longitude).
    pub x: f64,
    /// Y coordinate (e.g. northing in meters, or latitude).
    pub y: f64,
    /// Measured value at this location (e.g. Shannon entropy, Simpson index).
    pub value: f64,
}

/// Variogram configuration for spatial correlation modelling.
#[derive(Debug, Clone, Copy)]
pub struct VariogramConfig {
    /// Variogram model to use.
    pub model: VariogramModel,
}

impl VariogramConfig {
    /// Spherical variogram — bounded correlation, most common for ecology.
    #[must_use]
    pub const fn spherical(nugget: f64, sill: f64, range: f64) -> Self {
        Self {
            model: VariogramModel::Spherical {
                nugget,
                sill,
                range,
            },
        }
    }

    /// Exponential variogram — asymptotic approach to sill.
    #[must_use]
    pub const fn exponential(nugget: f64, sill: f64, range: f64) -> Self {
        Self {
            model: VariogramModel::Exponential {
                nugget,
                sill,
                range,
            },
        }
    }

    /// Gaussian variogram — very smooth surfaces.
    #[must_use]
    pub const fn gaussian(nugget: f64, sill: f64, range: f64) -> Self {
        Self {
            model: VariogramModel::Gaussian {
                nugget,
                sill,
                range,
            },
        }
    }

    /// Linear variogram — simple monotonic.
    #[must_use]
    pub const fn linear(nugget: f64, sill: f64, range: f64) -> Self {
        Self {
            model: VariogramModel::Linear {
                nugget,
                sill,
                range,
            },
        }
    }
}

/// Result of spatial diversity interpolation.
#[derive(Debug, Clone)]
pub struct SpatialResult {
    /// Interpolated values at target locations.
    pub values: Vec<f64>,
    /// Kriging variance (uncertainty) at each target location.
    pub variances: Vec<f64>,
}

/// Interpolate a diversity metric across geographic space.
///
/// Uses ordinary kriging (unknown mean) via `ToadStool`'s `KrigingF64`.
///
/// # Arguments
///
/// * `gpu` — GPU context (kriging uses CPU internally but requires device).
/// * `sites` — Known sampling sites with measured diversity values.
/// * `targets` — (x, y) coordinates at which to interpolate.
/// * `config` — Variogram model and parameters.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if interpolation fails, or [`Error::InvalidInput`]
/// if fewer than 2 known sites are provided.
pub fn interpolate_diversity(
    gpu: &GpuF64,
    sites: &[SpatialSample],
    targets: &[(f64, f64)],
    config: &VariogramConfig,
) -> Result<SpatialResult> {
    if sites.len() < 2 {
        return Err(Error::InvalidInput(
            "kriging requires at least 2 known sites".into(),
        ));
    }
    if targets.is_empty() {
        return Ok(SpatialResult {
            values: vec![],
            variances: vec![],
        });
    }

    let known: Vec<(f64, f64, f64)> = sites.iter().map(|s| (s.x, s.y, s.value)).collect();

    let kriging = KrigingF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("KrigingF64 init: {e}")))?;

    let result = kriging
        .interpolate(&known, targets, config.model)
        .map_err(|e| Error::Gpu(format!("kriging interpolation: {e}")))?;

    Ok(from_kriging_result(&result))
}

/// Interpolate with a known population mean (simple kriging).
///
/// Use when the global mean of the diversity metric is known a priori
/// (e.g. from a large reference dataset). Produces tighter variance
/// estimates than ordinary kriging.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if interpolation fails, or [`Error::InvalidInput`]
/// if fewer than 2 known sites are provided.
pub fn interpolate_diversity_simple(
    gpu: &GpuF64,
    sites: &[SpatialSample],
    targets: &[(f64, f64)],
    config: &VariogramConfig,
    known_mean: f64,
) -> Result<SpatialResult> {
    if sites.len() < 2 {
        return Err(Error::InvalidInput(
            "kriging requires at least 2 known sites".into(),
        ));
    }
    if targets.is_empty() {
        return Ok(SpatialResult {
            values: vec![],
            variances: vec![],
        });
    }

    let known: Vec<(f64, f64, f64)> = sites.iter().map(|s| (s.x, s.y, s.value)).collect();

    let kriging = KrigingF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("KrigingF64 init: {e}")))?;

    let result = kriging
        .interpolate_simple(&known, targets, config.model, known_mean)
        .map_err(|e| Error::Gpu(format!("simple kriging: {e}")))?;

    Ok(from_kriging_result(&result))
}

/// Compute empirical variogram from observed spatial data.
///
/// Returns `(lag_distances, lag_semivariances)` for fitting a variogram
/// model to your data before interpolation.
///
/// # Arguments
///
/// * `sites` — Known sampling sites with measured values.
/// * `n_lags` — Number of distance bins.
/// * `max_distance` — Maximum lag distance to consider.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if fewer than 2 sites are provided.
pub fn empirical_variogram(
    sites: &[SpatialSample],
    n_lags: usize,
    max_distance: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if sites.len() < 2 {
        return Err(Error::InvalidInput(
            "variogram estimation requires at least 2 sites".into(),
        ));
    }

    let known: Vec<(f64, f64, f64)> = sites.iter().map(|s| (s.x, s.y, s.value)).collect();

    KrigingF64::fit_variogram(&known, n_lags, max_distance)
        .map_err(|e| Error::Gpu(format!("variogram fitting: {e}")))
}

/// Convert `ToadStool`'s `KrigingResult` to our domain type.
fn from_kriging_result(result: &KrigingResult) -> SpatialResult {
    SpatialResult {
        values: result.values.clone(),
        variances: result.variances.clone(),
    }
}
