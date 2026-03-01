// SPDX-License-Identifier: AGPL-3.0-or-later
//! MS2 spectrum and statistics types.

/// A single MS2 spectrum.
#[derive(Debug, Clone)]
pub struct Ms2Spectrum {
    /// Scan number.
    pub scan: u32,
    /// Precursor m/z.
    pub precursor_mz: f64,
    /// Retention time in minutes.
    pub rt_minutes: f64,
    /// Total ion current.
    pub tic: f64,
    /// Base peak intensity.
    pub bpi: f64,
    /// Charge state.
    pub charge: u32,
    /// Fragment m/z values.
    pub mz_array: Vec<f64>,
    /// Fragment intensities.
    pub intensity_array: Vec<f64>,
}

/// Summary statistics from parsed MS2 spectra.
#[derive(Debug, Clone)]
pub struct Ms2Stats {
    /// Number of spectra.
    pub num_spectra: usize,
    /// Total number of fragment peaks.
    pub total_peaks: usize,
    /// Minimum precursor m/z, or `None` if empty.
    pub min_precursor_mz: Option<f64>,
    /// Maximum precursor m/z, or `None` if empty.
    pub max_precursor_mz: Option<f64>,
    /// Minimum retention time, or `None` if empty.
    pub min_rt: Option<f64>,
    /// Maximum retention time, or `None` if empty.
    pub max_rt: Option<f64>,
    /// Mean number of peaks per spectrum.
    pub mean_peaks_per_spectrum: f64,
}
