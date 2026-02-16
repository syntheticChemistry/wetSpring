//! MS2 text format parser for tandem mass spectrometry data.
//!
//! Format (ProteoWizard / MassHunter):
//! ```text
//! H  Header lines
//! S  scan  scan  precursor_mz
//! I  NativeID  ...
//! I  RTime  rt_minutes
//! I  BPI  base_peak_intensity
//! I  TIC  total_ion_current
//! Z  charge  mass
//! mz  intensity    (peak list, tab-separated)
//! ```

use std::path::Path;

/// A single MS2 spectrum.
#[derive(Debug, Clone)]
pub struct Ms2Spectrum {
    /// Scan number
    pub scan: u32,
    /// Precursor m/z
    pub precursor_mz: f64,
    /// Retention time in minutes
    pub rt_minutes: f64,
    /// Total ion current
    pub tic: f64,
    /// Base peak intensity
    pub bpi: f64,
    /// Charge state
    pub charge: u32,
    /// Fragment m/z values
    pub mz_array: Vec<f64>,
    /// Fragment intensities
    pub intensity_array: Vec<f64>,
}

/// Parse an MS2 file and return all spectra.
pub fn parse_ms2(path: &Path) -> Result<Vec<Ms2Spectrum>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Read error {}: {}", path.display(), e))?;

    let mut spectra = Vec::new();
    let mut current: Option<Ms2Spectrum> = None;

    for line in content.lines() {
        if line.is_empty() {
            continue;
        }

        let first_char = line.as_bytes()[0];
        match first_char {
            b'H' => {
                // Header line — skip
            }
            b'S' => {
                // Save previous spectrum
                if let Some(spec) = current.take() {
                    spectra.push(spec);
                }
                // Parse: S scan scan precursor_mz
                let parts: Vec<&str> = line.split_whitespace().collect();
                let scan = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                let pmz = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                current = Some(Ms2Spectrum {
                    scan,
                    precursor_mz: pmz,
                    rt_minutes: 0.0,
                    tic: 0.0,
                    bpi: 0.0,
                    charge: 0,
                    mz_array: Vec::new(),
                    intensity_array: Vec::new(),
                });
            }
            b'I' => {
                if let Some(ref mut spec) = current {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        match parts[1] {
                            "RTime" => {
                                spec.rt_minutes = parts[2].parse().unwrap_or(0.0);
                            }
                            "TIC" => {
                                spec.tic = parts[2].parse().unwrap_or(0.0);
                            }
                            "BPI" => {
                                spec.bpi = parts[2].parse().unwrap_or(0.0);
                            }
                            _ => {}
                        }
                    }
                }
            }
            b'Z' => {
                if let Some(ref mut spec) = current {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        spec.charge = parts[1].parse().unwrap_or(0);
                    }
                }
            }
            _ => {
                // Peak line: mz intensity (tab or space separated)
                if let Some(ref mut spec) = current {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let (Ok(mz), Ok(intensity)) =
                            (parts[0].parse::<f64>(), parts[1].parse::<f64>())
                        {
                            spec.mz_array.push(mz);
                            spec.intensity_array.push(intensity);
                        }
                    }
                }
            }
        }
    }

    // Don't forget the last spectrum
    if let Some(spec) = current {
        spectra.push(spec);
    }

    Ok(spectra)
}

/// Summary statistics from parsed MS2 spectra.
#[derive(Debug, Clone)]
pub struct Ms2Stats {
    pub num_spectra: usize,
    pub total_peaks: usize,
    pub min_precursor_mz: f64,
    pub max_precursor_mz: f64,
    pub min_rt: f64,
    pub max_rt: f64,
    pub mean_peaks_per_spectrum: f64,
}

/// Compute summary statistics.
pub fn compute_stats(spectra: &[Ms2Spectrum]) -> Ms2Stats {
    if spectra.is_empty() {
        return Ms2Stats {
            num_spectra: 0,
            total_peaks: 0,
            min_precursor_mz: 0.0,
            max_precursor_mz: 0.0,
            min_rt: 0.0,
            max_rt: 0.0,
            mean_peaks_per_spectrum: 0.0,
        };
    }

    let mut total_peaks = 0usize;
    let mut min_pmz = f64::MAX;
    let mut max_pmz = f64::MIN;
    let mut min_rt = f64::MAX;
    let mut max_rt = f64::MIN;

    for s in spectra {
        total_peaks += s.mz_array.len();
        if s.precursor_mz < min_pmz {
            min_pmz = s.precursor_mz;
        }
        if s.precursor_mz > max_pmz {
            max_pmz = s.precursor_mz;
        }
        if s.rt_minutes < min_rt {
            min_rt = s.rt_minutes;
        }
        if s.rt_minutes > max_rt {
            max_rt = s.rt_minutes;
        }
    }

    Ms2Stats {
        num_spectra: spectra.len(),
        total_peaks,
        min_precursor_mz: min_pmz,
        max_precursor_mz: max_pmz,
        min_rt,
        max_rt,
        mean_peaks_per_spectrum: total_peaks as f64 / spectra.len() as f64,
    }
}
