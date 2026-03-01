// SPDX-License-Identifier: AGPL-3.0-or-later
//! MS2 summary statistics.

use super::types::{Ms2Spectrum, Ms2Stats};
use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Compute summary statistics in a single streaming pass.
///
/// Allocation-free per spectrum — lines are streamed via `BufReader` without
/// buffering full spectra. Use this for large files where only aggregate
/// statistics are needed.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened or read.
#[must_use = "computed stats are discarded if not used"]
pub fn stats_from_file(path: &Path) -> Result<Ms2Stats> {
    let file = File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);
    let mut line_buf = String::new();

    let mut num_spectra = 0_usize;
    let mut total_peaks = 0_usize;
    let mut min_pmz: Option<f64> = None;
    let mut max_pmz: Option<f64> = None;
    let mut min_rt: Option<f64> = None;
    let mut max_rt: Option<f64> = None;
    let mut in_spectrum = false;

    loop {
        line_buf.clear();
        let bytes_read = reader.read_line(&mut line_buf).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if bytes_read == 0 {
            break;
        }
        let line = line_buf.trim_end_matches(['\n', '\r']);

        if line.is_empty() {
            continue;
        }

        let first_byte = line.as_bytes()[0];
        match first_byte {
            b'H' | b'Z' => {}
            b'S' => {
                if in_spectrum {
                    num_spectra += 1;
                }
                in_spectrum = true;
                if let Some(pmz) = line
                    .split_whitespace()
                    .nth(3)
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    min_pmz = Some(min_pmz.map_or(pmz, |v: f64| v.min(pmz)));
                    max_pmz = Some(max_pmz.map_or(pmz, |v: f64| v.max(pmz)));
                }
            }
            b'I' => {
                let mut parts = line.split_whitespace();
                if let (Some(_), Some("RTime"), Some(val)) =
                    (parts.next(), parts.next(), parts.next())
                {
                    if let Ok(rt) = val.parse::<f64>() {
                        min_rt = Some(min_rt.map_or(rt, |v: f64| v.min(rt)));
                        max_rt = Some(max_rt.map_or(rt, |v: f64| v.max(rt)));
                    }
                }
            }
            _ => {
                if in_spectrum {
                    let mut parts = line.split_whitespace();
                    if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                        if a.parse::<f64>().is_ok() && b.parse::<f64>().is_ok() {
                            total_peaks += 1;
                        }
                    }
                }
            }
        }
    }

    if in_spectrum {
        num_spectra += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    Ok(Ms2Stats {
        num_spectra,
        total_peaks,
        min_precursor_mz: min_pmz,
        max_precursor_mz: max_pmz,
        min_rt,
        max_rt,
        mean_peaks_per_spectrum: if num_spectra > 0 {
            total_peaks as f64 / num_spectra as f64
        } else {
            0.0
        },
    })
}

/// Compute summary statistics.
#[must_use]
pub fn compute_stats(spectra: &[Ms2Spectrum]) -> Ms2Stats {
    if spectra.is_empty() {
        return Ms2Stats {
            num_spectra: 0,
            total_peaks: 0,
            min_precursor_mz: None,
            max_precursor_mz: None,
            min_rt: None,
            max_rt: None,
            mean_peaks_per_spectrum: 0.0,
        };
    }

    let mut total_peaks = 0_usize;
    let mut min_pmz: Option<f64> = None;
    let mut max_pmz: Option<f64> = None;
    let mut min_rt: Option<f64> = None;
    let mut max_rt: Option<f64> = None;

    for s in spectra {
        total_peaks += s.mz_array.len();
        min_pmz = Some(min_pmz.map_or(s.precursor_mz, |v: f64| v.min(s.precursor_mz)));
        max_pmz = Some(max_pmz.map_or(s.precursor_mz, |v: f64| v.max(s.precursor_mz)));
        min_rt = Some(min_rt.map_or(s.rt_minutes, |v: f64| v.min(s.rt_minutes)));
        max_rt = Some(max_rt.map_or(s.rt_minutes, |v: f64| v.max(s.rt_minutes)));
    }

    #[allow(clippy::cast_precision_loss)] // counts are small
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
