// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzML statistics computation — from spectra or streaming.

use super::{MzmlSpectrum, MzmlStats};
use crate::error::{Error, Result};
use crate::io::xml::{XmlEvent, XmlReader};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Compute summary statistics in a single streaming pass (zero-copy path).
///
/// Skips binary array decoding entirely — only spectrum-level `cvParam`
/// metadata is inspected.  Use this for large files where only aggregate
/// statistics are needed.
///
/// # Errors
///
/// Returns [`crate::error::Error::Io`] if the file cannot be opened or [`crate::error::Error::Xml`]
/// for XML structure errors.
pub fn stats_from_file(path: &Path) -> Result<MzmlStats> {
    let file = File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let buf_reader = BufReader::new(file);
    let mut reader = XmlReader::new(buf_reader);
    reader.set_trim_text(true);

    let mut ms1 = 0_usize;
    let mut ms2 = 0_usize;
    let mut min_rt: Option<f64> = None;
    let mut max_rt: Option<f64> = None;
    let mut min_mz: Option<f64> = None;
    let mut max_mz: Option<f64> = None;
    let mut total_peaks = 0_usize;
    let mut num_spectra = 0_usize;

    let mut in_spectrum = false;
    let mut current_ms_level: u32 = 1;
    let mut current_array_len: usize = 0;

    loop {
        match reader.next_event()? {
            XmlEvent::StartElement {
                ref name,
                ref attrs,
            } => match name.as_str() {
                "spectrum" => {
                    in_spectrum = true;
                    current_ms_level = 1;
                    current_array_len = attrs
                        .iter()
                        .find(|(k, _)| k == "defaultArrayLength")
                        .and_then(|(_, v)| v.parse().ok())
                        .unwrap_or(0);
                }
                "cvParam" if in_spectrum => {
                    let accession = attrs
                        .iter()
                        .find(|(k, _)| k == "accession")
                        .map_or("", |(_, v)| v.as_str());
                    let cv_value = attrs
                        .iter()
                        .find(|(k, _)| k == "value")
                        .map_or("", |(_, v)| v.as_str());
                    match accession {
                        "MS:1000511" => {
                            // MS level defaults to 1 (MS1) when the cvParam
                            // value is absent or non-numeric — matches
                            // ProteoWizard's default behavior.
                            current_ms_level = cv_value.parse().unwrap_or(1);
                        }
                        "MS:1000016" => {
                            if let Ok(rt) = cv_value.parse::<f64>() {
                                min_rt = Some(min_rt.map_or(rt, |v: f64| v.min(rt)));
                                max_rt = Some(max_rt.map_or(rt, |v: f64| v.max(rt)));
                            }
                        }
                        "MS:1000528" => {
                            if let Ok(lo) = cv_value.parse::<f64>() {
                                if lo > 0.0 {
                                    min_mz = Some(min_mz.map_or(lo, |v: f64| v.min(lo)));
                                }
                            }
                        }
                        "MS:1000527" => {
                            if let Ok(hi) = cv_value.parse::<f64>() {
                                if hi > 0.0 {
                                    max_mz = Some(max_mz.map_or(hi, |v: f64| v.max(hi)));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            XmlEvent::EndElement { ref name } if name == "spectrum" => {
                num_spectra += 1;
                match current_ms_level {
                    1 => ms1 += 1,
                    2 => ms2 += 1,
                    _ => {}
                }
                total_peaks += current_array_len;
                in_spectrum = false;
            }
            XmlEvent::Eof => break,
            _ => {}
        }
    }

    Ok(MzmlStats {
        num_spectra,
        num_ms1: ms1,
        num_ms2: ms2,
        min_rt,
        max_rt,
        min_mz,
        max_mz,
        total_peaks,
    })
}

/// Compute summary statistics from parsed spectra.
#[must_use]
pub fn compute_stats(spectra: &[MzmlSpectrum]) -> MzmlStats {
    let mut ms1 = 0_usize;
    let mut ms2 = 0_usize;
    let mut min_rt: Option<f64> = None;
    let mut max_rt: Option<f64> = None;
    let mut min_mz: Option<f64> = None;
    let mut max_mz: Option<f64> = None;
    let mut total_peaks = 0_usize;

    for s in spectra {
        match s.ms_level {
            1 => ms1 += 1,
            2 => ms2 += 1,
            _ => {}
        }

        min_rt = Some(min_rt.map_or(s.rt_minutes, |v: f64| v.min(s.rt_minutes)));
        max_rt = Some(max_rt.map_or(s.rt_minutes, |v: f64| v.max(s.rt_minutes)));

        if s.lowest_mz > 0.0 {
            min_mz = Some(min_mz.map_or(s.lowest_mz, |v: f64| v.min(s.lowest_mz)));
        }
        if s.highest_mz > 0.0 {
            max_mz = Some(max_mz.map_or(s.highest_mz, |v: f64| v.max(s.highest_mz)));
        }

        total_peaks += s.mz_array.len();
    }

    MzmlStats {
        num_spectra: spectra.len(),
        num_ms1: ms1,
        num_ms2: ms2,
        min_rt,
        max_rt,
        min_mz,
        max_mz,
        total_peaks,
    }
}
