//! mzML parser — vendor-neutral mass spectrometry data I/O.
//!
//! Parses mzML XML files using `quick-xml`, decoding base64-encoded
//! and zlib-compressed m/z + intensity arrays (64-bit float).

use base64::Engine;
use flate2::read::ZlibDecoder;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::io::Read;
use std::path::Path;

/// A single mass spectrum parsed from mzML.
#[derive(Debug, Clone)]
pub struct MzmlSpectrum {
    /// Spectrum index (0-based)
    pub index: usize,
    /// MS level (1 = MS1, 2 = MS2, etc.)
    pub ms_level: u32,
    /// Retention time in minutes
    pub rt_minutes: f64,
    /// Total ion current
    pub tic: f64,
    /// Base peak m/z
    pub base_peak_mz: f64,
    /// Base peak intensity
    pub base_peak_intensity: f64,
    /// Lowest observed m/z
    pub lowest_mz: f64,
    /// Highest observed m/z
    pub highest_mz: f64,
    /// m/z array
    pub mz_array: Vec<f64>,
    /// Intensity array
    pub intensity_array: Vec<f64>,
}

/// Summary statistics from an mzML file.
#[derive(Debug, Clone)]
pub struct MzmlStats {
    pub num_spectra: usize,
    pub num_ms1: usize,
    pub num_ms2: usize,
    pub min_rt: f64,
    pub max_rt: f64,
    pub min_mz: f64,
    pub max_mz: f64,
    pub total_peaks: usize,
}

/// Decode a base64 + zlib-compressed array of f64 values.
fn decode_binary_array(encoded: &str, is_zlib: bool, is_64bit: bool) -> Result<Vec<f64>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|e| format!("Base64 decode error: {}", e))?;

    let decompressed = if is_zlib {
        let mut decoder = ZlibDecoder::new(&bytes[..]);
        let mut buf = Vec::new();
        decoder
            .read_to_end(&mut buf)
            .map_err(|e| format!("Zlib decompress error: {}", e))?;
        buf
    } else {
        bytes
    };

    if is_64bit {
        if decompressed.len() % 8 != 0 {
            return Err(format!(
                "Binary array length {} not divisible by 8",
                decompressed.len()
            ));
        }
        Ok(decompressed
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    } else {
        if decompressed.len() % 4 != 0 {
            return Err(format!(
                "Binary array length {} not divisible by 4",
                decompressed.len()
            ));
        }
        Ok(decompressed
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()) as f64)
            .collect())
    }
}

/// Parse an mzML file and return all spectra.
pub fn parse_mzml(path: &Path) -> Result<Vec<MzmlSpectrum>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Read error {}: {}", path.display(), e))?;

    let mut reader = Reader::from_str(&content);
    reader.config_mut().trim_text(true);

    let mut spectra = Vec::new();
    let mut in_spectrum = false;
    let mut in_binary_data_array = false;
    let mut in_binary = false;

    // Current spectrum being built
    let mut current_index: usize = 0;
    let mut current_ms_level: u32 = 1;
    let mut current_rt: f64 = 0.0;
    let mut current_tic: f64 = 0.0;
    let mut current_base_peak_mz: f64 = 0.0;
    let mut current_base_peak_intensity: f64 = 0.0;
    let mut current_lowest_mz: f64 = 0.0;
    let mut current_highest_mz: f64 = 0.0;

    // Binary data array state
    let mut is_mz_array = false;
    let mut is_intensity_array = false;
    let mut is_zlib = false;
    let mut is_64bit = true;
    let mut binary_text = String::new();
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f64> = Vec::new();

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local_name = e.local_name();
                let name = std::str::from_utf8(local_name.as_ref()).unwrap_or("");

                match name {
                    "spectrum" => {
                        in_spectrum = true;
                        current_ms_level = 1;
                        current_rt = 0.0;
                        current_tic = 0.0;
                        current_base_peak_mz = 0.0;
                        current_base_peak_intensity = 0.0;
                        current_lowest_mz = 0.0;
                        current_highest_mz = 0.0;
                        mz_array.clear();
                        intensity_array.clear();

                        for attr in e.attributes().flatten() {
                            let key = std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
                            let val = std::str::from_utf8(&attr.value).unwrap_or("");
                            if key == "index" {
                                current_index = val.parse().unwrap_or(0);
                            }
                        }
                    }
                    "binaryDataArray" => {
                        in_binary_data_array = true;
                        is_mz_array = false;
                        is_intensity_array = false;
                        is_zlib = false;
                        is_64bit = true;
                        binary_text.clear();
                    }
                    "binary" => {
                        in_binary = true;
                        binary_text.clear();
                    }
                    "cvParam" if in_spectrum => {
                        let mut accession = String::new();
                        let mut value = String::new();
                        for attr in e.attributes().flatten() {
                            let key = std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
                            let val = std::str::from_utf8(&attr.value).unwrap_or("");
                            match key {
                                "accession" => accession = val.to_string(),
                                "value" => value = val.to_string(),
                                _ => {}
                            }
                        }

                        if in_binary_data_array {
                            match accession.as_str() {
                                "MS:1000514" => is_mz_array = true,      // m/z array
                                "MS:1000515" => is_intensity_array = true, // intensity array
                                "MS:1000574" => is_zlib = true,           // zlib compression
                                "MS:1000576" => is_zlib = false,          // no compression
                                "MS:1000523" => is_64bit = true,          // 64-bit float
                                "MS:1000521" => is_64bit = false,         // 32-bit float
                                _ => {}
                            }
                        } else {
                            match accession.as_str() {
                                "MS:1000511" => {
                                    current_ms_level = value.parse().unwrap_or(1);
                                }
                                "MS:1000016" => {
                                    current_rt = value.parse().unwrap_or(0.0);
                                }
                                "MS:1000285" => {
                                    current_tic = value.parse().unwrap_or(0.0);
                                }
                                "MS:1000504" => {
                                    current_base_peak_mz = value.parse().unwrap_or(0.0);
                                }
                                "MS:1000505" => {
                                    current_base_peak_intensity = value.parse().unwrap_or(0.0);
                                }
                                "MS:1000528" => {
                                    current_lowest_mz = value.parse().unwrap_or(0.0);
                                }
                                "MS:1000527" => {
                                    current_highest_mz = value.parse().unwrap_or(0.0);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_binary {
                    binary_text.push_str(&e.unescape().unwrap_or_default());
                }
            }
            Ok(Event::End(ref e)) => {
                let local_name = e.local_name();
                let name = std::str::from_utf8(local_name.as_ref()).unwrap_or("");

                match name {
                    "spectrum" => {
                        spectra.push(MzmlSpectrum {
                            index: current_index,
                            ms_level: current_ms_level,
                            rt_minutes: current_rt,
                            tic: current_tic,
                            base_peak_mz: current_base_peak_mz,
                            base_peak_intensity: current_base_peak_intensity,
                            lowest_mz: current_lowest_mz,
                            highest_mz: current_highest_mz,
                            mz_array: std::mem::take(&mut mz_array),
                            intensity_array: std::mem::take(&mut intensity_array),
                        });
                        in_spectrum = false;
                    }
                    "binary" => {
                        in_binary = false;
                    }
                    "binaryDataArray" => {
                        if !binary_text.is_empty() {
                            match decode_binary_array(&binary_text, is_zlib, is_64bit) {
                                Ok(arr) => {
                                    if is_mz_array {
                                        mz_array = arr;
                                    } else if is_intensity_array {
                                        intensity_array = arr;
                                    }
                                }
                                Err(e) => {
                                    eprintln!(
                                        "Warning: binary decode failed at spectrum {}: {}",
                                        current_index, e
                                    );
                                }
                            }
                        }
                        in_binary_data_array = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {}", e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(spectra)
}

/// Compute summary statistics from parsed spectra.
pub fn compute_stats(spectra: &[MzmlSpectrum]) -> MzmlStats {
    let mut ms1 = 0;
    let mut ms2 = 0;
    let mut min_rt = f64::MAX;
    let mut max_rt = f64::MIN;
    let mut min_mz = f64::MAX;
    let mut max_mz = f64::MIN;
    let mut total_peaks = 0usize;

    for s in spectra {
        match s.ms_level {
            1 => ms1 += 1,
            2 => ms2 += 1,
            _ => {}
        }
        if s.rt_minutes < min_rt {
            min_rt = s.rt_minutes;
        }
        if s.rt_minutes > max_rt {
            max_rt = s.rt_minutes;
        }
        if s.lowest_mz > 0.0 && s.lowest_mz < min_mz {
            min_mz = s.lowest_mz;
        }
        if s.highest_mz > max_mz {
            max_mz = s.highest_mz;
        }
        total_peaks += s.mz_array.len();
    }

    if min_rt == f64::MAX {
        min_rt = 0.0;
    }
    if max_rt == f64::MIN {
        max_rt = 0.0;
    }
    if min_mz == f64::MAX {
        min_mz = 0.0;
    }
    if max_mz == f64::MIN {
        max_mz = 0.0;
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
