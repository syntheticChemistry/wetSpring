// SPDX-License-Identifier: AGPL-3.0-or-later
//! mzXML type definitions: spectrum structs and internal builder/enums.
//!
//! Field layout of [`MzxmlSpectrum`] matches [`crate::io::mzml::MzmlSpectrum`]
//! for downstream interoperability.

/// A single mass spectrum parsed from mzXML.
///
/// Field layout matches [`crate::io::mzml::MzmlSpectrum`] for downstream
/// interoperability — callers can use either parser and feed results
/// into the same EIC / feature-detection pipeline.
#[derive(Debug, Clone)]
pub struct MzxmlSpectrum {
    /// Scan number (1-based in file, stored 0-based here).
    pub index: usize,
    /// MS level (1 = MS1, 2 = MS2, etc.).
    pub ms_level: u32,
    /// Retention time in minutes.
    pub rt_minutes: f64,
    /// Total ion current.
    pub tic: f64,
    /// Base peak m/z.
    pub base_peak_mz: f64,
    /// Base peak intensity.
    pub base_peak_intensity: f64,
    /// Lowest observed m/z.
    pub lowest_mz: f64,
    /// Highest observed m/z.
    pub highest_mz: f64,
    /// m/z array (deinterleaved from `<peaks>`).
    pub mz_array: Vec<f64>,
    /// Intensity array (deinterleaved from `<peaks>`).
    pub intensity_array: Vec<f64>,
}

/// Reusable zlib decompression buffer for `<peaks>` decoding.
///
/// Amortizes allocation across scans — same pattern as
/// [`crate::io::mzml::decode::DecodeBuffer`].
#[derive(Default)]
pub(super) struct ZlibBuffer {
    pub(super) buf: Vec<u8>,
}

/// Internal builder for assembling a spectrum from scan/peaks events.
#[derive(Default)]
pub(super) struct ScanBuilder {
    pub(super) index: usize,
    pub(super) ms_level: u32,
    pub(super) rt: f64,
    pub(super) tic: f64,
    pub(super) base_peak_mz: f64,
    pub(super) base_peak_intensity: f64,
    pub(super) lowest_mz: f64,
    pub(super) highest_mz: f64,
    pub(super) mz_array: Vec<f64>,
    pub(super) intensity_array: Vec<f64>,
}

impl ScanBuilder {
    pub(super) fn build(self) -> MzxmlSpectrum {
        MzxmlSpectrum {
            index: self.index.saturating_sub(1), // 1-based → 0-based
            ms_level: self.ms_level,
            rt_minutes: self.rt,
            tic: self.tic,
            base_peak_mz: self.base_peak_mz,
            base_peak_intensity: self.base_peak_intensity,
            lowest_mz: self.lowest_mz,
            highest_mz: self.highest_mz,
            mz_array: self.mz_array,
            intensity_array: self.intensity_array,
        }
    }
}

/// Byte order for decoded peak arrays.
#[derive(Clone, Copy)]
pub(super) enum ByteOrder {
    /// Big-endian (mzXML default).
    Network,
    Little,
}

/// Compression type for `<peaks>` content.
#[derive(Clone, Copy)]
pub(super) enum Compression {
    None,
    Zlib,
}
