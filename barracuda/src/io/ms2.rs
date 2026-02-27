// SPDX-License-Identifier: AGPL-3.0-or-later
//! MS2 text format parser for tandem mass spectrometry data.
//!
//! Streams from disk via `BufReader` — the file is never loaded into memory.
//!
//! Format (`ProteoWizard` / `MassHunter`):
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

use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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

/// Collect all spectra from an MS2 file into memory via [`Ms2Iter`].
///
/// Convenience wrapper — streams from disk, then collects.
/// For large files, prefer iterating with [`Ms2Iter`] directly.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened, or
/// [`Error::Ms2`] if a record is malformed.
#[must_use = "parsed spectra are discarded if not used"]
pub fn parse_ms2(path: &Path) -> Result<Vec<Ms2Spectrum>> {
    Ms2Iter::open(path)?.collect()
}

/// Streaming iterator that yields one [`Ms2Spectrum`] at a time without
/// buffering the entire file.
///
/// Uses `read_line` into a reusable buffer to avoid per-line allocation
/// (unlike `Lines` which allocates a fresh `String` per line).
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use wetspring_barracuda::io::ms2;
///
/// let iter = ms2::Ms2Iter::open(Path::new("spectra.ms2")).unwrap();
/// for result in iter {
///     let spectrum = result.unwrap();
///     println!("scan {} — {} peaks", spectrum.scan, spectrum.mz_array.len());
/// }
/// ```
pub struct Ms2Iter {
    reader: Box<dyn BufRead>,
    line_buf: String,
    path: std::path::PathBuf,
    pending: Option<Ms2Spectrum>,
    done: bool,
}

impl Ms2Iter {
    /// Open an MS2 file for streaming iteration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let reader: Box<dyn BufRead> = Box::new(BufReader::new(file));
        Ok(Self {
            reader,
            line_buf: String::new(),
            path: path.to_path_buf(),
            pending: None,
            done: false,
        })
    }

    fn parse_i_line(spec: &mut Ms2Spectrum, line: &str) -> std::result::Result<(), Error> {
        let mut parts = line.split_whitespace();
        if let (Some(_), Some(key), Some(val)) = (parts.next(), parts.next(), parts.next()) {
            match key {
                "RTime" => {
                    spec.rt_minutes = val
                        .parse()
                        .map_err(|_| Error::Ms2(format!("I-line invalid RTime: {val}")))?;
                }
                "TIC" => {
                    spec.tic = val
                        .parse()
                        .map_err(|_| Error::Ms2(format!("I-line invalid TIC: {val}")))?;
                }
                "BPI" => {
                    spec.bpi = val
                        .parse()
                        .map_err(|_| Error::Ms2(format!("I-line invalid BPI: {val}")))?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl Iterator for Ms2Iter {
    type Item = Result<Ms2Spectrum>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            self.line_buf.clear();
            match self.reader.read_line(&mut self.line_buf) {
                Ok(0) => {
                    self.done = true;
                    return self.pending.take().map(Ok);
                }
                Ok(_) => {
                    let line = self.line_buf.trim_end_matches(['\n', '\r']);
                    if line.is_empty() {
                        continue;
                    }
                    let first_byte = line.as_bytes()[0];
                    match first_byte {
                        b'H' => {}
                        b'S' => {
                            let emit = self.pending.take();
                            let mut parts = line.split_whitespace();
                            let scan: u32 = match parts.nth(1).and_then(|s| s.parse().ok()) {
                                Some(v) => v,
                                None => {
                                    return Some(Err(Error::Ms2(format!(
                                        "S-line missing scan number: {line}"
                                    ))));
                                }
                            };
                            let pmz: f64 = match parts.nth(1).and_then(|s| s.parse().ok()) {
                                Some(v) => v,
                                None => {
                                    return Some(Err(Error::Ms2(format!(
                                        "S-line missing precursor m/z: {line}"
                                    ))));
                                }
                            };
                            self.pending = Some(Ms2Spectrum {
                                scan,
                                precursor_mz: pmz,
                                rt_minutes: 0.0,
                                tic: 0.0,
                                bpi: 0.0,
                                charge: 0,
                                mz_array: Vec::new(),
                                intensity_array: Vec::new(),
                            });
                            if emit.is_some() {
                                return emit.map(Ok);
                            }
                        }
                        b'I' => {
                            if let Some(ref mut spec) = self.pending {
                                if let Err(e) = Self::parse_i_line(spec, line) {
                                    return Some(Err(e));
                                }
                            }
                        }
                        b'Z' => {
                            if let Some(ref mut spec) = self.pending {
                                if let Some(cs) = line.split_whitespace().nth(1) {
                                    match cs.parse() {
                                        Ok(v) => spec.charge = v,
                                        Err(_) => {
                                            return Some(Err(Error::Ms2(format!(
                                                "Z-line invalid charge: {cs}"
                                            ))));
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            if let Some(ref mut spec) = self.pending {
                                let mut parts = line.split_whitespace();
                                if let (Some(mz_str), Some(int_str)) = (parts.next(), parts.next())
                                {
                                    if let (Ok(mz), Ok(intensity)) =
                                        (mz_str.parse::<f64>(), int_str.parse::<f64>())
                                    {
                                        spec.mz_array.push(mz);
                                        spec.intensity_array.push(intensity);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(Error::Io {
                        path: self.path.clone(),
                        source: e,
                    }));
                }
            }
        }
    }
}

/// Process each spectrum without collecting to a `Vec`.
///
/// The callback receives each [`Ms2Spectrum`] as it is parsed. For
/// single-pass aggregation (statistics, filtering), this avoids
/// buffering all spectra in memory.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be opened,
/// [`Error::Ms2`] if a record is malformed, or propagates the
/// callback's [`Result`].
pub fn for_each_spectrum<F>(path: &Path, mut f: F) -> Result<()>
where
    F: FnMut(Ms2Spectrum) -> Result<()>,
{
    for result in Ms2Iter::open(path)? {
        f(result?)?;
    }
    Ok(())
}

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

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, deprecated)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_ms2(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("test.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "H\tCreatedBy\ttest").unwrap();
        writeln!(f, "S\t1\t1\t500.25").unwrap();
        writeln!(f, "I\tRTime\t5.50").unwrap();
        writeln!(f, "I\tTIC\t10000.0").unwrap();
        writeln!(f, "I\tBPI\t8000.0").unwrap();
        writeln!(f, "Z\t2\t999.49").unwrap();
        writeln!(f, "100.0\t500.0").unwrap();
        writeln!(f, "200.0\t300.0").unwrap();
        writeln!(f, "300.0\t100.0").unwrap();
        writeln!(f, "S\t2\t2\t600.50").unwrap();
        writeln!(f, "I\tRTime\t6.00").unwrap();
        writeln!(f, "Z\t1\t599.49").unwrap();
        writeln!(f, "150.0\t400.0").unwrap();
        path
    }

    #[test]
    fn parse_synthetic_ms2() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_ms2(dir.path());
        let spectra = parse_ms2(&path).unwrap();

        assert_eq!(spectra.len(), 2);

        assert_eq!(spectra[0].scan, 1);
        assert!((spectra[0].precursor_mz - 500.25).abs() < f64::EPSILON);
        assert!((spectra[0].rt_minutes - 5.5).abs() < f64::EPSILON);
        assert_eq!(spectra[0].charge, 2);
        assert_eq!(spectra[0].mz_array.len(), 3);

        assert_eq!(spectra[1].scan, 2);
        assert_eq!(spectra[1].mz_array.len(), 1);
    }

    #[test]
    fn compute_stats_two_spectra() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_ms2(dir.path());
        let spectra = parse_ms2(&path).unwrap();
        let stats = compute_stats(&spectra);

        assert_eq!(stats.num_spectra, 2);
        assert_eq!(stats.total_peaks, 4);
        assert!((stats.min_precursor_mz.unwrap() - 500.25).abs() < f64::EPSILON);
        assert!((stats.max_precursor_mz.unwrap() - 600.50).abs() < f64::EPSILON);
        assert!((stats.mean_peaks_per_spectrum - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.num_spectra, 0);
        assert!(stats.min_precursor_mz.is_none());
    }

    #[test]
    fn parse_nonexistent_file() {
        let path = std::env::temp_dir().join("nonexistent_wetspring_9f8a2.ms2");
        let result = parse_ms2(&path);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ms2_with_empty_lines_and_unknown_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edges.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "H\tCreatedBy\ttest").unwrap();
        writeln!(f).unwrap(); // empty line
        writeln!(f, "S\t1\t1\t400.00").unwrap();
        writeln!(f, "I\tRTime\t3.00").unwrap();
        writeln!(f, "I\tUnknownKey\tignored").unwrap(); // unknown I-line key
        writeln!(f, "I\tShort").unwrap(); // I-line with < 3 parts
        writeln!(f, "Z").unwrap(); // Z-line with < 2 parts
        writeln!(f).unwrap(); // another empty line
        writeln!(f, "100.0\t200.0").unwrap();
        writeln!(f, "bad_peak_line").unwrap(); // peak line with < 2 fields

        let spectra = parse_ms2(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert!((spectra[0].precursor_mz - 400.0).abs() < f64::EPSILON);
        assert!((spectra[0].rt_minutes - 3.0).abs() < f64::EPSILON);
        assert_eq!(spectra[0].charge, 0); // Z had no value
        assert_eq!(spectra[0].mz_array.len(), 1); // only 1 valid peak
    }

    #[test]
    fn parse_ms2_peaks_before_spectrum() {
        // Peak lines before any S-line should be ignored
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orphan.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "100.0\t200.0").unwrap(); // orphan peak
        writeln!(f, "S\t1\t1\t300.00").unwrap();
        writeln!(f, "50.0\t100.0").unwrap();

        let spectra = parse_ms2(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].mz_array.len(), 1);
        assert!((spectra[0].mz_array[0] - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_ms2_orphan_i_and_z_lines() {
        // I and Z lines before any S-line should be silently ignored
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orphan_iz.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "I\tRTime\t5.0").unwrap(); // orphan I
        writeln!(f, "Z\t2\t400.0").unwrap(); // orphan Z
        writeln!(f, "S\t1\t1\t300.00").unwrap();
        writeln!(f, "I\tRTime\t10.0").unwrap();
        writeln!(f, "Z\t3\t600.0").unwrap();
        writeln!(f, "50.0\t100.0").unwrap();

        let spectra = parse_ms2(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert!((spectra[0].rt_minutes - 10.0).abs() < f64::EPSILON);
        assert_eq!(spectra[0].charge, 3);
    }

    #[test]
    fn ms2_iter_matches_parse() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_ms2(dir.path());

        let buffered = parse_ms2(&path).unwrap();
        let streamed: Vec<Ms2Spectrum> = Ms2Iter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(buffered.len(), streamed.len());
        for (b, s) in buffered.iter().zip(streamed.iter()) {
            assert_eq!(b.scan, s.scan);
            assert!((b.precursor_mz - s.precursor_mz).abs() < f64::EPSILON);
            assert!((b.rt_minutes - s.rt_minutes).abs() < f64::EPSILON);
            assert_eq!(b.charge, s.charge);
            assert_eq!(b.mz_array.len(), s.mz_array.len());
        }
    }

    #[test]
    fn ms2_iter_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.ms2");
        File::create(&path).unwrap();

        let spectra: Vec<Ms2Spectrum> = Ms2Iter::open(&path)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(spectra.is_empty());
    }

    #[test]
    fn stats_from_file_multiple_spectra() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_ms2(dir.path());
        let stats = stats_from_file(&path).unwrap();

        assert_eq!(stats.num_spectra, 2);
        assert_eq!(stats.total_peaks, 4);
        assert!((stats.min_precursor_mz.unwrap() - 500.25).abs() < f64::EPSILON);
        assert!((stats.max_precursor_mz.unwrap() - 600.50).abs() < f64::EPSILON);
        assert!((stats.min_rt.unwrap() - 5.5).abs() < f64::EPSILON);
        assert!((stats.max_rt.unwrap() - 6.0).abs() < f64::EPSILON);
        assert!((stats.mean_peaks_per_spectrum - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_from_file_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.ms2");
        File::create(&path).unwrap();

        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_spectra, 0);
        assert_eq!(stats.total_peaks, 0);
        assert!(stats.min_precursor_mz.is_none());
        assert!(stats.min_rt.is_none());
    }

    #[test]
    fn stats_from_file_single_spectrum() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "S\t1\t1\t400.00").unwrap();
        writeln!(f, "I\tRTime\t2.5").unwrap();
        writeln!(f, "100.0\t500.0").unwrap();
        writeln!(f, "200.0\t300.0").unwrap();

        let stats = stats_from_file(&path).unwrap();
        assert_eq!(stats.num_spectra, 1);
        assert_eq!(stats.total_peaks, 2);
        assert!((stats.min_precursor_mz.unwrap() - 400.0).abs() < f64::EPSILON);
        assert!((stats.min_rt.unwrap() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_from_file_nonexistent() {
        let path = std::env::temp_dir().join("nonexistent_wetspring_stats_9f8a2.ms2");
        let result = stats_from_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn compute_stats_single_spectrum() {
        let s = Ms2Spectrum {
            scan: 1,
            precursor_mz: 450.0,
            rt_minutes: 3.0,
            tic: 5000.0,
            bpi: 2000.0,
            charge: 2,
            mz_array: vec![100.0, 200.0],
            intensity_array: vec![500.0, 300.0],
        };
        let stats = compute_stats(&[s]);
        assert_eq!(stats.num_spectra, 1);
        assert_eq!(stats.total_peaks, 2);
        assert!((stats.min_precursor_mz.unwrap() - 450.0).abs() < f64::EPSILON);
        assert!((stats.min_rt.unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_ms2_s_line_missing_scan_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("malformed_s.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "S").unwrap();
        writeln!(f, "50.0\t100.0").unwrap();

        let result = parse_ms2(&path);
        assert!(result.is_err(), "S-line without scan should error");
    }

    #[test]
    fn parse_ms2_s_line_missing_pmz_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("malformed_pmz.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "S\t1").unwrap();
        writeln!(f, "100.0\t200.0").unwrap();

        let result = parse_ms2(&path);
        assert!(result.is_err(), "S-line without precursor m/z should error");
    }

    #[test]
    fn ms2_iter_io_error_invalid_utf8() {
        // Invalid UTF-8 in file causes Lines to return Err
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_utf8.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "S\t1\t1\t500.0").unwrap();
        writeln!(f, "100.0\t200.0").unwrap();
        f.write_all(&[0xFF, 0xFE, 0xFD]).unwrap(); // invalid UTF-8
        f.write_all(b"\n").unwrap();

        let results: Vec<Result<Ms2Spectrum>> = Ms2Iter::open(&path).unwrap().collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }
}
