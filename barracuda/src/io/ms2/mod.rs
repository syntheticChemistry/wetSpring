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

mod parser;
mod stats;
mod types;

#[expect(deprecated)]
pub use parser::{Ms2Iter, for_each_spectrum, parse_ms2};
pub use stats::{compute_stats, stats_from_file};
pub use types::{Ms2Spectrum, Ms2Stats};

#[cfg(test)]
#[expect(clippy::unwrap_used, deprecated)]
mod tests {
    use super::*;
    use crate::error::Result;
    use std::fs::File;
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

    #[test]
    fn parse_ms2_valid_single_spectrum_block() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "H\tCreatedBy\ttest").unwrap();
        writeln!(f, "S\t1\t1\t500.25").unwrap();
        writeln!(f, "I\tRTime\t5.50").unwrap();
        writeln!(f, "I\tTIC\t10000.0").unwrap();
        writeln!(f, "Z\t2\t999.49").unwrap();
        writeln!(f, "100.0\t500.0").unwrap();
        writeln!(f, "200.0\t300.0").unwrap();

        let spectra = parse_ms2(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].scan, 1);
        assert!((spectra[0].precursor_mz - 500.25).abs() < f64::EPSILON);
        assert_eq!(spectra[0].mz_array.len(), 2);
    }

    #[test]
    fn parse_ms2_i_line_invalid_rtime_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_rtime.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "S\t1\t1\t500.0").unwrap();
        writeln!(f, "I\tRTime\tnot_a_number").unwrap();
        writeln!(f, "100.0\t200.0").unwrap();

        let result = parse_ms2(&path);
        assert!(result.is_err(), "I-line with invalid RTime should error");
    }

    #[test]
    fn parse_ms2_z_line_invalid_charge_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_charge.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "S\t1\t1\t500.0").unwrap();
        writeln!(f, "I\tRTime\t5.0").unwrap();
        writeln!(f, "Z\t2\t999.49").unwrap();
        writeln!(f, "Z\tbad\t999.49").unwrap();
        writeln!(f, "100.0\t200.0").unwrap();

        let result = parse_ms2(&path);
        assert!(result.is_err(), "Z-line with invalid charge should error");
    }

    #[test]
    fn parse_ms2_peak_line_bad_numbers_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_peaks.ms2");
        let mut f = File::create(&path).unwrap();
        writeln!(f, "S\t1\t1\t400.0").unwrap();
        writeln!(f, "I\tRTime\t3.0").unwrap();
        writeln!(f, "100.0\t200.0").unwrap();
        writeln!(f, "abc\tdef").unwrap();
        writeln!(f, "300.0\t400.0").unwrap();

        let spectra = parse_ms2(&path).unwrap();
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].mz_array.len(), 2);
    }
}
