// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nanopore raw signal I/O — streaming parser for POD5 format.
//!
//! POD5 is Oxford Nanopore's forward-looking file format (Apache Arrow IPC
//! internally). This module provides a sovereign Rust parser that extracts
//! raw ionic current signal, calibration, and read metadata without external
//! C dependencies (no HDF5 lib).
//!
//! # Design
//!
//! - **Streaming**: reads are yielded one at a time via [`NanoporeIter`]
//! - **Zero external deps**: minimal Arrow IPC reader for POD5-specific columns
//! - **Synthetic generation**: [`SyntheticSignalGenerator`] produces test data
//!   for pre-hardware validation (Exp196a–c)
//!
//! # Formats
//!
//! | Format | Status | Notes |
//! |--------|--------|-------|
//! | POD5   | Active | Pure Rust Arrow IPC reader |
//! | FAST5  | Planned (Phase 2) | HDF5 — needs sovereign parser or C binding |
//!
//! # Usage
//!
//! ```
//! use wetspring_barracuda::io::nanopore::{NanoporeRead, SyntheticSignalGenerator};
//!
//! let sig = SyntheticSignalGenerator::new(42);
//! let read = sig.generate_read(1, 4000, 4000.0);
//! assert_eq!(read.signal.len(), 4000);
//! let cal = read.calibrated_signal();
//! assert_eq!(cal.len(), 4000);
//! ```

mod nrs;
mod synthetic;
mod types;

pub use nrs::{NanoporeIter, write_nrs};
pub use synthetic::{SyntheticSignalGenerator, simple_basecall, threshold_basecall};
pub use types::{NanoporeRead, SignalStats};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn synthetic_read_has_correct_length() {
        let sig = SyntheticSignalGenerator::new(42);
        let read = sig.generate_read(1, 4000, 4000.0);
        assert_eq!(read.signal.len(), 4000);
        assert_eq!(read.channel, 1);
        assert!((read.sample_rate - 4000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn calibrated_signal_applies_affine() {
        let read = NanoporeRead {
            read_id: [0; 16],
            signal: vec![0, 100, -100],
            channel: 1,
            sample_rate: 4000.0,
            calibration_offset: 200.0,
            calibration_scale: 0.15,
        };
        let cal = read.calibrated_signal();
        assert!((cal[0] - 200.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((cal[1] - 215.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((cal[2] - 185.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn duration_seconds_correct() {
        let read = NanoporeRead {
            read_id: [0; 16],
            signal: vec![0; 8000],
            channel: 1,
            sample_rate: 4000.0,
            calibration_offset: 0.0,
            calibration_scale: 1.0,
        };
        assert!((read.duration_seconds() - 2.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn read_id_hex_format() {
        let mut id = [0u8; 16];
        id[0] = 0xAB;
        id[15] = 0xCD;
        let read = NanoporeRead {
            read_id: id,
            signal: vec![],
            channel: 1,
            sample_rate: 4000.0,
            calibration_offset: 0.0,
            calibration_scale: 1.0,
        };
        let hex = read.read_id_hex();
        assert_eq!(hex.len(), 32);
        assert!(hex.starts_with("ab"));
        assert!(hex.ends_with("cd"));
    }

    #[test]
    fn signal_stats_basic() {
        let read = NanoporeRead {
            read_id: [0; 16],
            signal: vec![10, 20, 30, 40, 50],
            channel: 1,
            sample_rate: 4000.0,
            calibration_offset: 0.0,
            calibration_scale: 1.0,
        };
        let stats = read.signal_stats();
        assert_eq!(stats.n_samples, 5);
        assert!((stats.mean - 30.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert_eq!(stats.min, 10);
        assert_eq!(stats.max, 50);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn signal_stats_empty() {
        let read = NanoporeRead {
            read_id: [0; 16],
            signal: vec![],
            channel: 1,
            sample_rate: 4000.0,
            calibration_offset: 0.0,
            calibration_scale: 1.0,
        };
        let stats = read.signal_stats();
        assert_eq!(stats.n_samples, 0);
        assert!((stats.mean).abs() < f64::EPSILON);
    }

    #[test]
    fn batch_generation() {
        let sig = SyntheticSignalGenerator::new(99);
        let batch = sig.generate_batch(10, 1000, 4000.0);
        assert_eq!(batch.len(), 10);
        for read in &batch {
            assert_eq!(read.signal.len(), 1000);
        }
    }

    #[test]
    fn sequence_based_signal_has_discrete_levels() {
        let sig = SyntheticSignalGenerator::new(42);
        let seq = b"ACGT";
        let read = sig.generate_from_sequence(1, seq, 100, 4000.0);
        assert_eq!(read.signal.len(), 400);

        let cal = read.calibrated_signal();
        let a_mean: f64 = cal[..100].iter().sum::<f64>() / 100.0;
        let t_mean: f64 = cal[300..400].iter().sum::<f64>() / 100.0;
        assert!(
            a_mean < t_mean,
            "A level ({a_mean:.1}) should be lower than T level ({t_mean:.1})"
        );
    }

    #[test]
    fn simple_basecall_recovers_sequence() {
        let sig = SyntheticSignalGenerator::new(42);
        let seq = b"ACGTACGT";
        let read = sig.generate_from_sequence(1, seq, 200, 4000.0);
        let cal = read.calibrated_signal();
        let called = simple_basecall(&cal, 8);
        assert_eq!(called.len(), 8);
        let accuracy = called
            .iter()
            .zip(seq.iter())
            .filter(|(a, b)| a == b)
            .count() as f64
            / 8.0;
        assert!(
            accuracy >= 0.75,
            "basecall accuracy {accuracy:.2} should be >= 0.75"
        );
    }

    #[test]
    fn nrs_round_trip() {
        let sig = SyntheticSignalGenerator::new(42);
        let reads = sig.generate_batch(5, 500, 4000.0);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        write_nrs(path, &reads).unwrap();

        let iter = NanoporeIter::open(path).unwrap();
        let loaded: Vec<NanoporeRead> = iter.map(|r| r.unwrap()).collect();
        assert_eq!(loaded.len(), 5);

        for (orig, loaded) in reads.iter().zip(loaded.iter()) {
            assert_eq!(orig.read_id, loaded.read_id);
            assert_eq!(orig.channel, loaded.channel);
            assert!((orig.sample_rate - loaded.sample_rate).abs() < f64::EPSILON);
            assert!((orig.calibration_offset - loaded.calibration_offset).abs() < f64::EPSILON);
            assert!((orig.calibration_scale - loaded.calibration_scale).abs() < f64::EPSILON);
            assert_eq!(orig.signal, loaded.signal);
        }
    }

    #[test]
    fn nrs_single_read_round_trip() {
        let sig = SyntheticSignalGenerator::new(123);
        let read = sig.generate_read(7, 100, 5000.0);
        let reads = vec![read];

        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_nrs(tmp.path(), &reads).unwrap();

        let loaded: Vec<NanoporeRead> = NanoporeIter::open(tmp.path())
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(loaded.len(), 1);
        assert_eq!(reads[0].signal, loaded[0].signal);
        assert_eq!(reads[0].channel, loaded[0].channel);
    }

    #[test]
    fn nrs_signal_integrity_after_write_read() {
        let read = NanoporeRead {
            read_id: [1u8; 16],
            signal: vec![100, -50, 200, 0, i16::MAX, i16::MIN],
            channel: 42,
            sample_rate: 4000.0,
            calibration_offset: 200.0,
            calibration_scale: 0.15,
        };
        let reads = vec![read];

        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_nrs(tmp.path(), &reads).unwrap();

        let loaded: NanoporeRead = NanoporeIter::open(tmp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(reads[0].signal, loaded.signal);
    }

    #[test]
    fn nrs_invalid_magic_rejected() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"BAD!").unwrap();
        let result = NanoporeIter::open(tmp.path());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("invalid NRS magic"));
    }

    #[test]
    fn nrs_empty_file_reads_zero() {
        let reads: Vec<NanoporeRead> = Vec::new();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_nrs(tmp.path(), &reads).unwrap();

        let count = NanoporeIter::open(tmp.path()).unwrap().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn nrs_truncated_header_rejected() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"NRS1").unwrap();
        let result = NanoporeIter::open(tmp.path());
        assert!(
            result.is_err(),
            "truncated n_reads (magic only, no u64) should fail"
        );
    }

    #[test]
    fn nrs_signal_length_exceeds_limit_rejected() {
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut f = std::fs::File::create(tmp.path()).unwrap();
        f.write_all(b"NRS1").unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&4000.0f64.to_le_bytes()).unwrap();
        f.write_all(&0.0f64.to_le_bytes()).unwrap();
        f.write_all(&1.0f64.to_le_bytes()).unwrap();
        f.write_all(&(100_000_001u64).to_le_bytes()).unwrap();
        drop(f);

        let mut iter = NanoporeIter::open(tmp.path()).unwrap();
        let result = iter.next().unwrap();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("100M") || msg.contains("corrupt"));
    }

    #[test]
    fn nrs_truncated_read_rejected() {
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut f = std::fs::File::create(tmp.path()).unwrap();
        f.write_all(b"NRS1").unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&4000.0f64.to_le_bytes()).unwrap();
        f.write_all(&0.0f64.to_le_bytes()).unwrap();
        f.write_all(&1.0f64.to_le_bytes()).unwrap();
        f.write_all(&100u64.to_le_bytes()).unwrap();
        f.write_all(&[0u8; 50]).unwrap();
        drop(f);

        let mut iter = NanoporeIter::open(tmp.path()).unwrap();
        let result = iter.next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn nrs_empty_file_zero_bytes() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), []).unwrap();
        let result = NanoporeIter::open(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn nrs_read_extraction_wire_format() {
        let read = NanoporeRead {
            read_id: [0xAB; 16],
            signal: vec![100, -50, 0],
            channel: 99,
            sample_rate: 5000.0,
            calibration_offset: -10.5,
            calibration_scale: 0.02,
        };
        let reads = vec![read.clone()];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_nrs(tmp.path(), &reads).unwrap();

        let loaded: NanoporeRead = NanoporeIter::open(tmp.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(loaded.read_id, read.read_id);
        assert_eq!(loaded.channel, read.channel);
        assert!((loaded.sample_rate - read.sample_rate).abs() < f64::EPSILON);
        assert!((loaded.calibration_offset - read.calibration_offset).abs() < f64::EPSILON);
        assert!((loaded.calibration_scale - read.calibration_scale).abs() < f64::EPSILON);
        assert_eq!(loaded.signal, read.signal);
    }

    #[test]
    fn threshold_basecall_boundaries() {
        assert_eq!(threshold_basecall(260.0), b'A');
        assert_eq!(threshold_basecall(275.0), b'C');
        assert_eq!(threshold_basecall(290.0), b'G');
        assert_eq!(threshold_basecall(305.0), b'T');
    }

    #[test]
    fn uuid_v4_format_bits() {
        let sig = SyntheticSignalGenerator::new(42);
        let read = sig.generate_read(1, 100, 4000.0);
        assert_eq!(read.read_id[6] >> 4, 4, "version nibble should be 4");
        assert_eq!(read.read_id[8] >> 6, 2, "variant bits should be 10xxxxxx");
    }

    #[test]
    fn synthetic_deterministic_same_seed() {
        let sig1 = SyntheticSignalGenerator::new(999);
        let sig2 = SyntheticSignalGenerator::new(999);
        let read1 = sig1.generate_read(1, 200, 4000.0);
        let read2 = sig2.generate_read(1, 200, 4000.0);
        assert_eq!(read1.signal, read2.signal);
        assert_eq!(read1.read_id, read2.read_id);
    }

    #[test]
    fn simple_basecall_empty_returns_empty() {
        let result = simple_basecall(&[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn simple_basecall_n_bases_zero_returns_empty() {
        let cal = vec![260.0, 275.0, 290.0];
        let result = simple_basecall(&cal, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn simple_basecall_chunk_size_zero_returns_empty() {
        let cal = vec![260.0, 275.0];
        let result = simple_basecall(&cal, 100);
        assert!(result.is_empty());
    }

    #[test]
    fn threshold_basecall_edge_values() {
        assert_eq!(threshold_basecall(267.9), b'A');
        assert_eq!(threshold_basecall(268.0), b'C');
        assert_eq!(threshold_basecall(282.9), b'C');
        assert_eq!(threshold_basecall(283.0), b'G');
        assert_eq!(threshold_basecall(297.9), b'G');
        assert_eq!(threshold_basecall(298.0), b'T');
    }
}
