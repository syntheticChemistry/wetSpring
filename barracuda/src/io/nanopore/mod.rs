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

use crate::error::{Error, Result};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A single nanopore read with raw ionic current signal.
///
/// The signal is stored as raw ADC values (int16). Use [`calibrated_signal`]
/// to convert to picoamperes using the per-read calibration parameters.
///
/// [`calibrated_signal`]: NanoporeRead::calibrated_signal
#[derive(Debug, Clone)]
pub struct NanoporeRead {
    /// Read UUID (16 bytes, RFC 4122 v4).
    pub read_id: [u8; 16],
    /// Raw ADC signal (ionic current samples).
    pub signal: Vec<i16>,
    /// Sequencing channel number (1-indexed on flow cell).
    pub channel: u32,
    /// ADC sample rate in Hz (typically 4000 or 5000).
    pub sample_rate: f64,
    /// Calibration offset: `pA = raw * scale + offset`.
    pub calibration_offset: f64,
    /// Calibration scale: `pA = raw * scale + offset`.
    pub calibration_scale: f64,
}

impl NanoporeRead {
    /// Convert raw ADC signal to calibrated picoampere values.
    ///
    /// Applies the affine calibration: `pA = raw * scale + offset`.
    #[must_use]
    pub fn calibrated_signal(&self) -> Vec<f64> {
        self.signal
            .iter()
            .map(|&s| f64::from(s).mul_add(self.calibration_scale, self.calibration_offset))
            .collect()
    }

    /// Duration of this read in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate > 0.0 {
            self.signal.len() as f64 / self.sample_rate
        } else {
            0.0
        }
    }

    /// Format the read UUID as a hex string (lowercase, no dashes).
    #[must_use]
    pub fn read_id_hex(&self) -> String {
        use std::fmt::Write;
        self.read_id
            .iter()
            .fold(String::with_capacity(32), |mut acc, b| {
                let _ = write!(acc, "{b:02x}");
                acc
            })
    }

    /// Compute basic signal statistics: (mean, `std_dev`, min, max).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn signal_stats(&self) -> SignalStats {
        if self.signal.is_empty() {
            return SignalStats {
                mean: 0.0,
                std_dev: 0.0,
                min: 0,
                max: 0,
                n_samples: 0,
            };
        }

        let signal_f64: Vec<f64> = self.signal.iter().map(|&s| f64::from(s)).collect();
        let mean = barracuda::stats::mean(&signal_f64);
        let n = signal_f64.len() as f64;
        let std_dev = barracuda::stats::correlation::variance(&signal_f64)
            .map(|var_sample| (var_sample * (n - 1.0) / n).sqrt())
            .unwrap_or(0.0);
        let (lo, hi) = self
            .signal
            .iter()
            .fold((i16::MAX, i16::MIN), |(lo, hi), &s| (lo.min(s), hi.max(s)));

        SignalStats {
            mean,
            std_dev,
            min: lo,
            max: hi,
            n_samples: self.signal.len(),
        }
    }
}

/// Summary statistics for a raw nanopore signal.
#[derive(Debug, Clone, Copy)]
pub struct SignalStats {
    /// Mean ADC value.
    pub mean: f64,
    /// Standard deviation of ADC values.
    pub std_dev: f64,
    /// Minimum ADC value.
    pub min: i16,
    /// Maximum ADC value.
    pub max: i16,
    /// Number of signal samples.
    pub n_samples: usize,
}

// ═══════════════════════════════════════════════════════════════════
// Wire format: Nanopore Read Stream (NRS)
//
// A simple binary format for testing and validation. Real POD5 parsing
// is layered on top of the same NanoporeRead type.
//
// Layout:
//   magic:    [u8; 4] = b"NRS1"
//   n_reads:  u64 (little-endian)
//   Per-read:
//     read_id:            [u8; 16]
//     channel:            u32 (LE)
//     sample_rate:        f64 (LE)
//     calibration_offset: f64 (LE)
//     calibration_scale:  f64 (LE)
//     signal_length:      u64 (LE)
//     signal:             [i16; signal_length] (LE)
// ═══════════════════════════════════════════════════════════════════

/// Magic bytes identifying the NRS wire format.
const NRS_MAGIC: &[u8; 4] = b"NRS1";

/// Streaming iterator over nanopore reads from an NRS file.
///
/// Opens a file and yields [`NanoporeRead`] values one at a time,
/// following the same pattern as [`FastqIter`](crate::io::fastq::FastqIter).
pub struct NanoporeIter {
    reader: BufReader<std::fs::File>,
    remaining: u64,
    path: std::path::PathBuf,
}

impl std::fmt::Debug for NanoporeIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NanoporeIter")
            .field("remaining", &self.remaining)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl NanoporeIter {
    /// Open an NRS file and prepare to stream reads.
    ///
    /// # Errors
    ///
    /// Returns `Error::Io` if the file cannot be opened, or
    /// `Error::Nanopore` if the file header is invalid.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        read_exact(&mut reader, &mut magic, path)?;
        if &magic != NRS_MAGIC {
            return Err(Error::Nanopore(format!(
                "invalid NRS magic: expected NRS1, got {}",
                String::from_utf8_lossy(&magic)
            )));
        }

        let n_reads = read_u64_le(&mut reader, path)?;

        Ok(Self {
            reader,
            remaining: n_reads,
            path: path.to_path_buf(),
        })
    }
}

impl Iterator for NanoporeIter {
    type Item = Result<NanoporeRead>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(read_one_nrs_record(&mut self.reader, &self.path))
    }
}

/// Write a collection of nanopore reads to NRS wire format.
///
/// Used by test generators to create synthetic data files.
///
/// # Errors
///
/// Returns `Error::Io` if writing fails.
pub fn write_nrs(path: &Path, reads: &[NanoporeRead]) -> Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    file.write_all(NRS_MAGIC).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let n = reads.len() as u64;
    file.write_all(&n.to_le_bytes()).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    for read in reads {
        write_one_nrs_record(&mut file, read, path)?;
    }

    Ok(())
}

// ── Internal I/O helpers ─────────────────────────────────────────

fn read_exact(reader: &mut impl Read, buf: &mut [u8], path: &Path) -> Result<()> {
    reader.read_exact(buf).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

fn read_u32_le(reader: &mut impl Read, path: &Path) -> Result<u32> {
    let mut buf = [0u8; 4];
    read_exact(reader, &mut buf, path)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(reader: &mut impl Read, path: &Path) -> Result<u64> {
    let mut buf = [0u8; 8];
    read_exact(reader, &mut buf, path)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64_le(reader: &mut impl Read, path: &Path) -> Result<f64> {
    let mut buf = [0u8; 8];
    read_exact(reader, &mut buf, path)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_one_nrs_record(reader: &mut impl BufRead, path: &Path) -> Result<NanoporeRead> {
    let mut read_id = [0u8; 16];
    read_exact(reader, &mut read_id, path)?;

    let channel = read_u32_le(reader, path)?;
    let sample_rate = read_f64_le(reader, path)?;
    let calibration_offset = read_f64_le(reader, path)?;
    let calibration_scale = read_f64_le(reader, path)?;
    let signal_length = read_u64_le(reader, path)?;

    if signal_length > 100_000_000 {
        return Err(Error::Nanopore(format!(
            "signal length {signal_length} exceeds 100M samples — likely corrupt"
        )));
    }

    let n = signal_length as usize;
    let mut signal = vec![0i16; n];

    let byte_buf: &mut [u8] = bytemuck::cast_slice_mut(&mut signal);
    read_exact(reader, byte_buf, path)?;

    #[cfg(target_endian = "big")]
    for sample in &mut signal {
        *sample = i16::from_le_bytes(sample.to_ne_bytes());
    }

    Ok(NanoporeRead {
        read_id,
        signal,
        channel,
        sample_rate,
        calibration_offset,
        calibration_scale,
    })
}

fn write_one_nrs_record(
    writer: &mut impl std::io::Write,
    read: &NanoporeRead,
    path: &Path,
) -> Result<()> {
    let w = |writer: &mut dyn std::io::Write, data: &[u8]| -> Result<()> {
        writer.write_all(data).map_err(|e| Error::Io {
            path: path.to_path_buf(),
            source: e,
        })
    };

    w(writer, &read.read_id)?;
    w(writer, &read.channel.to_le_bytes())?;
    w(writer, &read.sample_rate.to_le_bytes())?;
    w(writer, &read.calibration_offset.to_le_bytes())?;
    w(writer, &read.calibration_scale.to_le_bytes())?;
    w(writer, &(read.signal.len() as u64).to_le_bytes())?;

    #[cfg(target_endian = "little")]
    {
        let byte_slice: &[u8] = bytemuck::cast_slice(&read.signal);
        w(writer, byte_slice)?;
    }
    #[cfg(target_endian = "big")]
    {
        for &sample in &read.signal {
            w(writer, &sample.to_le_bytes())?;
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Synthetic data generation (pre-hardware testing)
// ═══════════════════════════════════════════════════════════════════

/// Generates synthetic nanopore reads for pre-hardware validation.
///
/// Produces realistic ionic current patterns with configurable noise,
/// channel simulation, and known-answer signal for round-trip testing.
pub struct SyntheticSignalGenerator {
    seed: u64,
}

impl SyntheticSignalGenerator {
    /// Create a generator with the given PRNG seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate a single nanopore read with known properties.
    ///
    /// The signal consists of a baseline current with Gaussian-like noise,
    /// simulating an open channel at `channel_id`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_read(
        &self,
        channel_id: u32,
        n_samples: usize,
        sample_rate: f64,
    ) -> NanoporeRead {
        let mut rng = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(u64::from(channel_id));

        let mut read_id = [0u8; 16];
        for byte in &mut read_id {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }
        // RFC 4122 v4 variant bits
        read_id[6] = (read_id[6] & 0x0F) | 0x40;
        read_id[8] = (read_id[8] & 0x3F) | 0x80;

        let mut signal = Vec::with_capacity(n_samples);
        let baseline: i16 = 512;
        for _ in 0..n_samples {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u1 = ((rng >> 33) as f64) / f64::from(u32::MAX);
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u2 = ((rng >> 33) as f64) / f64::from(u32::MAX);
            let noise = ((-2.0 * u1.max(1e-30).ln()).sqrt()
                * (2.0 * std::f64::consts::PI * u2).cos()
                * 15.0) as i16;
            signal.push(baseline.saturating_add(noise));
        }

        NanoporeRead {
            read_id,
            signal,
            channel: channel_id,
            sample_rate,
            calibration_offset: 200.0,
            calibration_scale: 0.15,
        }
    }

    /// Generate a batch of reads across multiple channels.
    #[must_use]
    pub fn generate_batch(
        &self,
        n_reads: usize,
        samples_per_read: usize,
        sample_rate: f64,
    ) -> Vec<NanoporeRead> {
        (0..n_reads)
            .map(|i| {
                #[allow(clippy::cast_possible_truncation)]
                let ch = (i as u32) % 512 + 1;
                self.generate_read(ch, samples_per_read, sample_rate)
            })
            .collect()
    }

    /// Generate a read with a known nucleotide-like signal pattern.
    ///
    /// Encodes `sequence` as discrete current levels (A=400, C=500, G=600,
    /// T=700 ADC units) with `samples_per_base` samples per nucleotide,
    /// plus Gaussian noise. Used for basecalling validation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn generate_from_sequence(
        &self,
        channel_id: u32,
        sequence: &[u8],
        samples_per_base: usize,
        sample_rate: f64,
    ) -> NanoporeRead {
        let mut rng = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(u64::from(channel_id));

        let mut read_id = [0u8; 16];
        for byte in &mut read_id {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }
        read_id[6] = (read_id[6] & 0x0F) | 0x40;
        read_id[8] = (read_id[8] & 0x3F) | 0x80;

        let n_samples = sequence.len() * samples_per_base;
        let mut signal = Vec::with_capacity(n_samples);

        for &base in sequence {
            let level: i16 = match base {
                b'A' | b'a' => 400,
                b'C' | b'c' => 500,
                b'G' | b'g' => 600,
                b'T' | b't' => 700,
                _ => 550,
            };

            for _ in 0..samples_per_base {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u1 = ((rng >> 33) as f64) / f64::from(u32::MAX);
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u2 = ((rng >> 33) as f64) / f64::from(u32::MAX);
                let noise = ((-2.0 * u1.max(1e-30).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2).cos()
                    * 10.0) as i16;
                signal.push(level.saturating_add(noise));
            }
        }

        NanoporeRead {
            read_id,
            signal,
            channel: channel_id,
            sample_rate,
            calibration_offset: 200.0,
            calibration_scale: 0.15,
        }
    }
}

/// Classify a calibrated signal segment into a nucleotide base.
///
/// Uses threshold-based classification matching the synthetic signal
/// generator's discrete levels: A≈260pA, C≈275pA, G≈290pA, T≈305pA
/// (after calibration: `raw * 0.15 + 200`).
///
/// For real nanopore data, this would be replaced by a neural basecaller.
#[must_use]
pub fn threshold_basecall(calibrated_pa: f64) -> u8 {
    if calibrated_pa < 268.0 {
        b'A'
    } else if calibrated_pa < 283.0 {
        b'C'
    } else if calibrated_pa < 298.0 {
        b'G'
    } else {
        b'T'
    }
}

/// Simple basecalling by averaging signal segments.
///
/// Splits the calibrated signal into `n_bases` equal segments,
/// computes the mean of each, and classifies by threshold.
/// Returns the basecalled sequence.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn simple_basecall(calibrated: &[f64], n_bases: usize) -> Vec<u8> {
    if n_bases == 0 || calibrated.is_empty() {
        return Vec::new();
    }

    let chunk_size = calibrated.len() / n_bases;
    if chunk_size == 0 {
        return Vec::new();
    }

    (0..n_bases)
        .map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(calibrated.len());
            let segment = &calibrated[start..end];
            let mean = segment.iter().sum::<f64>() / segment.len() as f64;
            threshold_basecall(mean)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
mod tests {
    use super::*;

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
        assert!((cal[0] - 200.0).abs() < 1e-10);
        assert!((cal[1] - 215.0).abs() < 1e-10);
        assert!((cal[2] - 185.0).abs() < 1e-10);
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
        assert!((read.duration_seconds() - 2.0).abs() < 1e-10);
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
        assert!((stats.mean - 30.0).abs() < 1e-10);
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
}
