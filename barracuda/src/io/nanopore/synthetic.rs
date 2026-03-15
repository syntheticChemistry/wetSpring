// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic nanopore signal generation for pre-hardware validation.

use crate::io::nanopore::types::NanoporeRead;

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
    #[expect(clippy::cast_precision_loss)] // Precision: n_samples and RNG values fit f64
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
            let noise = ((-2.0
                * u1.max(crate::tolerances::BOX_MULLER_U1_FLOOR_SYNTHETIC)
                    .ln())
            .sqrt()
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
                #[expect(clippy::cast_possible_truncation)] // Truncation: i < n_reads, 512 fits u32
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
    #[expect(clippy::cast_precision_loss)] // Precision: sequence length and RNG fit f64
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
                let noise = ((-2.0
                    * u1.max(crate::tolerances::BOX_MULLER_U1_FLOOR_SYNTHETIC)
                        .ln())
                .sqrt()
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
#[expect(clippy::cast_precision_loss)] // Precision: segment.len() bounded by calibrated len
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
