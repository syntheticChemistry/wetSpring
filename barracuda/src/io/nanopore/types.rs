// SPDX-License-Identifier: AGPL-3.0-or-later
//! Core types for nanopore reads and signal statistics.

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
    #[expect(clippy::cast_precision_loss)] // Precision: signal.len() fits f64
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
    ///
    /// Uses Welford's online algorithm for numerically stable single-pass
    /// computation — no intermediate `Vec<f64>` allocation.
    #[must_use]
    #[expect(clippy::cast_precision_loss)] // Precision: signal.len() fits f64
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

        let mut lo = i16::MAX;
        let mut hi = i16::MIN;
        let mut welford_mean = 0.0_f64;
        let mut m2 = 0.0_f64;

        for (i, &s) in self.signal.iter().enumerate() {
            lo = lo.min(s);
            hi = hi.max(s);
            let x = f64::from(s);
            let count = (i + 1) as f64;
            let delta = x - welford_mean;
            welford_mean += delta / count;
            let delta2 = x - welford_mean;
            m2 = delta.mul_add(delta2, m2);
        }

        let n = self.signal.len() as f64;
        let std_dev = if n > 1.0 { (m2 / n).sqrt() } else { 0.0 };

        SignalStats {
            mean: welford_mean,
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
