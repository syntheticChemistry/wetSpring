// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-generation frequency thresholds for LTEE variant calling (WS-11).
//!
//! In the Long-Term Evolution Experiment, the minimum variant frequency
//! that constitutes a real polymorphism depends on population dynamics:
//!
//! - **Early generations** (< 2000): Low clonal interference, emerging mutations
//!   detectable at low frequency. Polymorphism threshold ≈ 5%.
//! - **Mid generations** (2000–20000): Clonal interference begins; multiple
//!   beneficial lineages coexist. Noise floor rises. Threshold ≈ 10–15%.
//! - **Late generations** (> 20000): Strong clonal interference, hitchhiking,
//!   and sweeps. Fixed mutations dominate; polymorphisms require higher
//!   confidence. Threshold ≈ 15–20%.
//!
//! For **consensus calling** (clonal isolates), the threshold is the
//! complement: a variant is "fixed" when its frequency exceeds
//! `1 - polymorphism_threshold`.
//!
//! # Model
//!
//! The threshold is computed using an exponential saturation curve:
//!
//! ```text
//! f_min(g) = floor + (ceiling - floor) × (1 - e^(-g / τ))
//! ```
//!
//! where `g` is the generation number, `floor` is the early-generation
//! threshold, `ceiling` is the asymptotic late-generation threshold, and
//! `τ` (tau) is the characteristic generation timescale for clonal
//! interference onset.

#[cfg(test)]
mod tests;

/// Population parameters for LTEE generation-aware thresholds.
#[derive(Debug, Clone)]
pub struct LteeThresholds {
    /// Minimum polymorphism frequency at generation 0 (floor).
    /// breseq default for LTEE: 0.05 (5%)
    pub floor: f64,
    /// Asymptotic threshold at late generations (ceiling).
    /// Reflects the noise floor imposed by clonal interference.
    pub ceiling: f64,
    /// Characteristic timescale (τ) in generations for the threshold
    /// to approach the ceiling. Higher τ = slower ramp.
    /// For *E. coli* LTEE: ~10,000 generations.
    pub tau: f64,
    /// Generation number for threshold computation.
    pub generation: u64,
}

impl Default for LteeThresholds {
    fn default() -> Self {
        Self {
            floor: 0.05,
            ceiling: 0.20,
            tau: 10_000.0,
            generation: 0,
        }
    }
}

impl LteeThresholds {
    /// Create thresholds for a specific generation using default LTEE parameters.
    #[must_use]
    pub fn at_generation(generation: u64) -> Self {
        Self {
            generation,
            ..Self::default()
        }
    }

    /// Create with custom population parameters.
    #[must_use]
    pub const fn custom(floor: f64, ceiling: f64, tau: f64, generation: u64) -> Self {
        Self {
            floor,
            ceiling,
            tau,
            generation,
        }
    }

    /// Compute the polymorphism detection threshold at the configured generation.
    ///
    /// Returns the minimum allele frequency to call a polymorphism:
    /// `floor + (ceiling - floor) × (1 - e^(-generation / τ))`
    #[must_use]
    pub fn polymorphism_threshold(&self) -> f64 {
        if self.generation == 0 {
            return self.floor;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "generation numbers are well within f64 precision for LTEE"
        )]
        let g = self.generation as f64;
        let saturation = 1.0 - (-g / self.tau).exp();
        (self.ceiling - self.floor).mul_add(saturation, self.floor)
    }

    /// Compute the consensus (fixation) threshold at the configured generation.
    ///
    /// A variant is considered fixed when its frequency exceeds this value.
    /// This is `1 - polymorphism_threshold`, ensuring symmetry.
    #[must_use]
    pub fn consensus_threshold(&self) -> f64 {
        1.0 - self.polymorphism_threshold()
    }

    /// Compute the minimum quality score for the configured generation.
    ///
    /// Early generations use lower quality thresholds (more permissive)
    /// because true polymorphisms may be at very low frequency with
    /// fewer supporting reads. Late generations require higher confidence.
    ///
    /// Model: `Q_min(g) = Q_floor + (Q_ceiling - Q_floor) × saturation`
    #[must_use]
    pub fn quality_threshold(&self) -> f64 {
        let q_floor = 6.0;
        let q_ceiling = 20.0;
        if self.generation == 0 {
            return q_floor;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "generation numbers are well within f64 precision for LTEE"
        )]
        let g = self.generation as f64;
        let saturation = 1.0 - (-g / self.tau).exp();
        (q_ceiling - q_floor).mul_add(saturation, q_floor)
    }

    /// Apply these generation-aware thresholds to a [`CallerConfig`].
    ///
    /// Overwrites `min_alt_frequency` and `min_quality` with the
    /// generation-appropriate values. Other config fields are preserved.
    #[must_use]
    pub fn apply_to(&self, config: &super::CallerConfig) -> super::CallerConfig {
        let mut cfg = config.clone();
        cfg.min_alt_frequency = self.polymorphism_threshold();
        cfg.min_quality = self.quality_threshold();
        cfg
    }
}
