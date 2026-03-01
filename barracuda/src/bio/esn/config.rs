// SPDX-License-Identifier: AGPL-3.0-or-later
//! Echo State Network configuration.

/// Echo State Network configuration.
#[derive(Debug, Clone)]
pub struct EsnConfig {
    /// Number of input dimensions per time step.
    pub input_size: usize,
    /// Number of reservoir neurons (state dimensionality).
    pub reservoir_size: usize,
    /// Number of output classes or regression targets.
    pub output_size: usize,
    /// Spectral radius of `W_res` (controls echo state property, typically < 1.0).
    pub spectral_radius: f64,
    /// Fraction of non-zero reservoir connections (sparsity = 1 − connectivity).
    pub connectivity: f64,
    /// Leaky integration rate: `state = (1-α)·old + α·tanh(...)`.
    pub leak_rate: f64,
    /// Ridge regression regularization λ (Tikhonov).
    pub regularization: f64,
    /// PRNG seed for deterministic reservoir generation.
    pub seed: u64,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            input_size: 5,
            reservoir_size: 200,
            output_size: 3,
            spectral_radius: 0.9,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: crate::tolerances::ESN_REGULARIZATION,
            seed: 42,
        }
    }
}
