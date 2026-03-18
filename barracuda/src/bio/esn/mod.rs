// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::needless_range_loop)]
//! Minimal Echo State Network (ESN) for NPU deployment validation.
//!
//! Implements reservoir computing: a fixed random recurrent network (reservoir)
//! projects input into a high-dimensional state space. Only the readout layer
//! (`W_out`) is trained via ridge regression. The reservoir weights (`W_in`, `W_res`)
//! are generated once and never updated.
//!
//! # Implementations
//!
//! - **Legacy** (`LegacyEsn`): CPU-only, minimal LCG PRNG, custom reservoir.
//!   Always available. Use for non-GPU builds and backward compatibility.
//! - **barraCuda bridge** (`BioEsn`): Wraps `barracuda::esn_v2::ESN` for bio use cases.
//!   Hardware-agnostic (Tensor), WGSL fused reservoir update. Requires `gpu` feature.
//! - **Multi-head** (`MultiHeadBioEsn`): Wraps barraCuda `MultiHeadEsn` (S79).
//!   Shared reservoir, per-head bio readout, head disagreement uncertainty.
//!   Cross-spring provenance: hotSpring (36-head) + wetSpring (bio heads).
//!
//! # NPU Deployment Path
//!
//! ```text
//! GPU/CPU training data (Exp108-113)
//!   → ESN train (ridge regression readout)
//!     → export W_out as f64
//!       → quantize to int8 (NpuReadoutWeights)
//!         → Akida AKD1000 FC layer inference
//! ```

mod config;
pub mod heads;
mod npu;
mod reservoir;
mod training;

#[cfg(feature = "gpu")]
mod toadstool_bridge;

pub use config::EsnConfig;
pub use heads::{AttentionState, BioHeadGroupDisagreement};
pub use npu::NpuReadoutWeights;
pub use reservoir::Lcg;

#[cfg(feature = "gpu")]
pub use toadstool_bridge::{BioEsn, BioEsnConfig, BioHeadKind, MultiHeadBioEsn};

use reservoir::{build_w_in, build_w_res, update_state};
use training::solve_ridge;

/// Legacy CPU-only Echo State Network.
///
/// Uses minimal LCG PRNG, custom reservoir building, ridge regression training.
/// Always available. For GPU/NPU-capable ESN when `gpu` feature is enabled,
/// use `BioEsn` (requires `gpu` feature) which wraps barraCuda's `barracuda::esn_v2::ESN`.
pub struct LegacyEsn {
    w_in: Vec<f64>,
    w_res: Vec<f64>,
    w_out: Vec<f64>,
    state: Vec<f64>,
    config: EsnConfig,
}

impl LegacyEsn {
    /// Create a new ESN with random reservoir.
    #[must_use]
    pub fn new(config: EsnConfig) -> Self {
        let mut rng = reservoir::Lcg::new(config.seed);
        let w_in = build_w_in(&config, &mut rng);
        let w_res = build_w_res(&config, &mut rng);
        let n_res = config.reservoir_size;
        let n_out = config.output_size;

        Self {
            w_in,
            w_res,
            w_out: vec![0.0; n_out * n_res],
            state: vec![0.0; n_res],
            config,
        }
    }

    /// Reset reservoir state to zero.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Run one reservoir update step.
    pub fn update(&mut self, input: &[f64]) {
        update_state(
            &mut self.state,
            &self.w_in,
            &self.w_res,
            input,
            &self.config,
        );
    }

    /// Get a reference to the current reservoir state (for NPU inference).
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Input weight matrix `W_in` (flat, `input_size × reservoir_size`).
    #[must_use]
    pub fn w_in(&self) -> &[f64] {
        &self.w_in
    }

    /// Reservoir weight matrix `W_res` (flat, `reservoir_size × reservoir_size`).
    #[must_use]
    pub fn w_res(&self) -> &[f64] {
        &self.w_res
    }

    /// Readout weight matrix `W_out` (flat, `output_size × reservoir_size`).
    #[must_use]
    pub fn w_out(&self) -> &[f64] {
        &self.w_out
    }

    /// Mutable access to readout weights for online learning / evolution.
    pub fn w_out_mut(&mut self) -> &mut [f64] {
        &mut self.w_out
    }

    /// Configuration used to build this ESN.
    #[must_use]
    pub const fn config(&self) -> &EsnConfig {
        &self.config
    }

    /// Compute readout from current state.
    #[must_use]
    pub fn readout(&self) -> Vec<f64> {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let mut output = vec![0.0_f64; n_out];
        for o in 0..n_out {
            for r in 0..n_res {
                output[o] = self.w_out[o * n_res + r].mul_add(self.state[r], output[o]);
            }
        }
        output
    }

    /// Train readout weights via ridge regression on collected states.
    pub fn train(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let n_samples = inputs.len().min(targets.len());

        self.reset_state();

        let mut flat_states = Vec::with_capacity(n_samples * n_res);
        for input in inputs.iter().take(n_samples) {
            self.update(input);
            flat_states.extend_from_slice(&self.state);
        }

        let mut flat_targets = vec![0.0_f64; n_samples * n_out];
        for (t, target) in targets.iter().take(n_samples).enumerate() {
            for (o, &val) in target.iter().take(n_out).enumerate() {
                flat_targets[t * n_out + o] = val;
            }
        }

        solve_ridge(
            &mut self.w_out,
            &flat_states,
            &flat_targets,
            n_samples,
            n_res,
            n_out,
            self.config.regularization,
        );

        self.reset_state();
    }

    /// Train from sequences with reset between each trajectory.
    pub fn train_stateful(&mut self, trajectories: &[Vec<(Vec<f64>, Vec<f64>)>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let mut flat_states = Vec::new();
        let mut flat_targets = Vec::new();
        let mut n_samples = 0;

        for traj in trajectories {
            self.reset_state();
            for (input, target) in traj {
                self.update(input);
                flat_states.extend_from_slice(&self.state);
                for o in 0..n_out {
                    flat_targets.push(if o < target.len() { target[o] } else { 0.0 });
                }
                n_samples += 1;
            }
        }

        if n_samples == 0 {
            return;
        }

        solve_ridge(
            &mut self.w_out,
            &flat_states,
            &flat_targets,
            n_samples,
            n_res,
            n_out,
            self.config.regularization,
        );

        self.reset_state();
    }

    /// Train with reset before each sample (stateless: each window independent).
    pub fn train_stateless(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) {
        let n_res = self.config.reservoir_size;
        let n_out = self.config.output_size;
        let n_samples = inputs.len().min(targets.len());

        let mut flat_states = Vec::with_capacity(n_samples * n_res);
        for input in inputs.iter().take(n_samples) {
            self.reset_state();
            self.update(input);
            flat_states.extend_from_slice(&self.state);
        }

        let mut flat_targets = vec![0.0_f64; n_samples * n_out];
        for (t, target) in targets.iter().take(n_samples).enumerate() {
            for (o, &val) in target.iter().take(n_out).enumerate() {
                flat_targets[t * n_out + o] = val;
            }
        }

        solve_ridge(
            &mut self.w_out,
            &flat_states,
            &flat_targets,
            n_samples,
            n_res,
            n_out,
            self.config.regularization,
        );

        self.reset_state();
    }

    /// Predict on a sequence of inputs (full drive + readout).
    #[must_use]
    pub fn predict(&mut self, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.reset_state();
        inputs
            .iter()
            .map(|input| {
                self.update(input);
                self.readout()
            })
            .collect()
    }

    /// Export readout weights as int8 for NPU deployment.
    #[must_use]
    pub fn to_npu_weights(&self) -> NpuReadoutWeights {
        NpuReadoutWeights::from_readout_weights(
            &self.w_out,
            self.config.output_size,
            self.config.reservoir_size,
        )
    }
}

/// Backward-compatible alias for [`LegacyEsn`].
pub type Esn = LegacyEsn;

#[cfg(test)]
mod tests;
