// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::missing_errors_doc, clippy::missing_const_for_fn)]
//! Bridge to `ToadStool`'s `barracuda::esn_v2` for bio use cases.
//!
//! Wraps the hardware-agnostic ESN (Tensor, WGSL shaders) with bio-specific
//! feature extraction and multi-head classifier support.

use super::{EsnConfig, NpuReadoutWeights};
use barracuda::esn_v2::{ESN, ESNConfig, ExportedWeights};
use barracuda::tensor::Tensor;
use std::sync::OnceLock;

/// Bio classifier head kinds for multi-head ESN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BioHeadKind {
    /// Diversity / community structure.
    Diversity,
    /// Taxonomy classification.
    Taxonomy,
    /// AMR (antimicrobial resistance) profile.
    Amr,
    /// Bloom sentinel (QS, healthy, stressed).
    Bloom,
    /// Disorder / phase classification.
    Disorder,
    /// Custom head index.
    Custom(usize),
}

impl BioHeadKind {
    /// Default head index for single-head or legacy compatibility.
    #[must_use]
    pub const fn default_index() -> usize {
        0
    }

    /// Head index for multi-head ESN (`0..output_size`).
    #[must_use]
    pub fn index(self, output_size: usize) -> usize {
        let i = match self {
            Self::Diversity => 0,
            Self::Taxonomy => 1,
            Self::Amr => 2,
            Self::Bloom => 3,
            Self::Disorder => 4,
            Self::Custom(idx) => idx,
        };
        i.min(output_size.saturating_sub(1))
    }
}

/// Bio-specific ESN configuration mapping to `ToadStool` `ESNConfig`.
#[derive(Debug, Clone)]
pub struct BioEsnConfig {
    /// Input feature size (e.g. diversity metrics, k-mer counts).
    pub input_size: usize,
    /// Reservoir size.
    pub reservoir_size: usize,
    /// Output size (number of heads for multi-head classifiers).
    pub output_size: usize,
    /// Spectral radius (typically 0.9).
    pub spectral_radius: f64,
    /// Reservoir connectivity (fraction of non-zero weights).
    pub connectivity: f64,
    /// Leak rate for temporal integration.
    pub leak_rate: f64,
    /// Ridge regression regularization.
    pub regularization: f64,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for BioEsnConfig {
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

impl From<&EsnConfig> for BioEsnConfig {
    fn from(c: &EsnConfig) -> Self {
        Self {
            input_size: c.input_size,
            reservoir_size: c.reservoir_size,
            output_size: c.output_size,
            spectral_radius: c.spectral_radius,
            connectivity: c.connectivity,
            leak_rate: c.leak_rate,
            regularization: c.regularization,
            seed: c.seed,
        }
    }
}

impl BioEsnConfig {
    /// Convert to `ToadStool` `ESNConfig` (f32).
    #[must_use]
    pub fn to_esn_config(&self) -> ESNConfig {
        ESNConfig {
            input_size: self.input_size,
            reservoir_size: self.reservoir_size,
            output_size: self.output_size,
            spectral_radius: self.spectral_radius as f32,
            connectivity: self.connectivity as f32,
            leak_rate: self.leak_rate as f32,
            regularization: self.regularization as f32,
            seed: self.seed,
        }
    }

    /// Multi-head config for bio classifiers (diversity, taxonomy, AMR, bloom, disorder).
    #[must_use]
    pub fn multi_head(input_size: usize, head_count: usize) -> Self {
        Self {
            input_size,
            reservoir_size: 500,
            output_size: head_count,
            spectral_radius: 0.95,
            connectivity: 0.05,
            leak_rate: 0.3,
            regularization: crate::tolerances::ESN_REGULARIZATION,
            seed: 42,
        }
    }
}

/// Bio ESN wrapping `ToadStool`'s hardware-agnostic ESN.
///
/// Provides sync API over async `ToadStool` ESN for bio pipelines.
/// Exports weights for NPU deployment via `ExportedWeights` and `NpuReadoutWeights`.
pub struct BioEsn {
    inner: ESN,
}

impl BioEsn {
    /// Create a new `BioEsn` (blocks on async `ToadStool` ESN init).
    ///
    /// # Errors
    /// Returns `barracuda::error::BarracudaError` if config is invalid or device init fails.
    pub fn new(config: &BioEsnConfig) -> Result<Self, barracuda::error::BarracudaError> {
        let esn_config = config.to_esn_config();
        let inner = block_on(ESN::new(esn_config))?;
        Ok(Self { inner })
    }

    /// Reset reservoir state to zero.
    pub fn reset_state(&mut self) -> Result<(), barracuda::error::BarracudaError> {
        block_on(self.inner.reset_state())
    }

    /// Run one reservoir update step.
    pub fn update(&mut self, input: &[f64]) -> Result<(), barracuda::error::BarracudaError> {
        let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        let device = self.inner.state().device().clone();
        let input_tensor =
            Tensor::from_data(&input_f32, vec![self.inner.config().input_size, 1], device)?;
        block_on(self.inner.update(&input_tensor))?;
        Ok(())
    }

    /// Get current reservoir state (for NPU inference).
    pub fn state(&self) -> Result<Vec<f64>, barracuda::error::BarracudaError> {
        let vec_f32 = self.inner.state().to_vec()?;
        Ok(vec_f32.iter().map(|&x| f64::from(x)).collect())
    }

    /// Train readout via ridge regression on collected states.
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> Result<f32, barracuda::error::BarracudaError> {
        let inputs_f32: Vec<Vec<f32>> = inputs
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();
        let targets_f32: Vec<Vec<f32>> = targets
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();
        block_on(self.inner.train(&inputs_f32, &targets_f32))
    }

    /// Predict on a sequence of inputs.
    pub fn predict(
        &mut self,
        inputs: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, barracuda::error::BarracudaError> {
        self.reset_state()?;
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
            let out = block_on(self.inner.predict(&input_f32))?;
            outputs.push(out.iter().map(|&x| f64::from(x)).collect());
        }
        Ok(outputs)
    }

    /// Export readout weights as int8 for NPU deployment (wetSpring format).
    pub fn to_npu_weights(&self) -> Result<NpuReadoutWeights, barracuda::error::BarracudaError> {
        let exported = self.inner.export_weights()?;
        let w_out =
            exported
                .w_out
                .ok_or_else(|| barracuda::error::BarracudaError::InvalidOperation {
                    op: "BioEsn::to_npu_weights".to_string(),
                    reason: "ESN has not been trained yet — call train() first".to_string(),
                })?;
        let n_res = self.inner.config().reservoir_size;
        let n_out = self.inner.config().output_size;
        // ToadStool stores [reservoir_size, output_size]; wetSpring expects [output_size, reservoir_size].
        let mut w_out_f64 = vec![0.0_f64; n_out * n_res];
        for r in 0..n_res {
            for o in 0..n_out {
                w_out_f64[o * n_res + r] = f64::from(w_out[r * n_out + o]);
            }
        }
        Ok(NpuReadoutWeights::from_readout_weights(
            &w_out_f64, n_out, n_res,
        ))
    }

    /// Export all weights for cross-device deployment.
    pub fn export_weights(&self) -> Result<ExportedWeights, barracuda::error::BarracudaError> {
        self.inner.export_weights()
    }

    /// Migrate single-head weights to multi-head.
    pub fn migrate_to_multi_head(
        weights: &ExportedWeights,
        reservoir_size: usize,
        new_output_size: usize,
    ) -> Result<ExportedWeights, barracuda::error::BarracudaError> {
        weights.migrate_to_multi_head(reservoir_size, new_output_size)
    }

    /// Configuration used to build this ESN.
    #[must_use]
    pub fn config(&self) -> &ESNConfig {
        self.inner.config()
    }

    /// Whether the ESN has been trained.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }
}

/// Block on async using a shared runtime.
fn block_on<F, O>(f: F) -> O
where
    F: std::future::Future<Output = O>,
{
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    let rt = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|e| panic!("tokio runtime for ESN bridge: {e}"))
    });
    rt.block_on(f)
}
