// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::missing_errors_doc, clippy::missing_const_for_fn)]
//! barraCuda ESN bridge for bio use cases.
//!
//! Two tiers:
//! - [`BioEsn`]: single-head ESN for backward compatibility and simple classifiers.
//! - [`MultiHeadBioEsn`]: wraps barraCuda `MultiHeadEsn` (S79) — shared reservoir
//!   with per-head bio readouts, head disagreement uncertainty, and labeled exports.
//!
//! # Cross-spring provenance
//!
//! - ESN core: hotSpring V0615 (36-head concept, `MultiHeadEsn`)
//! - Reservoir WGSL shaders: neuralSpring V24 (`LstmReservoir`, `EsnClassifier`)
//! - Ridge regression readout: barraCuda `barracuda::linalg::solve_f64_cpu`
//! - Bio feature mapping: wetSpring (diversity, taxonomy, AMR, bloom, disorder heads)

use super::{EsnConfig, NpuReadoutWeights};
use barracuda::esn_v2::{ESN, ESNConfig, ExportedWeights, HeadConfig, MultiHeadEsn};
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

/// Bio-specific ESN configuration mapping to barraCuda `ESNConfig`.
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
    /// Convert to barraCuda `ESNConfig` (f32).
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

/// Bio ESN wrapping barraCuda's hardware-agnostic ESN.
///
/// Provides sync API over async barraCuda ESN for bio pipelines.
/// Exports weights for NPU deployment via `ExportedWeights` and `NpuReadoutWeights`.
pub struct BioEsn {
    inner: ESN,
}

impl BioEsn {
    /// Create a new `BioEsn` (blocks on async barraCuda ESN init).
    ///
    /// # Errors
    /// Returns `barracuda::error::BarracudaError` if config is invalid or device init fails.
    pub fn new(config: &BioEsnConfig) -> Result<Self, barracuda::error::BarracudaError> {
        let esn_config = config.to_esn_config();
        let inner = block_on(ESN::new(esn_config))??;
        Ok(Self { inner })
    }

    /// Reset reservoir state to zero.
    pub fn reset_state(&mut self) -> Result<(), barracuda::error::BarracudaError> {
        block_on(self.inner.reset_state())?
    }

    /// Run one reservoir update step.
    pub fn update(&mut self, input: &[f64]) -> Result<(), barracuda::error::BarracudaError> {
        let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        let device = self.inner.state().device().clone();
        let input_tensor =
            Tensor::from_data(&input_f32, vec![self.inner.config().input_size, 1], device)?;
        block_on(self.inner.update(&input_tensor))??;
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
        block_on(self.inner.train(&inputs_f32, &targets_f32))?
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
            let out = block_on(self.inner.predict(&input_f32))??;
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
        // barraCuda stores [reservoir_size, output_size]; wetSpring expects [output_size, reservoir_size].
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
    ///
    /// Includes `head_labels` for bio heads (barraCuda S79+).
    pub fn export_weights(&self) -> Result<ExportedWeights, barracuda::error::BarracudaError> {
        let mut weights = self.inner.export_weights()?;
        if weights.head_labels.is_empty() && self.inner.config().output_size > 1 {
            weights.head_labels = bio_head_labels(self.inner.config().output_size);
        }
        Ok(weights)
    }

    /// Migrate single-head weights to multi-head with bio labels.
    pub fn migrate_to_multi_head(
        weights: &ExportedWeights,
        reservoir_size: usize,
        new_output_size: usize,
    ) -> Result<ExportedWeights, barracuda::error::BarracudaError> {
        let mut migrated = weights.migrate_to_multi_head(reservoir_size, new_output_size)?;
        if migrated.head_labels.is_empty() {
            migrated.head_labels = bio_head_labels(new_output_size);
        }
        Ok(migrated)
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

/// Multi-head bio ESN wrapping barraCuda's `MultiHeadEsn` (S79).
///
/// Shared reservoir with per-head readouts for bio classification domains.
/// Each head is independently trainable and exportable. Head disagreement
/// measures uncertainty across bio domains.
///
/// # Cross-spring provenance
///
/// - Shared reservoir: hotSpring V0615 (36-head concept → `ToadStool` `MultiHeadEsn`)
/// - Per-head ridge regression: barraCuda `barracuda::linalg::solve_f64_cpu`
/// - Bio head mapping: wetSpring (diversity/taxonomy/AMR/bloom/disorder)
/// - Head disagreement: hotSpring uncertainty metric (mean pairwise L2)
pub struct MultiHeadBioEsn {
    inner: MultiHeadEsn,
    head_labels: Vec<String>,
}

impl MultiHeadBioEsn {
    /// Create a multi-head bio ESN with the standard 5 bio heads.
    ///
    /// Heads: diversity (0), taxonomy (1), AMR (2), bloom (3), disorder (4).
    pub fn new_bio5(config: &BioEsnConfig) -> Result<Self, barracuda::error::BarracudaError> {
        let heads = vec![
            HeadConfig {
                group: barracuda::esn_v2::HeadGroup::Meta,
                label: "diversity".to_string(),
                output_size: 1,
            },
            HeadConfig {
                group: barracuda::esn_v2::HeadGroup::Meta,
                label: "taxonomy".to_string(),
                output_size: 1,
            },
            HeadConfig {
                group: barracuda::esn_v2::HeadGroup::Meta,
                label: "amr".to_string(),
                output_size: 1,
            },
            HeadConfig {
                group: barracuda::esn_v2::HeadGroup::Meta,
                label: "bloom".to_string(),
                output_size: 1,
            },
            HeadConfig {
                group: barracuda::esn_v2::HeadGroup::Meta,
                label: "disorder".to_string(),
                output_size: 1,
            },
        ];
        Self::new(config, heads)
    }

    /// Create a multi-head bio ESN with custom head configuration.
    pub fn new(
        config: &BioEsnConfig,
        heads: Vec<HeadConfig>,
    ) -> Result<Self, barracuda::error::BarracudaError> {
        let head_labels: Vec<String> = heads.iter().map(|h| h.label.clone()).collect();
        let esn_config = config.to_esn_config();
        let inner = block_on(MultiHeadEsn::new(esn_config, heads))??;
        Ok(Self { inner, head_labels })
    }

    /// Train a single bio head from pre-collected reservoir states.
    ///
    /// `states`: flat `[reservoir_size × n_samples]` (column-major).
    /// `targets`: flat `[head_output_size × n_samples]`.
    pub fn train_head(
        &mut self,
        head: BioHeadKind,
        states: &[f64],
        targets: &[f64],
        lambda: f64,
    ) -> Result<(), barracuda::error::BarracudaError> {
        let idx = head.index(self.head_labels.len());
        self.inner.train_head(idx, states, targets, lambda)
    }

    /// Mean pairwise L2 distance between head predictions (uncertainty signal).
    ///
    /// Higher disagreement → lower confidence in the consensus prediction.
    /// Requires a reservoir state `Tensor` (from barraCuda ESN forward pass).
    pub fn head_disagreement(
        &self,
        state: &Tensor,
    ) -> Result<f64, barracuda::error::BarracudaError> {
        self.inner.head_disagreement(state)
    }

    /// Export weights with bio head labels populated.
    pub fn export_weights(&self) -> Result<ExportedWeights, barracuda::error::BarracudaError> {
        self.inner.export_weights()
    }

    /// Head labels for this ESN.
    #[must_use]
    pub fn head_labels(&self) -> &[String] {
        &self.head_labels
    }

    /// Number of bio heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.head_labels.len()
    }
}

/// Default bio head labels for multi-head ESN (matches [`BioHeadKind`] order).
fn bio_head_labels(output_size: usize) -> Vec<String> {
    const NAMES: &[&str] = &["diversity", "taxonomy", "amr", "bloom", "disorder"];
    (0..output_size)
        .map(|i| {
            NAMES
                .get(i)
                .map_or_else(|| format!("custom_{i}"), |s| (*s).to_string())
        })
        .collect()
}

/// Block on async using a shared runtime.
///
/// # Errors
///
/// Returns `BarracudaError::InvalidOperation` if the tokio runtime cannot be created.
fn block_on<F, O>(f: F) -> Result<O, barracuda::error::BarracudaError>
where
    F: std::future::Future<Output = O>,
{
    static RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    let rt_result = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())
    });
    match rt_result {
        Ok(rt) => Ok(rt.block_on(f)),
        Err(msg) => Err(barracuda::error::BarracudaError::InvalidOperation {
            op: "ESN bridge runtime init".to_string(),
            reason: msg.clone(),
        }),
    }
}
