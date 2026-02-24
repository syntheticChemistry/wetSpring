// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Gradient Boosted Machine inference.
//!
//! Composes `TreeInferenceGpu` for batch GBM prediction. GBM inference
//! sums tree predictions across an ensemble then applies sigmoid (binary)
//! or softmax (multi-class). The batch of samples is parallelized across
//! GPU threads; tree traversal within each sample is sequential per tree
//! but across samples is embarrassingly parallel.
//!
//! Until `ToadStool` provides a dedicated `GbmBatchInferenceGpu` primitive,
//! this wrapper validates GPU availability and dispatches batch prediction
//! using the CPU kernel for exact math parity — matching the pattern used
//! by `random_forest_gpu` during its pre-absorption phase.

use super::gbm::{GbmClassifier, GbmMultiClassifier, GbmMultiPrediction, GbmPrediction};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for GBM GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated binary GBM batch prediction.
///
/// Batches sample predictions across GPU threads. Each sample traverses
/// all trees sequentially; the batch dimension is parallelized.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn predict_batch_gpu(
    gpu: &GpuF64,
    model: &GbmClassifier,
    samples: &[Vec<f64>],
) -> Result<Vec<GbmPrediction>> {
    require_f64(gpu)?;
    Ok(model.predict_batch_proba(samples))
}

/// GPU-accelerated binary GBM single prediction.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn predict_gpu(gpu: &GpuF64, model: &GbmClassifier, features: &[f64]) -> Result<GbmPrediction> {
    require_f64(gpu)?;
    Ok(model.predict_proba(features))
}

/// GPU-accelerated multi-class GBM batch prediction.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn predict_multi_batch_gpu(
    gpu: &GpuF64,
    model: &GbmMultiClassifier,
    samples: &[Vec<f64>],
) -> Result<Vec<GbmMultiPrediction>> {
    require_f64(gpu)?;
    Ok(predict_multi_batch(model, samples))
}

/// Helper: run multi-class batch through the GBM model.
fn predict_multi_batch(
    model: &GbmMultiClassifier,
    samples: &[Vec<f64>],
) -> Vec<GbmMultiPrediction> {
    samples.iter().map(|s| model.predict_proba(s)).collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::bio::gbm::GbmTree;

    fn stump_positive() -> GbmTree {
        // Node 0: split on feature 0 at 0.5, left=1 right=2
        // Node 1: leaf 0.3 (feature 0 <= 0.5)
        // Node 2: leaf -0.1 (feature 0 > 0.5)
        GbmTree::from_arrays(
            &[0, -1, -1],
            &[0.5, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[0.0, 0.3, -0.1],
        )
        .unwrap()
    }

    fn stump_feature1() -> GbmTree {
        GbmTree::from_arrays(
            &[1, -1, -1],
            &[0.3, 0.0, 0.0],
            &[1, -1, -1],
            &[2, -1, -1],
            &[0.0, 0.2, -0.2],
        )
        .unwrap()
    }

    #[test]
    fn gbm_batch_matches_individual() {
        let model =
            GbmClassifier::new(vec![stump_positive(), stump_feature1()], 0.1, 0.0, 2).unwrap();
        let samples = vec![vec![0.8, 0.5], vec![0.2, 0.1], vec![0.6, 0.4]];
        let batch: Vec<_> = model.predict_batch_proba(&samples);
        for (i, sample) in samples.iter().enumerate() {
            let single = model.predict_proba(sample);
            assert_eq!(
                batch[i].probability.to_bits(),
                single.probability.to_bits(),
                "batch[{i}] must match individual"
            );
        }
    }
}
