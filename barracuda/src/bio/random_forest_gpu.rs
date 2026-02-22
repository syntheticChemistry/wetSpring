// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch Random Forest inference via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::rf_inference::RfBatchInferenceGpu` —
//! the absorbed shader from wetSpring handoff v5. wetSpring provides the
//! high-level API that marshals `RandomForest` to `SoA` GPU buffers and
//! reduces per-tree predictions via majority vote.
//!
//! Uses a `SoA` (Structure of Arrays) layout with separate buffers for
//! node features (i32), thresholds (f64), and children (i32) to avoid
//! bitcast issues in WGSL.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::rf_inference::RfBatchInferenceGpu;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::random_forest::{RandomForest, RfPrediction};

/// GPU-accelerated random forest batch inference.
pub struct RandomForestGpu {
    device: Arc<WgpuDevice>,
    inner: RfBatchInferenceGpu,
}

impl RandomForestGpu {
    /// Create a new random forest GPU instance.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = RfBatchInferenceGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Run batch RF inference on GPU, returns per-sample predictions.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU buffer creation or readback fails.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub fn predict_batch(
        &self,
        forest: &RandomForest,
        samples: &[Vec<f64>],
    ) -> crate::error::Result<Vec<RfPrediction>> {
        let n_samples = samples.len();
        let n_trees = forest.n_trees();
        let n_features = forest.n_features();
        let n_classes = forest.n_classes();

        if n_samples == 0 || n_trees == 0 {
            return Ok(Vec::new());
        }

        let n_nodes_max = (0..n_trees)
            .map(|i| forest.tree_at(i).n_nodes())
            .max()
            .unwrap_or(1);

        let flat_nodes = n_trees * n_nodes_max;
        let mut node_features_flat: Vec<i32> = vec![-1; flat_nodes];
        let mut node_thresh_flat: Vec<f64> = vec![0.0; flat_nodes];
        let mut node_children_flat: Vec<i32> = vec![0; flat_nodes * 2];

        for t in 0..n_trees {
            let tree = forest.tree_at(t);
            let base = t * n_nodes_max;
            for n in 0..tree.n_nodes() {
                let node = tree.node_at(n);
                node_features_flat[base + n] = node.feature;
                node_thresh_flat[base + n] = node.threshold;
                let coff = (base + n) * 2;
                if node.feature < 0 {
                    node_children_flat[coff] = node.prediction.unwrap_or(0) as i32;
                    node_children_flat[coff + 1] = -1;
                } else {
                    node_children_flat[coff] = node.left_child;
                    node_children_flat[coff + 1] = node.right_child;
                }
            }
        }

        let mut features_flat: Vec<f64> = vec![0.0; n_samples * n_features];
        for (i, sample) in samples.iter().enumerate() {
            for (j, &val) in sample.iter().enumerate() {
                if j < n_features {
                    features_flat[i * n_features + j] = val;
                }
            }
        }

        let d = self.device.device();

        let nf_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF node features"),
            contents: bytemuck::cast_slice(&node_features_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nt_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF node thresholds"),
            contents: bytemuck::cast_slice(&node_thresh_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nc_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF node children"),
            contents: bytemuck::cast_slice(&node_children_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let feat_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF features"),
            contents: bytemuck::cast_slice(&features_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_gpu = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RF predictions"),
            contents: bytemuck::cast_slice(&vec![0u32; n_samples * n_trees]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner.dispatch(
            &nf_gpu,
            &nt_gpu,
            &nc_gpu,
            &feat_gpu,
            &out_gpu,
            n_samples as u32,
            n_trees as u32,
            n_nodes_max as u32,
            n_features as u32,
        );

        d.poll(wgpu::Maintain::Wait);

        let raw = self
            .device
            .read_buffer_u32(&out_gpu, n_samples * n_trees)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        let mut results = Vec::with_capacity(n_samples);
        for s in 0..n_samples {
            let mut votes = vec![0usize; n_classes];
            for t in 0..n_trees {
                let pred = raw[s * n_trees + t] as usize;
                if pred < n_classes {
                    votes[pred] += 1;
                }
            }
            let (class, &max_votes) = votes
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .unwrap_or((0, &0));
            #[allow(clippy::cast_precision_loss)]
            let confidence = max_votes as f64 / n_trees as f64;
            results.push(RfPrediction {
                class,
                votes,
                confidence,
            });
        }

        Ok(results)
    }
}
