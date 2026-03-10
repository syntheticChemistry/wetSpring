// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated `UniFrac` tree propagation via barraCuda.
//!
//! Delegates to `barracuda::ops::bio::unifrac_propagate::UniFracPropagateGpu`.
//! wetSpring provides the high-level API that builds CSR tree buffers from
//! the CPU `FlatTree` representation and dispatches `leaf_init` + bottom-up
//! propagation on GPU.
//!
//! # GPU Strategy
//!
//! Two-pass dispatch: (1) `leaf_init` copies sample abundances into leaf slots,
//! (2) `propagate_level` sums child contributions x branch lengths bottom-up.
//! Each node is independent within a level — one thread per node.

use barracuda::UniFracPropagateGpu;
use barracuda::device::WgpuDevice;
use barracuda::ops::bio::unifrac_propagate::UniFracConfig;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU `UniFrac` propagation engine.
pub struct UniFracGpu {
    device: Arc<WgpuDevice>,
    inner: UniFracPropagateGpu,
}

/// GPU `UniFrac` propagation result.
pub struct UniFracGpuResult {
    /// Node sums: flat `[n_nodes × n_samples]` f64 array.
    pub node_sums: Vec<f64>,
    /// Number of nodes in the tree.
    pub n_nodes: usize,
    /// Number of samples.
    pub n_samples: usize,
}

impl UniFracGpu {
    /// Create a new `UniFrac` GPU instance.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = UniFracPropagateGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Propagate sample abundances through a CSR phylogenetic tree on GPU.
    ///
    /// # Arguments
    ///
    /// * `parent_array` — parent index for each node (`[n_nodes]`, root's parent = root)
    /// * `branch_lengths` — branch length for each node (`[n_nodes]`)
    /// * `sample_matrix` — leaf abundances, flat `[n_leaves × n_samples]`
    /// * `n_nodes` — total nodes in tree
    /// * `n_samples` — number of samples
    /// * `n_leaves` — number of leaf nodes (must be first `n_leaves` entries)
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or readback fails.
    #[expect(clippy::cast_possible_truncation)]
    pub fn propagate(
        &self,
        parent_array: &[u32],
        branch_lengths: &[f64],
        sample_matrix: &[f64],
        n_nodes: usize,
        n_samples: usize,
        n_leaves: usize,
    ) -> crate::error::Result<UniFracGpuResult> {
        let d = self.device.device();

        let config = UniFracConfig {
            n_nodes: n_nodes as u32,
            n_samples: n_samples as u32,
            n_leaves: n_leaves as u32,
            _pad: 0,
        };

        let parent_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UniFrac parents"),
            contents: bytemuck::cast_slice(parent_array),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let branch_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UniFrac branches"),
            contents: bytemuck::cast_slice(branch_lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sample_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UniFrac samples"),
            contents: bytemuck::cast_slice(sample_matrix),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sums_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UniFrac node_sums"),
            contents: bytemuck::cast_slice(&vec![0.0f64; n_nodes * n_samples]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner
            .dispatch_leaf_init(&config, &parent_buf, &branch_buf, &sample_buf, &sums_buf);
        let _ = d.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        self.inner.dispatch_propagate_level(
            &config,
            &parent_buf,
            &branch_buf,
            &sample_buf,
            &sums_buf,
        );
        let _ = d.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let node_sums = self
            .device
            .read_buffer_f64(&sums_buf, n_nodes * n_samples)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(UniFracGpuResult {
            node_sums,
            n_nodes,
            n_samples,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::manual_let_else
)]
mod tests {
    use super::*;

    #[test]
    fn api_surface_compiles() {
        fn _assert_result(_: &UniFracGpuResult) {}
        let _: fn(
            &UniFracGpu,
            &[u32],
            &[f64],
            &[f64],
            usize,
            usize,
            usize,
        ) -> crate::error::Result<UniFracGpuResult> = UniFracGpu::propagate;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match crate::gpu::GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let device = gpu.to_wgpu_device();
        let unifrac = UniFracGpu::new(&device);
        let parent_array = [0u32, 0, 1];
        let branch_lengths = [0.0, 1.0, 1.0];
        let sample_matrix = [1.0, 0.0, 0.0, 1.0];
        let result = unifrac.propagate(&parent_array, &branch_lengths, &sample_matrix, 3, 2, 2);
        assert!(result.is_ok(), "propagate should succeed with valid input");
    }
}
