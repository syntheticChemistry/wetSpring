// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated pairwise Jaccard distance via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::pairwise_jaccard::PairwiseJaccardGpu` —
//! evolved by `neuralSpring`, absorbed in `ToadStool` session 31f.
//!
//! `wetSpring` provides the high-level API that accepts a presence/absence
//! matrix and returns Jaccard distances.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::pairwise_jaccard::PairwiseJaccardGpu;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Pairwise Jaccard distance result.
pub struct JaccardGpuResult {
    /// Upper-triangle Jaccard distances, N*(N-1)/2 values.
    /// Each value is 1 - |intersection|/|union|.
    pub distances: Vec<f32>,
    /// Number of genomes.
    pub n_genomes: usize,
}

/// GPU-accelerated pairwise Jaccard distance for pangenome analysis.
///
/// Cross-spring provenance: `neuralSpring` (Write) → `ToadStool` (Absorb) → `wetSpring` (Lean).
pub struct JaccardGpu {
    device: Arc<WgpuDevice>,
    inner: PairwiseJaccardGpu,
}

impl JaccardGpu {
    /// Creates a new GPU Jaccard distance evaluator.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = PairwiseJaccardGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Compute pairwise Jaccard distances from a presence/absence matrix.
    ///
    /// `pa_matrix`: column-major `[n_genes × n_genomes]` f32.
    /// Values > 0.5 treated as present, <= 0.5 as absent.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix size mismatches or GPU read fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn pairwise_jaccard(
        &self,
        pa_matrix: &[f32],
        n_genomes: usize,
        n_genes: usize,
    ) -> crate::error::Result<JaccardGpuResult> {
        if pa_matrix.len() != n_genomes * n_genes {
            return Err(crate::error::Error::Gpu(format!(
                "PA matrix size mismatch: {} != {}×{}",
                pa_matrix.len(),
                n_genes,
                n_genomes
            )));
        }

        let d = self.device.device();
        let pa_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Jaccard PA"),
            contents: bytemuck::cast_slice(pa_matrix),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let n_pairs = n_genomes * (n_genomes - 1) / 2;
        let dist_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Jaccard dists"),
            size: (n_pairs * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.inner
            .dispatch(&pa_buf, &dist_buf, n_genomes as u32, n_genes as u32);
        d.poll(wgpu::Maintain::Wait);

        let distances = self
            .device
            .read_buffer_f32(&dist_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(JaccardGpuResult {
            distances,
            n_genomes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[tokio::test]
    async fn jaccard_gpu_basic() {
        let gpu = match GpuF64::new().await {
            Ok(g) => g,
            Err(_) => return,
        };
        let device = gpu.to_wgpu_device();
        let jg = JaccardGpu::new(&device);

        // 3 genomes, 4 genes, column-major
        #[rustfmt::skip]
        let pa = vec![
            1.0, 1.0, 0.0, // gene 0: g0=yes, g1=yes, g2=no
            1.0, 0.0, 1.0, // gene 1
            0.0, 1.0, 1.0, // gene 2
            1.0, 1.0, 1.0, // gene 3: all present
        ];

        let result = jg.pairwise_jaccard(&pa, 3, 4).expect("jaccard dispatch");
        assert_eq!(result.distances.len(), 3);
        // g0 vs g1: intersection={0,3}=2, union={0,1,2,3}=4, J=2/4=0.5, dist=0.5
        assert!((result.distances[0] - 0.5).abs() < 1e-5);
    }
}
