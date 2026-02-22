// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated locus variance (FST decomposition) via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::locus_variance::LocusVarianceGpu` —
//! evolved by `neuralSpring`, absorbed in `ToadStool` session 31f.
//!
//! Computes per-locus allele frequency variance across populations,
//! the core building block for Weir-Cockerham FST estimation.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::locus_variance::LocusVarianceGpu;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-accelerated locus variance for FST decomposition.
///
/// Cross-spring provenance: `neuralSpring` (Write) → `ToadStool` (Absorb) → `wetSpring` (Lean).
pub struct LocusVarianceGpuWrapper {
    device: Arc<WgpuDevice>,
    inner: LocusVarianceGpu,
}

impl LocusVarianceGpuWrapper {
    /// Creates a new GPU locus variance evaluator.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = LocusVarianceGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Compute per-locus allele frequency variance across populations.
    ///
    /// `allele_freqs`: row-major `[n_pops × n_loci]` f32.
    /// Returns `[n_loci]` population variances.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix size mismatches or GPU read fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn compute(
        &self,
        allele_freqs: &[f32],
        n_pops: usize,
        n_loci: usize,
    ) -> crate::error::Result<Vec<f32>> {
        if allele_freqs.len() != n_pops * n_loci {
            return Err(crate::error::Error::Gpu(format!(
                "AF matrix size mismatch: {} != {}×{}",
                allele_freqs.len(),
                n_pops,
                n_loci
            )));
        }

        let d = self.device.device();

        let freq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LocusVar freqs"),
            contents: bytemuck::cast_slice(allele_freqs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let var_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LocusVar output"),
            size: (n_loci * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.inner
            .dispatch(&freq_buf, &var_buf, n_pops as u32, n_loci as u32);
        d.poll(wgpu::Maintain::Wait);

        self.device
            .read_buffer_f32(&var_buf, n_loci)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[tokio::test]
    async fn locus_variance_uniform_is_zero() {
        let gpu = match GpuF64::new().await {
            Ok(g) => g,
            Err(_) => return,
        };
        let device = gpu.to_wgpu_device();
        let lv = LocusVarianceGpuWrapper::new(&device);

        // 3 pops × 2 loci, row-major
        // Locus 0: uniform 0.5 → variance = 0
        // Locus 1: 0.1, 0.5, 0.9 → mean=0.5, var = (0.16+0+0.16)/3 ≈ 0.1067
        let freqs = vec![
            0.5f32, 0.1, // pop0
            0.5, 0.5, // pop1
            0.5, 0.9, // pop2
        ];
        let result = lv.compute(&freqs, 3, 2).expect("locus variance");
        assert_eq!(result.len(), 2);
        assert!(result[0].abs() < 1e-5, "uniform locus should be 0");
        assert!((result[1] - 0.10666667).abs() < 1e-4);
    }
}
