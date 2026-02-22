// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated pairwise Hamming distance via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::pairwise_hamming::PairwiseHammingGpu` —
//! evolved by `neuralSpring`, absorbed in `ToadStool` session 31f.
//!
//! `wetSpring` provides the high-level API that accepts `&[u8]` nucleotide
//! sequences and returns normalized Hamming distances.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::pairwise_hamming::PairwiseHammingGpu;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Pairwise Hamming distance result.
pub struct HammingGpuResult {
    /// Upper-triangle distances in row-major order, N*(N-1)/2 values.
    /// Each value is normalized: `differing_sites` / `seq_len`.
    pub distances: Vec<f32>,
    /// Number of sequences.
    pub n_seqs: usize,
}

/// GPU-accelerated pairwise Hamming distance.
///
/// Cross-spring provenance: `neuralSpring` (Write) → `ToadStool` (Absorb) → `wetSpring` (Lean).
pub struct HammingGpu {
    device: Arc<WgpuDevice>,
    inner: PairwiseHammingGpu,
}

impl HammingGpu {
    /// Creates a new GPU Hamming distance evaluator.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = PairwiseHammingGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Compute pairwise Hamming distances for encoded sequences.
    ///
    /// `sequences`: slice of equal-length encoded sequences (each `&[u32]`
    /// where 0=A, 1=C, 2=G, 3=T).
    ///
    /// Returns N*(N-1)/2 normalized distances in upper-triangle order.
    ///
    /// # Errors
    ///
    /// Returns an error if sequences have unequal lengths or GPU read fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn pairwise_hamming(&self, sequences: &[&[u32]]) -> crate::error::Result<HammingGpuResult> {
        let n_seqs = sequences.len();
        if n_seqs < 2 {
            return Ok(HammingGpuResult {
                distances: Vec::new(),
                n_seqs,
            });
        }
        let seq_len = sequences[0].len();
        if sequences.iter().any(|s| s.len() != seq_len) {
            return Err(crate::error::Error::Gpu(
                "All sequences must have equal length".into(),
            ));
        }

        let flat: Vec<u32> = sequences.iter().flat_map(|s| s.iter().copied()).collect();
        let d = self.device.device();

        let seq_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hamming seqs"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let n_pairs = n_seqs * (n_seqs - 1) / 2;
        let dist_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hamming dists"),
            size: (n_pairs * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.inner
            .dispatch(&seq_buf, &dist_buf, n_seqs as u32, seq_len as u32);
        d.poll(wgpu::Maintain::Wait);

        let distances = self
            .device
            .read_buffer_f32(&dist_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(HammingGpuResult { distances, n_seqs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[tokio::test]
    async fn hamming_gpu_matches_cpu() {
        let gpu = match GpuF64::new().await {
            Ok(g) => g,
            Err(_) => return,
        };
        let device = gpu.to_wgpu_device();
        let hg = HammingGpu::new(&device);

        let s0: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let s1: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 0]; // ACGTACGA (differs at pos 7)
        let s2: Vec<u32> = vec![3, 3, 3, 3, 3, 3, 3, 3]; // TTTTTTTT

        let result = hg
            .pairwise_hamming(&[&s0, &s1, &s2])
            .expect("hamming dispatch");
        assert_eq!(result.distances.len(), 3);
        assert!((result.distances[0] - 0.125).abs() < 1e-6); // s0 vs s1
        assert!((result.distances[1] - 0.75).abs() < 1e-6); // s0 vs s2
        assert!((result.distances[2] - 0.875).abs() < 1e-6); // s1 vs s2
    }
}
