// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated k-mer histogram via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::kmer_histogram::KmerHistogramGpu`.
//! wetSpring provides the high-level API that encodes DNA sequences to
//! 2-bit packed k-mer indices and dispatches to GPU for histogram counting.
//!
//! # GPU Strategy
//!
//! One GPU thread per k-mer. Atomic increments into a `4^k` histogram buffer.
//! For short k (4-8), the histogram fits in L1; for k=16, 4^16 = 4 GB — only
//! practical with streaming or hash-based approaches.

use barracuda::KmerHistogramGpu;
use barracuda::device::WgpuDevice;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU k-mer histogram result.
pub struct KmerGpuResult {
    /// Histogram of k-mer counts. Length = `4^k`.
    pub histogram: Vec<u32>,
    /// Total k-mers dispatched to GPU.
    pub n_kmers: usize,
}

/// GPU-accelerated k-mer counting.
pub struct KmerGpu {
    device: Arc<WgpuDevice>,
    inner: KmerHistogramGpu,
}

impl KmerGpu {
    /// Create a new k-mer GPU compute instance.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let inner = KmerHistogramGpu::new(Arc::clone(device));
        Self {
            device: Arc::clone(device),
            inner,
        }
    }

    /// Count k-mers on the GPU from pre-encoded indices.
    ///
    /// `kmer_indices`: each value must be < `4^k`.
    /// Returns a histogram of length `4^k`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU buffer readback fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_histogram(
        &self,
        kmer_indices: &[u32],
        k: u32,
    ) -> crate::error::Result<KmerGpuResult> {
        let hist_len = 4_usize.pow(k);
        let n_kmers = kmer_indices.len();

        if n_kmers == 0 {
            return Ok(KmerGpuResult {
                histogram: vec![0u32; hist_len],
                n_kmers: 0,
            });
        }

        let d = self.device.device();

        let kmer_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KmerGpu indices"),
            contents: bytemuck::cast_slice(kmer_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let hist_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KmerGpu histogram"),
            contents: bytemuck::cast_slice(&vec![0u32; hist_len]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner.dispatch(&kmer_buf, &hist_buf, n_kmers as u32, k);
        d.poll(wgpu::Maintain::Wait);

        let histogram = self
            .device
            .read_buffer_u32(&hist_buf, hist_len)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(KmerGpuResult { histogram, n_kmers })
    }

    /// Encode a DNA sequence to k-mer indices and count on GPU.
    ///
    /// Encodes `A=0, C=1, G=2, T=3`. Ambiguous bases skip the k-mer window.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or readback fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn count_from_sequence(
        &self,
        sequence: &[u8],
        k: u32,
    ) -> crate::error::Result<KmerGpuResult> {
        let ku = k as usize;
        if sequence.len() < ku {
            return Ok(KmerGpuResult {
                histogram: vec![0u32; 4_usize.pow(k)],
                n_kmers: 0,
            });
        }

        let mask = (1u32 << (2 * k)) - 1;
        let mut indices = Vec::with_capacity(sequence.len() - ku + 1);
        let mut window = 0u32;
        let mut valid = 0usize;

        for (i, &base) in sequence.iter().enumerate() {
            let encoded = match base {
                b'A' | b'a' => 0u32,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => {
                    valid = 0;
                    window = 0;
                    continue;
                }
            };
            window = ((window << 2) | encoded) & mask;
            valid += 1;
            if valid >= ku {
                indices.push(window);
            }
            let _ = i;
        }

        self.count_histogram(&indices, k)
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        fn _assert_kmer_result(_: &KmerGpuResult) {}
        let _ = KmerGpu::new;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn kmer_gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let device = gpu.to_wgpu_device();
        let kmer = KmerGpu::new(&device);
        let result = kmer.count_from_sequence(b"ACGT", 2);
        assert!(result.is_ok(), "count_from_sequence should succeed");
    }
}
