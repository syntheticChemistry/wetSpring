// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch pairwise ANI via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::ani::AniBatchF64` — the absorbed
//! shader from wetSpring handoff v6. wetSpring provides the high-level
//! API that encodes byte sequences to GPU buffers.
//!
//! # GPU Strategy
//!
//! ANI is position-sequential but pair-parallel. For N pairs of length
//! L, dispatch N threads. No transcendentals needed (integer counting
//! + one f64 division), so no polyfill required.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::ani::AniBatchF64;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU batch ANI result.
pub struct AniGpuResult {
    /// ANI values per pair, in [0.0, 1.0].
    pub ani_values: Vec<f64>,
    /// Aligned positions per pair.
    pub aligned_counts: Vec<u32>,
    /// Identical positions per pair.
    pub identical_counts: Vec<u32>,
}

/// GPU-accelerated batch ANI computation.
pub struct AniGpu {
    device: Arc<WgpuDevice>,
    inner: AniBatchF64,
}

impl AniGpu {
    /// Create a new ANI GPU compute instance.
    ///
    /// # Errors
    ///
    /// Returns an error if `ToadStool` shader compilation fails.
    pub fn new(device: &Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let inner = AniBatchF64::new(Arc::clone(device))
            .map_err(|e| crate::error::Error::Gpu(format!("AniBatchF64: {e}")))?;
        Ok(Self {
            device: Arc::clone(device),
            inner,
        })
    }

    /// Compute ANI for a batch of sequence pairs on the GPU.
    ///
    /// Sequences are `(seq_a, seq_b)` pairs of equal-length byte slices.
    /// Bases: A/C/G/T as ASCII. Gaps (`-`, `.`) and `N` are excluded.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or buffer readback fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn batch_ani(&self, pairs: &[(&[u8], &[u8])]) -> crate::error::Result<AniGpuResult> {
        let n_pairs = pairs.len();
        if n_pairs == 0 {
            return Ok(AniGpuResult {
                ani_values: Vec::new(),
                aligned_counts: Vec::new(),
                identical_counts: Vec::new(),
            });
        }

        let max_seq_len = pairs
            .iter()
            .map(|(a, b)| a.len().max(b.len()))
            .max()
            .unwrap_or(0);

        let encode_base = |b: u8| -> u32 {
            match b.to_ascii_uppercase() {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                b'-' | b'.' => 4,
                _ => 5,
            }
        };

        let mut seq_a_flat: Vec<u32> = vec![5; n_pairs * max_seq_len];
        let mut seq_b_flat: Vec<u32> = vec![5; n_pairs * max_seq_len];
        for (i, (a, b)) in pairs.iter().enumerate() {
            let base = i * max_seq_len;
            for (j, &byte) in a.iter().enumerate() {
                seq_a_flat[base + j] = encode_base(byte);
            }
            for (j, &byte) in b.iter().enumerate() {
                seq_b_flat[base + j] = encode_base(byte);
            }
        }

        let d = self.device.device();

        let seq_a_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI seq_a"),
            contents: bytemuck::cast_slice(&seq_a_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let seq_b_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI seq_b"),
            contents: bytemuck::cast_slice(&seq_b_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ani_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI out"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let aligned_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI aligned"),
            contents: bytemuck::cast_slice(&vec![0u32; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let identical_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ANI identical"),
            contents: bytemuck::cast_slice(&vec![0u32; n_pairs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner
            .dispatch(
                n_pairs as u32,
                max_seq_len as u32,
                &seq_a_buf,
                &seq_b_buf,
                &ani_buf,
                &aligned_buf,
                &identical_buf,
            )
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        d.poll(wgpu::Maintain::Wait);

        let ani_values = self
            .device
            .read_buffer_f64(&ani_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let aligned_counts = self
            .device
            .read_buffer_u32(&aligned_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let identical_counts = self
            .device
            .read_buffer_u32(&identical_buf, n_pairs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(AniGpuResult {
            ani_values,
            aligned_counts,
            identical_counts,
        })
    }
}
