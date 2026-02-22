// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated HMM batch forward algorithm via `ToadStool`.
//!
//! Delegates to `barracuda::ops::bio::hmm::HmmBatchForwardF64` — the
//! absorbed shader from wetSpring handoff v6. wetSpring provides the
//! high-level API that marshals [`HmmModel`] to GPU buffers.
//!
//! # GPU Strategy
//!
//! The forward algorithm is sequential over time steps but independent
//! across observation sequences. Each thread runs the full T-step
//! forward for one sequence, yielding one log-likelihood per sequence.

use barracuda::device::WgpuDevice;
use barracuda::ops::bio::hmm::HmmBatchForwardF64;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::hmm::HmmModel;

/// GPU-accelerated HMM batch forward algorithm.
pub struct HmmGpuForward {
    device: Arc<WgpuDevice>,
    inner: HmmBatchForwardF64,
}

/// Result of GPU batch forward.
pub struct HmmGpuResult {
    /// Per-sequence log-forward variables: `[n_seqs × n_steps × n_states]`.
    pub log_alpha: Vec<f64>,
    /// Per-sequence log-likelihoods: `[n_seqs]`.
    pub log_likelihoods: Vec<f64>,
    /// Number of sequences.
    pub n_seqs: usize,
    /// Number of time steps per sequence.
    pub n_steps: usize,
    /// Number of HMM states.
    pub n_states: usize,
}

impl HmmGpuForward {
    /// Create a new HMM GPU forward instance.
    ///
    /// # Errors
    ///
    /// Returns an error if `ToadStool` shader compilation fails.
    pub fn new(device: &Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let inner = HmmBatchForwardF64::new(Arc::clone(device))
            .map_err(|e| crate::error::Error::Gpu(format!("HmmBatchForwardF64: {e}")))?;
        Ok(Self {
            device: Arc::clone(device),
            inner,
        })
    }

    /// Run batch forward on N observation sequences.
    ///
    /// All sequences must have the same length `n_steps`.
    /// `observations` is row-major `[n_seqs × n_steps]` with symbol indices.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or buffer readback fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn forward_batch(
        &self,
        model: &HmmModel,
        observations: &[u32],
        n_seqs: usize,
        n_steps: usize,
    ) -> crate::error::Result<HmmGpuResult> {
        let d = self.device.device();
        let s = model.n_states;
        let alpha_size = n_seqs * n_steps * s;

        let trans_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM log_trans"),
            contents: bytemuck::cast_slice(&model.log_trans),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let emit_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM log_emit"),
            contents: bytemuck::cast_slice(&model.log_emit),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pi_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM log_pi"),
            contents: bytemuck::cast_slice(&model.log_pi),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let obs_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM observations"),
            contents: bytemuck::cast_slice(observations),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let alpha_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM log_alpha"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; alpha_size]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let lik_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM log_lik"),
            contents: bytemuck::cast_slice(&vec![0.0_f64; n_seqs]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        self.inner
            .dispatch(
                s as u32,
                model.n_symbols as u32,
                n_steps as u32,
                n_seqs as u32,
                &trans_buf,
                &emit_buf,
                &pi_buf,
                &obs_buf,
                &alpha_buf,
                &lik_buf,
            )
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        d.poll(wgpu::Maintain::Wait);

        let log_alpha = self
            .device
            .read_buffer_f64(&alpha_buf, alpha_size)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let log_likelihoods = self
            .device
            .read_buffer_f64(&lik_buf, n_seqs)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;

        Ok(HmmGpuResult {
            log_alpha,
            log_likelihoods,
            n_seqs,
            n_steps,
            n_states: s,
        })
    }
}
