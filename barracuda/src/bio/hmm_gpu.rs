// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated HMM batch forward algorithm.
//!
//! Runs the forward algorithm for N independent observation sequences
//! through the same HMM model, with one thread per sequence.
//!
//! Uses a local WGSL shader (`hmm_forward_f64.wgsl`) with native f64
//! — a ToadStool absorption candidate following Write → Absorb → Lean.
//!
//! # GPU Strategy
//!
//! The forward algorithm is sequential over time steps but independent
//! across observation sequences. Each thread runs the full T-step
//! forward for one sequence, yielding one log-likelihood per sequence.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::hmm::HmmModel;

const HMM_WGSL: &str = include_str!("../shaders/hmm_forward_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmParams {
    n_states: u32,
    n_symbols: u32,
    n_steps: u32,
    n_seqs: u32,
}

pub struct HmmGpuForward {
    device: Arc<WgpuDevice>,
}

/// Result of GPU batch forward.
pub struct HmmGpuResult {
    /// Per-sequence log-forward variables: `[n_seqs × n_steps × n_states]`.
    pub log_alpha: Vec<f64>,
    /// Per-sequence log-likelihoods: `[n_seqs]`.
    pub log_likelihoods: Vec<f64>,
    pub n_seqs: usize,
    pub n_steps: usize,
    pub n_states: usize,
}

impl HmmGpuForward {
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }

    /// Run batch forward on N observation sequences.
    ///
    /// All sequences must have the same length `n_steps`.
    /// `observations` is row-major `[n_seqs × n_steps]` with symbol indices.
    pub fn forward_batch(
        &self,
        model: &HmmModel,
        observations: &[u32],
        n_seqs: usize,
        n_steps: usize,
    ) -> crate::error::Result<HmmGpuResult> {
        let dev = &self.device;
        let d = dev.device();
        let s = model.n_states;

        let alpha_size = n_seqs * n_steps * s;
        let alpha_init: Vec<f64> = vec![0.0; alpha_size];
        let lik_init: Vec<f64> = vec![0.0; n_seqs];

        let params = HmmParams {
            n_states: s as u32,
            n_symbols: model.n_symbols as u32,
            n_steps: n_steps as u32,
            n_seqs: n_seqs as u32,
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
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
            contents: bytemuck::cast_slice(&alpha_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let lik_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HMM log_lik"),
            contents: bytemuck::cast_slice(&lik_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Force exp/log polyfill — RTX 4070 NVVM can't compile native f64 transcendentals
        let patched = ShaderTemplate::for_driver_auto(HMM_WGSL, true);
        let module = dev.compile_shader(&patched, Some("HmmForwardF64"));
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HmmForwardF64"),
            layout: None,
            module: &module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: trans_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: emit_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pi_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: obs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: alpha_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: lik_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("HMM forward"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let n_workgroups = (n_seqs as u32).div_ceil(256);
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }

        dev.queue().submit(Some(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        let log_alpha = dev
            .read_buffer_f64(&alpha_buf, alpha_size)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))?;
        let log_likelihoods = dev
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
