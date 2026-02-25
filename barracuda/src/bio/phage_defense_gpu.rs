// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated phage defense ODE parameter sweep.
//!
//! **Lean phase complete**: Uses `ToadStool`'s `BatchedOdeRK4<PhageDefenseOde>::generate_shader()`
//! via the `OdeSystem` trait (see `bio::ode_systems::PhageDefenseOde`).
//! Local WGSL file deleted — shader now generated from trait impl at runtime.

use barracuda::device::{WgpuDevice, storage_bgl_entry, uniform_bgl_entry};
use barracuda::numerical::ode_generic::BatchedOdeRK4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::phage_defense::{self, PhageDefenseParams};
use barracuda::numerical::PhageDefenseOde;

/// Number of state variables.
pub const N_VARS: usize = phage_defense::N_VARS;
/// Number of f64 parameters per batch.
pub const N_PARAMS: usize = phage_defense::N_PARAMS;

/// Config for batched phage defense ODE sweep.
pub struct PhageDefenseOdeConfig {
    /// Number of batch elements to integrate in parallel.
    pub n_batches: u32,
    /// Number of RK4 steps per batch element.
    pub n_steps: u32,
    /// Step size (time step) for integration.
    pub h: f64,
    /// Initial time value.
    pub t0: f64,
    /// Maximum clamp value for state variables.
    pub clamp_max: f64,
    /// Minimum clamp value for state variables.
    pub clamp_min: f64,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuConfig {
    n_batches: u32,
    n_steps: u32,
    _pad0: u32,
    _pad1: u32,
    h: f64,
    t0: f64,
    clamp_max: f64,
    clamp_min: f64,
}

/// GPU-backed batched phage defense ODE integrator.
pub struct PhageDefenseGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl PhageDefenseGpu {
    /// Compile the local WGSL shader and create the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns `Err` if shader compilation fails.
    pub fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        let d = device.device();
        let wgsl = BatchedOdeRK4::<PhageDefenseOde>::generate_shader();
        let module = device.compile_shader_f64(&wgsl, Some("PhageDefense ODE"));

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PhageDefense BGL"),
            entries: &Self::bgl_entries(),
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PhageDefense Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PhageDefense Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bgl,
            device,
        })
    }

    fn bgl_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
        [
            uniform_bgl_entry(0),
            storage_bgl_entry(1, true),
            storage_bgl_entry(2, true),
            storage_bgl_entry(3, false),
        ]
    }

    /// Integrate all batches and return final states `[n_batches × N_VARS]`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or readback fails.
    ///
    /// # Panics
    ///
    /// Panics if buffer sizes don't match config.
    pub fn integrate(
        &self,
        config: &PhageDefenseOdeConfig,
        initial_states: &[f64],
        batch_params: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        let b = config.n_batches as usize;
        assert_eq!(initial_states.len(), b * N_VARS);
        assert_eq!(batch_params.len(), b * N_PARAMS);

        let d = self.device.device();
        let q = self.device.queue();

        let gpu_cfg = GpuConfig {
            n_batches: config.n_batches,
            n_steps: config.n_steps,
            _pad0: 0,
            _pad1: 0,
            h: config.h,
            t0: config.t0,
            clamp_max: config.clamp_max,
            clamp_min: config.clamp_min,
        };

        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PhageDefense config"),
            contents: bytemuck::bytes_of(&gpu_cfg),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let states_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PhageDefense initial_states"),
            contents: bytemuck::cast_slice(initial_states),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PhageDefense params"),
            contents: bytemuck::cast_slice(batch_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PhageDefense output"),
            contents: bytemuck::cast_slice(&vec![0.0f64; b * N_VARS]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PhageDefense BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: states_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PhageDefense Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PhageDefense Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(config.n_batches.div_ceil(64), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
        d.poll(wgpu::Maintain::Wait);

        self.device
            .read_buffer_f64(&out_buf, b * N_VARS)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))
    }

    /// Convenience: integrate from `PhageDefenseParams` structs.
    ///
    /// # Errors
    ///
    /// Returns `Err` if GPU dispatch or readback fails.
    ///
    /// # Panics
    ///
    /// Panics if `params_list.len() != initial_states.len()`.
    pub fn integrate_params(
        &self,
        params_list: &[PhageDefenseParams],
        initial_states: &[[f64; N_VARS]],
        n_steps: u32,
        h: f64,
    ) -> crate::error::Result<Vec<[f64; N_VARS]>> {
        assert_eq!(params_list.len(), initial_states.len());
        let n = params_list.len();

        let flat_y0: Vec<f64> = initial_states
            .iter()
            .flat_map(|y| y.iter().copied())
            .collect();
        let flat_params: Vec<f64> = params_list
            .iter()
            .flat_map(PhageDefenseParams::to_flat)
            .collect();

        let config = PhageDefenseOdeConfig {
            n_batches: n as u32,
            n_steps,
            h,
            t0: 0.0,
            clamp_max: 1e12,
            clamp_min: 0.0,
        };

        let raw = self.integrate(&config, &flat_y0, &flat_params)?;
        Ok(raw
            .chunks_exact(N_VARS)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect())
    }
}
