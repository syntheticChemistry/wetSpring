// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU ODE parameter sweep for QS/c-di-GMP 5-variable system.
//!
//! Local workaround for `ToadStool`'s `BatchedOdeRK4F64`. As of session 39
//! (d45fdfb3), `ToadStool` removed `enable f64;` from the shader (line 35 is
//! now a comment), BUT `batched_ode_rk4.rs:209` calls `compile_shader()`
//! instead of `compile_shader_f64()`. Without f64 preamble injection the
//! shader won't compile on naga/Vulkan backends. This local copy uses
//! `ShaderTemplate::for_driver_auto` which injects the f64 preamble and
//! `pow_f64`/`exp_f64`/`log_f64` polyfills for Ada Lovelace GPUs.
//!
//! **Write → Absorb → Lean**: once `ToadStool` switches to `compile_shader_f64()`
//! in `batched_ode_rk4.rs`, this module becomes a thin wrapper around
//! `barracuda::ops::BatchedOdeRK4F64`, matching the pattern used by the
//! other 19 rewired GPU modules. Filed as handoff feedback item.

use barracuda::device::WgpuDevice;
use barracuda::shaders::precision::ShaderTemplate;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const ODE_WGSL: &str = include_str!("../shaders/batched_qs_ode_rk4_f64.wgsl");

/// Workgroup size — must match `@workgroup_size(N)` in `shaders/batched_qs_ode_rk4_f64.wgsl`.
const WORKGROUP_SIZE: u32 = 256;

/// Number of state variables in the QS biofilm ODE system.
pub const N_VARS: usize = 5;
/// Number of parameters per ODE batch element.
pub const N_PARAMS: usize = 17;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QsOdeConfigGpu {
    n_batches: u32,
    n_steps: u32,
    _pad0: u32,
    _pad1: u32,
    h: f64,
    t0: f64,
    clamp_max: f64,
    clamp_min: f64,
}

/// Configuration for ODE parameter sweep.
pub struct OdeSweepConfig {
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

/// GPU-backed ODE parameter sweep for the QS biofilm 5-variable system.
pub struct OdeSweepGpu {
    device: Arc<WgpuDevice>,
}

impl OdeSweepGpu {
    /// Create a new ODE sweep instance for the given device.
    #[must_use]
    pub const fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Run RK4 integration for all batch elements.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch, buffer mapping, or readback fails.
    ///
    /// # Panics
    ///
    /// Panics if `initial_states.len() != n_batches * N_VARS` or
    /// `batch_params.len() != n_batches * N_PARAMS`.
    pub fn integrate(
        &self,
        config: &OdeSweepConfig,
        initial_states: &[f64],
        batch_params: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        let b = config.n_batches as usize;
        assert_eq!(initial_states.len(), b * N_VARS);
        assert_eq!(batch_params.len(), b * N_PARAMS);

        let dev = &self.device;
        let d = dev.device();

        let cfg_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("OdeSweep Config"),
            contents: bytemuck::bytes_of(&QsOdeConfigGpu {
                n_batches: config.n_batches,
                n_steps: config.n_steps,
                _pad0: 0,
                _pad1: 0,
                h: config.h,
                t0: config.t0,
                clamp_max: config.clamp_max,
                clamp_min: config.clamp_min,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let init_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("OdeSweep InitStates"),
            contents: bytemuck::cast_slice(initial_states),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let param_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("OdeSweep Params"),
            contents: bytemuck::cast_slice(batch_params),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_size = (b * N_VARS * std::mem::size_of::<f64>()) as u64;
        let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OdeSweep Output"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("OdeSweep BGL"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("OdeSweep BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: init_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        // Force polyfill: RTX 4070 NVVM cannot compile f64 pow() natively
        let patched = ShaderTemplate::for_driver_auto(ODE_WGSL, true);
        let module = dev.compile_shader(&patched, Some("OdeSweepRK4"));

        let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("OdeSweep PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("OdeSweep Pipeline"),
            layout: Some(&pl),
            module: &module,
            entry_point: "main",
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("OdeSweep"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("OdeSweep Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((b as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        dev.queue().submit(Some(encoder.finish()));

        let staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OdeSweep Staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc2 = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy"),
        });
        enc2.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        dev.queue().submit(Some(enc2.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| crate::error::Error::Gpu(format!("map recv: {e}")))?
            .map_err(|e| crate::error::Error::Gpu(format!("map: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f64> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }
}

const fn bgl_entry(idx: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
