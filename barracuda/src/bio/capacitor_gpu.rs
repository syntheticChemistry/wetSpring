// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated phenotypic capacitor ODE parameter sweep.
//!
//! **Lean phase complete**: Uses `ToadStool`'s `BatchedOdeRK4<CapacitorOde>::generate_shader()`
//! via the `OdeSystem` trait (see `bio::ode_systems::CapacitorOde`).
//! Local WGSL file deleted — shader now generated from trait impl at runtime.
//!
//! **Precision (S68+)**: Compiled via `compile_shader_universal` at `Precision::F64`
//! (f64 canonical). `Precision::Df64` available for ~10× throughput on consumer
//! FP32 cores (requires host buffer protocol adaptation: `vec2<f32>` storage).

use barracuda::device::{ComputeDispatch, WgpuDevice};
use barracuda::numerical::CapacitorOde;
use barracuda::numerical::ode_generic::BatchedOdeRK4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::capacitor::{self, CapacitorParams};

/// Number of state variables.
pub const N_VARS: usize = capacitor::N_VARS;
/// Number of f64 parameters per batch.
pub const N_PARAMS: usize = capacitor::N_PARAMS;

/// Config for batched capacitor ODE sweep.
pub struct CapacitorOdeConfig {
    /// Number of batch elements to integrate in parallel.
    pub n_batches: u32,
    /// Number of RK4 steps per batch element.
    pub n_steps: u32,
    /// Step size for integration.
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

/// GPU-backed batched capacitor ODE integrator.
pub struct CapacitorGpu {
    device: Arc<WgpuDevice>,
}

impl CapacitorGpu {
    /// Compile the local WGSL shader and create the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns `Err` if shader compilation fails.
    pub const fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
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
        config: &CapacitorOdeConfig,
        initial_states: &[f64],
        batch_params: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        let b = config.n_batches as usize;
        assert_eq!(initial_states.len(), b * N_VARS);
        assert_eq!(batch_params.len(), b * N_PARAMS);

        let d = self.device.device();

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
            label: Some("Capacitor config"),
            contents: bytemuck::bytes_of(&gpu_cfg),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let states_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Capacitor initial_states"),
            contents: bytemuck::cast_slice(initial_states),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Capacitor params"),
            contents: bytemuck::cast_slice(batch_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Capacitor output"),
            contents: bytemuck::cast_slice(&vec![0.0f64; b * N_VARS]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let wgsl = BatchedOdeRK4::<CapacitorOde>::generate_shader();
        ComputeDispatch::new(self.device.as_ref(), "Capacitor ODE")
            .shader(&wgsl, "main")
            .f64()
            .uniform(0, &config_buf)
            .storage_read(1, &states_buf)
            .storage_read(2, &params_buf)
            .storage_rw(3, &out_buf)
            .dispatch(config.n_batches.div_ceil(64), 1, 1)
            .submit();

        self.device
            .read_buffer_f64(&out_buf, b * N_VARS)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))
    }

    /// Convenience: integrate from `CapacitorParams` structs.
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
        params_list: &[CapacitorParams],
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
            .flat_map(CapacitorParams::to_flat)
            .collect();

        let config = CapacitorOdeConfig {
            n_batches: n as u32,
            n_steps,
            h,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };

        let raw = self.integrate(&config, &flat_y0, &flat_params)?;
        Ok(raw
            .chunks_exact(N_VARS)
            .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5]])
            .collect())
    }
}
