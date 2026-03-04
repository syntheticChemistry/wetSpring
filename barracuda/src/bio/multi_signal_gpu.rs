// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated dual-signal QS ODE parameter sweep.
//!
//! **Lean phase complete**: Uses barraCuda's `BatchedOdeRK4<MultiSignalOde>::generate_shader()`
//! via the `OdeSystem` trait (see `bio::ode_systems::MultiSignalOde`).
//! Local WGSL file deleted — shader now generated from trait impl at runtime.
//!
//! **Precision (S68+)**: Compiled via `compile_shader_universal` at `Precision::F64`
//! (f64 canonical). `Precision::Df64` available for ~10× throughput on consumer
//! FP32 cores (requires host buffer protocol adaptation: `vec2<f32>` storage).

use barracuda::device::{ComputeDispatch, WgpuDevice};
use barracuda::numerical::MultiSignalOde;
use barracuda::numerical::ode_generic::BatchedOdeRK4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::multi_signal::{self, MultiSignalParams};

/// Number of state variables.
pub const N_VARS: usize = multi_signal::N_VARS;
/// Number of f64 parameters per batch.
pub const N_PARAMS: usize = multi_signal::N_PARAMS;

/// Config for batched multi-signal ODE sweep.
pub struct MultiSignalOdeConfig {
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

/// GPU-backed batched dual-signal QS ODE integrator.
pub struct MultiSignalGpu {
    device: Arc<WgpuDevice>,
}

impl MultiSignalGpu {
    /// Compile the local WGSL shader and create the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns `Err` if shader compilation fails.
    pub const fn new(device: Arc<WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    /// Integrate all batches and return final states `[n_batches x N_VARS]`.
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
        config: &MultiSignalOdeConfig,
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
            label: Some("MultiSignal config"),
            contents: bytemuck::bytes_of(&gpu_cfg),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let states_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MultiSignal initial_states"),
            contents: bytemuck::cast_slice(initial_states),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MultiSignal params"),
            contents: bytemuck::cast_slice(batch_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MultiSignal output"),
            contents: bytemuck::cast_slice(&vec![0.0f64; b * N_VARS]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let wgsl = BatchedOdeRK4::<MultiSignalOde>::generate_shader();
        ComputeDispatch::new(self.device.as_ref(), "MultiSignal ODE")
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

    /// Convenience: integrate from `MultiSignalParams` structs.
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
        params_list: &[MultiSignalParams],
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
            .flat_map(MultiSignalParams::to_flat)
            .collect();

        let config = MultiSignalOdeConfig {
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
            .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6]])
            .collect())
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::manual_let_else
)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        fn _assert_config(_: &MultiSignalOdeConfig) {}
        let _: fn(
            &MultiSignalGpu,
            &MultiSignalOdeConfig,
            &[f64],
            &[f64],
        ) -> crate::error::Result<Vec<f64>> = MultiSignalGpu::integrate;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let device = gpu.to_wgpu_device();
        let multi_signal = MultiSignalGpu::new(device).expect("MultiSignalGpu::new");
        let config = MultiSignalOdeConfig {
            n_batches: 1,
            n_steps: 10,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        let initial_states = [0.5; N_VARS];
        let batch_params = super::super::multi_signal::MultiSignalParams::default().to_flat();
        let result = multi_signal.integrate(&config, &initial_states, &batch_params);
        assert!(result.is_ok(), "integrate should succeed with valid input");
    }
}
