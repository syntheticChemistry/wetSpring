// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU ODE parameter sweep for QS/c-di-GMP 5-variable system.
//!
//! **Lean phase**: thin wrapper around `barracuda::ops::BatchedOdeRK4F64`.
//! The local WGSL workaround was retired once `ToadStool` S41 switched to
//! `compile_shader_f64()` in `batched_ode_rk4.rs`. This module preserves
//! the wetSpring-facing API (`OdeSweepConfig`, `OdeSweepGpu`) while
//! delegating all GPU work to upstream.

use barracuda::device::WgpuDevice;
use barracuda::ops::{BatchedOdeRK4F64, BatchedRk4Config};
use std::sync::Arc;

/// Number of state variables in the QS biofilm ODE system.
pub const N_VARS: usize = 5;
/// Number of parameters per ODE batch element.
pub const N_PARAMS: usize = 17;

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
///
/// Delegates to `barracuda::ops::BatchedOdeRK4F64` (`ToadStool` upstream).
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

        let upstream_config = BatchedRk4Config {
            n_batches: config.n_batches,
            n_steps: config.n_steps,
            h: config.h,
            t0: config.t0,
            clamp_max: config.clamp_max,
            clamp_min: config.clamp_min,
        };

        let integrator = BatchedOdeRK4F64::new(self.device.clone(), upstream_config);
        integrator
            .integrate(initial_states, batch_params)
            .map_err(|e| crate::error::Error::Gpu(format!("{e}")))
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
        fn _assert_config(_: &OdeSweepConfig) {}
        let _: fn(&OdeSweepGpu, &OdeSweepConfig, &[f64], &[f64]) -> crate::error::Result<Vec<f64>> =
            OdeSweepGpu::integrate;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let device = gpu.to_wgpu_device();
        let ode = OdeSweepGpu::new(device);
        let config = OdeSweepConfig {
            n_batches: 1,
            n_steps: 10,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        };
        let initial_states = vec![0.5; N_VARS];
        let batch_params = vec![0.0; N_PARAMS];
        let result = ode.integrate(&config, &initial_states, &batch_params);
        assert!(result.is_ok(), "integrate should succeed with valid input");
    }
}
