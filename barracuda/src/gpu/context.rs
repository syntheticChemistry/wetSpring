// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared GPU context for [`barracuda::session::TensorSession`] and barraCuda primitives.
//!
//! Bridges wetSpring's [`super::GpuF64`] (wgpu limits, `TensorContext`, capabilities) with
//! neuralSpring-style ML session batching: one [`Arc`] of [`barracuda::device::WgpuDevice`],
//! many [`TensorSession`] instances.

use std::sync::Arc;

use barracuda::device::WgpuDevice;
use barracuda::session::TensorSession;

use super::GpuF64;

/// Shared GPU device handle for tensor sessions and barraCuda ops.
///
/// Prefer constructing from [`GpuF64`](GpuF64) via [`Self::from_gpu_f64`] so adapter name and
/// `SHADER_F64` availability match the rest of wetSpring.
#[derive(Clone)]
pub struct GpuContext {
    device: Arc<WgpuDevice>,
    device_name: String,
    has_f64: bool,
}

impl GpuContext {
    /// Build from the canonical wetSpring GPU wrapper (same `WgpuDevice` as FP64 science paths).
    #[must_use]
    pub fn from_gpu_f64(gpu: &GpuF64) -> Self {
        Self {
            device: gpu.to_wgpu_device(),
            device_name: gpu.adapter_name.clone(),
            has_f64: gpu.has_f64,
        }
    }

    /// Wrap an existing device (e.g. IPC handoff) with explicit metadata.
    #[must_use]
    pub const fn from_device(device: Arc<WgpuDevice>, device_name: String, has_f64: bool) -> Self {
        Self {
            device,
            device_name,
            has_f64,
        }
    }

    /// Underlying barraCuda device (shared).
    #[must_use]
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Owned clone of the device `Arc` for APIs that need it.
    #[must_use]
    pub fn device_arc(&self) -> Arc<WgpuDevice> {
        Arc::clone(&self.device)
    }

    /// Adapter-reported name (e.g. `"NVIDIA GeForce RTX 4070"`).
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Whether `SHADER_F64` was available at device creation.
    #[must_use]
    pub const fn has_f64(&self) -> bool {
        self.has_f64
    }

    /// Fresh [`TensorSession`] sharing this device (pipelines compiled once per session).
    #[must_use]
    pub fn tensor_session(&self) -> TensorSession {
        TensorSession::with_device(Arc::clone(&self.device))
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device_name", &self.device_name)
            .field("has_f64", &self.has_f64)
            .field("device", &"<WgpuDevice>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn from_gpu_f64_matches_name_and_f64_flag() {
        let Ok(gpu) = GpuF64::new().await else {
            return;
        };
        let ctx = GpuContext::from_gpu_f64(&gpu);
        assert_eq!(ctx.device_name(), gpu.adapter_name.as_str());
        assert_eq!(ctx.has_f64(), gpu.has_f64);
    }

    #[tokio::test]
    async fn tensor_session_from_context_records_zero_ops_initially() {
        let Ok(gpu) = GpuF64::new().await else {
            return;
        };
        let ctx = GpuContext::from_gpu_f64(&gpu);
        let session = ctx.tensor_session();
        assert_eq!(session.num_ops(), 0);
    }

    #[tokio::test]
    async fn device_arc_matches_gpu_f64_wgpu_device() {
        let Ok(gpu) = GpuF64::new().await else {
            return;
        };
        let ctx = GpuContext::from_gpu_f64(&gpu);
        assert!(Arc::ptr_eq(&ctx.device_arc(), &gpu.to_wgpu_device()));
    }
}
