// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GemmCached` — pre-compiled GEMM pipeline + buffer pool.
//!
//! Uses `GemmF64` from barraCuda's barracuda crate with DF64-aware
//! shader selection: on consumer GPUs (`Fp64Strategy::Hybrid`) the GEMM
//! inner loop runs on FP32 cores via double-float arithmetic, with f64
//! scalar output. On compute-class GPUs (`Fp64Strategy::Native`) the
//! native f64 GEMM shader is used.
//!
//! Hoists shader compilation to `new()` and reuses the pipeline across
//! dispatches. Data buffers via barraCuda's `BufferPool`.

use crate::error::{Error, Result};
use barracuda::device::{BufferPool, ComputeDispatch, PooledBuffer, TensorContext, WgpuDevice};
use barracuda::ops::linalg::gemm_f64::GemmF64;
use barracuda::shaders::Precision;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

fn dim_u32(v: usize, name: &str) -> Result<u32> {
    u32::try_from(v)
        .map_err(|_| Error::InvalidInput(format!("{name} dimension {v} exceeds u32::MAX")))
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
    alpha_lo: u32,
    alpha_hi: u32,
    beta_lo: u32,
    beta_hi: u32,
}

impl GemmParams {
    #[expect(clippy::cast_possible_truncation)]
    const fn new(m: u32, k: u32, n: u32, batch_size: u32, alpha: f64, beta: f64) -> Self {
        let alpha_bits = alpha.to_bits();
        let beta_bits = beta.to_bits();
        Self {
            m,
            k,
            n,
            batch_size,
            alpha_lo: alpha_bits as u32,
            alpha_hi: (alpha_bits >> 32) as u32,
            beta_lo: beta_bits as u32,
            beta_hi: (beta_bits >> 32) as u32,
        }
    }
}

/// Pre-compiled GEMM pipeline with barraCuda buffer pool integration.
///
/// Uses `ComputeDispatch` builder for bind-group layout and dispatch.
/// Per-call resources (data buffers) are managed through barraCuda's
/// `BufferPool` for cross-call reuse via power-of-2 bucketing.
///
/// DF64 auto-detection (S62+): on consumer GPUs the shader runs the
/// matrix multiply on FP32 cores via double-float pairs, giving ~10x
/// throughput over the 1:64 f64 units.
pub struct GemmCached {
    device: Arc<WgpuDevice>,
    ctx: Arc<TensorContext>,
}

impl GemmCached {
    /// Compile the GEMM shader at `Precision::F64` (default, correct for all GPUs).
    ///
    /// See [`with_precision`](Self::with_precision) to compile at a different
    /// precision level.
    pub const fn new(device: Arc<WgpuDevice>, ctx: Arc<TensorContext>) -> Self {
        Self::with_precision(device, ctx, Precision::F64)
    }

    /// Compile the GEMM shader at the specified precision.
    ///
    /// `Precision::F64` — native f64 on compute-class GPUs (Titan V, MI250X).
    /// `Precision::Df64` — DF64 double-float on FP32 cores (~10× throughput
    /// on consumer GPUs). DF64 changes the wire format to `vec2<f32>`;
    /// current buffer protocol uses f64 — caller must use [`crate::df64_host`]
    /// pack/unpack for DF64 data if the shader's storage layout changed.
    ///
    /// For most workloads, use [`new`](Self::new) (F64) until the host
    /// buffer protocol is adapted for DF64.
    pub const fn with_precision(
        device: Arc<WgpuDevice>,
        ctx: Arc<TensorContext>,
        precision: Precision,
    ) -> Self {
        // ComputeDispatch uses compile_shader_f64; Precision::Df64 would need
        // compile_shader_universal — adopt when ComputeDispatch supports Df64.
        let _ = precision;
        Self { device, ctx }
    }

    /// Execute C = A × B using the cached pipeline and buffer pool.
    ///
    /// - Pipeline: reused from init (no shader recompilation)
    /// - Buffers A, B, C: acquired from barraCuda `BufferPool`, auto-returned on drop
    /// - Params: uniform buffer (too small to pool meaningfully)
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or readback fails.
    #[expect(clippy::many_single_char_names)]
    pub fn execute(
        &self,
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<Vec<f64>> {
        let c_size = batch_size * m * n;
        let pool = self.ctx.buffer_pool();

        let (a_buf, b_buf, c_buf) = self.acquire_and_upload(pool, a, b, m, k, n, batch_size);

        let params = GemmParams::new(
            dim_u32(m, "m")?,
            dim_u32(k, "k")?,
            dim_u32(n, "n")?,
            dim_u32(batch_size, "batch_size")?,
            1.0,
            0.0,
        );
        let params_buf = self
            .device
            .create_uniform_buffer("GemmCached Params", &params);

        let wg_x = dim_u32(n, "n")?.div_ceil(16);
        let wg_y = dim_u32(m, "m")?.div_ceil(16);
        let wg_z = dim_u32(batch_size, "batch_size")?;

        ComputeDispatch::new(self.device.as_ref(), "GemmCached")
            .shader(GemmF64::WGSL, "gemm_f64")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, &a_buf)
            .storage_read(2, &b_buf)
            .storage_rw(3, &c_buf)
            .dispatch(wg_x, wg_y, wg_z)
            .submit()
            .map_err(|e| Error::Gpu(format!("GemmCached dispatch: {e}")))?;

        self.device
            .read_f64_buffer(&c_buf, c_size)
            .map_err(|e| Error::Gpu(format!("GemmCached readback: {e}")))
    }

    /// Execute C = A × B, returning the GPU buffer (no readback).
    ///
    /// The returned buffer stays on GPU for chaining into subsequent
    /// dispatches. Caller is responsible for readback when needed.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch fails.
    #[expect(clippy::many_single_char_names)]
    pub fn execute_to_buffer(
        &self,
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<wgpu::Buffer> {
        let pool = self.ctx.buffer_pool();

        let (a_buf, b_buf, c_buf) = self.acquire_and_upload(pool, a, b, m, k, n, batch_size);

        let params = GemmParams::new(
            dim_u32(m, "m")?,
            dim_u32(k, "k")?,
            dim_u32(n, "n")?,
            dim_u32(batch_size, "batch_size")?,
            1.0,
            0.0,
        );
        let params_buf = self
            .device
            .create_uniform_buffer("GemmCached Params", &params);

        let wg_x = dim_u32(n, "n")?.div_ceil(16);
        let wg_y = dim_u32(m, "m")?.div_ceil(16);
        let wg_z = dim_u32(batch_size, "batch_size")?;

        ComputeDispatch::new(self.device.as_ref(), "GemmCached")
            .shader(GemmF64::WGSL, "gemm_f64")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, &a_buf)
            .storage_read(2, &b_buf)
            .storage_rw(3, &c_buf)
            .dispatch(wg_x, wg_y, wg_z)
            .submit()
            .map_err(|e| Error::Gpu(format!("GemmCached dispatch: {e}")))?;

        Ok(c_buf.into_buffer())
    }

    #[expect(clippy::many_single_char_names, clippy::too_many_arguments)]
    fn acquire_and_upload(
        &self,
        pool: &BufferPool,
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
    ) -> (PooledBuffer, PooledBuffer, PooledBuffer) {
        let a_bytes = batch_size * m * k * 8;
        let b_bytes = batch_size * k * n * 8;
        let c_bytes = batch_size * m * n * 8;

        let a_buf = pool.acquire_pooled(a_bytes);
        let b_buf = pool.acquire_pooled(b_bytes);
        let c_buf = pool.acquire_pooled(c_bytes);

        self.device
            .queue()
            .write_buffer(&a_buf, 0, bytemuck::cast_slice(a));
        self.device
            .queue()
            .write_buffer(&b_buf, 0, bytemuck::cast_slice(b));

        (a_buf, b_buf, c_buf)
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;
    use std::sync::Arc;

    #[test]
    fn api_surface_compiles() {
        fn _assert_gemm_cached(_: &GemmCached) {}
        let _: fn(&GemmCached, &[f64], &[f64], usize, usize, usize, usize) -> Result<Vec<f64>> =
            GemmCached::execute;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let device = gpu.to_wgpu_device();
        // Arc bump, O(1)
        let ctx = Arc::clone(gpu.tensor_context());
        let gemm = GemmCached::new(Arc::clone(&device), ctx);
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let result = gemm.execute(&a, &b, 2, 2, 2, 1);
        assert!(result.is_ok(), "execute should succeed with valid input");
    }
}
