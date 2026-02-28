// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GemmCached` — pre-compiled GEMM pipeline + buffer pool.
//!
//! Uses `GemmF64` from `ToadStool`'s barracuda crate with DF64-aware
//! shader selection: on consumer GPUs (`Fp64Strategy::Hybrid`) the GEMM
//! inner loop runs on FP32 cores via double-float arithmetic, with f64
//! scalar output. On compute-class GPUs (`Fp64Strategy::Native`) the
//! native f64 GEMM shader is used.
//!
//! Hoists shader compilation to `new()` and reuses the pipeline across
//! dispatches. Data buffers via `ToadStool`'s `BufferPool`.

use crate::error::{Error, Result};
use barracuda::device::{
    BufferPool, PooledBuffer, TensorContext, WgpuDevice, storage_bgl_entry, uniform_bgl_entry,
};
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
    #[allow(clippy::cast_possible_truncation)]
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

/// Pre-compiled GEMM pipeline with `ToadStool` buffer pool integration.
///
/// Holds the shader module, bind group layout, and compute pipeline
/// for the lifetime of the session. Per-call resources (data buffers)
/// are managed through `ToadStool`'s `BufferPool` for cross-call reuse
/// via power-of-2 bucketing.
///
/// DF64 auto-detection (S62+): on consumer GPUs the shader runs the
/// matrix multiply on FP32 cores via double-float pairs, giving ~10x
/// throughput over the 1:64 f64 units.
pub struct GemmCached {
    device: Arc<WgpuDevice>,
    ctx: Arc<TensorContext>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GemmCached {
    /// Compile the GEMM shader at `Precision::F64` (default, correct for all GPUs).
    ///
    /// See [`with_precision`](Self::with_precision) to compile at a different
    /// precision level.
    pub fn new(device: Arc<WgpuDevice>, ctx: Arc<TensorContext>) -> Self {
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
    pub fn with_precision(
        device: Arc<WgpuDevice>,
        ctx: Arc<TensorContext>,
        precision: Precision,
    ) -> Self {
        let label = match precision {
            Precision::F64 => "GemmCached f64",
            Precision::Df64 => "GemmCached df64",
            _ => "GemmCached",
        };
        let shader = device.compile_shader_universal(GemmF64::WGSL, precision, Some(label));

        let bgl = device
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GemmCached BGL"),
                entries: &[
                    uniform_bgl_entry(0),
                    storage_bgl_entry(1, true),
                    storage_bgl_entry(2, true),
                    storage_bgl_entry(3, false),
                ],
            });

        let pl = device
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GemmCached PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GemmCached f64"),
                layout: Some(&pl),
                module: &shader,
                entry_point: "gemm_f64",
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Self {
            device,
            ctx,
            pipeline,
            bgl,
        }
    }

    /// Execute C = A × B using the cached pipeline and buffer pool.
    ///
    /// - Pipeline: reused from init (no shader recompilation)
    /// - Buffers A, B, C: acquired from `ToadStool` `BufferPool`, auto-returned on drop
    /// - Params: uniform buffer (too small to pool meaningfully)
    ///
    /// # Errors
    ///
    /// Returns an error if GPU dispatch or readback fails.
    #[allow(clippy::many_single_char_names)]
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

        let bg = self.create_bind_group(&params_buf, &a_buf, &b_buf, &c_buf);
        self.dispatch_wg(&bg, m, n, batch_size)?;

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
    #[allow(clippy::many_single_char_names)]
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

        let bg = self.create_bind_group(&params_buf, &a_buf, &b_buf, &c_buf);
        self.dispatch_wg(&bg, m, n, batch_size)?;

        Ok(c_buf.into_buffer())
    }

    #[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
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

    fn create_bind_group(
        &self,
        params: &wgpu::Buffer,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GemmCached BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: c.as_entire_binding(),
                    },
                ],
            })
    }

    fn dispatch_wg(&self, bg: &wgpu::BindGroup, m: usize, n: usize, batch_size: usize) -> Result<()> {
        let wg_x = dim_u32(n, "n")?.div_ceil(16);
        let wg_y = dim_u32(m, "m")?.div_ceil(16);
        let wg_z = dim_u32(batch_size, "batch_size")?;

        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GemmCached Encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GemmCached Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
        self.device.submit_and_poll(Some(encoder.finish()));
        Ok(())
    }
}
