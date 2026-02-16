// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU FP64 compute for wetSpring science workloads.
//!
//! Creates a wgpu device with `SHADER_F64` enabled, and provides helpers
//! for running f64 compute shaders. Bridges to `ToadStool`'s `WgpuDevice`
//! and `TensorContext` for access to `BarraCUDA` GPU primitives (reduce,
//! linalg, pairwise distance).
//!
//! Following hotSpring's validated pattern:
//! - Request `SHADER_F64` if available, report capability
//! - Bridge to `ToadStool` via `WgpuDevice::from_existing_simple`
//! - Raw dispatch helpers for custom WGSL shaders
//! - Readback via staging buffer + `map_async`

use barracuda::device::{TensorContext, WgpuDevice};
use std::sync::Arc;

/// GPU context with FP64 support for science workloads.
///
/// Wraps a wgpu device with `SHADER_F64` + `ToadStool`'s `TensorContext`
/// for batched dispatch and buffer pooling.
pub struct GpuF64 {
    /// Raw wgpu device handle.
    pub device: Arc<wgpu::Device>,
    /// Raw wgpu queue handle.
    pub queue: Arc<wgpu::Queue>,
    /// GPU adapter name (e.g., "NVIDIA RTX 4070").
    pub adapter_name: String,
    /// Whether the GPU supports f64 in shaders.
    pub has_f64: bool,
    wgpu_device: Arc<WgpuDevice>,
    tensor_ctx: Arc<TensorContext>,
}

impl GpuF64 {
    /// Create GPU device requesting `SHADER_F64`.
    ///
    /// Falls back gracefully if f64 is unavailable (`has_f64 = false`).
    /// Respects `WETSPRING_WGPU_BACKEND` env var (`vulkan`, `metal`, `dx12`).
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Gpu`] if no GPU adapter is found
    /// or device creation fails.
    pub async fn new() -> crate::error::Result<Self> {
        let backends = match std::env::var("WETSPRING_WGPU_BACKEND").as_deref() {
            Ok("vulkan") => wgpu::Backends::VULKAN,
            Ok("metal") => wgpu::Backends::METAL,
            Ok("dx12") => wgpu::Backends::DX12,
            _ => wgpu::Backends::all(),
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| crate::error::Error::Gpu("no GPU adapter found".into()))?;

        let info = adapter.get_info();
        let features = adapter.features();
        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let limits = adapter.limits();

        let mut required_features = wgpu::Features::empty();
        if has_f64 {
            required_features |= wgpu::Features::SHADER_F64;
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wetSpring FP64 Science Device"),
                    required_features,
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: limits
                            .max_storage_buffer_binding_size
                            .min(512 * 1024 * 1024),
                        max_buffer_size: limits.max_buffer_size.min(1024 * 1024 * 1024),
                        max_storage_buffers_per_shader_stage: limits
                            .max_storage_buffers_per_shader_stage
                            .min(16),
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .map_err(|e| crate::error::Error::Gpu(format!("device creation: {e}")))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let wgpu_device = Arc::new(WgpuDevice::from_existing_simple(
            device.clone(),
            queue.clone(),
        ));
        let tensor_ctx = Arc::new(TensorContext::new(wgpu_device.clone()));

        Ok(Self {
            device,
            queue,
            adapter_name: info.name.clone(),
            has_f64,
            wgpu_device,
            tensor_ctx,
        })
    }

    /// Bridge to `ToadStool`'s `WgpuDevice` for `SumReduceF64`, `BatchedEighGpu`, etc.
    #[must_use]
    pub fn to_wgpu_device(&self) -> Arc<WgpuDevice> {
        self.wgpu_device.clone()
    }

    /// Access the `TensorContext` for batched dispatch and buffer pooling.
    #[must_use]
    pub const fn tensor_context(&self) -> &Arc<TensorContext> {
        &self.tensor_ctx
    }

    /// Print GPU capabilities to stdout.
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
    }

    /// Create a compute pipeline from WGSL shader source.
    #[must_use]
    pub fn create_pipeline(&self, shader_source: &str, label: &str) -> wgpu::ComputePipeline {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: "main",
            })
    }

    /// Create a read-only storage buffer from f64 data.
    #[must_use]
    pub fn create_f64_buffer(&self, data: &[f64], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Create a writable storage buffer for f64 output.
    #[must_use]
    pub fn create_f64_output_buffer(&self, count: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (count * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer from a `bytemuck::Pod` value.
    #[must_use]
    pub fn create_uniform_buffer<T: bytemuck::Pod>(&self, data: &T, label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Dispatch a compute pipeline and read back f64 results.
    ///
    /// Encodes a single compute pass, copies output to a staging buffer,
    /// and maps it back to CPU memory.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Gpu`] if the buffer mapping callback
    /// channel is closed or the buffer mapping itself fails.
    pub fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
        output_buffer: &wgpu::Buffer,
        output_count: usize,
    ) -> crate::error::Result<Vec<f64>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (output_count * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(output_buffer, 0, &staging, 0, (output_count * 8) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| crate::error::Error::Gpu(format!("map callback channel: {e}")))?
            .map_err(|e| crate::error::Error::Gpu(format!("buffer mapping: {e}")))?;

        let data = slice.get_mapped_range();
        let result: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| {
                // chunks_exact(8) guarantees this slice is exactly 8 bytes,
                // so the conversion always succeeds.
                let mut arr = [0u8; 8];
                arr.copy_from_slice(chunk);
                f64::from_le_bytes(arr)
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }
}
