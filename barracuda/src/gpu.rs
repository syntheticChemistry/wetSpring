// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU FP64 compute for wetSpring science workloads.
//!
//! Creates a wgpu device with `SHADER_F64` and bridges to `ToadStool`'s
//! `WgpuDevice` + `TensorContext`. GPU dispatch goes through `ToadStool`
//! primitives — wetSpring has 4 local WGSL shaders (ODE, kmer, unifrac, taxonomy; ODE blocked on
//! `enable f64;` in `ToadStool`'s upstream).
//!
//! `ToadStool` primitives used:
//! - `FusedMapReduceF64` — Shannon, Simpson, alpha diversity
//! - `BrayCurtisF64` — condensed distance matrices
//! - `BatchedEighGpu` — `PCoA` eigendecomposition
//! - `GemmF64` / `GemmCachedF64` — batch spectral cosine, taxonomy
//! - `KrigingF64` — spatial interpolation
//! - `VarianceF64` — population/sample variance, std dev
//! - `CorrelationF64` — Pearson correlation
//! - `CovarianceF64` — sample covariance
//! - `WeightedDotF64` — weighted/plain dot product
//! - `FelsensteinGpu` — phylogenetic likelihood (site-parallel pruning)
//! - `GillespieGpu` — parallel stochastic simulation (N trajectories)
//! - `SmithWatermanGpu` — banded local alignment (wavefront)
//! - `TreeInferenceGpu` — decision tree / random forest inference

use barracuda::device::{GpuDriverProfile, TensorContext, WgpuDevice};
use std::sync::Arc;

/// GPU context with FP64 support for science workloads.
///
/// Wraps a wgpu device with `SHADER_F64` + `ToadStool`'s `TensorContext`
/// for batched dispatch and buffer pooling. All compute goes through
/// `ToadStool` primitives via [`to_wgpu_device`](Self::to_wgpu_device).
///
/// `ToadStool` primitives used (15 total):
/// - `FusedMapReduceF64` — Shannon, Simpson, alpha diversity
/// - `BrayCurtisF64` — condensed distance matrices
/// - `BatchedEighGpu` — `PCoA` eigendecomposition
/// - `GemmF64` / `GemmCachedF64` — GEMM, spectral cosine, taxonomy
/// - `KrigingF64` — spatial interpolation
/// - `VarianceF64` / `CorrelationF64` / `CovarianceF64` / `WeightedDotF64`
/// - `FelsensteinGpu` — phylogenetic pruning (site × node parallel)
/// - `GillespieGpu` — parallel SSA (N independent trajectories)
/// - `SmithWatermanGpu` — banded SW alignment (anti-diagonal wavefront)
/// - `TreeInferenceGpu` — decision tree / RF inference (sample × tree)
///
/// Driver-specific capabilities (NVK workarounds, eigensolve strategy,
/// latency model) are available via [`driver_profile`](Self::driver_profile).
pub struct GpuF64 {
    /// GPU adapter name (e.g., "NVIDIA RTX 4070").
    pub adapter_name: String,
    /// Whether the GPU supports f64 in shaders.
    pub has_f64: bool,
    wgpu_device: Arc<WgpuDevice>,
    tensor_ctx: Arc<TensorContext>,
    driver_profile: GpuDriverProfile,
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
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| crate::error::Error::Gpu(format!("device creation: {e}")))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let adapter_name = info.name.clone();

        let wgpu_device = Arc::new(WgpuDevice::from_existing(device, queue, info));
        let tensor_ctx = Arc::new(TensorContext::new(wgpu_device.clone()));
        let driver_profile = GpuDriverProfile::from_device(&wgpu_device);

        Ok(Self {
            adapter_name,
            has_f64,
            wgpu_device,
            tensor_ctx,
            driver_profile,
        })
    }

    /// Bridge to `ToadStool`'s `WgpuDevice` for all GPU primitives.
    #[must_use]
    pub fn to_wgpu_device(&self) -> Arc<WgpuDevice> {
        self.wgpu_device.clone()
    }

    /// Access the `TensorContext` for batched dispatch and buffer pooling.
    #[must_use]
    pub const fn tensor_context(&self) -> &Arc<TensorContext> {
        &self.tensor_ctx
    }

    /// `ToadStool` driver profile for this GPU.
    ///
    /// Provides driver-specific capabilities: f64 workarounds, optimal
    /// eigensolve strategy, latency model for `WgslOptimizer`.
    #[must_use]
    pub const fn driver_profile(&self) -> &GpuDriverProfile {
        &self.driver_profile
    }

    /// Whether this GPU's driver needs f64 `exp`/`log` polyfills.
    #[must_use]
    pub fn needs_f64_workaround(&self) -> bool {
        self.driver_profile.needs_exp_f64_workaround()
            || self.driver_profile.needs_log_f64_workaround()
    }

    /// Print GPU capabilities to stdout.
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
        println!("  Arch: {:?}", self.driver_profile.arch);
        println!("  Driver: {:?}", self.driver_profile.driver);
        println!(
            "  f64 workarounds: exp={}, log={}",
            self.driver_profile.needs_exp_f64_workaround(),
            self.driver_profile.needs_log_f64_workaround()
        );
    }
}

/// Default minimum element count for GPU dispatch to outperform CPU.
///
/// Below this threshold, GPU kernel launch overhead dominates.
/// Measured on RTX 4070 for `FusedMapReduceF64` (Shannon/Simpson).
/// Callers should fall back to the CPU path when `n < GPU_DISPATCH_THRESHOLD`.
///
/// Use [`GpuF64::dispatch_threshold`] for a capability-aware value that
/// adapts to the detected GPU's latency profile.
pub const GPU_DISPATCH_THRESHOLD: usize = 10_000;

impl GpuF64 {
    /// Capability-aware dispatch threshold for this GPU.
    ///
    /// Uses the driver profile's instruction-level latency to estimate
    /// the crossover point where GPU dispatch outperforms CPU. Higher
    /// instruction latency (software f64 emulation) means a higher
    /// element count is needed to amortize launch overhead.
    ///
    /// Falls back to [`GPU_DISPATCH_THRESHOLD`] for unknown hardware.
    #[must_use]
    pub fn dispatch_threshold(&self) -> usize {
        use barracuda::device::latency::WgslOpClass;

        let model = self.driver_profile.latency_model();
        let dfma_cycles = model.raw_latency(WgslOpClass::F64Fma);

        match dfma_cycles {
            0..=4 => 5_000,   // Fast f64 (AMD RDNA2+): lower threshold
            5..=8 => 10_000,  // Native f64 (NVIDIA Volta+): default
            9..=16 => 25_000, // Software f64 (Apple M, Intel): higher threshold
            _ => 50_000,      // Unknown / very slow: conservative
        }
    }
}
