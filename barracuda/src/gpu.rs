// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU FP64 compute for wetSpring science workloads.
//!
//! Creates a wgpu device with `SHADER_F64` and bridges to `barraCuda`'s
//! `WgpuDevice` + `TensorContext`. Core dispatch goes through `barraCuda`
//! primitives (144+ consumed, zero fallback code). ODE domains use
//! runtime-generated WGSL via `BatchedOdeRK4::generate_shader()`.
//!
//! # Precision (barraCuda universal precision)
//!
//! All 694+ `barraCuda` WGSL shaders are f64-canonical (zero f32-only remain).
//! `compile_shader_universal(source, precision)` routes through:
//!
//! - `Precision::F64` — native f64 (compute-class GPUs: Titan V, V100, MI250X)
//! - `Precision::Df64` — double-float f32-pair (~48-bit mantissa, ~10x throughput
//!   on consumer FP32 cores: RTX 4070, RDNA2+)
//! - `Precision::F32` — downcast from f64 canonical (inference, bandwidth-bound)
//! - `Precision::F16` — half-precision (inference only)
//!
//! Selection: [`GpuF64::optimal_precision`] returns F64 or Df64 based on
//! [`GpuF64::fp64_strategy`] (Native vs Hybrid). ODE modules currently use
//! F64 directly; DF64 promotion requires host buffer protocol adaptation.
//!
//! # Consumed `barraCuda` GPU primitives
//!
//! - `FusedMapReduceF64` — Shannon, Simpson, alpha diversity
//! - `BrayCurtisF64` — condensed distance matrices
//! - `BatchedEighGpu` — `PCoA` eigendecomposition
//! - `BatchedOdeRK4F64` — QS/c-di-GMP ODE integration (5 ODE systems)
//! - `GemmF64` / `GemmCachedF64` — batch spectral cosine, taxonomy
//! - `KrigingF64` — spatial interpolation
//! - `KmerHistogramGpu` — k-mer counting (atomic histogram)
//! - `UniFracPropagateGpu` — phylogenetic tree propagation
//! - `VarianceF64` / `CorrelationF64` / `CovarianceF64` / `WeightedDotF64`
//! - `FelsensteinGpu` — phylogenetic pruning (site x node parallel)
//! - `GillespieGpu` — parallel SSA (N independent trajectories)
//! - `SmithWatermanGpu` — banded SW alignment (anti-diagonal wavefront)
//! - `TreeInferenceGpu` — decision tree / RF inference (sample x tree)
//! - `DiversityFusionGpu` — Shannon + Simpson + evenness fused
//! - `MultiHeadEsn` — shared reservoir, per-head readout (`head_disagreement`)
//! - Plus 20+ bio ops: ANI, SNP, dN/dS, pangenome, HMM, DADA2, etc.
//!
//! # WGSL Generation (Lean — zero local shaders)
//!
//! All ODE shaders are generated at runtime via `BatchedOdeRK4::generate_shader()`
//! from `barraCuda`. wetSpring holds zero local `.wgsl` files.

use barracuda::device::{GpuDriverProfile, TensorContext, WgpuDevice};
use std::sync::Arc;

const DEVICE_LABEL: &str = "FP64 Science Device";

/// wgpu `maxStorageBufferBindingSize` override (512 MiB).
///
/// Default wgpu limit is 128 MiB; increased for large biology matrices
/// (e.g. 4000 × 18000 drug-disease NMF, pairwise distance N=10000).
/// Upper bound from Vulkan spec `maxStorageBufferRange` (typically 2 GiB
/// on discrete GPUs). Validated on RTX 4070 / Titan V / AMD RDNA2.
const MAX_STORAGE_BINDING_BYTES: u32 = 512 * 1024 * 1024;

/// wgpu `maxBufferSize` override (1 GiB).
///
/// Default wgpu limit is 256 MiB; increased for large genomics datasets.
/// Must be ≤ device `maxBufferSize` (typically 2-4 GiB on modern GPUs).
/// Validated on RTX 4070 (12 GiB VRAM) and Titan V (12 GiB HBM2).
const MAX_BUFFER_SIZE_BYTES: u64 = 1024 * 1024 * 1024;

/// wgpu `maxStorageBuffersPerShaderStage` override.
///
/// Default wgpu limit is 8; many bio kernels need input + output + scratch
/// buffers (e.g. NMF: V, W, H, WH, scratch = 5). 16 is conservative vs
/// Vulkan minimum guarantee of 4 but matches D3D12/Metal typical limits.
const MAX_STORAGE_BUFFERS_PER_STAGE: u32 = 16;

/// GPU context with FP64 support for science workloads.
///
/// Wraps a wgpu device with `SHADER_F64` + `barraCuda`'s `TensorContext`
/// for batched dispatch and buffer pooling.
///
/// Most domains dispatch through `barraCuda` primitives via
/// [`to_wgpu_device`](Self::to_wgpu_device). ODE domains use runtime-generated
/// WGSL via `BatchedOdeRK4::generate_shader()` (zero local shaders).
///
/// Driver-specific capabilities (NVK workarounds, eigensolve strategy,
/// latency model) are available via [`driver_profile`](Self::driver_profile).
pub struct GpuF64 {
    /// GPU adapter name (e.g., `"NVIDIA GeForce RTX 4070"`).
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

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
            .map_err(|e| crate::error::Error::Gpu(format!("no GPU adapter found: {e}")))?;

        let info = adapter.get_info();
        let features = adapter.features();
        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let limits = adapter.limits();

        let mut required_features = wgpu::Features::empty();
        if has_f64 {
            required_features |= wgpu::Features::SHADER_F64;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(DEVICE_LABEL),
                required_features,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: limits
                        .max_storage_buffer_binding_size
                        .min(MAX_STORAGE_BINDING_BYTES),
                    max_buffer_size: limits.max_buffer_size.min(MAX_BUFFER_SIZE_BYTES),
                    max_storage_buffers_per_shader_stage: limits
                        .max_storage_buffers_per_shader_stage
                        .min(MAX_STORAGE_BUFFERS_PER_STAGE),
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::default(),
            })
            .await
            .map_err(|e| crate::error::Error::Gpu(format!("device creation: {e}")))?;

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

    /// Bridge to `barraCuda`'s `WgpuDevice` for all GPU primitives.
    #[must_use]
    pub fn to_wgpu_device(&self) -> Arc<WgpuDevice> {
        self.wgpu_device.clone()
    }

    /// Access the `TensorContext` for batched dispatch and buffer pooling.
    #[must_use]
    pub const fn tensor_context(&self) -> &Arc<TensorContext> {
        &self.tensor_ctx
    }

    /// `barraCuda` driver profile for this GPU.
    ///
    /// Provides driver-specific capabilities: f64 workarounds, optimal
    /// eigensolve strategy, latency model for `WgslOptimizer`.
    #[must_use]
    pub const fn driver_profile(&self) -> &GpuDriverProfile {
        &self.driver_profile
    }

    /// Whether the underlying GPU device has been lost.
    ///
    /// Delegates to `WgpuDevice::is_lost()`. When `true`,
    /// callers should fall back to the CPU path — the GPU will not accept
    /// new work until the device is recreated.
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.wgpu_device.is_lost()
    }

    /// Whether this GPU's driver needs f64 `exp`/`log` polyfills.
    #[must_use]
    pub fn needs_f64_workaround(&self) -> bool {
        self.driver_profile.needs_exp_f64_workaround()
            || self.driver_profile.needs_log_f64_workaround()
    }

    /// Runtime precision strategy for this GPU (`barraCuda` universal precision).
    ///
    /// `Native` = compute-class GPU with full-rate f64 (Titan V, V100, MI250X).
    /// `Hybrid` = consumer GPU with throttled f64 — bulk math routes through
    /// DF64 (f32-pair, ~14 digits) on FP32 cores, native f64 for reductions.
    #[must_use]
    pub fn fp64_strategy(&self) -> barracuda::device::Fp64Strategy {
        self.driver_profile.fp64_strategy()
    }

    /// Optimal `Precision` for `compile_shader_universal` on this GPU.
    ///
    /// Returns `Precision::F64` for compute-class GPUs (Native strategy),
    /// `Precision::Df64` for consumer GPUs (Hybrid strategy — routes through
    /// FP32 cores via DF64 core-streaming for ~10x effective throughput).
    /// `Concurrent` mode runs both DF64 and native f64 side-by-side for
    /// validation; defaults to F64 for the primary path.
    #[must_use]
    pub fn optimal_precision(&self) -> barracuda::shaders::Precision {
        use barracuda::device::Fp64Strategy;
        match self.fp64_strategy() {
            Fp64Strategy::Native | Fp64Strategy::Concurrent => barracuda::shaders::Precision::F64,
            Fp64Strategy::Hybrid => barracuda::shaders::Precision::Df64,
        }
    }

    /// Print GPU capabilities to stdout.
    pub fn print_info(&self) {
        println!("  GPU: {}", self.adapter_name);
        println!("  SHADER_F64: {}", if self.has_f64 { "YES" } else { "NO" });
        println!("  Arch: {:?}", self.driver_profile.arch);
        println!("  Driver: {:?}", self.driver_profile.driver);
        println!("  Fp64Strategy: {:?}", self.fp64_strategy());
        println!("  Optimal precision: {:?}", self.optimal_precision());
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
/// Measured on representative f64-capable GPUs for `FusedMapReduceF64`.
/// Callers should fall back to the CPU path when `n < GPU_DISPATCH_THRESHOLD`.
///
/// Use [`GpuF64::dispatch_threshold`] for a capability-aware value that
/// adapts to the detected GPU's latency profile.
pub const GPU_DISPATCH_THRESHOLD: usize = 10_000;

/// Element count thresholds for GPU vs CPU dispatch crossover.
/// Derived from benchmarking `FusedMapReduceF64` (hotSpring Exp001 §4.4, wetSpring Exp064/087).
const DISPATCH_THRESHOLD_FAST: usize = 5_000; // Fast f64 (AMD RDNA2+): ~4 cycles/DFMA
const DISPATCH_THRESHOLD_NATIVE: usize = 10_000; // Native f64 (NVIDIA Volta+): ~8 cycles/DFMA
const DISPATCH_THRESHOLD_SOFTWARE: usize = 25_000; // Software f64 (Apple M, Intel): emulated
const DISPATCH_THRESHOLD_UNKNOWN: usize = 50_000; // Unknown / very slow: conservative

impl GpuF64 {
    /// Capability-aware dispatch threshold for this GPU.
    ///
    /// Uses the driver profile's instruction-level latency to estimate
    /// the crossover point where GPU dispatch outperforms CPU. Higher
    /// instruction latency (software exp/log transcendental fallback) means a higher
    /// element count is needed to amortize launch overhead.
    ///
    /// Falls back to [`GPU_DISPATCH_THRESHOLD`] for unknown hardware.
    #[must_use]
    pub fn dispatch_threshold(&self) -> usize {
        use barracuda::device::latency::WgslOpClass;

        let model = self.driver_profile.latency_model();
        let dfma_cycles = model.raw_latency(WgslOpClass::F64Fma);

        // DFMA cycle ranges: 0..=4 fast, 5..=8 native, 9..=16 software, _ unknown
        match dfma_cycles {
            0..=4 => DISPATCH_THRESHOLD_FAST,
            5..=8 => DISPATCH_THRESHOLD_NATIVE,
            9..=16 => DISPATCH_THRESHOLD_SOFTWARE,
            _ => DISPATCH_THRESHOLD_UNKNOWN,
        }
    }
}
