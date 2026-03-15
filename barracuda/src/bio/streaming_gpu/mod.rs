// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unidirectional streaming GPU pipeline — full-stage GPU coverage.
//!
//! # Architecture (v3 — full pipeline GPU)
//!
//! ```text
//! GpuPipelineSession::new(gpu)
//!   ├── TensorContext          ← buffer pool + bind group cache + batching
//!   ├── QualityFilterCached    ← pre-compiled QF shader (local extension)
//!   ├── Dada2Gpu              ← pre-compiled DADA2 E-step shader (local extension)
//!   ├── GemmCached             ← pre-compiled GEMM pipeline (local extension)
//!   ├── FusedMapReduceF64      ← pre-compiled FMR pipeline (barraCuda)
//!   └── warmup dispatches      ← prime driver caches
//!
//! session.filter_reads_gpu(reads, params)
//!   └── QualityFilterCached::execute()  ← per-read parallel trimming
//!
//! session.denoise_gpu(uniques, params)
//!   └── Dada2Gpu::batch_log_p_error()  ← E-step on GPU, EM control on CPU
//!
//! session.stream_sample(...)
//!   ├── GemmCached::execute()  ← cached pipeline, per-call buffers only
//!   ├── FMR::shannon()         ← reuses compiled pipeline
//!   ├── FMR::simpson()         ← reuses compiled pipeline
//!   └── FMR::observed()        ← reuses compiled pipeline
//! ```
//!
//! # Pipeline stages on GPU
//!
//! | Stage         | Primitive            | Status          |
//! |---------------|----------------------|-----------------|
//! | Quality filter| `QualityFilterCached`  | GPU (per-read)  |
//! | Dereplication | —                    | CPU (hash)      |
//! | DADA2 denoise | `Dada2Gpu`             | GPU E-step      |
//! | Chimera       | —                    | CPU (k-mer)     |
//! | Taxonomy      | `GemmCached`           | GPU (GEMM)      |
//! | Diversity     | `FusedMapReduceF64`    | GPU/CPU (FMR)   |
//!
//! # Local extensions (for barraCuda absorption)
//!
//! - `QualityFilterCached`: per-read parallel quality trimming WGSL shader
//! - `Dada2Gpu`: batch pair-wise `log_p_error` with precomputed log-err table
//! - `GemmCached`: caches shader + pipeline + BGL at init
//! - `execute_to_buffer()`: returns GPU buffer without readback (chaining)

mod analytics;
mod stages;

pub use analytics::{
    AlphaDiversity, FullStreamingResult, StreamingGpuResult, stream_classify_and_diversity_cpu,
};

use crate::bio::dada2_gpu::Dada2Gpu;
use crate::bio::gemm_cached::GemmCached;
use crate::bio::quality_gpu::QualityFilterCached;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::bray_curtis_f64::BrayCurtisF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use std::sync::Arc;
use std::time::Instant;

/// Pre-warmed GPU pipeline session with full-stage coverage.
///
/// Holds pre-compiled pipelines for all GPU-accelerated stages:
/// quality filter, DADA2 E-step, taxonomy GEMM, and diversity FMR.
/// Quality and DADA2 now delegate to barraCuda absorbed primitives.
pub struct GpuPipelineSession {
    pub(super) qf: QualityFilterCached,
    pub(super) dada2: Dada2Gpu,
    pub(super) fmr: FusedMapReduceF64,
    pub(super) gemm: GemmCached,
    pub(super) bc: BrayCurtisF64,
    /// GPU warmup time in milliseconds.
    pub warmup_ms: f64,
}

impl GpuPipelineSession {
    /// Create a pre-warmed session with all GPU pipelines compiled.
    ///
    /// # Errors
    ///
    /// Returns an error if the device lacks `SHADER_F64` or pipeline compilation fails.
    pub fn new(gpu: &GpuF64) -> Result<Self> {
        if !gpu.has_f64 {
            return Err(Error::Gpu("SHADER_F64 required".into()));
        }

        let warmup_start = Instant::now();
        let device = gpu.to_wgpu_device();
        let ctx = Arc::clone(gpu.tensor_context());

        let qf = QualityFilterCached::new(Arc::clone(&device))?;
        let dada2 = Dada2Gpu::new(Arc::clone(&device))?;
        let fmr = FusedMapReduceF64::new(Arc::clone(&device))
            .map_err(|e| Error::Gpu(format!("FMR init: {e}")))?;
        let gemm = GemmCached::new(Arc::clone(&device), ctx);
        let bc = BrayCurtisF64::new(device)
            .map_err(|e| Error::Gpu(format!("BrayCurtisF64 init: {e}")))?;

        let _ = fmr.sum(&[1.0, 2.0, 3.0]);
        let _ = gemm.execute(&[1.0; 4], &[1.0; 4], 2, 2, 2, 1);

        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

        Ok(Self {
            qf,
            dada2,
            fmr,
            gemm,
            bc,
            warmup_ms,
        })
    }

    /// Pipeline session info string.
    #[must_use]
    pub fn ctx_stats(&self) -> String {
        format!("warmup={:.1}ms", self.warmup_ms)
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        fn _assert_session(_: &GpuPipelineSession) {}
        fn _assert_alpha(_: &AlphaDiversity) {}
        fn _assert_streaming_result(_: &StreamingGpuResult) {}
        fn _assert_full_result(_: &FullStreamingResult) {}
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let session = GpuPipelineSession::new(&gpu).expect("GpuPipelineSession::new");
        let counts = vec![1.0, 2.0, 3.0];
        let result = session.shannon(&counts);
        assert!(result.is_ok(), "shannon should succeed with valid input");
    }
}
