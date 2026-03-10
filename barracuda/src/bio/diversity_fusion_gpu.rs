// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused diversity metrics GPU — Shannon + Simpson + evenness in one dispatch.
//!
//! **Lean phase**: Delegates to `barracuda::ops::bio::diversity_fusion` (S63 absorption).
//! The local WGSL shader has been deleted; this module is a thin re-export
//! following the Write → Absorb → Lean cycle.

pub use barracuda::ops::bio::diversity_fusion::{
    DiversityFusionGpu, DiversityResult, diversity_fusion_cpu,
};

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::gpu::GpuF64;
    use std::sync::Arc;

    #[test]
    fn api_surface_compiles() {
        fn _assert_result(_: &DiversityResult) {}
        let _: fn(&[f64], usize) -> Vec<DiversityResult> = diversity_fusion_cpu;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let device = gpu.to_wgpu_device();
        let fusion = DiversityFusionGpu::new(Arc::clone(&device)).expect("DiversityFusionGpu::new");
        let abundances = vec![1.0, 1.0, 1.0, 1.0];
        let result = fusion.compute(&abundances, 1, 4);
        assert!(result.is_ok(), "compute should succeed with valid input");
    }
}
