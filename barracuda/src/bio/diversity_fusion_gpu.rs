// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused diversity metrics GPU — Shannon + Simpson + evenness in one dispatch.
//!
//! **Lean phase**: Delegates to `barracuda::ops::bio::diversity_fusion` (S63 absorption).
//! The local WGSL shader has been deleted; this module is a thin re-export
//! following the Write → Absorb → Lean cycle.

pub use barracuda::ops::bio::diversity_fusion::{
    DiversityFusionGpu, DiversityResult, diversity_fusion_cpu,
};
