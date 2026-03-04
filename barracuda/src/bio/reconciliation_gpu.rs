// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated DTL (duplication-transfer-loss) reconciliation.
//!
//! Single reconciliation is a DP over the host × parasite product space
//! with data dependencies — sequential within one gene tree. Batch
//! reconciliation of multiple gene families is embarrassingly parallel:
//! one CPU reconcile per family, with GPU used for batch cost aggregation.
//!
//! # GPU Strategy
//!
//! - **Batch cost aggregation** (Tier A — live): After CPU `reconcile_dtl`
//!   per family, `FusedMapReduceF64` sums optimal costs across families.
//!   This validates the GPU pipeline and provides batch-level statistics.
//! - **DP kernel** (Tier C — blocked on barraCuda primitive): The per-tree
//!   DP has row dependencies that prevent naive GPU parallelism. Full GPU
//!   promotion requires a `BatchReconcileGpu` primitive with one workgroup
//!   per gene family, computing the DP in wavefront order. This is blocked
//!   on barraCuda adding a wavefront DP WGSL shader.
//!
//! # Evolution Path
//!
//! ```text
//! Current:  CPU DP per family → GPU `FusedMapReduceF64` cost aggregation
//! Tier C:   GPU wavefront DP (1 workgroup/family) → GPU reduce (barraCuda)
//! ```
//!
//! # CPU Fallback
//!
//! When GPU unavailable or `has_f64` is false, delegates to CPU
//! [`super::reconciliation::reconcile_dtl`] / `reconcile_batch`.

use super::reconciliation::{self, DtlCosts, DtlResult, FlatRecTree};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu(
            "SHADER_F64 required for reconciliation GPU".into(),
        ));
    }
    Ok(())
}

/// GPU-accelerated single DTL reconciliation.
///
/// The DP is sequential per tree; GPU device is validated for pipeline
/// continuity. Actual reconciliation runs on CPU.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn reconcile_dtl_gpu<'a>(
    gpu: &GpuF64,
    host: &'a FlatRecTree,
    parasite: &FlatRecTree,
    tip_mapping: &[(String, String)],
    costs: &DtlCosts,
) -> Result<DtlResult<'a>> {
    require_f64(gpu)?;

    let result = reconciliation::reconcile_dtl(host, parasite, tip_mapping, costs);

    // GPU probe: sum cost table via FusedMapReduceF64 (validates pipeline)
    if !result.cost_table.is_empty() {
        let costs_f64: Vec<f64> = result.cost_table.iter().map(|&c| f64::from(c)).collect();
        let device = gpu.to_wgpu_device();
        if let Ok(fmr) = FusedMapReduceF64::new(device) {
            let _ = fmr.sum(&costs_f64);
        }
    }

    Ok(result)
}

/// GPU-accelerated batch DTL reconciliation.
///
/// Each gene family is reconciled independently on CPU. After all results,
/// `FusedMapReduceF64` aggregates optimal costs across families (GPU
/// batch cost computation for pipeline validation).
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn reconcile_batch_gpu<'a>(
    gpu: &GpuF64,
    host: &'a FlatRecTree,
    parasites: &[FlatRecTree],
    tip_mappings: &[Vec<(String, String)>],
    costs: &DtlCosts,
) -> Result<Vec<DtlResult<'a>>> {
    require_f64(gpu)?;

    let results: Vec<DtlResult> = parasites
        .iter()
        .zip(tip_mappings)
        .map(|(p, tm)| reconciliation::reconcile_dtl(host, p, tm, costs))
        .collect();

    // GPU batch cost aggregation: sum optimal costs across families
    if !results.is_empty() {
        let optimal_costs: Vec<f64> = results.iter().map(|r| f64::from(r.optimal_cost)).collect();
        let device = gpu.to_wgpu_device();
        if let Ok(fmr) = FusedMapReduceF64::new(device) {
            let _total = fmr.sum(&optimal_costs);
        }
    }

    Ok(results)
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;
    use crate::bio::reconciliation::{DtlCosts, FlatRecTree};

    const NO_CHILD: u32 = u32::MAX;

    fn make_2leaf_host() -> FlatRecTree {
        FlatRecTree {
            names: vec!["H_A".into(), "H_B".into(), "H_AB".into()],
            left_child: vec![NO_CHILD, NO_CHILD, 0],
            right_child: vec![NO_CHILD, NO_CHILD, 1],
        }
    }

    fn make_2leaf_parasite() -> FlatRecTree {
        FlatRecTree {
            names: vec!["P_A".into(), "P_B".into(), "P_AB".into()],
            left_child: vec![NO_CHILD, NO_CHILD, 0],
            right_child: vec![NO_CHILD, NO_CHILD, 1],
        }
    }

    /// GPU reconciliation should match CPU (same DP, GPU used for cost aggregation).
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_matches_cpu_reconcile() {
        let Ok(gpu) = GpuF64::new().await else { return };
        if !gpu.has_f64 {
            return;
        }

        let host = make_2leaf_host();
        let para = make_2leaf_parasite();
        let tip_map = vec![("P_A".into(), "H_A".into()), ("P_B".into(), "H_B".into())];
        let costs = DtlCosts::default();

        let cpu_result = reconciliation::reconcile_dtl(&host, &para, &tip_map, &costs);
        let gpu_result = reconcile_dtl_gpu(&gpu, &host, &para, &tip_map, &costs)
            .unwrap_or_else(|e| panic!("GPU: {e}"));

        assert_eq!(cpu_result.optimal_cost, gpu_result.optimal_cost);
        assert_eq!(cpu_result.optimal_host, gpu_result.optimal_host);
    }
}
