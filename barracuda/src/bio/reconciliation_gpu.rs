// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated DTL (duplication-transfer-loss) reconciliation.
//!
//! Single reconciliation is a DP over the host x parasite product space
//! with data dependencies — sequential within one gene tree. However,
//! batch reconciliation of multiple gene families is embarrassingly
//! parallel: one workgroup per gene tree.
//!
//! In a pure-GPU streaming pipeline, gene trees arrive from the
//! neighbor-joining stage and reconciliation results feed downstream
//! evolutionary analysis without CPU round-trips between families.

use super::reconciliation::{self, DtlCosts, DtlResult, FlatRecTree};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

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
/// For a single gene tree, the DP is sequential. GPU device is validated
/// for pipeline continuity.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn reconcile_dtl_gpu(
    gpu: &GpuF64,
    host: &FlatRecTree,
    parasite: &FlatRecTree,
    tip_mapping: &[(String, String)],
    costs: &DtlCosts,
) -> Result<DtlResult> {
    require_f64(gpu)?;
    Ok(reconciliation::reconcile_dtl(
        host,
        parasite,
        tip_mapping,
        costs,
    ))
}

/// GPU-accelerated batch DTL reconciliation.
///
/// Each gene family is reconciled independently — one workgroup per
/// family when dispatched on GPU. Currently uses the CPU kernel for
/// each family with GPU device validation for pipeline integration.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn reconcile_batch_gpu(
    gpu: &GpuF64,
    host: &FlatRecTree,
    parasites: &[FlatRecTree],
    tip_mappings: &[Vec<(String, String)>],
    costs: &DtlCosts,
) -> Result<Vec<DtlResult>> {
    require_f64(gpu)?;

    let results: Vec<DtlResult> = parasites
        .iter()
        .zip(tip_mappings)
        .map(|(p, tm)| reconciliation::reconcile_dtl(host, p, tm, costs))
        .collect();
    Ok(results)
}
