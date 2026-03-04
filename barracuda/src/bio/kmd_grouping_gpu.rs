// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Kendrick mass defect via barraCuda's `KmdGroupingF64`.
//!
//! Computes per-ion `[KM, NKM, KMD]` on GPU using the exact repeat unit
//! transform. Homologue grouping uses barraCuda's `group()` method which
//! returns per-ion group labels.
//!
//! # Cross-spring provenance
//!
//! - **wetSpring** (Write): CPU `kmd::kendrick_mass_defect` + `group_homologues`
//! - **barraCuda** (Absorb): `KmdGroupingF64` GPU shader (`kmd_grouping_f64.wgsl`)
//! - **wetSpring** (Lean): This module delegates to barraCuda's GPU primitive

use barracuda::ops::kmd_grouping_f64;

use super::kmd::{self, KmdResult};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu(
            "SHADER_F64 required for KMD grouping GPU".into(),
        ));
    }
    Ok(())
}

/// GPU-accelerated Kendrick mass defect via `KmdGroupingF64`.
///
/// Returns per-ion `KmdResult` computed entirely on GPU (element-wise
/// Kendrick transform). For small arrays (< 64), falls back to CPU.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn kendrick_mass_defect_native_gpu(
    gpu: &GpuF64,
    exact_masses: &[f64],
    exact_unit: f64,
    nominal_unit: f64,
) -> Result<Vec<KmdResult>> {
    require_f64(gpu)?;

    if exact_masses.len() < 64 {
        return Ok(kmd::kendrick_mass_defect(
            exact_masses,
            exact_unit,
            nominal_unit,
        ));
    }

    let kmd_op =
        kmd_grouping_f64::KmdGroupingF64::new(gpu.to_wgpu_device(), exact_unit, nominal_unit);

    let gpu_results = kmd_op
        .compute(exact_masses)
        .map_err(|e| Error::Gpu(format!("KmdGroupingF64 compute: {e}")))?;

    Ok(gpu_results
        .iter()
        .zip(exact_masses)
        .map(|(gr, &mass)| KmdResult {
            exact_mass: mass,
            kendrick_mass: gr.km,
            kmd: gr.kmd,
            nominal_km: gr.nkm,
        })
        .collect())
}

/// GPU KMD + homologue grouping via `KmdGroupingF64::group()`.
///
/// Returns per-ion group labels (0-indexed). Ions in the same homologous
/// series share the same label.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn kmd_group_gpu(
    gpu: &GpuF64,
    exact_masses: &[f64],
    exact_unit: f64,
    nominal_unit: f64,
    kmd_tolerance: f64,
) -> Result<Vec<usize>> {
    require_f64(gpu)?;

    let kmd_op =
        kmd_grouping_f64::KmdGroupingF64::new(gpu.to_wgpu_device(), exact_unit, nominal_unit);

    kmd_op
        .group(exact_masses, kmd_tolerance)
        .map_err(|e| Error::Gpu(format!("KmdGroupingF64 group: {e}")))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn small_array_fallback() {
        let masses = vec![498.930, 398.936, 298.943];
        let cpu =
            kmd::kendrick_mass_defect(&masses, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);
        assert_eq!(cpu.len(), 3);
        let kmd0 = cpu[0].kmd;
        for r in &cpu[1..] {
            assert!(
                (r.kmd - kmd0).abs() < 0.01,
                "homologous KMDs should match: {kmd0} vs {}",
                r.kmd
            );
        }
    }
}
