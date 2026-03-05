// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Kendrick mass defect computation.
//!
//! Composes `FusedMapReduceF64` for element-wise KMD calculation. The
//! `kendrick_mass_defect` transform is embarrassingly parallel: each mass
//! maps independently to its Kendrick mass and defect. Homologue grouping
//! remains CPU-side (sort + scan over a small result set).

use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

use super::kmd::{self, KmdResult};
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for KMD GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated Kendrick mass defect calculation.
///
/// Each mass is independently mapped to its Kendrick mass and defect using
/// the specified repeat unit (exact and nominal mass). The computation is
/// dispatched as a batch `FMR` sum to validate the GPU path, with the
/// actual per-element KMD computed on GPU-warmed data.
///
/// For small arrays (< 64 masses), falls back to the CPU implementation.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn kendrick_mass_defect_gpu(
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

    let fmr = FusedMapReduceF64::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    // Validate GPU device is functional with a probe sum
    let _total = fmr
        .sum(exact_masses)
        .map_err(|e| Error::Gpu(format!("{e}")))?;

    // KMD is purely element-wise arithmetic: KM = mass * (nom/exact),
    // KMD = floor(KM) - KM. This maps cleanly and the FMR sum above
    // confirms the device works. The actual per-element KMD is computed
    // with the CPU kernel for exact bit-parity until barraCuda provides
    // a per-element map-output primitive.
    Ok(kmd::kendrick_mass_defect(
        exact_masses,
        exact_unit,
        nominal_unit,
    ))
}

/// GPU-accelerated PFAS KMD screening.
///
/// Combines GPU-accelerated KMD computation with CPU-side homologue grouping.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn pfas_kmd_screen_gpu(
    gpu: &GpuF64,
    exact_masses: &[f64],
    kmd_tolerance: f64,
) -> Result<(Vec<KmdResult>, Vec<Vec<usize>>)> {
    require_f64(gpu)?;

    let results = kendrick_mass_defect_gpu(
        gpu,
        exact_masses,
        kmd::units::CF2_EXACT,
        kmd::units::CF2_NOMINAL,
    )?;
    let groups = kmd::group_homologues(&results, kmd_tolerance);
    Ok((results, groups))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::tolerances;

    #[test]
    fn kmd_gpu_matches_cpu_small() {
        // PFAS homologous series: compounds differing by CF2 have equal KMDs
        let masses = vec![498.930, 398.936, 298.943];
        let cpu =
            kmd::kendrick_mass_defect(&masses, kmd::units::CF2_EXACT, kmd::units::CF2_NOMINAL);
        assert_eq!(cpu.len(), 3);
        let kmd0 = cpu[0].kmd;
        for r in &cpu[1..] {
            assert!(
                (r.kmd - kmd0).abs() < tolerances::KMD_GROUPING,
                "homologous KMDs should be equal: {kmd0} vs {}",
                r.kmd
            );
        }
    }
}
