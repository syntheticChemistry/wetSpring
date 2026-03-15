// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated tolerance-based m/z matching via barraCuda's `BatchToleranceSearchF64`.
//!
//! Dispatches S sample ions against R reference ions in a single GPU pass,
//! returning `[S × R]` match scores in `[0, 1]`. Replaces the CPU binary-search
//! path for large-scale PFAS suspect screening.
//!
//! # Cross-spring provenance
//!
//! - **wetSpring** (Write): CPU `find_within_ppm` / `find_within_da` binary search
//! - **barraCuda** (Absorb): `BatchToleranceSearchF64` GPU shader (`batch_tolerance_search_f64.wgsl`)
//! - **wetSpring** (Lean): This module delegates to barraCuda's GPU primitive

use barracuda::ops::batch_tolerance_search_f64::BatchToleranceSearchF64;

use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu(
            "SHADER_F64 required for tolerance search GPU".into(),
        ));
    }
    Ok(())
}

/// GPU batch tolerance search: match sample masses against reference masses.
///
/// Returns flat `[S × R]` row-major `f32` scores in `[0, 1]`:
/// - `1.0` = exact match
/// - `0.0` = outside tolerance
/// - Linear interpolation between
///
/// For small inputs (< 64 samples or < 8 refs), falls back to CPU binary search.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn batch_tolerance_search_gpu(
    gpu: &GpuF64,
    sample_masses: &[f64],
    ref_masses: &[f64],
    ppm_tol: f64,
    da_tol: f64,
) -> Result<Vec<f32>> {
    require_f64(gpu)?;

    if sample_masses.len() < 64 || ref_masses.len() < 8 {
        return Ok(cpu_fallback(sample_masses, ref_masses, ppm_tol, da_tol));
    }

    let searcher = BatchToleranceSearchF64::new(gpu.to_wgpu_device(), ppm_tol, da_tol);
    searcher
        .search(sample_masses, ref_masses)
        .map_err(|e| Error::Gpu(format!("BatchToleranceSearchF64: {e}")))
}

/// CPU fallback for small arrays: ppm-based match scoring.
fn cpu_fallback(samples: &[f64], refs: &[f64], ppm_tol: f64, da_tol: f64) -> Vec<f32> {
    let mut scores = Vec::with_capacity(samples.len() * refs.len());
    for &s in samples {
        let ppm_window = s * ppm_tol * crate::tolerances::PPM_FACTOR;
        let tol = ppm_window.max(da_tol);
        for &r in refs {
            let diff = (s - r).abs();
            if diff <= tol {
                #[expect(clippy::cast_possible_truncation)] // Truncation: score in [0,1], fits f32
                scores.push((1.0 - diff / tol) as f32);
            } else {
                scores.push(0.0);
            }
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fallback_basic() {
        let samples = vec![100.0, 200.0];
        let refs = vec![100.0, 150.0];
        let scores = cpu_fallback(&samples, &refs, 10.0, 0.01);
        assert_eq!(scores.len(), 4);
        assert!(scores[0] > 0.99, "exact match should score ~1.0");
        assert!(scores[1] < f32::EPSILON, "no match at 50 Da distance");
    }

    #[test]
    fn cpu_fallback_empty() {
        let scores = cpu_fallback(&[], &[100.0], 10.0, 0.01);
        assert!(scores.is_empty());
    }
}
