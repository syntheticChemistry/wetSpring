// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extended GPU statistics via barraCuda v0.3.3 primitives.
//!
//! Wires barraCuda's GPU-accelerated jackknife, bootstrap, Kimura fixation,
//! and Hargreaves ET₀ — avoiding CPU round-trips for large-batch operations.
//!
//! # Cross-spring provenance
//!
//! | Primitive | Origin | Evolved by | Absorbed |
//! |-----------|--------|------------|----------|
//! | `JackknifeMeanGpu` | wetSpring diversity jackknife | hotSpring precision shaders | barraCuda S60 |
//! | `BootstrapMeanGpu` | wetSpring RAWR bootstrap | hotSpring PRNG alignment | barraCuda S60 |
//! | `KimuraGpu` | groundSpring population genetics | neuralSpring batch dispatch | barraCuda S58 |
//! | `HargreavesBatchGpu` | airSpring hydrology | hotSpring f64 polyfills | barraCuda S66 |

use barracuda::stats::bootstrap::BootstrapMeanGpu;
use barracuda::stats::evolution::KimuraGpu;
use barracuda::stats::hydrology::HargreavesBatchGpu;
use barracuda::stats::jackknife::JackknifeMeanGpu;

use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu(
            "SHADER_F64 required for extended stats GPU".into(),
        ));
    }
    Ok(())
}

/// GPU jackknife mean with variance and standard error.
///
/// Dispatches N leave-one-out resamples in a single GPU pass.
/// Returns `(estimate, variance, std_error)`.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if the device lacks `SHADER_F64` or dispatch fails.
pub fn jackknife_mean_gpu(gpu: &GpuF64, data: &[f64]) -> Result<(f64, f64, f64)> {
    require_f64(gpu)?;

    let jk = JackknifeMeanGpu::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("JackknifeMeanGpu: {e}")))?;

    let result = jk
        .dispatch(data)
        .map_err(|e| Error::Gpu(format!("jackknife GPU: {e}")))?;

    Ok((result.estimate, result.variance, result.std_error))
}

/// GPU bootstrap resampling means.
///
/// Returns `n_bootstrap` resampled means computed on GPU. Use with
/// `barracuda::stats::bootstrap_ci` to compute confidence intervals.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or device lacks `SHADER_F64`.
pub fn bootstrap_means_gpu(
    gpu: &GpuF64,
    data: &[f64],
    n_bootstrap: u32,
    seed: u32,
) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let boot = BootstrapMeanGpu::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("BootstrapMeanGpu: {e}")))?;

    boot.dispatch(data, n_bootstrap, seed)
        .map_err(|e| Error::Gpu(format!("bootstrap GPU: {e}")))
}

/// GPU batch Kimura fixation probability.
///
/// For each `(N, s, p)` triplet, computes the probability of fixation
/// of an allele at initial frequency `p` in a population of size `N`
/// under selection coefficient `s`.
///
/// All three slices must have equal length.
///
/// # Cross-spring note
///
/// Kimura's diffusion approximation was first GPU-promoted by groundSpring
/// (population genetics), with f64 precision shaders from hotSpring.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or arrays differ in length.
pub fn kimura_fixation_gpu(
    gpu: &GpuF64,
    pop_sizes: &[f64],
    selections: &[f64],
    freqs: &[f64],
) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let kim =
        KimuraGpu::new(gpu.to_wgpu_device()).map_err(|e| Error::Gpu(format!("KimuraGpu: {e}")))?;

    kim.dispatch(pop_sizes, selections, freqs)
        .map_err(|e| Error::Gpu(format!("Kimura GPU: {e}")))
}

/// GPU batch Hargreaves ET₀ (reference evapotranspiration).
///
/// Computes Hargreaves-Samani ET₀ for each day from extraterrestrial
/// radiation, daily max/min temperature.
///
/// # Cross-spring note
///
/// Hargreaves ET₀ was first implemented by airSpring (hydrology),
/// with f64 polyfills contributed by hotSpring. Now available as a
/// barraCuda primitive for all springs.
///
/// # Errors
///
/// Returns [`Error::Gpu`] if dispatch fails or arrays differ in length.
pub fn hargreaves_et0_gpu(
    gpu: &GpuF64,
    ra: &[f64],
    t_max: &[f64],
    t_min: &[f64],
) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let hg = HargreavesBatchGpu::new(gpu.to_wgpu_device())
        .map_err(|e| Error::Gpu(format!("HargreavesBatchGpu: {e}")))?;

    hg.dispatch(ra, t_max, t_min)
        .map_err(|e| Error::Gpu(format!("Hargreaves GPU: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    type JkFn = fn(&GpuF64, &[f64]) -> Result<(f64, f64, f64)>;
    type BootFn = fn(&GpuF64, &[f64], u32, u32) -> Result<Vec<f64>>;
    type TriVecFn = fn(&GpuF64, &[f64], &[f64], &[f64]) -> Result<Vec<f64>>;

    #[test]
    fn api_surface_compiles() {
        let jk: JkFn = jackknife_mean_gpu;
        let boot: BootFn = bootstrap_means_gpu;
        let kim: TriVecFn = kimura_fixation_gpu;
        let hg: TriVecFn = hargreaves_et0_gpu;
        // Force the compiler to see these as used.
        assert!(std::mem::size_of_val(&jk) > 0);
        assert!(std::mem::size_of_val(&boot) > 0);
        assert!(std::mem::size_of_val(&kim) > 0);
        assert!(std::mem::size_of_val(&hg) > 0);
    }
}
