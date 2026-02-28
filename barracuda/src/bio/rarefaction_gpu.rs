// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated rarefaction with bootstrap confidence intervals.
//!
//! Uses `ToadStool`'s `BatchedMultinomialGpu` for batched multinomial
//! subsampling across all bootstrap replicates in one GPU dispatch, then
//! `DiversityFusionGpu` for fused Shannon + Simpson + evenness per replicate.
//!
//! # Architecture
//!
//! - **Batched multinomial**: All replicates sampled in one GPU kernel via
//!   `BatchedMultinomialGpu` (xoshiro128** PRNG).
//! - **Fused diversity**: `DiversityFusionGpu::compute` yields Shannon, Simpson,
//!   evenness for all replicates in one pass.
//! - **CPU fallback**: For small communities (< 50 species), CPU is faster
//!   due to dispatch overhead.
//!
//! # References
//!
//! - Gotelli & Colwell (2001). "Quantifying biodiversity."
//! - `ToadStool` `batched_multinomial`, `diversity_fusion`.

use crate::bio::diversity;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::bio::{BatchedMultinomialGpu, DiversityFusionGpu};

/// Minimum species count to justify GPU rarefaction dispatch.
///
/// Below 50 species, CPU bootstrap is faster than GPU dispatch +
/// buffer transfer. Determined empirically (Exp066).
const GPU_MIN_SPECIES: usize = 50;
const DEFAULT_BOOTSTRAP_N: usize = 1000;

/// Bootstrap confidence interval.
#[derive(Debug, Clone)]
pub struct BootstrapCi {
    /// Point estimate (mean of bootstrap distribution).
    pub mean: f64,
    /// Lower bound (2.5th percentile).
    pub lower: f64,
    /// Upper bound (97.5th percentile).
    pub upper: f64,
    /// Standard error (std dev of bootstrap distribution).
    pub se: f64,
}

/// Parameters for GPU rarefaction.
#[derive(Debug, Clone)]
pub struct RarefactionGpuParams {
    /// Number of bootstrap replicates. Default: 1000.
    pub n_bootstrap: usize,
    /// Rarefaction depth (number of reads to subsample). If `None`, uses
    /// the minimum sample total.
    pub depth: Option<usize>,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RarefactionGpuParams {
    fn default() -> Self {
        Self {
            n_bootstrap: DEFAULT_BOOTSTRAP_N,
            depth: None,
            seed: 42,
        }
    }
}

/// Result of GPU rarefaction for one sample.
#[derive(Debug, Clone)]
pub struct RarefactionResult {
    /// Shannon entropy with 95% CI.
    pub shannon: BootstrapCi,
    /// Simpson diversity with 95% CI.
    pub simpson: BootstrapCi,
    /// Observed features (species richness) with 95% CI.
    pub observed: BootstrapCi,
    /// Rarefaction depth used.
    pub depth: usize,
}

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for rarefaction GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated rarefaction with bootstrap confidence intervals.
///
/// For each bootstrap replicate, subsamples `depth` reads from the community
/// and computes diversity metrics on GPU.
///
/// # Errors
///
/// Returns an error if GPU dispatch fails or the device lacks f64 support.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn rarefaction_bootstrap_gpu(
    gpu: &GpuF64,
    counts: &[f64],
    params: &RarefactionGpuParams,
) -> Result<RarefactionResult> {
    require_f64(gpu)?;

    let total: f64 = counts.iter().sum();
    let depth = params.depth.unwrap_or(total as usize);

    if counts.is_empty() || depth == 0 {
        return Ok(RarefactionResult {
            shannon: BootstrapCi {
                mean: 0.0,
                lower: 0.0,
                upper: 0.0,
                se: 0.0,
            },
            simpson: BootstrapCi {
                mean: 0.0,
                lower: 0.0,
                upper: 0.0,
                se: 0.0,
            },
            observed: BootstrapCi {
                mean: 0.0,
                lower: 0.0,
                upper: 0.0,
                se: 0.0,
            },
            depth,
        });
    }

    let use_gpu = counts.len() >= GPU_MIN_SPECIES;

    let (shannon_samples, simpson_samples, observed_samples) = if use_gpu {
        let device = gpu.to_wgpu_device();
        let multinomial = BatchedMultinomialGpu::new(device.clone())
            .map_err(|e| Error::Gpu(format!("BatchedMultinomialGpu: {e}")))?;
        let diversity_fusion = DiversityFusionGpu::new(device)
            .map_err(|e| Error::Gpu(format!("DiversityFusionGpu: {e}")))?;

        let n_taxa = counts.len();
        let n_reps = params.n_bootstrap;
        if total <= 0.0 {
            return Ok(RarefactionResult {
                shannon: BootstrapCi {
                    mean: 0.0,
                    lower: 0.0,
                    upper: 0.0,
                    se: 0.0,
                },
                simpson: BootstrapCi {
                    mean: 0.0,
                    lower: 0.0,
                    upper: 0.0,
                    se: 0.0,
                },
                observed: BootstrapCi {
                    mean: 0.0,
                    lower: 0.0,
                    upper: 0.0,
                    se: 0.0,
                },
                depth,
            });
        }

        let mut cumulative = Vec::with_capacity(n_taxa);
        let mut cum = 0.0;
        for &c in counts {
            cum += c / total;
            cumulative.push(cum);
        }

        let depth_u32 = u32::try_from(depth)
            .map_err(|_| Error::InvalidInput("rarefaction depth exceeds u32::MAX".into()))?;
        let n_reps_u32 = u32::try_from(n_reps)
            .map_err(|_| Error::InvalidInput("n_bootstrap exceeds u32::MAX".into()))?;

        let mut seeds: Vec<u32> = (0..n_reps * 4)
            .map(|i| {
                let s = params
                    .seed
                    .wrapping_add((i / 4) as u64 * 0x9e37_79b9_7f4a_7c15);
                ((s >> ((i % 4) * 16)) as u32).wrapping_add(0x9e37_79b9)
            })
            .collect();

        let counts_u32 = multinomial
            .sample(&cumulative, &mut seeds, depth_u32, n_reps_u32)
            .map_err(|e| Error::Gpu(format!("BatchedMultinomialGpu::sample: {e}")))?;

        let abundances: Vec<f64> = counts_u32.iter().map(|&c| f64::from(c)).collect();

        let results = diversity_fusion
            .compute(&abundances, n_reps, n_taxa)
            .map_err(|e| Error::Gpu(format!("DiversityFusionGpu::compute: {e}")))?;

        let shannon_samples: Vec<f64> = results.iter().map(|r| r.shannon).collect();
        let simpson_samples: Vec<f64> = results.iter().map(|r| r.simpson).collect();
        let observed_samples: Vec<f64> = (0..n_reps)
            .map(|r| {
                let row = &counts_u32[r * n_taxa..(r + 1) * n_taxa];
                row.iter().filter(|&&c| c > 0).count() as f64
            })
            .collect();

        (shannon_samples, simpson_samples, observed_samples)
    } else {
        let mut shannon_samples = Vec::with_capacity(params.n_bootstrap);
        let mut simpson_samples = Vec::with_capacity(params.n_bootstrap);
        let mut observed_samples = Vec::with_capacity(params.n_bootstrap);
        let mut rng_state = params.seed;

        for _ in 0..params.n_bootstrap {
            let rarefied = subsample_community(counts, depth, &mut rng_state);
            shannon_samples.push(diversity::shannon(&rarefied));
            simpson_samples.push(diversity::simpson(&rarefied));
            observed_samples.push(diversity::observed_features(&rarefied));
        }

        (shannon_samples, simpson_samples, observed_samples)
    };

    Ok(RarefactionResult {
        shannon: compute_ci(&shannon_samples),
        simpson: compute_ci(&simpson_samples),
        observed: compute_ci(&observed_samples),
        depth,
    })
}

/// Subsample a community to the given depth using multinomial sampling.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn subsample_community(counts: &[f64], depth: usize, rng: &mut u64) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return vec![0.0; counts.len()];
    }

    // Cumulative probabilities
    let mut cumulative = Vec::with_capacity(counts.len());
    let mut cum = 0.0;
    for &c in counts {
        cum += c / total;
        cumulative.push(cum);
    }

    let mut result = vec![0.0_f64; counts.len()];

    for _ in 0..depth {
        // Generate uniform random in [0, 1)
        *rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (*rng >> 11) as f64 / (1_u64 << 53) as f64;

        // Binary search for the species
        let idx = match cumulative.binary_search_by(|p| p.total_cmp(&u)) {
            Ok(i) => i,
            Err(i) => i.min(counts.len() - 1),
        };
        result[idx] += 1.0;
    }

    result
}

/// Compute 95% confidence interval from bootstrap samples.
///
/// Uses [`barracuda::stats::percentile`] for interpolated bounds.
fn compute_ci(samples: &[f64]) -> BootstrapCi {
    if samples.is_empty() {
        return BootstrapCi {
            mean: 0.0,
            lower: 0.0,
            upper: 0.0,
            se: 0.0,
        };
    }

    let mean = barracuda::stats::mean(samples);
    let se = barracuda::stats::correlation::variance(samples)
        .map(f64::sqrt)
        .unwrap_or(0.0);

    BootstrapCi {
        mean,
        lower: barracuda::stats::percentile(samples, 2.5),
        upper: barracuda::stats::percentile(samples, 97.5),
        se,
    }
}

/// Batch GPU rarefaction for multiple samples.
///
/// Rarefies each sample to the specified depth (or minimum total across samples)
/// and computes diversity with bootstrap CIs.
///
/// # Errors
///
/// Returns an error if GPU dispatch fails or the device lacks f64 support.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn batch_rarefaction_gpu(
    gpu: &GpuF64,
    samples: &[Vec<f64>],
    params: &RarefactionGpuParams,
) -> Result<Vec<RarefactionResult>> {
    let min_depth = params.depth.unwrap_or_else(|| {
        samples
            .iter()
            .map(|s| s.iter().sum::<f64>() as usize)
            .filter(|&d| d > 0)
            .min()
            .unwrap_or(0)
    });

    let effective_params = RarefactionGpuParams {
        depth: Some(min_depth),
        ..params.clone()
    };

    samples
        .iter()
        .enumerate()
        .map(|(i, sample)| {
            let mut p = effective_params.clone();
            p.seed = params.seed.wrapping_add(i as u64 * 1_000_000);
            rarefaction_bootstrap_gpu(gpu, sample, &p)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Single-species community: every bootstrap replicate yields the same
    /// rarefied vector, so `Shannon=0`, `Simpson=0`, `observed=1` deterministically.
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn known_value_single_species_shannon_simpson_observed() {
        let gpu = crate::gpu::GpuF64::new().await.expect("GPU init");
        if !gpu.has_f64 {
            return;
        }
        // Single species: all counts in one bin
        let counts = vec![100.0];
        let params = RarefactionGpuParams {
            depth: Some(50),
            ..Default::default()
        };
        let result = rarefaction_bootstrap_gpu(&gpu, &counts, &params).unwrap();
        // Deterministic: every subsample gives [50.0], so Shannon=0, Simpson=0, observed=1
        assert!(
            result.shannon.mean.abs() < 1e-10,
            "Shannon single-species: got {}",
            result.shannon.mean
        );
        assert!(
            result.simpson.mean.abs() < 1e-10,
            "Simpson single-species: got {}",
            result.simpson.mean
        );
        assert!(
            (result.observed.mean - 1.0).abs() < 1e-10,
            "Observed single-species: got {}",
            result.observed.mean
        );
    }
}
