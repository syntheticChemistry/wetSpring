// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated rarefaction with bootstrap confidence intervals.
//!
//! Uses `ToadStool`'s `FusedMapReduceF64` for parallel diversity computation
//! across bootstrap replicates. The random subsampling uses a simple
//! multiplicative LCG seeded per-replicate, with diversity metrics computed
//! on GPU.
//!
//! # Architecture
//!
//! - **Bootstrap loop**: For each replicate, generate a rarefied abundance
//!   vector by random sampling, then compute Shannon/Simpson on GPU.
//! - **GPU dispatch**: `FusedMapReduceF64::shannon_entropy` and
//!   `::simpson_index` for each replicate.
//! - **CPU fallback**: For small communities (< 50 species) or few bootstrap
//!   iterations, CPU is faster due to dispatch overhead.
//!
//! # References
//!
//! - Gotelli & Colwell (2001). "Quantifying biodiversity."
//! - `ToadStool` `prng_xoshiro` for future full-GPU random sampling.

use crate::bio::diversity;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;

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

    let fmr = if use_gpu {
        Some(
            FusedMapReduceF64::new(gpu.to_wgpu_device())
                .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?,
        )
    } else {
        None
    };

    let mut shannon_samples = Vec::with_capacity(params.n_bootstrap);
    let mut simpson_samples = Vec::with_capacity(params.n_bootstrap);
    let mut observed_samples = Vec::with_capacity(params.n_bootstrap);

    let mut rng_state = params.seed;

    for _ in 0..params.n_bootstrap {
        let rarefied = subsample_community(counts, depth, &mut rng_state);

        if let Some(ref fmr) = fmr {
            let h = fmr
                .shannon_entropy(&rarefied)
                .map_err(|e| Error::Gpu(format!("Shannon GPU: {e}")))?;
            shannon_samples.push(h);

            let d = fmr
                .simpson_index(&rarefied)
                .map_err(|e| Error::Gpu(format!("Simpson GPU: {e}")))?;
            simpson_samples.push(1.0 - d);
        } else {
            shannon_samples.push(diversity::shannon(&rarefied));
            simpson_samples.push(diversity::simpson(&rarefied));
        }

        observed_samples.push(diversity::observed_features(&rarefied));
    }

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
        let idx = match cumulative
            .binary_search_by(|p| p.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i,
            Err(i) => i.min(counts.len() - 1),
        };
        result[idx] += 1.0;
    }

    result
}

/// Compute 95% confidence interval from bootstrap samples.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn compute_ci(samples: &[f64]) -> BootstrapCi {
    if samples.is_empty() {
        return BootstrapCi {
            mean: 0.0,
            lower: 0.0,
            upper: 0.0,
            se: 0.0,
        };
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;

    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let se = variance.sqrt();

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower_idx = ((samples.len() as f64 * 0.025).floor() as usize).min(samples.len() - 1);
    let upper_idx = ((samples.len() as f64 * 0.975).floor() as usize).min(samples.len() - 1);

    BootstrapCi {
        mean,
        lower: sorted[lower_idx],
        upper: sorted[upper_idx],
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
mod tests {
    // GPU tests require hardware; integration tests in validate_diversity_gpu.rs
}
