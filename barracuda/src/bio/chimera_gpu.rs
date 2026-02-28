// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated chimera detection.
//!
//! Uses the same UCHIME-style algorithm as [`super::chimera`] but accelerates
//! the k-mer sketch and parent-candidate scoring on GPU.
//!
//! # GPU Strategy
//!
//! - **K-mer histograms**: `KmerHistogramGpu` for batch k-mer counting per
//!   sequence. Each sequence gets a dense histogram of length 4^k (k=8).
//! - **Batch sketch scoring**: `GemmF64` for dot products between histograms
//!   (A × A^T), `FusedMapReduceF64` for vector norms. Cosine similarity
//!   ranks parent candidates (proxy for Jaccard-like shared k-mer count).
//! - **Chimera classification**: Stays CPU-side — `test_chimera_fast` uses
//!   prefix-sum crossover evaluation with early termination (sequential).
//!
//! # CPU Fallback
//!
//! When GPU is unavailable, `has_f64` is false, or sequence count is below
//! the dispatch threshold (16), delegates to [`super::chimera::detect_chimeras`].

use crate::bio::chimera::{self, ChimeraParams, ChimeraResult, ChimeraStats};
use crate::bio::dada2::Asv;
use crate::bio::kmer_gpu::KmerGpu;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::linalg::gemm_f64::GemmF64;
use std::sync::Arc;

/// Minimum sequence count to use GPU path (avoids dispatch overhead for tiny inputs).
const GPU_DISPATCH_THRESHOLD: usize = 16;

/// k-mer size for chimera sketch (matches CPU `SKETCH_K`).
const SKETCH_K: u32 = 8;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for chimera GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated chimera detection with CPU-identical classification.
///
/// Uses `KmerHistogramGpu` for batch k-mer counting and `GemmF64` +
/// `FusedMapReduceF64` for batch sketch similarity (cosine between k-mer
/// profiles). Parent candidate selection uses GPU similarity; the final
/// chimera test (`test_chimera_fast`) runs on CPU for exact math parity.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
#[allow(clippy::cast_precision_loss)]
pub fn detect_chimeras_gpu(
    gpu: &GpuF64,
    seqs: &[Asv],
    params: &ChimeraParams,
) -> Result<(Vec<ChimeraResult>, ChimeraStats)> {
    const MAX_PARENT_CANDIDATES: usize = 8;
    require_f64(gpu)?;

    if seqs.len() < GPU_DISPATCH_THRESHOLD {
        return Ok(chimera::detect_chimeras(seqs, params));
    }

    let device = gpu.to_wgpu_device();
    let hist_len = 4_usize.pow(SKETCH_K);

    // 1. Batch k-mer histograms via KmerHistogramGpu
    let kmer_gpu = KmerGpu::new(&device);
    let mut histograms: Vec<Vec<f64>> = Vec::with_capacity(seqs.len());
    for asv in seqs {
        let result = kmer_gpu
            .count_from_sequence(&asv.sequence, SKETCH_K)
            .map_err(|e| Error::Gpu(format!("KmerHistogramGpu: {e}")))?;
        let vec_f64: Vec<f64> = result.histogram.iter().map(|&c| f64::from(c)).collect();
        histograms.push(vec_f64);
    }

    // 2. Batch sketch similarity via GemmF64 (dot products) + FusedMapReduceF64 (norms)
    let n = histograms.len();
    let flat_a: Vec<f64> = histograms.iter().flat_map(|h| h.iter().copied()).collect();

    // A^T [D, N] for GEMM: A [N, D] × A^T [D, N] → [N, N] dot products
    let mut flat_at = vec![0.0_f64; hist_len * n];
    for i in 0..n {
        for j in 0..hist_len {
            flat_at[j * n + i] = flat_a[i * hist_len + j];
        }
    }

    let dot_matrix = GemmF64::execute(Arc::clone(&device), &flat_a, &flat_at, n, hist_len, n, 1)
        .map_err(|e| Error::Gpu(format!("GemmF64 sketch similarity: {e}")))?;

    let fmr = FusedMapReduceF64::new(device)
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    let mut norms = Vec::with_capacity(n);
    for h in &histograms {
        let sum_sq = fmr
            .sum_of_squares(h)
            .map_err(|e| Error::Gpu(format!("norm reduce: {e}")))?;
        norms.push(sum_sq.sqrt());
    }

    // 3. CPU: run chimera loop with GPU-derived similarity for parent ranking
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        if i < 2 {
            results.push(ChimeraResult {
                query_idx: i,
                is_chimera: false,
                score: 0.0,
                left_parent: None,
                right_parent: None,
                crossover: None,
            });
            continue;
        }

        let query = &seqs[i];

        let eligible: Vec<usize> = (0..i)
            .filter(|&j| {
                seqs[j].abundance as f64 >= query.abundance as f64 * params.min_parent_fold
            })
            .collect();

        if eligible.len() < 2 {
            results.push(ChimeraResult {
                query_idx: i,
                is_chimera: false,
                score: 0.0,
                left_parent: None,
                right_parent: None,
                crossover: None,
            });
            continue;
        }

        // Rank parents by cosine similarity (GPU-computed)
        let mut scored: Vec<(usize, f64)> = eligible
            .iter()
            .map(|&j| {
                let dot = dot_matrix[i * n + j];
                let denom = norms[i] * norms[j];
                let sim = if denom > 0.0 { dot / denom } else { 0.0 };
                (j, sim)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = scored.len().min(MAX_PARENT_CANDIDATES);
        let candidates: Vec<usize> = scored[..top_k].iter().map(|&(j, _)| j).collect();

        let result = chimera::test_chimera_fast(query, seqs, &candidates, i, params);
        results.push(result);
    }

    let chimeras_found = results.iter().filter(|r| r.is_chimera).count();
    let stats = ChimeraStats {
        input_sequences: n,
        chimeras_found,
        retained: n - chimeras_found,
    };

    Ok((results, stats))
}

/// GPU-accelerated chimera removal.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn remove_chimeras_gpu(
    gpu: &GpuF64,
    seqs: &[Asv],
    params: &ChimeraParams,
) -> Result<(Vec<Asv>, ChimeraStats)> {
    let (results, stats) = detect_chimeras_gpu(gpu, seqs, params)?;
    let filtered: Vec<Asv> = results
        .iter()
        .filter(|r| !r.is_chimera)
        .map(|r| seqs[r.query_idx].clone())
        .collect();
    Ok((filtered, stats))
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;
    use crate::bio::chimera::ChimeraParams;
    use crate::bio::dada2::Asv;

    fn make_asv(seq: &[u8], abundance: usize) -> Asv {
        Asv {
            sequence: seq.to_vec(),
            abundance,
            n_members: 1,
        }
    }

    /// GPU output should match CPU for chimera detection (within algorithm variance).
    /// Uses cosine similarity for parent ranking vs CPU's sum-of-min; classification
    /// is identical via `test_chimera_fast`.
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_matches_cpu_chimera() {
        let Ok(gpu) = GpuF64::new().await else { return };
        if !gpu.has_f64 {
            return;
        }

        let parent_a = b"ACGTACGTACTTTTTTTTTT";
        let parent_b = b"TTTTTTTTTTACGTACGTAC";
        let chimera = b"ACGTACGTACACGTACGTAC";

        // Need >= 16 ASVs to hit GPU path; pad with unique sequences
        let mut asvs = vec![
            make_asv(parent_a, 5000),
            make_asv(parent_b, 3000),
            make_asv(chimera, 50),
        ];
        let bases = [b'A', b'C', b'G', b'T'];
        for i in 0..14 {
            let mut seq = b"ACGTACGTACGTAC".to_vec();
            seq.push(bases[i % 4]);
            seq.push(bases[i / 4]);
            asvs.push(make_asv(&seq, 100 + i));
        }

        let (cpu_results, cpu_stats) = chimera::detect_chimeras(&asvs, &ChimeraParams::default());
        let (gpu_results, gpu_stats) = detect_chimeras_gpu(&gpu, &asvs, &ChimeraParams::default())
            .unwrap_or_else(|e| panic!("GPU: {e}"));

        assert_eq!(cpu_stats.chimeras_found, gpu_stats.chimeras_found);
        assert_eq!(cpu_stats.retained, gpu_stats.retained);
        for i in 0..asvs.len() {
            assert_eq!(
                cpu_results[i].is_chimera, gpu_results[i].is_chimera,
                "query_idx={i}"
            );
        }
    }
}
