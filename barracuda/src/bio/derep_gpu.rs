// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated sequence dereplication.
//!
//! Uses `KmerHistogramGpu` for batch k-mer signature computation and
//! `GemmF64` + `FusedMapReduceF64` for batch pairwise similarity.
//! Identical sequences have identical k-mer histograms (cosine = 1.0).
//! Greedy clustering on CPU groups by similarity; best representative
//! selection stays CPU-side.
//!
//! # GPU Strategy
//!
//! - **K-mer histograms**: `KmerHistogramGpu` for each sequence (k=8).
//! - **Batch similarity**: `GemmF64` for dot products, `FusedMapReduceF64`
//!   for norms. Cosine similarity matrix identifies identical sequences.
//! - **Clustering**: Greedy union-find on CPU — sequences with cosine >=
//!   threshold (1.0 - ε) are grouped. Best rep = highest mean quality.
//!
//! # CPU Fallback
//!
//! When GPU unavailable, sequence count < 16, or any sequence length < 8,
//! delegates to [`super::derep::dereplicate`].

use super::derep::{DerepSort, DerepStats, UniqueSequence, mean_quality};
use crate::bio::kmer_gpu::KmerGpu;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;
use crate::io::fastq::FastqRecord;
use barracuda::ops::fused_map_reduce_f64::FusedMapReduceF64;
use barracuda::ops::linalg::gemm_f64::GemmF64;
use std::sync::Arc;

/// Minimum sequence count to use GPU path.
const GPU_DISPATCH_THRESHOLD: usize = 16;

/// k-mer size for derep signatures.
const SKETCH_K: u32 = 8;

/// Cosine threshold for "identical" sequences (1.0 - ε).
const IDENTICAL_THRESHOLD: f64 = 1.0 - 1e-10;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for derep GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated sequence dereplication.
///
/// Uses `KmerHistogramGpu` for batch k-mer signatures and `GemmF64` +
/// `FusedMapReduceF64` for pairwise cosine similarity. Sequences with
/// cosine >= 1.0 - ε are grouped; clustering and best-rep selection
/// run on CPU.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
#[allow(clippy::cast_precision_loss)]
pub fn dereplicate_gpu(
    gpu: &GpuF64,
    records: &[FastqRecord],
    sort: DerepSort,
    min_abundance: usize,
) -> Result<(Vec<UniqueSequence>, DerepStats)> {
    fn find(p: &[usize], i: usize) -> usize {
        if p[i] == i { i } else { find(p, p[i]) }
    }
    fn union(p: &mut [usize], i: usize, j: usize) {
        let pi = find(p, i);
        let pj = find(p, j);
        if pi != pj {
            p[pi] = pj;
        }
    }
    require_f64(gpu)?;

    if records.len() < GPU_DISPATCH_THRESHOLD {
        return Ok(super::derep::dereplicate(records, sort, min_abundance));
    }

    // Uppercase sequences (match CPU)
    let sequences: Vec<Vec<u8>> = records
        .iter()
        .map(|r| r.sequence.iter().map(u8::to_ascii_uppercase).collect())
        .collect();

    // Fall back to CPU if any sequence too short for k-mer
    if sequences.iter().any(|s| s.len() < SKETCH_K as usize) {
        return Ok(super::derep::dereplicate(records, sort, min_abundance));
    }

    let device = gpu.to_wgpu_device();
    let hist_len = 4_usize.pow(SKETCH_K);
    let kmer_gpu = KmerGpu::new(&device);

    // 1. Batch k-mer histograms
    let mut histograms: Vec<Vec<f64>> = Vec::with_capacity(records.len());
    for seq in &sequences {
        let result = kmer_gpu
            .count_from_sequence(seq, SKETCH_K)
            .map_err(|e| Error::Gpu(format!("KmerHistogramGpu: {e}")))?;
        let vec_f64: Vec<f64> = result.histogram.iter().map(|&c| f64::from(c)).collect();
        histograms.push(vec_f64);
    }

    // 2. Batch similarity via GemmF64 + FusedMapReduceF64
    let n = histograms.len();
    let flat_a: Vec<f64> = histograms.iter().flat_map(|h| h.iter().copied()).collect();

    let mut flat_at = vec![0.0_f64; hist_len * n];
    for i in 0..n {
        for j in 0..hist_len {
            flat_at[j * n + i] = flat_a[i * hist_len + j];
        }
    }

    let dot_matrix = GemmF64::execute(Arc::clone(&device), &flat_a, &flat_at, n, hist_len, n, 1)
        .map_err(|e| Error::Gpu(format!("GemmF64 derep similarity: {e}")))?;

    let fmr = FusedMapReduceF64::new(device)
        .map_err(|e| Error::Gpu(format!("FusedMapReduceF64: {e}")))?;

    let mut norms = Vec::with_capacity(n);
    for h in &histograms {
        let sum_sq = fmr
            .sum_of_squares(h)
            .map_err(|e| Error::Gpu(format!("norm reduce: {e}")))?;
        norms.push(sum_sq.sqrt());
    }

    // 3. CPU: union-find grouping by cosine >= IDENTICAL_THRESHOLD
    let mut parent: Vec<usize> = (0..n).collect();

    for i in 1..n {
        for j in 0..i {
            let dot = dot_matrix[i * n + j];
            let denom = norms[i] * norms[j];
            let cos = if denom > 0.0 { dot / denom } else { 0.0 };
            if cos >= IDENTICAL_THRESHOLD {
                union(&mut parent, i, j);
                break; // i is now in same group as j
            }
        }
    }

    // 4. Aggregate groups: root -> (members, best_rep_idx)
    let mut root_to_members: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let r = find(&parent, i);
        root_to_members.entry(r).or_default().push(i);
    }

    let mean_qualities: Vec<f64> = records.iter().map(|r| mean_quality(&r.quality)).collect();

    let mut uniques: Vec<UniqueSequence> = Vec::new();
    for (_root, members) in root_to_members {
        let best_idx = members
            .iter()
            .copied()
            .max_by(|&a, &b| {
                mean_qualities[a]
                    .partial_cmp(&mean_qualities[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(members[0]);

        let rec = &records[best_idx];
        uniques.push(UniqueSequence {
            sequence: sequences[members[0]].clone(),
            abundance: members.len(),
            best_quality: mean_qualities[best_idx],
            representative_id: rec.id.clone(),
            representative_quality: rec.quality.clone(),
        });
    }

    // 5. Filter, sort, stats (match CPU)
    let effective_min = if min_abundance == 0 { 1 } else { min_abundance };
    uniques.retain(|u| u.abundance >= effective_min);

    match sort {
        DerepSort::Abundance => {
            uniques.sort_by(|a, b| {
                b.abundance
                    .cmp(&a.abundance)
                    .then_with(|| a.sequence.cmp(&b.sequence))
            });
        }
        DerepSort::Sequence => {
            uniques.sort_by(|a, b| a.sequence.cmp(&b.sequence));
        }
    }

    let n_unique = uniques.len();
    let max_abundance = uniques.iter().map(|u| u.abundance).max().unwrap_or(0);
    let singletons = uniques.iter().filter(|u| u.abundance == 1).count();
    let total_abundance: usize = uniques.iter().map(|u| u.abundance).sum();

    let stats = DerepStats {
        input_sequences: records.len(),
        unique_sequences: n_unique,
        max_abundance,
        singletons,
        mean_abundance: if n_unique > 0 {
            total_abundance as f64 / n_unique as f64
        } else {
            0.0
        },
    };

    Ok((uniques, stats))
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    fn make_record(id: &str, seq: &[u8], q: u8) -> FastqRecord {
        FastqRecord {
            id: id.to_string(),
            sequence: seq.to_vec(),
            quality: vec![33 + q; seq.len()],
        }
    }

    /// GPU output should match CPU for dereplication (identical grouping).
    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_matches_cpu_derep() {
        let Ok(gpu) = GpuF64::new().await else { return };
        if !gpu.has_f64 {
            return;
        }

        // Need >= 16 records to hit GPU path
        let mut records = vec![
            make_record("r1", b"ACGTACGTACGTACGT", 30),
            make_record("r2", b"ACGTACGTACGTACGT", 35),
            make_record("r3", b"GCTAGCTAGCTAGCTA", 30),
            make_record("r4", b"ACGTACGTACGTACGT", 25),
        ];
        for i in 5..=20 {
            records.push(make_record(
                &format!("r{i}"),
                if i % 2 == 0 {
                    b"ACGTACGTACGTACGT"
                } else {
                    b"GCTAGCTAGCTAGCTA"
                },
                30,
            ));
        }

        let (cpu_uniques, cpu_stats) =
            super::super::derep::dereplicate(&records, DerepSort::Abundance, 0);
        let (gpu_uniques, gpu_stats) = dereplicate_gpu(&gpu, &records, DerepSort::Abundance, 0)
            .unwrap_or_else(|e| panic!("GPU: {e}"));

        assert_eq!(cpu_stats.unique_sequences, gpu_stats.unique_sequences);
        assert_eq!(cpu_stats.max_abundance, gpu_stats.max_abundance);
        assert_eq!(cpu_uniques.len(), gpu_uniques.len());
        for (c, g) in cpu_uniques.iter().zip(gpu_uniques.iter()) {
            assert_eq!(c.sequence, g.sequence);
            assert_eq!(c.abundance, g.abundance);
        }
    }
}
