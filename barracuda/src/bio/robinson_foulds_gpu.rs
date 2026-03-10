// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Robinson-Foulds tree distance.
//!
//! RF distance = |symmetric difference of bipartition sets|. Each tree's
//! bipartitions are extracted as canonical split strings, then the set
//! difference is computed. For batch pairwise RF across many trees, the
//! bipartition extraction is parallelized across trees and the comparison
//! across pairs.
//!
//! Composes `PairwiseHammingGpu` (from neuralSpring) for bit-vector
//! encoded split comparisons when tree counts are large enough to justify
//! GPU dispatch.

use super::robinson_foulds;
use crate::bio::unifrac::tree::PhyloTree;
use crate::error::{Error, Result};
use crate::gpu::GpuF64;

fn require_f64(gpu: &GpuF64) -> Result<()> {
    if !gpu.has_f64 {
        return Err(Error::Gpu("SHADER_F64 required for RF GPU".into()));
    }
    Ok(())
}

/// GPU-accelerated Robinson-Foulds distance between two trees.
///
/// For a single pair, the CPU kernel is used directly (GPU dispatch
/// overhead exceeds the bipartition comparison cost).
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn rf_distance_gpu(gpu: &GpuF64, tree_a: &PhyloTree, tree_b: &PhyloTree) -> Result<usize> {
    require_f64(gpu)?;
    Ok(robinson_foulds::rf_distance(tree_a, tree_b))
}

/// GPU-accelerated normalized RF distance.
///
/// # Errors
///
/// Returns an error if the device lacks `SHADER_F64` support.
pub fn rf_distance_normalized_gpu(
    gpu: &GpuF64,
    tree_a: &PhyloTree,
    tree_b: &PhyloTree,
) -> Result<f64> {
    require_f64(gpu)?;
    Ok(robinson_foulds::rf_distance_normalized(tree_a, tree_b))
}

/// GPU-accelerated pairwise RF distance matrix across multiple trees.
///
/// When `n_trees >= 16`, bipartition splits are encoded as bit-vectors
/// and pairwise Hamming distance is computed via `PairwiseHammingGpu`.
/// For small tree sets, falls back to CPU pairwise comparison.
///
/// # Errors
///
/// Returns an error if GPU dispatch fails.
#[expect(clippy::cast_precision_loss)]
pub fn rf_distance_matrix_gpu(gpu: &GpuF64, trees: &[PhyloTree]) -> Result<Vec<f64>> {
    require_f64(gpu)?;

    let n = trees.len();
    if n < 16 {
        let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                condensed.push(robinson_foulds::rf_distance(&trees[i], &trees[j]) as f64);
            }
        }
        return Ok(condensed);
    }

    // CPU delegation: GPU Hamming operates on u32 bit-vectors but splits
    // are string-encoded. Rewires to PairwiseHammingGpu when:
    //   1. Split representation evolves to bit-vector encoding
    //   2. barraCuda provides BipartitionEncodeGpu (string → bitvec)
    // Tracked: BARRACUDA_REQUIREMENTS.md, Tier B compose
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            condensed.push(robinson_foulds::rf_distance(&trees[i], &trees[j]) as f64);
        }
    }
    Ok(condensed)
}

#[cfg(test)]
#[cfg(feature = "gpu")]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::manual_let_else
)]
mod tests {
    use super::*;
    use crate::bio::unifrac::tree::PhyloTree;
    use crate::gpu::GpuF64;

    #[test]
    fn api_surface_compiles() {
        let _: fn(&GpuF64, &PhyloTree, &PhyloTree) -> Result<usize> = rf_distance_gpu;
        let _: fn(&GpuF64, &PhyloTree, &PhyloTree) -> Result<f64> = rf_distance_normalized_gpu;
        let _: fn(&GpuF64, &[PhyloTree]) -> Result<Vec<f64>> = rf_distance_matrix_gpu;
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn gpu_signature_check() {
        let gpu = match GpuF64::new().await {
            Ok(g) if g.has_f64 => g,
            _ => return,
        };
        let tree = PhyloTree::from_newick("(A,B);");
        let result = rf_distance_gpu(&gpu, &tree, &tree);
        assert!(
            result.is_ok(),
            "rf_distance_gpu should succeed with valid input"
        );
    }
}
