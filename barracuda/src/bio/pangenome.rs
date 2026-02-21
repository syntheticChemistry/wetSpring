// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pangenome analysis: gene presence-absence and core/accessory partitioning.
//!
//! Constructs a binary presence-absence matrix from gene annotations across
//! multiple genomes, then partitions genes into core (present in all),
//! accessory (present in some), and unique (present in one) categories.
//! Fits Heap's law to estimate pangenome openness.
//!
//! Used in Moulana & Anderson 2020 (Sulfurovum pangenomics).

/// A gene cluster with its presence across genomes.
#[derive(Debug, Clone)]
pub struct GeneCluster {
    /// Cluster identifier (e.g., COG ID or representative gene).
    pub id: String,
    /// Presence in each genome (true = present).
    pub presence: Vec<bool>,
}

/// Result of pangenome analysis.
#[derive(Debug, Clone)]
pub struct PangenomeResult {
    /// Number of gene clusters in the core genome (present in all).
    pub core_size: usize,
    /// Number of gene clusters in the accessory genome (present in 2+ but not all).
    pub accessory_size: usize,
    /// Number of unique gene clusters (present in exactly one genome).
    pub unique_size: usize,
    /// Total number of gene clusters (core + accessory + unique).
    pub total_size: usize,
    /// Number of genomes.
    pub n_genomes: usize,
    /// Heap's law alpha parameter (< 1 → open pangenome).
    pub heaps_alpha: Option<f64>,
}

/// Analyze a pangenome from a presence-absence matrix.
///
/// Each `GeneCluster` has a boolean vector of length `n_genomes`.
#[must_use]
pub fn analyze(clusters: &[GeneCluster], n_genomes: usize) -> PangenomeResult {
    let mut core = 0;
    let mut accessory = 0;
    let mut unique = 0;

    for cluster in clusters {
        let count = cluster.presence.iter().filter(|&&p| p).count();
        if count == n_genomes {
            core += 1;
        } else if count == 1 {
            unique += 1;
        } else if count > 1 {
            accessory += 1;
        }
    }

    let heaps_alpha = fit_heaps_law(clusters, n_genomes);

    PangenomeResult {
        core_size: core,
        accessory_size: accessory,
        unique_size: unique,
        total_size: core + accessory + unique,
        n_genomes,
        heaps_alpha,
    }
}

/// GPU uniform parameters for batched pangenome dispatch.
///
/// Maps directly to WGSL `var<uniform>` for future GPU absorption.
/// The presence-absence matrix is stored as a flat `u8` array
/// (`n_genes * n_genomes`), row-major — one thread per gene for
/// core/accessory/unique classification.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PangenomeParams {
    /// Number of gene clusters.
    pub n_genes: u32,
    /// Number of genomes.
    pub n_genomes: u32,
}

/// Flat presence-absence matrix for GPU storage buffer binding.
///
/// Row-major `[n_genes × n_genomes]` packed as `u8` (0/1).
/// This is the GPU absorption target: one thread reads one row,
/// counts presence, classifies as core/accessory/unique.
#[must_use]
pub fn presence_matrix_flat(clusters: &[GeneCluster], n_genomes: usize) -> Vec<u8> {
    let mut flat = Vec::with_capacity(clusters.len() * n_genomes);
    for cluster in clusters {
        for g in 0..n_genomes {
            flat.push(u8::from(cluster.presence.get(g).copied().unwrap_or(false)));
        }
    }
    flat
}

/// Construct gene clusters from a raw presence-absence matrix.
///
/// `matrix` is `[n_genes x n_genomes]`, row-major. Values > 0 indicate presence.
#[must_use]
pub fn clusters_from_matrix(matrix: &[Vec<f64>], gene_names: &[String]) -> Vec<GeneCluster> {
    matrix
        .iter()
        .enumerate()
        .map(|(i, row)| GeneCluster {
            id: gene_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("gene_{i}")),
            presence: row.iter().map(|&v| v > 0.0).collect(),
        })
        .collect()
}

/// Fit Heap's law: `n(g) = κ * g^α` where `g` is number of genomes,
/// `n(g)` is total gene count. Alpha < 1 indicates open pangenome.
///
/// Uses simple linear regression on log-log data.
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn fit_heaps_law(clusters: &[GeneCluster], n_genomes: usize) -> Option<f64> {
    if n_genomes < 3 || clusters.is_empty() {
        return None;
    }

    // Simulate adding genomes one at a time
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut seen: Vec<bool> = vec![false; clusters.len()];

    for g in 0..n_genomes {
        for (i, cluster) in clusters.iter().enumerate() {
            if g < cluster.presence.len() && cluster.presence[g] && !seen[i] {
                seen[i] = true;
            }
        }
        let total = seen.iter().filter(|&&s| s).count();
        if total > 0 {
            points.push(((g + 1) as f64, total as f64));
        }
    }

    if points.len() < 2 {
        return None;
    }

    // log-log regression: ln(n) = ln(κ) + α * ln(g)
    let log_points: Vec<(f64, f64)> = points.iter().map(|(x, y)| (x.ln(), y.ln())).collect();

    let n = log_points.len() as f64;
    let sum_x: f64 = log_points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = log_points.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = log_points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = log_points.iter().map(|(x, _)| x * x).sum();

    // Linear regression denominator: n*Σ(x²) - (Σx)². Clippy
    // `suspicious_operation_groupings` is a false positive here.
    #[allow(clippy::suspicious_operation_groupings)]
    let denom = n.mul_add(sum_xx, -(sum_x * sum_x));
    if denom.abs() < 1e-15 {
        return None;
    }

    let alpha = n.mul_add(sum_xy, -(sum_x * sum_y)) / denom;
    Some(alpha)
}

/// Hypergeometric test for enrichment (Fisher exact approximation).
///
/// Tests whether `k` successes in `n` draws from a population of `big_n`
/// with `big_k` total successes is significant.
#[must_use]
pub fn hypergeometric_pvalue(k: usize, n: usize, big_k: usize, big_n: usize) -> f64 {
    if big_n == 0 || n == 0 || big_k == 0 {
        return 1.0;
    }

    let expected = n as f64 * big_k as f64 / big_n as f64;
    if (k as f64) <= expected {
        return 1.0;
    }

    // Normal approximation to hypergeometric
    let var = n as f64 * big_k as f64 * (big_n - big_k) as f64 * (big_n - n) as f64
        / (big_n as f64 * big_n as f64 * (big_n - 1).max(1) as f64);

    if var <= 0.0 {
        return if k as f64 > expected { 0.0 } else { 1.0 };
    }

    let z = (k as f64 - expected) / var.sqrt();
    1.0 - normal_cdf(z)
}

/// Benjamini-Hochberg FDR correction.
#[must_use]
pub fn benjamini_hochberg(pvalues: &[f64]) -> Vec<f64> {
    let n = pvalues.len();
    if n == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f64)> = pvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0; n];
    let mut cummin = f64::INFINITY;
    for i in (0..n).rev() {
        let rank = i + 1;
        let adj = indexed[i].1 * n as f64 / rank as f64;
        cummin = cummin.min(adj).min(1.0);
        adjusted[indexed[i].0] = cummin;
    }
    adjusted
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = ((a5 * t + a4).mul_add(t, a3).mul_add(t, a2).mul_add(t, a1) * t)
        .mul_add(-(-x * x).exp(), 1.0);
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_clusters() -> Vec<GeneCluster> {
        vec![
            GeneCluster {
                id: "core1".into(),
                presence: vec![true, true, true],
            },
            GeneCluster {
                id: "core2".into(),
                presence: vec![true, true, true],
            },
            GeneCluster {
                id: "acc1".into(),
                presence: vec![true, true, false],
            },
            GeneCluster {
                id: "unique1".into(),
                presence: vec![true, false, false],
            },
            GeneCluster {
                id: "unique2".into(),
                presence: vec![false, false, true],
            },
        ]
    }

    #[test]
    fn core_accessory_unique() {
        let clusters = sample_clusters();
        let result = analyze(&clusters, 3);
        assert_eq!(result.core_size, 2);
        assert_eq!(result.accessory_size, 1);
        assert_eq!(result.unique_size, 2);
        assert_eq!(result.total_size, 5);
    }

    #[test]
    fn all_core() {
        let clusters = vec![
            GeneCluster {
                id: "g1".into(),
                presence: vec![true, true],
            },
            GeneCluster {
                id: "g2".into(),
                presence: vec![true, true],
            },
        ];
        let result = analyze(&clusters, 2);
        assert_eq!(result.core_size, 2);
        assert_eq!(result.accessory_size, 0);
        assert_eq!(result.unique_size, 0);
    }

    #[test]
    fn heaps_law_computed() {
        let clusters = sample_clusters();
        let result = analyze(&clusters, 3);
        assert!(result.heaps_alpha.is_some());
    }

    #[test]
    fn matrix_construction() {
        let matrix = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
        let names = vec!["geneA".into(), "geneB".into()];
        let clusters = clusters_from_matrix(&matrix, &names);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].id, "geneA");
        assert!(clusters[0].presence[0]);
        assert!(!clusters[0].presence[1]);
    }

    #[test]
    fn hypergeometric_enriched() {
        let p = hypergeometric_pvalue(8, 10, 20, 100);
        assert!(p < 0.05);
    }

    #[test]
    fn hypergeometric_not_enriched() {
        let p = hypergeometric_pvalue(2, 10, 20, 100);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bh_correction_monotonic() {
        let pvals = vec![0.01, 0.04, 0.03, 0.5];
        let adj = benjamini_hochberg(&pvals);
        assert!(adj.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert!(adj[0] <= adj[3]);
    }

    #[test]
    fn bh_empty() {
        let adj = benjamini_hochberg(&[]);
        assert!(adj.is_empty());
    }
}
