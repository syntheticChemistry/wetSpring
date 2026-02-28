// SPDX-License-Identifier: AGPL-3.0-or-later

//! Node atomic compute pipeline — Tower + Nest + compute dispatch.
//!
//! Coordinates the full NUCLEUS data flow:
//! 1. **Tower** discovers compute substrates (local + mesh via Songbird)
//! 2. **Nest** provides data (local dir / `NestGate` cache / NCBI fetch)
//! 3. **Node** dispatches computation to the best available substrate
//!
//! # Assembly Statistics
//!
//! Core genomics computations on NCBI assembly collections:
//! - **N50**: Assembly contiguity metric
//! - **GC content**: Guanine-cytosine fraction
//! - **Genome size**: Total assembly length
//! - **Diversity**: Shannon entropy across assembly GC distribution
//!
//! All math uses `f64` for scientific precision, compatible with
//! barracuda GPU promotion via `FusedMapReduceF64`.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::data::{self, DataSource};
use crate::dispatch::{self, Reason, Workload};
use crate::inventory;
use crate::substrate::{Capability, SubstrateKind};
#[cfg(test)]
use wetspring_barracuda::tolerances;

/// Assembly statistics computed from a genome FASTA.
#[derive(Debug, Clone)]
pub struct AssemblyStats {
    /// Assembly accession.
    pub accession: String,
    /// Number of contigs/scaffolds.
    pub num_contigs: usize,
    /// Total assembly length in base pairs.
    pub total_length: u64,
    /// N50 value (50% of assembly in contigs this size or larger).
    pub n50: u64,
    /// GC content as a fraction (0.0 to 1.0).
    pub gc_content: f64,
    /// Largest contig length.
    pub largest_contig: u64,
}

/// Collection-level statistics across multiple assemblies.
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Dataset name.
    pub dataset: String,
    /// Number of assemblies analyzed.
    pub assembly_count: usize,
    /// Mean genome size across assemblies.
    pub mean_genome_size: f64,
    /// Standard deviation of genome sizes.
    pub genome_size_std: f64,
    /// Mean GC content across assemblies.
    pub mean_gc: f64,
    /// Standard deviation of GC content.
    pub gc_std: f64,
    /// Mean N50 across assemblies.
    pub mean_n50: f64,
    /// Shannon entropy of the GC distribution (binned).
    pub gc_shannon_entropy: f64,
    /// Individual assembly stats.
    pub assemblies: Vec<AssemblyStats>,
}

/// Result of a Node compute pipeline run.
#[derive(Debug)]
pub struct PipelineResult {
    /// Data source used.
    pub data_source: DataSource,
    /// Compute substrate chosen.
    pub substrate_kind: SubstrateKind,
    /// Dispatch reason.
    pub dispatch_reason: Reason,
    /// Computed collection statistics.
    pub stats: CollectionStats,
}

/// Compute assembly statistics for a dataset using the full NUCLEUS pipeline.
///
/// 1. Resolves the dataset via the three-tier chain (env / `NestGate` / synthetic)
/// 2. Discovers compute substrates via Tower (local + Songbird mesh)
/// 3. Routes the computation workload to the best substrate
/// 4. Computes statistics on the resolved assemblies
///
/// # Errors
///
/// Returns an error if no capable substrate is found, the dataset has no local
/// path, or assembly file parsing fails.
pub fn compute_assembly_stats(dataset: &str) -> Result<PipelineResult, String> {
    let resolution = data::resolve_dataset(dataset);

    let substrates = inventory::discover_with_tower();

    let workload = Workload::new(
        format!("{dataset}_assembly_stats"),
        vec![Capability::F64Compute],
    );

    let decision = dispatch::route_bandwidth_aware(&workload, &substrates)
        .ok_or_else(|| "no capable substrate for f64 assembly stats".to_string())?;

    let stats = match &resolution.path {
        Some(dir) if dir.is_dir() => compute_collection_from_dir(dataset, dir)?,
        _ => {
            return Err(format!(
                "dataset {dataset} resolved as {:?} but no local path available for compute",
                resolution.source
            ));
        }
    };

    Ok(PipelineResult {
        data_source: resolution.source,
        substrate_kind: decision.substrate.kind,
        dispatch_reason: decision.reason,
        stats,
    })
}

/// Compute collection statistics from a directory of `.fna.gz` files.
///
/// # Errors
///
/// Returns an error if the directory cannot be read, contains no `.fna.gz`
/// files, or all assemblies fail to parse.
pub fn compute_collection_from_dir(dataset: &str, dir: &Path) -> Result<CollectionStats, String> {
    let entries = list_assembly_files(dir)?;
    if entries.is_empty() {
        return Err(format!("no .fna.gz files in {}", dir.display()));
    }

    let mut assemblies = Vec::with_capacity(entries.len());
    for path in &entries {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let accession = stem.strip_suffix(".fna").unwrap_or(stem).to_string();

        match compute_assembly_stats_from_file(&accession, path) {
            Ok(stats) => assemblies.push(stats),
            Err(e) => {
                eprintln!("[node] skip {}: {e}", path.display());
            }
        }
    }

    if assemblies.is_empty() {
        return Err("all assemblies failed to parse".to_string());
    }

    Ok(aggregate_collection(dataset, assemblies))
}

/// Compute statistics for a single assembly file (.fna.gz).
///
/// # Errors
///
/// Returns an error if the file cannot be read, decompressed, or parsed as
/// FASTA, or if it contains no sequences.
pub fn compute_assembly_stats_from_file(
    accession: &str,
    path: &Path,
) -> Result<AssemblyStats, String> {
    let sequences = read_fasta_gz(path)?;

    if sequences.is_empty() {
        return Err("no sequences in FASTA".to_string());
    }

    #[allow(clippy::cast_possible_truncation)]
    let mut contig_lengths: Vec<u64> = sequences.iter().map(|s| s.len() as u64).collect();
    contig_lengths.sort_unstable_by(|a, b| b.cmp(a));

    let total_length: u64 = contig_lengths.iter().sum();
    let largest_contig = contig_lengths[0];
    let n50 = compute_n50(&contig_lengths, total_length);

    let (gc_count, total_bases) = count_gc(&sequences);
    #[allow(clippy::cast_precision_loss)]
    let gc_content = if total_bases > 0 {
        gc_count as f64 / total_bases as f64
    } else {
        0.0
    };

    Ok(AssemblyStats {
        accession: accession.to_string(),
        num_contigs: contig_lengths.len(),
        total_length,
        n50,
        gc_content,
        largest_contig,
    })
}

/// Aggregate individual assembly stats into collection-level metrics.
#[allow(clippy::cast_precision_loss)]
fn aggregate_collection(dataset: &str, assemblies: Vec<AssemblyStats>) -> CollectionStats {
    let sizes: Vec<f64> = assemblies.iter().map(|a| a.total_length as f64).collect();
    let gcs: Vec<f64> = assemblies.iter().map(|a| a.gc_content).collect();
    let n50s: Vec<f64> = assemblies.iter().map(|a| a.n50 as f64).collect();

    let mean_genome_size = mean(&sizes);
    let genome_size_std = std_dev(&sizes, mean_genome_size);
    let mean_gc = mean(&gcs);
    let gc_std = std_dev(&gcs, mean_gc);
    let mean_n50 = mean(&n50s);
    let gc_shannon_entropy = shannon_entropy_binned(&gcs, 20);

    CollectionStats {
        dataset: dataset.to_string(),
        assembly_count: assemblies.len(),
        mean_genome_size,
        genome_size_std,
        mean_gc,
        gc_std,
        mean_n50,
        gc_shannon_entropy,
        assemblies,
    }
}

// ── FASTA gz reader ─────────────────────────────────────────────────

fn read_fasta_gz(path: &Path) -> Result<Vec<Vec<u8>>, String> {
    let output = std::process::Command::new("gzip")
        .args(["-dc", &path.display().to_string()])
        .output()
        .map_err(|e| format!("gzip: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "gzip -dc failed ({})",
            output.status.code().unwrap_or(-1)
        ));
    }

    let reader = BufReader::new(&output.stdout[..]);
    parse_fasta_sequences(reader)
}

fn parse_fasta_sequences<R: BufRead>(reader: R) -> Result<Vec<Vec<u8>>, String> {
    let mut sequences = Vec::new();
    let mut current = Vec::new();
    let mut in_seq = false;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read line: {e}"))?;
        if line.starts_with('>') {
            if in_seq && !current.is_empty() {
                sequences.push(std::mem::take(&mut current));
            }
            in_seq = true;
        } else if in_seq {
            current.extend(line.trim().as_bytes());
        }
    }
    if !current.is_empty() {
        sequences.push(current);
    }

    Ok(sequences)
}

// ── Statistics ───────────────────────────────────────────────────────

fn compute_n50(sorted_desc: &[u64], total: u64) -> u64 {
    let half = total / 2;
    let mut cumulative = 0u64;
    for &len in sorted_desc {
        cumulative += len;
        if cumulative >= half {
            return len;
        }
    }
    sorted_desc.last().copied().unwrap_or(0)
}

fn count_gc(sequences: &[Vec<u8>]) -> (u64, u64) {
    let mut gc = 0u64;
    let mut total = 0u64;
    for seq in sequences {
        for &b in seq {
            match b {
                b'G' | b'g' | b'C' | b'c' => {
                    gc += 1;
                    total += 1;
                }
                b'A' | b'a' | b'T' | b't' => {
                    total += 1;
                }
                b'N' | b'n' => total += 1,
                _ => {}
            }
        }
    }
    (gc, total)
}

fn mean(values: &[f64]) -> f64 {
    barracuda::stats::mean(values)
}

fn std_dev(values: &[f64], _mean_val: f64) -> f64 {
    barracuda::stats::correlation::std_dev(values).unwrap_or(0.0)
}

/// Shannon entropy of a distribution binned into `n_bins` equal-width bins.
///
/// Used to quantify GC content diversity across an assembly collection.
/// Higher entropy = more diverse GC distribution.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn shannon_entropy_binned(values: &[f64], n_bins: usize) -> f64 {
    if values.is_empty() || n_bins == 0 {
        return 0.0;
    }

    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range < f64::EPSILON {
        return 0.0;
    }

    let mut bins = vec![0u64; n_bins];
    let bin_width = range / n_bins as f64;

    for &v in values {
        let idx = ((v - min_val) / bin_width) as usize;
        let idx = idx.min(n_bins - 1);
        bins[idx] += 1;
    }

    let n = values.len() as f64;
    let mut entropy = 0.0;
    for &count in &bins {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// List `.fna.gz` assembly files in a directory.
///
/// # Errors
///
/// Returns an error if the directory cannot be read.
pub fn list_assembly_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read dir {}: {e}", dir.display()))?;

    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".fna.gz"))
        })
        .collect();

    paths.sort();
    Ok(paths)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn n50_basic() {
        let contigs = vec![100, 80, 50, 30, 20, 10];
        let total: u64 = contigs.iter().sum();
        assert_eq!(compute_n50(&contigs, total), 80);
    }

    #[test]
    fn n50_single_contig() {
        assert_eq!(compute_n50(&[1000], 1000), 1000);
    }

    #[test]
    fn n50_empty() {
        assert_eq!(compute_n50(&[], 0), 0);
    }

    #[test]
    fn gc_count_basic() {
        let seqs = vec![b"ATGCGC".to_vec()];
        let (gc, total) = count_gc(&seqs);
        assert_eq!(gc, 4);
        assert_eq!(total, 6);
    }

    #[test]
    fn gc_handles_n_bases() {
        let seqs = vec![b"ATGCN".to_vec()];
        let (gc, total) = count_gc(&seqs);
        assert_eq!(gc, 2);
        assert_eq!(total, 5);
    }

    #[test]
    fn gc_case_insensitive() {
        let seqs = vec![b"atgcGC".to_vec()];
        let (gc, total) = count_gc(&seqs);
        assert_eq!(gc, 4);
        assert_eq!(total, 6);
    }

    #[test]
    fn mean_basic() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_empty() {
        assert!((mean(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn std_dev_basic() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let m = mean(&values);
        let sd = std_dev(&values, m);
        // Sample std dev (N-1 denominator): sqrt(32/7) ≈ 2.138
        assert!(
            (sd - 2.138).abs() < tolerances::ODE_STEADY_STATE,
            "sample std dev: {sd}"
        );
    }

    #[test]
    fn std_dev_single() {
        assert!((std_dev(&[42.0], 42.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn shannon_entropy_uniform() {
        let values: Vec<f64> = (0..100).map(|i| f64::from(i) / 100.0).collect();
        let h = shannon_entropy_binned(&values, 10);
        assert!(
            h > 2.0,
            "uniform distribution should have high entropy, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_constant() {
        let values = vec![0.5; 100];
        let h = shannon_entropy_binned(&values, 10);
        assert!(
            h.abs() < f64::EPSILON,
            "constant values should have zero entropy"
        );
    }

    #[test]
    fn shannon_entropy_empty() {
        assert!((shannon_entropy_binned(&[], 10) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_fasta_basic() {
        let fasta = b">seq1\nATGC\nATGC\n>seq2\nGGGG\n";
        let reader = BufReader::new(&fasta[..]);
        let seqs = parse_fasta_sequences(reader).unwrap();
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[0], b"ATGCATGC");
        assert_eq!(seqs[1], b"GGGG");
    }

    #[test]
    fn parse_fasta_empty() {
        let reader = BufReader::new(&b""[..]);
        let seqs = parse_fasta_sequences(reader).unwrap();
        assert!(seqs.is_empty());
    }

    #[test]
    fn aggregate_collection_basic() {
        let assemblies = vec![
            AssemblyStats {
                accession: "A".to_string(),
                num_contigs: 10,
                total_length: 5_000_000,
                n50: 100_000,
                gc_content: 0.45,
                largest_contig: 500_000,
            },
            AssemblyStats {
                accession: "B".to_string(),
                num_contigs: 20,
                total_length: 4_000_000,
                n50: 80_000,
                gc_content: 0.50,
                largest_contig: 300_000,
            },
        ];

        let coll = aggregate_collection("test", assemblies);
        assert_eq!(coll.assembly_count, 2);
        assert!((coll.mean_gc - 0.475).abs() < tolerances::ODE_STEADY_STATE);
        assert!((coll.mean_genome_size - 4_500_000.0).abs() < 1.0);
    }

    #[test]
    fn list_assembly_files_handles_missing_dir() {
        let result = list_assembly_files(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }
}
