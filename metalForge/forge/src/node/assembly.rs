// SPDX-License-Identifier: AGPL-3.0-or-later

//! Assembly statistics computation — FASTA parsing, N50, GC, aggregation.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::types::{AssemblyStats, CollectionStats};

/// Compute statistics for a single assembly file (.fna.gz).
///
/// # Errors
///
/// Returns an error if the file cannot be read, decompressed, or parsed as
/// FASTA, or if it contains no sequences.
pub fn compute_assembly_stats_from_file(
    accession: &str,
    path: &Path,
) -> Result<AssemblyStats, crate::error::AssemblyError> {
    let sequences = read_fasta_gz(path)?;

    if sequences.is_empty() {
        return Err(crate::error::AssemblyError::NoSequences);
    }

    let mut contig_lengths: Vec<u64> = sequences.iter().map(|s| s.len() as u64).collect();
    contig_lengths.sort_unstable_by(|a, b| b.cmp(a));

    let total_length: u64 = contig_lengths.iter().sum();
    let largest_contig = contig_lengths[0];
    let n50 = compute_n50(&contig_lengths, total_length);

    let (gc_count, total_bases) = count_gc(&sequences);
    #[expect(
        clippy::cast_precision_loss,
        reason = "precision: index→f64 lossless for array sizes below 2^53"
    )]
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
#[expect(
    clippy::cast_precision_loss,
    reason = "precision: index→f64 lossless for array sizes below 2^53"
)]
pub fn aggregate_collection(dataset: &str, assemblies: Vec<AssemblyStats>) -> CollectionStats {
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

pub fn read_fasta_gz(path: &Path) -> Result<Vec<Vec<u8>>, crate::error::AssemblyError> {
    let output = std::process::Command::new("gzip")
        .args(["-dc", &path.display().to_string()])
        .output()
        .map_err(|e| crate::error::AssemblyError::ToolFailed(format!("gzip: {e}")))?;

    if !output.status.success() {
        return Err(crate::error::AssemblyError::ToolFailed(format!(
            "gzip -dc failed ({})",
            output.status.code().unwrap_or(-1)
        )));
    }

    let reader = BufReader::new(&output.stdout[..]);
    parse_fasta_sequences(reader)
}

pub fn parse_fasta_sequences<R: BufRead>(
    reader: R,
) -> Result<Vec<Vec<u8>>, crate::error::AssemblyError> {
    let mut sequences = Vec::new();
    let mut current = Vec::new();
    let mut in_seq = false;

    for line in reader.lines() {
        let line = line.map_err(|e| crate::error::AssemblyError::ReadLine(e.to_string()))?;
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

pub fn compute_n50(sorted_desc: &[u64], total: u64) -> u64 {
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

pub fn count_gc(sequences: &[Vec<u8>]) -> (u64, u64) {
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
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "precision: bounded f64→integer for display ranges"
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
pub fn list_assembly_files(dir: &Path) -> Result<Vec<PathBuf>, crate::error::AssemblyError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        crate::error::AssemblyError::ReadDir(format!("read dir {}: {e}", dir.display()))
    })?;

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
