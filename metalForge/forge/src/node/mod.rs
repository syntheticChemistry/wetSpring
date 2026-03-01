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

mod assembly;
mod types;

use std::path::Path;

use crate::data;
use crate::dispatch::{self, Workload};
use crate::inventory;
use crate::substrate::Capability;

pub use assembly::{compute_assembly_stats_from_file, list_assembly_files, shannon_entropy_binned};
pub use types::{AssemblyStats, CollectionStats, PipelineResult};

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
    let entries = assembly::list_assembly_files(dir)?;
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

        match assembly::compute_assembly_stats_from_file(&accession, path) {
            Ok(stats) => assemblies.push(stats),
            Err(e) => {
                eprintln!("[node] skip {}: {e}", path.display());
            }
        }
    }

    if assemblies.is_empty() {
        return Err("all assemblies failed to parse".to_string());
    }

    Ok(assembly::aggregate_collection(dataset, assemblies))
}

#[cfg(test)]
mod tests;
