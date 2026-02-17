// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bioinformatics and analytical chemistry algorithms.

pub mod derep;
pub mod diversity;
#[cfg(feature = "gpu")]
pub mod diversity_gpu;
pub mod eic;
pub mod feature_table;
pub mod kmd;
pub mod kmer;
#[cfg(feature = "gpu")]
pub mod kriging;
pub mod merge_pairs;
pub mod pcoa;
#[cfg(feature = "gpu")]
pub mod pcoa_gpu;
pub mod quality;
pub mod signal;
pub mod spectral_match;
#[cfg(feature = "gpu")]
pub mod spectral_match_gpu;
pub mod tolerance_search;
