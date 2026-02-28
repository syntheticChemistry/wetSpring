// SPDX-License-Identifier: AGPL-3.0-or-later
//! FST (Fixation Index) variance decomposition — Weir-Cockerham estimator.
//!
//! Re-exports from `barracuda::ops::bio::fst_variance` for discoverability.
//! CPU-only: computes Wright's F-statistics (FST, FIS, FIT) from allele
//! frequencies and population sample sizes.
//!
//! # References
//!
//! - Weir & Cockerham (1984) Evolution 38:1358-1370

pub use barracuda::ops::bio::fst_variance::{FstResult, fst_variance_decomposition};
