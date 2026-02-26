// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared NCBI Entrez helpers for validation binaries.
//!
//! Provides API-key discovery, HTTP GET, and Entrez E-search wrappers.
//! Validation binaries that query NCBI share this module instead of
//! duplicating the same boilerplate.
//!
//! # Submodules
//!
//! - `api_key` — Capability-based API key cascade (env → data root → XDG)
//! - [`http`] — Runtime-discovered HTTP transport (curl / wget / custom)
//! - `entrez` — Entrez E-search XML query and response parsing
//! - `cache` — Capability-based cache file path resolution

mod api_key;
mod cache;
mod entrez;
pub mod http;

pub use api_key::api_key;
pub use cache::cache_file;
pub use entrez::esearch_count;
pub use http::get as http_get;
