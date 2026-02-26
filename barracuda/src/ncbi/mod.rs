// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared NCBI Entrez helpers for validation binaries.
//!
//! Provides API-key discovery, HTTP GET, Entrez E-search/E-fetch wrappers,
//! SRA download, and integrity-verified caching. Validation binaries that
//! query NCBI share this module instead of duplicating boilerplate.
//!
//! # Submodules
//!
//! - `api_key` — Capability-based API key cascade (env → data root → XDG)
//! - [`http`] — Runtime-discovered HTTP transport (curl / wget / custom)
//! - `entrez` — Entrez E-search XML query and response parsing
//! - `efetch` — Entrez E-fetch for FASTA/`GenBank` sequence download
//! - `sra` — SRA run download via capability-discovered `fasterq-dump` / `fastq-dump`
//! - [`nestgate`] — Optional `NestGate` data provider (JSON-RPC over Unix socket)
//! - `cache` — Capability-based cache with accession storage and SHA-256 integrity

mod api_key;
mod cache;
mod efetch;
mod entrez;
pub mod http;
pub mod nestgate;
mod sra;

pub use api_key::api_key;
pub use cache::{accession_dir, cache_file, verify_integrity, write_with_integrity};
pub use efetch::{efetch_fasta, efetch_fasta_batch, efetch_genbank};
pub use entrez::esearch_count;
pub use http::get as http_get;
pub use sra::{download_sra_run, sra_tools_available};
