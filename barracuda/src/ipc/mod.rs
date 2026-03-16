// SPDX-License-Identifier: AGPL-3.0-or-later
//! IPC server for wetSpring as a biomeOS science primal.
//!
//! Implements the Primal IPC Protocol v2.0 — JSON-RPC 2.0, newline-delimited,
//! over Unix domain socket. Each registered capability maps to existing
//! barracuda library functions; no math is duplicated.
//!
//! # Capabilities
//!
//! | Method                          | Description                                  |
//! |---------------------------------|----------------------------------------------|
//! | `health.check`                  | Health/readiness probe                       |
//! | `science.diversity`             | Alpha diversity metrics (Shannon, Simpson, …) |
//! | `science.anderson`              | Anderson spectral analysis (GPU-preferred)    |
//! | `science.qs_model`              | QS/c-di-GMP biofilm ODE integration          |
//! | `science.ncbi_fetch`            | NCBI sequence retrieval                      |
//! | `science.full_pipeline`         | Chains all above stages                      |
//! | `science.kinetics`              | Biogas production models (Gompertz, etc.)    |
//! | `science.alignment`             | Smith-Waterman local alignment               |
//! | `science.taxonomy`              | Naive Bayes classification (RDP-style)       |
//! | `science.phylogenetics`         | Robinson-Foulds tree distance                |
//! | `science.nmf`                   | Non-negative Matrix Factorization            |
//! | `science.timeseries`            | Cross-spring time series analysis            |
//! | `science.timeseries_diversity`  | Diversity on time series abundances          |
//! | `provenance.begin`              | Start provenance-tracked session             |
//! | `provenance.record`             | Record step in provenance DAG                |
//! | `provenance.complete`           | Dehydrate → commit → attribute               |
//!
//! # Socket discovery
//!
//! The server binds to:
//! 1. `WETSPRING_SOCKET` env var (explicit override)
//! 2. `$XDG_RUNTIME_DIR/biomeos/wetspring-default.sock`
//! 3. `<temp_dir>/wetspring-default.sock` (platform-agnostic fallback)
//!
//! # Design
//!
//! - **No cross-embedding**: wetSpring does not import biomeOS code — it
//!   implements the standard JSON-RPC protocol independently.
//! - **Sovereign fallback**: wetSpring works standalone; biomeOS is additive.
//! - **Existing math unchanged**: the server is a thin JSON-RPC wrapper over
//!   validated barracuda library functions.

pub mod capability_domains;
pub mod discover;
pub mod dispatch;
pub mod handlers;
pub mod metrics;
pub mod primal_names;
pub mod protocol;
pub mod provenance;
pub mod server;
pub mod songbird;
pub mod timeseries;
pub mod transport;

pub use server::Server;
