// SPDX-License-Identifier: AGPL-3.0-or-later
//! IPC server for wetSpring as a biomeOS science primal.
//!
//! Implements the Primal IPC Protocol v2.0 — JSON-RPC 2.0, newline-delimited,
//! over Unix domain socket. Each registered capability maps to existing
//! barracuda library functions; no math is duplicated.
//!
//! When only the `json` feature is enabled (without `ipc`), this module exposes
//! the shared [`stream_item::StreamItem`] protocol type for pipeline NDJSON.
//!
//! # Capabilities
//!
//! | Method                          | Description                                  |
//! |---------------------------------|----------------------------------------------|
//! | `health.check`                  | Full health/readiness probe (legacy)         |
//! | `health.liveness`               | Minimal liveness probe (`alive: true`)       |
//! | `health.readiness`              | Deep readiness probe with subsystem status   |
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

#[cfg(feature = "json")]
pub mod stream_item;

#[cfg(feature = "ipc")]
pub mod capability_domains;
#[cfg(feature = "ipc")]
pub mod compute_dispatch;
#[cfg(feature = "ipc")]
pub mod discover;
#[cfg(feature = "ipc")]
pub mod dispatch;
#[cfg(feature = "ipc")]
pub mod dispatch_strategy;
#[cfg(feature = "ipc")]
pub mod handlers;
#[cfg(feature = "ipc")]
pub mod mcp;
#[cfg(feature = "ipc")]
pub mod message;
#[cfg(feature = "ipc")]
pub mod metrics;
#[cfg(feature = "ipc")]
pub mod performance_surface;
/// Re-exports primal name constants for `use crate::ipc::primal_names::*` in graph binaries.
#[cfg(any(feature = "ipc", feature = "json"))]
pub mod primal_names;
#[cfg(feature = "ipc")]
pub mod protocol;
#[cfg(feature = "ipc")]
pub mod provenance;
#[cfg(feature = "ipc")]
pub mod resilience;
#[cfg(feature = "ipc")]
pub mod server;
#[cfg(feature = "ipc")]
pub mod songbird;
#[cfg(feature = "ipc")]
pub mod sweetgrass;
#[cfg(feature = "ipc")]
pub mod timeseries;
#[cfg(feature = "ipc")]
pub mod transport;

#[cfg(feature = "ipc")]
pub use server::Server;

#[cfg(all(test, feature = "ipc"))]
use std::path::{Path, PathBuf};

/// Create a unique socket path for this test, isolated from other tests.
/// Uses the system temp directory + test-specific suffix to prevent collisions.
#[cfg(all(test, feature = "ipc"))]
#[must_use]
pub fn test_socket_path(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("wetspring-test");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{test_name}-{}.sock", std::process::id()))
}

/// Best-effort removal of a test socket path (see [`test_socket_path`]).
///
/// Call before bind to clear stale files from interrupted runs, and after tests
/// that create a socket inode. Ignores errors (e.g. server still listening).
#[cfg(all(test, feature = "ipc"))]
pub fn cleanup_test_socket(path: &Path) {
    let _ = std::fs::remove_file(path);
}
