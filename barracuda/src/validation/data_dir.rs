// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based data directory resolution for validation binaries.
//!
//! Primal code has no hardcoded paths — data locations are discovered at runtime
//! via environment variables and standard conventions.

use std::path::PathBuf;

/// Discover the benchmark results directory via capability-based discovery.
///
/// Discovery order:
/// 1. `WETSPRING_BENCH_DIR` env var (explicit override)
/// 2. `{CARGO_MANIFEST_DIR}/../benchmarks/results` for development
/// 3. `benchmarks/results` relative to cwd for deployment
#[must_use]
pub fn discover_bench_dir() -> PathBuf {
    data_dir("WETSPRING_BENCH_DIR", "benchmarks/results")
}

/// Resolve a data directory using a cascading discovery strategy.
///
/// Implements capability-based discovery: primal code has no hardcoded paths and
/// discovers data at runtime via explicit configuration or environment.
///
/// # Discovery cascade (in order)
///
/// 1. **Explicit env var** — If `env_var` is set, use it. Overrides everything.
/// 2. **General data root** — If `WETSPRING_DATA_ROOT` is set and
///    `WETSPRING_DATA_ROOT/{default_subpath}` exists, use it.
/// 3. **Development fallback** — If `CARGO_MANIFEST_DIR/../{default_subpath}` exists,
///    use it.
/// 4. **Deployment fallback** — Otherwise return `default_subpath` (relative to cwd).
#[must_use]
pub fn data_dir(env_var: &str, default_subpath: &str) -> PathBuf {
    let specific = std::env::var(env_var).ok();
    let data_root = std::env::var("WETSPRING_DATA_ROOT").ok();
    resolve_data_dir(specific.as_deref(), data_root.as_deref(), default_subpath)
}

/// Pure logic for data directory resolution — no global state access.
///
/// Takes pre-read environment values so it can be tested without mutating
/// process-wide environment variables (which is `unsafe` in edition 2024).
#[must_use]
pub fn resolve_data_dir(
    specific_override: Option<&str>,
    data_root: Option<&str>,
    default_subpath: &str,
) -> PathBuf {
    if let Some(dir) = specific_override {
        return PathBuf::from(dir);
    }
    if let Some(root) = data_root {
        let p = std::path::Path::new(root).join(default_subpath);
        if p.exists() {
            return p;
        }
    }
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(default_subpath);
    if manifest.exists() {
        return manifest;
    }
    PathBuf::from(default_subpath)
}
