// SPDX-License-Identifier: AGPL-3.0-or-later
//! Socket discovery for `NestGate` and biomeOS.
//!
//! Delegates to the unified [`crate::ipc::discover`] module when the `ipc`
//! feature is enabled; otherwise falls back to a minimal standalone resolver
//! so `ncbi` works without pulling in the full IPC subsystem.

use std::path::PathBuf;

/// Whether `NestGate` routing is enabled via environment.
#[must_use]
pub fn is_enabled() -> bool {
    std::env::var("WETSPRING_DATA_PROVIDER")
        .is_ok_and(|v| v.trim().eq_ignore_ascii_case("nestgate"))
}

/// Discover the `NestGate` Unix socket path.
///
/// Capability-based discovery (no hardcoded absolute paths):
/// 1. `NESTGATE_SOCKET` env var (explicit override)
/// 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`
/// 3. `<temp_dir>/nestgate-default.sock` (platform-agnostic fallback)
#[must_use]
pub fn discover_socket() -> Option<PathBuf> {
    discover_primal_socket("NESTGATE_SOCKET", "nestgate")
}

/// Discover the biomeOS Neural API socket for capability-based routing.
///
/// 1. `BIOMEOS_SOCKET` env var (explicit override)
/// 2. `$XDG_RUNTIME_DIR/biomeos/biomeos-default.sock`
/// 3. `<temp_dir>/biomeos-default.sock`
#[must_use]
pub fn discover_biomeos_socket() -> Option<PathBuf> {
    discover_primal_socket("BIOMEOS_SOCKET", "biomeos")
}

/// Pure-logic socket path resolution for testing.
///
/// Accepts pre-resolved env values so tests don't pollute the environment.
#[cfg(test)]
#[must_use]
pub fn resolve_socket(explicit: Option<&str>, xdg_runtime: Option<&str>) -> Option<PathBuf> {
    resolve_primal_socket(explicit, xdg_runtime, "nestgate")
}

/// Pure-logic biomeOS socket path resolution for testing.
#[cfg(test)]
#[must_use]
pub fn resolve_biomeos_socket(
    explicit: Option<&str>,
    xdg_runtime: Option<&str>,
) -> Option<PathBuf> {
    resolve_primal_socket(explicit, xdg_runtime, "biomeos")
}

// ── Unified implementation ──────────────────────────────────────────────

fn discover_primal_socket(env_var: &str, primal: &str) -> Option<PathBuf> {
    #[cfg(feature = "ipc")]
    {
        crate::ipc::discover::discover_socket(env_var, primal)
    }
    #[cfg(not(feature = "ipc"))]
    {
        discover_standalone(env_var, primal)
    }
}

#[cfg(test)]
fn resolve_primal_socket(
    explicit: Option<&str>,
    xdg_runtime: Option<&str>,
    primal: &str,
) -> Option<PathBuf> {
    let xdg_subpath = format!("biomeos/{primal}-default.sock");
    let fallback_name = format!("{primal}-default.sock");

    #[cfg(feature = "ipc")]
    {
        crate::ipc::discover::resolve_socket_explicit(
            explicit,
            xdg_runtime,
            &xdg_subpath,
            &fallback_name,
        )
    }
    #[cfg(not(feature = "ipc"))]
    {
        resolve_standalone(explicit, xdg_runtime, &xdg_subpath, &fallback_name)
    }
}

/// Standalone discovery when the `ipc` feature is not enabled.
#[cfg(not(feature = "ipc"))]
fn discover_standalone(env_var: &str, primal: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_var) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join(format!("biomeos/{primal}-default.sock"));
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(format!("{primal}-default.sock"));
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

/// Standalone explicit resolution when the `ipc` feature is not enabled.
#[cfg(all(test, not(feature = "ipc")))]
fn resolve_standalone(
    explicit: Option<&str>,
    xdg_runtime: Option<&str>,
    xdg_subpath: &str,
    fallback_name: &str,
) -> Option<PathBuf> {
    if let Some(path) = explicit {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(xdg) = xdg_runtime {
        let p = PathBuf::from(xdg).join(xdg_subpath);
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(fallback_name);
    if fallback.exists() {
        return Some(fallback);
    }

    None
}
