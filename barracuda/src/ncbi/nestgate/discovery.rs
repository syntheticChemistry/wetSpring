// SPDX-License-Identifier: AGPL-3.0-or-later
//! Socket discovery for `NestGate` and biomeOS.

use std::path::PathBuf;

/// Default `NestGate` socket path under `XDG_RUNTIME_DIR` (`biomeos/nestgate-default.sock`).
pub(super) const DEFAULT_NESTGATE_PATH_XDG: &str = "biomeos/nestgate-default.sock";
/// Fallback `NestGate` socket filename when `XDG_RUNTIME_DIR` is unset (`nestgate-default.sock`).
pub(super) const DEFAULT_NESTGATE_PATH_FALLBACK: &str = "nestgate-default.sock";
/// Default `biomeOS` socket path under `XDG_RUNTIME_DIR` (`biomeos/biomeos-default.sock`).
pub(super) const DEFAULT_BIOMEOS_PATH_XDG: &str = "biomeos/biomeos-default.sock";
/// Fallback `biomeOS` socket filename when `XDG_RUNTIME_DIR` is unset (`biomeos-default.sock`).
pub(super) const DEFAULT_BIOMEOS_PATH_FALLBACK: &str = "biomeos-default.sock";

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
    let explicit = std::env::var("NESTGATE_SOCKET").ok();
    let xdg = std::env::var("XDG_RUNTIME_DIR").ok();
    resolve_socket(explicit.as_deref(), xdg.as_deref())
}

/// Pure-logic socket path resolution.
#[must_use]
pub fn resolve_socket(explicit: Option<&str>, xdg_runtime: Option<&str>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(xdg) = xdg_runtime {
        let p = PathBuf::from(xdg).join(DEFAULT_NESTGATE_PATH_XDG);
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(DEFAULT_NESTGATE_PATH_FALLBACK);
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

/// Discover the biomeOS Neural API socket for capability-based routing.
///
/// 1. `BIOMEOS_SOCKET` env var (explicit override)
/// 2. `$XDG_RUNTIME_DIR/biomeos/biomeos-default.sock`
/// 3. `<temp_dir>/biomeos-default.sock`
#[must_use]
pub fn discover_biomeos_socket() -> Option<PathBuf> {
    let explicit = std::env::var("BIOMEOS_SOCKET").ok();
    let xdg = std::env::var("XDG_RUNTIME_DIR").ok();
    resolve_biomeos_socket(explicit.as_deref(), xdg.as_deref())
}

/// Pure-logic biomeOS socket path resolution.
#[must_use]
pub fn resolve_biomeos_socket(
    explicit: Option<&str>,
    xdg_runtime: Option<&str>,
) -> Option<PathBuf> {
    if let Some(path) = explicit {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(xdg) = xdg_runtime {
        let p = PathBuf::from(xdg).join(DEFAULT_BIOMEOS_PATH_XDG);
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(DEFAULT_BIOMEOS_PATH_FALLBACK);
    if fallback.exists() {
        return Some(fallback);
    }

    None
}
