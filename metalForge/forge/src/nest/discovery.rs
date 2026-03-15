// SPDX-License-Identifier: AGPL-3.0-or-later

//! Socket discovery for `NestGate` Unix socket.
//!
//! Discovery order:
//! 1. `NESTGATE_SOCKET` environment variable (if path exists)
//! 2. `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock` (if exists)
//! 3. Platform-agnostic `temp_dir()/nestgate-default.sock` fallback

use std::path::PathBuf;

/// Default socket path when `NESTGATE_SOCKET` is not set.
///
/// Uses environment-based discovery: checks `NESTGATE_SOCKET` first,
/// then `$XDG_RUNTIME_DIR/biomeos/nestgate-default.sock`, then
/// platform-agnostic temp dir fallback. No hardcoded absolute paths.
#[must_use]
pub fn default_socket_path() -> PathBuf {
    if let Ok(path) = std::env::var("NESTGATE_SOCKET") {
        return PathBuf::from(path);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join("biomeos/nestgate-default.sock");
    }
    std::env::temp_dir().join("nestgate-default.sock")
}

/// Discover the `NestGate` Unix socket.
///
/// Public so `data.rs` can share the same discovery logic.
#[must_use]
pub fn discover_nestgate_socket() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("NESTGATE_SOCKET") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join("biomeos/nestgate-default.sock");
        if p.exists() {
            return Some(p);
        }
    }
    let fallback = default_socket_path();
    if fallback.exists() {
        return Some(fallback);
    }
    None
}
