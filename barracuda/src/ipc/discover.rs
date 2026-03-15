// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based socket discovery for biomeOS primals.
//!
//! Resolves Unix domain socket paths using a cascading strategy:
//! 1. Explicit env var override (e.g. `WETSPRING_SOCKET`)
//! 2. `$XDG_RUNTIME_DIR/biomeos/{primal}-default.sock`
//! 3. `<temp_dir>/{primal}-default.sock` (platform-agnostic fallback)
//!
//! No absolute paths or hardcoded primal names — discovery is parameterized
//! by env var and primal name, following the Primal IPC Protocol.

use std::path::PathBuf;

/// Discover Squirrel AI socket.
///
/// Priority: `SQUIRREL_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_squirrel() -> Option<PathBuf> {
    discover_socket("SQUIRREL_SOCKET", super::primal_names::SQUIRREL)
}

/// Discover an existing primal socket by env var and primal name.
///
/// Returns `Some(path)` if a socket file is found at one of the
/// standard locations, `None` otherwise (standalone mode).
#[must_use]
pub fn discover_socket(env_var: &str, primal: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_var) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join(format!("{}/{}-default.sock", super::primal_names::BIOMEOS, primal));
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

/// Resolve a bind path for a server socket.
///
/// Unlike [`discover_socket`], this returns a path even if the file does
/// not yet exist (the caller will create/bind it).
#[must_use]
pub fn resolve_bind_path(env_var: &str, primal: &str) -> PathBuf {
    if let Ok(path) = std::env::var(env_var) {
        return PathBuf::from(path);
    }

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join(format!("{}/{}-default.sock", super::primal_names::BIOMEOS, primal));
    }

    std::env::temp_dir().join(format!("{primal}-default.sock"))
}

/// Pure-logic socket resolution for testing (no env reads).
///
/// Accepts pre-resolved values so tests don't pollute the process environment.
#[cfg(test)]
#[must_use]
pub fn resolve_socket_explicit(
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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_when_no_socket() {
        let result = discover_socket("WETSPRING_TEST_NONEXISTENT_VAR_12345", "nonexistent_primal");
        assert!(result.is_none());
    }

    #[test]
    fn discover_explicit_env_override() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("custom.sock");
        std::fs::write(&sock, "").unwrap();

        temp_env::with_var(
            "WETSPRING_DISCOVER_TEST_SOCK",
            Some(sock.to_str().unwrap()),
            || {
                let found = discover_socket("WETSPRING_DISCOVER_TEST_SOCK", "irrelevant");
                assert_eq!(found, Some(sock.clone()));
            },
        );
    }

    #[test]
    fn resolve_bind_path_uses_env() {
        temp_env::with_var("WETSPRING_BIND_TEST", Some("/tmp/test-bind.sock"), || {
            let p = resolve_bind_path("WETSPRING_BIND_TEST", super::super::primal_names::SELF);
            assert_eq!(p, PathBuf::from("/tmp/test-bind.sock"));
        });
    }

    #[test]
    fn resolve_bind_path_falls_through_to_temp() {
        temp_env::with_vars(
            [
                ("WETSPRING_BIND_FALLBACK_TEST", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let p = resolve_bind_path("WETSPRING_BIND_FALLBACK_TEST", "myprimal");
                assert!(p.to_string_lossy().contains("myprimal"));
            },
        );
    }

    #[test]
    fn resolve_socket_explicit_finds_existing() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test.sock");
        std::fs::write(&sock, "").unwrap();

        let found = resolve_socket_explicit(Some(sock.to_str().unwrap()), None, "unused", "unused");
        assert_eq!(found, Some(sock));
    }

    #[test]
    fn resolve_socket_explicit_none_when_missing() {
        let found =
            resolve_socket_explicit(Some("/nonexistent/path.sock"), None, "unused", "unused");
        assert!(found.is_none());
    }
}
