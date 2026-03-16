// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport abstraction for IPC connections.
//!
//! Currently supports Unix domain sockets only. TCP transport is documented
//! in the `wetspring_server` binary (`WETSPRING_TCP_ADDR`) but not yet
//! implemented. This module provides the abstraction layer for future
//! multi-transport support without changing the server or dispatch logic.

use std::path::PathBuf;

/// Supported transport types for the Primal IPC Protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    /// Unix domain socket at a filesystem path.
    Unix(PathBuf),
}

impl Transport {
    /// Resolve the transport from environment configuration.
    ///
    /// Currently only resolves Unix domain sockets via [`super::discover`].
    #[must_use]
    pub fn resolve(env_var: &str, primal: &str) -> Self {
        Self::Unix(super::discover::resolve_bind_path(env_var, primal))
    }

    /// The filesystem path for Unix transports.
    #[must_use]
    pub fn path(&self) -> Option<&std::path::Path> {
        match self {
            Self::Unix(p) => Some(p),
        }
    }
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unix(p) => write!(f, "unix:{}", p.display()),
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn resolve_returns_unix() {
        temp_env::with_vars(
            [
                ("WETSPRING_TRANSPORT_TEST", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let t = Transport::resolve("WETSPRING_TRANSPORT_TEST", "test_primal");
                assert!(matches!(t, Transport::Unix(_)));
                assert!(t.path().is_some());
            },
        );
    }

    #[test]
    fn display_format() {
        let dir = tempfile::tempdir().unwrap();
        let sock_path = dir.path().join("test.sock");
        let t = Transport::Unix(sock_path.clone());
        assert_eq!(t.to_string(), format!("unix:{}", sock_path.display()));
    }
}
