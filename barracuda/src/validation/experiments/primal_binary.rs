// SPDX-License-Identifier: AGPL-3.0-or-later
//! Post-primordial primal binary discovery — plasmidBin is the sole source.
//!
//! Experiments that need to spawn primal processes use this module instead of
//! ad-hoc `target/release/` or `which` fallbacks.

use std::path::PathBuf;

/// Discover a primal binary in plasmidBin.
///
/// Search order:
/// 1. `{NAME}_BIN` env var (explicit override)
/// 2. `NUCLEUS_BIN_DIR` env var (set by nucleus_launcher.sh)
/// 3. `infra/plasmidBin/primals/{triple}/` (git checkout, triple-aware)
/// 4. `infra/plasmidBin/primals/` (git checkout, flat)
///
/// Returns `None` if the binary is not found — no PATH or `target/release/`
/// fallback (post-primordial standard, Wave 49).
pub fn discover(name: &str) -> Option<PathBuf> {
    let env_var = format!("{}_BIN", name.to_uppercase().replace('-', "_"));
    if let Ok(path) = std::env::var(&env_var) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(bin_dir) = std::env::var("NUCLEUS_BIN_DIR") {
        let p = PathBuf::from(&bin_dir).join(name);
        if p.exists() {
            return Some(p);
        }
    }

    let triple = host_triple();
    for depth in &["../..", "../../..", "../../../.."] {
        let triple_path =
            PathBuf::from(format!("{depth}/infra/plasmidBin/primals/{triple}/{name}"));
        if triple_path.exists() {
            return Some(triple_path);
        }
        let flat_path = PathBuf::from(format!("{depth}/infra/plasmidBin/primals/{name}"));
        if flat_path.exists() {
            return Some(flat_path);
        }
    }
    None
}

fn host_triple() -> &'static str {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        "x86_64-unknown-linux-musl"
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        "aarch64-unknown-linux-musl"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_os = "linux"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "macos"),
    )))]
    {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_triple_is_not_empty() {
        assert!(!host_triple().is_empty());
        assert_ne!(host_triple(), "unknown");
    }

    #[test]
    fn discover_returns_none_for_nonexistent() {
        assert!(discover("definitely_not_a_primal_binary_xyzzy").is_none());
    }
}
