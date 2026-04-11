// SPDX-License-Identifier: AGPL-3.0-or-later
//! Upstream contract assertions for barraCuda math primal.
//!
//! Pins expected barraCuda behavior so that upstream changes that break
//! wetSpring invariants are caught at compile time or in CI, not silently
//! absorbed. Absorbed from neuralSpring S174 `upstream_contract` pattern.

/// Minimum barraCuda version wetSpring was validated against.
pub const PINNED_BARRACUDA_VERSION: &str = "0.3.11";

/// Verify that the barraCuda Cargo.toml version matches expectations.
///
/// This is a documentation-only assertion — the real guard is `Cargo.lock`.
/// barraCuda does not export a `VERSION` constant; path dependencies are
/// pinned by lockfile resolution.
///
/// Call from CI or startup to remind operators that drift should be reviewed
/// before it causes silent precision regressions.
#[must_use]
pub const fn check_barracuda_version() -> bool {
    // With path dependencies, Cargo.lock pins the exact source.
    // [`PINNED_BARRACUDA_VERSION`] documents what version wetSpring was validated against.
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Semantic version comparison: `actual >= minimum`.
    fn version_ge(actual: &str, minimum: &str) -> bool {
        let parse = |s: &str| -> (u32, u32, u32) {
            let mut parts = s.split('.');
            let major = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
            let minor = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
            let patch = parts
                .next()
                .and_then(|p| p.split('-').next()?.parse().ok())
                .unwrap_or(0);
            (major, minor, patch)
        };
        parse(actual) >= parse(minimum)
    }

    /// Reads `[workspace.package] version = "…"` from barraCuda workspace root.
    fn workspace_package_version(cargo_toml: &str) -> Option<String> {
        let mut in_ws_pkg = false;
        for line in cargo_toml.lines() {
            let t = line.trim();
            if t == "[workspace.package]" {
                in_ws_pkg = true;
                continue;
            }
            if in_ws_pkg && t.starts_with('[') {
                break;
            }
            if in_ws_pkg {
                if let Some(rest) = t.strip_prefix("version = ") {
                    let v = rest.trim().trim_matches('"');
                    if v != "true" && !v.contains("workspace") {
                        return Some(v.to_string());
                    }
                }
            }
        }
        None
    }

    #[test]
    fn pinned_version_matches_linked() {
        assert!(check_barracuda_version());

        let manifest =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../primals/barraCuda/Cargo.toml");
        let Ok(content) = std::fs::read_to_string(&manifest) else {
            panic!(
                "expected path dependency at {}; clone barraCuda alongside wetSpring",
                manifest.display()
            );
        };
        let Some(linked) = workspace_package_version(&content) else {
            panic!("barraCuda workspace Cargo.toml missing [workspace.package] version");
        };
        assert!(
            version_ge(&linked, PINNED_BARRACUDA_VERSION),
            "barraCuda workspace version {linked} is below pin {PINNED_BARRACUDA_VERSION}"
        );
    }

    #[test]
    fn version_ge_basics() {
        assert!(version_ge("0.3.7", "0.3.7"));
        assert!(version_ge("0.3.8", "0.3.7"));
        assert!(version_ge("0.4.0", "0.3.7"));
        assert!(version_ge("1.0.0", "0.3.7"));
        assert!(!version_ge("0.3.6", "0.3.7"));
        assert!(!version_ge("0.2.9", "0.3.7"));
    }

    #[test]
    fn version_ge_handles_prerelease() {
        assert!(version_ge("0.3.7-dev", "0.3.7"));
    }
}
