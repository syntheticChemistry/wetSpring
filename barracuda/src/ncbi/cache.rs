// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based cache file path resolution.

use std::path::{Path, PathBuf};

/// Build a cache file path via capability-based data discovery.
///
/// Discovery order:
/// 1. `WETSPRING_DATA_ROOT/{filename}` if `WETSPRING_DATA_ROOT` is set
/// 2. `{CARGO_MANIFEST_DIR}/../data/{filename}` for development
/// 3. `data/{filename}` relative to cwd for deployment
#[must_use]
pub fn cache_file(filename: &str) -> PathBuf {
    let data_root = std::env::var("WETSPRING_DATA_ROOT").ok();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    resolve(data_root.as_deref(), &manifest, filename)
}

/// Pure-logic cache path resolution — no env access.
fn resolve(data_root: Option<&str>, manifest_dir: &str, filename: &str) -> PathBuf {
    if let Some(root) = data_root {
        let p = PathBuf::from(root).join(filename);
        if p.parent().is_some_and(Path::exists) {
            return p;
        }
    }

    let dev_path = PathBuf::from(manifest_dir).join("../data").join(filename);
    if dev_path.parent().is_some_and(Path::exists) {
        return dev_path;
    }

    PathBuf::from("data").join(filename)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn builds_relative_path() {
        let path = cache_file("test_cache.txt");
        assert!(
            path.to_string_lossy().contains("data")
                && path.to_string_lossy().contains("test_cache.txt")
        );
    }

    #[test]
    fn nested() {
        let path = cache_file("sub/nested.json");
        assert!(path.to_string_lossy().contains("nested.json"));
    }

    #[test]
    fn dev_fallback_contains_data() {
        let path = cache_file("test.txt");
        let s = path.to_string_lossy();
        assert!(s.contains("data") && s.contains("test.txt"));
    }

    #[test]
    fn uses_manifest_dir_fallback() {
        let path = cache_file("integration_test_file.json");
        let s = path.to_string_lossy();
        assert!(
            s.contains("data") && s.contains("integration_test_file.json"),
            "path should contain data dir: {s}"
        );
    }

    #[test]
    fn handles_path_separators() {
        let path = cache_file("subdir/with/slashes.json");
        let s = path.to_string_lossy();
        assert!(s.contains("slashes.json"));
        assert!(s.contains("subdir") || s.contains("with"));
    }

    #[test]
    fn resolve_data_root_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = resolve(Some(dir.path().to_str().unwrap()), ".", "test.json");
        assert_eq!(path, dir.path().join("test.json"));
    }

    #[test]
    fn resolve_data_root_nonexistent_falls_through() {
        let path = resolve(
            Some("/nonexistent_wetspring_test_dir_xyz"),
            ".",
            "test.json",
        );
        assert!(
            !path.starts_with("/nonexistent_wetspring_test_dir_xyz"),
            "should not use nonexistent data root: {path:?}"
        );
    }

    #[test]
    fn resolve_manifest_dir_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let manifest = dir.path().join("sub");
        std::fs::create_dir_all(&manifest).unwrap();
        let path = resolve(None, manifest.to_str().unwrap(), "cache.bin");
        assert!(path.to_string_lossy().contains("data"));
    }

    #[test]
    fn resolve_final_fallback() {
        let path = resolve(None, "/nonexistent_manifest_xyz", "fallback.json");
        assert_eq!(path, PathBuf::from("data").join("fallback.json"));
    }

    #[test]
    fn resolve_none_root_with_valid_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let manifest = dir.path().join("crate");
        std::fs::create_dir_all(&manifest).unwrap();
        let path = resolve(None, manifest.to_str().unwrap(), "x.json");
        assert!(path.to_string_lossy().contains("data"), "path = {path:?}");
    }
}
