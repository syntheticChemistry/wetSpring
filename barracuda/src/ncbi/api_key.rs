// SPDX-License-Identifier: AGPL-3.0-or-later
//! NCBI API key discovery via capability-based cascade.

use std::path::Path;

/// Resolve the NCBI API key via capability-based discovery.
///
/// Search order (first match wins):
/// 1. `NCBI_API_KEY` environment variable (production / CI)
/// 2. `WETSPRING_DATA_ROOT/api-keys.toml` (capability-based data root)
/// 3. `$HOME/.config/wetspring/api-keys.toml` (XDG convention)
///
/// Returns `None` if no key is found — callers must degrade gracefully
/// (NCBI allows 3 req/sec without a key, 10 req/sec with one).
#[must_use]
pub fn api_key() -> Option<String> {
    if let Ok(key) = std::env::var("NCBI_API_KEY") {
        return Some(key);
    }

    let data_root = std::env::var("WETSPRING_DATA_ROOT").ok();
    let home = std::env::var("HOME").ok();
    resolve(data_root.as_deref(), home.as_deref())
}

/// Pure-logic API key resolution from filesystem paths.
///
/// Checks `data_root/api-keys.toml` then `home/.config/wetspring/api-keys.toml`.
fn resolve(data_root: Option<&str>, home: Option<&str>) -> Option<String> {
    if let Some(root) = data_root {
        let toml_path = Path::new(root).join("api-keys.toml");
        if let Some(key) = parse_toml(&toml_path) {
            return Some(key);
        }
    }

    if let Some(h) = home {
        let xdg = Path::new(h).join(".config/wetspring/api-keys.toml");
        if let Some(key) = parse_toml(&xdg) {
            return Some(key);
        }
    }

    None
}

/// Extract `ncbi_api_key = "..."` from a TOML file.
fn parse_toml(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    content
        .lines()
        .find(|l| l.starts_with("ncbi_api_key"))
        .and_then(|l| l.split('"').nth(1))
        .map(String::from)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn returns_from_env() {
        if std::env::var("NCBI_API_KEY").is_ok() {
            assert!(api_key().is_some());
        }
    }

    #[test]
    fn parse_toml_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# API keys").unwrap();
        writeln!(f, "ncbi_api_key = \"abc123def456\"").unwrap();
        writeln!(f, "other_key = \"xyz\"").unwrap();

        let key = parse_toml(&path);
        assert_eq!(key, Some("abc123def456".to_string()));
    }

    #[test]
    fn parse_toml_missing_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nokey.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "other_key = \"xyz\"").unwrap();

        assert!(parse_toml(&path).is_none());
    }

    #[test]
    fn parse_toml_nonexistent_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wetspring_nonexistent_toml_abc123.toml");
        assert!(parse_toml(&path).is_none());
    }

    #[test]
    fn parse_toml_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.toml");
        std::fs::File::create(&path).unwrap();

        assert!(parse_toml(&path).is_none());
    }

    #[test]
    fn parse_toml_no_quotes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noquotes.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = bare_value").unwrap();

        assert!(parse_toml(&path).is_none());
    }

    #[test]
    fn parse_toml_with_spaces_around_equals() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spaces.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key   =   \"spaced_key\"").unwrap();

        let key = parse_toml(&path);
        assert_eq!(key, Some("spaced_key".to_string()));
    }

    #[test]
    fn parse_toml_with_comment_before_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("commented.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# NCBI API key for production use").unwrap();
        writeln!(f, "ncbi_api_key = \"comment_key\"").unwrap();

        let key = parse_toml(&path);
        assert_eq!(key, Some("comment_key".to_string()));
    }

    #[test]
    fn parse_toml_multiple_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"first_key\"").unwrap();
        writeln!(f, "other = \"x\"").unwrap();
        writeln!(f, "ncbi_api_key = \"second_key\"").unwrap();

        let key = parse_toml(&path);
        assert_eq!(key, Some("first_key".to_string()));
    }

    #[test]
    fn resolve_data_root_hit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("api-keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"root_key\"").unwrap();

        let key = resolve(Some(dir.path().to_str().unwrap()), None);
        assert_eq!(key, Some("root_key".to_string()));
    }

    #[test]
    fn resolve_home_xdg_hit() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().join(".config/wetspring");
        std::fs::create_dir_all(&xdg).unwrap();
        let path = xdg.join("api-keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"home_key\"").unwrap();

        let key = resolve(None, Some(dir.path().to_str().unwrap()));
        assert_eq!(key, Some("home_key".to_string()));
    }

    #[test]
    fn resolve_data_root_priority_over_home() {
        let root_dir = tempfile::tempdir().unwrap();
        let root_toml = root_dir.path().join("api-keys.toml");
        let mut f = std::fs::File::create(&root_toml).unwrap();
        writeln!(f, "ncbi_api_key = \"root_wins\"").unwrap();

        let home_dir = tempfile::tempdir().unwrap();
        let xdg = home_dir.path().join(".config/wetspring");
        std::fs::create_dir_all(&xdg).unwrap();
        let home_toml = xdg.join("api-keys.toml");
        let mut hf = std::fs::File::create(&home_toml).unwrap();
        writeln!(hf, "ncbi_api_key = \"home_loses\"").unwrap();

        let key = resolve(
            Some(root_dir.path().to_str().unwrap()),
            Some(home_dir.path().to_str().unwrap()),
        );
        assert_eq!(key, Some("root_wins".to_string()));
    }

    #[test]
    fn resolve_both_none() {
        assert!(resolve(None, None).is_none());
    }

    #[test]
    fn resolve_data_root_missing_file_falls_to_home() {
        let root_dir = tempfile::tempdir().unwrap();

        let home_dir = tempfile::tempdir().unwrap();
        let xdg = home_dir.path().join(".config/wetspring");
        std::fs::create_dir_all(&xdg).unwrap();
        let path = xdg.join("api-keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"fallback\"").unwrap();

        let key = resolve(
            Some(root_dir.path().to_str().unwrap()),
            Some(home_dir.path().to_str().unwrap()),
        );
        assert_eq!(key, Some("fallback".to_string()));
    }

    #[test]
    fn is_deterministic() {
        let k1 = api_key();
        let k2 = api_key();
        assert_eq!(k1, k2);
    }
}
