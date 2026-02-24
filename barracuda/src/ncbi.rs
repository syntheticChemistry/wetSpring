// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared NCBI Entrez helpers for validation binaries.
//!
//! Provides API-key discovery, HTTP GET via `curl` subprocess (sovereign
//! HTTPS without TLS crate deps), and Entrez E-search wrappers. Validation
//! binaries that query NCBI share this module instead of duplicating the
//! same boilerplate.
//!
//! # Why `curl`?
//!
//! HTTPS requires a TLS implementation. Adding `rustls`/`native-tls` would
//! pull in 20+ transitive crates and violate ecoBin's minimal-dependency
//! principle. Shelling out to `curl` keeps the binary self-contained while
//! still supporting HTTPS. Binaries fall back to cached/synthetic data when
//! `curl` is unavailable.
//!
//! # Evolution path
//!
//! | Phase | Strategy | Status |
//! |-------|----------|--------|
//! | Current | `curl` subprocess — zero compile deps, runtime dep on system curl | active |
//! | Phase 2 | metalForge HTTP substrate — route HTTPS through forge routing | blocked on forge HTTP |
//! | Phase 3 | Sovereign TLS (if HTTPS becomes pipeline-critical) | not needed for validation |
//!
//! The `curl` approach is the correct tradeoff for validation-only network
//! access. All callers degrade gracefully to cached/synthetic data when
//! `curl` is absent.

use std::path::{Path, PathBuf};

const ENTREZ_BASE: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";

/// Resolve the NCBI API key via capability-based discovery.
///
/// Search order (first match wins):
/// 1. `NCBI_API_KEY` environment variable (production / CI)
/// 2. `WETSPRING_DATA_ROOT/api-keys.toml` (capability-based data root)
/// 3. `$HOME/.config/wetspring/api-keys.toml` (XDG convention)
/// 4. Well-known development paths relative to workspace root
///
/// Returns `None` if no key is found — callers must degrade gracefully
/// (NCBI allows 3 req/sec without a key, 10 req/sec with one).
#[must_use]
pub fn api_key() -> Option<String> {
    if let Ok(key) = std::env::var("NCBI_API_KEY") {
        return Some(key);
    }

    // Capability-based: check the validation data root
    if let Ok(root) = std::env::var("WETSPRING_DATA_ROOT") {
        let toml_path = Path::new(&root).join("api-keys.toml");
        if let Some(key) = parse_api_key_toml(&toml_path) {
            return Some(key);
        }
    }

    // XDG config home (~/.config/wetspring/)
    if let Ok(home) = std::env::var("HOME") {
        let xdg = Path::new(&home).join(".config/wetspring/api-keys.toml");
        if let Some(key) = parse_api_key_toml(&xdg) {
            return Some(key);
        }
    }

    // Legacy dev paths (kept for backwards compatibility during transition)
    let legacy_paths = [
        "../../../testing-secrets/api-keys.toml",
        "../../testing-secrets/api-keys.toml",
    ];
    for path in &legacy_paths {
        if let Some(key) = parse_api_key_toml(Path::new(path)) {
            return Some(key);
        }
    }

    None
}

/// Extract `ncbi_api_key = "..."` from a TOML file.
fn parse_api_key_toml(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    content
        .lines()
        .find(|l| l.starts_with("ncbi_api_key"))
        .and_then(|l| l.split('"').nth(1))
        .map(String::from)
}

/// HTTP GET via `curl` subprocess — sovereign HTTPS without TLS crate deps.
///
/// Returns the response body as a `String`. Falls back to an error when
/// `curl` is not installed or the request fails.
///
/// # Errors
///
/// Returns `Err` if `curl` is not found, the request times out (30 s),
/// or the response contains invalid UTF-8.
pub fn http_get(url: &str) -> Result<String, String> {
    let output = std::process::Command::new("curl")
        .args(["-s", "-m", "30", url])
        .output()
        .map_err(|e| e.to_string())?;

    if output.status.success() {
        String::from_utf8(output.stdout).map_err(|e| e.to_string())
    } else {
        Err(format!(
            "curl failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// URL-encode a search term for Entrez E-utilities.
fn encode_entrez_term(term: &str) -> String {
    term.replace(' ', "+")
        .replace('"', "%22")
        .replace('[', "%5B")
        .replace(']', "%5D")
        .replace('(', "%28")
        .replace(')', "%29")
}

/// Query NCBI Entrez E-search and return the hit count.
///
/// Sends a `rettype=count` request to
/// `eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi` and parses the
/// `<Count>` element from the XML response.
///
/// # Errors
///
/// Returns `Err` if the HTTP request fails, the response is missing a
/// `<Count>` element, or the count cannot be parsed as `u64`.
pub fn esearch_count(db: &str, term: &str, api_key: &str) -> Result<u64, String> {
    let encoded = encode_entrez_term(term);
    let url = format!("{ENTREZ_BASE}?db={db}&term={encoded}&rettype=count&api_key={api_key}");
    let body = http_get(&url)?;
    parse_esearch_count(&body)
}

/// Parse `<Count>...</Count>` from an Entrez E-search XML response body.
fn parse_esearch_count(body: &str) -> Result<u64, String> {
    if let Some(start) = body.find("<Count>") {
        let rest = &body[start + 7..];
        if let Some(end) = rest.find("</Count>") {
            return rest[..end].trim().parse::<u64>().map_err(|e| e.to_string());
        }
    }
    Err(format!(
        "no <Count> in response: {}",
        &body[..body.len().min(crate::tolerances::ERROR_BODY_PREVIEW_LEN)]
    ))
}

/// Build a cache file path relative to the crate manifest directory.
///
/// Returns `{CARGO_MANIFEST_DIR}/../data/{filename}`.
#[must_use]
pub fn cache_file(filename: &str) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest).join("../data").join(filename)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn encode_entrez_term_spaces_and_brackets() {
        let encoded = encode_entrez_term("luxI[gene] AND (biofilm)");
        assert_eq!(encoded, "luxI%5Bgene%5D+AND+%28biofilm%29");
    }

    #[test]
    fn encode_entrez_term_quotes() {
        let encoded = encode_entrez_term("\"quorum sensing\"");
        assert_eq!(encoded, "%22quorum+sensing%22");
    }

    #[test]
    fn encode_entrez_term_empty() {
        assert_eq!(encode_entrez_term(""), "");
    }

    #[test]
    fn encode_entrez_term_no_special_chars() {
        assert_eq!(encode_entrez_term("simple"), "simple");
    }

    #[test]
    fn encode_entrez_term_all_special() {
        let encoded = encode_entrez_term("\"a\" [b] (c) d e");
        assert_eq!(encoded, "%22a%22+%5Bb%5D+%28c%29+d+e");
    }

    #[test]
    fn cache_file_builds_relative_path() {
        let path = cache_file("test_cache.txt");
        assert!(path.ends_with("data/test_cache.txt"));
    }

    #[test]
    fn cache_file_nested() {
        let path = cache_file("sub/nested.json");
        assert!(path.ends_with("data/sub/nested.json"));
    }

    #[test]
    fn api_key_returns_from_env() {
        if std::env::var("NCBI_API_KEY").is_ok() {
            assert!(api_key().is_some());
        }
    }

    #[test]
    fn parse_api_key_toml_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# API keys").unwrap();
        writeln!(f, "ncbi_api_key = \"abc123def456\"").unwrap();
        writeln!(f, "other_key = \"xyz\"").unwrap();

        let key = parse_api_key_toml(&path);
        assert_eq!(key, Some("abc123def456".to_string()));
    }

    #[test]
    fn parse_api_key_toml_missing_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nokey.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "other_key = \"xyz\"").unwrap();

        assert!(parse_api_key_toml(&path).is_none());
    }

    #[test]
    fn parse_api_key_toml_nonexistent_file() {
        let path = Path::new("/tmp/wetspring_nonexistent_toml_abc123.toml");
        assert!(parse_api_key_toml(path).is_none());
    }

    #[test]
    fn parse_api_key_toml_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.toml");
        std::fs::File::create(&path).unwrap();

        assert!(parse_api_key_toml(&path).is_none());
    }

    #[test]
    fn parse_api_key_toml_no_quotes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noquotes.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = bare_value").unwrap();

        assert!(parse_api_key_toml(&path).is_none());
    }

    #[test]
    fn http_get_localhost_refused() {
        let result = http_get("http://127.0.0.1:1/nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn parse_esearch_count_valid() {
        let xml = r#"<?xml version="1.0"?>
<eSearchResult><Count>42</Count><RetMax>0</RetMax></eSearchResult>"#;
        assert_eq!(parse_esearch_count(xml).unwrap(), 42);
    }

    #[test]
    fn parse_esearch_count_with_whitespace() {
        let xml = "<eSearchResult><Count>  1234  </Count></eSearchResult>";
        assert_eq!(parse_esearch_count(xml).unwrap(), 1234);
    }

    #[test]
    fn parse_esearch_count_missing_tag() {
        let xml = "<eSearchResult><RetMax>0</RetMax></eSearchResult>";
        assert!(parse_esearch_count(xml).is_err());
    }

    #[test]
    fn parse_esearch_count_empty_body() {
        assert!(parse_esearch_count("").is_err());
    }

    #[test]
    fn parse_esearch_count_unclosed_tag() {
        let xml = "<eSearchResult><Count>99";
        assert!(parse_esearch_count(xml).is_err());
    }

    #[test]
    fn parse_esearch_count_non_numeric() {
        let xml = "<Count>abc</Count>";
        assert!(parse_esearch_count(xml).is_err());
    }

    #[test]
    fn parse_esearch_count_zero() {
        let xml = "<Count>0</Count>";
        assert_eq!(parse_esearch_count(xml).unwrap(), 0);
    }

    #[test]
    fn parse_esearch_count_large_value() {
        let xml = "<Count>9999999999</Count>";
        assert_eq!(parse_esearch_count(xml).unwrap(), 9_999_999_999);
    }
}
