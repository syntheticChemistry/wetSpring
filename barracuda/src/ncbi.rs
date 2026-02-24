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

use std::path::{Path, PathBuf};

/// Resolve the NCBI API key from environment or well-known TOML files.
///
/// Search order:
/// 1. `NCBI_API_KEY` environment variable
/// 2. `../../../testing-secrets/api-keys.toml` (relative to manifest)
/// 3. `../../testing-secrets/api-keys.toml`
#[must_use]
pub fn api_key() -> Option<String> {
    if let Ok(key) = std::env::var("NCBI_API_KEY") {
        return Some(key);
    }
    let relative_paths = [
        "../../../testing-secrets/api-keys.toml",
        "../../testing-secrets/api-keys.toml",
    ];
    for path in &relative_paths {
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
    let url = format!(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?\
         db={db}&term={encoded}&rettype=count&api_key={api_key}"
    );
    let body = http_get(&url)?;

    if let Some(start) = body.find("<Count>") {
        let rest = &body[start + 7..];
        if let Some(end) = rest.find("</Count>") {
            return rest[..end].trim().parse::<u64>().map_err(|e| e.to_string());
        }
    }
    Err(format!(
        "no <Count> in response: {}",
        &body[..body.len().min(200)]
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
mod tests {
    use super::*;

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
    fn cache_file_builds_relative_path() {
        let path = cache_file("test_cache.txt");
        assert!(path.ends_with("data/test_cache.txt"));
    }

    #[test]
    fn api_key_returns_from_env() {
        // Only test the env var path — don't depend on secrets file
        if std::env::var("NCBI_API_KEY").is_ok() {
            assert!(api_key().is_some());
        }
    }
}
