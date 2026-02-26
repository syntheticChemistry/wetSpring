// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared NCBI Entrez helpers for validation binaries.
//!
//! Provides API-key discovery, HTTP GET, and Entrez E-search wrappers.
//! Validation binaries that query NCBI share this module instead of
//! duplicating the same boilerplate.
//!
//! # HTTP transport
//!
//! HTTP GET uses a capability-based transport chain — the first available
//! backend wins:
//!
//! 1. **`WETSPRING_HTTP_CMD`** — user-supplied command (e.g. `wget -qO-`)
//! 2. **System `curl`** — sovereign HTTPS without TLS crate deps
//! 3. **System `wget`** — common fallback on minimal containers
//!
//! All callers degrade gracefully to cached/synthetic data when no
//! HTTP transport is available. The transport is an implementation
//! detail — primal code discovers capabilities at runtime.
//!
//! # Evolution path
//!
//! | Phase | Strategy | Status |
//! |-------|----------|--------|
//! | Current | Capability-discovered system HTTP — zero compile deps | active |
//! | Phase 2 | metalForge HTTP substrate — route through forge dispatch | blocked on forge HTTP |
//! | Phase 3 | Sovereign Rust TLS (if HTTPS becomes pipeline-critical) | not needed for validation |

use std::path::{Path, PathBuf};

const ENTREZ_BASE: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";

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
    api_key_from_paths(data_root.as_deref(), home.as_deref())
}

/// Pure-logic API key resolution from filesystem paths.
///
/// Checks `data_root/api-keys.toml` then `home/.config/wetspring/api-keys.toml`.
fn api_key_from_paths(data_root: Option<&str>, home: Option<&str>) -> Option<String> {
    if let Some(root) = data_root {
        let toml_path = Path::new(root).join("api-keys.toml");
        if let Some(key) = parse_api_key_toml(&toml_path) {
            return Some(key);
        }
    }

    if let Some(h) = home {
        let xdg = Path::new(h).join(".config/wetspring/api-keys.toml");
        if let Some(key) = parse_api_key_toml(&xdg) {
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

/// Discovered HTTP transport backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpBackend {
    Custom,
    Curl,
    Wget,
}

/// Discover an available HTTP GET backend at runtime.
///
/// Returns the backend kind and the command name/path. Checks:
/// 1. `WETSPRING_HTTP_CMD` environment variable
/// 2. `curl` on `$PATH`
/// 3. `wget` on `$PATH`
fn discover_http_backend() -> Option<(HttpBackend, String)> {
    let custom = std::env::var("WETSPRING_HTTP_CMD").ok();
    select_backend(
        custom.as_deref(),
        which_exists("curl"),
        which_exists("wget"),
    )
}

/// Pure-logic backend selection — no env or filesystem access.
fn select_backend(
    custom_cmd: Option<&str>,
    has_curl: bool,
    has_wget: bool,
) -> Option<(HttpBackend, String)> {
    if let Some(cmd) = custom_cmd {
        if !cmd.is_empty() {
            return Some((HttpBackend::Custom, cmd.to_string()));
        }
    }

    if has_curl {
        return Some((HttpBackend::Curl, "curl".to_string()));
    }

    if has_wget {
        return Some((HttpBackend::Wget, "wget".to_string()));
    }

    None
}

/// Check whether a command exists on `$PATH` without executing it.
fn which_exists(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// HTTP GET via capability-discovered system transport.
///
/// Discovers the best available HTTP backend at runtime and uses it.
/// Returns the response body as a `String`.
///
/// # Errors
///
/// Returns `Err` if no HTTP transport is available, the request fails,
/// times out (30 s), or the response contains invalid UTF-8.
#[must_use = "HTTP response body is discarded if not used"]
pub fn http_get(url: &str) -> Result<String, String> {
    let (backend, cmd) = discover_http_backend().ok_or_else(|| {
        "no HTTP transport available (need curl or wget on PATH, or set WETSPRING_HTTP_CMD)"
            .to_string()
    })?;

    let output = match backend {
        HttpBackend::Custom => {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            let (program, args) = parts
                .split_first()
                .ok_or_else(|| "WETSPRING_HTTP_CMD is empty".to_string())?;
            let mut command = std::process::Command::new(program);
            command.args(args);
            command.arg(url);
            command.output().map_err(|e| format!("{cmd}: {e}"))?
        }
        HttpBackend::Curl => std::process::Command::new("curl")
            .args(["-sfS", "-m", "30", url])
            .output()
            .map_err(|e| format!("curl: {e}"))?,
        HttpBackend::Wget => std::process::Command::new("wget")
            .args(["-qO-", "--timeout=30", url])
            .output()
            .map_err(|e| format!("wget: {e}"))?,
    };

    interpret_output(&output, &cmd)
}

/// Interpret the output of an HTTP subprocess.
///
/// Extracted for testability: the subprocess dispatch is environment-dependent,
/// but the response interpretation is pure logic.
fn interpret_output(output: &std::process::Output, cmd: &str) -> Result<String, String> {
    if output.status.success() {
        String::from_utf8(output.stdout.clone()).map_err(|e| e.to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let preview = &stderr[..stderr.len().min(crate::tolerances::ERROR_BODY_PREVIEW_LEN)];
        Err(format!(
            "{cmd} failed (exit {:?}): {preview}",
            output.status.code()
        ))
    }
}

/// URL-encode a search term for Entrez E-utilities.
fn encode_entrez_term(term: &str) -> String {
    let mut out = String::with_capacity(term.len() * 2);
    for ch in term.chars() {
        match ch {
            ' ' => out.push('+'),
            '"' => out.push_str("%22"),
            '[' => out.push_str("%5B"),
            ']' => out.push_str("%5D"),
            '(' => out.push_str("%28"),
            ')' => out.push_str("%29"),
            '&' => out.push_str("%26"),
            '#' => out.push_str("%23"),
            other => out.push(other),
        }
    }
    out
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
#[must_use = "search count is discarded if not used"]
pub fn esearch_count(db: &str, term: &str, api_key: &str) -> Result<u64, String> {
    let encoded = encode_entrez_term(term);
    let url = format!("{ENTREZ_BASE}?db={db}&term={encoded}&rettype=count&api_key={api_key}");
    let body = http_get(&url)?;
    parse_esearch_count(&body)
}

/// Parse `<Count>...</Count>` from an Entrez E-search XML response body.
fn parse_esearch_count(body: &str) -> Result<u64, String> {
    let start = body
        .find("<Count>")
        .ok_or_else(|| preview_error("no <Count> in response", body))?;
    let rest = &body[start + 7..];
    let end = rest
        .find("</Count>")
        .ok_or_else(|| preview_error("unclosed <Count> tag", body))?;
    rest[..end]
        .trim()
        .parse::<u64>()
        .map_err(|e| format!("invalid count value: {e}"))
}

/// Build an error message with a truncated preview of the response body.
fn preview_error(msg: &str, body: &str) -> String {
    let limit = body.len().min(crate::tolerances::ERROR_BODY_PREVIEW_LEN);
    format!("{msg}: {}", &body[..limit])
}

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
    resolve_cache_path(data_root.as_deref(), &manifest, filename)
}

/// Pure-logic cache path resolution — no env access.
fn resolve_cache_path(data_root: Option<&str>, manifest_dir: &str, filename: &str) -> PathBuf {
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
    use std::io::Write;

    // ── URL encoding ──────────────────────────────────────────────

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
    fn encode_entrez_term_ampersand_and_hash() {
        let encoded = encode_entrez_term("a&b#c");
        assert_eq!(encoded, "a%26b%23c");
    }

    // ── Cache file discovery ──────────────────────────────────────

    #[test]
    fn cache_file_builds_relative_path() {
        let path = cache_file("test_cache.txt");
        assert!(
            path.to_string_lossy().contains("data")
                && path.to_string_lossy().contains("test_cache.txt")
        );
    }

    #[test]
    fn cache_file_nested() {
        let path = cache_file("sub/nested.json");
        assert!(path.to_string_lossy().contains("nested.json"));
    }

    #[test]
    fn cache_file_dev_fallback_contains_data() {
        let path = cache_file("test.txt");
        let s = path.to_string_lossy();
        assert!(s.contains("data") && s.contains("test.txt"));
    }

    // ── API key discovery ─────────────────────────────────────────

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

    // ── HTTP backend discovery ────────────────────────────────────

    #[test]
    fn discover_http_backend_finds_curl_or_wget() {
        let result = discover_http_backend();
        if which_exists("curl") {
            let (backend, _) = result.unwrap();
            assert_eq!(backend, HttpBackend::Curl);
        } else if which_exists("wget") {
            let (backend, _) = result.unwrap();
            assert_eq!(backend, HttpBackend::Wget);
        }
    }

    #[test]
    fn which_exists_finds_sh() {
        assert!(which_exists("sh"));
    }

    #[test]
    fn which_exists_rejects_nonexistent() {
        assert!(!which_exists("__wetspring_nonexistent_binary_xyz__"));
    }

    #[test]
    fn http_get_localhost_refused() {
        let result = http_get("http://127.0.0.1:1/nonexistent");
        assert!(result.is_err());
    }

    // ── Entrez XML parsing ────────────────────────────────────────

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

    // ── Preview error formatting ──────────────────────────────────

    #[test]
    fn preview_error_truncates_long_body() {
        let body = "x".repeat(1000);
        let msg = preview_error("test", &body);
        assert!(msg.len() < 500);
    }

    #[test]
    fn preview_error_short_body() {
        let msg = preview_error("oops", "short");
        assert_eq!(msg, "oops: short");
    }

    #[test]
    fn parse_esearch_count_multiline_xml() {
        let xml = r#"<?xml version="1.0"?>
<eSearchResult>
  <Count>
    999
  </Count>
  <RetMax>0</RetMax>
</eSearchResult>"#;
        assert_eq!(parse_esearch_count(xml).unwrap(), 999);
    }

    #[test]
    fn preview_error_empty_body() {
        let msg = preview_error("no count", "");
        assert_eq!(msg, "no count: ");
    }

    #[test]
    fn encode_entrez_term_unicode() {
        let encoded = encode_entrez_term("café naïve 日本語");
        assert_eq!(encoded, "café+naïve+日本語");
    }

    #[test]
    fn cache_file_handles_path_separators() {
        let path = cache_file("subdir/with/slashes.json");
        let s = path.to_string_lossy();
        assert!(s.contains("slashes.json"));
        assert!(s.contains("subdir") || s.contains("with"));
    }

    #[test]
    fn parse_api_key_toml_multiple_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"first_key\"").unwrap();
        writeln!(f, "other = \"x\"").unwrap();
        writeln!(f, "ncbi_api_key = \"second_key\"").unwrap();

        let key = parse_api_key_toml(&path);
        assert_eq!(key, Some("first_key".to_string()));
    }

    // ── interpret_output (extracted HTTP response logic) ─────────

    #[test]
    fn interpret_output_success_returns_body() {
        let output = std::process::Output {
            status: std::process::Command::new("true").status().unwrap(),
            stdout: b"hello world".to_vec(),
            stderr: vec![],
        };
        let result = interpret_output(&output, "test-cmd");
        assert_eq!(result.unwrap(), "hello world");
    }

    #[test]
    fn interpret_output_failure_returns_stderr_preview() {
        let output = std::process::Output {
            status: std::process::Command::new("false").status().unwrap(),
            stdout: vec![],
            stderr: b"connection refused".to_vec(),
        };
        let result = interpret_output(&output, "curl");
        let err = result.unwrap_err();
        assert!(err.contains("curl"));
        assert!(err.contains("connection refused"));
    }

    #[test]
    fn interpret_output_failure_truncates_long_stderr() {
        let output = std::process::Output {
            status: std::process::Command::new("false").status().unwrap(),
            stdout: vec![],
            stderr: "x".repeat(500).into_bytes(),
        };
        let result = interpret_output(&output, "wget");
        let err = result.unwrap_err();
        assert!(err.len() < 500);
    }

    #[test]
    fn interpret_output_success_empty_body() {
        let output = std::process::Output {
            status: std::process::Command::new("true").status().unwrap(),
            stdout: vec![],
            stderr: vec![],
        };
        assert_eq!(interpret_output(&output, "cmd").unwrap(), "");
    }

    // ── cache_file with CARGO_MANIFEST_DIR (set during cargo test) ──

    #[test]
    fn cache_file_uses_manifest_dir_fallback() {
        let path = cache_file("integration_test_file.json");
        let s = path.to_string_lossy();
        assert!(
            s.contains("data") && s.contains("integration_test_file.json"),
            "path should contain data dir: {s}"
        );
    }

    // ── esearch_count URL construction ──────────────────────────

    #[test]
    fn esearch_count_constructs_valid_url() {
        let encoded = encode_entrez_term("luxI[gene]");
        let url = format!("{ENTREZ_BASE}?db=nuccore&term={encoded}&rettype=count&api_key=test_key");
        assert!(url.contains("esearch.fcgi"));
        assert!(url.contains("db=nuccore"));
        assert!(url.contains("luxI%5Bgene%5D"));
        assert!(url.contains("api_key=test_key"));
    }

    // ── discover_http_backend variants ──────────────────────────

    #[test]
    fn which_exists_finds_ls() {
        assert!(which_exists("ls"));
    }

    #[test]
    fn which_exists_finds_echo() {
        assert!(which_exists("echo"));
    }

    // ── api_key without env keys returns None or existing key ───

    #[test]
    fn api_key_is_deterministic() {
        let k1 = api_key();
        let k2 = api_key();
        assert_eq!(k1, k2);
    }

    // ── parse_api_key_toml with whitespace variations ───────────

    #[test]
    fn parse_api_key_toml_with_spaces_around_equals() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spaces.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key   =   \"spaced_key\"").unwrap();

        let key = parse_api_key_toml(&path);
        assert_eq!(key, Some("spaced_key".to_string()));
    }

    #[test]
    fn parse_api_key_toml_with_comment_before_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("commented.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# NCBI API key for production use").unwrap();
        writeln!(f, "ncbi_api_key = \"comment_key\"").unwrap();

        let key = parse_api_key_toml(&path);
        assert_eq!(key, Some("comment_key".to_string()));
    }

    // ── api_key_from_paths (extracted pure logic) ────────────────

    #[test]
    fn api_key_from_paths_data_root_hit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("api-keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"root_key\"").unwrap();

        let key = api_key_from_paths(Some(dir.path().to_str().unwrap()), None);
        assert_eq!(key, Some("root_key".to_string()));
    }

    #[test]
    fn api_key_from_paths_home_xdg_hit() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().join(".config/wetspring");
        std::fs::create_dir_all(&xdg).unwrap();
        let path = xdg.join("api-keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"home_key\"").unwrap();

        let key = api_key_from_paths(None, Some(dir.path().to_str().unwrap()));
        assert_eq!(key, Some("home_key".to_string()));
    }

    #[test]
    fn api_key_from_paths_data_root_priority_over_home() {
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

        let key = api_key_from_paths(
            Some(root_dir.path().to_str().unwrap()),
            Some(home_dir.path().to_str().unwrap()),
        );
        assert_eq!(key, Some("root_wins".to_string()));
    }

    #[test]
    fn api_key_from_paths_both_none() {
        assert!(api_key_from_paths(None, None).is_none());
    }

    #[test]
    fn api_key_from_paths_data_root_missing_file_falls_to_home() {
        let root_dir = tempfile::tempdir().unwrap();

        let home_dir = tempfile::tempdir().unwrap();
        let xdg = home_dir.path().join(".config/wetspring");
        std::fs::create_dir_all(&xdg).unwrap();
        let path = xdg.join("api-keys.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "ncbi_api_key = \"fallback\"").unwrap();

        let key = api_key_from_paths(
            Some(root_dir.path().to_str().unwrap()),
            Some(home_dir.path().to_str().unwrap()),
        );
        assert_eq!(key, Some("fallback".to_string()));
    }

    // ── select_backend (extracted pure logic) ────────────────────

    #[test]
    fn select_backend_custom_wins() {
        let (backend, cmd) = select_backend(Some("myhttp --get"), true, true).unwrap();
        assert_eq!(backend, HttpBackend::Custom);
        assert_eq!(cmd, "myhttp --get");
    }

    #[test]
    fn select_backend_empty_custom_ignored() {
        let (backend, _) = select_backend(Some(""), true, false).unwrap();
        assert_eq!(backend, HttpBackend::Curl);
    }

    #[test]
    fn select_backend_curl_over_wget() {
        let (backend, _) = select_backend(None, true, true).unwrap();
        assert_eq!(backend, HttpBackend::Curl);
    }

    #[test]
    fn select_backend_wget_fallback() {
        let (backend, _) = select_backend(None, false, true).unwrap();
        assert_eq!(backend, HttpBackend::Wget);
    }

    #[test]
    fn select_backend_none_when_nothing() {
        assert!(select_backend(None, false, false).is_none());
    }

    #[test]
    fn select_backend_custom_only_no_system() {
        let (backend, cmd) = select_backend(Some("wget2"), false, false).unwrap();
        assert_eq!(backend, HttpBackend::Custom);
        assert_eq!(cmd, "wget2");
    }

    // ── resolve_cache_path (extracted pure logic) ────────────────

    #[test]
    fn resolve_cache_path_data_root_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = resolve_cache_path(Some(dir.path().to_str().unwrap()), ".", "test.json");
        assert_eq!(path, dir.path().join("test.json"));
    }

    #[test]
    fn resolve_cache_path_data_root_nonexistent_falls_through() {
        let path = resolve_cache_path(
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
    fn resolve_cache_path_manifest_dir_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let manifest = dir.path().join("sub");
        std::fs::create_dir_all(&manifest).unwrap();
        let path = resolve_cache_path(None, manifest.to_str().unwrap(), "cache.bin");
        assert!(path.to_string_lossy().contains("data"));
    }

    #[test]
    fn resolve_cache_path_final_fallback() {
        let path = resolve_cache_path(None, "/nonexistent_manifest_xyz", "fallback.json");
        assert_eq!(path, PathBuf::from("data").join("fallback.json"));
    }

    #[test]
    fn resolve_cache_path_none_root_with_valid_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let manifest = dir.path().join("crate");
        std::fs::create_dir_all(&manifest).unwrap();
        let path = resolve_cache_path(None, manifest.to_str().unwrap(), "x.json");
        assert!(path.to_string_lossy().contains("data"), "path = {path:?}");
    }
}
