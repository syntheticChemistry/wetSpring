// SPDX-License-Identifier: AGPL-3.0-or-later

//! NCBI E-utilities acquisition via the Tower atomic pattern.
//!
//! Acquires genomic data from NCBI using the Tower → Nest flow:
//! 1. **Tower** discovers `NestGate` via Songbird
//! 2. **Nest** checks if data is already stored (`storage.exists`)
//! 3. If not cached, fetches directly from NCBI E-utilities via `curl`
//! 4. Stores the result in `NestGate` for future use
//!
//! This avoids adding an HTTP client dependency to the forge crate. NCBI
//! HTTP calls are delegated to `curl` (available on all target systems).
//! When `NestGate` evolves to expose `ncbi.fetch` via RPC, the direct
//! curl path becomes a fallback.
//!
//! # Rate Limiting
//!
//! NCBI allows ~3 req/s without an API key, ~10 req/s with one.
//! The `NCBI_API_KEY` and `NCBI_EMAIL` env vars are forwarded to requests
//! when available.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::nest::NestClient;

const EUTILS_BASE: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";

/// NCBI acquisition client using the Tower → Nest pattern.
#[derive(Debug)]
pub struct NcbiClient {
    /// `NestGate` client for caching (if available).
    nest: Option<NestClient>,
    /// Optional API key for higher rate limits.
    api_key: Option<String>,
    /// Contact email (required by NCBI policy).
    email: Option<String>,
}

/// Result of an NCBI search (`ESearch`).
#[derive(Debug)]
pub struct SearchResult {
    /// Database searched.
    pub db: String,
    /// Total count of matching records.
    pub count: u64,
    /// IDs returned (up to `retmax`).
    pub ids: Vec<String>,
    /// Raw XML response.
    pub raw_xml: String,
}

/// Result of an NCBI summary (`ESummary`).
#[derive(Debug)]
pub struct SummaryResult {
    /// Document summaries (raw XML).
    pub raw_xml: String,
}

/// Result of an NCBI fetch (`EFetch`).
#[derive(Debug)]
pub struct FetchResult {
    /// Fetched content (`FASTA`, `GenBank`, etc.).
    pub content: String,
    /// Return type used.
    pub rettype: String,
}

/// Where an assembly was resolved from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssemblySource {
    /// Found in `NestGate` cache.
    NestCache,
    /// Found on local filesystem.
    LocalFile(PathBuf),
    /// Fetched from NCBI.
    Fetched,
}

/// Result of assembly acquisition.
#[derive(Debug)]
pub struct AssemblyResult {
    /// The accession requested.
    pub accession: String,
    /// Where the data came from.
    pub source: AssemblySource,
    /// File path (if saved to disk or found locally).
    pub path: Option<PathBuf>,
    /// Byte count of the assembly data.
    pub size_bytes: u64,
}

impl NcbiClient {
    /// Create a new NCBI client with Tower discovery.
    ///
    /// Attempts to discover `NestGate` for caching. Falls back to
    /// direct NCBI access if `NestGate` is unavailable.
    #[must_use]
    pub fn discover() -> Self {
        Self {
            nest: NestClient::discover(),
            api_key: std::env::var("NCBI_API_KEY").ok(),
            email: std::env::var("NCBI_EMAIL")
                .or_else(|_| std::env::var("USER_EMAIL"))
                .or_else(|_| std::env::var("EMAIL"))
                .ok(),
        }
    }

    /// Create a client with explicit `NestGate` socket.
    #[must_use]
    pub fn with_nest(nest: NestClient) -> Self {
        Self {
            nest: Some(nest),
            api_key: std::env::var("NCBI_API_KEY").ok(),
            email: std::env::var("NCBI_EMAIL").ok(),
        }
    }

    /// Create a client without `NestGate` (direct NCBI only).
    #[must_use]
    pub fn direct() -> Self {
        Self {
            nest: None,
            api_key: std::env::var("NCBI_API_KEY").ok(),
            email: std::env::var("NCBI_EMAIL").ok(),
        }
    }

    /// Whether this client has `NestGate` caching available.
    #[must_use]
    pub const fn has_nest(&self) -> bool {
        self.nest.is_some()
    }

    /// Search NCBI via `ESearch`.
    ///
    /// Returns matching IDs for the given database and query term.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails, the response is invalid UTF-8,
    /// or `NestGate` storage operations fail when caching.
    pub fn esearch(&self, db: &str, term: &str, retmax: u32) -> Result<SearchResult, String> {
        let cache_key = format!("ncbi:esearch:{db}:{term}:{retmax}");

        if let Some(ref nest) = self.nest {
            if nest.exists(&cache_key) == Ok(true) {
                if let Ok(result) = nest.retrieve(&cache_key) {
                    if let Some(xml) = result.value {
                        return Ok(parse_esearch_result(db, &xml));
                    }
                }
            }
        }

        let encoded_term = url_encode(term);
        let mut url = format!(
            "{EUTILS_BASE}/esearch.fcgi?db={db}&term={encoded_term}&retmax={retmax}&retmode=xml"
        );
        self.append_auth(&mut url);

        let xml = curl_get(&url)?;

        if let Some(ref nest) = self.nest {
            let escaped = xml.replace('\\', "\\\\").replace('"', "\\\"");
            let _ = nest.store(&cache_key, &format!("\"{escaped}\""));
        }

        Ok(parse_esearch_result(db, &xml))
    }

    /// Fetch summaries via `ESummary`.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the response is invalid UTF-8.
    pub fn esummary(&self, db: &str, ids: &[String]) -> Result<SummaryResult, String> {
        if ids.is_empty() {
            return Ok(SummaryResult {
                raw_xml: String::new(),
            });
        }
        let id_list = ids.join(",");
        let mut url = format!("{EUTILS_BASE}/esummary.fcgi?db={db}&id={id_list}&retmode=xml");
        self.append_auth(&mut url);

        let raw_xml = curl_get(&url)?;
        Ok(SummaryResult { raw_xml })
    }

    /// Fetch sequence data via `EFetch`.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails, the response is invalid UTF-8,
    /// or `NestGate` storage operations fail when caching.
    pub fn efetch(&self, db: &str, id: &str, rettype: &str) -> Result<FetchResult, String> {
        let cache_key = format!("ncbi:efetch:{db}:{id}:{rettype}");

        if let Some(ref nest) = self.nest {
            if nest.exists(&cache_key) == Ok(true) {
                if let Ok(Some(data)) = nest.retrieve_blob(&cache_key) {
                    let content = String::from_utf8(data).map_err(|e| format!("utf8: {e}"))?;
                    return Ok(FetchResult {
                        content,
                        rettype: rettype.to_string(),
                    });
                }
            }
        }

        let mut url =
            format!("{EUTILS_BASE}/efetch.fcgi?db={db}&id={id}&rettype={rettype}&retmode=text");
        self.append_auth(&mut url);

        let content = curl_get(&url)?;

        if let Some(ref nest) = self.nest {
            let _ = nest.store_blob(&cache_key, content.as_bytes());
        }

        Ok(FetchResult {
            content,
            rettype: rettype.to_string(),
        })
    }

    /// Acquire a genome assembly by accession.
    ///
    /// Resolution chain:
    /// 1. Check `NestGate` cache
    /// 2. Check local data directory
    /// 3. Fetch assembly summary from NCBI for metadata
    ///
    /// For actual assembly `FASTA` downloads (multi-GB), use the dedicated
    /// download scripts in `scripts/`. This method fetches metadata and
    /// verifies availability.
    ///
    /// # Errors
    ///
    /// Returns an error if the assembly is not found in NCBI, or if `ESearch`
    /// or local filesystem operations fail.
    pub fn acquire_assembly(
        &self,
        accession: &str,
        local_dir: Option<&Path>,
    ) -> Result<AssemblyResult, String> {
        let nest_key = format!("data:assembly:{accession}");

        if let Some(ref nest) = self.nest {
            if nest.exists(&nest_key) == Ok(true) {
                return Ok(AssemblyResult {
                    accession: accession.to_string(),
                    source: AssemblySource::NestCache,
                    path: None,
                    size_bytes: 0,
                });
            }
        }

        if let Some(dir) = local_dir {
            let filename = format!("{accession}.fna.gz");
            let path = dir.join(&filename);
            if path.exists() {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                return Ok(AssemblyResult {
                    accession: accession.to_string(),
                    source: AssemblySource::LocalFile(path.clone()),
                    path: Some(path),
                    size_bytes: size,
                });
            }
        }

        let search = self.esearch("assembly", accession, 1)?;
        if search.count == 0 {
            return Err(format!("assembly {accession} not found in NCBI"));
        }

        Ok(AssemblyResult {
            accession: accession.to_string(),
            source: AssemblySource::Fetched,
            path: None,
            size_bytes: 0,
        })
    }

    fn append_auth(&self, url: &mut String) {
        use std::fmt::Write;
        if let Some(ref key) = self.api_key {
            let _ = write!(url, "&api_key={key}");
        }
        if let Some(ref email) = self.email {
            let _ = write!(url, "&tool=wetspring-forge&email={email}");
        }
    }
}

// ── HTTP via curl ───────────────────────────────────────────────────

fn curl_get(url: &str) -> Result<String, String> {
    let output = Command::new("curl")
        .args(["-fsSL", "--max-time", "30", url])
        .output()
        .map_err(|e| format!("curl not available: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "curl failed ({}): {stderr}",
            output.status.code().unwrap_or(-1)
        ));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("invalid utf8: {e}"))
}

fn url_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            b' ' => result.push('+'),
            _ => {
                result.push('%');
                result.push(hex_char(byte >> 4));
                result.push(hex_char(byte & 0xF));
            }
        }
    }
    result
}

const fn hex_char(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'A' + nibble - 10) as char,
    }
}

// ── XML parsing (minimal, no serde/xml dep) ─────────────────────────

fn parse_esearch_result(db: &str, xml: &str) -> SearchResult {
    let count = extract_xml_tag(xml, "Count")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let ids = extract_xml_tags(xml, "Id");

    SearchResult {
        db: db.to_string(),
        count,
        ids,
        raw_xml: xml.to_string(),
    }
}

fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let content_start = start + open.len();
    let end = xml[content_start..].find(&close)?;
    Some(xml[content_start..content_start + end].to_string())
}

fn extract_xml_tags(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut results = Vec::new();
    let mut search_from = 0;

    while let Some(start) = xml[search_from..].find(&open) {
        let abs_start = search_from + start + open.len();
        if let Some(end) = xml[abs_start..].find(&close) {
            results.push(xml[abs_start..abs_start + end].to_string());
            search_from = abs_start + end + close.len();
        } else {
            break;
        }
    }
    results
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn url_encode_simple() {
        assert_eq!(url_encode("hello world"), "hello+world");
        assert_eq!(url_encode("Vibrio[Organism]"), "Vibrio%5BOrganism%5D");
    }

    #[test]
    fn url_encode_passthrough() {
        assert_eq!(url_encode("simple-term"), "simple-term");
        assert_eq!(url_encode("test_123"), "test_123");
    }

    #[test]
    fn parse_esearch_xml() {
        let xml = r#"<?xml version="1.0" ?>
<eSearchResult>
  <Count>42</Count>
  <RetMax>3</RetMax>
  <IdList>
    <Id>12345</Id>
    <Id>67890</Id>
    <Id>11111</Id>
  </IdList>
</eSearchResult>"#;
        let result = parse_esearch_result("assembly", xml);
        assert_eq!(result.db, "assembly");
        assert_eq!(result.count, 42);
        assert_eq!(result.ids.len(), 3);
        assert_eq!(result.ids[0], "12345");
    }

    #[test]
    fn parse_esearch_empty() {
        let xml = "<eSearchResult><Count>0</Count><IdList></IdList></eSearchResult>";
        let result = parse_esearch_result("nucleotide", xml);
        assert_eq!(result.count, 0);
        assert!(result.ids.is_empty());
    }

    #[test]
    fn extract_xml_tag_basic() {
        let xml = "<root><Count>42</Count></root>";
        assert_eq!(extract_xml_tag(xml, "Count"), Some("42".to_string()));
        assert!(extract_xml_tag(xml, "Missing").is_none());
    }

    #[test]
    fn extract_xml_tags_multiple() {
        let xml = "<IdList><Id>1</Id><Id>2</Id><Id>3</Id></IdList>";
        let ids = extract_xml_tags(xml, "Id");
        assert_eq!(ids, vec!["1", "2", "3"]);
    }

    #[test]
    fn ncbi_client_direct_has_no_nest() {
        let client = NcbiClient::direct();
        assert!(!client.has_nest());
    }

    #[test]
    fn ncbi_client_discover_checks_env() {
        let client = NcbiClient::discover();
        let _ = client.has_nest();
    }

    #[test]
    fn assembly_source_equality() {
        assert_eq!(AssemblySource::NestCache, AssemblySource::NestCache);
        assert_eq!(AssemblySource::Fetched, AssemblySource::Fetched);
        assert_ne!(AssemblySource::NestCache, AssemblySource::Fetched);
    }

    #[test]
    fn hex_char_values() {
        assert_eq!(hex_char(0), '0');
        assert_eq!(hex_char(9), '9');
        assert_eq!(hex_char(10), 'A');
        assert_eq!(hex_char(15), 'F');
    }

    #[test]
    fn url_encode_special_chars() {
        assert_eq!(url_encode("a&b=c"), "a%26b%3Dc");
        assert_eq!(url_encode("100%"), "100%25");
    }

    #[test]
    fn url_encode_empty() {
        assert_eq!(url_encode(""), "");
    }

    #[test]
    fn parse_esearch_missing_count() {
        let xml = "<eSearchResult><IdList><Id>1</Id></IdList></eSearchResult>";
        let result = parse_esearch_result("nucleotide", xml);
        assert_eq!(result.count, 0);
        assert_eq!(result.ids.len(), 1);
    }

    #[test]
    fn extract_xml_tag_nested() {
        let xml = "<root><outer><Count>10</Count></outer></root>";
        assert_eq!(extract_xml_tag(xml, "Count"), Some("10".to_string()));
    }

    #[test]
    fn extract_xml_tags_empty_list() {
        let xml = "<IdList></IdList>";
        let ids = extract_xml_tags(xml, "Id");
        assert!(ids.is_empty());
    }

    #[test]
    fn extract_xml_tags_no_match() {
        let xml = "<root><a>1</a></root>";
        let tags = extract_xml_tags(xml, "b");
        assert!(tags.is_empty());
    }

    #[test]
    fn ncbi_client_with_nest() {
        let sock = PathBuf::from("/tmp/fake_nestgate.sock");
        let nest = crate::nest::NestClient::new(sock);
        let client = NcbiClient::with_nest(nest);
        assert!(client.has_nest());
    }

    #[test]
    fn esummary_empty_ids() {
        let client = NcbiClient::direct();
        let result = client.esummary("assembly", &[]);
        assert!(result.is_ok());
        assert!(result.unwrap().raw_xml.is_empty());
    }

    #[test]
    fn assembly_source_display() {
        let local = AssemblySource::LocalFile(PathBuf::from("/data/assembly.fna.gz"));
        assert_ne!(local, AssemblySource::NestCache);
        assert_ne!(local, AssemblySource::Fetched);
    }
}
