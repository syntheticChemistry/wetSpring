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

use crate::error::NcbiError;
use crate::nest::NestClient;

const EUTILS_BASE_DEFAULT: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";

fn eutils_base() -> String {
    std::env::var("WETSPRING_NCBI_EUTILS_URL").unwrap_or_else(|_| EUTILS_BASE_DEFAULT.to_owned())
}

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
    pub fn esearch(&self, db: &str, term: &str, retmax: u32) -> Result<SearchResult, NcbiError> {
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
        let base = eutils_base();
        let mut url =
            format!("{base}/esearch.fcgi?db={db}&term={encoded_term}&retmax={retmax}&retmode=xml");
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
    pub fn esummary(&self, db: &str, ids: &[String]) -> Result<SummaryResult, NcbiError> {
        if ids.is_empty() {
            return Ok(SummaryResult {
                raw_xml: String::new(),
            });
        }
        let id_list = ids.join(",");
        let base = eutils_base();
        let mut url = format!("{base}/esummary.fcgi?db={db}&id={id_list}&retmode=xml");
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
    pub fn efetch(&self, db: &str, id: &str, rettype: &str) -> Result<FetchResult, NcbiError> {
        let cache_key = format!("ncbi:efetch:{db}:{id}:{rettype}");

        if let Some(ref nest) = self.nest {
            if nest.exists(&cache_key) == Ok(true) {
                if let Ok(Some(data)) = nest.retrieve_blob(&cache_key) {
                    let content = String::from_utf8(data)
                        .map_err(|e| NcbiError::InvalidUtf8(e.to_string()))?;
                    return Ok(FetchResult {
                        content,
                        rettype: rettype.to_string(),
                    });
                }
            }
        }

        let base = eutils_base();
        let mut url = format!("{base}/efetch.fcgi?db={db}&id={id}&rettype={rettype}&retmode=text");
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
    ) -> Result<AssemblyResult, NcbiError> {
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
            return Err(NcbiError::AssemblyNotFound(accession.to_string()));
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

fn curl_get(url: &str) -> Result<String, NcbiError> {
    let output = Command::new("curl")
        .args(["-fsSL", "--max-time", "30", url])
        .output()
        .map_err(|e| NcbiError::HttpRequest(format!("curl not available: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(NcbiError::HttpRequest(format!(
            "curl failed ({}): {stderr}",
            output.status.code().unwrap_or(-1)
        )));
    }

    String::from_utf8(output.stdout).map_err(|e| NcbiError::InvalidUtf8(e.to_string()))
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
#[path = "ncbi_tests.rs"]
mod tests;
