// SPDX-License-Identifier: AGPL-3.0-or-later
//! NCBI Entrez E-fetch — download sequences in FASTA or `GenBank` format.
//!
//! Wraps the `EFetch` endpoint (`efetch.fcgi`) to retrieve full sequence
//! records by accession or UID. Uses the same capability-discovered HTTP
//! transport as [`super::http`].

use crate::error::Error;

const EFETCH_BASE: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";

/// Fetch a single FASTA record from NCBI by accession or UID.
///
/// # Arguments
///
/// * `db` — NCBI database (e.g. `"nucleotide"`, `"protein"`, `"sra"`)
/// * `id` — accession number or UID (e.g. `"K03455"`, `"PRJNA283159"`)
/// * `api_key` — NCBI API key (10 req/s with key, 3 req/s without)
///
/// # Errors
///
/// Returns `Err` if the HTTP request fails, NCBI returns an error page,
/// or no HTTP transport is available.
#[must_use = "fetched sequence is discarded if not used"]
pub fn efetch_fasta(db: &str, id: &str, api_key: &str) -> crate::error::Result<String> {
    let url = build_url(db, id, "fasta", "text", api_key);
    let body = super::http::get(&url)?;
    validate_fasta(&body)?;
    Ok(body)
}

/// Fetch a `GenBank` flat-file record from NCBI by accession or UID.
///
/// # Errors
///
/// Returns `Err` if the HTTP request fails or the response does not
/// contain a LOCUS line.
#[must_use = "fetched record is discarded if not used"]
pub fn efetch_genbank(db: &str, id: &str, api_key: &str) -> crate::error::Result<String> {
    let url = build_url(db, id, "gb", "text", api_key);
    let body = super::http::get(&url)?;
    if !body.contains("LOCUS") {
        return Err(Error::Ncbi(preview_msg(
            "response does not look like GenBank",
            &body,
        )));
    }
    Ok(body)
}

/// Fetch a batch of FASTA records (multiple UIDs, comma-separated).
///
/// # Errors
///
/// Returns `Err` if the HTTP request fails or the response contains no
/// FASTA headers.
#[must_use = "fetched sequences are discarded if not used"]
pub fn efetch_fasta_batch(db: &str, ids: &[&str], api_key: &str) -> crate::error::Result<String> {
    if ids.is_empty() {
        return Err(Error::Ncbi("empty ID list".to_string()));
    }
    let joined = ids.join(",");
    let url = build_url(db, &joined, "fasta", "text", api_key);
    let body = super::http::get(&url)?;
    validate_fasta(&body)?;
    Ok(body)
}

/// Build an `EFetch` URL with the given parameters.
fn build_url(db: &str, id: &str, rettype: &str, retmode: &str, api_key: &str) -> String {
    format!("{EFETCH_BASE}?db={db}&id={id}&rettype={rettype}&retmode={retmode}&api_key={api_key}")
}

/// Validate that a response body looks like FASTA (starts with `>`).
fn validate_fasta(body: &str) -> crate::error::Result<()> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('>') {
        Ok(())
    } else {
        Err(Error::Ncbi(preview_msg(
            "response does not look like FASTA",
            body,
        )))
    }
}

/// Build an error message with a truncated preview of the response body.
fn preview_msg(msg: &str, body: &str) -> String {
    let limit = body.len().min(crate::tolerances::ERROR_BODY_PREVIEW_LEN);
    format!("{msg}: {}", &body[..limit])
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn build_url_nucleotide() {
        let url = build_url("nucleotide", "K03455", "fasta", "text", "test_key");
        assert!(url.contains("efetch.fcgi"));
        assert!(url.contains("db=nucleotide"));
        assert!(url.contains("id=K03455"));
        assert!(url.contains("rettype=fasta"));
        assert!(url.contains("retmode=text"));
        assert!(url.contains("api_key=test_key"));
    }

    #[test]
    fn build_url_protein() {
        let url = build_url("protein", "AAA44325", "fasta", "text", "k");
        assert!(url.contains("db=protein"));
        assert!(url.contains("id=AAA44325"));
    }

    #[test]
    fn build_url_batch_ids() {
        let url = build_url("nucleotide", "K03455,M54321", "fasta", "text", "k");
        assert!(url.contains("id=K03455,M54321"));
    }

    #[test]
    fn validate_fasta_valid() {
        let body = ">seq1 description\nATCGATCG\n>seq2\nGGGGAAAA\n";
        assert!(validate_fasta(body).is_ok());
    }

    #[test]
    fn validate_fasta_with_leading_whitespace() {
        let body = "\n  >seq1\nATCG\n";
        assert!(validate_fasta(body).is_ok());
    }

    #[test]
    fn validate_fasta_html_error() {
        let body = "<!DOCTYPE html><html><body>Error</body></html>";
        let err = validate_fasta(body).unwrap_err();
        assert!(err.to_string().contains("does not look like FASTA"));
    }

    #[test]
    fn validate_fasta_empty() {
        let err = validate_fasta("").unwrap_err();
        assert!(err.to_string().contains("does not look like FASTA"));
    }

    #[test]
    fn validate_fasta_whitespace_only() {
        let err = validate_fasta("   \n\n  ").unwrap_err();
        assert!(err.to_string().contains("does not look like FASTA"));
    }

    #[test]
    fn efetch_fasta_batch_empty_ids_error() {
        let ids: Vec<&str> = vec![];
        let err = efetch_fasta_batch("nucleotide", &ids, "key").unwrap_err();
        assert!(err.to_string().contains("empty ID list"));
    }

    #[test]
    fn preview_msg_truncates() {
        let body = "x".repeat(1000);
        let msg = preview_msg("test", &body);
        assert!(msg.len() < 500);
    }

    #[test]
    fn preview_msg_short() {
        let msg = preview_msg("oops", "short");
        assert_eq!(msg, "oops: short");
    }
}
