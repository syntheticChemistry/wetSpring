// SPDX-License-Identifier: AGPL-3.0-or-later
//! NCBI Entrez E-fetch — download sequences in FASTA or `GenBank` format.
//!
//! Wraps the `EFetch` endpoint (`efetch.fcgi`) to retrieve full sequence
//! records by accession or UID. Uses the same capability-discovered HTTP
//! transport as [`super::http`].

use crate::error::Error;

const EFETCH_BASE_DEFAULT: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";

fn efetch_base() -> String {
    std::env::var("WETSPRING_NCBI_EFETCH_URL").unwrap_or_else(|_| EFETCH_BASE_DEFAULT.to_owned())
}

/// Injectable HTTP GET function — enables mock injection for testing.
pub type HttpGetFn = fn(&str) -> crate::error::Result<String>;

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
    efetch_fasta_with(db, id, api_key, super::http::get)
}

/// Like [`efetch_fasta`] but with an injectable HTTP transport.
///
/// # Errors
///
/// Returns `Err` if the HTTP transport fails or the response is not FASTA.
pub fn efetch_fasta_with(
    db: &str,
    id: &str,
    api_key: &str,
    http_get: HttpGetFn,
) -> crate::error::Result<String> {
    let url = build_url(db, id, "fasta", "text", api_key);
    let body = http_get(&url)?;
    validate_fasta(&body)?;
    Ok(body)
}

/// Validate that a response body looks like `GenBank` (contains `LOCUS`).
fn validate_genbank(body: &str) -> crate::error::Result<()> {
    if body.contains("LOCUS") {
        Ok(())
    } else {
        Err(Error::Ncbi(preview_msg(
            "response does not look like GenBank",
            body,
        )))
    }
}

/// Fetch a `GenBank` flat-file record from NCBI by accession or UID.
///
/// # Errors
///
/// Returns `Err` if the HTTP request fails or the response does not
/// contain a LOCUS line.
#[must_use = "fetched record is discarded if not used"]
pub fn efetch_genbank(db: &str, id: &str, api_key: &str) -> crate::error::Result<String> {
    efetch_genbank_with(db, id, api_key, super::http::get)
}

/// Like [`efetch_genbank`] but with an injectable HTTP transport.
///
/// # Errors
///
/// Returns `Err` if the HTTP transport fails or the response is not `GenBank`.
pub fn efetch_genbank_with(
    db: &str,
    id: &str,
    api_key: &str,
    http_get: HttpGetFn,
) -> crate::error::Result<String> {
    let url = build_url(db, id, "gb", "text", api_key);
    let body = http_get(&url)?;
    validate_genbank(&body)?;
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
    efetch_fasta_batch_with(db, ids, api_key, super::http::get)
}

/// Like [`efetch_fasta_batch`] but with an injectable HTTP transport.
///
/// # Errors
///
/// Returns `Err` if the IDs are empty, the HTTP transport fails, or
/// the response is not FASTA.
pub fn efetch_fasta_batch_with(
    db: &str,
    ids: &[&str],
    api_key: &str,
    http_get: HttpGetFn,
) -> crate::error::Result<String> {
    if ids.is_empty() {
        return Err(Error::Ncbi("empty ID list".to_string()));
    }
    let joined = ids.join(",");
    let url = build_url(db, &joined, "fasta", "text", api_key);
    let body = http_get(&url)?;
    validate_fasta(&body)?;
    Ok(body)
}

/// Build an `EFetch` URL with the given parameters.
fn build_url(db: &str, id: &str, rettype: &str, retmode: &str, api_key: &str) -> String {
    let base = efetch_base();
    format!("{base}?db={db}&id={id}&rettype={rettype}&retmode={retmode}&api_key={api_key}")
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
#[expect(clippy::unwrap_used)]
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

    #[test]
    fn build_url_empty_api_key() {
        let url = build_url("nucleotide", "K03455", "fasta", "text", "");
        assert!(url.contains("api_key="));
    }

    #[test]
    fn build_url_genbank_rettype() {
        let url = build_url("nucleotide", "NC_000913", "gb", "text", "k");
        assert!(url.contains("rettype=gb"));
        assert!(url.contains("retmode=text"));
    }

    #[test]
    fn validate_genbank_valid() {
        let body =
            "LOCUS       NC_000913             4641652 bp    DNA     circular\nDEFINITION  ...";
        assert!(validate_genbank(body).is_ok());
    }

    #[test]
    fn validate_genbank_html_error() {
        let body = "<!DOCTYPE html><html><body>Error</body></html>";
        let err = validate_genbank(body).unwrap_err();
        assert!(err.to_string().contains("does not look like GenBank"));
    }

    #[test]
    fn validate_genbank_empty() {
        let err = validate_genbank("").unwrap_err();
        assert!(err.to_string().contains("does not look like GenBank"));
    }

    #[test]
    fn validate_genbank_partial_locus() {
        let body = "Header line\nLOCUS       NC_000913";
        assert!(validate_genbank(body).is_ok());
    }

    #[test]
    fn preview_msg_empty_body() {
        let msg = preview_msg("error", "");
        assert_eq!(msg, "error: ");
    }

    #[expect(clippy::unnecessary_wraps)]
    fn mock_fasta_get(_url: &str) -> crate::error::Result<String> {
        Ok(">K03455.1 Human immunodeficiency virus\nATCGATCGATCG\n".to_string())
    }

    #[expect(clippy::unnecessary_wraps)]
    fn mock_genbank_get(_url: &str) -> crate::error::Result<String> {
        Ok("LOCUS       K03455  9719 bp    RNA\nDEFINITION  HIV-1\n".to_string())
    }

    fn mock_error_get(_url: &str) -> crate::error::Result<String> {
        Err(crate::error::Error::Ncbi("connection refused".to_string()))
    }

    #[expect(clippy::unnecessary_wraps)]
    fn mock_html_error_get(_url: &str) -> crate::error::Result<String> {
        Ok("<!DOCTYPE html><html><body>Error</body></html>".to_string())
    }

    #[test]
    fn efetch_fasta_with_mock_success() {
        let result = efetch_fasta_with("nucleotide", "K03455", "key", mock_fasta_get);
        let fasta = result.unwrap();
        assert!(fasta.starts_with('>'));
        assert!(fasta.contains("ATCG"));
    }

    #[test]
    fn efetch_fasta_with_mock_network_error() {
        let err = efetch_fasta_with("nucleotide", "K03455", "key", mock_error_get).unwrap_err();
        assert!(err.to_string().contains("connection refused"));
    }

    #[test]
    fn efetch_fasta_with_mock_html_error() {
        let err =
            efetch_fasta_with("nucleotide", "K03455", "key", mock_html_error_get).unwrap_err();
        assert!(err.to_string().contains("does not look like FASTA"));
    }

    #[test]
    fn efetch_genbank_with_mock_success() {
        let result = efetch_genbank_with("nucleotide", "K03455", "key", mock_genbank_get);
        let gb = result.unwrap();
        assert!(gb.contains("LOCUS"));
    }

    #[test]
    fn efetch_genbank_with_mock_html_error() {
        let err =
            efetch_genbank_with("nucleotide", "K03455", "key", mock_html_error_get).unwrap_err();
        assert!(err.to_string().contains("does not look like GenBank"));
    }

    #[test]
    fn efetch_fasta_batch_with_mock_success() {
        let result =
            efetch_fasta_batch_with("nucleotide", &["K03455", "M54321"], "key", mock_fasta_get);
        assert!(result.is_ok());
    }

    #[test]
    fn efetch_fasta_batch_with_mock_empty_ids() {
        let err = efetch_fasta_batch_with("nucleotide", &[], "key", mock_fasta_get).unwrap_err();
        assert!(err.to_string().contains("empty ID list"));
    }

    #[test]
    fn efetch_fasta_batch_with_mock_error() {
        let err =
            efetch_fasta_batch_with("nucleotide", &["K03455"], "key", mock_error_get).unwrap_err();
        assert!(err.to_string().contains("connection refused"));
    }
}
