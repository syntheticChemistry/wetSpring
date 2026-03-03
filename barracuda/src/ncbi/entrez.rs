// SPDX-License-Identifier: AGPL-3.0-or-later
//! NCBI Entrez E-utilities (E-search) wrappers.

const ENTREZ_BASE_DEFAULT: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";

fn entrez_base() -> String {
    std::env::var("WETSPRING_NCBI_ESEARCH_URL").unwrap_or_else(|_| ENTREZ_BASE_DEFAULT.to_owned())
}

/// URL-encode a search term for Entrez E-utilities.
fn encode_term(term: &str) -> String {
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

use crate::error::Error;

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
pub fn esearch_count(db: &str, term: &str, api_key: &str) -> crate::error::Result<u64> {
    let encoded = encode_term(term);
    let base = entrez_base();
    let url = format!("{base}?db={db}&term={encoded}&rettype=count&api_key={api_key}");
    let body = super::http::get(&url)?;
    parse_count(&body)
}

/// Parse `<Count>...</Count>` from an Entrez E-search XML response body.
fn parse_count(body: &str) -> crate::error::Result<u64> {
    let start = body
        .find("<Count>")
        .ok_or_else(|| Error::Ncbi(preview_msg("no <Count> in response", body)))?;
    let rest = &body[start + 7..];
    let end = rest
        .find("</Count>")
        .ok_or_else(|| Error::Ncbi(preview_msg("unclosed <Count> tag", body)))?;
    rest[..end]
        .trim()
        .parse::<u64>()
        .map_err(|e| Error::Ncbi(format!("invalid count value: {e}")))
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
    fn encode_term_spaces_and_brackets() {
        let encoded = encode_term("luxI[gene] AND (biofilm)");
        assert_eq!(encoded, "luxI%5Bgene%5D+AND+%28biofilm%29");
    }

    #[test]
    fn encode_term_quotes() {
        let encoded = encode_term("\"quorum sensing\"");
        assert_eq!(encoded, "%22quorum+sensing%22");
    }

    #[test]
    fn encode_term_empty() {
        assert_eq!(encode_term(""), "");
    }

    #[test]
    fn encode_term_no_special_chars() {
        assert_eq!(encode_term("simple"), "simple");
    }

    #[test]
    fn encode_term_all_special() {
        let encoded = encode_term("\"a\" [b] (c) d e");
        assert_eq!(encoded, "%22a%22+%5Bb%5D+%28c%29+d+e");
    }

    #[test]
    fn encode_term_ampersand_and_hash() {
        let encoded = encode_term("a&b#c");
        assert_eq!(encoded, "a%26b%23c");
    }

    #[test]
    fn encode_term_unicode() {
        let encoded = encode_term("café naïve 日本語");
        assert_eq!(encoded, "café+naïve+日本語");
    }

    #[test]
    fn parse_count_valid() {
        let xml = r#"<?xml version="1.0"?>
<eSearchResult><Count>42</Count><RetMax>0</RetMax></eSearchResult>"#;
        assert_eq!(parse_count(xml).unwrap(), 42);
    }

    #[test]
    fn parse_count_with_whitespace() {
        let xml = "<eSearchResult><Count>  1234  </Count></eSearchResult>";
        assert_eq!(parse_count(xml).unwrap(), 1234);
    }

    #[test]
    fn parse_count_missing_tag() {
        let xml = "<eSearchResult><RetMax>0</RetMax></eSearchResult>";
        assert!(parse_count(xml).is_err());
    }

    #[test]
    fn parse_count_empty_body() {
        assert!(parse_count("").is_err());
    }

    #[test]
    fn parse_count_unclosed_tag() {
        let xml = "<eSearchResult><Count>99";
        assert!(parse_count(xml).is_err());
    }

    #[test]
    fn parse_count_non_numeric() {
        let xml = "<Count>abc</Count>";
        assert!(parse_count(xml).is_err());
    }

    #[test]
    fn parse_count_zero() {
        let xml = "<Count>0</Count>";
        assert_eq!(parse_count(xml).unwrap(), 0);
    }

    #[test]
    fn parse_count_large_value() {
        let xml = "<Count>9999999999</Count>";
        assert_eq!(parse_count(xml).unwrap(), 9_999_999_999);
    }

    #[test]
    fn parse_count_multiline_xml() {
        let xml = r#"<?xml version="1.0"?>
<eSearchResult>
  <Count>
    999
  </Count>
  <RetMax>0</RetMax>
</eSearchResult>"#;
        assert_eq!(parse_count(xml).unwrap(), 999);
    }

    #[test]
    fn preview_msg_truncates_long_body() {
        let body = "x".repeat(1000);
        let msg = preview_msg("test", &body);
        assert!(msg.len() < 500);
    }

    #[test]
    fn preview_msg_short_body() {
        let msg = preview_msg("oops", "short");
        assert_eq!(msg, "oops: short");
    }

    #[test]
    fn preview_msg_empty_body() {
        let msg = preview_msg("no count", "");
        assert_eq!(msg, "no count: ");
    }

    #[test]
    fn esearch_url_construction() {
        let encoded = encode_term("luxI[gene]");
        let base = entrez_base();
        let url = format!("{base}?db=nuccore&term={encoded}&rettype=count&api_key=test_key");
        assert!(url.contains("esearch.fcgi"));
        assert!(url.contains("db=nuccore"));
        assert!(url.contains("luxI%5Bgene%5D"));
        assert!(url.contains("api_key=test_key"));
    }
}
