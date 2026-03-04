// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(clippy::expect_used, clippy::unwrap_used)]

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
    let sock = std::env::temp_dir().join("fake_nestgate.sock");
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
    let local = AssemblySource::LocalFile(std::env::temp_dir().join("assembly.fna.gz"));
    assert_ne!(local, AssemblySource::NestCache);
    assert_ne!(local, AssemblySource::Fetched);
}
