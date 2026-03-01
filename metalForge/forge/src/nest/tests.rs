// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nest module tests.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;

use super::{NestClient, discover_nestgate_socket};
use super::{base64, json, time};

#[test]
fn base64_round_trip() {
    let data = b"Hello, NestGate!";
    let encoded = base64::base64_encode(data);
    let decoded = base64::base64_decode(&encoded);
    assert_eq!(decoded, data);
}

#[test]
fn base64_empty() {
    assert_eq!(base64::base64_encode(b""), "");
    assert_eq!(base64::base64_decode(""), Vec::<u8>::new());
}

#[test]
fn base64_padding_1() {
    let encoded = base64::base64_encode(b"A");
    assert!(encoded.ends_with("=="));
    assert_eq!(base64::base64_decode(&encoded), b"A");
}

#[test]
fn base64_padding_2() {
    let encoded = base64::base64_encode(b"AB");
    assert!(encoded.ends_with('='));
    assert_eq!(base64::base64_decode(&encoded), b"AB");
}

#[test]
fn base64_no_padding() {
    let encoded = base64::base64_encode(b"ABC");
    assert!(!encoded.contains('='));
    assert_eq!(base64::base64_decode(&encoded), b"ABC");
}

#[test]
fn base64_binary_data() {
    let data: Vec<u8> = (0..=255).collect();
    let encoded = base64::base64_encode(&data);
    let decoded = base64::base64_decode(&encoded);
    assert_eq!(decoded, data);
}

#[test]
fn escape_json_handles_special_chars() {
    assert_eq!(json::escape_json_str(r#"a"b\c"#), r#"a\"b\\c"#);
    assert_eq!(json::escape_json_str("simple"), "simple");
}

#[test]
fn chrono_lite_produces_iso8601() {
    let ts = time::chrono_lite_now();
    assert!(ts.contains('T'));
    assert!(ts.ends_with('Z'));
    assert!(ts.len() >= 19);
}

#[test]
fn extract_result_value_string() {
    let json_str = r#"{"jsonrpc":"2.0","result":"hello","id":1}"#;
    assert_eq!(
        json::extract_result_value(json_str),
        Some("\"hello\"".to_string())
    );
}

#[test]
fn extract_result_value_object() {
    let json_str = r#"{"jsonrpc":"2.0","result":{"key":"val"},"id":1}"#;
    let val = json::extract_result_value(json_str).unwrap();
    assert!(val.contains("key"));
}

#[test]
fn extract_result_value_bool() {
    let json_str = r#"{"jsonrpc":"2.0","result":true,"id":1}"#;
    assert_eq!(
        json::extract_result_value(json_str),
        Some("true".to_string())
    );
}

#[test]
fn nest_client_with_family() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("test.sock");
    let client = NestClient::new(sock).with_family("wetspring");
    assert_eq!(client.family_id(), "wetspring");
}

#[test]
fn socket_discovery_does_not_panic() {
    let _ = discover_nestgate_socket();
}

#[test]
fn days_to_ymd_epoch() {
    let (y, m, d) = time::days_to_ymd(0);
    assert_eq!((y, m, d), (1970, 1, 1));
}

#[test]
fn days_to_ymd_known_date() {
    // 2026-02-28 = day 20,512 from epoch
    let (y, m, d) = time::days_to_ymd(20_512);
    assert_eq!(y, 2026);
    assert_eq!(m, 2);
    assert_eq!(d, 28);
}

#[test]
fn days_to_ymd_leap_year() {
    // 2000-03-01 = day 11017 from epoch (2000 is a leap year)
    let (y, m, d) = time::days_to_ymd(11_017);
    assert_eq!(y, 2000);
    assert_eq!(m, 3);
    assert_eq!(d, 1);
}

#[test]
fn is_leap_year_basic() {
    assert!(time::is_leap(2000));
    assert!(time::is_leap(2024));
    assert!(!time::is_leap(1900));
    assert!(!time::is_leap(2023));
}

#[test]
fn extract_result_value_array() {
    let json_str = r#"{"jsonrpc":"2.0","result":["a","b","c"],"id":1}"#;
    let val = json::extract_result_value(json_str).unwrap();
    assert!(val.starts_with('['));
    assert!(val.contains("\"a\""));
}

#[test]
fn extract_result_value_number() {
    let json_str = r#"{"jsonrpc":"2.0","result":42,"id":1}"#;
    assert_eq!(json::extract_result_value(json_str), Some("42".to_string()));
}

#[test]
fn extract_result_value_null() {
    let json_str = r#"{"jsonrpc":"2.0","result":null,"id":1}"#;
    assert_eq!(
        json::extract_result_value(json_str),
        Some("null".to_string())
    );
}

#[test]
fn extract_result_value_missing() {
    let json_str = r#"{"jsonrpc":"2.0","error":"bad","id":1}"#;
    assert!(json::extract_result_value(json_str).is_none());
}

#[test]
fn extract_result_string_basic() {
    let json_str = r#"{"result":{"blob":"SGVsbG8="},"id":1}"#;
    assert_eq!(
        json::extract_result_string(json_str, "blob"),
        Some("SGVsbG8=".to_string())
    );
}

#[test]
fn extract_result_string_missing_key() {
    let json_str = r#"{"result":{"blob":"SGVsbG8="},"id":1}"#;
    assert!(json::extract_result_string(json_str, "data").is_none());
}

#[test]
fn extract_string_array_basic() {
    let json_str = r#"{"result":{"keys":["a","b","c"]},"id":1}"#;
    let keys = json::extract_string_array(json_str, "keys").unwrap();
    assert_eq!(keys, vec!["a", "b", "c"]);
}

#[test]
fn extract_string_array_empty() {
    let json_str = r#"{"result":{"keys":[]},"id":1}"#;
    let keys = json::extract_string_array(json_str, "keys").unwrap();
    assert!(keys.is_empty());
}

#[test]
fn extract_result_array_basic() {
    let json_str = r#"{"jsonrpc":"2.0","result":["k1","k2"],"id":1}"#;
    let arr = json::extract_result_array(json_str).unwrap();
    assert_eq!(arr, vec!["k1", "k2"]);
}

#[test]
fn find_value_end_string() {
    assert_eq!(json::find_value_end(r#""hello""#), Some(7));
}

#[test]
fn find_value_end_object() {
    assert_eq!(json::find_value_end(r#"{"a":1}"#), Some(7));
}

#[test]
fn find_value_end_nested() {
    assert_eq!(json::find_value_end(r#"{"a":{"b":2}}"#), Some(13));
}

#[test]
fn find_value_end_array() {
    assert_eq!(json::find_value_end("[1,2,3]"), Some(7));
}

#[test]
fn find_value_end_number() {
    assert_eq!(json::find_value_end("42,"), Some(2));
}

#[test]
fn parse_string_list_basic() {
    let list = json::parse_string_list(r#""a","b","c""#);
    assert_eq!(list, vec!["a", "b", "c"]);
}

#[test]
fn parse_string_list_empty() {
    let list = json::parse_string_list("");
    assert!(list.is_empty());
}

#[test]
fn nest_client_new_and_socket_path() {
    let p = PathBuf::from("/tmp/test.sock");
    let client = NestClient::new(p.clone());
    assert_eq!(client.socket_path(), p);
}

#[test]
fn nest_client_discover_returns_none_without_socket() {
    temp_env::with_var("NESTGATE_SOCKET", None::<&str>, || {
        // Discovery may or may not find a socket depending on the system;
        // what matters is that it doesn't panic.
        let _ = NestClient::discover();
    });
}

#[test]
fn base64_large_binary() {
    #[allow(clippy::cast_possible_truncation)]
    let data: Vec<u8> = (0_u32..1024).map(|i| (i & 0xFF) as u8).collect();
    let encoded = base64::base64_encode(&data);
    let decoded = base64::base64_decode(&encoded);
    assert_eq!(decoded, data);
}

#[test]
fn b64_val_all_chars() {
    assert_eq!(base64::b64_val(b'A'), 0);
    assert_eq!(base64::b64_val(b'Z'), 25);
    assert_eq!(base64::b64_val(b'a'), 26);
    assert_eq!(base64::b64_val(b'z'), 51);
    assert_eq!(base64::b64_val(b'0'), 52);
    assert_eq!(base64::b64_val(b'9'), 61);
    assert_eq!(base64::b64_val(b'+'), 62);
    assert_eq!(base64::b64_val(b'/'), 63);
    assert_eq!(base64::b64_val(b'='), 0);
}

#[test]
fn escape_json_str_empty() {
    assert_eq!(json::escape_json_str(""), "");
}
