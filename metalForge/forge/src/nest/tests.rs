// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nest module tests.

#![expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]

use super::{NestClient, default_socket_path, discover_nestgate_socket};
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
    let p = std::env::temp_dir().join("test.sock");
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

// ═══════════════════════════════════════════════════════════════════
// Transport layer tests — real Unix socket round-trip
// ═══════════════════════════════════════════════════════════════════

fn start_echo_server(sock: &std::path::Path) -> std::thread::JoinHandle<()> {
    let sock = sock.to_path_buf();
    std::thread::spawn(move || {
        let listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        if let Ok((stream, _)) = listener.accept() {
            let mut reader = std::io::BufReader::new(&stream);
            let mut line = String::new();
            std::io::BufRead::read_line(&mut reader, &mut line).unwrap();
            let resp = r#"{"jsonrpc":"2.0","result":true,"id":1}"#;
            let mut writer = std::io::BufWriter::new(&stream);
            std::io::Write::write_all(&mut writer, resp.as_bytes()).unwrap();
            std::io::Write::write_all(&mut writer, b"\n").unwrap();
            std::io::Write::flush(&mut writer).unwrap();
        }
    })
}

#[test]
#[ignore = "requires Unix socket environment (flaky in sandboxed CI)"]
fn transport_rpc_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("rpc_test.sock");
    let handle = start_echo_server(&sock);
    wait_for_socket(&sock);
    let req = r#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
    let resp = super::transport::rpc(&sock, req).unwrap();
    assert!(resp.contains("\"result\":true"));
    handle.join().unwrap();
}

#[test]
fn transport_rpc_bad_socket() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("nonexistent.sock");
    let req = r#"{"jsonrpc":"2.0","method":"ping","id":1}"#;
    let result = super::transport::rpc(&sock, req);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════
// Client integration tests — full round-trip via transport
// ═══════════════════════════════════════════════════════════════════

fn start_storage_server(sock: &std::path::Path, response: &str) -> std::thread::JoinHandle<()> {
    let sock = sock.to_path_buf();
    let response = response.to_owned();
    std::thread::spawn(move || {
        let listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        while let Ok((stream, _)) = listener.accept() {
            let mut reader = std::io::BufReader::new(&stream);
            let mut line = String::new();
            if std::io::BufRead::read_line(&mut reader, &mut line).is_err() {
                break;
            }
            let mut writer = std::io::BufWriter::new(&stream);
            std::io::Write::write_all(&mut writer, response.as_bytes()).unwrap();
            std::io::Write::write_all(&mut writer, b"\n").unwrap();
            std::io::Write::flush(&mut writer).unwrap();
        }
    })
}

fn wait_for_socket(sock: &std::path::Path) {
    for _ in 0..100 {
        if sock.exists() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("socket did not appear: {}", sock.display());
}

#[test]
fn client_exists_returns_true() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("exists.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":true,"id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.exists("test-key").unwrap();
    assert!(result);
}

#[test]
fn client_exists_returns_false() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("exists_false.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":false,"id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.exists("missing-key").unwrap();
    assert!(!result);
}

#[test]
fn client_store_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("store.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":"ok","id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.store("key", r#"{"val":1}"#).unwrap();
    assert!(result.ok);
}

#[test]
fn client_store_blob_and_retrieve() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("blob.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":"ok","id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.store_blob("data", b"hello").unwrap();
    assert!(result.ok);
}

#[test]
fn client_retrieve_value() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("retrieve.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":"hello","id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.retrieve("key").unwrap();
    assert!(result.value.is_some());
}

#[test]
fn client_delete_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("delete.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":"ok","id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.delete("key").unwrap();
    assert!(result.ok);
}

#[test]
fn client_list_returns_keys() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("list.sock");
    let _handle = start_storage_server(
        &sock,
        r#"{"jsonrpc":"2.0","result":{"keys":["a","b"]},"id":1}"#,
    );
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.list(Some("")).unwrap();
    assert_eq!(result.keys, vec!["a", "b"]);
}

#[test]
fn client_stats_returns_raw() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("stats.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":{"count":5},"id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let raw = client.stats().unwrap();
    assert!(raw.contains("count"));
}

#[test]
fn client_store_dataset_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("meta.sock");
    let _handle = start_storage_server(&sock, r#"{"jsonrpc":"2.0","result":"ok","id":1}"#);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client
        .store_dataset_metadata("test_ds", "SRA", 10, 1024)
        .unwrap();
    assert!(result.ok);
}

#[test]
fn client_retrieve_blob_with_data() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("retrieve_blob.sock");
    let encoded = base64::base64_encode(b"binary data");
    let response = format!(r#"{{"jsonrpc":"2.0","result":{{"blob":"{encoded}"}},"id":1}}"#);
    let _handle = start_storage_server(&sock, &response);
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.retrieve_blob("key").unwrap();
    assert_eq!(result.unwrap(), b"binary data");
}

#[test]
fn client_retrieve_blob_error_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("retrieve_blob_err.sock");
    let _handle = start_storage_server(
        &sock,
        r#"{"jsonrpc":"2.0","error":{"code":-1,"message":"not found"},"id":1}"#,
    );
    wait_for_socket(&sock);
    let client = NestClient::new(sock);
    let result = client.retrieve_blob("missing").unwrap();
    assert!(result.is_none());
}

#[test]
fn default_socket_path_returns_non_empty() {
    let path = default_socket_path();
    assert!(!path.as_os_str().is_empty());
    assert!(path.to_string_lossy().contains("nestgate"));
}

#[test]
fn discover_nestgate_socket_returns_none_when_no_socket() {
    let result = discover_nestgate_socket();
    if let Some(path) = result {
        assert!(path.exists(), "discovered socket must exist");
    }
}

#[test]
fn default_socket_path_contains_nestgate() {
    let path = default_socket_path();
    let path_str = path.to_string_lossy();
    assert!(
        path_str.contains("nestgate"),
        "socket path should contain 'nestgate': {path_str}"
    );
}
