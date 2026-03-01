// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal JSON helpers for JSON-RPC request/response handling.

/// Escape a string for JSON.
pub(super) fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Extract the `result` value from a JSON-RPC response.
pub(super) fn extract_result_value(json: &str) -> Option<String> {
    let start = json.find("\"result\"")?;
    let after = &json[start + 8..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    if rest.starts_with('{') || rest.starts_with('[') || rest.starts_with('"') {
        let end = find_value_end(rest)?;
        Some(rest[..end].to_string())
    } else {
        let end = rest.find([',', '}'])?;
        Some(rest[..end].trim().to_string())
    }
}

/// Extract a string value for a given key from JSON.
pub(super) fn extract_result_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let start = json.find(&pattern)?;
    let after = &json[start + pattern.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let inner = rest.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

/// Extract a string array for a given key from JSON.
pub(super) fn extract_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let pattern = format!("\"{key}\"");
    let start = json.find(&pattern)?;
    let after = &json[start + pattern.len()..];
    let arr_start = after.find('[')?;
    let arr_end = after[arr_start..].find(']')?;
    let content = &after[arr_start + 1..arr_start + arr_end];
    Some(parse_string_list(content))
}

/// Extract the top-level `result` array from JSON.
pub(super) fn extract_result_array(json: &str) -> Option<Vec<String>> {
    let start = json.find("\"result\"")?;
    let after = &json[start + 8..];
    let arr_start = after.find('[')?;
    let arr_end = after[arr_start..].find(']')?;
    let content = &after[arr_start + 1..arr_start + arr_end];
    Some(parse_string_list(content))
}

/// Parse a comma-separated list of quoted strings.
pub(super) fn parse_string_list(content: &str) -> Vec<String> {
    content
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim().trim_matches('"');
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect()
}

/// Find the end index of a JSON value (string, object, array, or primitive).
pub(super) fn find_value_end(s: &str) -> Option<usize> {
    let first = s.as_bytes().first()?;
    match first {
        b'"' => {
            let end = s[1..].find('"')? + 2;
            Some(end)
        }
        b'{' | b'[' => {
            let (open, close) = if *first == b'{' {
                (b'{', b'}')
            } else {
                (b'[', b']')
            };
            let mut depth = 0;
            for (i, &ch) in s.as_bytes().iter().enumerate() {
                if ch == open {
                    depth += 1;
                } else if ch == close {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
            }
            None
        }
        _ => s.find([',', '}']).or(Some(s.len())),
    }
}
