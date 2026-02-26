// SPDX-License-Identifier: AGPL-3.0-or-later
//! NCBI data loading for Phase 35 experiments.
//!
//! Reads JSON data produced by `scripts/fetch_ncbi_phase35.py`. Falls back to
//! synthetic data generation when NCBI data files are absent (offline / CI mode).
//!
//! Data lives in `data/ncbi_phase35/` relative to the crate manifest directory,
//! resolved via capability-based discovery (`WETSPRING_NCBI_DIR` env var or
//! `validation::data_dir` fallback chain).
//!
//! # Module structure
//!
//! Each experiment's record type lives in its own submodule:
//! - `vibrio` — Vibrio genome assemblies (Exp121)
//! - `campy` — Campylobacterota assemblies (Exp125)
//! - `biome` — 16S `BioProject` records (Exp126)

mod biome;
mod campy;
mod vibrio;

pub use biome::{
    BiomeProject, biome_diversity_params, load_biome_projects, try_load_biome_projects,
};
pub use campy::{CampyAssembly, load_campylobacterota, try_load_campylobacterota};
pub use vibrio::{VibrioAssembly, load_vibrio_assemblies, try_load_vibrio_assemblies};

use crate::validation;
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    validation::data_dir("WETSPRING_NCBI_DIR", "data/ncbi_phase35")
}

/// Load JSON array — returns an error when data is unavailable.
///
/// Production-grade API: callers handle the error explicitly.
fn try_load_json_array<T>(
    path: &std::path::Path,
    array_key: &str,
    from_json: impl Fn(&str) -> T,
    is_valid: impl Fn(&T) -> bool,
) -> crate::error::Result<Vec<T>> {
    let content = std::fs::read_to_string(path).map_err(|e| crate::error::Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let needle = format!("\"{array_key}\"");
    let arr_start = content
        .find(&needle)
        .ok_or_else(|| crate::error::Error::InvalidInput(format!("no \"{array_key}\" key")))?;
    let rest = &content[arr_start..];
    let bracket = rest
        .find('[')
        .ok_or_else(|| crate::error::Error::InvalidInput("no array bracket".into()))?;
    let arr = &rest[bracket..];
    let objects = split_json_objects(arr);
    let items: Vec<T> = objects
        .iter()
        .map(|obj| from_json(obj))
        .filter(is_valid)
        .collect();
    if items.is_empty() {
        return Err(crate::error::Error::InvalidInput(
            "no valid records in JSON array".into(),
        ));
    }
    Ok(items)
}

/// Extract a JSON string value from a line like `"key": "value"`.
fn json_str_value(json: &str, key: &str) -> String {
    let needle = format!("\"{key}\":");
    if let Some(pos) = json.find(&needle) {
        let rest = &json[pos + needle.len()..];
        let rest = rest.trim_start();
        if let Some(inner) = rest.strip_prefix('"') {
            if let Some(end) = inner.find('"') {
                return inner[..end].to_string();
            }
        }
    }
    String::new()
}

/// Extract a JSON integer value from a line like `"key": 12345`.
fn json_int_value(json: &str, key: &str) -> u64 {
    let needle = format!("\"{key}\":");
    if let Some(pos) = json.find(&needle) {
        let rest = &json[pos + needle.len()..];
        let rest = rest.trim_start();
        let num_str: String = rest.chars().take_while(char::is_ascii_digit).collect();
        return num_str.parse().unwrap_or(0);
    }
    0
}

/// Split JSON array into individual object strings (minimal parser).
fn split_json_objects(array_content: &str) -> Vec<String> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = None;

    for (i, ch) in array_content.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        objects.push(array_content[s..=i].to_string());
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    objects
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn json_str_value_basic() {
        let json = r#"{"accession": "GCF_000006745.1", "organism": "Vibrio cholerae"}"#;
        assert_eq!(json_str_value(json, "accession"), "GCF_000006745.1");
        assert_eq!(json_str_value(json, "organism"), "Vibrio cholerae");
    }

    #[test]
    fn json_str_value_missing_key() {
        let json = r#"{"accession": "GCF_000006745.1"}"#;
        assert_eq!(json_str_value(json, "missing"), "");
    }

    #[test]
    fn json_int_value_basic() {
        let json = r#"{"genome_size_bp": 4000000, "gene_count": 3800}"#;
        assert_eq!(json_int_value(json, "genome_size_bp"), 4_000_000);
        assert_eq!(json_int_value(json, "gene_count"), 3800);
    }

    #[test]
    fn json_int_value_missing() {
        assert_eq!(json_int_value(r#"{"x": 1}"#, "missing"), 0);
    }

    #[test]
    fn json_str_value_empty_value() {
        let json = r#"{"accession": "", "organism": "Vibrio"}"#;
        assert_eq!(json_str_value(json, "accession"), "");
        assert_eq!(json_str_value(json, "organism"), "Vibrio");
    }

    #[test]
    fn json_int_value_trailing_comma() {
        assert_eq!(
            json_int_value(r#"{"count": 123, "other": 456}"#, "count"),
            123
        );
    }

    #[test]
    fn split_json_objects_empty() {
        assert!(split_json_objects("").is_empty());
        assert!(split_json_objects("[]").is_empty());
    }

    #[test]
    fn split_json_objects_single() {
        let objects = split_json_objects(r#"[{"a": 1}]"#);
        assert_eq!(objects.len(), 1);
        assert!(objects[0].contains("\"a\""));
    }

    #[test]
    fn split_json_objects_multiple() {
        let objects = split_json_objects(r#"[{"a": 1}, {"b": 2}, {"c": 3}]"#);
        assert_eq!(objects.len(), 3);
    }

    #[test]
    fn split_json_objects_nested() {
        let objects = split_json_objects(r#"[{"a": {"inner": 1}}, {"b": 2}]"#);
        assert_eq!(objects.len(), 2);
        assert!(objects[0].contains("inner"));
    }

    #[test]
    fn data_dir_returns_path() {
        let dir = data_dir();
        assert!(dir.to_string_lossy().contains("ncbi_phase35"));
    }
}
