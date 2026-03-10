// SPDX-License-Identifier: AGPL-3.0-or-later
//! BIOM format parser (QIIME2 interop).
//!
//! Parses BIOM 1.0 (JSON) OTU/ASV tables into a dense matrix with
//! sample and observation metadata. BIOM is the standard exchange
//! format for QIIME2 feature tables.
//!
//! # Supported formats
//!
//! - **BIOM 1.0 (JSON)**: Dense and sparse matrix formats.
//!   Requires `json` feature (`serde_json`).
//!
//! # References
//!
//! - `McDonald` et al. 2012, *`GigaScience`* 1:7 (BIOM format)
//! - `<http://biom-format.org/documentation/format_versions/biom-1.0.html>`

use crate::error::{Error, Result};
use serde::Deserialize;
use std::io::BufReader;
use std::path::Path;

/// A parsed BIOM table.
#[derive(Debug, Clone)]
pub struct BiomTable {
    /// Table identifier.
    pub id: String,
    /// Format string (e.g. "1.0.0").
    pub format: String,
    /// Matrix type: "dense" or "sparse".
    pub matrix_type: String,
    /// Number of rows (observations / OTUs / ASVs).
    pub n_rows: usize,
    /// Number of columns (samples).
    pub n_cols: usize,
    /// Row (observation) identifiers.
    pub observation_ids: Vec<String>,
    /// Column (sample) identifiers.
    pub sample_ids: Vec<String>,
    /// Dense data matrix (row-major, `n_rows × n_cols`).
    pub data: Vec<f64>,
}

/// Typed deserialization target — eliminates the `serde_json::Value` DOM.
#[derive(Deserialize)]
struct BiomJson {
    #[serde(default)]
    id: Option<String>,
    #[serde(default = "default_format")]
    format: String,
    #[serde(default = "default_matrix_type")]
    matrix_type: String,
    rows: Vec<BiomRow>,
    columns: Vec<BiomRow>,
    data: serde_json::Value,
}

#[derive(Deserialize)]
struct BiomRow {
    #[serde(default)]
    id: String,
}

fn default_format() -> String {
    "1.0.0".into()
}

fn default_matrix_type() -> String {
    "dense".into()
}

impl BiomTable {
    /// Get the count for a specific observation and sample.
    ///
    /// Returns `0.0` if indices are out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row < self.n_rows && col < self.n_cols {
            self.data[row * self.n_cols + col]
        } else {
            0.0
        }
    }

    /// Get a row (observation) as a slice.
    #[must_use]
    pub fn row(&self, row: usize) -> &[f64] {
        let start = row * self.n_cols;
        &self.data[start..start + self.n_cols]
    }

    /// Get a column (sample) as a vector.
    #[must_use]
    pub fn column(&self, col: usize) -> Vec<f64> {
        (0..self.n_rows)
            .map(|r| self.data[r * self.n_cols + col])
            .collect()
    }

    /// Total count per sample (column sums).
    #[must_use]
    pub fn sample_totals(&self) -> Vec<f64> {
        (0..self.n_cols)
            .map(|c| self.column(c).iter().sum())
            .collect()
    }

    /// Total count per observation (row sums).
    #[must_use]
    pub fn observation_totals(&self) -> Vec<f64> {
        (0..self.n_rows).map(|r| self.row(r).iter().sum()).collect()
    }
}

/// Parse a BIOM 1.0 JSON file.
///
/// Uses typed `Deserialize` for direct struct construction — no
/// intermediate `serde_json::Value` DOM tree. The `data` field is
/// still parsed as `Value` to handle dense/sparse polymorphism, but
/// metadata (rows, columns, id, format) deserializes directly.
///
/// # Errors
///
/// Returns [`Error::Io`] if the file cannot be read, or
/// [`Error::InvalidInput`] for malformed BIOM JSON.
pub fn parse_biom(path: &Path) -> Result<BiomTable> {
    let file = std::fs::File::open(path).map_err(|e| Error::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let reader = BufReader::new(file);
    let raw: BiomJson = serde_json::from_reader(reader)
        .map_err(|e| Error::InvalidInput(format!("BIOM JSON: {e}")))?;
    biom_from_typed(raw)
}

/// Parse a BIOM 1.0 JSON string.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for malformed BIOM JSON.
pub fn parse_biom_str(json: &str) -> Result<BiomTable> {
    let raw: BiomJson =
        serde_json::from_str(json).map_err(|e| Error::InvalidInput(format!("BIOM JSON: {e}")))?;
    biom_from_typed(raw)
}

fn biom_from_typed(raw: BiomJson) -> Result<BiomTable> {
    let observation_ids: Vec<String> = raw.rows.into_iter().map(|r| r.id).collect();
    let sample_ids: Vec<String> = raw.columns.into_iter().map(|c| c.id).collect();
    let n_rows = observation_ids.len();
    let n_cols = sample_ids.len();

    let data = if raw.matrix_type == "sparse" {
        parse_sparse_data(&raw.data, n_rows, n_cols)?
    } else {
        parse_dense_data(&raw.data, n_rows, n_cols)?
    };

    Ok(BiomTable {
        id: raw.id.unwrap_or_default(),
        format: raw.format,
        matrix_type: raw.matrix_type,
        n_rows,
        n_cols,
        observation_ids,
        sample_ids,
        data,
    })
}

/// Parse dense matrix data (array of arrays).
fn parse_dense_data(val: &serde_json::Value, n_rows: usize, n_cols: usize) -> Result<Vec<f64>> {
    let outer = val
        .as_array()
        .ok_or_else(|| Error::InvalidInput("BIOM: dense data must be array of arrays".into()))?;

    if outer.len() != n_rows {
        return Err(Error::InvalidInput(format!(
            "BIOM: expected {n_rows} data rows, got {}",
            outer.len()
        )));
    }

    let mut data = Vec::with_capacity(n_rows * n_cols);
    for row in outer {
        let inner = row
            .as_array()
            .ok_or_else(|| Error::InvalidInput("BIOM: dense data rows must be arrays".into()))?;
        if inner.len() != n_cols {
            return Err(Error::InvalidInput(format!(
                "BIOM: expected {n_cols} columns, got {}",
                inner.len()
            )));
        }
        for v in inner {
            data.push(v.as_f64().unwrap_or(0.0));
        }
    }
    Ok(data)
}

/// Parse sparse COO data (array of `[row, col, value]` triplets).
fn parse_sparse_data(val: &serde_json::Value, n_rows: usize, n_cols: usize) -> Result<Vec<f64>> {
    let triplets = val.as_array().ok_or_else(|| {
        Error::InvalidInput("BIOM: sparse data must be array of [row,col,val]".into())
    })?;

    let mut data = vec![0.0_f64; n_rows * n_cols];
    for t in triplets {
        let arr = t.as_array().ok_or_else(|| {
            Error::InvalidInput("BIOM: sparse triplet must be [row,col,val]".into())
        })?;
        if arr.len() < 3 {
            return Err(Error::InvalidInput(
                "BIOM: sparse triplet needs 3 elements".into(),
            ));
        }
        let r = arr[0].as_u64().unwrap_or(0) as usize;
        let c = arr[1].as_u64().unwrap_or(0) as usize;
        let v = arr[2].as_f64().unwrap_or(0.0);
        if r < n_rows && c < n_cols {
            data[r * n_cols + c] = v;
        }
    }
    Ok(data)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::tolerances;

    fn dense_biom_json() -> &'static str {
        r#"{
            "id": "test_table",
            "format": "1.0.0",
            "matrix_type": "dense",
            "matrix_element_type": "float",
            "rows": [
                {"id": "OTU1", "metadata": null},
                {"id": "OTU2", "metadata": null},
                {"id": "OTU3", "metadata": null}
            ],
            "columns": [
                {"id": "Sample1", "metadata": null},
                {"id": "Sample2", "metadata": null}
            ],
            "data": [
                [10.0, 20.0],
                [30.0, 40.0],
                [50.0, 60.0]
            ]
        }"#
    }

    fn sparse_biom_json() -> &'static str {
        r#"{
            "id": "sparse_table",
            "format": "1.0.0",
            "matrix_type": "sparse",
            "matrix_element_type": "int",
            "rows": [
                {"id": "ASV1", "metadata": null},
                {"id": "ASV2", "metadata": null}
            ],
            "columns": [
                {"id": "S1", "metadata": null},
                {"id": "S2", "metadata": null},
                {"id": "S3", "metadata": null}
            ],
            "data": [
                [0, 0, 5],
                [0, 2, 10],
                [1, 1, 15]
            ]
        }"#
    }

    #[test]
    fn parse_dense_biom() {
        let table = parse_biom_str(dense_biom_json()).unwrap();
        assert_eq!(table.id, "test_table");
        assert_eq!(table.n_rows, 3);
        assert_eq!(table.n_cols, 2);
        assert_eq!(table.observation_ids, vec!["OTU1", "OTU2", "OTU3"]);
        assert_eq!(table.sample_ids, vec!["Sample1", "Sample2"]);
        assert!((table.get(0, 0) - 10.0).abs() < f64::EPSILON);
        assert!((table.get(2, 1) - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_sparse_biom() {
        let table = parse_biom_str(sparse_biom_json()).unwrap();
        assert_eq!(table.id, "sparse_table");
        assert_eq!(table.n_rows, 2);
        assert_eq!(table.n_cols, 3);
        assert!((table.get(0, 0) - 5.0).abs() < f64::EPSILON);
        assert!((table.get(0, 1) - 0.0).abs() < f64::EPSILON); // not set
        assert!((table.get(0, 2) - 10.0).abs() < f64::EPSILON);
        assert!((table.get(1, 1) - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn row_and_column_access() {
        let table = parse_biom_str(dense_biom_json()).unwrap();
        let row0 = table.row(0);
        assert!((row0[0] - 10.0).abs() < f64::EPSILON);
        assert!((row0[1] - 20.0).abs() < f64::EPSILON);

        let col1 = table.column(1);
        assert_eq!(col1.len(), 3);
        assert!((col1[0] - 20.0).abs() < f64::EPSILON);
        assert!((col1[2] - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sample_totals() {
        let table = parse_biom_str(dense_biom_json()).unwrap();
        let totals = table.sample_totals();
        assert!((totals[0] - 90.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((totals[1] - 120.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn observation_totals() {
        let table = parse_biom_str(dense_biom_json()).unwrap();
        let totals = table.observation_totals();
        assert!((totals[0] - 30.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((totals[1] - 70.0).abs() < tolerances::ANALYTICAL_LOOSE);
        assert!((totals[2] - 110.0).abs() < tolerances::ANALYTICAL_LOOSE);
    }

    #[test]
    fn out_of_bounds_returns_zero() {
        let table = parse_biom_str(dense_biom_json()).unwrap();
        assert!(table.get(99, 99).abs() < f64::EPSILON);
    }

    #[test]
    fn invalid_json_returns_error() {
        assert!(parse_biom_str("not json").is_err());
    }

    #[test]
    fn missing_rows_returns_error() {
        let json = r#"{"columns": [{"id":"S1"}], "data": []}"#;
        assert!(parse_biom_str(json).is_err());
    }

    #[test]
    fn parse_from_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.biom");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(dense_biom_json().as_bytes()).unwrap();

        let table = parse_biom(&path).unwrap();
        assert_eq!(table.n_rows, 3);
        assert_eq!(table.n_cols, 2);
    }
}
