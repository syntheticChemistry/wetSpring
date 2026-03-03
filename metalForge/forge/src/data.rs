// SPDX-License-Identifier: AGPL-3.0-or-later

//! Data resolution chain for validation binaries.
//!
//! Provides a three-tier data path resolution strategy:
//! 1. **Environment**: `WETSPRING_DATA_DIR` (explicit local path)
//! 2. **`NestGate` IPC**: Query `NestGate` for cached/stored datasets
//! 3. **Synthetic fallback**: Use generated test data when real data unavailable
//!
//! This wires the Nest atomic (Tower + `NestGate`) into metalForge so validation
//! binaries can discover real data at runtime without hardcoded paths.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::nest;

const NESTGATE_TIMEOUT: Duration = Duration::from_secs(5);

/// Where data was resolved from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSource {
    /// Resolved from a local directory via environment variable.
    LocalDir(PathBuf),
    /// Resolved from `NestGate` IPC cache.
    NestGate,
    /// No real data available — using synthetic fallback.
    Synthetic,
}

/// Result of a data path resolution attempt.
#[derive(Debug)]
pub struct DataResolution {
    /// Where the data came from.
    pub source: DataSource,
    /// Local file path (if data is on disk).
    pub path: Option<PathBuf>,
    /// Whether real (non-synthetic) data is available.
    pub is_real: bool,
}

/// Resolve a dataset by name using the three-tier chain.
///
/// Resolution order:
/// 1. `WETSPRING_DATA_DIR/<dataset>` (env var path)
/// 2. `NestGate` IPC `storage.exists` / `storage.retrieve`
/// 3. Synthetic fallback
#[must_use]
pub fn resolve_dataset(dataset: &str) -> DataResolution {
    if let Some(dir) = env_data_dir() {
        let path = dir.join(dataset);
        if path.exists() {
            return DataResolution {
                source: DataSource::LocalDir(dir),
                path: Some(path),
                is_real: true,
            };
        }
    }

    if let Some(socket) = discover_nestgate_socket() {
        if nestgate_has_dataset(&socket, dataset) {
            return DataResolution {
                source: DataSource::NestGate,
                path: None,
                is_real: true,
            };
        }
    }

    DataResolution {
        source: DataSource::Synthetic,
        path: None,
        is_real: false,
    }
}

/// Resolve a specific file within a dataset.
#[must_use]
pub fn resolve_file(dataset: &str, filename: &str) -> DataResolution {
    if let Some(dir) = env_data_dir() {
        let path = dir.join(dataset).join(filename);
        if path.exists() {
            return DataResolution {
                source: DataSource::LocalDir(dir),
                path: Some(path),
                is_real: true,
            };
        }
    }

    let cache_key = format!("{dataset}/{filename}");
    if let Some(socket) = discover_nestgate_socket() {
        if nestgate_has_dataset(&socket, &cache_key) {
            return DataResolution {
                source: DataSource::NestGate,
                path: None,
                is_real: true,
            };
        }
    }

    DataResolution {
        source: DataSource::Synthetic,
        path: None,
        is_real: false,
    }
}

/// Discover the data directory path (env var or XDG-compliant default).
///
/// Resolution order:
/// 1. `WETSPRING_DATA_DIR` (explicit override)
/// 2. `XDG_DATA_HOME/wetspring` (typically `~/.local/share/wetspring`)
/// 3. `~/.local/share/wetspring` (XDG default when `XDG_DATA_HOME` is unset)
///
/// Does not check existence — callers should create the directory if needed.
#[must_use]
pub fn discover_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("WETSPRING_DATA_DIR") {
        return PathBuf::from(dir);
    }

    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(xdg).join("wetspring");
    }

    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".local/share/wetspring");
    }

    PathBuf::from("/tmp/wetspring-data")
}

/// Get the data directory from environment, falling back to `data/` relative
/// to the workspace root.
fn env_data_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("WETSPRING_DATA_DIR") {
        let p = PathBuf::from(dir);
        if p.is_dir() {
            return Some(p);
        }
    }

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(|p| p.join("data"));

    workspace.filter(|p| p.is_dir())
}

/// Discover the `NestGate` Unix socket (delegates to [`nest::discover_nestgate_socket`]).
fn discover_nestgate_socket() -> Option<PathBuf> {
    nest::discover_nestgate_socket()
}

/// Check if `NestGate` has a dataset cached.
fn nestgate_has_dataset(socket: &Path, key: &str) -> bool {
    let escaped_key = key.replace('\\', "\\\\").replace('"', "\\\"");
    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"storage.exists","params":{{"key":"data:{escaped_key}","family_id":"default"}},"id":1}}"#,
    );
    nestgate_rpc(socket, &request)
        .map(|resp| resp.contains("true"))
        .unwrap_or(false)
}

/// Send a JSON-RPC request to `NestGate`.
fn nestgate_rpc(socket: &Path, request: &str) -> Result<String, String> {
    let addr = std::os::unix::net::SocketAddr::from_pathname(socket)
        .map_err(|e| format!("invalid NestGate socket: {e}"))?;
    let stream = UnixStream::connect_addr(&addr).map_err(|e| format!("NestGate connect: {e}"))?;
    stream
        .set_read_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| format!("timeout: {e}"))?;
    stream
        .set_write_timeout(Some(NESTGATE_TIMEOUT))
        .map_err(|e| format!("timeout: {e}"))?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer
        .write_all(request.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer.write_all(b"\n").map_err(|e| format!("write: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("NestGate returned empty response".to_string());
    }
    Ok(line)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn resolve_dataset_synthetic_when_no_env() {
        temp_env::with_var("WETSPRING_DATA_DIR", None::<&str>, || {
            let res = resolve_dataset("nonexistent_dataset_12345");
            assert_eq!(res.source, DataSource::Synthetic);
            assert!(!res.is_real);
            assert!(res.path.is_none());
        });
    }

    #[test]
    fn resolve_dataset_from_env() {
        let dir = tempfile::tempdir().unwrap();
        let dataset_dir = dir.path().join("test_dataset");
        std::fs::create_dir_all(&dataset_dir).unwrap();
        std::fs::write(dataset_dir.join("sample.fasta"), ">seq\nATCG").unwrap();

        temp_env::with_var(
            "WETSPRING_DATA_DIR",
            Some(dir.path().to_str().unwrap()),
            || {
                let res = resolve_dataset("test_dataset");
                assert!(matches!(res.source, DataSource::LocalDir(_)));
                assert!(res.is_real);
                assert!(res.path.unwrap().exists());
            },
        );
    }

    #[test]
    fn resolve_file_from_env() {
        let dir = tempfile::tempdir().unwrap();
        let dataset_dir = dir.path().join("vibrio_assemblies");
        std::fs::create_dir_all(&dataset_dir).unwrap();
        std::fs::write(dataset_dir.join("GCF_001.fna.gz"), "fake genome").unwrap();

        temp_env::with_var(
            "WETSPRING_DATA_DIR",
            Some(dir.path().to_str().unwrap()),
            || {
                let res = resolve_file("vibrio_assemblies", "GCF_001.fna.gz");
                assert!(res.is_real);
                assert!(res.path.unwrap().exists());
            },
        );
    }

    #[test]
    fn resolve_file_synthetic_fallback() {
        temp_env::with_var("WETSPRING_DATA_DIR", None::<&str>, || {
            let res = resolve_file("nonexistent", "nope.txt");
            assert_eq!(res.source, DataSource::Synthetic);
            assert!(!res.is_real);
        });
    }

    #[test]
    fn nestgate_socket_discovery_does_not_panic() {
        let _ = discover_nestgate_socket();
    }

    #[test]
    fn resolve_dataset_nonexistent_env_dir() {
        let dir = tempfile::tempdir().unwrap();
        temp_env::with_var(
            "WETSPRING_DATA_DIR",
            Some(dir.path().to_str().unwrap()),
            || {
                let res = resolve_dataset("missing_dataset");
                assert_eq!(res.source, DataSource::Synthetic);
                assert!(!res.is_real);
            },
        );
    }

    #[test]
    fn resolve_file_nonexistent_file_in_valid_dir() {
        let dir = tempfile::tempdir().unwrap();
        let dataset_dir = dir.path().join("ds");
        std::fs::create_dir_all(&dataset_dir).unwrap();

        temp_env::with_var(
            "WETSPRING_DATA_DIR",
            Some(dir.path().to_str().unwrap()),
            || {
                let res = resolve_file("ds", "missing.fa");
                assert_eq!(res.source, DataSource::Synthetic);
                assert!(!res.is_real);
            },
        );
    }

    #[test]
    fn data_source_equality() {
        assert_eq!(DataSource::Synthetic, DataSource::Synthetic);
        assert_eq!(DataSource::NestGate, DataSource::NestGate);
        assert_ne!(DataSource::Synthetic, DataSource::NestGate);
        let path = discover_data_dir();
        assert_eq!(
            DataSource::LocalDir(path.clone()),
            DataSource::LocalDir(path)
        );
    }

    #[test]
    fn data_resolution_debug() {
        let res = DataResolution {
            source: DataSource::Synthetic,
            path: None,
            is_real: false,
        };
        let debug = format!("{res:?}");
        assert!(debug.contains("Synthetic"));
    }

    #[test]
    fn env_data_dir_invalid_path() {
        temp_env::with_var(
            "WETSPRING_DATA_DIR",
            Some("/definitely/nonexistent/path/12345"),
            || {
                let dir = env_data_dir();
                assert!(dir.is_none() || !dir.unwrap().to_str().unwrap().contains("12345"));
            },
        );
    }
}
