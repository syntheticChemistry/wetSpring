// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based cache file path resolution and accession-based storage.

use crate::error::Error;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Build a cache file path via capability-based data discovery.
///
/// Discovery order:
/// 1. `WETSPRING_DATA_DIR/{filename}` if `WETSPRING_DATA_DIR` is set
/// 2. `{CARGO_MANIFEST_DIR}/../data/{filename}` for development
/// 3. `data/{filename}` relative to cwd for deployment
#[must_use]
pub fn cache_file(filename: &str) -> PathBuf {
    let data_dir = std::env::var("WETSPRING_DATA_DIR").ok();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    resolve(data_dir.as_deref(), &manifest, filename)
}

/// Build an accession-based cache directory.
///
/// Returns a path like `{data_root}/ncbi/{db}/{accession}/` and creates
/// the directory tree if it does not exist.
///
/// # Errors
///
/// Returns `Err` if the directory cannot be created.
pub fn accession_dir(db: &str, accession: &str) -> crate::error::Result<PathBuf> {
    let base = cache_file(&format!("ncbi/{db}/{accession}"));
    std::fs::create_dir_all(&base).map_err(|e| {
        Error::Ncbi(format!(
            "cannot create accession dir {}: {e}",
            base.display()
        ))
    })?;
    Ok(base)
}

/// Write content to an accession-based cache file with SHA-256 integrity.
///
/// Writes `content` to `{accession_dir}/{filename}` and a companion
/// `.sha256` sidecar file. Returns the path to the written file.
///
/// # Errors
///
/// Returns `Err` if the write fails.
pub fn write_with_integrity(
    dir: &Path,
    filename: &str,
    content: &str,
) -> crate::error::Result<PathBuf> {
    let file_path = dir.join(filename);
    std::fs::write(&file_path, content)
        .map_err(|e| Error::Ncbi(format!("write {}: {e}", file_path.display())))?;

    let hash = sha256_hex(content.as_bytes());
    let sidecar = dir.join(format!("{filename}.sha256"));
    std::fs::write(&sidecar, format!("{hash}  {filename}\n"))
        .map_err(|e| Error::Ncbi(format!("write sidecar {}: {e}", sidecar.display())))?;

    Ok(file_path)
}

/// Verify the SHA-256 integrity of a cached file.
///
/// Reads the `.sha256` sidecar and compares against the actual file hash.
///
/// # Errors
///
/// Returns `Err` if the sidecar is missing, the file is missing, or the
/// hashes do not match.
pub fn verify_integrity(dir: &Path, filename: &str) -> crate::error::Result<()> {
    let file_path = dir.join(filename);
    let sidecar = dir.join(format!("{filename}.sha256"));

    let expected = std::fs::read_to_string(&sidecar)
        .map_err(|e| Error::Ncbi(format!("read sidecar {}: {e}", sidecar.display())))?;
    let expected_hash = expected
        .split_whitespace()
        .next()
        .ok_or_else(|| Error::Ncbi("empty sidecar file".to_string()))?;

    let mut file = std::fs::File::open(&file_path)
        .map_err(|e| Error::Ncbi(format!("open {}: {e}", file_path.display())))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| Error::Ncbi(format!("read {}: {e}", file_path.display())))?;

    let actual_hash = sha256_hex(&buf);
    if actual_hash == expected_hash {
        Ok(())
    } else {
        Err(Error::Ncbi(format!(
            "integrity check failed for {filename}: expected {expected_hash}, got {actual_hash}"
        )))
    }
}

/// Compute a SHA-256 hex digest — pure Rust, no external deps.
///
/// Uses the standard NIST SHA-256 algorithm. This is a validation tool,
/// not a performance-critical path, so clarity > speed.
fn sha256_hex(data: &[u8]) -> String {
    use std::fmt::Write;
    let hash = sha256(data);
    let mut hex = String::with_capacity(64);
    for byte in &hash {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// SHA-256 implementation (FIPS 180-4). Pure Rust, zero dependencies.
#[expect(clippy::many_single_char_names, clippy::too_many_lines)]
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

/// Pure-logic cache path resolution — no env access.
///
/// Fallback cascade (when `data_dir` is `None` or path doesn't exist):
/// 1. `WETSPRING_DATA_DIR` (passed as `data_dir`) — explicit override
/// 2. `{manifest_dir}/../data` — development layout (crate in workspace)
/// 3. `data/` — deployment fallback relative to cwd
fn resolve(data_dir: Option<&str>, manifest_dir: &str, filename: &str) -> PathBuf {
    if let Some(dir) = data_dir {
        let p = PathBuf::from(dir).join(filename);
        if p.parent().is_some_and(Path::exists) {
            return p;
        }
    }

    let dev_path = PathBuf::from(manifest_dir).join("../data").join(filename);
    if dev_path.parent().is_some_and(Path::exists) {
        return dev_path;
    }

    PathBuf::from("data").join(filename)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn builds_relative_path() {
        let path = cache_file("test_cache.txt");
        assert!(
            path.to_string_lossy().contains("data")
                && path.to_string_lossy().contains("test_cache.txt")
        );
    }

    #[test]
    fn nested() {
        let path = cache_file("sub/nested.json");
        assert!(path.to_string_lossy().contains("nested.json"));
    }

    #[test]
    fn dev_fallback_contains_data() {
        let path = cache_file("test.txt");
        let s = path.to_string_lossy();
        assert!(s.contains("data") && s.contains("test.txt"));
    }

    #[test]
    fn uses_manifest_dir_fallback() {
        let path = cache_file("integration_test_file.json");
        let s = path.to_string_lossy();
        assert!(
            s.contains("data") && s.contains("integration_test_file.json"),
            "path should contain data dir: {s}"
        );
    }

    #[test]
    fn handles_path_separators() {
        let path = cache_file("subdir/with/slashes.json");
        let s = path.to_string_lossy();
        assert!(s.contains("slashes.json"));
        assert!(s.contains("subdir") || s.contains("with"));
    }

    #[test]
    fn resolve_data_root_existing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = resolve(Some(dir.path().to_str().unwrap()), ".", "test.json");
        assert_eq!(path, dir.path().join("test.json"));
    }

    #[test]
    fn resolve_data_root_nonexistent_falls_through() {
        let path = resolve(
            Some("/nonexistent_wetspring_test_dir_xyz"),
            ".",
            "test.json",
        );
        assert!(
            !path.starts_with("/nonexistent_wetspring_test_dir_xyz"),
            "should not use nonexistent data root: {path:?}"
        );
    }

    #[test]
    fn resolve_manifest_dir_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let manifest = dir.path().join("sub");
        std::fs::create_dir_all(&manifest).unwrap();
        let path = resolve(None, manifest.to_str().unwrap(), "cache.bin");
        assert!(path.to_string_lossy().contains("data"));
    }

    #[test]
    fn resolve_final_fallback() {
        let path = resolve(None, "/nonexistent_manifest_xyz", "fallback.json");
        assert_eq!(path, PathBuf::from("data").join("fallback.json"));
    }

    #[test]
    fn resolve_none_root_with_valid_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        std::fs::create_dir_all(&data).unwrap();
        let manifest = dir.path().join("crate");
        std::fs::create_dir_all(&manifest).unwrap();
        let path = resolve(None, manifest.to_str().unwrap(), "x.json");
        assert!(path.to_string_lossy().contains("data"), "path = {path:?}");
    }

    #[test]
    fn sha256_empty_input() {
        let hash = sha256_hex(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        let hash = sha256_hex(b"abc");
        assert_eq!(
            hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_longer_input() {
        let hash = sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hash,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn accession_dir_creates_tree() {
        let result = accession_dir("nucleotide", "K03455");
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().contains("ncbi"));
        assert!(path.to_string_lossy().contains("nucleotide"));
        assert!(path.to_string_lossy().contains("K03455"));
    }

    #[test]
    fn write_with_integrity_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let content = ">seq1\nATCGATCG\n";
        let path = write_with_integrity(dir.path(), "test.fasta", content).unwrap();
        assert!(path.exists());

        let sidecar = dir.path().join("test.fasta.sha256");
        assert!(sidecar.exists());

        assert!(verify_integrity(dir.path(), "test.fasta").is_ok());
    }

    #[test]
    fn verify_integrity_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        write_with_integrity(dir.path(), "test.fasta", ">seq1\nATCG\n").unwrap();

        std::fs::write(dir.path().join("test.fasta"), ">CORRUPTED\n").unwrap();

        let err = verify_integrity(dir.path(), "test.fasta").unwrap_err();
        assert!(err.to_string().contains("integrity check failed"));
    }

    #[test]
    fn verify_integrity_missing_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.fasta"), ">seq1\n").unwrap();

        let err = verify_integrity(dir.path(), "test.fasta").unwrap_err();
        assert!(err.to_string().contains("sidecar"));
    }

    #[test]
    fn verify_integrity_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.fasta.sha256"), "abc123  test.fasta\n").unwrap();

        let err = verify_integrity(dir.path(), "test.fasta").unwrap_err();
        assert!(err.to_string().contains("open"));
    }
}
