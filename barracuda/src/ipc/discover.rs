// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based socket discovery for biomeOS primals.
//!
//! Resolves Unix domain socket paths using a cascading strategy:
//! 1. Explicit env var override (e.g. `WETSPRING_SOCKET`)
//! 2. `$XDG_RUNTIME_DIR/biomeos/{primal}-{family_id}.sock`
//! 3. `<temp_dir>/{primal}-{family_id}.sock` (platform-agnostic fallback)
//!
//! `FAMILY_ID` / `BIOMEOS_FAMILY_ID` env var selects the instance (defaults
//! to `"default"`), enabling multi-instance deployments on the same host.
//!
//! No absolute paths or hardcoded primal names — discovery is parameterized
//! by env var and primal name, following the Primal IPC Protocol.

use std::path::PathBuf;

/// Resolve the active family identifier for multi-instance discovery.
///
/// Priority: `FAMILY_ID` → `BIOMEOS_FAMILY_ID` → `"default"`.
#[must_use]
pub fn family_id() -> String {
    std::env::var("FAMILY_ID")
        .or_else(|_| std::env::var("BIOMEOS_FAMILY_ID"))
        .unwrap_or_else(|_| "default".to_string())
}

/// Derive the standard env var name for a primal socket.
///
/// Convention: uppercase primal name + `_SOCKET` suffix.
///
/// ```
/// # use wetspring_barracuda::ipc::discover::socket_env_var;
/// assert_eq!(socket_env_var("songbird"), "SONGBIRD_SOCKET");
/// assert_eq!(socket_env_var("wetspring"), "WETSPRING_SOCKET");
/// ```
#[must_use]
pub fn socket_env_var(primal: &str) -> String {
    let mut var = primal.to_ascii_uppercase();
    var.push_str("_SOCKET");
    var
}

/// Discover a primal by name using the standard env var convention.
///
/// Equivalent to `discover_socket(socket_env_var(primal), primal)`.
#[must_use]
pub fn discover_primal(primal: &str) -> Option<PathBuf> {
    discover_socket(&socket_env_var(primal), primal)
}

/// Discover Squirrel AI socket.
///
/// Priority: `SQUIRREL_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_squirrel() -> Option<PathBuf> {
    discover_primal(super::primal_names::SQUIRREL)
}

/// Discover coralReef sovereign shader compiler socket.
///
/// coralReef compiles WGSL to native ISA for sovereign GPU dispatch.
/// Priority: `CORALREEF_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_coralreef() -> Option<PathBuf> {
    discover_primal(super::primal_names::CORALREEF)
}

/// Discover toadStool compute orchestrator socket.
///
/// toadStool handles hardware discovery, GPU/NPU routing, and compute dispatch.
/// Priority: `TOADSTOOL_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_toadstool() -> Option<PathBuf> {
    discover_primal(super::primal_names::TOADSTOOL)
}

/// Discover petalTongue visualization socket.
///
/// Priority: `PETALTONGUE_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_petaltongue() -> Option<PathBuf> {
    discover_primal(super::primal_names::PETALTONGUE)
}

/// Discover rhizoCrypt derivation DAG socket.
///
/// rhizoCrypt tracks content lineage and derivation chains for
/// scyBorg provenance trio integration.
/// Priority: `RHIZOCRYPT_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_rhizocrypt() -> Option<PathBuf> {
    discover_primal(super::primal_names::RHIZOCRYPT)
}

/// Discover loamSpine immutable ledger socket.
///
/// loamSpine stores provenance certificates and license proofs for
/// scyBorg provenance trio integration.
/// Priority: `LOAMSPINE_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_loamspine() -> Option<PathBuf> {
    discover_primal(super::primal_names::LOAMSPINE)
}

/// Discover sweetGrass provenance socket.
///
/// sweetGrass handles W3C PROV-O attribution braids.
/// Priority: `SWEETGRASS_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_sweetgrass() -> Option<PathBuf> {
    discover_primal(super::primal_names::SWEETGRASS)
}

/// Discover skunkBat audit socket.
///
/// skunkBat handles security audit logging and JH-5 forwarding.
/// Priority: `SKUNKBAT_SOCKET` env → XDG runtime → temp dir.
#[must_use]
pub fn discover_skunkbat() -> Option<PathBuf> {
    discover_primal(super::primal_names::SKUNKBAT)
}

/// Discover a primal by the **capability** it provides.
///
/// Maps a capability domain to the primal that serves it, then resolves
/// the socket via the standard name-based cascade. This is the
/// capability-oriented abstraction (PG-03 evolution path): when Songbird
/// implements `capability.resolve`, this function becomes the single
/// point of migration — callers never change.
///
/// Returns `None` if the capability domain is unrecognized or the
/// providing primal has no reachable socket.
///
/// # Known mappings
///
/// | Capability domain | Provider primal |
/// |-------------------|-----------------|
/// | `"tensor"`, `"stats"`, `"compute"`, `"spectral"`, `"linalg"` | barraCuda |
/// | `"crypto"`, `"security"` | BearDog |
/// | `"discovery"` | Songbird |
/// | `"storage"` | NestGate |
/// | `"dag"` | rhizoCrypt |
/// | `"spine"`, `"entry"` | loamSpine |
/// | `"braid"`, `"provenance"` | sweetGrass |
/// | `"render"`, `"shader"` | petalTongue |
/// | `"ai"`, `"inference"` | Squirrel |
/// | `"audit"` | skunkBat |
///
/// When Songbird is available, attempts `capability.resolve` RPC first
/// (Wave 199+ live resolution). Falls back to the static
/// [`capability_to_primal`] table when Songbird is absent or returns an error.
#[must_use]
pub fn discover_by_capability(capability_domain: &str) -> Option<PathBuf> {
    if let Some(socket) = resolve_via_songbird(capability_domain) {
        return Some(socket);
    }
    let primal = capability_to_primal(capability_domain)?;
    discover_primal(primal)
}

/// Attempt live capability resolution via Songbird `capability.resolve`.
///
/// Returns `Some(socket_path)` if Songbird is running and resolves the
/// domain to a live primal socket. Returns `None` on any failure
/// (Songbird absent, RPC error, domain not found) — callers fall back
/// to the static table.
fn resolve_via_songbird(capability_domain: &str) -> Option<PathBuf> {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let songbird_socket = discover_primal(super::primal_names::SONGBIRD)?;

    let request = format!(
        r#"{{"jsonrpc":"2.0","method":"capability.resolve","params":{{"domain":"{capability_domain}"}},"id":1}}"#,
    );

    let stream = UnixStream::connect(&songbird_socket).ok()?;
    stream
        .set_read_timeout(Some(super::timeouts::DISCOVERY))
        .ok()?;
    stream
        .set_write_timeout(Some(super::timeouts::DISCOVERY))
        .ok()?;

    let mut writer = std::io::BufWriter::new(&stream);
    writer.write_all(request.as_bytes()).ok()?;
    writer.write_all(b"\n").ok()?;
    writer.flush().ok()?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;

    let v: serde_json::Value = serde_json::from_str(&line).ok()?;
    let result = v.get("result")?;
    let socket_path = result.get("socket")?.as_str()?;
    let path = PathBuf::from(socket_path);
    if path.exists() { Some(path) } else { None }
}

/// Map a capability domain prefix to the canonical primal name that serves it.
///
/// Returns `None` for unrecognized domains.
#[must_use]
pub const fn capability_to_primal(domain: &str) -> Option<&str> {
    match domain.as_bytes() {
        b"tensor" | b"stats" | b"compute" | b"spectral" | b"linalg" | b"math" | b"noise"
        | b"activation" | b"fhe" | b"tolerances" | b"rng" | b"health" => {
            Some(super::primal_names::BARRACUDA)
        }
        b"crypto" | b"security" => Some(super::primal_names::BEARDOG),
        b"discovery" => Some(super::primal_names::SONGBIRD),
        b"storage" => Some(super::primal_names::NESTGATE),
        b"dag" => Some(super::primal_names::RHIZOCRYPT),
        b"spine" | b"entry" => Some(super::primal_names::LOAMSPINE),
        b"braid" | b"provenance" => Some(super::primal_names::SWEETGRASS),
        b"render" | b"shader" => Some(super::primal_names::PETALTONGUE),
        b"ai" | b"inference" => Some(super::primal_names::SQUIRREL),
        b"audit" => Some(super::primal_names::SKUNKBAT),
        _ => None,
    }
}

/// Discover an existing primal socket by env var and primal name.
///
/// Returns `Some(path)` if a socket file is found at one of the
/// standard locations, `None` otherwise (standalone mode).
#[must_use]
pub fn discover_socket(env_var: &str, primal: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var(env_var) {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    let fam = family_id();

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(xdg).join(format!(
            "{}/{}-{fam}.sock",
            super::primal_names::BIOMEOS,
            primal
        ));
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(format!("{primal}-{fam}.sock"));
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

/// Resolve a bind path for a server socket.
///
/// Unlike [`discover_socket`], this returns a path even if the file does
/// not yet exist (the caller will create/bind it).
#[must_use]
pub fn resolve_bind_path(env_var: &str, primal: &str) -> PathBuf {
    if let Ok(path) = std::env::var(env_var) {
        return PathBuf::from(path);
    }

    let fam = family_id();

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join(format!(
            "{}/{}-{fam}.sock",
            super::primal_names::BIOMEOS,
            primal
        ));
    }

    std::env::temp_dir().join(format!("{primal}-{fam}.sock"))
}

/// Pure-logic socket resolution for testing (no env reads).
///
/// Accepts pre-resolved values so tests don't pollute the process environment.
#[cfg(test)]
#[must_use]
pub fn resolve_socket_explicit(
    explicit: Option<&str>,
    xdg_runtime: Option<&str>,
    xdg_subpath: &str,
    fallback_name: &str,
) -> Option<PathBuf> {
    if let Some(path) = explicit {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(xdg) = xdg_runtime {
        let p = PathBuf::from(xdg).join(xdg_subpath);
        if p.exists() {
            return Some(p);
        }
    }

    let fallback = std::env::temp_dir().join(fallback_name);
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module: assertions use unwrap for clarity"
)]
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_when_no_socket() {
        let result = discover_socket("WETSPRING_TEST_NONEXISTENT_VAR_12345", "nonexistent_primal");
        assert!(result.is_none());
    }

    #[test]
    fn discover_explicit_env_override() {
        let sock = crate::ipc::test_socket_path("discover_explicit_env_override");
        crate::ipc::cleanup_test_socket(&sock);
        std::fs::write(&sock, "").unwrap();

        temp_env::with_var(
            "WETSPRING_DISCOVER_TEST_SOCK",
            Some(sock.to_str().unwrap()),
            || {
                let found = discover_socket("WETSPRING_DISCOVER_TEST_SOCK", "irrelevant");
                assert_eq!(found, Some(sock.clone()));
            },
        );
        crate::ipc::cleanup_test_socket(&sock);
    }

    #[test]
    fn resolve_bind_path_uses_env() {
        let sock_path = crate::ipc::test_socket_path("resolve_bind_path_uses_env");
        crate::ipc::cleanup_test_socket(&sock_path);
        temp_env::with_var(
            "WETSPRING_BIND_TEST",
            Some(sock_path.to_str().unwrap()),
            || {
                let p = resolve_bind_path("WETSPRING_BIND_TEST", super::super::primal_names::SELF);
                assert_eq!(p, sock_path);
            },
        );
        crate::ipc::cleanup_test_socket(&sock_path);
    }

    #[test]
    fn resolve_bind_path_falls_through_to_temp() {
        temp_env::with_vars(
            [
                ("WETSPRING_BIND_FALLBACK_TEST", None::<&str>),
                ("XDG_RUNTIME_DIR", None::<&str>),
            ],
            || {
                let p = resolve_bind_path("WETSPRING_BIND_FALLBACK_TEST", "myprimal");
                assert!(p.to_string_lossy().contains("myprimal"));
            },
        );
    }

    #[test]
    fn resolve_socket_explicit_finds_existing() {
        let sock = crate::ipc::test_socket_path("resolve_socket_explicit_finds_existing");
        crate::ipc::cleanup_test_socket(&sock);
        std::fs::write(&sock, "").unwrap();

        let found = resolve_socket_explicit(Some(sock.to_str().unwrap()), None, "unused", "unused");
        assert_eq!(found, Some(sock.clone()));
        crate::ipc::cleanup_test_socket(&sock);
    }

    #[test]
    fn resolve_socket_explicit_none_when_missing() {
        let found =
            resolve_socket_explicit(Some("/nonexistent/path.sock"), None, "unused", "unused");
        assert!(found.is_none());
    }

    #[test]
    fn socket_env_var_convention() {
        assert_eq!(socket_env_var("songbird"), "SONGBIRD_SOCKET");
        assert_eq!(socket_env_var("wetspring"), "WETSPRING_SOCKET");
        assert_eq!(socket_env_var("toadstool"), "TOADSTOOL_SOCKET");
        assert_eq!(socket_env_var("biomeos"), "BIOMEOS_SOCKET");
    }

    #[test]
    fn discover_primal_returns_none_when_no_socket() {
        let result = discover_primal("nonexistent_test_primal_xyzzy");
        assert!(result.is_none());
    }

    #[test]
    fn discover_coralreef_returns_none_when_absent() {
        assert!(discover_coralreef().is_none());
    }

    #[test]
    fn discover_toadstool_returns_none_when_absent() {
        assert!(discover_toadstool().is_none());
    }

    #[test]
    fn discover_petaltongue_returns_none_when_absent() {
        assert!(discover_petaltongue().is_none());
    }

    #[test]
    fn discover_rhizocrypt_returns_none_when_absent() {
        assert!(discover_rhizocrypt().is_none());
    }

    #[test]
    fn discover_loamspine_returns_none_when_absent() {
        assert!(discover_loamspine().is_none());
    }

    #[test]
    fn discover_sweetgrass_returns_none_when_absent() {
        assert!(discover_sweetgrass().is_none());
    }

    #[test]
    fn all_primal_discovery_uses_standard_env_convention() {
        assert_eq!(socket_env_var("coralreef"), "CORALREEF_SOCKET");
        assert_eq!(socket_env_var("rhizocrypt"), "RHIZOCRYPT_SOCKET");
        assert_eq!(socket_env_var("loamspine"), "LOAMSPINE_SOCKET");
        assert_eq!(socket_env_var("sweetgrass"), "SWEETGRASS_SOCKET");
        assert_eq!(socket_env_var("petaltongue"), "PETALTONGUE_SOCKET");
        assert_eq!(socket_env_var("squirrel"), "SQUIRREL_SOCKET");
        assert_eq!(socket_env_var("skunkbat"), "SKUNKBAT_SOCKET");
    }

    #[test]
    fn discover_skunkbat_returns_none_when_absent() {
        assert!(discover_skunkbat().is_none());
    }

    #[test]
    fn capability_to_primal_maps_known_domains() {
        assert_eq!(capability_to_primal("tensor"), Some("barracuda"));
        assert_eq!(capability_to_primal("stats"), Some("barracuda"));
        assert_eq!(capability_to_primal("compute"), Some("barracuda"));
        assert_eq!(capability_to_primal("spectral"), Some("barracuda"));
        assert_eq!(capability_to_primal("linalg"), Some("barracuda"));
        assert_eq!(capability_to_primal("crypto"), Some("beardog"));
        assert_eq!(capability_to_primal("discovery"), Some("songbird"));
        assert_eq!(capability_to_primal("storage"), Some("nestgate"));
        assert_eq!(capability_to_primal("dag"), Some("rhizocrypt"));
        assert_eq!(capability_to_primal("spine"), Some("loamspine"));
        assert_eq!(capability_to_primal("braid"), Some("sweetgrass"));
        assert_eq!(capability_to_primal("render"), Some("petaltongue"));
        assert_eq!(capability_to_primal("ai"), Some("squirrel"));
        assert_eq!(capability_to_primal("audit"), Some("skunkbat"));
    }

    #[test]
    fn capability_to_primal_none_for_unknown() {
        assert_eq!(capability_to_primal("nonexistent"), None);
        assert_eq!(capability_to_primal(""), None);
    }

    #[test]
    fn discover_by_capability_returns_none_for_absent_primals() {
        assert!(discover_by_capability("tensor").is_none());
        assert!(discover_by_capability("audit").is_none());
        assert!(discover_by_capability("nonexistent").is_none());
    }

    #[test]
    fn resolve_via_songbird_returns_none_when_absent() {
        assert!(resolve_via_songbird("tensor").is_none());
        assert!(resolve_via_songbird("unknown_domain").is_none());
    }
}
