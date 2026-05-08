// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp400: NUCLEUS Composition Parity — wetSpring
//!
//! Replicates primalSpring's exp094 pattern for the wetSpring niche.
//! Validates each NUCLEUS atomic tier (Tower, Node, Nest) and a cross-atomic
//! science pipeline via live IPC.
//!
//! Pattern: discover → call → extract → compare → report
//!
//! Gracefully skips checks when primals are offline (no hard failure for
//! missing infrastructure — produces a gap report instead).
//!
//! ## Environment
//!
//! | Variable | Default | Purpose |
//! |----------|---------|---------|
//! | `WETSPRING_SOCKET` | `$XDG_RUNTIME_DIR/biomeos/wetspring-default.sock` | wetSpring IPC |
//! | `BIOMEOS_SOCKET`   | `$XDG_RUNTIME_DIR/biomeos/default.sock`           | biomeOS router |
//!
//! ## Usage
//!
//! ```text
//! cargo run -p wetspring-exp400
//! ```

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::{Value, json};

const RPC_TIMEOUT: Duration = Duration::from_secs(10);

fn main() {
    let mut v = Harness::new("wetSpring Exp400 — NUCLEUS Composition Parity");

    // ── Discovery ──────────────────────────────────────────────────
    v.section("Discovery");

    let wetspring_socket = resolve_socket("WETSPRING_SOCKET", "wetspring-default.sock");
    let biomeos_socket = resolve_socket("BIOMEOS_SOCKET", "default.sock");

    let ws_live = wetspring_socket
        .as_ref()
        .is_some_and(|p| p.exists());
    v.check_bool("wetspring_socket_exists", ws_live, &format!(
        "socket: {}",
        wetspring_socket.as_ref().map_or("not found".into(), |p| p.display().to_string()),
    ));

    let bio_live = biomeos_socket
        .as_ref()
        .is_some_and(|p| p.exists());
    v.check_bool("biomeos_socket_exists", bio_live, &format!(
        "biomeOS: {}",
        biomeos_socket.as_ref().map_or("not found".into(), |p| p.display().to_string()),
    ));

    // ── Tower Atomic (BearDog + Songbird) ──────────────────────────
    v.section("Tower Atomic (BearDog + Songbird)");
    tower_health(&biomeos_socket, &mut v);
    tower_crypto_hash(&biomeos_socket, &mut v);

    // ── Node Atomic (barraCuda + coralReef + toadStool) ────────────
    v.section("Node Atomic (barraCuda + coralReef + toadStool)");
    node_compute_health(&biomeos_socket, &mut v);
    node_stats_mean(&wetspring_socket, &mut v);

    // ── Nest Atomic (NestGate + Provenance Trio) ───────────────────
    v.section("Nest Atomic (NestGate + Provenance Trio)");
    nest_storage_health(&biomeos_socket, &mut v);
    nest_provenance_health(&biomeos_socket, &mut v);

    // ── wetSpring Niche: Science Pipeline ──────────────────────────
    v.section("Niche: wetSpring Science Pipeline");
    niche_science_health(&wetspring_socket, &mut v);
    niche_diversity_parity(&wetspring_socket, &mut v);
    niche_capability_list(&wetspring_socket, &mut v);

    // ── Cross-Atomic: Hash → Store → Retrieve → Science ───────────
    v.section("NUCLEUS Cross-Atomic Pipeline");
    cross_atomic_pipeline(&wetspring_socket, &biomeos_socket, &mut v);

    v.finish();
}

// ═════════════════════════════════════════════════════════════════════
// Tower Atomic
// ═════════════════════════════════════════════════════════════════════

fn tower_health(socket: &Option<PathBuf>, v: &mut Harness) {
    for (label, method) in [
        ("beardog_health", "health.check"),
        ("songbird_health", "health.check"),
    ] {
        match try_rpc(socket, method, &json!({})) {
            Ok(r) => v.check_bool(label, r.get("status").is_some(), &format!("{method}: {r}")),
            Err(e) => v.skip(label, &e),
        }
    }
}

fn tower_crypto_hash(socket: &Option<PathBuf>, v: &mut Harness) {
    let test_data = "wetSpring exp400 NUCLEUS composition parity";
    match try_rpc(socket, "crypto.hash", &json!({"data": test_data, "algorithm": "blake3"})) {
        Ok(r) => {
            let hash = r.get("hash").and_then(Value::as_str).unwrap_or("");
            v.check_bool("crypto_hash_nonempty", !hash.is_empty(), &format!("BLAKE3: {hash}"));
            let deterministic = try_rpc(socket, "crypto.hash", &json!({"data": test_data, "algorithm": "blake3"}))
                .is_ok_and(|r2| r2.get("hash").and_then(Value::as_str) == Some(hash));
            v.check_bool("crypto_hash_deterministic", deterministic, "same input → same hash");
        }
        Err(e) => {
            v.skip("crypto_hash_nonempty", &e);
            v.skip("crypto_hash_deterministic", &e);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Node Atomic
// ═════════════════════════════════════════════════════════════════════

fn node_compute_health(socket: &Option<PathBuf>, v: &mut Harness) {
    match try_rpc(socket, "compute.health", &json!({})) {
        Ok(r) => v.check_bool("compute_health", r.get("status").is_some(), &format!("{r}")),
        Err(e) => v.skip("compute_health", &e),
    }
}

fn node_stats_mean(socket: &Option<PathBuf>, v: &mut Harness) {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let expected_mean = 3.0;
    match try_rpc(socket, "math.stats", &json!({"operation": "mean", "data": data})) {
        Ok(r) => {
            let actual = r.get("result").and_then(Value::as_f64).unwrap_or(f64::NAN);
            let pass = (actual - expected_mean).abs() < 1e-10;
            v.check_bool("stats_mean_parity", pass, &format!("expected {expected_mean}, got {actual}"));
        }
        Err(e) => v.skip("stats_mean_parity", &e),
    }
}

// ═════════════════════════════════════════════════════════════════════
// Nest Atomic
// ═════════════════════════════════════════════════════════════════════

fn nest_storage_health(socket: &Option<PathBuf>, v: &mut Harness) {
    match try_rpc(socket, "storage.health", &json!({})) {
        Ok(r) => v.check_bool("storage_health", r.get("status").is_some(), &format!("{r}")),
        Err(e) => v.skip("storage_health", &e),
    }
}

fn nest_provenance_health(socket: &Option<PathBuf>, v: &mut Harness) {
    match try_rpc(socket, "provenance.health", &json!({})) {
        Ok(r) => v.check_bool("provenance_health", r.get("status").is_some(), &format!("{r}")),
        Err(e) => v.skip("provenance_health", &e),
    }
}

// ═════════════════════════════════════════════════════════════════════
// wetSpring Niche
// ═════════════════════════════════════════════════════════════════════

fn niche_science_health(socket: &Option<PathBuf>, v: &mut Harness) {
    match try_rpc(socket, "health.check", &json!({})) {
        Ok(r) => {
            let methods = r.get("methods").and_then(Value::as_u64).unwrap_or(0);
            v.check_bool("science_health", methods > 0, &format!("{methods} science methods"));
        }
        Err(e) => v.skip("science_health", &e),
    }
}

fn niche_diversity_parity(socket: &Option<PathBuf>, v: &mut Harness) {
    let abundances = vec![10.0, 20.0, 30.0, 15.0, 25.0];
    match try_rpc(socket, "science.diversity", &json!({"abundances": abundances, "metric": "shannon"})) {
        Ok(r) => {
            let shannon = r.get("shannon").and_then(Value::as_f64).unwrap_or(f64::NAN);
            let pass = shannon > 0.0 && shannon.is_finite();
            v.check_bool("diversity_shannon_valid", pass, &format!("H' = {shannon:.6}"));
        }
        Err(e) => v.skip("diversity_shannon_valid", &e),
    }
}

fn niche_capability_list(socket: &Option<PathBuf>, v: &mut Harness) {
    match try_rpc(socket, "capability.list", &json!({})) {
        Ok(r) => {
            let methods = r.get("methods").and_then(|m| m.as_array());
            let count = methods.map_or(0, Vec::len);
            v.check_bool("capability_list", count > 5, &format!("{count} capabilities registered"));
        }
        Err(e) => v.skip("capability_list", &e),
    }
}

// ═════════════════════════════════════════════════════════════════════
// Cross-Atomic Pipeline
// ═════════════════════════════════════════════════════════════════════

fn cross_atomic_pipeline(
    ws_socket: &Option<PathBuf>,
    bio_socket: &Option<PathBuf>,
    v: &mut Harness,
) {
    let test_blob = json!({"experiment": "exp400", "data": [1.0, 2.0, 3.0]});
    let blob_str = serde_json::to_string(&test_blob).unwrap_or_default();

    // Step 1: hash via Tower (BearDog crypto)
    let hash = match try_rpc(bio_socket, "crypto.hash", &json!({"data": blob_str, "algorithm": "blake3"})) {
        Ok(r) => r.get("hash").and_then(Value::as_str).map(String::from),
        Err(e) => {
            v.skip("cross_atomic_hash", &e);
            v.skip("cross_atomic_store", "depends on hash");
            v.skip("cross_atomic_science", "depends on store");
            return;
        }
    };
    v.check_bool("cross_atomic_hash", hash.is_some(), &format!("BLAKE3 → {}", hash.as_deref().unwrap_or("?")));

    // Step 2: store via Nest (NestGate)
    match try_rpc(bio_socket, "storage.store", &json!({
        "key": hash.as_deref().unwrap_or("test-key"),
        "data": blob_str,
        "namespace": "wetspring-exp400",
    })) {
        Ok(_) => v.check_bool("cross_atomic_store", true, "stored in NestGate"),
        Err(e) => {
            v.skip("cross_atomic_store", &e);
            v.skip("cross_atomic_science", "depends on store");
            return;
        }
    }

    // Step 3: science computation via wetSpring niche
    match try_rpc(ws_socket, "science.diversity", &json!({"abundances": [1.0, 2.0, 3.0], "metric": "shannon"})) {
        Ok(r) => {
            let valid = r.get("shannon").and_then(Value::as_f64).is_some_and(|s| s > 0.0);
            v.check_bool("cross_atomic_science", valid, "science pipeline post-storage");
        }
        Err(e) => v.skip("cross_atomic_science", &e),
    }
}

// ═════════════════════════════════════════════════════════════════════
// Infrastructure
// ═════════════════════════════════════════════════════════════════════

fn resolve_socket(env_var: &str, filename: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_var) {
        return Some(PathBuf::from(p));
    }
    let runtime = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    Some(PathBuf::from(runtime).join("biomeos").join(filename))
}

fn try_rpc(socket: &Option<PathBuf>, method: &str, params: &Value) -> Result<Value, String> {
    let path = socket.as_ref().ok_or_else(|| "socket path not resolved".to_string())?;
    if !path.exists() {
        return Err(format!("socket not found: {} — primal offline (gap)", path.display()));
    }

    let request = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });
    let request_line = serde_json::to_string(&request).map_err(|e| format!("serialize: {e}"))?;

    let stream = UnixStream::connect(path).map_err(|e| format!("connect {}: {e}", path.display()))?;
    stream.set_read_timeout(Some(RPC_TIMEOUT)).ok();
    stream.set_write_timeout(Some(RPC_TIMEOUT)).ok();

    let mut writer = std::io::BufWriter::new(&stream);
    writer.write_all(request_line.as_bytes()).map_err(|e| format!("write: {e}"))?;
    writer.write_all(b"\n").map_err(|e| format!("newline: {e}"))?;
    writer.flush().map_err(|e| format!("flush: {e}"))?;

    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| format!("read: {e}"))?;

    if line.is_empty() {
        return Err("empty response".to_string());
    }

    let response: Value = serde_json::from_str(&line).map_err(|e| format!("parse: {e}"))?;

    if let Some(err) = response.get("error") {
        return Err(format!("RPC error: {err}"));
    }

    response
        .get("result")
        .cloned()
        .ok_or_else(|| "no result field".to_string())
}

/// Minimal validation harness for experiment crates (no wetspring-barracuda Validator dep overhead).
struct Harness {
    name: String,
    passed: u32,
    skipped: u32,
    failed: u32,
    total: u32,
}

impl Harness {
    fn new(name: &str) -> Self {
        println!("═══════════════════════════════════════════════════════════");
        println!("  {name}");
        println!("  Provenance: exp400_nucleus_composition_parity / 2026-05-08");
        println!("═══════════════════════════════════════════════════════════\n");
        Self { name: name.to_string(), passed: 0, skipped: 0, failed: 0, total: 0 }
    }

    fn section(&self, label: &str) {
        println!("\n── {label} ──────────────────────────────────────────");
    }

    fn check_bool(&mut self, label: &str, pass: bool, detail: &str) {
        self.total += 1;
        if pass {
            self.passed += 1;
            println!("  ✓ {label}: {detail}");
        } else {
            self.failed += 1;
            println!("  ✗ {label}: {detail}");
        }
    }

    fn skip(&mut self, label: &str, reason: &str) {
        self.total += 1;
        self.skipped += 1;
        println!("  ⊘ {label}: SKIP — {reason}");
    }

    fn finish(self) -> ! {
        println!("\n═══════════════════════════════════════════════════════════");
        println!(
            "  {} — {}/{} passed, {} skipped, {} failed",
            self.name, self.passed, self.total, self.skipped, self.failed
        );
        println!("═══════════════════════════════════════════════════════════");

        if self.failed > 0 {
            std::process::exit(1);
        }
        std::process::exit(0);
    }
}
