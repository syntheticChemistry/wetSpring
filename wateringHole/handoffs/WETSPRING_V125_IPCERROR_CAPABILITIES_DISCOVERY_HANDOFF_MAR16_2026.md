# wetSpring V125 — Structured IpcError + Dual-Format Capabilities + Discovery Helpers

**Date:** 2026-03-16
**From:** wetSpring V125
**To:** toadStool, coralReef, barraCuda, sibling springs
**Status:** Active handoff

---

## Summary

Deep debt cleanup implementing typed IPC errors, dual-format capability parsing,
and generic socket discovery helpers — three patterns identified during the
cross-ecosystem review in V124. Also fixes 18 binaries with misplaced `OrExit`
imports from the V123 zero-panic transformation.

## What Changed

### 1. Structured `IpcError` Enum

Replaced `Error::Ipc(String)` with a typed enum enabling programmatic error recovery:

```rust
pub enum IpcError {
    SocketPath(String),           // bind/create/remove failures
    Connect(String),              // connection refused / not found
    Transport(String),            // write/read/flush/timeout/shutdown
    Codec(String),                // serialize/deserialize failures
    RpcReject { code, message },  // JSON-RPC error from remote primal
    EmptyResponse,                // no response or missing result field
}
```

- 28 construction sites updated across 5 files
- `From<IpcError> for Error` for ergonomic `?` propagation
- `Error::Ipc` now properly chains `source()` through `IpcError`
- Callers can match on failure category: retry `Connect`, degrade `RpcReject`, abort `SocketPath`

**For sibling springs:** This pattern allows typed error recovery at call sites.
healthSpring and biomeOS already have similar patterns. Recommend adoption
across all springs for consistent IPC error handling.

### 2. Dual-Format Capability Parsing

New `protocol::extract_capabilities()` parses both flat and structured capability
responses from any primal:

```rust
pub struct CapabilityInfo {
    pub capabilities: Vec<String>,      // Format A: flat list
    pub domains: Vec<CapabilityDomain>, // Format B: structured domains
    pub primal: Option<String>,
    pub version: Option<String>,
}
```

**For toadStool/biomeOS:** Use this to parse `capability.list` responses from
any primal without knowing which format version it speaks.

### 3. Generic Socket Discovery Helpers

New `discover::socket_env_var()` and `discover::discover_primal()` replace
per-primal boilerplate:

```rust
// Before: one function per primal
pub fn discover_squirrel() -> Option<PathBuf> {
    discover_socket("SQUIRREL_SOCKET", SQUIRREL)
}

// After: generic by name
pub fn discover_primal(primal: &str) -> Option<PathBuf> {
    discover_socket(&socket_env_var(primal), primal)
}
```

**For all primals:** Adopt `socket_env_var()` convention so any primal can be
discovered by name without adding new functions.

### 4. Binary OrExit Import Fixes

Fixed 18 binaries where `use OrExit` was placed inside function bodies instead
of at module scope. All 354 binaries now compile with zero errors.

## What This Means for toadStool/barraCuda

- **No API changes** to existing IPC protocol
- **Typed errors** allow toadStool to distinguish "primal down" from "bad request"
  when forwarding dispatch results
- **`extract_capabilities()`** can be promoted to barraCuda for shared use
- **`socket_env_var()`** formalizes the `{PRIMAL_UPPER}_SOCKET` convention

## Test Results

- **Library tests:** 1,475 passed (117 IPC tests, 9 error tests)
- **metalForge tests:** 252 passed
- **Clippy:** zero warnings (pedantic + nursery) across both crates
- **Compilation:** zero errors across all 354 binaries

## Next Steps (V126 candidates)

- `temp-env` for environment variable isolation in all env-reading tests
- aarch64 CI cross-compilation validation (groundSpring pattern)
- `petalTongue` dashboard bindings for real-time validation monitoring
- `rhizoCrypt` NDJSON streaming for provenance event pipelines
