# wetSpring V126 Handoff — DispatchOutcome, Health Probes, IpcError Helpers

**Date:** 2026-03-16
**From:** wetSpring V126
**To:** toadStool, biomeOS, healthSpring, sweetGrass, all springs

---

## What Changed

### 1. `DispatchOutcome<T>` Enum (compute_dispatch.rs)

Protocol vs application error separation following groundSpring/airSpring/sweetGrass:

```rust
pub enum DispatchOutcome<T> {
    Success(T),                          // RPC succeeded
    Protocol(DispatchError),             // Transport/socket failure (retriable)
    Application { code: i64, message: String }, // toadStool rejected (deterministic)
}
```

- `is_success()`, `is_retriable()`, `into_result()` methods
- `submit_outcome()` + `submit_outcome_to()` public API
- Internal `rpc_call_outcome()` separates transport from JSON-RPC errors
- Callers: retry on `Protocol`, report on `Application`, use on `Success`

### 2. `health.liveness` + `health.readiness` Probes

Following healthSpring V32 / sweetGrass orchestration pattern:

| Method | Response | Purpose |
|--------|----------|---------|
| `health.liveness` | `{"alive": true, "primal": "wetspring"}` | Fast keep-alive polling |
| `health.readiness` | `{"ready": true, "subsystems": {"math": true, "gpu": false, "ipc": true}}` | Deep subsystem check |
| `health.check` | Delegates to `health.readiness` | Backward compatible |

- New `health` capability domain (3 methods)
- 24 total capabilities across 16 domains
- biomeOS can poll liveness cheaply, readiness for routing decisions

### 3. `IpcError` Query Helpers

Circuit-breaker / backoff methods on the structured `IpcError` enum:

| Method | Returns `true` for |
|--------|--------------------|
| `is_retriable()` | `Connect`, `Transport`, `EmptyResponse` |
| `is_timeout_likely()` | Transport/Connect containing "timeout"/"timed out"/"wouldblock" |
| `is_method_not_found()` | `RpcReject { code: -32601, .. }` |
| `is_connection_error()` | `Connect`, `SocketPath` |

- Enables circuit-breaker logic without string matching
- sweetGrass-compatible: same semantic intent, Rust-idiomatic

---

## Ecosystem Relevance

### For toadStool
- `DispatchOutcome` enables smarter retry logic in `compute.dispatch` callers
- Can adopt similar pattern for its own dispatch responses

### For biomeOS
- `health.liveness` / `health.readiness` now standard — poll all springs uniformly
- Subsystem status enables degraded-mode routing (CPU fallback on `gpu: false`)

### For healthSpring / sweetGrass
- Pattern convergence: wetSpring now speaks the same health/circuit-breaker dialect
- `IpcError::is_retriable()` aligns with sweetGrass's backoff recommendations

### For all springs
- 16 capability domains, 24 methods — full ecosystem introspection
- `DispatchOutcome` pattern recommended for all toadStool dispatch clients

---

## Test Summary

- 62 targeted tests pass (error, compute_dispatch, dispatch, capability_domains)
- 1,440 lib tests pass (3 pre-existing GPU-hardware-specific failures excluded)
- Zero new clippy warnings (pedantic + nursery)
- Full audit: zero unsafe, zero production mocks, zero hardcoding, zero TODO/FIXME

---

## Files Changed

| File | Change |
|------|--------|
| `barracuda/src/error.rs` | `IpcError` query helpers: `is_retriable()`, `is_timeout_likely()`, `is_method_not_found()`, `is_connection_error()` |
| `barracuda/src/ipc/compute_dispatch.rs` | `DispatchOutcome<T>`, `submit_outcome()`, `rpc_call_outcome()` |
| `barracuda/src/ipc/handlers/mod.rs` | `handle_health_liveness()`, `handle_health_readiness()`, CAPABILITIES updated to 24 |
| `barracuda/src/ipc/dispatch.rs` | Routes for `health.liveness`, `health.readiness` |
| `barracuda/src/ipc/capability_domains.rs` | `health` domain (3 methods), 16 total domains |
| `barracuda/src/ipc/mod.rs` | Doc table updated with new health methods |
