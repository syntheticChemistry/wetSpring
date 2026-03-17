# wetSpring V124 — compute.dispatch + Structured Tracing + deny.toml

**Date:** 2026-03-16
**From:** wetSpring V124
**To:** toadStool, coralReef, barraCuda, sibling springs
**Status:** Active handoff

---

## Summary

Cross-ecosystem absorption executing priorities identified from sibling spring
and primal review. Three major patterns absorbed:

1. **Workspace `deny.toml`** — groundSpring/airSpring pattern
2. **Typed `compute.dispatch` IPC client** — healthSpring/ludoSpring pattern
3. **Structured `tracing`** — coralReef Phase 10 pattern

## What Changed

### 1. Workspace `deny.toml`

Root `deny.toml` covers both workspace members (`barracuda`, `metalForge/forge`):

- `wildcards = "deny"` — no `*` version specs
- `yanked = "deny"` — fail on yanked crates
- `confidence-threshold = 0.8` for license detection
- Advisory DB explicitly configured
- `ring` license clarified (transitive via wgpu)

### 2. Typed `compute.dispatch` IPC Client

New `barracuda/src/ipc/compute_dispatch.rs` — typed client for toadStool S156+
dispatch protocol. Three RPC methods:

| Method | Returns | Purpose |
|--------|---------|---------|
| `compute.dispatch.submit` | `DispatchHandle` | Submit GPU/NPU/CPU workload |
| `compute.dispatch.result` | `DispatchResult` | Poll job by ID |
| `compute.dispatch.capabilities` | `Vec<ComputeBackend>` | Query available backends |

Discovery: `TOADSTOOL_SOCKET` env → XDG runtime → temp dir. Graceful
`DispatchError::NoComputePrimal` fallback in standalone mode.

**For toadStool team:** wetSpring is now wired to discover and use
`compute.dispatch.*` endpoints. Next step: wire actual GPU/NPU workload
submission through this client (currently discovery + typed parsing only).

### 3. Structured `tracing`

Replaced 14 `eprintln!` calls in production library code with structured
`tracing` macros:

- `tracing::info!` for operational events (listening, registered, discovered)
- `tracing::warn!` for recoverable failures (accept error, heartbeat failed)
- `tracing::error!` for fatal failures (cannot bind socket)

Server binary initializes `tracing_subscriber::fmt::init()` for structured
stderr output with timestamps, levels, and span context.

**For coralReef team:** wetSpring now emits structured traces compatible with
the ecosystem tracing standard. Ready for `tracing-opentelemetry` integration
when the primal observability pipeline is available.

## Quality Gate

| Metric | Value |
|--------|-------|
| Tests | 1,719 passed, 0 failed |
| Clippy | ZERO warnings (pedantic + nursery) |
| `cargo check` | ZERO errors |
| `cargo fmt` | Clean |
| `deny.toml` | `wildcards=deny`, `yanked=deny` |

## Dependencies Added

| Crate | Version | Purpose | Pure Rust |
|-------|---------|---------|-----------|
| `tracing` | 0.1 | Structured diagnostics | Yes |
| `tracing-subscriber` | 0.3 (optional, `ipc` feature) | stderr output | Yes |

## Cross-Spring Patterns Absorbed

| Pattern | Source | Implementation |
|---------|--------|----------------|
| `deny.toml` workspace-level | groundSpring V109, airSpring | Root `deny.toml` |
| Typed compute.dispatch | healthSpring V29, ludoSpring V22 | `ipc::compute_dispatch` |
| Structured tracing | coralReef Phase 10 | `tracing` in server/songbird/fetch |
| `yanked = "deny"` | groundSpring V109 | Root + crate `deny.toml` |

## Next Steps (V125)

1. Wire `compute.dispatch.submit` for real GPU workloads via toadStool
2. Add aarch64 cross-compile CI target (4 springs already have it)
3. Wire Track 3 GPU primitives when available in barraCuda
4. `tracing-opentelemetry` integration when primal observability lands
