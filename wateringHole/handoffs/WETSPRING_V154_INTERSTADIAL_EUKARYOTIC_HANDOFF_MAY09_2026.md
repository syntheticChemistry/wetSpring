# wetSpring V154 — Interstadial Eukaryotic Evolution Handoff

**Date**: 2026-05-09
**From**: wetSpring (Tier 2 → Tier 3 target)
**Context**: primalSpring v0.9.25 Interstadial Primordial Extinction wave

---

## What Changed

### 1. Per-Trio Provenance Module Split

The monolithic `ipc/provenance.rs` (553 lines) and `ipc/sweetgrass.rs` (181 lines)
were split into per-trio modules following the ecosystem standard:

| Module | Primal | Domain |
|--------|--------|--------|
| `ipc/provenance/rhizocrypt.rs` | rhizoCrypt | Ephemeral DAG sessions (`dag.*`) |
| `ipc/provenance/loamspine.rs` | loamSpine | Immutable ledger commit (`session.commit`) |
| `ipc/provenance/sweetgrass.rs` | sweetGrass | W3C PROV-O attribution braids (`braid.*`) |
| `ipc/provenance/mod.rs` | — | Shared types, Neural API socket, capability_call, IPC handlers |

All public API preserved. Existing callers (`use crate::ipc::provenance as trio`) unchanged.

### 2. primal-proof Cargo Feature Flag

New `primal-proof` feature gates IPC-first compute routing:

```toml
primal-proof = ["ipc"]
```

Dual-lane pattern:
- `cargo build --features ipc,primal-proof` → IPC-only (sovereign)
- `cargo build --features ipc` → in-process library (default)

### 3. Certification Module (Eukaryotic Organelle)

`certification/` library module absorbs guidestone layers:

| Layer | Name | Requires |
|-------|------|----------|
| 0 | Bare Science Baselines | Nothing |
| 1 | Tolerance Provenance | Nothing |
| 2 | Checksum Verification | Nothing |
| 3 | NUCLEUS Liveness | Deployed primals |
| 4 | Manifest IPC Parity | Live barraCuda |
| 5 | Domain Science IPC | Live barraCuda |
| 6 | Cross-Atomic Pipeline | Live NUCLEUS |

Entry point: `certification::certify(max_layer) -> ValidationResult`

### 4. Validation Scenarios (Two-Tier)

`validation/scenarios/` with `ScenarioMeta` registry:

| Scenario | Track | Tier | Description |
|----------|-------|------|-------------|
| `bare-science` | Science | Rust | Deterministic baselines (Shannon, Hill, stats) |
| `manifest-ipc-parity` | Composition | Live | 15 downstream capabilities vs live barraCuda |
| `cross-atomic-pipeline` | Pipeline | Live | BearDog→NestGate integrity |
| `gonzales-provenance` | Pharmacology | Both | IC50 dose-response + trio tracking |

### 5. UniBin Binary

`wetspring_unibin` — single binary with subcommands:

```
wetspring_unibin certify [--layer N] [--bare]
wetspring_unibin validate [--track T] [--scenario S] [--tier T] [--list]
wetspring_unibin serve
wetspring_unibin status
wetspring_unibin version
```

### 6. Fossilization

Pre-extinction guidestone archived to `fossilRecord/guidestone_prokaryotic_may2026/`
with provenance README. Original binary retained for backward compatibility.

### 7. Debt String Cleanup

All `debt`/`deep debt` strings in runtime output replaced with `evolution` terminology.
Zero DEBT/FIXME/TODO/HACK comment markers in active `.rs` code.

---

## Build Status

| Check | Result |
|-------|--------|
| `cargo build --workspace` | PASS |
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace --all-targets` | 0 errors (pre-existing warnings in exp400/benchmarks only) |
| `cargo test -p wetspring-barracuda --lib` | 1,209 passed |
| `cargo test -p wetspring-barracuda --features guidestone --lib` | 1,594 passed |
| `#[allow()]` without reason | 0 |
| `#[deprecated]` without note | 0 |
| TODO/FIXME/HACK/DEBT markers | 0 |
| `PrimalClient` | 0 |
| `AtomicHarness` | 0 |
| `spawn_primal` | 0 |
| `probe_primal` | 0 |
| `neural_api_healthy` | 0 |

## Interstadial Compliance

| Requirement | Status |
|-------------|--------|
| Replace deprecated IPC patterns | DONE (new code uses `CompositionContext`) |
| `#[deprecated]` with note | N/A (no deprecated attrs in wetSpring) |
| `#[allow]` with reason | DONE (0 `#[allow]`, all `#[expect(reason)]`) |
| Zero TODO/FIXME/HACK/DEBT | DONE |
| cargo build/fmt/clippy/test clean | DONE |
| Begin UniBin evolution | DONE (`wetspring_unibin`) |
| Certification module | DONE (`certification/`) |
| Validation scenarios | DONE (`validation/scenarios/`, 4 scenarios) |
| Two-tier validation | DONE (Tier::Rust + Tier::Live) |
| Pin primalSpring v0.9.25 | DONE (path dep resolves to v0.9.25) |
| Fossilize pre-extinction | DONE (`fossilRecord/`) |
| Per-trio provenance split | DONE (`ipc/provenance/{rhizocrypt,loamspine,sweetgrass}`) |
| `primal-proof` feature flag | DONE |

## Remaining Evolution Targets (Tier 3)

1. **Route handler compute through ecobin barraCuda IPC** — 15 handlers still call `crate::bio::*` and `barracuda::*` in-process. Gate behind `primal-proof` feature.
2. **Absorb remaining experiment binaries** — 342 bins can be selectively absorbed as scenarios. Start with the most exercised validation binaries.
3. **Server absorption** — `wetspring serve` (current IPC server) should be consolidated into the UniBin `serve` subcommand.
4. **`discover_primal()` full deprecation** — Legacy binaries still use `discover::discover_primal()` directly. New code uses `CompositionContext`.

## For primalSpring

wetSpring is now at Interstadial compliance for v0.9.25 patterns:
- Per-trio provenance split ✓
- UniBin with certify/validate/serve/status/version ✓
- Two-tier validation with ScenarioMeta ✓
- Certification organelle ✓
- `primal-proof` feature flag ✓

The guidestone prokaryotic binary is fossilized. All new composition code
uses `CompositionContext::from_live_discovery_with_fallback()` and `ctx.call()`.

## For Other Springs

wetSpring's per-trio module split is a clean pattern to follow:
- Shared types in `provenance/mod.rs`
- Per-primal operations in `provenance/{rhizocrypt,loamspine,sweetgrass}.rs`
- `capability_call()` as `pub(crate)` for internal reuse
- IPC handlers in `mod.rs` for the JSON-RPC dispatch surface

The `primal-proof` feature flag follows healthSpring's dual-lane pattern.
