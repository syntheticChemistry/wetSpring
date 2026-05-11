# wetSpring V159 — Deep Debt Audit, barraCuda IPC Routing & Upstream Primal Handoff

**Date:** 2026-05-11
**Version:** V159
**From:** wetSpring
**To:** primalSpring, upstream primal teams, other springs, projectNUCLEUS, foundation

---

## Summary

V159 completes a comprehensive deep debt audit and wires the first handler-level
`primal-proof` IPC routing for barraCuda. New `ipc/barracuda_route.rs` module
enables sovereign compute routing. Three science handlers now attempt barraCuda
IPC first with graceful in-process fallback. All documentation synchronized to
current metrics. Paper count reconciled (63). Broken provenance reference fixed.

**Status after V159:** 8 gaps open, 14 resolved/closed. 1,865 lib tests +
97 integration. Zero clippy warnings. Zero unsafe code. Zero production mocks.
Zero `todo!()` in library code. Zero hardcoded paths. All external deps pure Rust.

---

## 1. barraCuda IPC Routing Module (`ipc/barracuda_route.rs`)

New module for sovereign NUCLEUS deployment — routes math calls through a
remote barraCuda primal over JSON-RPC when `primal-proof` feature is active:

- `discover()` — standard socket cascade (`BARRACUDA_SOCKET` → XDG → temp)
- `forward(socket, method, params)` — typed JSON-RPC call with `COMPUTE` timeout
- `try_forward(method, params)` — discover + forward, returns `None` on any failure
- `is_available()` — health check for circuit-breaker logic
- Atomic request IDs for multiplexed connections
- 5 tests (discovery, availability, forwarding, error paths, ID sequencing)

**For barraCuda team:** wetSpring now has a complete IPC client targeting
barraCuda's v0.9.17 surface. When a barraCuda primal is live on the same
host, `cargo build --features ipc,primal-proof` will route diversity, Anderson
spectral, and QS ODE calls through IPC automatically.

## 2. Handler-Level `primal-proof` Wiring (PG-09)

`#[cfg(feature = "primal-proof")]` branches wired into three science handlers:

| Handler | IPC Method | Fallback |
|---------|-----------|----------|
| `handle_diversity` | `stats.diversity` | In-process `bio::diversity` + optional GPU |
| `handle_anderson` | `spectral.anderson_3d` | In-process `barracuda::spectral` (GPU) or `-32001` error |
| `handle_qs_model` | `compute.ode_rk4` | In-process `bio::qs_biofilm` ODE |

Pattern follows ludoSpring's `local` feature gate. Each handler attempts IPC
first; on any failure (socket absent, transport error, RPC reject), falls
through to the existing in-process path transparently.

**For primalSpring:** This is the dual-lane pattern documented in JH-11. When
barraCuda is live, wetSpring can run as a pure IPC compositor with zero
in-process math dependencies. The `barracuda` Cargo dependency becomes
optional for the `primal-proof` build path.

## 3. Deep Debt Audit Results

Comprehensive audit across all categories — wetSpring is debt-free:

| Category | Result |
|----------|--------|
| Unsafe code | Zero — `#![forbid(unsafe_code)]` workspace-level |
| Production mocks | Zero — all mocks isolated to `#[cfg(test)]` |
| `todo!()` / `unimplemented!()` | Zero in library (only in `no_run` doc examples) |
| Hardcoded paths/URLs | Zero — all env-configurable via standard vars |
| External non-Rust deps | Zero (wgpu native driver layer is inherent) |
| Library files >800L | Zero — max 617L; 4 GPU bins >800L use shared harness |
| `#[allow()]` | Zero — all suppressions use `#[expect(reason)]` |
| Dead code | Intentional (deserialization fields, BearDog scaffold) with `reason` |

**For all springs:** This audit methodology can be replicated. The key checks:
`forbid(unsafe_code)`, `#[expect(reason)]` for all suppressions, env-var
configs for all URLs/paths, `provenance_registry.rs` for script baselines.

## 4. Broken Provenance Reference Fixed

`scripts/diversity/compute_stats.py` was referenced in `wetspring_guidestone`
and `certification/bare` but the file did not exist. The check validates
`mean([10..50]) = 30.0`, which is an analytical identity — updated provenance
string to `"Analytical identity: mean(10..50) = 30.0 (exact)"`. Fossil record
copy (`fossilRecord/guidestone_prokaryotic_may2026/`) also fixed.

## 5. Documentation Synchronization

All docs now consistently report V159 metrics:

| Metric | Value |
|--------|-------|
| Lib tests | 1,865 |
| Integration tests | 97 |
| IPC roundtrip tests | 18 |
| Binaries | 364 (342 barracuda + 22 forge) |
| Validation checks | 5,900+ |
| Experiments | 383 (380 + 3 proposed) |
| Papers | 63/63 |
| Primal gaps | 8 open, 14 resolved/closed |
| Consumed capabilities | 48 (33 v0.9.17 + 15 legacy) |
| Deploy graphs | 7 |
| Coverage | 91.20% line / 90.30% function |

**Files updated:** README.md, CONTEXT.md, GAPS.md, CHANGELOG.md,
whitePaper/README.md, whitePaper/baseCamp/README.md,
whitePaper/baseCamp/EXTENSION_PLAN.md, experiments/README.md,
barracuda/README.md, barracuda/EVOLUTION_READINESS.md,
barracuda/ABSORPTION_MANIFEST.md, specs/README.md, docs/PRIMAL_GAPS.md.

---

## Upstream Primal Team Guidance

### For barraCuda

- wetSpring's `ipc/barracuda_route.rs` is a complete IPC client for v0.9.17
- PG-17 (tensor.matmul handle-based only) remains open — inline data path
  would simplify integration for springs without handle caching
- The `COMPUTE` timeout (30s) may need adjustment for large spectral workloads

### For toadStool

- PG-05 (toadStool discovery + barraCuda optional) remains open
- wetSpring registers `compute.dispatch` via `ipc/compute_dispatch.rs` but
  toadStool discovery depends on live sovereign dispatch wiring
- ToadStool method `compute.dispatch.submit` confirmed working in test harness

### For Songbird

- PG-03 (capability-based discovery) is partial — `discover_by_capability()`
  provides a compile-time map as bridge until Songbird implements
  `capability.resolve` at runtime
- The mapping covers 15 domains across 10 primals — Songbird should expose
  at least these domains in its capability registry

### For BearDog

- `witness_signature` scaffolded in `facade/provenance.rs` for Tier 2
  signature witnessing (BearDog `crypto.sign_ed25519`)
- Currently `dead_code` with `reason` — activates when BearDog IPC is live

### For skunkBat

- V158 wired `ipc/skunkbat.rs` with `audit.event` and `audit.forward`
- Emitting structured events with domain/action/severity payloads
- Waiting for Phase 3 (JH-5 forwarding) to enable cross-primal audit chain

### For Provenance Trio (rhizoCrypt, loamSpine, sweetGrass)

- PG-02 and PG-18 remain open — trio IPC readiness and UDS connection reset
- wetSpring has full provenance module wired (`ipc/provenance/`) with session
  lifecycle (begin → record → complete → dehydrate → commit → attribute)
- Foundation seeding (thread04) depends on sweetGrass braid integration

### For NestGate

- PG-04 remains open — NestGate IPC wired but deployment pending
- NCBI fetch uses tiered strategy: biomeOS → NestGate → sovereign HTTP

### For primalSpring

- wetSpring is now at 8 open gaps (was 15 per May 10 audit)
- CI cross-sync validates against 413 canonical methods
- `capability_to_primal()` mapping should be reviewed for completeness
  when new primals are added to the registry
- All 7 deploy graphs validate structurally via `graph_validate.rs`

---

## Composition Patterns for NUCLEUS Deployment

### Current State

wetSpring exposes a complete UniBin (`wetspring_unibin`) with subcommands:
`validate`, `certify`, `serve`, `list`, `info`. NUCLEUS workloads in
`projectNUCLEUS/workloads/wetspring/` (11 TOMLs) currently target per-validator
binaries. Migration to UniBin dispatch is the next deployment step.

### neuralAPI / biomeOS Integration

- `method.register` consumed (biomeOS v3.51) — wetSpring registers capabilities
  on startup when biomeOS is live
- `composition.status` consumed — health reporting feeds deployment graphs
- `composition.deploy(graph)` — 7 deploy graphs ready for NUCLEUS dispatch
- Science facade (`wetspring_science_facade`) serves HTTP API for extracellular
  access via primals.eco CDN

### Cell Membrane Awareness

- **Extracellular:** primals.eco CDN (science facade HTTP endpoints)
- **Membrane:** lab/git.primals.eco tunnel (source, CI)
- **Intracellular:** sovereign compute (all math, IPC, GPU dispatch)
- wetSpring respects `GATE_HOME`, `ABG_SHARED` when set
- No hardcoded paths — all discovery is env-driven

---

## Foundation Seeding Status

V158 handoff proposed `thread04_enviro_targets.toml` entries for:
1. NCBI 16S pipeline (PRJNA488170 — community reconstruction)
2. Cold seep metagenomics (PRJNA382322 — multi-biome assembly)
3. Fajgenbaum pathway scoring (NMF + drug-target networks)

**Current foundation state:** Thread04 has `data/sources/thread04_enviro.toml`
but no `data/targets/thread04_enviro_targets.toml` and no
`expressions/ENVIRONMENTAL_GENOMICS.md`. These are the highest-priority
foundation items for environmental genomics validation.

---

## Remaining Open Gaps by Owner

| Owner | Gaps | Items |
|-------|------|-------|
| **wetSpring** | 2 | PG-09 (handler wired, remaining: gonzales/kinetics), PG-12 (Exp403 legacy migration) |
| **External** | 5 | PG-02 (trio IPC), PG-04 (NestGate), PG-05 (toadStool), PG-06 (ionic bond), PG-18 (trio UDS) |
| **Mixed** | 2 | PG-03 (Songbird capability.resolve), PG-10 (primalSpring routing) |
| **barraCuda** | 1 | PG-17 (matmul inline data path) |

---

*This handoff is published to `wateringHole/handoffs/` per the NUCLEUS_SPRING_ALIGNMENT.md
feedback protocol and is available for upstream primalSpring audit.*
