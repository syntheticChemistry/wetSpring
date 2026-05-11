<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# wetSpring V161 — Upstream Primal & Spring Team Handoff

**Date:** May 11, 2026
**Version:** V161
**From:** wetSpring
**To:** All primal teams, all spring teams, projectNUCLEUS, foundation

---

## Context

wetSpring has zero internal gaps. All 8 remaining open PGs are external
dependencies. The next round of evolution requires **full data and compute
chains** end-to-end. This handoff identifies what each upstream team needs
to deliver for that to happen, prioritized by blocking impact.

**wetSpring state:** 1,962 tests, 0 failures. 63/63 papers. guideStone
Level 4. Foundation Thread 04 seeded (36 targets). All handlers primal-proof
wired. Deep debt audit clean across all dimensions.

---

## HIGH PRIORITY — NestGate Live Deployment (PG-04)

**Owner:** NestGate team
**Impact:** Blocks ALL real-data pipelines. Without NestGate, no spring can
fetch external datasets through the primal composition. Every `data.fetch`
call degrades to gap-report mode.

**What wetSpring has done:**
- `data_fetch.rs` routes all external fetches through
  `capability.call("storage", "fetch_external")` via biomeOS → NestGate
- `capability.call("storage", "store")` and `capability.call("storage", "retrieve")`
  for cache persistence and retrieval
- Exp400 validates NestGate health and cross-atomic store→retrieve pipeline
- Pure primal composition: no fallbacks, gap reports on missing primals

**What NestGate needs to deliver:**
1. Live deployment in NUCLEUS stack (UDS socket at discoverable path)
2. `storage.fetch_external` responding to JSON-RPC with HTTP proxy semantics
3. `storage.store` / `storage.retrieve` for cache persistence
4. Health endpoint (`health.liveness`) on the NestGate socket

**Why this is P0:** Without NestGate, the EMP real OTU tables, KBS LTER
data, SRA longitudinal atlas, and AMR gene databases cannot be fetched
through composition. Springs fall back to direct HTTP (breaking the
membrane architecture) or skip entirely.

---

## HIGH PRIORITY — Provenance Trio IPC Readiness (PG-02, PG-18)

**Owner:** rhizoCrypt, loamSpine, sweetGrass teams
**Impact:** Blocks provenance DAG, ledger, and braid for all springs.
Compositions run without provenance tracking.

**What wetSpring has done:**
- `ipc/provenance.rs` sends `capability.call` to `dag.session.create`,
  `dag.event.append`, `dag.dehydrate` (rhizoCrypt), `session.commit`
  (loamSpine), and `braid.create` (sweetGrass) via Neural API socket
- `ipc/sweetgrass.rs` has typed `BraidRequest`/`BraidCommitRequest` structs
- `WireWitnessRef` added per Attestation Encoding Standard v2.0
- Graceful degradation on all trio paths
- `CONSUMED_CAPABILITIES` declares `dag.session.create`, `entry.append`,
  `braid.create`, `braid.commit` per Wire Standard L3

**What trio teams need to deliver:**
1. JSON-RPC on UDS (currently: sockets accept connections but reset on
   JSON-RPC send — PG-18)
2. Stable endpoints for the methods listed above
3. Health endpoints (`health.liveness`) on trio sockets

**Why this is P0:** Foundation seeding is geological-layer work. Without
provenance, validated results lack BLAKE3-anchored DAG chains and
sweetGrass braids. The science is correct but not load-bearing.

---

## MEDIUM PRIORITY — Songbird Capability Resolution (PG-03)

**Owner:** Songbird team / biomeOS
**Impact:** Springs use name-based discovery with a capability abstraction
layer. Full capability-based routing blocked on Songbird.

**What wetSpring has done:**
- `discover_by_capability(domain)` maps 15 capability domains to provider
  primals and resolves sockets — single migration point
- `capability_to_primal(domain)` provides canonical mapping at `const` time
- Tests cover all known domain mappings and unknown-domain handling

**What Songbird needs to deliver:**
- `capability.resolve` JSON-RPC method: given a capability domain string,
  return the provider primal name and socket path
- When this ships, `discover_by_capability` swaps its internals from
  `capability_to_primal → discover_primal` to a single Songbird RPC call.
  All callers unchanged.

---

## MEDIUM PRIORITY — toadStool Sovereign Dispatch (PG-05)

**Owner:** toadStool team
**Impact:** Full compute dispatch via toadStool IPC for GPU workloads.

**What wetSpring has done:**
- `discover_toadstool()` helper resolves toadStool socket
- `barracuda` marked `optional = true` in `Cargo.toml` with `barracuda-lib`
  feature gate
- Exp400 validates `compute.health` via biomeOS
- 5 science handlers primal-proof wired with IPC-first routing

**What toadStool needs to deliver:**
- Sovereign dispatch path: `compute.dispatch.submit` with compiled GPU binary
  bytes (coralReef pipeline)
- Currently responds "Missing 'binary' array" — the path exists but needs
  the coralReef → toadStool pipeline active

---

## MEDIUM PRIORITY — barraCuda tensor.matmul Inline Data (PG-17)

**Owner:** barraCuda team
**Impact:** `tensor.matmul` requires pre-created tensor handles. Springs
cannot do simple inline matrix multiply without the create→matmul→query flow.

**What wetSpring has done:**
- guideStone works around via create→matmul→check-shape
- `validate_parity` cannot be used for matmul (expects scalar `result`)

**Suggestion:** Add an inline-data convenience path to `tensor.matmul`
that accepts `{"a": [[...]], "b": [[...]]}` and returns `{"result": [[...]]}`.
Alternatively, add a `tensor.matmul_inline` method.

---

## LOWER PRIORITY — primalSpring Composition Routing (PG-10)

**Owner:** primalSpring
**Impact:** `method_to_capability_domain()` misroutes `spectral.*` and
`linalg.*` methods.

**Fix:** Add `"spectral" | "linalg"` to the `"tensor"` match arm in
`method_to_capability_domain()`. One-line fix.

---

## LOWER PRIORITY — Ionic Bond Protocol (PG-06)

**Owner:** primalSpring Track 4
**Impact:** No automated bond negotiation. Bonding metadata is declared
but unused at runtime.

**Status:** Architectural — needs spec from Track 4 before implementation.

---

## For Spring Teams — Patterns to Absorb

### Handler-Level primal-proof Wiring (wetSpring exemplar)

wetSpring's V159-V160 pattern for graceful IPC-first routing:

```rust
pub fn handle_diversity(params: &Value) -> Result<Value, RpcError> {
    #[cfg(feature = "primal-proof")]
    if let Some(result) = super::super::barracuda_route::try_forward(
        "stats.diversity", params
    ) {
        return Ok(result);
    }
    // in-process fallback
    // ...
}
```

Key properties:
- `#[cfg(feature = "primal-proof")]` — zero-cost when feature is off
- `try_forward` returns `Option<Value>` — `None` on any IPC failure
- Falls back to in-process computation transparently
- Each handler independently decides its IPC method name

### Legacy Surface Separation (wetSpring V161 exemplar)

Split canonical and legacy capabilities into separate constants:
- `CONSUMED_CAPABILITIES`: v0.9.17 canonical surface
- `CONSUMED_CAPABILITIES_LEGACY`: pre-canonical methods

Makes CI and composition tools able to distinguish without parsing comments.

### Foundation Seeding (airSpring + wetSpring exemplars)

airSpring seeded Thread 06 (36/36 targets). wetSpring seeded Thread 04
(36 targets). Pattern:
1. Create `data/targets/threadNN_*.toml` with validated targets
2. Each target: `expected_value`, `unit`, `tolerance`, `source`, `spring`, `validated`
3. Update `THREAD_INDEX.toml`: status → "seeded", set `data_targets` path

---

## For projectNUCLEUS — Composition Deployment Readiness

| Spring | UniBin | Workloads | plasmidBin | Foundation | Notes |
|--------|--------|-----------|------------|------------|-------|
| wetSpring | ready | 11 TOMLs | 1.4M binary | Thread 04 seeded | All handlers primal-proof wired |
| airSpring | ready | 1 TOML | 3.0M + 2.4M | Thread 06 seeded | 9 UniBin scenarios |
| neuralSpring | ready | 2 TOMLs | pending | Threads 5+7 documented | `compute.dispatch` wired |
| hotSpring | ready | 1 TOML | pending | Thread 02 seeded | 188 experiments, gS L6 |
| groundSpring | ready | 1 TOML | pending | pending | Modularized guidestone (833→128L) |
| ludoSpring | ready | 2 TOMLs | pending | pending | Tier 4 exemplar |
| healthSpring | ready | 3 TOMLs | pending | pending | skunkBat in deploy graphs |

**Critical dependency chain for full science validation:**
1. NestGate live → data fetching through membrane
2. Provenance trio JSON-RPC → DAG/braid tracking
3. Songbird `capability.resolve` → name-free discovery
4. plasmidBin binaries for all 7 springs → NUCLEUS workload dispatch

---

## Next Round of Evolution — Full Data and Compute Chains

The post-interstadial non-compositional validation layer is complete (12,900+
tests, 8/8 springs). The **next phase** requires:

1. **NestGate deployed** — so `data.fetch` routes through the cell membrane
   instead of direct HTTP. Every spring's real-data pipeline depends on this.
2. **Provenance trio speaking JSON-RPC** — so validated results get BLAKE3
   DAG chains and sweetGrass braids. Foundation geological layers need this.
3. **BearDog TLS shadow run** (H2-12, port 8443) — upstream shipped, ready
   for ops. Enables secure external access through primals.eco.
4. **Songbird NAT VPS relay** (H2-14) — upstream shipped, ~$5/mo commodity.
   Enables discovery across network boundaries.

Without items 1-2, the "Python → Rust → Primal (NUCLEUS composition)
validation of peer-reviewed science" narrative stops at "Primal" — the
composition can validate math but cannot fetch, store, or provenance-track
real-world data through the membrane.

---

*This document is maintained by wetSpring and fed back to primalSpring via
the wateringHole handoff protocol.*
