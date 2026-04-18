<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# wetSpring V145 — Ecosystem Evolution Handoff

**Date:** 2026-04-17
**From:** wetSpring
**To:** primalSpring, barraCuda, toadStool, coralReef, biomeOS, BearDog,
        Songbird, NestGate, Squirrel, petalTongue, rhizoCrypt, loamSpine,
        sweetGrass, all spring teams

---

## Purpose

This handoff summarizes what wetSpring learned during its evolution from Python
baseline validation (Level 1) through Rust parity (Level 2), GPU acceleration
(Levels 3-4), and now primal composition proof (Level 5 IPC-WIRED). It
documents composition patterns, NUCLEUS deployment readiness, and gaps that
feed back to primal teams for ecosystem-wide evolution.

---

## The Validation Ladder — What wetSpring Proved

```
Level 1: Python baseline (58 scripts, 63 papers reproduced)
  → Peer-reviewed science, documented provenance, public data only

Level 2: Rust validation (1,592 lib tests, 363 binaries)
  → Faithful port, forbid(unsafe_code), 91.20% coverage, zero #[allow()]

Level 3: barraCuda CPU (150+ primitives, v0.3.12)
  → Same math via barraCuda library calls, zero local WGSL

Level 4: barraCuda GPU (47 GPU modules, 800+ shaders consumed)
  → Sovereign shader execution via wgpu, toadStool dispatch

Level 5: Primal composition — IPC-WIRED (V145)  ← CURRENT
  → Exp403: live UDS calls to barraCuda, NestGate, Squirrel, BearDog, toadStool
  → 22 barraCuda consumed capabilities declared in niche.rs
  → check_skip for absent primals (CI-safe: exit 2 = skipped)

Level 6: NUCLEUS deployment — plasmidBin ecobins  ← NEXT
  → biomeos deploy --graph wetspring_science_nucleus.toml
  → Clean machine, no source code, just ecoBin static binaries
```

---

## Composition Patterns Discovered

### 1. Three-Tier Composition Validation

Every spring should implement three tiers of composition validation:

```
Tier 1: LOCAL_CAPABILITIES (in-process dispatch simulating JSON-RPC)
  - Always green in CI — no external dependencies
  - Validates: method routing, parameter parsing, result shapes
  - wetSpring: Exp401 (IPC parity), Exp402 (niche parity)

Tier 2: IPC-WIRED (live primal calls with check_skip)
  - Calls primals over real UDS sockets
  - Degrades gracefully when primals absent (exit 2 = skip)
  - Validates: IPC transport, serialization, primal discovery
  - wetSpring: Exp403 (primal parity — 5 primals, 6 domains)

Tier 3: FULL NUCLEUS (proto-nucleate from plasmidBin ecobins)
  - biomeOS deploys the graph, spring validates externally
  - No source code at runtime — just static binaries
  - wetSpring: not yet implemented (Level 6)
```

### 2. Socket Discovery Cascade

All primal discovery should follow the three-step cascade:

1. Explicit env var: `{PRIMAL}_SOCKET` (operator override)
2. XDG runtime: `$XDG_RUNTIME_DIR/biomeos/{primal}-{family_id}.sock`
3. Temp fallback: `{temp_dir}/{primal}-{family_id}.sock`

Family ID (`FAMILY_ID` or `BIOMEOS_FAMILY_ID` env var) enables multi-instance
deployments on the same host.

### 3. check_skip Pattern for CI Safety

Every composition validation binary should implement:

```rust
fn check_skip(label: &str, socket: Option<&PathBuf>, primal: &str) -> bool {
    if socket.is_none() {
        println!("    [SKIP] {label} — {primal} socket not found");
        return false;
    }
    true
}
```

Exit semantics: **0** = pass, **1** = fail, **2** = all skipped (no primals).
This keeps CI green when primals are not deployed.

### 4. CONSUMED_CAPABILITIES Declaration

Every spring's `niche.rs` should declare both:
- `CAPABILITIES` — what the spring exposes to biomeOS
- `CONSUMED_CAPABILITIES` — what the spring consumes from other primals

This enables biomeOS to validate composition completeness without hardcoded
knowledge. wetSpring declares 42 provided + 45 consumed capabilities.

### 5. Evaporation Pattern (Library → IPC Migration)

For the primal proof, domain math migrates from library calls to IPC:

```
Level 2 (keep):  barracuda::stats::mean(&data)     // library dep
Level 5 (add):   rpc_call(sock, "stats.mean", params) // IPC dep
Validation:      assert!((ipc_result - lib_result).abs() <= tolerance)
```

The library dep stays for Level 2 comparison. The IPC path is additive.
Both paths are validated against Python baselines.

---

## What Each Primal Team Needs to Know

### barraCuda

- wetSpring now calls 22 of your 32 JSON-RPC methods over IPC (Exp403)
- The gap is in wetSpring's wiring, not in barraCuda's capabilities
- PG-09 documents the full evaporation surface (library → IPC migration table)
- Tier 2 validates: `stats.mean`, `stats.std_dev`, `stats.weighted_mean`,
  `tensor.matmul`, `rng.uniform`, `health.liveness`, `capabilities.list`,
  `identity.get`
- All 32 methods are consumed capability declarations in `niche.rs`

### toadStool

- wetSpring has `discover_toadstool()` wired in `ipc/discover.rs`
- Exp403 D06 calls `compute.dispatch` over UDS (check_skip if absent)
- PG-05: toadStool compute IPC is declared but not wired for live workloads
- For sovereign dispatch (coralReef native), compute requests would route
  through toadStool IPC — this is the expected evolution path

### NestGate

- Exp403 D03 calls `storage.store` + `storage.retrieve` over UDS
- PG-04: NestGate storage is declared as optional niche dependency
- `data.fetch.*` handlers currently store locally with BLAKE3 hashes
- Cross-spring data retrieval with provenance continuity needs NestGate IPC

### Squirrel / neuralSpring

- Exp403 D04 calls `inference.complete` over UDS (check_skip if absent)
- wetSpring has `discover_squirrel()` in `ipc/discover.rs`
- `ai.ecology_interpret` handler already dispatches to Squirrel
- Adding Squirrel to the composition immediately gains `inference.*` methods

### BearDog

- Exp403 D05 calls `crypto.hash` over UDS (check_skip if absent)
- V144: Ed25519 → BLAKE3 keyed MAC (Tower Atomic delegation compliance)
- `ed25519-dalek` removed from `Cargo.toml` — crypto delegated to BearDog
- PG-06: Ionic bond negotiation protocol not yet defined

### Songbird

- PG-03: Socket discovery is name-based, not capability-based
- `by_capability` field in proto-nucleate graphs is metadata only
- True capability-based discovery needs `capability.resolve` → socket path
- All discovery uses canonical names from `primal_names.rs`

### biomeOS

- wetSpring is ready for `biomeos deploy --graph wetspring_science_nucleus.toml`
- V144: Universal composition health methods (`composition.*_health`) removed
  from wetSpring — biomeOS v3.04+ owns orchestration health
- `composition.science_health` is the only remaining spring-specific health
- Socket discovery uses standard env var / XDG / temp cascade
- Exit codes: 0 = pass, 1 = fail, 2 = skip (for CI integration)

### Provenance Trio (rhizoCrypt, loamSpine, sweetGrass)

- PG-02: IPC clients exist (`ipc/provenance.rs`, `ipc/sweetgrass.rs`)
- Typed structs: `BraidRequest`, `BraidCommitRequest`, `WireWitnessRef`
- `CONSUMED_CAPABILITIES` declares: `dag.session.create`, `dag.event.append`,
  `spine.create`, `entry.append`, `braid.create`, `braid.commit`
- Graceful degradation: falls back to local session tracking when absent
- Blocked only by trio primals reaching IPC-ready status

---

## Composition Gaps — Feedback to primalSpring

7 open gaps, 2 resolved. Full details in `docs/PRIMAL_GAPS.md`.

| # | Gap | Owner | Status |
|---|-----|-------|--------|
| PG-01 | Proto-nucleate not parsed | wetSpring | **Resolved V141** |
| PG-02 | Provenance trio IPC | Trio teams | Partial V142 |
| PG-03 | Name-based discovery | Songbird/biomeOS | Structural |
| PG-04 | NestGate not wired | NestGate | Declared |
| PG-05 | toadStool compute IPC | toadStool | Declared |
| PG-06 | Ionic bond protocol | primalSpring Track 4 | Metadata only |
| PG-07 | Capability drift | wetSpring | **Resolved V141** |
| PG-08 | Validate manifest binary name | primalSpring | Informational |
| PG-09 | barraCuda IPC evaporation | wetSpring | In progress V145 |

---

## NUCLEUS Deployment via neuralAPI from biomeOS

The deployment story for Level 6:

```
1. Clone infra/plasmidBin/ (static ecoBin binaries for all primals)
2. Start biomeOS (the orchestrator)
3. biomeos deploy --graph wetspring_science_nucleus.toml
   → biomeOS spawns 9 UDS sockets:
     beardog, songbird, toadstool, barracuda, coralreef,
     nestgate, rhizocrypt, loamspine, sweetgrass
4. Run spring validation externally:
   cargo run --features ipc --bin validate_primal_parity_v1
   → Exp403 discovers all 5 sockets, calls by capability, validates
5. Every check: PASS / FAIL / SKIP
6. Exit 0 = primal proof passed. Exit 1 = regression. Exit 2 = skipped.
```

This is the proof that peer-reviewed science, ported from Python to Rust,
runs correctly through sovereign primal compositions deployed as static
binaries. No vendor lock-in. No cloud dependency. No C libraries. No source
code required at runtime.

---

## Architectural Gaps (GAPS.md — 7 Open)

These are ecosystem-level gaps, not composition-specific:

1. **Ionic contract negotiation** — no automated protocol
2. **Cross-spring data exchange** — RootPulse not yet designed
3. **Sovereign shader validation** — coralReef compilation not wired
4. **Neural API model registry** — NestGate ML model lifecycle
5. **Deployment health rollup** — biomeOS health aggregation
6. **Mesh topology discovery** — multi-node Songbird
7. **NPU model versioning** — toadStool model lifecycle

All 7 are external team dependencies. wetSpring code quality is green.

---

## Code Quality Summary

| Metric | Value |
|--------|-------|
| `cargo clippy` (pedantic + nursery) | 0 warnings |
| `forbid(unsafe_code)` | workspace + per-crate |
| `#[allow()]` in production | 0 (uses `#[expect(reason)]`) |
| TODO / FIXME / HACK in .rs | 0 |
| Mocks outside `#[cfg(test)]` | 0 |
| C dependencies | 0 (`flate2` uses `rust_backend`) |
| `dyn` dispatch in production | 0 (enum dispatch for ValidationSink) |
| `async-trait` crate | 0 (native async fn in traits) |
| Edition | 2024 |
| MSRV | 1.87 |
| SPDX headers | All `.rs` files |

---

*This handoff follows the `{SPRING}_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`
convention per `wateringHole/` naming standard.*
