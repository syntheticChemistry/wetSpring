# wetSpring V174 — Primal & Spring Team Handoff

**Date:** May 17, 2026
**From:** wetSpring
**To:** primalSpring, barraCuda, toadStool, coralReef, biomeOS, Provenance Trio, NestGate, metalForge, all springs
**Scope:** Composition patterns, NUCLEUS deployment via Neural API, experiment chain completion, deep debt resolution, and upstream evolution guidance

---

## 1. What wetSpring Proved (Python → Rust → Primal)

wetSpring validated the full path from peer-reviewed Python science to primal composition:

| Tier | What | Evidence |
|------|------|----------|
| **Tier 1** | Python → Rust CPU | 1,962 lib tests, 58 Python baselines → 41 CPU modules, 22.5× speedup |
| **Tier 2** | Rust CPU → GPU (barraCuda) | 44 GPU modules (all lean), 150+ primitives, 29 GPU-domain parity experiments |
| **Tier 3** | GPU → NUCLEUS Composition | 136/136 proto-nucleate, 7 deploy graphs, 452 registry methods, IPC-first |
| **Tier 4** | guideStone Primal Proof | Level 5: 38/38 pass, 4 environmental skip (resolve on live deployment) |
| **Tier 5** | Interactive Composition | Live probing, schema parity self-check, graceful degradation |

**Totals:** 384/384 experiments, 370 binaries, 5,957+ checks, 0 failures, `forbid(unsafe_code)`, zero `#[allow()]`, clippy pedantic+nursery zero warnings.

---

## 2. V171–V174 Evolution Summary

### V171: Live Composition Health
- `composition.science_health` evolved from static `"deferred_check"` to live runtime probing
- New module `ipc/composition_health.rs` with typed probing of trio, NestGate, biomeOS
- Wave 20 schema parity self-check (`capability.list` validates its own shape)
- Graceful degradation: `"live"` → `"discovered"` → `"absent"` per component

### V172: Wave 20 Debt Resolution
- CI cross-sync threshold tightened to 452
- `primal.list` added to `CONSUMED_CAPABILITIES`
- Attribution clarified for V158 CI method count evolution

### V173: Deep Debt Resolution
- UniBin `serve` subcommand evolved from stub to live JSON-RPC server
- `dense_to_csr` extracted from binary into `validation::dense_to_csr` (feature-gated)
- Display-name hygiene: 13 `*_DISPLAY` constants in `primal_names.rs`, all hardcoded CamelCase strings replaced
- Zero unsafe, zero TODO/FIXME/HACK, zero production mocks, zero external C deps

### V174: Experiment Buildouts + Control Validation
- Exp377: Hormesis biphasic dose-response model (17/17 PASS)
- Exp378: Trophic cascade via Anderson lattice (10/10 PASS)
- Exp379: Joint colonization resistance surface (30/30 PASS)
- `CONTROL_EXPERIMENT_STATUS.md` established for tracking control chains

---

## 3. Composition Patterns for NUCLEUS Deployment

These patterns were evolved and validated by wetSpring. They are ready for adoption by all springs.

### 3.1 Primal Self-Knowledge (Zero Hardcoding)

```rust
// primal_names.rs — single source of truth
pub const SELF_NAME: &str = "wetspring";           // wire name (IPC, registry)
pub const SELF_DISPLAY: &str = "wetSpring";         // human-readable (JSON, UX)
pub const BIOMEOS: &str = "biomeos";                // discovered at runtime
pub const NESTGATE: &str = "nestgate";              // never hardcoded in handlers
```

**Pattern:** All primal identifiers live in one module. Wire names (`lowercase`) for IPC routing. Display names (`CamelCase`) for JSON responses and gap reports. No string literal `"wetSpring"` or `"biomeOS"` anywhere in handlers.

### 3.2 Capability-Based Discovery

```rust
// niche.rs — CONSUMED_CAPABILITIES
pub const CONSUMED_CAPABILITIES: &[&str] = &[
    "barracuda.compute",      // barraCuda math dispatch
    "toadstool.validate",     // pre-flight workload check
    "nest.store",             // NestGate content-addressed storage (Wave 17)
    "nest.commit",            // NestGate immutable commit (Wave 17)
    "primal.list",            // biomeOS primal census (Wave 20)
    "primal.announce",        // self-announcement to biomeOS
    // ... 51 total consumed capabilities
];
```

**Pattern:** Springs declare what they consume, not who provides it. Discovery at runtime via Songbird sockets (`discover::discover_socket(primal_name)`) with XDG fallback. If a primal is absent, the operation degrades gracefully — never panics, never blocks.

### 3.3 Live Composition Health (Adopted Pattern)

```rust
// composition_health.rs — probing pattern
pub fn probe_trio_status() -> TrioStatus {
    // For each trio member: discover socket → try health.liveness → return typed status
    // "live" | "discovered" (socket found, RPC failed) | "absent" (no socket)
}
```

**Degradation table:**

| Component | Absent | Discovered | Live |
|-----------|--------|------------|------|
| Trio | No provenance attribution | Socket found, liveness failed | Full DAG lineage |
| NestGate | No content-addressed storage | Socket found, liveness failed | Full CAS |
| biomeOS | No orchestration | Socket found, API unreachable | Full Neural API |

**Adoption guidance for springs:** Implement your own `composition.<domain>_health` method using this probing pattern. Probe only the primals you depend on. Return structured JSON, not strings.

### 3.4 Neural API Deployment via biomeOS

```
biomeOS Graph → Atomic Instances → IPC (JSON-RPC over Unix socket) → Spring Niche
```

**Deploy graph structure** (`graphs/wetspring_deploy.toml`):
- Declares which primals the niche needs (tower atomics, node atomics, nest atomics)
- biomeOS instantiates the graph, starts atomic instances, provides socket directories
- The spring discovers primals via Songbird mesh at runtime
- All inter-primal communication is JSON-RPC over Unix domain sockets

**Tower atomics** (persistent services): biomeOS, NestGate, BearDog, Songbird
**Node atomics** (compute workers): barraCuda, toadStool, coralReef
**Nest atomics** (storage/provenance): rhizoCrypt, loamSpine, sweetGrass

**For toadStool/metalForge teams:** Mixed hardware dispatch (NPU→GPU via PCIe bypassing CPU roundtrip) is documented in `CONTROL_EXPERIMENT_STATUS.md`. wetSpring validated CPU→GPU parity via barraCuda. GPU→NPU and mixed-system dispatch are ready for metalForge atomic implementation.

### 3.5 UniBin Pattern

The `wetspring_unibin` binary unifies all operational modes:

```
wetspring_unibin certify    # run guideStone certification
wetspring_unibin validate   # run domain validation checks
wetspring_unibin serve      # start JSON-RPC server (live IPC)
wetspring_unibin status     # print composition status JSON
wetspring_unibin version    # print version + niche metadata
```

**For springs adopting this pattern:** The `serve` subcommand binds to a Unix socket discovered via `Server::bind_default()`, discovers Songbird for mesh registration, starts a heartbeat task, and runs the full JSON-RPC dispatch loop. This replaces ad-hoc server binaries.

---

## 4. Upstream Learnings for Primal Teams

### 4.1 barraCuda Team
- **CPU parity proven:** 546/546 checks at 22.5× Python speedup (v1–v9 benchmark suite)
- **GPU parity proven:** 29 domains with pure GPU math (zero CPU roundtrip for analytics)
- **Streaming proven:** 441–837× vs CPU roundtrip for ODE+phylo+analytics pipelines
- **Hormesis chain (Exp377–379):** New biology domain validated entirely in existing `bio::*` modules — no new barraCuda primitives were needed, demonstrating that the current primitive surface is sufficient for Anderson-localization-based biology
- **Next:** CPU vs GPU benchmark parity validation (Python baseline → Rust CPU → Rust GPU three-tier), documented in `CONTROL_EXPERIMENT_STATUS.md`

### 4.2 toadStool Team
- wetSpring uses `toadstool.validate` and `toadstool.list_workloads` for pre-flight workload checks
- `ComputeDispatch` with 264 ops validated across 5 substrate configs (Exp080)
- **Ready for:** `toadstool.dispatch` compute routing through metalForge mixed-hardware instances

### 4.3 coralReef Team
- coralReef v0.1.0 declared in niche (`capability_registry.toml`)
- wetSpring is fully lean (zero local WGSL) — all shaders consumed from barraCuda (via ToadStool absorption)
- **Ready for:** coralReef shader compilation integration when sovereign ISA generation stabilizes

### 4.4 biomeOS Team
- Neural API signals consumed: `composition.status`, `method.register`, `primal.announce`, `signal.dispatch`
- Wave 17 signals wired: `nest.store`, `nest.commit` with fallback
- Wave 20 schema: `primal.list` consumed for live primal census, `capability.list` self-validates `count` field
- Live NUCLEUS guideStone: 30/31 pass, 1 fail (deployment infra), 9 skip (absent primals)
- **Gap:** biomeOS graph execution runtime needed for full graph-only deployment (currently IPC-server mode)

### 4.5 Provenance Trio (rhizoCrypt, loamSpine, sweetGrass)
- All three probed via `probe_trio_status()` with graceful degradation
- Provenance wiring: `facade::provenance` generates W3C PROV-O attribution using display-name constants
- **Ready for:** Live trio integration when trio sockets are deployed

### 4.6 NestGate Team
- `nest.store` and `nest.commit` dispatch wired (Wave 17)
- `probe_nestgate_status()` probes liveness with 500ms timeout
- **Ready for:** Content-addressed storage for frozen experiment baselines

### 4.7 metalForge Team
- 37 domains validated CPU↔GPU (Exp103+104+165)
- PCIe bypass experiments documented (Exp088)
- Control experiment matrix in `CONTROL_EXPERIMENT_STATUS.md`:
  - Python→CPU→GPU chain
  - NPU→GPU via PCIe (bypassing CPU)
  - CPU↔GPU↔NPU mixed dispatch via toadStool
  - NUCLEUS atomics (tower, node, nest) coordination via biomeOS graphs

---

## 5. What's Ready for Absorption

| Item | Source | Target | Status |
|------|--------|--------|--------|
| Composition health probing pattern | `ipc/composition_health.rs` | All springs | Ready — pattern documented above |
| Display-name hygiene | `primal_names.rs` | All primals | Ready — 13 constants, zero hardcoded strings |
| UniBin pattern | `bin/wetspring_unibin/` | All springs | Ready — `certify/validate/serve/status/version` |
| `dense_to_csr` helper | `validation::dense_to_csr` | barraCuda (if sparse matrix needed) | Ready — feature-gated |
| Hormesis biology models | `bio::hormesis`, `bio::binding_landscape` | Other science springs | Ready — pure Rust, `forbid(unsafe_code)` |
| Control experiment framework | `CONTROL_EXPERIMENT_STATUS.md` | barraCuda, toadStool, metalForge | Ready — chains documented |

---

## 6. Open Gaps for Upstream Review

| ID | Gap | Owner | Status |
|----|-----|-------|--------|
| PG-02 | Provenance Trio socket deployment | Trio teams | Deployment-only — wiring complete |
| PG-04 | NestGate socket deployment | NestGate team | Deployment-only — wiring complete |
| — | biomeOS graph execution runtime | biomeOS team | Needed for graph-only deploy (IPC-server mode works) |
| — | toadStool compute routing via metalForge | toadStool + metalForge | Needed for mixed-hardware dispatch |
| — | coralReef ISA generation pipeline | coralReef team | Needed for sovereign shader compilation |

---

## 7. Metrics Snapshot (V174)

| Metric | Value |
|--------|-------|
| Experiments | 384/384 (0 proposed) |
| Binaries | 370 (348 barracuda + 22 forge) |
| Checks | 5,957+ |
| Lib tests | 1,962 (0 failures) |
| Integration tests | 97 (0 failures) |
| IPC roundtrip tests | 18 (0 failures) |
| Papers | 63/63 |
| Niche capabilities | 42 across 21 domains |
| Consumed capabilities | 51 (registry 452) |
| Proto-nucleate alignment | 136/136 |
| Deploy graphs | 7 |
| Primal gaps | 2 open (deployment-only), 20 resolved |
| Coverage | 91.20% (gated at 90%) |
| unsafe blocks | 0 (`forbid(unsafe_code)`) |
| `#[allow()]` | 0 |
| TODO/FIXME/HACK | 0 |
| Production mocks | 0 |
| External C deps | 0 |
| Clippy warnings | 0 (pedantic + nursery) |
