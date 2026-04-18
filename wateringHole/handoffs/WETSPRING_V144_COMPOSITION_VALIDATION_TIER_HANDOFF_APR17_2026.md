<!--
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# wetSpring V144 — Composition Validation Tier: Python → Rust → Primal

| Field | Value |
|-------|-------|
| **Spring** | wetSpring |
| **Version** | V144 |
| **Date** | 2026-04-17 |
| **barraCuda** | 0.3.12 |
| **Wire Standard** | L2 + L3 |
| **Proto-nucleate** | 136/136 (D01–D07) |
| **IPC parity** | 43/43 (Exp401) |
| **Niche parity** | 63/63 (Exp402) |
| **IPC roundtrips** | 18 integration tests |
| **Deploy graphs** | 7 canonical (`[[graph.nodes]]`) |
| **Niche capabilities** | 42 (37 dispatch) |
| **Status** | All quality gates green |

---

## 1. Evolution Narrative

wetSpring completes the three-tier validation evolution:

```
Tier 1: Python validates Rust (science fidelity)
  71 Python/R scripts → 1,592+ Rust lib tests → 47 GPU modules
  Python was the validation target for Rust.

Tier 2: Rust validates NUCLEUS composition (primal patterns)
  340 validation binaries → 136/136 proto-nucleate → 7 deploy graphs
  Rust + Python are now validation targets for NUCLEUS composition.

Tier 3: Composition validates deployment (IPC parity → ecoBin → biomeOS)
  Exp401 (43/43) → Exp402 (63/63) → 18 IPC roundtrips → plasmidBin harvest
  IPC dispatch results match local Rust baselines.
```

This is the pattern every spring should follow: prove Python baselines, prove
Rust math, then prove that NUCLEUS composition preserves fidelity through IPC.

---

## 2. What Changed (V144 Composition Tier)

### Composition Validation Binaries

| Binary | Exp | Checks | What it validates |
|--------|-----|--------|-------------------|
| `validate_composition_nucleus_v1` | 400 | 136/136 | Proto-nucleate alignment: niche self-knowledge, capability surface, deploy graph structure, bonding metadata, atomic fragments |
| `validate_composition_parity_v1` | 401 | 43/43 | IPC science parity: dispatch results vs local Rust baselines for diversity, QS, Gonzales, Anderson, brain, provenance, AI + 7-graph structural validation via `graph_validate` + Wire Standard L2/L3 compliance |
| `validate_niche_parity_v1` | 402 | 63/63 | NICHE_STARTER_PATTERNS gate: every non-aspirational capability dispatches, niche↔graph alignment, consumed/provided accounting, science coverage, Wire Standard probes |

### IPC Roundtrip Coverage (8 → 18 tests)

New end-to-end Unix socket tests:

| Test | Methods |
|------|---------|
| `gonzales_dose_response_roundtrip` | `science.gonzales.dose_response` |
| `gonzales_pk_decay_roundtrip` | `science.gonzales.pk_decay` |
| `anderson_disorder_sweep_roundtrip` | `science.anderson.disorder_sweep` |
| `anderson_biome_atlas_roundtrip` | `science.anderson.biome_atlas` |
| `provenance_lifecycle_roundtrip` | `provenance.begin` → `record` → `complete` |
| `brain_observe_roundtrip` | `brain.observe` (36 head_outputs) |
| `brain_attention_roundtrip` | `brain.attention` |
| `brain_urgency_roundtrip` | `brain.urgency` |
| `metrics_snapshot_roundtrip` | `metrics.snapshot` |
| `ai_ecology_interpret_roundtrip` | `ai.ecology_interpret` |

### Ed25519 → BLAKE3 Keyed MAC (Tower Atomic Delegation)

`barracuda/src/vault/consent.rs` previously imported `ed25519-dalek` directly
for `ConsentTicket` signing. This violated Tower Atomic delegation — BearDog
owns crypto. Replaced with BLAKE3 keyed MAC for local self-signed consent
tickets. `ed25519-dalek` dependency removed from `Cargo.toml`.

**For BearDog team:** When `crypto.sign_ed25519` is available via IPC,
wetSpring will migrate `ConsentTicket` to asymmetric signing through the Tower
Atomic. The BLAKE3 keyed MAC is the compliant bridge.

### `metrics.snapshot` Handler

Surfaced by Exp402 as a niche/dispatch gap. Implemented in
`barracuda/src/ipc/handlers/metrics.rs`. Returns primal name, capabilities
count, dependency stats, and deploy graph path. Wired into dispatch.

### biomeOS V144 Composition Health Ownership

Universal composition methods removed from wetSpring dispatch:
- `composition.tower_health` → biomeOS
- `composition.node_health` → biomeOS
- `composition.nest_health` → biomeOS
- `composition.nucleus_health` → biomeOS

Retained: `composition.science_health` (spring-specific domain health).

Capabilities: 46 → 42 niche, 42 → 37 dispatch. Exp400 guard: 141 → 136.

### Other Changes

- **barraCuda v0.3.12** — all docs and `upstream_contract.rs` pin aligned
- **`deny.toml`** — crate-level C-dependency bans aligned with workspace root
- **Provenance registry** — `BaselineProvenance` struct extended with `commit` and `date` fields (58 entries, all `None` pending Python script re-runs)
- **Deploy graph** — `wetspring_deploy.toml` universal composition capabilities removed, `pattern_version` updated
- **`sovereign-dispatch`** feature added to `Cargo.toml`
- **`akida-driver`** path case fix (`toadStool` → `toadstool` on case-sensitive FS)

---

## 3. For primalSpring (Composition Evolution)

### Patterns Established

wetSpring demonstrates the composition validation pattern that other springs
can follow:

1. **Static composition validation** (Exp400 pattern): Niche self-knowledge
   checks, deploy graph structural validation, proto-nucleate primal coverage,
   bonding metadata, atomic fragment alignment. Guard constant enforces check
   count stability.

2. **IPC parity validation** (Exp401 pattern): Dispatch every science method,
   extract numeric results, compare against local Rust baselines with named
   tolerances. Validate all deploy graphs through `graph_validate`. Check Wire
   Standard L2/L3 shape. This is primalSpring's `validate_parity` /
   `validate_parity_vec` pattern adapted for spring IPC.

3. **Niche parity validation** (Exp402 pattern): Following
   `NICHE_STARTER_PATTERNS.md` — dispatch every niche capability with correct
   params, distinguish "method not found" (real gap) from "domain error"
   (GPU required, service offline), validate niche↔graph alignment and
   consumed/provided accounting.

### Gaps Handed Back

| ID | Gap | Owner |
|----|-----|-------|
| PG-01 | Proto-nucleate manifest → consolidated downstream_manifest.toml | primalSpring (resolved) |
| PG-02 | Provenance trio IPC (rhizoCrypt, loamSpine, sweetGrass) | Trio teams |
| PG-03 | Name-based → capability-based primal discovery | Songbird + biomeOS |
| PG-04 | NestGate `storage.fetch_external` for TLS fetch | NestGate team |
| PG-05 | Squirrel `inference.*` capabilities not yet discoverable | neuralSpring + Squirrel |
| PG-06 | petalTongue `render.dashboard` IPC | petalTongue team |
| PG-07 | biomeOS `graph.deploy` / `graph.status` IPC | biomeOS team |
| PG-08 | `spring_validate_manifest.toml` binary name inconsistency (`wetspring` vs `wetspring_primal`) | primalSpring |

### Downstream Manifest Alignment

wetSpring's entry in `primalSpring/graphs/downstream/downstream_manifest.toml`
should be updated to reflect:

- `capabilities` count: 42 (was 46)
- `barracuda_version`: 0.3.12
- Remove `composition.nucleus_health` from any capability lists
- Add `metrics.snapshot` to capability list

---

## 4. For Other Springs

### Composition Validation Checklist

Every spring reaching the composition tier should:

- [ ] Create Exp400-style static composition validator (niche + graph + proto-nucleate)
- [ ] Create Exp401-style IPC parity validator (dispatch results vs local baselines)
- [ ] Create Exp402-style niche parity validator (NICHE_STARTER_PATTERNS gate)
- [ ] Expand IPC roundtrip integration tests (every science method end-to-end)
- [ ] Remove universal `composition.*_health` handlers (biomeOS V144+)
- [ ] Retain `composition.<domain>_health` (spring-specific)
- [ ] Remove `ed25519-dalek` / `ring` / `rustls` direct deps (Tower Atomic delegation)
- [ ] Align `deny.toml` with workspace root C-dependency bans
- [ ] Update `upstream_contract.rs` or equivalent version pin
- [ ] Update plasmidBin metadata with current test/binary counts

### Brain Observer Pattern

`brain.observe` requires a `head_outputs` array of 36 floats (one per ESN
attention head). Springs implementing brain IPC must provide this array.

---

## 5. For biomeOS / Neural API

### Deployment Pattern

```toml
# biomeOS deploys wetSpring's NUCLEUS graph:
biomeos deploy --graph wetspring_science_nucleus.toml

# wetSpring exposes 37 dispatch methods via JSON-RPC 2.0 on Unix socket:
# /tmp/wetspring.sock (or WETSPRING_SOCKET env)

# biomeOS Neural API routes natural-language to wetSpring capabilities:
# "What is the Shannon diversity of this sample?"
# → ecology_semantic_mappings → science.diversity → dispatch → result
```

### Discovery

wetSpring registers 42 niche capabilities with Songbird. biomeOS discovers
wetSpring via `capability.list` (Wire Standard L2+L3). The `provided_capabilities`
array contains 21 structured domain groups with method lists. The
`consumed_capabilities` array declares 22 capabilities wetSpring needs from
Tower/Node/Nest/Meta primals.

### Health Probes

| Method | Owner | Response |
|--------|-------|----------|
| `health.liveness` | wetSpring | `{"alive": true}` |
| `health.readiness` | wetSpring | `{"ready": true, "primal": "wetspring", "subsystems": {...}}` |
| `health.check` | wetSpring | `{"healthy": true, ...}` |
| `composition.science_health` | wetSpring | `{"healthy": true, "spring": "wetSpring", "subsystems": {...}}` |
| `composition.tower_health` | biomeOS | aggregates Tower primals |
| `composition.node_health` | biomeOS | aggregates Node primals |
| `composition.nest_health` | biomeOS | aggregates Nest primals |
| `composition.nucleus_health` | biomeOS | aggregates all tiers |

---

## 6. For barraCuda

### Version Pin

wetSpring now pins `PINNED_BARRACUDA_VERSION = "0.3.12"`. All docs, metadata,
and comments align. The `sovereign-dispatch` feature is wired for testing the
sovereign dispatch pipeline via coralReef.

### Absorption Candidates

No new local WGSL (fully lean). The `metrics.snapshot` handler is pure Rust
niche metadata — no math to absorb.

---

## 7. Verification

```
cargo check --features json,ipc,facade     → clean
cargo test --features ipc --lib             → 1,592 passed, 0 failed
cargo test --features ipc --test ipc_roundtrip → 18 passed, 0 failed
Exp400: validate_composition_nucleus_v1     → 136/136 PASS
Exp401: validate_composition_parity_v1      → 43/43 PASS
Exp402: validate_niche_parity_v1            → 63/63 PASS
cargo clippy --workspace --all-targets      → 0 warnings
cargo deny check                            → advisories ok, bans ok
```

---

*Handoff crafted during V144 composition validation tier sprint.*
*Archive previous handoffs → `handoffs/archive/`.*
*primalSpring upstream will audit composition patterns and manifest alignment.*
