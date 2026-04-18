<!--
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# wetSpring V144 — Composition Evolution: biomeOS v3.04 Alignment

| Field | Value |
|-------|-------|
| **Spring** | wetSpring |
| **Version** | V144 |
| **Date** | 2026-04-12 |
| **barraCuda** | 0.3.11 |
| **Wire Standard** | L2 + L3 |
| **Proto-nucleate** | 141/141 (D01–D07) |
| **Deploy graphs** | 7 canonical (`[[graph.nodes]]`) |
| **Status** | All quality gates green |

---

## 1. What Changed

### Composition Health Method Ownership

biomeOS v3.04 implemented `COMPOSITION_HEALTH_STANDARD.md` with dedicated
routing for universal composition health methods:

- `composition.tower_health`
- `composition.node_health`
- `composition.nest_health`
- `composition.nucleus_health`

These methods are orchestration concerns. Springs should not implement them —
biomeOS aggregates subsystem health from its lifecycle manager which has
visibility into all running primals.

**Action taken:** Removed all 4 universal composition handlers from wetSpring.
Retained `composition.science_health` — the spring-specific method that
reports science domain health (tracks, experiments, validation checks).

### Capability Count Changes

| Metric | V143 | V144 | Delta |
|--------|------|------|-------|
| Niche capabilities | 46 | 42 | -4 |
| Dispatch methods | 41 | 37 | -4 |
| Domain methods | 41 | 37 | -4 |
| Composition domain | 5 methods | 1 method | -4 |

### akida-driver Path Fix

The optional `akida-driver` dependency in `barracuda/Cargo.toml` referenced
`../../../primals/toadstool/` but the actual directory is `toadStool/`
(camelCase). This broke Cargo manifest resolution on case-sensitive Linux
filesystems even when the `npu` feature was not enabled (Cargo resolves
all path dependencies at load time).

**Fix:** `toadstool` → `toadStool` in the path.

---

## 2. Files Changed

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | akida-driver path case fix |
| `barracuda/src/ipc/handlers/mod.rs` | Removed 4 universal composition handlers + `probe_capability` helper; CAPABILITIES 46→42 |
| `barracuda/src/ipc/dispatch.rs` | Removed 4 dispatch arms; test counts 41→37 |
| `barracuda/src/ipc/capability_domains.rs` | composition domain trimmed to science_health only |
| `barracuda/src/niche.rs` | CAPABILITIES count 46→42 |
| `barracuda/tests/ipc_roundtrip.rs` | Universal composition methods now assert "method not found" |
| `capability_registry.toml` | Removed 4 universal entries |

---

## 3. Validation

```
cargo clippy --workspace --all-targets -- -D warnings   → 0 warnings
cargo test --workspace                                   → 1,949 passed, 0 failed
cargo doc --workspace --no-deps                          → 0 warnings
```

---

## 4. Ecosystem Coordination

### For biomeOS

biomeOS v3.04 now owns all universal composition health routing. Springs
calling `composition.tower_health` etc. against a biomeOS Neural API server
will get correct responses from the lifecycle handler. Springs calling these
methods against each other will correctly get "method not found."

### For Other Springs

All springs should follow the same pattern:
- **Remove** universal `composition.*_health` handlers
- **Retain** spring-specific composition health (e.g., `composition.science_health`)
- biomeOS handles orchestration-level health aggregation

### For primalSpring

The composition ownership boundary is now clear:
- **biomeOS:** `composition.tower_health`, `composition.node_health`,
  `composition.nest_health`, `composition.nucleus_health`
- **Springs:** `composition.<domain>_health` (spring-specific)

---

## 5. Open Gaps (Unchanged from V143)

All 7 architectural gaps remain external team dependencies. See `GAPS.md`.

---

*Handoff authored during biomeOS v3.04 composition elevation sprint.*
*See also: `ecoPrimals/infra/wateringHole/handoffs/WETSPRING_V144_BIOMEOS_V304_COMPOSITION_ALIGNMENT_HANDOFF_APR12_2026.md`*
