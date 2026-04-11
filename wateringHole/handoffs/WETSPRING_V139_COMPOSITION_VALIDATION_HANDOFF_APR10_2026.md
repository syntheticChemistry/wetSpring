# WETSPRING V139 — Composition Validation Handoff

| Field | Value |
|-------|-------|
| Date | 2026-04-10 |
| From | wetSpring V139 |
| To | primalSpring, biomeOS, wateringHole |
| License | AGPL-3.0-or-later |
| Supersedes | WETSPRING_V138_PRIMAL_COMPOSITION_PATTERNS_HANDOFF_APR07_2026.md |

## Executive Summary

wetSpring evolves from Rust-validates-Python to **NUCLEUS composition
validates Rust+Python**. This handoff documents the composition validation
tier: IPC surface alignment, deploy graph structural validation, and
proto-nucleate coverage verification.

### Evolution Path

```
Python baseline → Rust validation (1,580 tests, 355 binaries)
  → Primal composition validation (proto-nucleate alignment)
    → ecoBin harvest to plasmidBin
```

## Canonical Metrics

| Metric | Value |
|--------|-------|
| Library tests | 1,580 |
| Validation binaries | 356 (334 barracuda + 22 forge) |
| IPC methods routed | 37 |
| Capabilities advertised | 45 |
| Composition health handlers | 5 (science, tower, node, nest, nucleus) |
| Deploy graphs | 7 |
| Proto-nucleate primals covered | 11/11 |
| Named tolerances | 38+ |
| LOC (Rust) | ~214,690 |
| unsafe blocks | 0 |
| #[allow()] in production | 0 |

## Changes in V139

### Critical Fix: akida-driver Path Resolution (C1)

The `akida-driver` path dependency referenced `toadStool` (camelCase) but
the filesystem directory is `toadstool` (lowercase). Fixed case to match
actual directory. This unblocked `cargo fmt`, `cargo clippy`, and
`cargo test` for the full workspace.

### Composition Validation Binary (Exp400)

New: `validate_composition_nucleus_v1` — validates the full NUCLEUS
composition pattern against the proto-nucleate graph:

- **D01**: Health triad (liveness, readiness, check)
- **D02**: Capability discovery surface completeness (45 capabilities)
- **D03**: Composition health JSON shape validation (all 5 handlers)
- **D04**: Deploy graph structural validation (5 graphs)
- **D05**: Proto-nucleate primal coverage (11/11 primals)
- **D06**: Niche self-knowledge consistency
- **D07**: Bonding metadata and atomic alignment

### JSON Shape Fix: composition.nucleus_health

Fixed shape mismatch in `validate_nucleus_live_gonzales.rs` — the handler
returns `tiers.tower` (boolean) but the validator was reading
`nucleus["tower"]["healthy"]` (nested object). Aligned to actual handler
shape: `tiers.{tower,node,nest,provenance_trio}` and top-level `healthy`.

### Capability Registry Reconciliation (H3)

Three sources of truth synchronized:
- `ipc/dispatch.rs` routes 37 methods
- `handlers/mod.rs` CAPABILITIES lists all 37
- `niche.rs` CAPABILITIES expanded from 24 → 45 (added health, gonzales,
  data.fetch, vault, composition, anderson sub-methods)
- `capability_registry.toml` expanded with 20 new entries
- Ecology semantic mappings updated with 7 new Gonzales/Anderson entries

### Dispatch Proptest Surface (Correctness)

The proptest `known` method list synchronized with the full dispatch
surface (added 11 missing methods: data.fetch.*, vault.*, composition.*).
Prevents false negatives in fuzz testing.

### IPC Round-Trip Tests (Composition)

Five new integration tests in `ipc_roundtrip.rs`:
- `composition_science_health_roundtrip`
- `composition_nucleus_health_roundtrip`
- `composition_tower_health_roundtrip`
- `vault_store_roundtrip`
- `data_fetch_roundtrip`

### Squirrel Added to Niche Dependencies

Added Squirrel (AI inference primal) to `niche::DEPENDENCIES` as
`required: false`. The proto-nucleate graph lists Squirrel for AI-driven
sample triage; wetSpring exposes `ai.ecology_interpret` but had not
declared the Squirrel dependency. Now 9 niche dependencies (5 required +
4 optional) covering all 11 proto-nucleate primals.

### Code Quality Fixes

- 3x `#[allow(dead_code)]` → `#[expect(dead_code, reason = "...")]`
- 2x `.unwrap()` → `?` in facade binary (proper error propagation)
- 5x `/tmp` fallback → `std::env::temp_dir()` (ecoBin platform-agnostic)
- Anderson handler documented as GPU-required (not GPU-preferred)
- Doc example `todo!()` → hidden helper functions
- barraCuda v0.3.7 pinned in reproduction manifest
- Validation binary clippy expect attributes aligned

### plasmidBin Metadata Updated

`infra/plasmidBin/wetspring/metadata.toml` updated:
- Version 0.7.0 → 0.8.0
- Full capability list (33 capabilities)
- Composition model declared (nucleated, 4 fragments)
- Proto-nucleate reference
- barraCuda version pinned in provenance

## Composition Validation Matrix

| Pattern | Binary | Status |
|---------|--------|--------|
| Health triad | `validate_composition_nucleus_v1` D01 | PASS |
| Capability surface | `validate_composition_nucleus_v1` D02 | PASS |
| Composition health shapes | `validate_composition_nucleus_v1` D03 | PASS |
| Deploy graph structure | `validate_composition_nucleus_v1` D04 | PASS |
| Proto-nucleate coverage | `validate_composition_nucleus_v1` D05 | PASS |
| Niche self-knowledge | `validate_composition_nucleus_v1` D06 | PASS |
| Bonding/atomic alignment | `validate_composition_nucleus_v1` D07 | PASS |
| IPC science roundtrip | `ipc_roundtrip.rs` | PASS |
| IPC composition roundtrip | `ipc_roundtrip.rs` (5 new) | PASS |
| NUCLEUS live gonzales | `validate_nucleus_live_gonzales` | FIXED |
| Tower/Node orchestration | `validate_nucleus_tower_node` | EXISTING |
| Cross-primal pipeline | `validate_cross_primal_pipeline_v98` | EXISTING |
| biomeOS NUCLEUS | `validate_biomeos_nucleus_v98` | EXISTING |

## ecoBin Harvest Path

```bash
cd /home/westgate/Development/ecoPrimals/springs/wetSpring
cargo build --release --features ipc --bin wetspring
cp target/release/wetspring ../../infra/plasmidBin/wetspring/wetspring
cd ../../infra/plasmidBin
./harvest.sh wetspring --tag v0.8.0
```

## Upstream barraCuda Feature Gate Issue

When `ipc` or `facade` features are activated directly on
`wetspring-barracuda`, the upstream `barracuda` crate (v0.3.11) fails to
compile: `tolerances/precision.rs:138` references `crate::device` without
`#[cfg(feature = "gpu")]` gating. This blocks:
- `cargo test --features ipc` (IPC integration tests)
- `cargo check --features facade` (facade binary)

**Workaround:** `cargo test --workspace --features json` activates json
(subset of ipc) via workspace resolver-2, avoiding the barraCuda bug.
All 1,580 tests + 46 dispatch tests (including proptests) pass via this path.

**Resolution:** upstream barraCuda fix needed: gate
`tolerances/precision.rs:138` behind `#[cfg(feature = "gpu")]`.

## Gaps Handed Back

### To primalSpring

1. **coralReef direct IPC** — wetSpring reaches coralReef only via
   barraCuda shader compilation. No direct `shader.compile.wgsl` IPC from
   wetSpring. Proto-nucleate lists coralReef; composition health doesn't
   probe it. Propose: barraCuda expose shader compilation status via
   `compute.shader.status` that wetSpring can forward.

2. **Deploy graph schema unification** — `wetspring_deploy.toml` uses
   `[[nodes]]` while others use `[[graph.node]]`. The `graph_validate.rs`
   module validates `[[graph.node]]` only. Propose: primalSpring document
   canonical schema, retire `[[nodes]]` variant.

3. **Proto-nucleate validation node** — the proto-nucleate declares
   `validate_wetspring_lifescience` with binary `primalspring_primal` and
   capability `coordination.validate_composition`. This binary is external
   (lives in primalSpring, not wetSpring). Confirm primalSpring implements
   this coordination validator.

### To biomeOS

4. **Neural API socket standardization** — composition health probes
   require `FAMILY_ID` + `XDG_RUNTIME_DIR` to discover Neural API. When
   neither is set, all composition tiers report "unreachable". Propose:
   biomeOS provide a `BIOMEOS_NEURAL_API_SOCKET` env var as direct override.

### Known Ecosystem Gaps (unchanged from V138)

See `GAPS.md` for 7 documented architectural gaps (ionic negotiation,
RootPulse exchange, public chain anchor, petalTongue WASM, ludoSpring
composition, hotSpring physics, radiating attribution).

---

*This handoff documents the evolution from Rust validation to primal
composition validation. Python was the validation target for Rust; now
Rust + Python are the validation targets for NUCLEUS patterns.*
