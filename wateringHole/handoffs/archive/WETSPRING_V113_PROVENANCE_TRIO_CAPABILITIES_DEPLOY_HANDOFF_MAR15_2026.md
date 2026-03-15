# wetSpring V113 — Provenance Trio + Expanded Capabilities + Deploy Graph

**Date:** 2026-03-15
**From:** wetSpring (life science validation)
**To:** biomeOS, BarraCUDA, ToadStool, rhizoCrypt, loamSpine, sweetGrass
**Status:** V113 — 19 IPC capabilities, provenance trio integration, biomeOS deploy graph

---

## Executive Summary

wetSpring V113 implements the provenance trio integration pattern from wateringHole,
expands the IPC surface from 9 to 19 biomeOS capabilities, adds cross-spring
time series exchange, and provides a complete biomeOS deploy graph. Domain
logic degrades gracefully when trio primals are unavailable.

## Key Changes

### 1. Provenance Trio Integration

- **Module**: `ipc/provenance.rs`
- **Pattern**: Follows `SPRING_PROVENANCE_TRIO_INTEGRATION_PATTERN` exactly
- **Capabilities**: `provenance.begin`, `provenance.record`, `provenance.complete`
- **Protocol**: `capability.call` over Neural API Unix socket — zero compile-time coupling
- **Degradation**: All three handlers return valid JSON when trio is unavailable
- **Socket discovery**: 4-tier cascade (explicit → BIOMEOS_SOCKET_DIR → XDG_RUNTIME_DIR → temp)

### 2. Expanded Science Capabilities

| New Capability | Wraps | Domain |
|----------------|-------|--------|
| `science.kinetics` | Gompertz, first-order models | Track 4 (soil/biogas) |
| `science.alignment` | `bio::alignment::smith_waterman` | Track 1b (genomics) |
| `science.taxonomy` | `bio::taxonomy::NaiveBayesClassifier` | Track 1 (16S) |
| `science.phylogenetics` | `bio::robinson_foulds::rf_distance` | Track 1b |
| `science.nmf` | Self-contained Lee & Seung MU | Track 3 (drug repurposing) |
| `science.timeseries` | Statistics on `ecoPrimals/time-series/v1` | Cross-spring |
| `science.timeseries_diversity` | Diversity on time series abundances | Cross-spring |

### 3. biomeOS Deploy Graph

- **File**: `graphs/wetspring_deploy.toml`
- **Standard**: `SPRING_AS_NICHE_DEPLOYMENT_STANDARD` compliant
- **Germination**: BearDog → Songbird → Provenance Trio → nestGate/petalTongue (optional) → wetSpring → Validation
- **Composition**: Graph can be extended by biomeOS to compose wetSpring with other springs

## For biomeOS

1. **Capability Registry**: wetSpring now advertises 19 capabilities. Update `config/capability_registry.toml`
   with the expanded set if wetSpring is deployed as a niche.
2. **Deploy Graph**: `graphs/wetspring_deploy.toml` is ready for integration into biomeOS graph library.
3. **Composition Evolution**: The graph declares dependencies on provenance trio — biomeOS can compose
   wetSpring with airSpring or healthSpring that consume `science.timeseries` payloads.

## For BarraCUDA / ToadStool

1. **NMF**: wetSpring has a self-contained NMF (Lee & Seung). When `SparseGemmF64` is available
   in BarraCUDA, the inner matmul loops can be replaced for GPU acceleration.
2. **Time Series**: Cross-spring time series exchange is schema-validated. Consider if BarraCUDA
   should provide time series statistics primitives.

## For Provenance Trio (rhizoCrypt, loamSpine, sweetGrass)

1. **First Spring Integration**: wetSpring is the first spring to fully implement the trio pattern.
2. **API Observations**: The three-phase flow (dehydrate → commit → create_braid) works well.
   Consider adding a `complete_experiment` convenience operation that chains all three.
3. **Testing**: All 9 provenance tests verify graceful degradation when trio is offline.

## Quality Gates

| Gate | Status |
|------|--------|
| `cargo check --features ipc,json` | PASS |
| `cargo clippy` (pedantic + nursery) | ZERO new warnings |
| `cargo test` (1,326 tests) | 0 failures |
| Zero unsafe code | PASS |
| Zero unwrap outside tests | PASS |
