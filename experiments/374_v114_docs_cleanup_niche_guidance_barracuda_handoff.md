# Experiment 374: V114 — Documentation Cleanup + Niche Setup Guidance + BarraCUDA Absorption Handoff

**Date:** 2026-03-15
**Status:** PASS
**Version:** V114
**Predecessor:** Exp373 (V113 provenance trio + capabilities + deploy graph)

---

## Hypothesis

Cleaning stale documentation, archiving superseded handoffs, creating niche
setup guidance for springs, and crafting a comprehensive BarraCUDA/ToadStool
absorption handoff will:

1. Reduce documentation debt and eliminate stale version references
2. Provide a reusable niche setup template other springs can follow
3. Accelerate BarraCUDA GPU primitive evolution by documenting wetSpring needs
4. Create a clean fossil record with archived handoffs preserved in archive/

## Method

### Phase 1: Documentation Refresh

- Updated README.md, CHANGELOG.md, whitePaper/README.md to V114
- Updated whitePaper/STUDY.md with current experiment/binary/check counts
- Updated whitePaper/baseCamp/README.md status line

### Phase 2: Archive Cleanup

Archived 6 superseded handoffs to `wateringHole/handoffs/archive/`:
- V111: BARRACUDA_TOADSTOOL_ABSORPTION, DEEP_DEBT_EVOLUTION
- V112: STREAMING_PEDANTIC_CAPABILITY, NVIDIA_HARDWARE_LEARNING
- V113: PAPER_EXTENSION_ROADMAP, BARRACUDA_TOADSTOOL_EVOLUTION (Mar 11)

Active handoffs retained:
- V113 Provenance Trio (Mar 15) — current integration handoff
- V114 Deep Audit (Mar 12) — ongoing audit
- V114 Niche + Absorption (new)

### Phase 3: BarraCUDA Absorption Handoff

Reviewed all 19 wetSpring IPC capabilities and mapped each to BarraCUDA
primitive opportunities:

| Capability | BarraCUDA Primitive Needed | Precision | Priority |
|------------|---------------------------|-----------|----------|
| `science.nmf` | `SparseGemmF64`, `MatMulF64` | F64 | High |
| `science.alignment` | `SmithWatermanGpu` | Integer | Medium |
| `science.kinetics` | `OdeSolverGpu` (Gompertz, first-order) | F64 | Medium |
| `science.phylogenetics` | `TreeTraversalGpu` (Robinson-Foulds) | Integer | Low |
| `science.taxonomy` | `KmerHashGpu`, `NaiveBayesGpu` | Mixed | Medium |
| `science.diversity` | Already uses `barracuda` reduce ops | F64 | Done |
| `science.peak_detect` | `SignalConvGpu` | F64 | Low |
| `science.ode` | `AdaptiveOdeGpu` (RK45) | F64 | High |

### Phase 4: Niche Setup Guidance

Created wateringHole niche setup handoff documenting the wetSpring niche
pattern as a reference implementation:
- UniBin structure, socket binding, capability registration
- Deploy graph authoring, provenance trio wiring
- Neural API workflow graphs for automated experiment pipelines

## Results

- 6 handoffs archived, 0 stale references in active docs
- STUDY.md updated from "281 experiments" to "374 experiments"
- baseCamp status refreshed from V112 to V114
- BarraCUDA absorption handoff identifies 8 GPU primitive opportunities
- Niche setup guidance reusable by any spring

## Quality

- `cargo check --features ipc,json` — clean
- `cargo clippy --features ipc,json -- -W clippy::pedantic -W clippy::nursery` — zero warnings
- `cargo test` — 1,326 tests pass, 0 fail
- No code changes — documentation-only experiment
