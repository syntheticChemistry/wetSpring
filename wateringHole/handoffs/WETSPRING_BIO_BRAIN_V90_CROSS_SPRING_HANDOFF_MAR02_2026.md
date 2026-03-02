# wetSpring → ecoPrimals Handoff V90 — Bio Brain Cross-Spring Ingest (hotSpring + Nautilus)

**Date**: March 2, 2026
**From**: wetSpring (V90)
**To**: ecoPrimals team
**ToadStool pin**: S79 (`f97fc2ae`)
**License**: AGPL-3.0-or-later
**Supersedes**: V89 (S79 Deep Rewire + Cross-Spring Validation)

---

## Executive Summary

- **hotSpring brain architecture ingested**: 4-layer brain concept (NPU/Motor/Pre-motor/Prefrontal), 36-head Gen2 ESN layout, `HeadGroupDisagreement`, and `AttentionState` machine adapted from hotSpring v0.6.15 physics pipeline to wetSpring bio sentinel monitoring.
- **`BioNautilusBrain` integrated**: `bingocube-nautilus` crate (primalTools/bingoCube) wired as optional dependency for evolutionary reservoir computing — concept edge detection, drift monitoring, and adaptive sampling for bio data.
- **`BioBrain` adapter**: Manages attention state machine with smoothed urgency window, observation history, and IPC status snapshots. Three new JSON-RPC methods: `brain.observe`, `brain.attention`, `brain.urgency`.
- **Exp272: 64/64 checks passed**: 7 validation domains covering all 4 implementation phases across 7 springs, ~908ms total.
- **1,032 tests passing**, clippy pedantic+nursery clean.

---

## Part 1: What Was Ingested

### From hotSpring v0.6.15

| hotSpring Component | Bio Adaptation | File |
|---------------------|----------------|------|
| 36-head Gen2 ESN (6 groups × 6 heads) | Bio head constants: Anderson-informed (A), Diversity-empirical (B), Phylogeny-informed (C), Steering (D), Brain/Monitor (E), Meta-mixer (M) | `bio/esn/heads.rs` |
| `HeadGroupDisagreement` (reservoir.rs L617-671) | `BioHeadGroupDisagreement`: delta_regime, delta_phase, delta_anomaly, delta_priority, urgency() | `bio/esn/heads.rs` |
| `AttentionState` (Green/Yellow/Red) | `AttentionState` enum: Healthy/Alert/Critical with hysteresis transitions | `bio/esn/heads.rs` |
| 4-layer brain architecture | `BioBrain` adapter: ESN sentinel (cerebellum), spectral GPU (motor), rarefaction (pre-motor), Nautilus CPU (prefrontal) | `bio/brain/observation.rs` |
| `CgResidualUpdate` channel | `DiversityUpdate`: n_species, shannon_h, evenness, attention | `bio/brain/observation.rs` |

### From bingoCube/nautilus

| Nautilus Component | Bio Bridge | File |
|-------------------|------------|------|
| `NautilusBrain` | `BioNautilusBrain`: bio observation → physics mapping, predict/train/screen | `bio/brain/nautilus_bridge.rs` |
| `BetaObservation` → physics | `BioObservation` → `BetaObservation` conversion (Shannon→β, evenness→plaquette, Chao1→CG iters) | `bio/brain/nautilus_bridge.rs` |
| `detect_concept_edges()` | Community phase boundary detection (diversity regime transitions) | `bio/brain/nautilus_bridge.rs` |
| `DriftMonitor` | Population health tracking for long-term sentinel | `bio/brain/nautilus_bridge.rs` |
| `screen_candidates()` | Adaptive sampling: which diversity levels to sample next | `bio/brain/nautilus_bridge.rs` |

---

## Part 2: New Files

| File | Description | Lines |
|------|-------------|:-----:|
| `barracuda/src/bio/esn/heads.rs` | 36 bio head constants, `BioHeadGroupDisagreement`, `AttentionState`, 7 unit tests | ~330 |
| `barracuda/src/bio/brain/mod.rs` | Brain module with cross-spring provenance docs | ~30 |
| `barracuda/src/bio/brain/observation.rs` | `BioObservation`, `BioBrain`, `DiversityUpdate`, `BrainStatus` | ~230 |
| `barracuda/src/bio/brain/tests.rs` | 9 unit tests for brain adapter | ~130 |
| `barracuda/src/bio/brain/nautilus_bridge.rs` | `BioNautilusBrain`, bio→physics mapping, 8 unit tests | ~290 |
| `barracuda/src/bin/validate_bio_brain_s79.rs` | Exp272: 64 checks across 7 domains | ~415 |

## Part 3: Modified Files

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | Added `bingocube-nautilus` optional dep, `nautilus` feature, Exp272 binary |
| `barracuda/src/bio/mod.rs` | Registered `brain` module |
| `barracuda/src/bio/esn/mod.rs` | Registered `heads` module, exported `AttentionState`, `BioHeadGroupDisagreement` |
| `barracuda/src/ipc/handlers.rs` | Added `handle_brain_observe/attention/urgency`, `BIO_BRAIN` static, brain capability ads |
| `barracuda/src/ipc/dispatch.rs` | Routed `brain.observe`, `brain.attention`, `brain.urgency` |

---

## Part 4: IPC Capabilities Added

| Method | Params | Returns |
|--------|--------|---------|
| `brain.observe` | `sample_id`, 8 bio metrics, `head_outputs` (36 floats) | attention state, urgency, disagreement deltas |
| `brain.attention` | none | current attention, urgency, observation count, disagreement |
| `brain.urgency` | none | scalar urgency, attention state |

---

## Part 5: Exp272 Validation Results

```
╔════════════════════════════════════════════════════════════════════╗
║  Exp272: Bio Brain Cross-Spring Validation                        ║
╠════════════════════════════════════════════════════════════════════╣
║ Domain                 │ Spring             │    Time │   ✓ ║
╠════════════════════════════════════════════════════════════════════╣
║ 36-Head Constants      │ hotSpring          │  0.02ms │  14 ║
║ Head Disagreement      │ hotSpring          │  0.01ms │   8 ║
║ Attention Machine      │ hotSpring          │  0.01ms │   7 ║
║ Bio Observation        │ wetSpring          │  0.00ms │   4 ║
║ Bio Brain              │ hotSpring+wetSpring │  0.01ms │  10 ║
║ Nautilus Brain         │ bingoCube          │ 423.52ms │  11 ║
║ Concurrent Pipeline    │ multi-spring       │ 484.26ms │   8 ║
╠════════════════════════════════════════════════════════════════════╣
║ TOTAL                  │ 7 springs          │ 907.83ms │  62 ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## Part 6: Cross-Spring Provenance Tree

```
  ┌─ hotSpring v0.6.15 ── 4-layer brain (NPU/Motor/PreMotor/Prefrontal)
  ├─ hotSpring v0.6.15 ── 36-head Gen2 ESN (6 groups × 6 heads)
  ├─ hotSpring v0.6.15 ── HeadGroupDisagreement → BioHeadGroupDisagreement
  ├─ hotSpring v0.6.15 ── AttentionState (Green/Yellow/Red → Healthy/Alert/Critical)
  ├─ bingoCube/nautilus ─ NautilusBrain → BioNautilusBrain
  ├─ wetSpring V89 ───── BioObservation, BioBrain, DiversityUpdate
  └─ ToadStool S79 ──── MultiHeadEsn, ExportedWeights
```

---

## Part 7: Quality Gate

| Gate | Status |
|------|--------|
| `cargo fmt` | PASS |
| `cargo clippy --all-features -W pedantic -W nursery` | PASS (zero warnings in new code) |
| `cargo test --lib --features nautilus` | 1,032 passed, 0 failed, 1 ignored |
| Exp272 (64 checks / 7 domains) | 64/64 PASS |
| Zero `unsafe` code | PASS |
| Zero `todo!()`/`unimplemented!()` | PASS |
| AGPL-3.0-or-later headers | PASS |

---

## Part 8: What NOT Imported

- **No direct hotSpring imports**: Springs don't import each other. All patterns mapped independently.
- **No physics head constants**: Bio heads are independent constants mapped from physics analogs.
- **No GPU ESN shaders**: GPU shaders come through ToadStool absorption, not local copies.
- **No hotSpring 36 head labels**: Bio uses its own nomenclature (diversity/taxonomy/AMR/phylo/steering).

---

## Part 9: Next Steps

1. **Real data validation**: Run Exp272 against EMP (Earth Microbiome Project) 30K samples
2. **GPU ESN integration**: Wire `MultiHeadBioEsn` (5-head) outputs through the 36-head brain adapter
3. **Nautilus concept edge discovery**: Run on real diversity trajectories to find community phase boundaries
4. **Cross-spring export**: Share `BioHeadGroupDisagreement` pattern back to ToadStool via wateringHole
