# wetSpring → ToadStool/BarraCUDA Handoff V87 — blueFish WhitePaper + hotSpring Brain Architecture Review

**Date**: March 1, 2026
**From**: wetSpring (V87)
**To**: ToadStool/BarraCUDA team + Phase 2 primal teams (RootPulse, SweetGrass, LoamSpine)
**ToadStool pin**: S71+++ (`1dd7e338`)
**License**: AGPL-3.0-or-later
**Supersedes**: V86 (Cross-Spring Evolution)

---

## Executive Summary

- **blueFish whitePaper launched**: 7 documents at `whitePaper/blueFish/` establishing chemistry as an irreducible research programme within ecoPrimals — from PFAS analytical chemistry (57+ validated checks) through computational chemistry (DFT, MD, ML force fields) with full RootPulse provenance integration
- **hotSpring brain architecture reviewed**: 4-layer concurrent brain (NPU+GPU+CPU+GPU), Gen 2 36-head ESN with `HeadGroupDisagreement`, NautilusBrain evolutionary reservoir — all mapped to wetSpring bio workload patterns
- **Phase 2 primal use cases specified**: Concrete requirements for LoamSpine, SweetGrass, rhizoCrypt, BearDog, NestGate, Songbird driven by computational chemistry provenance needs
- **Isomorphism proof extended**: 29 computational chemistry operations decomposed into 3 tiers — 14 direct mappings (100% exist in BarraCUDA), 9 compositions, 6 genuinely new kernels

---

## Part 1: blueFish WhitePaper

### What Was Created

| Document | Content |
|----------|---------|
| `README.md` | Overview, hierarchy diagram, reading order |
| `00_INTRODUCTION.md` | Non-reducibility argument (Fodor, Lakatos, Anderson "More Is Different"), chemistry as special science |
| `01_ARCHITECTURE.md` | Two-arm structure (analytical + computational), cross-Spring interactions, phase2 primal demands |
| `02_ISOMORPHISM.md` | 29 comp chem operations mapped to BarraCUDA: Tier 1 (14 direct), Tier 2 (9 compose), Tier 3 (6 new) |
| `03_PROVENANCE.md` | 5 RootPulse use cases: DFT provenance, pipeline attribution, collaborative campaigns, dataset versioning, cross-group reproducibility |
| `04_CURRENT_STATE.md` | Validated analytical arm: 57+ checks, 5 binaries, 2 GPU modules, 6 experiments |
| `05_EVOLUTION.md` | 5-phase roadmap: PFAS → all analytical → DFT/MD core → ML force fields → full loop |
| `06_COMMUNITY.md` | Computational chemistry community pain points mapped to ecoPrimals solutions |

### Why This Matters for ToadStool

The isomorphism proof (02) shows that computational chemistry adds only 6 genuinely new kernel classes to BarraCUDA's ~40 existing primitives — a 15% expansion for an entirely new scientific domain. The new kernels:

1. **Gaussian basis function evaluation**: `exp(-alpha * r²) * Y_lm * r^l` — follows df64 transcendental patterns
2. **Electron repulsion integrals (ERIs)**: 4-center 2-electron integrals via Obara-Saika recurrence
3. **Schwarz screening**: Upper bound → skip negligible integrals (sparse dispatch pattern)
4. **Boys function**: Incomplete gamma → df64 transcendentals
5. **Becke partitioning**: Atomic grid weights for multicenter integration
6. **Resolution of identity (RI)**: 3-center → 2-center decomposition (Cholesky pattern)

ERIs are particularly interesting as a new WGSL shader class — the 4-center integral pattern has analogs in lattice QCD (4-point correlation functions).

---

## Part 2: hotSpring Brain Architecture Ingest

### What Was Reviewed

hotSpring v0.6.15 shipped:
- 4-layer concurrent brain: NPU (cerebellum) + Titan V (pre-motor) + CPU (prefrontal) + RTX 3090 (motor)
- `gpu_cg_solve_brain`: CG solver with `CgResidualUpdate` → NPU and `BrainInterrupt` ← NPU
- Gen 2 36-head ESN: 6 groups (Anderson, QCD, Potts, Steering, Monitor, Meta-mixer) with `HeadGroupDisagreement`
- NautilusBrain: Evolutionary reservoir computing via BingoCube boards (design complete, crate planned)
- Concept edge detection: LOO cross-validation identifies phase boundaries via head group disagreement

### wetSpring Bio Mapping

| hotSpring Pattern | wetSpring Bio Analog | Document |
|---|---|---|
| `CgResidualUpdate` | `DiversityUpdate { n_species, shannon_h, evenness }` | 05_EVOLUTION §4 |
| `BrainInterrupt::KillCg` | `BrainInterrupt::FlagNovelState` | 05_EVOLUTION §4 |
| `AttentionState` (Green/Yellow/Red) | Same — monitors diversity trajectory | 05_EVOLUTION §4 |
| `HeadGroupDisagreement` | Bio head groups: Anderson-informed, diversity-empirical, phylogeny-informed | 05_EVOLUTION §4 |
| `NautilusBrain` with `BetaObservation` | `NautilusBrain` with `ChemObservation` or `QsObservation` | 05_EVOLUTION §4 |
| `ExportedWeights` JSON persistence | Same format, cross-run learning for bio | 05_EVOLUTION §4 |

---

## Part 3: Phase 2 Primal Requirements (from 03_PROVENANCE.md)

### For RootPulse / LoamSpine Team

| Requirement | Value | Justification |
|---|---|---|
| Append rate | 100+ entries/sec | Active learning batch DFT appends |
| Entry size | 1-10 KB | DFT metadata + hashes |
| Total entries | 100K-1M | Large AL campaigns |
| Query by hash | O(log N) | Dataset assembly |
| Merkle proof | O(log N) | Cross-group reproducibility |

### For SweetGrass Team

| Requirement | Value |
|---|---|
| Chemistry Entity types | `Molecule`, `BasisSet`, `Functional`, `DftCampaign`, `TrainingPipeline`, `Validation` |
| Chemistry Braid relations | `DependsOn`, `ValidatedBy`, `ComputedWith`, `TrainedOn` |
| Multi-year tracking | PhD-timescale campaigns |

### For rhizoCrypt Team

| Requirement | Value |
|---|---|
| Multi-agent concurrent writes | Researchers running DFT simultaneously |
| Lock-free DAG append | No coordination overhead |
| Session dehydration | Campaign checkpointing |

---

## Part 4: Absorption Targets for ToadStool

### Priority 1: Brain Architecture Generalization

The `gpu_cg_solve_brain` pattern (GPU loop with NPU interrupt channel) should be generalized in BarraCUDA as a reusable pattern. wetSpring, neuralSpring, and groundSpring all have GPU-intensive loops that would benefit from NPU monitoring.

### Priority 2: ERI Shader Class (Future)

When blueFish's computational arm matures, the Obara-Saika ERI kernel will be a new WGSL shader class for ToadStool. Early design alignment is valuable.

### Priority 3: `esn_v2` Shape Bug

neuralSpring V67 handoff confirms: `ESN::train()` stores readout as `[reservoir_size, output_size]` but `set_readout_weights()` expects `[output_size, reservoir_size]`. wetSpring's `BioEsn` bridge works around this. Fix would unblock all Springs using the higher-level ESN API.

---

## Part 5: Quality State

| Gate | Status |
|------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy -- -W clippy::pedantic` | 0 warnings |
| `cargo test --workspace` | 1,247 PASS |
| `cargo doc --no-deps` | Clean |
| Unsafe code | 0 |
| Production mocks | 0 |
| Files > 1000 LOC | 0 |
| blueFish whitePaper | 7 documents, 1,768 lines |

---

## Reproduction

```bash
cd wetSpring
cargo test --workspace                    # 1,247 tests
cargo clippy --all-targets -- -W clippy::pedantic  # 0 warnings
ls whitePaper/blueFish/*.md               # 8 files (README + 00-06)
```

---

*Unidirectional handoff — no response expected. wetSpring continues autonomous evolution.*
