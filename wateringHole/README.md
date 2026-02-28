# wetSpring wateringHole

**Date:** February 28, 2026
**Purpose:** Spring-local handoff documents to ToadStool/BarraCuda and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V75** | `../wateringHole/handoffs/WETSPRING_TOADSTOOL_V75_COMPUTE_DISPATCH_REWIRE_FEB28_2026.md` | Feb 28 | **current** — 82 primitives consumed (+`ComputeDispatch`, +`BatchedMultinomialGpu`, +`PairwiseL2Gpu`), 6 GPU modules refactored from manual BGL to ComputeDispatch builder, `rarefaction_gpu` evolved to dedicated multinomial GPU, `pairwise_l2_gpu` + `fst_variance` adopted |
| V73 | `handoffs/WETSPRING_TOADSTOOL_V73_DEBT_REDUCTION_FIVE_TIER_HANDOFF_FEB28_2026.md` | Feb 28 | V73 deep debt + V72 five-tier chain (Exp224-228), 229 experiments 5,743+ checks ALL PASS |

## NestGate Handoffs

| File | Date | Scope |
|------|------|-------|
| `handoffs/WETSPRING_NESTGATE_DATA_TYPES_EVOLUTION_HANDOFF_FEB26_2026.md` | Feb 26 | Data type profiling, P0-P3 gap analysis, evolution roadmap for biology-aware storage |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | 700+ WGSL shader provenance map (cross-spring, ToadStool S68 universal precision) |

## Archive

Superseded handoffs in `handoffs/archive/` (V7-V72, API report, rewire, cross-spring provenance — 56 files).
V34-V39 archived (superseded by V40). V40-V43 archived (superseded by V47). V44-V45 archived (superseded by V48). V47-V50 archived (superseded by V52). V52 archived (superseded by V53+V54). V53-V55 archived (superseded by V56-V59). V56-V58 archived (superseded by V59+V60). V59-V66 archived (superseded by V73). V61+V70 archived at ecosystem wateringHole (superseded by V75).
Cross-spring docs moved to archive: `CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md`, `CROSS_SPRING_PROVENANCE_FEB22_2026.md`.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_V{NNN}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs are unidirectional: wetSpring → ToadStool.
ToadStool absorbs what it finds useful; wetSpring leans on upstream.
