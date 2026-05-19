# wetSpring → lithoSpore: Sealed Barrick 2009 USB Artifacts

**Date:** May 19, 2026
**From:** wetSpring (southGate)
**To:** lithoSpore team
**Priority:** HIGH — first complete E2E ecosystem proof
**License:** AGPL-3.0-or-later

---

## Status: SEALED

Barrick 2009 sovereign resequencing pipeline — 7/7 clones complete,
13/13 validation checks PASS, braids sealed, parity documented.

---

## Artifacts for USB

### 1. Sealed Braids

| File | Status | Contents |
|------|--------|----------|
| `provenance/braids/barrick_2009_systems_study_v2.json` | **SEALED** | Full 7-clone systems study. 486 sovereign variants, 569 breseq, parity evolution v0→v1→v2. |
| `provenance/braids/barrick_2009_refined_v1.json` | **PARTIAL** | Single-clone REL1164M calibration anchor. 34K→78 variants (99.8% FP reduction). |
| `provenance/braids/barrick_2009_sovereign.json` | **AUTO** | Pipeline auto-generated aggregate. BLAKE3: `d19be4bf...`. |
| `provenance/braids/barrick_2009_mutations.json` | **BASELINE** | breseq mutation counts from Exp381. |

### 2. Parity Document

| File | Contents |
|------|----------|
| `experiments/results/barrick_2009_parity.json` | Formal L1 (breseq 0.40.1) vs L2 (sovereign Rust) comparison. Per-clone variant counts, position matches, tolerances, assessment. |

### 3. Pipeline Source

| Component | Path | Description |
|-----------|------|-------------|
| Sovereign pipeline | `barracuda/src/bin/validate_sovereign_resequencing.rs` | Full GPU-adaptive pipeline (630 lines) |
| Variant caller | `barracuda/src/bio/variant_caller.rs` | Quality-weighted, strand-bias, Q20 filter |
| Pileup | `barracuda/src/bio/pileup.rs` | Base quality filtering |
| Socket discovery | `barracuda/src/ipc/discover.rs` | Wave 22 connect-probe + DEAD_SOCKET_CACHE |
| Parity types | `barracuda/src/validation/parity.rs` | `ParityResult`, `CloneParity`, `ParitySummary` |

### 4. Results Summary

| Clone | Gen | Sovereign | breseq | Ratio | Time |
|-------|-----|-----------|--------|-------|------|
| REL1164M | 2K | 78 | 4 | 19.5x | 1574s |
| REL2179M | 5K | 78 | 11 | 7.1x | 1577s |
| REL4536M | 10K | 60 | 11 | 5.5x | 1632s |
| REL7177M | 15K | 58 | — | — | 1610s |
| REL8593M | 20K | 68 | 32 | 2.1x | 1092s |
| REL10379 | 30K | 63 | 253 | 0.25x | 1475s |
| REL10926 | 40K | 81 | 258 | 0.31x | 1672s |
| **Total** | | **486** | **569** | **0.85x** | **10,662s** |

### 5. Known Gaps (for USB README)

- Zero position-level overlap between L1 and L2 — coordinate representation
  mismatch (breseq genome diff vs. sovereign reference-relative positions)
- Parity crossover: over-call at early generations, under-call at late
- L3 (primal-composed via live trio) not yet validated — requires live
  rhizoCrypt+loamSpine+sweetGrass deployment
- Tenaillon 2016 (264 clones, 590 GB) queued as next dataset

---

## USB Packaging Instructions

```bash
# Mount USB at /mnt/usb
USB=/mnt/usb/wetSpring

mkdir -p "$USB"/{provenance/braids,experiments/results,src}

# Braids
cp provenance/braids/barrick_2009_*.json "$USB/provenance/braids/"

# Parity
cp experiments/results/barrick_2009_parity.json "$USB/experiments/results/"

# Source snapshot
git archive HEAD | tar -x -C "$USB/src/"
```

---

## References

- `barrick_2009_systems_study_v2.json` — sealed braid with full results
- `barrick_2009_parity.json` — L1 vs L2 parity contract
- `GAPS.md` — Gap #9 (parity), Gap #11 (variant caller)
- Wave 23 audit: `WAVE23_WETSPRING_E2E_COMPLETION_MAY18_2026.md`
