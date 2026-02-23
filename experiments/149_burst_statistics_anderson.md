# Exp149: QS Burst Statistics as Anderson Localization

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (6/6 checks) |
| **Binary**     | `validate_burst_statistics_anderson` |
| **Date**       | 2026-02-23 |
| **Phase**      | 38 — Extension Papers |

## Core Idea

Reinterpret Jemielita et al. (SciRep 2019) findings as Anderson localization.
Their spatial colony-growth heterogeneity observations map exactly onto
Anderson regimes. Low-cost reinterpretation paper — no new experiments needed.

## Translation Table

| Their Term | Anderson Term |
|-----------|--------------|
| "clustered cells" | high local density → disorder |
| "homogeneous distribution" | uniform lattice → low W |
| "localized QS burst" | LOCALIZED STATE |
| "synchronized QS burst" | EXTENDED STATE |
| "spatial heterogeneity" | Anderson disorder W |

## 4 Observations Reinterpreted

| Config | Burst Timing | Sync | Anderson Regime |
|--------|:------------:|:----:|:---------------:|
| Clustered | EARLY | LOW | Localized (high W) |
| Homogeneous | DELAYED | HIGH | Extended (low W) |
| Random | INTERMEDIATE | MID | Near-critical |
| Very sparse | NONE | NONE | Dilution-suppressed |

## Novel Predictions Beyond Their Data

1. Diversity experiment: 5-species community → always patchy QS in 2D
2. 3D extension: synchronization RECOVERS in 3D biofilm reactor
3. Colony size scaling: P(sync) ~ exp(-L/ξ) — exponential decay
4. Temporal transition: sharp QS onset at 75% occupancy (from Exp137)

## Paper Strategy

Reinterpretation paper (short communication / PRL letter):
- Cost: no new experiments, reanalyze published data
- Impact: first connection of QS burst statistics to Anderson theory
- Novel analysis: compute ⟨r⟩ level spacing ratio from real cell coordinates
  (FIRST time this has been done with bacterial colony spatial data)
