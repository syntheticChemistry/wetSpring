# Exp156: Dictyostelium cAMP Relay — Non-Hermitian Anderson

| Field          | Value |
|----------------|-------|
| **Status**     | PASS (8/8 checks) |
| **Binary**     | `validate_dictyostelium_relay` |
| **Date**       | 2026-02-24 |
| **Phase**      | 39 — Paper Queue Extension |
| **Paper**      | 38 (Frontiers Cell Dev Biol 2023) |

## Core Idea

Analyze Dictyostelium cAMP relay as NP Solution #3 — active signal amplification
that defeats Anderson localization via non-Hermitian extension (gain term).

## Key Findings

- Relay gain: 10× per cell (exponential compounding defeats localization)
- cAMP wave speed: 0.2 mm/min (matches published value)
- After 50 hops: effective gain = 10⁵⁰ — localization impossible
- Non-Hermitian Anderson: gain G >> G_c → all states extended (lasing transition)
- Relay range = 100 cells vs passive localization length = 5 cells (20× amplification)
- HIGH evolutionary cost: each cell needs complete relay circuit (ACA, cAR, PDE, PKA)

## Three NP Solutions Compared

| Solution | Strategy | Network Architecture | Cost |
|----------|----------|---------------------|------|
| V. cholerae | Logic inversion | Inverse detection | Low |
| Myxococcus | Geometry bootstrap | Self-built medium | High |
| Dictyostelium | Signal relay | Active repeaters | Very high |

## Connection to Sub-thesis 02

Maps the three NP solutions to network architecture paradigms.
Each defeats Anderson localization through a fundamentally different mechanism.
