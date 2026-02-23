# Exp112: Real-Bloom GPU Surveillance at Scale

**Date**: February 23, 2026
**Status**: PASS — 23/23 checks
**Binary**: `validate_real_bloom_gpu` (requires `gpu` feature)
**Faculty**: Cahill, Smallwood (Sandia)

## Purpose

Validates bloom detection pipeline on GPU using realistic multi-ecosystem
community data at 500+ timepoints (Lake Erie HAB, Baltic cyanobacterial,
Florida red tide). Demonstrates GPU-accelerated diversity computation at
real surveillance workloads.

## Data

Synthetic: three ecosystems with biologically plausible bloom signatures.
- Lake Erie: 520 timepoints, 200 species, bloom at t=180-220
- Baltic: 480 timepoints, 150 species, bloom at t=200-250
- Florida: 365 timepoints, 100 species, bloom at t=120-160

Mirrors NCBI: PRJNA649075 (Lake Erie), PRJNA524461 (Baltic), PRJNA552483 (Florida).

## Results

| Ecosystem | Bloom Events | H Drop Ratio | Max Dominance | Max BC Shift |
|-----------|:---:|:---:|:---:|:---:|
| Lake Erie | 40 timepoints | 0.480 | 0.762 | 0.833 |
| Baltic | 50 timepoints | 0.403 | 0.804 | 0.854 |
| Florida | 40 timepoints | 0.322 | 0.864 | 0.862 |

- GPU-CPU parity: max |Shannon| = 0.0, max |BC| = 0.0 (exact match)
- CPU total: 3.5 ms (1365 timepoints × 3 ecosystems)
- All three ecosystems: bloom detected, dominance > 0.5, recovery confirmed
- Scale: N=500 timepoints → 30.5 ms CPU (O(N²) for Bray-Curtis)

## Key Findings

1. Bloom detection signatures generalize across all three ecosystem types:
   Shannon crash > 50%, dominance spike > 0.5, BC shift > 0.8.
2. Florida red tide shows the strongest bloom signature (H drop to 32% of
   pre-bloom), consistent with Karenia brevis monoculture dynamics.
3. Recovery detection works in all ecosystems — post-bloom Shannon recovers
   to pre-bloom levels.
4. GPU diversity produces exact CPU parity for Shannon and Bray-Curtis,
   confirming that surveillance decisions are hardware-agnostic.

## Reproduction

```bash
cargo run --features gpu --release --bin validate_real_bloom_gpu
```
