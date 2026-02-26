# Exp191: GPU V59 Science Parity — Diversity + Anderson

**Date:** February 26, 2026
**Phase:** V59 — Three-tier controls
**Binary:** `validate_gpu_v59_science`
**Command:** `cargo run --features gpu --release --bin validate_gpu_v59_science`
**Status:** DONE (GPU, 29 checks PASS)

## Purpose

Proves that GPU-dispatched Anderson spectral analysis produces physically
correct results and integrates with the diversity→Anderson pipeline.
Exercises `barracuda::spectral::*` primitives across multiple lattice
sizes, disorder values, and synthetic cold seep communities.

## Domains Validated

| Domain | Checks | barracuda Primitives |
|--------|:------:|---------------------|
| G01: Anderson 3D spectral analysis | 18 | `spectral::{anderson_3d, lanczos, lanczos_eigenvalues, level_spacing_ratio}` |
| G02: Diversity → Anderson pipeline | 6 | `bio::diversity::*` + `spectral::*` |
| G03: W_c determination | 2 | `spectral::anderson_3d` + level spacing |
| G04: Cold seep spectral classification | 3 | Full pipeline: synthetic community → diversity → Anderson → classification |

## Acceptance Criteria

- All 29 checks PASS with --features gpu
- Extended regime (r > midpoint) for low-disorder communities
- Localized regime (r < midpoint) for high-disorder communities
- W_c crossing detected in plausible range [10, 25]
