# Exp328: CPU vs GPU Viz Math Parity

**Phase:** V100 (petalTongue integration + GPU parity)
**Date:** 2026-03-09
**Binary:** `validate_cpu_gpu_viz_math`
**Features:** `gpu`, `json`

## Status: PASS (27/27)

## Scope

Validates that barraCuda CPU math matches barraCuda GPU math for all
scientific domains exercised by the petalTongue visualization scenarios.

## Domains

| Domain | Checks | Tolerance | CPU Function | GPU Function |
|--------|--------|-----------|-------------|-------------|
| G1 Shannon | 1 | GPU_VS_CPU_F64 (1e-6) | `diversity::shannon` | `diversity_gpu::shannon_gpu` |
| G2 Simpson | 1 | GPU_VS_CPU_F64 | `diversity::simpson` | `diversity_gpu::simpson_gpu` |
| G3 Observed | 1 | GPU_VS_CPU_F64 | `diversity::observed_features` | `diversity_gpu::observed_features_gpu` |
| G4 Pielou | 1 | GPU_VS_CPU_F64 | `diversity::pielou_evenness` | `diversity_gpu::pielou_evenness_gpu` |
| G5 Bray-Curtis | 4 | GPU_VS_CPU_F64 | `diversity::bray_curtis_matrix` → condensed | `diversity_gpu::bray_curtis_condensed_gpu` |
| G6 PCoA | 8 | GPU_VS_CPU_F64 × 10 | `pcoa::pcoa` | `pcoa_gpu::pcoa_gpu` |
| G7 KMD | 5 | GPU_VS_CPU_F64 | `kmd::kendrick_mass_defect` | `kmd_gpu::kendrick_mass_defect_gpu` |
| G8 ODE | 6 | 0.0 (bit-exact) | `qs_biofilm::run_scenario` ×2 | CPU determinism (no GPU ODE in viz) |

## Key Findings

- All diversity metrics match CPU↔GPU within 1e-6 tolerance
- PCoA eigenvalues and coordinates match within 1e-5 (sign-invariant comparison)
- KMD values match within 1e-6
- ODE integration is deterministic (bit-exact across runs)
- Bray-Curtis: GPU returns condensed upper-triangle, CPU returns full matrix — test extracts condensed from CPU for apples-to-apples comparison

## Chain Position

CPU v25 (46/46) → GPU v14 (27/27) → metalForge v17 (29/29) → Viz V1 (45/45) → **CPU↔GPU Math (27/27)**
