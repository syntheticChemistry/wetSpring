# Exp256: EMP-Scale Anderson Atlas — 30K Biome QS Classification

**Status:** PASS (35/35 checks)
**Date:** 2026-03-01
**Binary:** `validate_emp_anderson_atlas`
**Command:** `cargo run --release --bin validate_emp_anderson_atlas`
**Feature gate:** none (CPU only; GPU-accelerated Anderson available via `--features gpu`)

## Purpose

Proves the Anderson-QS classification pipeline scales to Earth Microbiome
Project (EMP) dimensions. Uses 30,002 synthetic samples calibrated to EMP's
14 EMPO biome categories with realistic diversity distributions.

## Science

Tests Paper 01's central prediction: all natural 3D biomes sustain QS
(extended Anderson states, r > midpoint). At EMP scale, this produces the
first comprehensive Anderson regime atlas across 14 biome types.

## Pipeline

```
Community (N=150 taxa) → Shannon H' → Pielou J → W = 0.5 + 14.5×J → Anderson 3D → r → classify
```

## NUCLEUS Integration

When biomeOS NUCLEUS is running:
1. Discovers orchestrator via socket scan
2. Routes through `capability.call` → `science.full_pipeline`
3. Reports active tier (1=biomeOS, 2=NestGate, 3=standalone)

When standalone: all math runs directly through barracuda (same results).

## Results

| Biome | H' | J | W | r | QS% |
|-------|---:|---:|---:|---:|----:|
| Animal corpus | 3.73 | 0.804 | 12.2 | 0.499 | 100% |
| Animal distal gut | 4.23 | 0.881 | 13.3 | 0.496 | 100% |
| Non-saline water | 4.42 | 0.912 | 13.7 | 0.495 | 100% |
| Saline sediment | 4.55 | 0.932 | 14.0 | 0.494 | 100% |
| Soil (non-saline) | 4.64 | 0.946 | 14.2 | 0.494 | 100% |
| Plant rhizosphere | 4.45 | 0.917 | 13.8 | 0.495 | 100% |

**Global:** 30,002/30,002 QS-active (100%) in 55ms standalone.

## Cross-Biome Statistics

- Jackknife mean r: 0.4962 ± 0.000465
- Bootstrap 95% CI: [0.4954, 0.4970] (entirely above midpoint 0.459)
- W↔r Pearson: -1.000 (higher disorder → lower r, as Anderson theory predicts)

## Next Steps

1. Replace synthetic communities with real EMP OTU tables (NestGate BIOM fetch)
2. Run through NUCLEUS when biomeOS is active (Exp257 probes readiness)
3. GPU-accelerate Anderson spectral via ToadStool (Exp254 proves portability)

## Chain

Paper (Exp251) → CPU (Exp252) → Python parity (Exp253) → GPU (Exp254) →
Streaming (Exp255) → **EMP Atlas (Exp256)** → NUCLEUS Pipeline (Exp257) →
Tower-Node (Exp258)
