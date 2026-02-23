# Exp104: metalForge Cross-Substrate v6 — Complete Three-Tier Coverage

| Field    | Value                                       |
|----------|---------------------------------------------|
| Script   | `validate_metalforge_v6`                    |
| Command  | `cargo run --features gpu --release --bin validate_metalforge_v6` |
| Status   | **PASS** (24/24)                            |
| Phase    | 29                                          |
| Depends  | Exp093 (v3), Exp100 (v4), Exp103 (v5)      |

## Purpose

Closes all remaining Three-Tier Matrix gaps. After Exp103 (metalForge v5)
proved 29 GPU domains substrate-independent, 8 of 25 actionable papers still
lacked metalForge coverage because their key algorithms had never been exercised
through an MF routing validation binary. This experiment adds the 5 missing
gap domains:

| # | Domain | Paper(s) | Primitive | CPU↔GPU |
|---|--------|----------|-----------|---------|
| V6-01 | QS ODE sweep | Paper 5 (Waters 2008) | `BatchedOdeRK4F64` | parity < 0.15 |
| V6-02 | UniFrac propagation | Paper 1 (Galaxy 16S) | `UniFracPropagateGpu` | Exact |
| V6-03 | DADA2 denoising | Paper 1 (Galaxy 16S) | `Dada2EStepGpu` | Exact |
| V6-04 | K-mer histogram | Paper 28 (Anderson 2014) | `KmerHistogramGpu` | Exact |
| V6-05 | Felsenstein pruning | Papers 16, 17, 20 | `FelsensteinGpu` | GPU_VS_CPU_F64 |

## Results

| Section | Checks | Status | Notes |
|---------|--------|--------|-------|
| MF-V6-01: QS ODE Sweep | 4/4 | PASS | 8-batch sweep, max |CPU−GPU| = 4.7e-2 |
| MF-V6-02: UniFrac | 8/8 | PASS | 5-node tree, 3 leaves, 2 samples, all exact |
| MF-V6-03: DADA2 | 4/4 | PASS | 3 sequences → 1 ASV, GPU E-step parity |
| MF-V6-04: K-mer | 3/3 | PASS | k=4, 256-bin histogram, 24 total k-mers exact |
| MF-V6-05: Felsenstein | 5/5 | PASS | 3-taxon tree, CPU LL = GPU LL = −29.262870 |
| **Total** | **24/24** | **PASS** | |

## Three-Tier Matrix Impact

### Papers promoted to full three-tier (CPU + GPU + metalForge)

| Paper | Domain(s) | MF validation |
|-------|-----------|---------------|
| Paper 1 (Galaxy 16S) | UniFrac, DADA2, chimera (v5), derep (v5) | **Exp104 + Exp103** |
| Paper 5 (Waters 2008) | QS ODE | **Exp104** |
| Paper 7 (Hsueh 2022) | Phage defense ODE | Already validated in **Exp100** (v4) |
| Paper 8 (Fernandez 2020) | Bistable ODE | Already validated in **Exp100** (v4) |
| Paper 12 (Srivastava 2011) | Multi-signal ODE | Already validated in **Exp100** (v4) |
| Paper 16 (Alamin 2024) | Felsenstein (placement) | **Exp104** |
| Paper 17 (Liu 2009) | NJ (v5) + Felsenstein | **Exp104 + Exp103** |
| Paper 20 (Wang 2021) | Felsenstein (bootstrap) | **Exp104** |
| Paper 28 (Anderson 2014) | K-mer histogram | **Exp104** |

### Updated coverage

| Before (Exp103) | After (Exp104) |
|-----------------|----------------|
| 17/25 full three-tier | **25/25 full three-tier** |
| 8 papers CPU+GPU only | **0 papers CPU+GPU only** |

### Only remaining exclusions (by design)

| Paper | Reason |
|-------|--------|
| Paper 11 (Waters 2021) | Reference only — no reproduction target |
| Paper 19 (Liu fungi) | Manuscript in progress |
| Paper 23 (Kachkovskiy) | Cross-spring — reproduction in groundSpring |
| PCoA in metalForge | naga WGSL compiler bug (tracked upstream) |

## metalForge Domain Coverage History

| Version | Exp | Domains | Total |
|---------|-----|---------|-------|
| v1 | 060 | 4 core (ANI, SNP, pangenome, dN/dS) | 4 |
| v2 | 084 | +8 (SW, Gillespie, DT, spectral, diversity, Bray-Curtis, HMM, RF) | 12 |
| v3 | 093 | +4 (EIC, PCoA, Kriging, Rarefaction) | 16 |
| v4 | 100 | +3 (phage_defense, bistable, multi_signal ODE) + NPU routing | 19 |
| v5 | 103 | +13 (pure GPU promotion: cooperation, capacitor, KMD, etc.) | 32 |
| **v6** | **104** | **+5 (QS ODE, UniFrac, DADA2, K-mer, Felsenstein)** | **37** |

## Workload Catalog Update

Added 3 new workloads to `metalForge/forge/src/workloads.rs`:
- `dada2()` — DADA2 denoising via `Dada2EStepGpu`
- `bootstrap()` — phylogenetic bootstrap via `FelsensteinGpu`
- `placement()` — metagenomic placement via `FelsensteinGpu`

Updated counts: 22 absorbed, 5 local WGSL, 1 CPU-only = 28 total.

## Reproduction

```bash
cargo run --features gpu --release --bin validate_metalforge_v6
```

## Provenance

| Field | Value |
|-------|-------|
| Binary | `validate_metalforge_v6` |
| Date | 2026-02-23 |
| Hardware | i9-12900K, RTX 4070, 64 GB DDR5 |
| Data | Synthetic (self-contained) |
