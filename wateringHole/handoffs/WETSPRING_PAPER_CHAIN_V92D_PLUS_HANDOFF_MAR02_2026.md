# wetSpring → ToadStool/BarraCuda: Paper-Math Chain Handoff (V92D+)

**Date:** March 2, 2026
**From:** wetSpring V92D+
**To:** ToadStool/BarraCuda
**ToadStool pin:** S79 (`f97fc2ae`)
**Supersedes:** V92D (deep debt)

---

## What Changed

Built the complete paper-math validation chain (Exp291-295):

```
Open Data (NCBI, Zenodo, EPA, journal tables)
  → Paper Math Control v4 (52 papers, 45/45)
    → BarraCuda CPU v22 (8 domains, 40/40, 0.8 ms)
      → CPU vs GPU v9 (5 tracks, 35/35)
        → Pure GPU Streaming v9 (full pipeline, 16/16)
          → metalForge v14 (cross-system, 28/28)
```

### New Validation Binaries

| Binary | Exp | Checks | Purpose |
|--------|-----|:------:|---------|
| `validate_paper_math_control_v4` | 291 | 45 | All 52 papers' core equations |
| `validate_barracuda_cpu_v22` | 292 | 40 | 8-domain CPU paper parity |
| `validate_cpu_vs_gpu_v9` | 293 | 35 | 5-track GPU portability |
| `validate_pure_gpu_streaming_v9` | 294 | 16 | End-to-end pipeline chain |
| `validate_metalforge_v14_paper_chain` | 295 | 28 | Cross-system paper math |

### Papers Added to Math Control (P33-P47)

- P33: Meyer 2020 — QS spatial propagation wave
- P34: Nitrifying QS — luxR:luxI ratio 2.3:1
- P35: Marine interkingdom — plankton QS prevalence
- P36: Myxococcus — critical cell density threshold
- P37: Dictyostelium — cAMP relay ODE
- P38: Fajgenbaum 2025 — MATRIX pharmacophenomics (NMF)
- P39: Gao 2020 — repoDB NMF drug repurposing
- P40: ROBOKOP — knowledge graph embedding (TransE)
- P41: Mukherjee 2024 — cell distancing colonization
- P42-P47: Gonzales Track 5 (IC50, PK decay, IL-31, pruritus, three-compartment Anderson, JAK1 selectivity)

---

## What ToadStool Should Absorb

### CPU Primitives (Already Upstream)
All math primitives used in Exp291-295 are already in barracuda:
- `barracuda::linalg::nmf` (NMF, cosine_similarity)
- `barracuda::stats` (bootstrap_ci, jackknife, pearson_correlation, fit_linear, norm_cdf, chao1)
- `barracuda::special` (erf, erfc)
- `barracuda::linalg::ridge_regression`

### GPU Promotion Candidates
The CPU v22 8-domain coverage creates clear GPU promotion targets:
- **Hill equation vectorization**: Batch dose-response curves on GPU (trivially parallel)
- **PK decay curves**: Batch exponential decay on GPU
- **TransE scoring**: Already upstream (`barracuda::ops::transe_score_f64`)
- **Anderson W→P(QS) mapping**: Batch norm_cdf on GPU (already erf in WGSL)

### Streaming Pipeline Pattern
Exp294 demonstrates the pattern ToadStool should optimize:
1. Diversity batch (FusedMapReduceF64)
2. Bray-Curtis pairwise (BrayCurtisF64)
3. NMF factorization (GPU NMF when available)
4. Anderson W-mapping (batch norm_cdf)
5. Statistics aggregation (CPU readback)

Key metric: streaming < individual dispatch by ~33% (buffer reuse).

### metalForge Lessons
- 47 workloads in catalog, 45 absorbed, 0 local
- PCIe streaming: 3 GPU-chained / 0 CPU round-trips for pure GPU pipeline
- Sovereign mode: 45/47 route locally without NestGate or Songbird
- Cross-substrate transitions: GPU→CPU→GPU→CPU proven stable

---

## Quality State

- `cargo fmt --all`: CLEAN
- `cargo clippy --all-features -W pedantic`: 0 warnings
- `cargo test --workspace`: 1,309 tests, 0 failures
- 277 experiments, 7,384+ validation checks
- Zero TODO/FIXME/HACK in .rs files
- Zero unsafe code
- Zero local WGSL (all upstream via ToadStool)

---

## Totals

| Metric | Value |
|--------|-------|
| Experiments | 277 |
| Validation checks | 7,384+ |
| Tests | 1,309 |
| Binaries | 260 |
| Papers | 52 + 6 reproduced |
| Tolerances | 103 named |
| ToadStool primitives | 93 consumed |
