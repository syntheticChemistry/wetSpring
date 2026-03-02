# Experiment 301: CPU vs GPU Full Domain Parity — V92G ComputeDispatch

**Date:** March 2, 2026
**Status:** DONE
**Phase:** V92H
**Objective:** Validate CPU↔GPU parity across 15 domains via ToadStool ComputeDispatch

---

## Sections

| Section | Domain | ComputeDispatch Op | Checks | Status |
|---------|--------|-------------------|:------:|--------|
| D01 | Diversity (Shannon/Simpson/Observed) | FusedMapReduceF64 | 12 | PASS |
| D02 | Diversity Fusion | DiversityFusionGpu | 1 | PASS |
| D03 | Bray-Curtis condensed | BrayCurtisF64 | 2 | PASS |
| D04 | PCoA eigendecomposition | BatchedEighGpu | 2 | PASS |
| D05 | GEMM f64 | GemmF64 | 1 | PASS |
| D06 | GEMM cached | GemmCachedF64 | 2 | PASS |
| D07 | NMF | nmf::nmf | 3 | PASS |
| D08 | Graph Laplacian | barracuda::linalg | 2 | PASS |
| D09 | Anderson localization | spectral::anderson_3d + lanczos | 3 | PASS |
| D10 | Bootstrap + Jackknife | stats | 4 | PASS |
| D11 | Hydrology ET₀ | stats (6 methods) | 7 | PASS |
| D12 | Boltzmann sampling | sample | 2 | PASS |
| D13 | LHS + Sobol | sample | 2 | PASS |
| D14 | DF64 Host protocol | df64_host | 3 | PASS |
| D15 | Regression | stats::fit_all | 2 | PASS |
| **Total** | **15 domains** | **17+ ComputeDispatch ops** | **48** | **ALL PASS** |

## Command

```bash
cargo run --features gpu --release --bin validate_cpu_gpu_full_domain_v92g
```

## Result

48/48 checks — GPU: RTX 4070 (Hybrid Fp64), 227.6 ms total.
