# wetSpring → ToadStool/BarraCUDA Handoff v10

**Date:** February 22, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Context:** Phase 22 — Cross-Spring Evolution validated, neuralSpring primitives wired

---

## Executive Summary

wetSpring has completed **95 experiments, 2,219+ validation checks, 728 Rust
tests, and 85 binaries** — all passing. **24 of 25 GPU modules** lean on
upstream ToadStool primitives (19 wetSpring-evolved + 5 neuralSpring-evolved).
4 local WGSL shaders remain in Write phase pending absorption.

**New in v10:**

1. **5 neuralSpring-evolved primitives wired and validated** (Exp094, 39/39 PASS)
2. **Cross-spring scaling benchmark** (Exp095) — up to 277× GPU speedup
3. **Cross-spring provenance** fully traced and documented
4. **ODE blocker** remains: `compile_shader()` vs `compile_shader_f64()` (unchanged from v9)

---

## Part 1: neuralSpring Primitive Wiring (NEW)

5 primitives evolved by neuralSpring, absorbed by ToadStool in session 31f,
now validated and benchmarked in wetSpring:

| Primitive | Use Case | Exp094 | Exp095 Speedup |
|-----------|---------|--------|---------------|
| `PairwiseHammingGpu` | Sequence distance (metagenomics) | 10/10 PASS | **16.4×** |
| `PairwiseJaccardGpu` | Pangenome gene P/A distance | 6/6 PASS | **276.7×** |
| `SpatialPayoffGpu` | Game theory cooperation fitness | 1/1 PASS | **19.6×** |
| `BatchFitnessGpu` | Population fitness evaluation | 16/16 PASS | **6.5×** |
| `LocusVarianceGpu` | Weir-Cockerham FST (allele freq) | 6/6 PASS | **19.2×** |

### Key Technical Notes

- **LocusVariance layout**: Row-major `allele_freqs[pop * n_loci + locus]` —
  CPU baselines must match for correct parity
- **Buffer-based API**: All 5 primitives use raw `wgpu::Buffer` I/O, requiring
  manual buffer creation + readback (unlike slice-based FMR/GEMM APIs)
- **All f32**: These primitives operate in f32 (unlike wetSpring's f64 bio ops)

---

## Part 2: Absorption State

### Lean (24 primitives — consuming upstream)

| # | Primitive | Evolved By | Absorbed |
|---|-----------|-----------|----------|
| 1–12 | 12 bio ops (SW, Felsenstein, HMM, ANI, etc.) | wetSpring | Sessions 18–31g |
| 13 | `FusedMapReduceF64` | hotSpring | Session 18 |
| 14 | `GemmCachedF64` | wetSpring | Session 18 |
| 15 | `TreeInferenceF64` | wetSpring | Session 18 |
| 16–19 | DADA2, QualityFilter, SNP, Pangenome | wetSpring | Session 31g |
| 20 | `PairwiseHammingGpu` | neuralSpring | Session 31f |
| 21 | `PairwiseJaccardGpu` | neuralSpring | Session 31f |
| 22 | `SpatialPayoffGpu` | neuralSpring | Session 31f |
| 23 | `BatchFitnessGpu` | neuralSpring | Session 31f |
| 24 | `LocusVarianceGpu` | neuralSpring | Session 31f |

### Write Phase (4 local WGSL shaders)

| Shader | Status | Blocker |
|--------|--------|---------|
| `batched_qs_ode_rk4_f64.wgsl` | Local workaround | `compile_shader()` → needs `compile_shader_f64()` |
| `kmer_histogram_f64.wgsl` | Written, needs test | None |
| `unifrac_propagate_f64.wgsl` | Written, needs test | None |
| `taxonomy_fc_f64.wgsl` | Written, needs test | None |

### Blocked (1)

- **PCoA GPU** (`BatchedEighGpu`): `naga` shader validation error persists.
  Wrapped with `catch_unwind` for graceful degradation.

---

## Part 3: Cross-Spring Evolution Proof

### The Biome Model Works

```
hotSpring  → precision shaders, eigensolvers, driver workarounds
             ↓ absorbed by ToadStool
wetSpring  → 12 bio shaders, GEMM 60× speedup, f64 precision fix
             ↓ absorbed by ToadStool  
neuralSpring → 5 evolutionary/distance primitives
             ↓ absorbed by ToadStool
All Springs benefit from all contributions
```

### Concrete Cross-Pollination Examples

1. **wetSpring GEMM → hotSpring HFB**: The 60× matmul speedup evolved by
   wetSpring for diversity matrices now powers hotSpring's nuclear HFB solver
2. **hotSpring ShaderTemplate → wetSpring bio**: The f64 preamble injection
   evolved by hotSpring for NVK compatibility enables all wetSpring f64 shaders
3. **neuralSpring distances → wetSpring metagenomics**: Pairwise Hamming/Jaccard
   evolved for neural architecture search now power wetSpring pangenome analysis

---

## Part 4: Feedback for ToadStool Team

### Critical (unchanged from v9)

1. **ODE `compile_shader` bug**: `batched_ode_rk4.rs:209` calls `compile_shader()`
   instead of `compile_shader_f64()`. Without f64 preamble injection, the shader
   won't compile on naga/Vulkan. Fix: `dev.compile_shader_f64(source, label)`

### Suggestions

1. **Crate-level re-exports** for new bio primitives — currently require deep
   `barracuda::ops::bio::pairwise_hamming::PairwiseHammingGpu` paths
2. **Consistent API surface**: Bio primitives use buffer-based API while FMR/GEMM
   use slice-based. Consider offering both for ergonomics
3. **PCoA naga issue**: `BatchedEighGpu` still fails shader validation. Any
   progress on the `naga` "invalid function call" error?

---

*Supersedes: v9 (archived)*
*Previous: WETSPRING_TOADSTOOL_V9_FEB22_2026.md*
