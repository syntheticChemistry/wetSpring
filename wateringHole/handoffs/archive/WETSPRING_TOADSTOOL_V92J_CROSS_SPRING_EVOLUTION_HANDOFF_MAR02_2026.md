# V92J Handoff: Cross-Spring Evolution + S87 Modern Systems

**Date:** March 2, 2026
**From:** wetSpring
**To:** ToadStool / BarraCuda team
**ToadStool Pin:** S87 (`2dc26792`)
**Phase:** V92J — 1,219 tests, 280 experiments, 8,241+ checks, 284 binaries
**Status:** Full 5-tier GREEN, fully lean (0 local WGSL, 0 local derivative math)

---

## Executive Summary

- wetSpring is **fully lean** on ToadStool S87: 144 primitives consumed, 264
  ComputeDispatch ops exercised, zero local WGSL, zero local derivative math.
- **S87 absorption** validated with zero regressions across 14 binaries and 1,219 tests.
- **Exp304** (61/61): comprehensive cross-spring evolution benchmark tracking shader
  provenance across 6 springs with GPU benchmarks on RTX 4070.
- **GEMM 256×256: 7.1× GPU speedup** over CPU; BrayCurtis 50×200: 2.0ms GPU;
  DF64 π roundtrip: 3.55e-15.
- **3 absorption opportunities** remain for ToadStool: GPU-streamable spectral
  primitives, GPU Boltzmann sampling, GPU Hydrology ET₀.

---

## Part 1: Cross-Spring Shader Evolution

ToadStool's 264 ComputeDispatch ops evolved from contributions across 6 springs.
Exp304 validates and benchmarks this evolution with full provenance tracking.

### When + Where + Who Benefits

| Spring | Written | Absorbed (Session) | Key Primitives | Consumed By |
|--------|---------|-------------------|----------------|-------------|
| **hotSpring** | v0.4.0 (Feb 14) | S26, S58, S64, S80 | DF64 precision, Anderson spectral, grid/mixing PDE, NVK workarounds, Sovereign compiler | ALL springs (universal precision layer) |
| **wetSpring** | V6 (Feb 16) | S27, S31, S58, S63, S64 | Bio diversity (Shannon/Simpson/Bray-Curtis), 5 ODE systems, DADA2/HMM/alignment, NMF, ridge regression | neuralSpring (brain diversity), groundSpring (ecological) |
| **neuralSpring** | S-01 (Feb 20) | S27, S54, S56, S64 | GemmF64, graph linalg, pairwise metrics (Hamming/Jaccard/L2), AlphaFold2, BatchedEncoder | wetSpring (NMF, PCoA), all springs |
| **airSpring** | V039 (Feb 22) | S40, S70, S81 | 6 hydrology ET₀ methods, Richards PDE, seasonal pipeline, Kriging | groundSpring (soil moisture), airSpring (seasonal forecast) |
| **groundSpring** | V54 (Feb 26) | S66, S70, S81 | Bootstrap, jackknife, Wright-Fisher, InterconnectTopology, FitResult accessors | wetSpring (population genomics), airSpring (calibration) |
| **wateringHole** | V69 (Feb 28) | S76, S80, S83 | Boltzmann sampling, Sobol/LHS, BrentGpu, L-BFGS, Omelyan integrator | all springs (optimization) |

### Key Cross-Spring Compositions

Springs don't import each other — they compose via ToadStool:

```
wetSpring NMF pipeline = wetSpring bio (NMF, S64) × neuralSpring (GEMM, S64) × hotSpring (DF64, S58)
wetSpring PCoA pipeline = wetSpring bio (BC, S82) × neuralSpring (Eigh, S64) × hotSpring (precision)
All GPU paths = spring-specific science × hotSpring universal precision layer
```

---

## Part 2: S87 Upstream Evolution

ToadStool S87 is a deep debt evolution and resilience cycle:

| Category | Change | Impact |
|----------|--------|--------|
| **async-trait** | 75 TODO(afit) → NOTE(async-dyn) | Conscious decision documented; not debt |
| **FHE shaders** | NTT/INTT bit-by-bit reduction rewrite; pointwise mul fix | 19 FHE tests pass |
| **gpu_helpers** | 663-line monolith → 3 submodules | Cleaner buffer/BGL/pipeline structure |
| **Device recovery** | `BarracudaError::is_device_lost()` + `with_device_retry` | Production resilience |
| **Unsafe audit** | ~60+ sites documented with SAFETY comments | All verified necessary (GPU/FFI) |
| **Tests fixed** | 9 pre-existing failures (kernel router, adapter, fault) | 2,866+ barracuda tests pass |

**No breaking API changes.** wetSpring compiled and validated without modifications.

---

## Part 3: wetSpring Barracuda Usage Summary

### Module Inventory

| Category | Count | Description |
|----------|-------|-------------|
| CPU modules | 47 | Pure Rust bioinformatics (diversity, alignment, phylo, ODE, etc.) |
| GPU modules | 42 | All lean on ToadStool primitives; 0 local WGSL |
| Compose modules | 7 | Wire upstream primitives (KMD, merge_pairs, RF, derep, NJ, reconciliation, molecular_clock) |
| IPC module | 1 | JSON-RPC over Unix socket (biomeOS integration) |
| Vault module | 1 | Encrypted storage, consent, provenance (BearDog crypto) |
| Validation binaries | 284 | 5-tier chain: Paper Math → CPU → GPU → Streaming → metalForge |

### ToadStool Primitives Consumed (144 of 264)

**Top-used by wetSpring:**
- `FusedMapReduceF64` — diversity, spectral, stats
- `BrayCurtisF64` — metagenomics distance matrices
- `BatchedEighGpu` — PCoA eigendecomposition
- `GemmF64` / `GemmCached` — NMF drug repurposing
- `BatchedOdeRK4F64` — 5 ODE systems (ComputeDispatch)
- `DiversityFusionGpu` — fused Shannon/Simpson/Chao1
- `ComputeDispatch` — all 5 ODE modules + GEMM
- `BufferPool` / `TensorContext` — GPU memory management
- `Fp64Strategy` / `GpuDriverProfile` — precision selection
- 30+ bio ops (SmithWaterman, Felsenstein, HMM, Gillespie, etc.)

### Write → Absorb → Lean Status

| Phase | Count | Status |
|-------|-------|--------|
| **Lean** | 34 | Use upstream ops directly (FMR, BC, Eigh, etc.) |
| **Compose** | 7 | Wire upstream primitives (KMD, merge_pairs, RF, etc.) |
| **ODE (trait-generated)** | 5 | `BatchedOdeRK4::<S>::generate_shader()` — no local WGSL |
| **Passthrough** | 0 | All promoted |
| **Pending absorption** | 0 | Nothing blocking |

---

## Part 4: Benchmark Results (RTX 4070, DF64 Hybrid)

### GPU vs CPU Scaling (Exp304)

| Operation | CPU (ms) | GPU (ms) | Speedup | Origin |
|-----------|----------|----------|---------|--------|
| GEMM 64×64 | 0.4 | 18.6 | 0.02× | neuralSpring (dispatch overhead dominates) |
| GEMM 128×128 | 3.6 | 3.5 | **1.0×** | neuralSpring (crossover point) |
| GEMM 256×256 | 28.9 | 4.1 | **7.1×** | neuralSpring (GPU wins) |
| BrayCurtis 50×200 | — | 2.0 | — | wetSpring |
| GemmCached 256×128×64 | — | 2.5 | — | wetSpring (composed) |

### CPU Throughput (Cross-Spring)

| Primitive | Origin | µs/op | ops/sec |
|-----------|--------|-------|---------|
| Shannon | wetSpring→S63 | <0.001 | 41.7B |
| Simpson | wetSpring→S63 | <0.001 | 90.9B |
| Bray-Curtis | wetSpring→S82 | <0.001 | 43.5B |
| Pearson r | neuralSpring→S66 | 1.47 | 681K |
| Ridge (50×10) | hotSpring→S59 | 2.53 | 395K |
| Trapz | hotSpring→S59 | 0.82 | 1.2M |

---

## Part 5: Absorption Opportunities for ToadStool

### 5.1 GPU-Streamable Spectral Primitives

**Current:** `spectral_match` and `spectral_density` run on CPU.
**Opportunity:** Absorb as WGSL shaders for on-device spectral pipeline.
**Benefit:** Eliminates CPU round-trip in LC-MS PFAS screening workflow.
**Complexity:** Medium (existing FusedMapReduceF64 pattern applies).

### 5.2 GPU Boltzmann Sampling

**Current:** CPU Boltzmann sampling with softmax/temperature.
**Opportunity:** `BoltzmannSamplingGpu` shader (wateringHole S76 started this).
**Benefit:** On-device hyperparameter search for all springs.
**Complexity:** Low (softmax shader exists; add temperature + proposal).

### 5.3 GPU Hydrology ET₀

**Current:** 6 CPU methods (Hargreaves, FAO-56, Thornthwaite, Makkink, Turc, Hamon).
**Opportunity:** Fused map-reduce shader for batch ET₀ computation.
**Benefit:** Seasonal pipeline stays on GPU (airSpring, groundSpring).
**Complexity:** Low (6 simple formulas → single fused shader).

### 5.4 Compose → Lean Upgrades (Lower Priority)

| Module | Current | Needed from ToadStool |
|--------|---------|----------------------|
| `merge_pairs_gpu` | CPU merge + FMR stats | `BatchMergePairsGpu` (overlap scoring) |
| `robinson_foulds_gpu` | CPU bipartition compare | `BipartitionEncodeGpu` (string→bitvec) |
| `kmd_gpu` | `KmerHistogramGpu` compose | Native KMD shader |
| `derep_gpu` | `KmerHistogramGpu` compose | Native DerepGpu shader |

---

## Part 6: Learnings for ToadStool Evolution

1. **barracuda always-on works.** No `#[cfg(not(feature = "gpu"))]` fallbacks.
   CPU math comes from barracuda. Feature gates only control GPU/NPU/IPC.

2. **ODE trait pattern is powerful.** `BatchedOdeRK4::<S>::generate_shader()` generates
   WGSL from trait impls — no handwritten shaders needed. 5 ODE systems use this.

3. **ComputeDispatch builder is clean.** All 6 wetSpring GPU modules use it. The
   `.f64()` path works; `.df64()` would be welcome when ready.

4. **GEMM crossover at 128×128.** Below that, CPU wins due to dispatch overhead.
   Consider latency-aware dispatch hints for small matrices.

5. **DF64 Hybrid strategy is correct.** RTX 4070 runs native f64 at 1:64 rate;
   DF64 on FP32 cores gives ~10× throughput for bulk math.

6. **S87 device-lost recovery is not yet consumed.** wetSpring uses sync validation
   binaries, not async tokio tests. A sync `with_device_retry` equivalent would help.

7. **Cross-spring provenance matters.** Exp304 proves the ecosystem model works:
   6 springs contribute independently, ToadStool absorbs, everyone benefits.

---

## Metrics Delta

| Metric | V92I (before) | V92J (after) |
|--------|---------------|--------------|
| ToadStool pin | S87 (`2dc26792`) | S87 (`2dc26792`) |
| Experiments | 279 | 280 (+1: Exp304) |
| Validation checks | 8,180+ | 8,241+ (+61) |
| Binaries | 267 | 284 (+17: 6 recovered unregistered + 1 Exp304 + 10 prior) |
| Tests | 1,219 | 1,219 |
| ComputeDispatch ops | 144/264 | 144/264 |
| Cross-spring provenance | partial | Exp304 full map |
