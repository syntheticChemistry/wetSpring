# wetSpring V98+ Cross-Spring Evolution Handoff

**Date:** March 8, 2026
**Version:** V98+
**Author:** wetSpring validation framework
**Recipients:** barraCuda team, toadStool team, coralReef team, all Springs

## 1. Upstream Pins

| Primal | Commit | Version |
|--------|--------|---------|
| barraCuda | `a898dee` | v0.3.3 (deep debt: typed errors, named constants) |
| toadStool | `bfe7977b` | S130+ (deep debt, spring sync, clippy pedantic) |
| coralReef | `d29a734` | Iteration 10 (AMD E2E verified, 990 tests) |

## 2. Cross-Spring Evolution Validation (Exp319)

**52/52 PASS** — exercises primitives from all 5 springs:

### §0: Provenance Registry
- 28 total shaders in registry, 22 cross-spring
- wetSpring authored 5, consumes 17

### §1: hotSpring Precision (DF64 + Special Functions)
- `erf(0)=0`, `erf(1)≈0.8427`, `Φ(0)=0.5` — all PASS
- DF64 pack→unpack roundtrip: 5/5 values verified (π, e, 1/3, 1e15, 1e-15)
- **Evolution:** hotSpring S58 → DF64 core → used by ALL springs

### §2: wetSpring Bio (Diversity + Phylogenetics)
- Shannon, Simpson, Chao1, Bray-Curtis, Pielou — all analytical PASS
- Felsenstein pruning: log-L finite ✓
- HMM forward: log-prob finite and < 0 ✓
- **Evolution:** wetSpring V6 → ToadStool S63/S64 (absorbed)
- **Used by:** neuralSpring (pop-gen), groundSpring (ecology)

### §3: hotSpring Spectral (Anderson Localization)
- Anderson 3D (4×4×4, W=2): N=64 lattice, Lanczos eigenvalues computed
- Level spacing ratio r=0.4008 (GOE≈0.5307, Poisson≈0.3863)
- **Evolution:** hotSpring v0.6.0 → barraCuda spectral
- **Used by:** wetSpring (QS-disorder), groundSpring (noise theory)

### §4: neuralSpring ML (Graph Linalg + Statistics)
- Graph Laplacian 50×50: n² elements, effective_rank > 0
- Pearson r > 0.99 on linear data
- **Evolution:** neuralSpring V64 → ToadStool S72 (ComputeDispatch)
- **Used by:** wetSpring (drug repurposing, spectral match)

### §5: airSpring Hydrology (6 ET₀ Methods)
- Hargreaves, FAO-56 PM, Thornthwaite, Makkink, Turc, Hamon — all > 0
- **Evolution:** airSpring V039 → barraCuda stats → S70
- **Used by:** wetSpring (soil QS models), groundSpring (Richards PDE)

### §6: groundSpring Stats (Bootstrap + Jackknife + Regression)
- mean(1..100)=50.5, var=841.667, jackknife estimate=3.0 — all PASS
- Bootstrap CI: lower < mean < upper ✓
- Linear regression: slope=2.0, intercept=0.0 ✓
- **Evolution:** groundSpring V73 → barraCuda stats → all springs

### §7: wetSpring NMF (Drug Repurposing)
- NMF 20×10 rank 3: W non-negative, H non-negative, 65 iterations
- **Evolution:** wetSpring V6 → ToadStool S64 → SparseGemmF64 S82

### §8: GPU Cross-Spring (Diversity)
- GPU Shannon ≈ CPU (within GPU_VS_CPU_F64 tolerance) ✓
- GPU Simpson ≈ CPU ✓
- GPU Bray-Curtis ≈ CPU ✓
- RTX 4070 (Hybrid/DF64), FusedMapReduceF64 via DF64 shaders
- **Evolution:** wetSpring Write → ToadStool S63 Absorb → Lean

## 3. Cross-Spring Benchmark (Exp320)

24 primitives profiled CPU + GPU with evolution provenance:

| Category | Primitive | Speed | Origin |
|----------|-----------|-------|--------|
| Bio | Shannon, Simpson, BC, Chao1, Pielou (n=1k) | sub-µs | wetSpring |
| Precision | erf (1k pts), norm_cdf | sub-µs | hotSpring |
| Spectral | Anderson 3D, Lanczos 216×50 | 3-288 µs | hotSpring |
| ML | Graph Laplacian 100², Pearson, Linear fit | 1-7 µs | neuralSpring |
| Hydrology | Hargreaves, FAO-56 PM | sub-µs–0.08 µs | airSpring |
| Stats | Mean, Jackknife, Bootstrap, Kimura | 0.4-2433 µs | groundSpring |
| GPU | Shannon, Simpson, BC (1k) | 979-2982 µs | wetSpring+hot |

GPU warmup (first dispatch): ~47ms on RTX 4070.

## 4. Cross-Spring Dependency Matrix

```
               hot   wet  neur   air  grnd
hotSpring        0     5     3     1     3
wetSpring        1     0     3     1     0
neuralSpring     3     2     0     1     2
airSpring        0     3     1     0     0
groundSpring     3     2     3     2     0
```

Key flows:
- hotSpring → wetSpring: 5 shaders (DF64, spectral theory)
- wetSpring → neuralSpring: 3 shaders (bio diversity, alignment, HMM)
- neuralSpring → hotSpring: 3 shaders (GEMM, graph linalg, optimizer)
- airSpring → wetSpring: 3 shaders (hydrology, seasonal pipeline)
- groundSpring → hotSpring: 3 shaders (stats, error theory, validation)

## 5. wetSpring-Authored Shaders

| Shader | Consumers |
|--------|-----------|
| `bio/smith_waterman_banded_f64.wgsl` | wetSpring, neuralSpring |
| `bio/felsenstein_f64.wgsl` | wetSpring |
| `bio/gillespie_ssa_f64.wgsl` | wetSpring, neuralSpring |
| `bio/hmm_forward_f64.wgsl` | wetSpring, neuralSpring |
| `reduce/fused_map_reduce_f64.wgsl` | wetSpring, airSpring, hotSpring |

The `fused_map_reduce_f64.wgsl` shader is particularly impactful: authored by
wetSpring for Shannon/Simpson GPU computation, it is now consumed by airSpring
(hydrology GPU reduction) and hotSpring (spectral diagnostics).

## 6. Ecosystem Impact & Learnings

### Cross-Spring Evolution Benefits
1. **hotSpring precision shaders** enable consumer GPU f64 for ALL springs
2. **wetSpring bio shaders** flow to neuralSpring (pop-gen) and groundSpring (ecology)
3. **neuralSpring GEMM** enables wetSpring drug repurposing and hotSpring lattice QCD
4. **airSpring hydrology** enriches wetSpring soil QS models
5. **groundSpring tolerances** provide the 13-tier validation backbone for all springs

### Performance Observations
- CPU bio diversity (n=1k): sub-microsecond (pure Rust optimization)
- GPU dispatch overhead dominates for small workloads (~1-3ms per dispatch)
- GPU beneficial above n≈10k where shader parallelism amortizes dispatch
- Hybrid GPU (DF64 fallback) works correctly for map-reduce patterns
- DF64 zero-output issue isolated to dot-product shaders (not map-reduce)

### Recommendations for toadStool/barraCuda
1. **GPU dispatch batching**: Fuse multiple small ops into single dispatches
2. **DF64 dot-product fix**: Investigate zero-output for dot_gpu on Hybrid GPUs
3. **Provenance registry expansion**: Add shader creation dates as ISO timestamps
4. **Cross-spring CI**: Automated validation of cross-spring shader consumption

---

**Status:** Complete. 295 experiments, 298 binaries, 8,656+ checks.
