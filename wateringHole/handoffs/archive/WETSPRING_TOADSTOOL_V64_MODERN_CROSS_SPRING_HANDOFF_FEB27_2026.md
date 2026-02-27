# wetSpring → ToadStool/BarraCuda Handoff V64

**Date**: February 27, 2026
**Phase**: V64 — Modern Cross-Spring Rewiring
**ToadStool pin**: `e96576ee` (S68+)
**Previous**: V63 (S68+ Realignment) — archived

---

## Executive Summary

V64 completes the modern ToadStool rewiring by surfacing `Fp64Strategy` and
`optimal_precision()` in wetSpring's `GpuF64` wrapper, migrating all 6 GPU
dispatch modules to ToadStool's resilient `submit_and_poll()`, and creating
a comprehensive cross-spring evolution benchmark (Exp210, 24/24) that
traces provenance across all 5 springs.

---

## Part 1: Modern Capabilities Wired

### GpuF64::fp64_strategy() (hotSpring S58 → ToadStool S67)
- Surfaces `GpuDriverProfile::fp64_strategy()` — `Native` vs `Hybrid`
- RTX 4070 detected as `Hybrid` (1:64 f64:f32 ratio → DF64 beneficial)

### GpuF64::optimal_precision() (ToadStool S68 universal)
- Returns `Precision::F64` on compute-class GPUs (Titan V, V100, A100)
- Returns `Precision::Df64` on consumer GPUs (RTX 4070 → DF64 core-streaming)
- Callers can pass this to `compile_shader_universal()` for automatic precision routing

### submit_and_poll Migration (ToadStool S68+)
6 modules migrated from raw `q.submit() + d.poll()` to `self.device.submit_and_poll()`:

| Module | Domain | Status |
|--------|--------|--------|
| `bistable_gpu` | Phenotypic switching ODE | PASS |
| `capacitor_gpu` | Phenotypic capacitor ODE | PASS |
| `cooperation_gpu` | QS game theory ODE | PASS |
| `multi_signal_gpu` | Multi-input QS ODE | PASS |
| `phage_defense_gpu` | Phage-defense ODE | PASS |
| `gemm_cached` | GEMM f64 pipeline | PASS |

Benefits:
- `DispatchSemaphore` — concurrency management (dGPU=8, iGPU=4, CPU=2 permits)
- `catch_unwind` — device-lost panic → graceful `is_lost()` flag
- `gpu_lock` — mutex-protected submit for thread safety

---

## Part 2: Cross-Spring Evolution Benchmark (Exp210)

24/24 checks PASS. Validates primitives from all 5 springs with provenance tracking:

| Spring | Validated Capabilities | Sessions |
|--------|----------------------|----------|
| hotSpring | Fp64Strategy, erf, ln_gamma, norm_cdf, Anderson spectral | S58-S68 |
| wetSpring | diversity, DiversityFusion GPU, GEMM cached, NMF, ridge | S41-S68 |
| neuralSpring | (via ToadStool pairwise ops, graph Laplacian) | S54-S56 |
| airSpring | Pearson correlation, MAE, RMSE, R², trapz | S64-S66 |
| groundSpring | (via ToadStool bootstrap, batched multinomial) | S64-S66 |

### Timing Results (RTX 4070, release)

| Benchmark | Origin | Time |
|-----------|--------|------|
| CPU diversity 500 taxa | wetSpring→S64 | 0.013ms |
| GPU DiversityFusion 5×10k | wetSpring→S63 | 52.7ms |
| Anderson 3D L=8 | hotSpring→S59 | 2.6ms |
| GEMM 256×256 cold | wetSpring→S62 | 2.5ms |
| GEMM 256×256 cached | wetSpring→S68+ | 2.1ms |
| NMF 20×10 KL k=3 | wetSpring→S58 | 0.22ms |
| Ridge 20×5→2 | wetSpring→S59 | 0.002ms |

---

## Part 3: Full Validation Sweep

| Binary | Experiment | Checks | Status |
|--------|-----------|--------|--------|
| `benchmark_cross_spring_modern_s68plus` | Exp210 | 24/24 | PASS |
| `benchmark_cross_spring_s68` | Exp189 | 28/28 | PASS |
| `validate_cross_spring_evolution` | Cross-spring | all | PASS |
| `validate_gpu_diversity_fusion` | Diversity GPU | all | PASS |
| `validate_gpu_drug_repurposing` | Drug repurposing | all | PASS |
| `validate_knowledge_graph_embedding` | KG embedding | all | PASS |
| `validate_pure_gpu_pipeline` | Pure GPU | all | PASS |
| `validate_barracuda_cpu_v11` | Exp206 IPC CPU | 64/64 | PASS |
| `validate_barracuda_gpu_v4` | Exp207 IPC GPU | 54/54 | PASS |
| `validate_metalforge_v7_mixed` | Exp208 metalForge | 75/75 | PASS |
| `validate_bistable` | Bistable ODE | all | PASS |
| `validate_capacitor` | Capacitor ODE | all | PASS |
| `validate_cooperation` | Cooperation ODE | all | PASS |
| `validate_multi_signal` | Multi-signal ODE | all | PASS |
| `validate_phage_defense` | Phage defense ODE | all | PASS |
| `cargo test` | Library | 20/20 | PASS |
| `cargo clippy --features gpu,ipc` | Linting | 0 warnings | CLEAN |

---

## Part 4: Recommendations for ToadStool

### High Priority
1. **Universal precision ODE shaders**: `BatchedOdeRK4::generate_shader()` currently emits raw `f64` WGSL. Evolving to emit `Scalar`/`op_*` universal precision code would let consumer GPUs route ODE integration through DF64 core-streaming for ~10× throughput.

### Medium Priority
2. **`ComputeDispatch` builder for wetSpring**: Would eliminate ~80 lines of bind-group/pipeline boilerplate per GPU module (5 ODE + 1 GEMM = ~480 lines).
3. **DF64 GEMM**: `GemmCached` uses `Precision::F64`; switching to `Precision::Df64` when DF64 GEMM shader is bind-group-compatible would give ~10× on consumer GPUs.

### Low Priority / Future
4. **`BandwidthTier` in metalForge**: PCIe-aware routing for cross-substrate dispatch (GPU→NPU→CPU).
5. **`SparseGemmF64`**: CSR × dense for drug-disease NMF sparse matrices.

### Cross-Spring Evolution Notes
- **hotSpring precision shaders** (Fp64Strategy, DF64 core, NVK workarounds) are now surfaced in wetSpring via `GpuF64::fp64_strategy()` and `optimal_precision()` — all springs benefit from hotSpring's precision work.
- **wetSpring bio shaders** (35 absorbed) continue to serve neuralSpring for evolutionary ecology workloads (spatial payoff, swarm NN using wetSpring's ODE trait system).
- **neuralSpring pairwise ops** (L2, Hamming, Jaccard) are used by wetSpring for genomics distance matrices.
- The cross-spring cycle is fully validated: each spring contributes → ToadStool absorbs → all springs benefit.

---

## Part 5: Files Changed (V64)

| File | Change |
|------|--------|
| `barracuda/src/gpu.rs` | Added `fp64_strategy()`, `optimal_precision()`, enhanced `print_info()` |
| `barracuda/src/bio/bistable_gpu.rs` | Migrated to `submit_and_poll()` |
| `barracuda/src/bio/capacitor_gpu.rs` | Migrated to `submit_and_poll()` |
| `barracuda/src/bio/cooperation_gpu.rs` | Migrated to `submit_and_poll()` |
| `barracuda/src/bio/multi_signal_gpu.rs` | Migrated to `submit_and_poll()` |
| `barracuda/src/bio/phage_defense_gpu.rs` | Migrated to `submit_and_poll()` |
| `barracuda/src/bio/gemm_cached.rs` | Migrated to `submit_and_poll()` |
| `barracuda/src/bin/benchmark_cross_spring_modern_s68plus.rs` | New: Exp210 |
| `barracuda/Cargo.toml` | Added Exp210 binary entry |
| `experiments/210_cross_spring_modern_s68plus.md` | New: experiment doc |
| `CHANGELOG.md` | Added V64 entry |
| `barracuda/EVOLUTION_READINESS.md` | Updated to V64 |
| `wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md` | Updated to V64 |
