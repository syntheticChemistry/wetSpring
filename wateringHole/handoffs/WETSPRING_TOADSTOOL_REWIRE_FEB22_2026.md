# wetSpring → ToadStool: Bio Primitive Rewire Results

**Date:** 2026-02-22
**From:** wetSpring (ecoPrimals — Life Science & Analytical Chemistry)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Supersedes:** Central handoff `WETSPRING_TOADSTOOL_TIER_A_SHADERS_FEB21_2026.md`
  (shader specs remain valid reference; this documents rewire outcomes)
**ToadStool sessions:** 31d + 31g (bio absorption)

---

## Executive Summary

On Feb 22, wetSpring rewired all 8 bio GPU modules to delegate to
ToadStool `barracuda::ops::bio::*` primitives, deleted 8 local WGSL
shaders (25 KB), and confirmed all 451 GPU validation checks pass.
Two ToadStool bugs were found and fixed during the process. 23 ToadStool
primitives are now consumed (up from 15). 4 local WGSL shaders in Write phase
(ODE, kmer, unifrac, taxonomy; ODE blocked on `enable f64;`).

---

## 1. Modules Rewired

| wetSpring Module | ToadStool Primitive | Shader Deleted | Checks |
|-----------------|---------------------|----------------|:------:|
| `hmm_gpu` | `ops::bio::hmm::HmmBatchForwardF64` | `hmm_forward_f64.wgsl` | 13 |
| `ani_gpu` | `ops::bio::ani::AniBatchF64` | `ani_batch_f64.wgsl` | 27 |
| `snp_gpu` | `ops::bio::snp::SnpCallingF64` | `snp_calling_f64.wgsl` | 27 |
| `dnds_gpu` | `ops::bio::dnds::DnDsBatchF64` | `dnds_batch_f64.wgsl` | 27 |
| `pangenome_gpu` | `ops::bio::pangenome::PangenomeClassifyGpu` | `pangenome_classify.wgsl` | 27 |
| `quality_gpu` | `ops::bio::quality::QualityFilterGpu` | `quality_filter.wgsl` | 88 |
| `dada2_gpu` | `ops::bio::dada2::Dada2EStepGpu` | `dada2_e_step.wgsl` | 88 |
| `random_forest_gpu` | `ops::bio::rf::RfBatchInferenceGpu` | `rf_batch_inference.wgsl` | 13 |

All modules now hold `inner: ToadStoolPrimitive` and delegate GPU dispatch.
Constructors return `Result<Self>`.

---

## 2. Bugs Found and Fixed

### Bug 1: SNP Binding Layout Mismatch

**Location:** `toadstool/crates/barracuda/src/ops/bio/snp.rs`
**Symptom:** `wgpu::ValidationError` — storage class mismatch for `is_variant`
**Root cause:** `make_bgl()` called with `&[true, true, false, false, false, false]`.
Binding 1 (`is_variant`) was marked read-only, but the WGSL shader declares
`var<storage, read_write>`. The array also contained a phantom 6th binding
with no shader counterpart.
**Fix:** Changed to `&[true, false, false, false, false]`.

```rust
// Before (incorrect):
let bgl = make_bgl(&device, &[true, true, false, false, false, false]);

// After (correct):
let bgl = make_bgl(&device, &[true, false, false, false, false]);
```

**Impact:** SNP shader would fail on any consumer invoking `SnpCallingF64`.

### Bug 2: AdapterInfo Propagation Failure

**Location:** `wetSpring/barracuda/src/gpu.rs`
**Symptom:** NVVM compilation error — `ptxas` fails on f64 `exp`/`log` for
RTX 4070 (Ada Lovelace, sm_89)
**Root cause:** wetSpring's `GpuF64::new()` used `WgpuDevice::from_existing_simple()`
which sets `adapter_info.name = "External Device"`. ToadStool's
`needs_f64_exp_log_workaround()` checks the adapter name for "RTX 40" and
"Ada" — the synthetic name bypassed detection, so f64 polyfills were not
injected.
**Fix:** Use `WgpuDevice::from_existing(device, queue, info)` with the real
`wgpu::AdapterInfo`.

**Recommendation for ToadStool:** Consider logging a warning when
`from_existing_simple()` is used, since it silently disables all
driver-specific workarounds. Or deprecate it in favor of requiring
explicit `AdapterInfo`.

---

## 3. Remaining Local Shader

| Shader | Blocker |
|--------|---------|
| `batched_qs_ode_rk4_f64.wgsl` | ToadStool's upstream `BatchedOdeRK4F64` shader contains `enable f64;` (line 35), which causes Naga compilation failure on many drivers. wetSpring's local copy avoids this directive by using `compile_shader_f64()` (math_f64 preamble injection). |

**Recommended fix:** Remove `enable f64;` from ToadStool's ODE shader and
rely on `compile_shader_f64()` preamble injection, matching the pattern
used by all other bio shaders.

---

## 4. Validation Results

All 451 GPU validation checks pass after rewire:

| Binary | Experiment | Checks | Result |
|--------|:----------:|:------:|--------|
| `validate_gpu_hmm_forward` | 047 | 13/13 | PASS |
| `validate_gpu_track1c` | 058 | 27/27 | PASS |
| `validate_gpu_rf` | 063 | 13/13 | PASS |
| `validate_16s_pipeline_gpu` | 016 | 88/88 | PASS |
| `validate_barracuda_gpu_v1` | 064 | 26/26 | PASS |
| `validate_metalforge_full` | 065 | 35/35 | PASS |
| `validate_gpu_streaming_pipeline` | 072 | 17/17 | PASS |
| `validate_cross_substrate` | 060 | 20/20 | PASS |
| `validate_barracuda_gpu_full` | 071 | 24/24 | PASS |
| `validate_gpu_ode_sweep` | 049 | 12/12 | PASS |

Quality gate: `cargo fmt`, `cargo clippy --pedantic`, `cargo test --lib` (633 passed),
`cargo doc --no-deps` — all clean, 0 warnings.

---

## 5. Benchmark Observations

| Benchmark | Finding | Implication for ToadStool |
|-----------|---------|--------------------------|
| HMM batch (256 seqs × 200 obs) | 0.46× GPU/CPU — dispatch-dominated | Small-N bio workloads need batching or dispatch amortization |
| Dispatch overhead | 4.2ms average per domain | Pipeline caching (`GpuPipelineSession`) is correct default |
| dN/dS crossover | GPU wins at N≈256+ | Consider auto-dispatch threshold in bio primitives |

---

## 6. Cross-Spring Evolution Story

ToadStool serves as convergence hub for three Springs:

```
hotSpring (physics) ──→ barracuda ←── wetSpring (bio)
                            ↑
                      neuralSpring (ML)
```

### What Each Spring Contributed (Absorbed by ToadStool)

| Spring | Contributions | Date | Cross-Spring Beneficiaries |
|--------|--------------|------|---------------------------|
| **wetSpring** | 8 bio WGSL shaders (HMM, ANI, SNP, dN/dS, pangenome, QF, DADA2, RF) | Feb 22 | neuralSpring (HMM for sequence models, RF for ensemble) |
| **wetSpring** | `log_f64` precision fix (~1e-3 → ~1e-15), `(zero+literal)` f64 constant pattern | Feb 16 | hotSpring (BCS bisection convergence), all Springs |
| **wetSpring** | Bray-Curtis f64, Shannon/Simpson FMR, Hill kinetics | Feb 16 | ecology/diversity in any Spring |
| **hotSpring** | `complex_f64.wgsl`, SU(3), lattice QCD, CG solver, FFT f64 | Feb 19 | any Spring doing field theory or transforms |
| **hotSpring** | NVK eigensolve profiling, warp-packed strategy, `GpuDriverProfile` | Feb 17–18 | all Springs on NVK/nouveau drivers |
| **hotSpring** | Spectral theory (Anderson, Lanczos, CSR, Hofstadter) | Feb 21–22 | wetSpring (large covariance), neuralSpring (sparse ops) |
| **neuralSpring** | TensorSession, matmul router, activations, BatchIprGpu | Feb 20 | hotSpring (Anderson IPR), wetSpring (taxonomy matmul) |

### Concrete Cross-Spring Effects Observed in This Rewire

1. **hotSpring → wetSpring**: f64 polyfill detection (Ada Lovelace workaround)
   correctly protects wetSpring's rewired bio shaders on RTX 4070.
2. **neuralSpring → wetSpring**: `BatchedEighGpu` continues to power PCoA
   eigendecomposition and bifurcation analysis.
3. **wetSpring → all**: Bio primitives now available at `barracuda::ops::bio::*`
   for any Spring that needs HMM, alignment, or phylogenetics.

---

## 7. ToadStool Primitives Now Consumed by wetSpring (23)

| # | Primitive | Origin | Absorbed |
|---|-----------|--------|----------|
| 1 | `FusedMapReduceF64` (Shannon) | barracuda core | Feb 16 |
| 2 | `FusedMapReduceF64` (Simpson) | barracuda core | Feb 16 |
| 3 | `FusedMapReduceF64` (observed) | barracuda core | Feb 16 |
| 4 | `BrayCurtisF64` | wetSpring | Feb 16 |
| 5 | `GemmF64` | barracuda core | Feb 16 |
| 6 | `BatchedEighGpu` | barracuda core | Feb 16 |
| 7 | `KrigingF64` | barracuda core | Feb 16 |
| 8 | `VarianceF64` | barracuda core | Feb 16 |
| 9 | `CorrelationF64` | barracuda core | Feb 16 |
| 10 | `CovarianceF64` | barracuda core | Feb 16 |
| 11 | `WeightedDotF64` | barracuda core | Feb 16 |
| 12 | `PrngXoshiro` | barracuda core | Feb 16 |
| 13 | `SmithWatermanGpu` | wetSpring | Feb 20 |
| 14 | `GillespieGpu` | wetSpring | Feb 20 |
| 15 | `TreeInferenceGpu` | wetSpring | Feb 20 |
| 16 | `HmmBatchForwardF64` | wetSpring | Feb 22 |
| 17 | `AniBatchF64` | wetSpring | Feb 22 |
| 18 | `SnpCallingF64` | wetSpring | Feb 22 |
| 19 | `DnDsBatchF64` | wetSpring | Feb 22 |
| 20 | `PangenomeClassifyGpu` | wetSpring | Feb 22 |
| 21 | `QualityFilterGpu` | wetSpring | Feb 22 |
| 22 | `Dada2EStepGpu` | wetSpring | Feb 22 |
| 23 | `RfBatchInferenceGpu` | wetSpring | Feb 22 |

---

## 8. Recommendations for ToadStool Team

### Priority 1: Fix ODE shader `enable f64;`
Remove `enable f64;` from `batched_qs_ode_rk4_f64.wgsl` line 35.
Use `compile_shader_f64()` preamble injection instead. This is the only
blocker preventing wetSpring from fully leaning on ToadStool.

### Priority 2: Deprecate `from_existing_simple()`
This method silently creates `AdapterInfo` with name `"External Device"`,
disabling all driver-specific workarounds. Either:
- Deprecate it with a warning
- Add `#[deprecated]` attribute
- Log a runtime warning when workaround detection is bypassed

### Priority 3: Auto-dispatch threshold for bio primitives
Small-N bio workloads (e.g., HMM at N=256) are dispatch-dominated.
Consider adding optional batch-size thresholds to bio primitive constructors
that auto-fall-back to CPU below the crossover point.

### Priority 4: CPU math feature
wetSpring has 4 local math functions (`erf`, `ln_gamma`, `regularized_gamma`,
`trapz`) that duplicate `barracuda::special`/`numerical`. A `barracuda::math`
feature (CPU-only, no wgpu dependency) would let wetSpring remove these.

### Priority 5: Audit other Springs for `from_existing_simple()` usage
If hotSpring or neuralSpring also use `from_existing_simple()`, they may
silently lack f64 workarounds on their hardware.

---

## 9. Evolution Timeline (wetSpring)

```
Feb 16 — wetSpring: log_f64 precision fix → math_f64.wgsl (all Springs benefit)
Feb 16 — wetSpring: Bray-Curtis, Shannon/Simpson FMR → barracuda
Feb 20 — wetSpring: SmithWaterman, Gillespie, TreeInference, Felsenstein → barracuda
Feb 21 — wetSpring: handoff v6 with 9 WGSL shader specs, NVVM driver findings
Feb 22 — ToadStool: sessions 31d+31g absorb 8 bio shaders
Feb 22 — wetSpring: rewires 8 GPU modules to barracuda::ops::bio::*
Feb 22 — wetSpring: deletes 8 local WGSL shaders (25 KB)
Feb 22 — wetSpring: discovers SNP binding bug → fixed in ToadStool
Feb 22 — wetSpring: discovers AdapterInfo propagation failure → fixed locally
Feb 22 — wetSpring: full revalidation (451 GPU checks PASS) + benchmarks
Feb 22 — wetSpring: cross-spring evolution documented
```

---

*License: AGPL-3.0-or-later. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required.*
