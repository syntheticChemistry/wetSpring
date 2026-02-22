# Exp077: ToadStool Bio Primitive Rewire

**Date:** February 22, 2026
**Status:** COMPLETE
**Binary:** All existing GPU validation binaries (no new binary)
**Tracks:** GPU, cross

---

## Objective

Rewire all 8 wetSpring GPU bio modules from local WGSL shaders to
ToadStool `barracuda::ops::bio::*` absorbed primitives, validate that
all existing GPU checks continue to pass, and delete local shader code.

## Background

ToadStool sessions 31d and 31g absorbed 8 wetSpring bio WGSL shaders
(HMM, ANI, SNP, dN/dS, Pangenome, Quality Filter, DADA2, RF) from
the handoff submitted on Feb 21. This experiment validates the rewire
by running the full GPU validation suite against the ToadStool-backed
implementations.

## Protocol

### Step 1: Rewire GPU modules

Replace local shader compilation (`include_str!`, `ShaderTemplate`,
`compile_shader_f64`) with delegation to ToadStool primitives:

| wetSpring Module | ToadStool Primitive | Local Shader (deleted) |
|-----------------|-------------------|----------------------|
| `hmm_gpu` | `HmmBatchForwardF64` | `hmm_forward_f64.wgsl` |
| `ani_gpu` | `AniBatchF64` | `ani_batch_f64.wgsl` |
| `snp_gpu` | `SnpCallingF64` | `snp_calling_f64.wgsl` |
| `dnds_gpu` | `DnDsBatchF64` | `dnds_batch_f64.wgsl` |
| `pangenome_gpu` | `PangenomeClassifyGpu` | `pangenome_classify.wgsl` |
| `quality_gpu` | `QualityFilterGpu` | `quality_filter.wgsl` |
| `dada2_gpu` | `Dada2EStepGpu` | `dada2_e_step.wgsl` |
| `random_forest_gpu` | `RfBatchInferenceGpu` | `rf_batch_inference.wgsl` |

### Step 2: Fix ToadStool bugs discovered during validation

1. **SNP binding layout** — `is_variant` (binding 2) was marked `read_only`
   in ToadStool's `snp.rs` but the shader declares `read_write`. Also had
   a phantom 6th storage binding. Fixed: `&[true, false, false, false, false]`.

2. **AdapterInfo propagation** — wetSpring's `GpuF64::new()` used
   `WgpuDevice::from_existing_simple()` which sets `adapter_info.name =
   "External Device"`, breaking ToadStool's RTX 4070 Ada Lovelace detection
   and f64 exp/log polyfill. Fixed: use `WgpuDevice::from_existing()` with
   real adapter info.

### Step 3: Validate all GPU binaries

| Binary | Checks | Result |
|--------|:------:|--------|
| `validate_gpu_hmm_forward` (Exp047) | 13/13 | PASS |
| `validate_gpu_track1c` (Exp058) | 27/27 | PASS |
| `validate_gpu_rf` (Exp063) | 13/13 | PASS |
| `validate_16s_pipeline_gpu` (Exp016) | 88/88 | PASS |
| `validate_barracuda_gpu_v1` (Exp064) | 26/26 | PASS |
| `validate_metalforge_full` (Exp065) | 35/35 | PASS |
| `validate_gpu_streaming_pipeline` (Exp072) | 17/17 | PASS |
| `validate_cross_substrate` (Exp060) | 20/20 | PASS |
| `validate_barracuda_gpu_full` (Exp071) | 24/24 | PASS |
| `validate_gpu_ode_sweep` (Exp049-050) | 12/12 | PASS |

### Step 4: Benchmark modern ToadStool shaders

| Benchmark | Key Finding |
|-----------|-------------|
| `benchmark_phylo_hmm_gpu` | HMM batch: 0.46× at 256×200obs (dispatch-dominated) |
| `benchmark_dispatch_overhead` | Average 4.2ms fixed dispatch cost per domain |
| `benchmark_all_domains_cpu_gpu` | dN/dS crosses over at N≈256; others GPU-friendly at large N |

### Step 5: Quality gate

- `cargo fmt --check` — clean
- `cargo clippy --pedantic` — 0 warnings
- `cargo test --lib` — 633 passed, 0 failed
- `cargo doc --no-deps` — clean

## Results

All 451 GPU validation checks pass with ToadStool-backed primitives.
8 local WGSL shaders deleted (25 KB). 23 ToadStool primitives consumed
(up from 15). Only 1 local shader remains (ODE, blocked).

## Cross-Spring Evolution

The rewire validates that ToadStool's cross-spring absorption model works:
- hotSpring precision shaders (f64 polyfills) correctly protect wetSpring's
  bio shaders on RTX 4070 Ada Lovelace
- neuralSpring's eigensolver (`BatchedEighGpu`) continues to power wetSpring's
  PCoA and bifurcation analysis
- wetSpring's bio primitives are now available to all springs via
  `barracuda::ops::bio::*`
