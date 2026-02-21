# Absorption Strategy: wetSpring → ToadStool

**Date:** February 21, 2026
**Pattern:** Write → Absorb → Lean (from hotSpring)
**Status:** 5 absorbed, 9 Tier A handoff candidates, 4 barracuda CPU math opportunities

---

## How wetSpring Writes Absorbable Extensions

### 1. GPU-Friendly Rust First

Before writing any WGSL, the Rust CPU module must be shaped for absorption:

| Pattern | Why | Example |
|---------|-----|---------|
| Flat arrays (SoA) | GPU buffers are 1D; no pointers, no Box/Vec nesting | `FlatTree` in `bio::felsenstein` |
| `#[repr(C)]` params | WGSL uniform/storage layout matches Rust struct | `GemmParams`, `FelsensteinParams` |
| Batch APIs | Single dispatch for N instances | `forward_batch()`, `score_batch()`, `align_batch()` |
| Preallocated buffers | No alloc in hot path; reuse via `BufferPool` | `GemmCached` pipeline pattern |
| Deterministic math | Same input → same output (bit-exact f64) | All `bio::*` modules |

### 2. Local WGSL Development

Write the shader locally, validate, then hand off.

**File placement:**
```
barracuda/src/shaders/<domain>_<op>_f64.wgsl
```

**Naming convention:**
- `_f64` suffix for f64 shaders
- Domain prefix: `felsenstein_`, `sw_`, `gillespie_`, `hmm_`
- Entry point: `@compute @workgroup_size(64) fn <name>_f64(...)`

**Include strategy:**
```rust
const SHADER_SRC: &str = include_str!("../shaders/hmm_forward_f64.wgsl");
```

Or for small shaders, inline as `pub const`:
```rust
pub const WGSL_HMM_FORWARD: &str = "
@group(0) @binding(0) var<storage, read> obs: array<u32>;
...
";
```

### 3. Validation Before Handoff

Every local shader must pass before handoff:

| Gate | Tool | Threshold |
|------|------|-----------|
| CPU parity | `validate_<domain>.rs` | GPU == CPU within `GPU_VS_CPU_*` tolerance |
| Determinism | Run 3x, compare bit-exact | Zero differences |
| Edge cases | Empty input, single element, max size | No panics |
| Clippy | `cargo clippy --features gpu` | Zero warnings in new code |

### 4. Handoff Format

```markdown
# Handoff: wetSpring → ToadStool

## Shader: <name>_f64.wgsl
- Source: barracuda/src/shaders/<name>_f64.wgsl
- Binding layout: [group, binding, type, size]
- Dispatch geometry: [x, y, z] formula
- CPU reference: bio::<module>::<function>
- Validation: Exp<N>, <K>/K checks PASS
- Tolerances: <list with rationale>
```

---

## Current Absorption Ledger

### Absorbed into ToadStool (crate-level re-exports)

| Primitive | wetSpring Source | ToadStool Module | Handoff Date |
|-----------|-----------------|------------------|-------------|
| `SmithWatermanGpu` | `bio::alignment` (Exp028) | `ops::bio::smith_waterman` | Feb 20 |
| `GillespieGpu` | `bio::gillespie` (Exp022) | `ops::bio::gillespie` | Feb 20 |
| `TreeInferenceGpu` | `bio::decision_tree` (Exp008) | `ops::bio::tree_inference` | Feb 20 |
| `FelsensteinGpu` | `bio::felsenstein` (Exp029) | `ops::bio::felsenstein` | Feb 20 |
| `GemmF64::WGSL` | `bio::gemm_cached` | `ops::linalg::gemm_f64` | Feb 20 |

### Local WGSL Shaders (Tier A — handoff candidates)

| Shader | File | GPU Checks | Absorption Path |
|--------|------|:----------:|-----------------|
| `dada2_e_step.wgsl` | `barracuda/src/shaders/` | 88 (pipeline) | New `BatchPairReduce<f64>` |
| `quality_filter.wgsl` | `barracuda/src/shaders/` | 88 (pipeline) | New `ParallelFilter<T>` |
| `hmm_forward_f64.wgsl` | `barracuda/src/shaders/` | 13 (Exp047) | New `HmmBatchForwardF64` |
| `batched_qs_ode_rk4_f64.wgsl` | `barracuda/src/shaders/` | 7 (Exp049) | Fix upstream `BatchedOdeRK4F64` |
| `ani_batch_f64.wgsl` | `barracuda/src/shaders/` | 7 (Exp058) | New `AniBatchF64` |
| `snp_calling_f64.wgsl` | `barracuda/src/shaders/` | 5 (Exp058) | New `SnpCallingF64` |
| `dnds_batch_f64.wgsl` | `barracuda/src/shaders/` | 9 (Exp058) | New `DnDsBatchF64` |
| `pangenome_classify.wgsl` | `barracuda/src/shaders/` | 6 (Exp058) | New `PangenomeClassifyGpu` |
| `rf_batch_inference.wgsl` | `barracuda/src/shaders/` | 13 (Exp063) | New `RfBatchInferenceGpu` |

### Completed Compositions (were candidates, now validated)

| Algorithm | Composed From | Exp | Checks |
|-----------|-------------|-----|:------:|
| Bootstrap resampling | `FelsensteinGpu` per replicate | 046 | 15 |
| Phylogenetic placement | `FelsensteinGpu` per edge | 046 | 15 |
| ODE parameter sweep | Local WGSL + `for_driver_auto` | 049 | 7 |
| Bifurcation eigenvalues | CPU Jacobian + `BatchedEighGpu` | 050 | 5 |

### Track 1c — GPU-Promoted (Exp058, Feb 21, 2026)

| Algorithm | Shader | GPU Checks | Notes |
|-----------|--------|:----------:|-------|
| ANI pairwise | `ani_batch_f64.wgsl` | **7/7** | One thread per pair, integer counting + f64 division |
| SNP calling | `snp_calling_f64.wgsl` | **5/5** | One thread per position, allele frequency |
| dN/dS (Nei-Gojobori) | `dnds_batch_f64.wgsl` | **9/9** | One thread per pair, genetic code table, `log()` polyfill |
| Pangenome classify | `pangenome_classify.wgsl` | **6/6** | One thread per gene, presence counting |
| Molecular clock | CPU-only | — | Sequential tree traversal, too small for GPU |

All 4 WGSL shaders validated against CPU baselines in `validate_gpu_track1c`.
Total: 27/27 GPU checks PASS. These are ToadStool absorption candidates.

### ML Ensembles — GPU-Promoted (Exp063, Feb 20, 2026)

| Algorithm | Shader | GPU Checks | Notes |
|-----------|--------|:----------:|-------|
| Random Forest | `rf_batch_inference.wgsl` | **13/13** | One thread per (sample, tree) pair, SoA layout |
| GBM | CPU-only | — | Sequential boosting across rounds |

RF batch inference validated against CPU in `validate_gpu_rf`.
SoA layout: separate node_features, node_thresholds (f64), node_children buffers.

### Remaining Shader Candidates (Original)

| Algorithm | Shader Design | Priority | Status |
|-----------|--------------|----------|--------|
| K-mer counting | GPU hash table (lock-free) | P3 | Blocked |
| UniFrac traversal | GPU tree traversal primitive | P3 | Blocked |
| Taxonomy FC model | Naive Bayes → int8 FC → NPU | P3 | NPU candidate |

---

## BarraCUDA CPU Math Evolution

wetSpring maintains local implementations of 4 math functions that barracuda
already provides. These should switch to barracuda when a `math` feature
(CPU-only, no wgpu) is available:

| Local Implementation | File | Lines | BarraCUDA Primitive | Impact |
|---------------------|------|:-----:|---------------------|--------|
| `erf()`, `normal_cdf()` | `bio/pangenome.rs` | 218–236 | `barracuda::special::erf` | Pangenome FDR |
| `ln_gamma()` | `bio/dada2.rs` | 431–453 | `barracuda::special::ln_gamma` | DADA2 Poisson p-value |
| `regularized_gamma_lower()` | `bio/dada2.rs` | 399–427 | `barracuda::special::regularized_gamma_p` | DADA2 Poisson p-value |
| `integrate_peak()` | `bio/eic.rs` | 165–176 | `barracuda::numerical::trapz` | LC-MS peak area |

**Why not now?** `barracuda` currently requires wgpu + akida-driver + toadstool-core
as mandatory dependencies. Importing it for CPU math would force GPU/NPU stack into
all builds. hotSpring accepts this (always-GPU). wetSpring keeps barracuda optional
(`--features gpu`) until barracuda adds a granular `math` feature.

**Proposal for ToadStool:** Add `[features] math = []` to barracuda crate that
gates `numerical`, `special`, `stats`, `optimize`, `sample` modules without
pulling in wgpu/akida/toadstool-core. This enables Springs to lean on shared
CPU math without forcing GPU stack.

---

## Lessons from hotSpring

| Lesson | How We Apply It |
|--------|----------------|
| Local shaders get absorbed faster than design docs | Write WGSL first, doc second |
| `pub const WGSL: &str` enables clean imports | Use for small shaders; `include_str!` for large |
| `shaders_toadstool_ref/` for divergence tracking | Copy upstream shader at absorption time; diff when upgrading |
| Validation binary per shader | `validate_toadstool_bio.rs` pattern — `catch_unwind` for driver panics |
| Driver-level failures are real | Gillespie SSA NVVM failure on RTX 4070; design for graceful skip |
| `compile_shader_f64` not `compile_shader` | All f64 WGSL must use `ShaderTemplate::for_driver_auto` |
| Force polyfills for Ada Lovelace | `for_driver_auto(source, true)` bypasses broken driver profile |
| naga rejects `enable f64;` | Omit from all WGSL; `ShaderTemplate` handles f64 enablement |
| Explicit `f64()` casts required | `max(x, f64(0.0))` not `max(x, 0.0)` — naga type promotion bug |
| `EVOLUTION_READINESS.md` tracks everything | Adopted from hotSpring; lives alongside code in `barracuda/` |
