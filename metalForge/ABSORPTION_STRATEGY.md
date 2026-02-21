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

wetSpring consolidated local math into `bio::special` (erf, ln_gamma,
`regularized_gamma_lower`) during Phase 15. These functions duplicate
barracuda upstream primitives and are shaped for extraction:

| Local Implementation | File | BarraCUDA Primitive | Status |
|---------------------|------|---------------------|--------|
| `erf()`, `normal_cdf()` | `bio/special.rs` | `barracuda::special::erf` | Consolidated; uses `mul_add` chains |
| `ln_gamma()` | `bio/special.rs` | `barracuda::special::ln_gamma` | Lanczos approximation, Horner form |
| `regularized_gamma_lower()` | `bio/special.rs` | `barracuda::special::regularized_gamma_p` | Series expansion, 1e-15 convergence |
| `integrate_peak()` | `bio/eic.rs` | `barracuda::numerical::trapz` | Trapezoidal integration |

### Extraction Plan

The `bio::special` module is designed for clean extraction:

1. **No internal dependencies** — does not import any other `bio::*` module
2. **Pure math** — no I/O, no allocation, no state
3. **`mul_add` optimized** — FMA-friendly for both CPU and GPU
4. **Test coverage** — known-value tests with documented precision bounds
5. **Consumers**: `bio::dada2` (via `regularized_gamma_lower`), `bio::pangenome` (via `normal_cdf`)

When barracuda adds `[features] math = []`, the migration is:
```
bio::special::erf              → barracuda::special::erf
bio::special::ln_gamma         → barracuda::special::ln_gamma
bio::special::regularized_gamma_lower → barracuda::special::regularized_gamma_p
bio::eic::integrate_peak       → barracuda::numerical::trapz
```

**Why not now?** `barracuda` currently requires wgpu + akida-driver + toadstool-core
as mandatory dependencies. Importing it for CPU math would force GPU/NPU stack into
all builds. hotSpring accepts this (always-GPU). wetSpring keeps barracuda optional
(`--features gpu`) until barracuda adds a granular `math` feature.

**Proposal for ToadStool:** Add `[features] math = []` to barracuda crate that
gates `numerical`, `special`, `stats`, `optimize`, `sample` modules without
pulling in wgpu/akida/toadstool-core. This enables Springs to lean on shared
CPU math without forcing GPU stack.

## Absorption Engineering Patterns

Rust modules shaped for ToadStool absorption follow these patterns from
hotSpring's proven methodology:

| Pattern | Why | wetSpring Example |
|---------|-----|-------------------|
| Flat arrays (SoA) | GPU buffers are 1D; no pointers | `FlatTree` in `bio::felsenstein` |
| `#[repr(C)]` params | WGSL uniform/storage layout matches Rust | `GemmParams`, `FelsensteinParams` |
| Batch APIs | Single dispatch for N instances | `forward_batch()`, `score_batch()` |
| Preallocated buffers | No alloc in hot path; reuse via `BufferPool` | `GemmCached` pipeline |
| Deterministic math | Same input → same output (bit-exact f64) | All `bio::*` modules |
| `mul_add` chains | FMA-friendly for both CPU SIMD and GPU | `bio::special::erf` polynomial |
| Named tolerances | Central constants, not magic numbers | 32 constants in `tolerances.rs` |
| Provenance headers | Script, commit, command, hardware | All 73 validation binaries |

### What Makes Code Absorbable (Lessons from hotSpring)

1. WGSL templates in Rust source (or structured `.wgsl` files), not opaque blobs
2. Binding layout docs: binding index, type, purpose
3. Dispatch docs: workgroup size, grid size
4. CPU reference validated against known values (Python baselines)
5. Tolerances in `tolerances.rs`, not ad-hoc constants
6. Handoff document with locations and validation results
7. Zero unsafe code — `#![forbid(unsafe_code)]` enforced

---

## Dispatch Overhead Data (Exp067/068)

Pipeline caching reduced dispatch overhead by 38% (Exp068). These numbers
inform ToadStool's absorption priorities — shaders with highest overhead
benefit most from upstream pipeline management:

| Shader | Before Cache | After Cache | Remaining Overhead |
|--------|:----------:|:---------:|:-----------------:|
| `ani_batch_f64.wgsl` | 5,855µs | 5,398µs | Buffer alloc + readback |
| `snp_calling_f64.wgsl` | 10,169µs | 6,384µs | Large flat buffer |
| `dnds_batch_f64.wgsl` | 9,900µs | 4,703µs | Polyfill + genetic code table |
| `pangenome_classify.wgsl` | 5,788µs | 3,043µs | Presence matrix upload |
| `rf_batch_inference.wgsl` | 2,493µs | 1,600µs | SoA tree upload |
| `hmm_forward_f64.wgsl` | 5,086µs | 3,229µs | Model params + alpha buffer |

ToadStool streaming (1 upload + N dispatches + 1 readback) would reduce
the remaining ~3ms average to ~0.5ms by amortizing buffer operations across
chained stages.

### Streaming Pipeline Proof (Exp072-076)

| Exp | What | Result |
|-----|------|--------|
| 072 | Pre-warmed FMR vs individual dispatch | 1.27x speedup, 5µs vs 110ms first call |
| 073 | Dispatch overhead at [64, 256, 1K, 4K] | Streaming beats individual at all sizes |
| 074 | Substrate router (GPU↔NPU↔CPU) | Correct routing, fallback parity |
| 075 | Pure GPU 5-stage pipeline | 0.1% overhead, 31 checks PASS |
| 076 | Cross-substrate GPU→NPU→CPU | Latency profiled, 17 checks PASS |

These experiments validate the full ToadStool dispatch model: pre-compile
pipelines once, share buffer pools, chain stages without CPU round-trips,
and route workloads to the optimal substrate.

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
