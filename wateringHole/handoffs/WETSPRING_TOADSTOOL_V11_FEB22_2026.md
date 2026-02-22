# wetSpring → ToadStool/BarraCUDA Handoff v11

**Date:** February 22, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Context:** Phase 23 — Structural evolution complete, flat layouts + DRY models + zero-clone APIs

---

## Executive Summary

wetSpring has completed **97 experiments, 2,229+ validation checks, 728 Rust
tests, and 87 binaries** — all passing. This handoff covers the structural
evolution work (Exp097) and its implications for ToadStool absorption.

**New in v11:**

1. **All trajectory types flattened** — ODE (`Vec<f64>` + `n_vars`), Gillespie
   (`Vec<i64>` + `n_species`), PCoA (`Vec<f64>` + `n_samples × n_axes`) now use
   contiguous memory. These layouts align directly with GPU buffer binding.
2. **DADA2 error model unified** — 5 functions shared from `dada2.rs` as single
   source of truth. GPU module delegates. Ready for extraction to
   `barracuda::ops::bio::dada2::ErrorModel`.
3. **UniFrac distance matrix now condensed** — returns upper-triangle `Vec<f64>`
   instead of N×N `Vec<Vec<f64>>`. Halves memory, matches `pcoa()` condensed input
   format directly. No conversion needed in pipelines.
4. **Zero-clone APIs throughout** — adapter trim returns `Option`, ODE/Gillespie
   use `extend_from_slice` instead of per-step clones, ODE closures borrow params
   instead of cloning.
5. **Full audit clean** — zero unsafe, zero TODO/FIXME, zero cross-primal coupling,
   zero `unimplemented!()`, zero production mocks.

---

## Part 1: What ToadStool Should Absorb

### 1.1 Flat Data Layout Patterns

wetSpring's flat layouts are directly GPU-buffer-compatible. When ToadStool
absorbs these bio primitives, the flat formats eliminate host-side conversion:

| Structure | Layout | Access API | GPU Binding |
|-----------|--------|-----------|-------------|
| `OdeResult` | `Vec<f64>`, row-major `[y0_t0, y1_t0, ..., y0_t1, ...]` | `state_at(i)`, `var_at(i, j)`, `states()` | Single `storage<read>` buffer, stride = `n_vars` |
| `Trajectory` | `Vec<i64>`, row-major `[s0_e0, s1_e0, ..., s0_e1, ...]` | `state_at(i)`, `states_iter()`, `final_state()` | Single `storage<read>` buffer, stride = `n_species` |
| `PcoaResult` | `Vec<f64>`, row-major `[ax0_s0, ax1_s0, ..., ax0_s1, ...]` | `coord(sample, axis)`, `sample_coords(sample)` | Single `storage<read>` buffer, stride = `n_axes` |
| `UnifracDistanceMatrix` | Condensed `Vec<f64>`, lower-triangle order | `condensed_index(i, j)` from `diversity.rs` | Single `storage<read>` buffer, `N*(N-1)/2` elements |

**Recommendation:** When upstream `barracuda::ops::bio::ode_rk4` absorbs the ODE
sweep shader, use this exact flat layout. The stride metadata (`n_vars`, `n_species`,
`n_samples`, `n_axes`) should be passed as uniform parameters in the binding layout.

### 1.2 Unified DADA2 Error Model

The DADA2 error model functions are now `pub(crate)` in `dada2.rs`:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `init_error_model()` | `→ ErrorModel` | Phred-based initial error rates |
| `estimate_error_model(seqs, partition, centers)` | `→ ErrorModel` | EM re-estimation from observed substitutions |
| `err_model_converged(old, new)` | `→ bool` | Max-diff convergence check (threshold: `DADA2_ERR_CONVERGENCE`) |
| `base_to_idx(b: u8)` | `→ usize` | ACGT → 0123 with IUPAC fallback |

Constants: `NUM_BASES=4`, `MAX_QUAL=42`, `MIN_ERR=1e-7`, `MAX_ERR=0.25`, `MAX_ERR_ITERS=6`.

**Type alias:** `ErrorModel = [[[f64; MAX_QUAL]; NUM_BASES]; NUM_BASES]` — 3D array
that flattens to `NUM_BASES * NUM_BASES * MAX_QUAL` f64 values for GPU upload.
The `flatten_log_error_model()` function in `dada2_gpu.rs` already does this.

**Recommendation:** When `barracuda::ops::bio::dada2` evolves, extract these
functions into a shared `dada2::error` module that both CPU and GPU paths consume.

### 1.3 Remaining Write-Phase Shaders (unchanged from v10)

| Shader | Domain | Blocker |
|--------|--------|---------|
| `batched_qs_ode_rk4_f64.wgsl` | ODE parameter sweep | `compile_shader` needs `compile_shader_f64` variant |
| `kmer_histogram_f64.wgsl` | K-mer counting | Ready for absorption |
| `unifrac_propagate_f64.wgsl` | UniFrac tree propagation | Ready for absorption |
| `taxonomy_fc_f64.wgsl` | Taxonomy scoring | Ready for absorption (NPU candidate) |

---

## Part 2: What wetSpring Learned (Relevant to ToadStool Evolution)

### 2.1 Vec<Vec<>> is the #1 Anti-Pattern for GPU Promotion

Every `Vec<Vec<T>>` in Rust becomes a pointer-chasing indirection on GPU.
wetSpring found 15+ instances in lib code. The fix is always the same:
flat `Vec<T>` + stride metadata + accessor methods. This should be a
ToadStool design guideline: **all batch-oriented ops should use flat layouts
from the start**, not after refactoring.

### 2.2 Clone Elimination Yields Better GPU Buffer Reuse

Rust's `.clone()` on trajectory data (ODE, SSA) allocates new vectors every
step. With flat layouts, `extend_from_slice()` appends to a single pre-allocated
buffer. This directly maps to GPU buffer growth patterns (append to staging
buffer, upload once). ToadStool's streaming pipeline already does this — Springs
should match the pattern on the CPU side.

### 2.3 DRY Error Models Prevent CPU/GPU Drift

DADA2's CPU and GPU implementations had identical error model functions that
could have silently diverged. The fix (GPU delegates to CPU) ensures the
convergence threshold comes from `tolerances.rs` consistently. **ToadStool
should enforce this pattern**: CPU reference functions should be the single
source, GPU code should import or delegate.

### 2.4 Condensed Distance Matrices Save 50% Memory

Symmetric N×N distance matrices waste half their storage. The condensed
upper-triangle format (already used by `bray_curtis_condensed` and `pcoa()`)
should be the default for all pairwise distance ops. This halves GPU buffer
size for distance matrix uploads.

### 2.5 Option<T> Over (T, bool) for Conditional Construction

Returning `Option<FastqRecord>` instead of `(FastqRecord, bool)` avoids
constructing (and cloning) the record when no transformation occurred. This
is idiomatic Rust and should be the pattern for all conditional transform ops.

---

## Part 3: Cross-Spring Status

| Spring | Tests | Checks | GPU Modules | Lean on ToadStool |
|--------|-------|--------|------------|-------------------|
| wetSpring | 728 | 2,229+ | 25 (20 lean + 5 neuralSpring) | 24 lean + 4 Write |
| hotSpring | 637 | ~195 | ~34 | Most lean |
| neuralSpring | — | — | 5 absorbed | Provides → consumed by wetSpring |

wetSpring consumes 5 neuralSpring-evolved primitives (validated Exp094, benchmarked
Exp095). Zero compile-time coupling between Springs — all discovery is runtime.

---

## Part 4: Codebase Health Summary

| Metric | Value |
|--------|-------|
| `cargo clippy --features gpu -- -D warnings` | 0 warnings |
| `#![forbid(unsafe_code)]` | Enforced crate-wide |
| `#![deny(clippy::expect_used, clippy::unwrap_used)]` | Enforced crate-wide |
| TODO/FIXME/HACK markers | 0 |
| `unimplemented!()` / `todo!()` | 0 |
| Cross-primal compile-time coupling | 0 |
| Production mocks | 0 |
| External C dependencies | 0 |
| Named tolerance constants | 39 |
| Max file size | All under 1000 LOC |

---

## Part 5: Related Handoffs

| Version | Date | Key Content |
|---------|------|-------------|
| v7 | Feb 22 | Dispatch routing, ODE flat APIs, Tier B→A promotion |
| v8 | Feb 22 | Pure GPU streaming, absorption strategy |
| v9 | Feb 22 | Local WGSL shaders, 96 experiments |
| v10 | Feb 22 | neuralSpring primitive wiring, cross-spring scaling |
| **v11** | **Feb 22** | **Structural evolution: flat layouts, DRY models, zero-clone APIs** |
