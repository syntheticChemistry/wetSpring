# Exp097: Structural Evolution Pass — Flat Layouts, DRY Models, Zero-Clone APIs

**Date:** February 22, 2026
**Binary:** N/A (codebase evolution, validated by full test suite + GPU validation)
**Status:** COMPLETE
**Commits:** `3a5e7e4` (Phase 1), `b2b7125` (Phase 2)

---

## Objective

Deep structural evolution of the barracuda codebase targeting four axes:
1. **Flat data layouts** — eliminate `Vec<Vec<>>` in performance-critical structures
2. **DRY error models** — unify duplicated CPU/GPU logic into single source
3. **Zero-clone APIs** — eliminate unnecessary allocations on hot paths
4. **Capability-based dispatch** — remove hardcoded hardware assumptions

## Method

Two-pass audit-and-execute cycle. First pass audited all lib code for unsafe,
hardcoding, mocks, large files, external dependencies, and non-idiomatic
patterns. Second pass audited remaining Vec<Vec<>>, clone(), TODO/FIXME, and
cross-primal coupling. Each finding was executed immediately and validated
by cargo clippy + cargo test + GPU validation.

## Results — Phase 1 (`3a5e7e4`)

| Change | Module | Impact |
|--------|--------|--------|
| PCoA coordinates flat | `pcoa.rs`, `pcoa_gpu.rs` | Vec<Vec<f64>> → flat Vec<f64> + n_samples/n_axes + accessors |
| Capability-based ODE polyfill | `ode_sweep_gpu.rs` | `ShaderTemplate::for_driver_auto(_, true)` → `dev.needs_f64_exp_log_workaround()` |
| Named GPU constants | `gpu.rs` | 3 magic numbers → DEVICE_LABEL, MAX_STORAGE_BINDING_BYTES, etc. |
| Zero-clone ODE closures | 6 ODE modules | Removed `params.clone()` — borrow into Fn closure directly |
| StatsAccumulator | `fastq.rs` | Unified duplicated stats logic between batch/streaming paths |
| HashMap→HashSet | `taxonomy.rs` | `HashMap<u64, bool>` → `HashSet<u64>` for k-mer presence |
| std::mem::take | `taxonomy.rs` | Eliminated clone in FASTA parser ownership transfer |
| XML buffer constants | `xml.rs` | Magic capacity numbers → INITIAL_TEXT_BUF_CAPACITY etc. |
| Hardware-agnostic docs | 5 files | Removed "RTX 4070" / "Ada Lovelace" from lib comments |
| #[inline] hot paths | diversity, quality, kmer, taxonomy | 12 functions annotated |

## Results — Phase 2 (`b2b7125`)

| Change | Module | Impact |
|--------|--------|--------|
| ODE trajectory flat | `ode.rs` | Vec<Vec<f64>> → flat Vec<f64> + n_vars + state_at/states/var_at |
| ODE per-step clone elimination | `ode.rs` | `y_vec.push(y.clone())` → `y_flat.extend_from_slice(&y)` |
| Gillespie trajectory flat | `gillespie.rs` | Vec<Vec<i64>> → flat Vec<i64> + n_species + state_at/states_iter |
| Gillespie per-event clone elimination | `gillespie.rs` | `states.push(state.clone())` → `flat_states.extend_from_slice(&state)` |
| DADA2 error model unification | `dada2.rs`, `dada2_gpu.rs` | 5 functions shared: init, estimate, converged, base_to_idx, constants |
| UniFrac condensed distance matrix | `unifrac.rs` | Vec<Vec<f64>> N×N → condensed upper-triangle Vec<f64> |
| Adapter trim zero-clone | `quality.rs` | `(FastqRecord, bool)` → `Option<FastqRecord>` — no clone on common path |

## Validation

| Suite | Result |
|-------|--------|
| `cargo clippy --features gpu -- -D warnings` | 0 warnings |
| `cargo test` | **728/728** pass (654 lib + 74 integration/doc) |
| `validate_cpu_vs_gpu_all_domains` | **48/48** — 16 domains proven |
| `validate_cross_spring_evolution` | **39/39** — 5 neuralSpring primitives |

## Audit Findings (for the record)

- **unsafe code:** Zero. `#![forbid(unsafe_code)]` enforced crate-wide.
- **TODO/FIXME/HACK:** Zero markers in any .rs file.
- **Cross-primal coupling:** Zero. No compile-time imports of other Springs.
- **unimplemented!/todo! macros:** Zero.
- **External dependencies:** All appropriate. flate2 uses rust_backend. No C deps.
- **Production mocks:** Zero. All mocks confined to `#[cfg(test)]`.

## Remaining Vec<Vec<>> Opportunities

Identified but deferred (marginal impact or jagged/variable-length data):
- `streaming_gpu.rs` / `taxonomy_gpu.rs` query k-mer lists (GPU batch candidate)
- `bootstrap.rs` column-major alignment (flat layout candidate)
- `hmm.rs` posterior_batch (per-sequence variable length)
- `felsenstein.rs` leaf sequences (GPU flat tree candidate)
- `neighbor_joining.rs` distance matrix batch (symmetric, condensed candidate)
- `kmd.rs` homologue groups (naturally jagged)
- `dnds.rs` permutations (recursive, naturally jagged)

## Files Modified

22 files across lib, bin, and integration tests:
- `bio/ode.rs`, `bio/gillespie.rs`, `bio/quality.rs`, `bio/unifrac.rs`
- `bio/dada2.rs`, `bio/dada2_gpu.rs`
- `bio/cooperation.rs`, `bio/bistable.rs`, `bio/capacitor.rs`
- `bio/multi_signal.rs`, `bio/phage_defense.rs`, `bio/qs_biofilm.rs`
- `bin/validate_capacitor.rs`, `bin/validate_cooperation.rs`
- `bin/validate_bistable.rs`, `bin/validate_multi_signal.rs`
- `bin/validate_phage_defense.rs`, `bin/validate_qs_ode.rs`
- `bin/validate_gillespie.rs`, `bin/validate_fastq.rs`
- `bin/validate_16s_pipeline.rs`
- `tests/bio_integration.rs`
