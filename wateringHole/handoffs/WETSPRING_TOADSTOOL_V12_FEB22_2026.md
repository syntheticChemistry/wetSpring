# wetSpring → ToadStool/BarraCUDA Handoff v12

**Date:** February 22, 2026
**From:** wetSpring (life science & analytical chemistry biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-or-later
**Context:** Phase 24 — Edition 2024, structural audit, coverage verification, bio::special migration

---

## Executive Summary

wetSpring has completed **97 experiments, 2,229+ validation checks, 740 Rust
tests, and 87 binaries** — all passing. **28 ToadStool primitives consumed**
(15 original + 8 bio absorbed + 5 neuralSpring). 4 local WGSL shaders remain
in Write phase. Rust edition 2024, MSRV 1.85.

**New in v12 (since v11):**

1. **Rust edition 2024** — migrated from 2021, MSRV 1.85, all import/formatting rules applied
2. **`forbid(unsafe_code)` → `deny(unsafe_code)`** — Rust 2024 makes `std::env::set_var` unsafe;
   `#[allow(unsafe_code)]` confined to test-only env-var manipulation with SAFETY docs
3. **CI hardened** — `RUSTDOCFLAGS="-D warnings"`, `clippy -D pedantic -D nursery`,
   `cargo check --features json` added to GitHub Actions workflow
4. **`bio::special` shim removed** — migration to `crate::special` complete, zero consumers
5. **New clippy lints resolved** — `f64::midpoint()`, `usize::midpoint()`, `const fn` promotions
6. **Coverage verified** — `cargo-llvm-cov` confirms bio+io modules avg ~97% line coverage;
   12 modules at 100%, lowest module (taxonomy/classifier) at 87% with 2 new tests added
7. **`rustfmt.toml` edition 2024** — aligned with `Cargo.toml`
8. **740 tests pass** (666 lib + 60 integration + 14 doc). Zero warnings across fmt, clippy, doc.

---

## Part 1: What wetSpring Has Ready for Absorption

### 4 Local WGSL Shaders (Write Phase)

| Shader | File | Domain | Bindings | Workgroup | Tests | Blocker |
|--------|------|--------|----------|-----------|-------|---------|
| `batched_qs_ode_rk4_f64.wgsl` | `src/shaders/` | ODE parameter sweep | `[params, y_in, y_out, uniforms]` | `(64,1,1)` | 7 | `compile_shader` needs `compile_shader_f64` |
| `kmer_histogram_f64.wgsl` | `src/shaders/` | K-mer counting | `[sequences, histograms, uniforms]` | `(256,1,1)` | — | Ready for absorption |
| `unifrac_propagate_f64.wgsl` | `src/shaders/` | UniFrac tree propagation | `[flat_tree, sample_matrix, result, uniforms]` | `(64,1,1)` | — | Ready for absorption |
| `taxonomy_fc_f64.wgsl` | `src/shaders/` | Taxonomy NB scoring (NPU) | `[weights, query_kmers, scores, uniforms]` | `(64,1,1)` | — | Ready for absorption |

### CPU Math Ready for `barracuda::math`

| Function | File | Algorithm | Tests |
|----------|------|-----------|-------|
| `erf(x)` | `src/special.rs` | Abramowitz & Stegun 7.1.26 | 5 |
| `normal_cdf(x)` | `src/special.rs` | Φ(x) = 0.5 × (1 + erf(x/√2)) | 4 |
| `ln_gamma(x)` | `src/special.rs` | Lanczos g=5, n=6 | 3 |
| `regularized_gamma_lower(a,x)` | `src/special.rs` | DLMF §8.2 series | 2 |

These are sovereign implementations (no libm). The `bio::special` re-export shim was
removed in Phase 24 — all consumers now use `crate::special::*` directly.

### Tier A Modules (GPU/NPU-ready, flat layouts)

| Module | Flat API | GPU Buffer | Tests |
|--------|----------|------------|-------|
| `kmer` | `to_histogram()` → `Vec<u32>` 4^k | Direct GPU buffer | 60+ |
| `unifrac` | `FlatTree` (CSR) + `to_sample_matrix()` | Direct GPU pairwise dispatch | 19+ |
| `taxonomy` | `to_int8_weights()` → `NpuWeights` | int8 affine quantization for NPU | 19+ |
| `ode` (all 6) | `to_flat()`/`from_flat()` → `Vec<f64>` | Batched RK4 via WGSL | 48+ |
| `diversity` | Flat `&[f64]` inputs | Via `FusedMapReduceF64` | 27+ |
| `pcoa` | Flat coordinates + `coord()` accessor | Via `BatchedEighGpu` | 33+ |
| `random_forest` | SoA layout | Via `RfBatchInferenceGpu` | 12+ |

---

## Part 2: What ToadStool Has That wetSpring Should Use More

| Primitive | Current wetSpring Use | Opportunity |
|-----------|----------------------|-------------|
| `compile_shader_f64()` | Not available (blocker) | Unblock ODE shader absorption |
| `FlatTree` generic | Not available | Replace wetSpring's local `FlatTree` with shared primitive |
| `KmerHistogramGpu` | Not available | Absorb `kmer_histogram_f64.wgsl` |
| `UniFracPairwiseGpu` | Not available | Absorb `unifrac_propagate_f64.wgsl` |
| `TaxonomyNpuInference` | Not available | Absorb `taxonomy_fc_f64.wgsl` for NPU int8 path |
| `prng_xoshiro` | Not used | Enable full-GPU rarefaction (random sampling on GPU) |

### Priority Order for ToadStool Team

1. **P1 — `compile_shader_f64()`** — Unblocks ODE shader (only remaining blocker from v8)
2. **P2 — `KmerHistogramGpu`** — k-mer counting is a universal bio primitive
3. **P3 — `UniFracPairwiseGpu`** — tree traversal for phylogenetic distance
4. **P4 — `TaxonomyNpuInference`** — int8 FC model for NPU (AKD1000) path
5. **P5 — `barracuda::math` feature** — CPU-only math (erf, gamma) without wgpu stack

---

## Part 3: Phase 24 Code Quality Evolution

### Edition 2024 Migration

| Change | Before | After | Why |
|--------|--------|-------|-----|
| Cargo.toml edition | `"2021"` | `"2024"` | Modern Rust idioms, import ordering |
| MSRV | 1.82 | 1.85 | Required by edition 2024 |
| `forbid(unsafe_code)` | `#![forbid]` | `#![deny]` | Rust 2024 makes `std::env::set_var` unsafe |
| Test env vars | safe | `unsafe { set_var() }` | `#[allow(unsafe_code)]` on test module only |
| `rustfmt.toml` | `edition = "2021"` | `edition = "2024"` | Aligned with Cargo.toml |
| `f64::midpoint()` | `(a + b) / 2.0` | `f64::midpoint(a, b)` | Overflow-safe |
| `usize::midpoint()` | `(a + b) / 2` | `usize::midpoint(a, b)` | Overflow-safe |
| `const fn` | regular `fn` | `const fn` | Compile-time evaluation where possible |

### Coverage Verification

`cargo-llvm-cov` confirms:

| Module Group | Line Coverage | Functions |
|-------------|:------------:|:---------:|
| bio/* (41 modules) | 93-100% | 80-100% |
| io/* (5 modules) | 90-100% | 80-100% |
| special.rs | 99% | 100% |
| encoding.rs | 90% | 93% |
| tolerances.rs | 100% | 100% |
| **bio+io weighted avg** | **~97%** | **~95%** |

12 bio modules at exactly 100%: `phage_defense`, `phred`, `qs_biofilm`, `bistable`,
`cooperation`, `tolerance_search`, `spectral_match`, `unifrac/mod`, `unifrac/flat_tree`,
`taxonomy/mod`, `taxonomy/types`, `placement`.

### Dead Module Cleanup

- `bio::special` re-export shim **deleted** (migration to `crate::special` complete)
- All `#[allow(dead_code)]` in lib code audited — only `mzml::decode_into()` remains
  (public API used only in tests; production path uses `decode_into_with_buffer`)

---

## Part 4: Lessons Learned

1. **Edition 2024 `unsafe` change** — `std::env::set_var/remove_var` becoming `unsafe`
   in Rust 2024 breaks `#![forbid(unsafe_code)]` for any crate that sets env vars in
   tests. Solution: `#![deny(unsafe_code)]` + `#[allow(unsafe_code)]` on test modules
   with `// SAFETY:` comments. CI still catches production unsafe.

2. **`bio::special` lifecycle** — Promoted from `bio::special` to `crate::special`
   (Phase 19), kept backward-compatible re-export, then removed shim (Phase 24) when
   grep confirmed zero consumers. Clean three-step migration.

3. **Coverage as a gate** — Running `cargo-llvm-cov` revealed that claimed coverage
   numbers were stale. Taxonomy classifier accessors (`taxon_priors`, `n_kmers_total`)
   were untested. Adding 2 targeted tests brought classifier coverage from ~85% to ~90%.
   Coverage should be verified, not claimed.

4. **`rustfmt.toml` edition sync** — When `Cargo.toml` says `edition = "2024"`, the
   `rustfmt.toml` must also say `edition = "2024"` or formatting will use wrong rules.
   Both must be updated together.

---

## Part 5: Concrete Next Steps

### For ToadStool/BarraCUDA Team

1. Implement `compile_shader_f64()` to unblock ODE shader absorption (P1)
2. Absorb `kmer_histogram_f64.wgsl` as `barracuda::ops::bio::kmer_histogram` (P2)
3. Absorb `unifrac_propagate_f64.wgsl` as `barracuda::ops::bio::unifrac_propagate` (P3)
4. Consider `barracuda::math` feature (CPU-only, no wgpu) for `erf`, `ln_gamma`,
   `regularized_gamma_lower` — these are used by `dada2` and `pangenome` consumers
5. Int8 quantization path for NPU taxonomy inference (P4)

### For wetSpring

1. When ToadStool absorbs shaders, rewire wetSpring GPU modules (same as Feb 22 bio rewire)
2. When `barracuda::math` becomes available, replace `crate::special` with upstream imports
3. Continue coverage improvements in `ms2.rs` (89.81%) and `xml.rs` (94.87%)
4. Track Rust stable for any new clippy lints from edition 2024

---

## Appendix: File Inventory

### Crate Structure

| Path | Purpose | Lines |
|------|---------|:-----:|
| `src/lib.rs` | Crate root, lint config | 68 |
| `src/special.rs` | Sovereign math (erf, gamma) | 162 |
| `src/tolerances.rs` | 43 named constants | ~105 |
| `src/encoding.rs` | Sovereign base64 (RFC 4648) | 191 |
| `src/error.rs` | Error types (no external crates) | ~75 |
| `src/validation.rs` | hotSpring validation framework | ~340 |
| `src/bio/` | 41 CPU + 25 GPU modules | ~15K |
| `src/io/` | FASTQ, mzML, MS2, XML parsers | ~3K |
| `src/bench/` | Benchmark harness + power | ~820 |
| `src/bin/` | 78 validate + 9 benchmark | ~10K |
| `src/shaders/` | 4 local WGSL (ODE, kmer, unifrac, taxonomy) | ~400 |

### Active Handoff Index

```
wateringHole/handoffs/WETSPRING_TOADSTOOL_V12_FEB22_2026.md  ← THIS (LATEST)
wateringHole/handoffs/WETSPRING_TOADSTOOL_V11_FEB22_2026.md
wateringHole/handoffs/WETSPRING_TOADSTOOL_V10_FEB22_2026.md
wateringHole/handoffs/WETSPRING_TOADSTOOL_REWIRE_FEB22_2026.md
wateringHole/handoffs/CROSS_SPRING_PROVENANCE_FEB22_2026.md
wateringHole/handoffs/CROSS_SPRING_EVOLUTION_WETSPRING_FEB22_2026.md
archive/WETSPRING_TOADSTOOL_V8_FEB22_2026.md
archive/WETSPRING_TOADSTOOL_V7_FEB22_2026.md
```

### Validation Summary

| Tier | What | Count |
|------|------|:-----:|
| **BarraCUDA CPU** | Rust math matches Python across 25 domains | 205/205 |
| **BarraCUDA GPU** | GPU math matches CPU across 16 domains | 609 checks |
| **metalForge** | Substrate-independent output across CPU/GPU/NPU | 80 dispatch + 35 layout + 57 transfer |
| **Cross-spring** | neuralSpring primitives validated in wetSpring | 39/39 |
| **Streaming** | Pure GPU pipeline, zero CPU round-trips | 80/80 (441-837× over round-trip) |
| **Tests** | 740 Rust tests (666 lib + 60 integration + 14 doc) | All pass |
| **Quality** | fmt, clippy (pedantic+nursery), doc | Zero warnings |
