# Evolution Readiness — Rust Module → WGSL Shader → Pipeline Stage

**Date**: 2026-02-16
**Status**: Phase 4 active — 16S pipeline complete through dereplication, 31/31 GPU + 63/63 CPU validation PASS

---

## Current Rust Modules (wetspring-barracuda)

| Module | Lines | Tests | Coverage | Status |
|--------|-------|-------|----------|--------|
| `io::fastq` | 658 | 26 | 99.3% | Sovereign — in-tree parser, gzip-aware |
| `io::mzml` | 740 | 19 | 99.4% | Sovereign — in-tree XML parser, f32/f64, zlib |
| `io::xml` | 473 | 31 | 94.9% | Sovereign — minimal pull parser |
| `io::ms2` | 278 | 12 | 98.1% | Production — streaming via `BufReader` |
| `bio::kmer` | 349 | 20 | 99.5% | Production — canonical 2-bit encoding |
| `bio::diversity` | ~400 | 27 | 100% | Production — Shannon/Simpson/Chao1/BC + Pielou + rarefaction |
| `bio::quality` | ~340 | 20 | 98% | Production — sliding window trim + adapter removal (Trimmomatic) |
| `bio::merge_pairs` | ~390 | 10 | 98% | Production — paired-end merging (VSEARCH/FLASH equivalent) |
| `bio::derep` | ~250 | 9 | 98% | Production — dereplication + abundance tracking (VSEARCH) |
| `bio::tolerance_search` | 260 | 11 | 100% | Production — NaN-safe ppm/Da search + PFAS screen |
| `bio::signal` | ~300 | 11 | 98% | Production — 1D peak detection (scipy.find_peaks equivalent) |
| `bio::eic` | ~200 | 8 | 98% | Production — EIC/XIC extraction + mass track detection |
| `bio::feature_table` | ~250 | 6 | 98% | Production — end-to-end asari-style feature extraction |
| `bio::spectral_match` | ~220 | 7 | 100% | Production — MS2 cosine similarity (matched + weighted) |
| `bio::kmd` | ~200 | 6 | 100% | Production — Kendrick mass defect + homologue grouping |
| `encoding` | 172 | 10 | 90.4%* | Sovereign — replaces base64 crate |
| `error` | 141 | 5 | 100% | Production — typed error chain |
| `validation` | 245 | 10 | 92.5%** | Framework — `Validator` struct + `check_count`/`check_count_u64` |
| `bio::pcoa` | 190 | 7 | 97.9% | Production — CPU `PCoA` (Jacobi eigensolve) |
| `gpu` | 262 | — | — | GPU bridge — `GpuF64` wraps wgpu + `ToadStool` `TensorContext` |
| `bio::diversity_gpu` | ~220 | — | — | GPU: Shannon/Simpson/observed/evenness/alpha via `FusedMapReduceF64`, BC via WGSL |
| `bio::pcoa_gpu` | 120 | — | — | GPU `PCoA` via `ToadStool`'s `BatchedEighGpu` |
| `bio::spectral_match_gpu` | ~180 | — | — | GPU pairwise cosine via `GemmF64` + `FusedMapReduceF64` |
| `tolerances` | 84 | — | — | Centralized CPU + GPU tolerance constants |

**Total: 293 tests (251 unit + 42 integration)**

\* `encoding.rs` — `build_decode_table()` is a `const fn` evaluated at compile time; llvm-cov cannot instrument it.
\** `validation.rs` — remaining uncovered code is `exit_with_result()`/`exit_skipped()`/`finish()` which call `process::exit()` — untestable without forking.

GPU modules require `--features gpu` and are validated by `validate_diversity_gpu` (31/31 PASS).

---

## Code Quality Evolution (Feb 16, 2026)

| Improvement | Before | After |
|-------------|--------|-------|
| Tests | 103 (75 unit + 28 integration) | 293 (251 unit + 42 integration) |
| Coverage | ~78% | ~98% |
| `#[allow(clippy::...)]` | ~30 annotations | ~20 (all justified) |
| `too_many_lines` | 4 binaries | 0 — all refactored into functions |
| `cast_precision_loss` | ~20 count→f64 casts | 0 — `Validator::check_count`/`check_count_u64` |
| Tolerance search | `unwrap()` panics on NaN | NaN-safe via `unwrap_or(Greater)` |
| GPU dispatch | `dispatch_and_read` panics | Returns `Result<Vec<f64>>` |
| mzML binary decode | bare `unwrap()` | infallible `copy_from_slice` (zero `expect`) |
| Validation pattern | manual `total += 1` / `passed += 1` | `Validator` struct with typed methods |
| `unsafe` blocks | 0 | 0 |
| Production `panic!` | 0 | 0 |
| Production `unwrap()`/`expect()` | ~15 calls | 0 — all replaced with `?` / `copy_from_slice` / `unwrap_or` |

---

## Evolution Map: Rust → GPU

### Tier A: Pure Rewire (existing ToadStool shaders, new orchestrators only)

| Rust Module | ToadStool Primitive / WGSL Shader | Pipeline Stage | Status |
|-------------|----------------------------------|----------------|--------|
| `bio::diversity::shannon` | `FusedMapReduceF64.shannon_entropy()` | Fused map-reduce (single dispatch) | **DONE** — rewired to ToadStool |
| `bio::diversity::simpson` | `FusedMapReduceF64.simpson_index()` | Fused map-reduce (1 − Σ p²) | **DONE** — rewired to ToadStool |
| `bio::diversity::bray_curtis_condensed` | `bray_curtis_pairs_f64.wgsl` (custom) | All-pairs BC, 1 thread/pair | **DONE** — custom shader |
| `bio::pcoa` | `BatchedEighGpu.execute_f64()` | BC → double-center → eigensolve | **DONE** — CPU + GPU validated |
| `bio::diversity::observed_features` | `FusedMapReduceF64.sum()` | Binarize + reduce | **DONE** — GPU validated |
| `bio::diversity::pielou_evenness` | Shannon GPU + observed GPU | Compose H / ln(S) | **DONE** — GPU validated |
| `bio::diversity::alpha_diversity` | `FusedMapReduceF64` compose | Full alpha suite on GPU | **DONE** — GPU validated |
| `bio::spectral_match::pairwise_cosine` | `GemmF64` + `FusedMapReduceF64` | Dot product matrix + norms | **DONE** — GPU validated |
| Rarefaction curves | `prng_xoshiro.wgsl` + `FusedMapReduceF64` | Random subsample + diversity | Ready — ToadStool PRNG available |

### Tier B: Adapt Existing Patterns (minor shader modifications)

| Rust Module | WGSL Shader (base) | Modification Needed | Pipeline Stage |
|-------------|---------------------|---------------------|----------------|
| `bio::tolerance_search::find_within_ppm` | `batched_bisection_f64.wgsl` | Adapt objective to ±ppm bounds | Suspect PFAS screening |
| `bio::tolerance_search::screen_pfas_fragments` | `pairwise_distance.wgsl` | Fragment diff matrix + threshold | PFAS difference scan |
| Sparse feature tables | `sparse_matvec_f64.wgsl` | CSR format for asari tables | LC-MS quantification |
| Peak shape fitting | `weighted_dot_f64.wgsl` | Gauss-Newton inner products | mzML peak fitting |

### Tier C: New Shaders (fresh WGSL, no existing ancestor)

| Rust Module | New WGSL Shader | Complexity | Pipeline Stage |
|-------------|-----------------|------------|----------------|
| `bio::kmer::count_kmers` | `hash_table_u64.wgsl` | High — GPU lock-free hash table | K-mer dereplication |
| mzML peak detection | `find_peaks_f64.wgsl` | Medium — local maxima + prominence | LC-MS peak detect |
| Mass track smoothing | `uniform_filter_f64.wgsl` | Low — 1D moving average | LC-MS preprocessing |
| m/z sort | `sort_f64.wgsl` | Medium — bitonic merge sort | Spectral ordering |
| MS2 spectral matching | ~~`cosine_similarity_f64.wgsl`~~ | **DONE** — via `GemmF64` + `FusedMapReduceF64` | Compound ID |

---

## Blockers for GPU Promotion

| Blocker | Affects | Resolution |
|---------|---------|------------|
| GPU hash table | Tier C (k-mer) | New shader — research lock-free GPU hash design |
| ~~barracuda feature gate~~ | ~~All~~ | **DONE** — `#[cfg(feature = "gpu")]` wired, `GpuF64` bridge active |
| ~~Test parity~~ | ~~All~~ | **DONE** — 17/17 GPU checks match CPU exactly |
| ~~f64 shader variants~~ | ~~Tier A~~ | **DONE** — ToadStool `FusedMapReduceF64` + `BatchedEighGpu` f64 |
| ~~ToadStool log bug~~ | ~~Shannon~~ | **DONE** — coefficients fixed upstream (commit `0c477306`) |
| Data size threshold | All | GPU dispatch only beneficial above ~10K elements; CPU fallback needed |

---

## Dependency Sovereignty Status

| Dependency | Status | Evolution |
|------------|--------|-----------|
| `base64` | **Replaced** — `src/encoding.rs` | Sovereign ✓ |
| `serde` | **Removed** — unused | Clean ✓ |
| `serde_json` | **Removed** — unused | Clean ✓ |
| `rayon` | **Removed** — unused | Clean ✓ |
| `needletail` | **Replaced** — `io::fastq` sovereign parser | Sovereign ✓ |
| `quick-xml` | **Replaced** — `io::xml` sovereign pull parser | Sovereign ✓ |
| `flate2` | Active | Keep — wraps pure-Rust miniz_oxide (zlib + gzip) |
| `barracuda` | Feature-gated (`gpu`) | **Active** — `GpuF64` bridge to `ToadStool` `WgpuDevice`/`TensorContext` |
| `wgpu` | Feature-gated (`gpu`) | **Active** — 0.19, matching barracuda pin |
| `tokio` | Feature-gated (`gpu`) | **Active** — async GPU device creation |
| `bytemuck` | Feature-gated (`gpu`) | **Active** — shader param structs |
| `tempfile` | Dev only | Test infrastructure only |

---

## Validation Binary Coverage

### CPU Validation (default build)

| Binary | Checks | Status | Data Source |
|--------|--------|--------|-------------|
| `validate_fastq` | 28/28 | PASS | Quality filtering + merge pairs + derep + Zenodo 800651 (MiSeq SOP) |
| `validate_diversity` | 18/18 | PASS | Analytical + simulated + evenness + rarefaction |
| `validate_mzml` | 7/7 | PASS | shuzhao-li-lab/data (MT02) |
| `validate_pfas` | 10/10 | PASS | Cosine similarity + KMD + FindPFAS (external data optional) |
| **CPU Total** | **63/63** | **PASS** | |

### GPU Validation (`--features gpu`)

| Binary | Checks | Status | Comparison |
|--------|--------|--------|------------|
| `validate_diversity_gpu` | 31/31 | PASS | GPU f64 vs CPU f64 (RTX 4070) |
| **GPU Total** | **31/31** | **PASS** | |

Checks: 3 Shannon + 3 Simpson + 6 BC + 5 PCoA + 6 Alpha + 8 Spectral Match.

---

## WGSL Shaders and ToadStool Primitives

### Custom Shaders (src/shaders/)

| Shader | Type | Input | Output |
|--------|------|-------|--------|
| `bray_curtis_pairs_f64.wgsl` | Parallel pairs | samples[N×D] | distances[N*(N-1)/2] |

### ToadStool Primitives Used (via `barracuda` crate)

| Primitive | API | wetSpring Use |
|-----------|-----|---------------|
| `FusedMapReduceF64` | `shannon_entropy()` | Shannon H = -Σ p ln(p) — single-dispatch fused map-reduce |
| `FusedMapReduceF64` | `simpson_index()` | Simpson D = Σ p² — diversity = 1 - D |
| `FusedMapReduceF64` | `sum()` | Observed features (binarize + reduce) |
| `FusedMapReduceF64` | `sum_of_squares()` | Vector norms for cosine similarity |
| `BatchedEighGpu` | `execute_f64()` / `execute_single_dispatch()` | PCoA eigendecomposition on double-centered BC matrix |
| `GemmF64` | `execute()` | Pairwise dot products for spectral cosine similarity |

**Deprecated shaders removed:** `shannon_map_f64.wgsl` and `simpson_map_f64.wgsl` — replaced
by `FusedMapReduceF64` which uses ToadStool's corrected `log_f64` (bug fixed upstream).

---

## Handoff to ToadStool/BarraCUDA Team

**Full handoff document:** [`HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md`](HANDOFF_WETSPRING_TO_TOADSTOOL_FEB_16_2026.md)

**Absorbed from ToadStool (commit `0c477306`):**

1. **`FusedMapReduceF64`** — Shannon and Simpson now use single-dispatch fused
   map-reduce instead of custom WGSL shaders + CPU sum. Deprecated shaders removed.
2. **`BatchedEighGpu`** — PCoA eigendecomposition wired and validated (5/5 checks).
3. **`log_f64` coefficient fix** — confirmed absorbed upstream, ToadStool's
   `math_f64.wgsl` now has corrected coefficients.

**Remaining evolution opportunities:**

1. **Bray-Curtis → ToadStool**: Custom `bray_curtis_pairs_f64.wgsl` could be
   promoted to ToadStool as a general-purpose condensed distance kernel.
2. **Rarefaction GPU (bootstrap CI)**: Wire `prng_xoshiro.wgsl` for Monte Carlo
   confidence intervals (CPU exact rarefaction is analytically sufficient).
3. **m/z tolerance search GPU**: Adapt `batched_bisection_f64.wgsl` for ppm-bounded
   binary search on sorted m/z arrays.
4. **EIC extraction GPU**: Parallel m/z binning across scans via custom WGSL kernel.
5. **Peak detection GPU**: Promote `bio::signal::find_peaks` to GPU for parallel
   chromatogram processing (STFT windowing available in ToadStool).
6. **DADA2-equivalent denoising**: Sequence error correction model for 16S amplicons.
7. **Chimera detection**: Reference-free chimera filtering (uchime3 equivalent).
8. **Taxonomy classification**: Naive Bayes / BLAST for 16S ASV taxonomy assignment.
9. **UniFrac distance**: Phylogeny-weighted beta diversity.

---

*Updated from wetSpring Phase 4 — 16S pipeline through dereplication + LC-MS feature extraction, February 16, 2026.*
