# WETSPRING V121 — Deep Debt Evolution + Absorption Handoff

**Date:** 2026-03-16
**From:** wetSpring V121
**To:** barraCuda, toadStool, All Springs
**License:** AGPL-3.0-or-later
**Covers:** V120 → V121 (deep audit execution, tolerance centralization, lint evolution)

---

## Executive Summary

- **Full-spectrum audit** executed against all wateringHole ecosystem standards
- **Zero clippy warnings** (pedantic + nursery), zero `#[allow()]` in production code
- **214 named tolerance constants** — all inline literals centralized (14 new, ~50 replacements)
- **All hardcoded primal names** replaced with `primal_names::*` constants
- **Pure Rust deps evolved**: blake3 `default-features=false, features=["pure"]`
- **1,685 tests pass**, 354 binaries, 376 experiments, 5,707+ validation checks

---

## Part 1: What Changed (V120 → V121)

### Tolerance Centralization (14 new constants, ~50 replacements)

New constants added to `barracuda/src/tolerances/`:

| Constant | Value | Module | Purpose |
|----------|-------|--------|---------|
| `NORM_CDF_SYMMETRY` | 1e-7 | mod.rs | CDF Φ(x)+Φ(−x)=1 |
| `RT_PARSE_PARITY` | 1e-6 | instrument.rs | Retention time parse roundtrip |
| `JCAMP_Y_PARSE` | 1e-9 | instrument.rs | JCAMP-DX Y-value precision |
| `FEATURE_MZ_MATCH` | 1.0 | instrument.rs | Feature m/z match (Da) |
| `FEATURE_RT_APEX` | 0.2 | instrument.rs | Feature RT apex (min) |
| `RI_PAPER_DEVIATION` | 5.0 | instrument.rs | RI vs published (%) |
| `RI_SEARCH_RELATIVE` | 0.05 | instrument.rs | RI search fraction |
| `GPU_F32_PAIRWISE_L2` | 1e-3 | gpu.rs | PairwiseL2 f32 kernel |
| `TRANSFER_TIME_PARITY` | 1e-6 | gpu.rs | metalForge transfer time |
| `ODE_CARRYING_CAPACITY_LOOSE` | 0.05 | bio/ode.rs | K_cap loose check |
| `ODE_DT_SWEEP` | 0.01 | bio/ode.rs | Sweep/proptest time step |
| `GILLESPIE_ENSEMBLE_MEAN_PCT` | 0.15 | bio/misc.rs | SSA ensemble % |
| `GILLESPIE_PROPTEST_PCT` | 0.25 | bio/misc.rs | Proptest steady-state % |
| `ASSEMBLY_MEAN_SIZE_TOL` | 1.0 | bio/misc.rs | Genome assembly size |

Inline literals replaced in: `bio/ode.rs`, `bio/qs_biofilm.rs`, `bio/multi_signal.rs`,
`bio/feature_table.rs`, `bio/cooperation.rs`, `bio/gillespie.rs`, `io/mzxml`,
`io/mzml`, `io/jcamp`, `special.rs`, and 9 validation binaries.

### Lint Evolution

- Crate-level `#[allow()]` → three `#[expect(reason)]` in `lib.rs`
- 8 stale `#[expect()]` removed across 7 validation binaries
- Long literal separator (`131072` → `131_072`)
- `deploy_graph_path()` → `const fn`
- `pub(crate)` → `pub` in private chimera submodule
- 10 doc-backtick fixes in `metalForge/forge/src/error.rs` and `primal_names.rs`

### Primal Name Centralization

Hardcoded `"squirrel"`, `"biomeOS"`, `"toadstool"`, `"beardog"`, `"songbird"`,
`"nestgate"`, `"wetspring"` replaced with `primal_names::*` in 7 validation binaries.

### Dependency Evolution

- `blake3 = { default-features = false, features = ["pure"] }` — eliminates `cc` build-dep
- All paths remain pure Rust, ecoBin compliant
- `wgpu` `renderdoc-sys` noted as known upstream issue (not application code)

### Infrastructure Fixes

- `verify_baseline_outputs.sh` rerun comparison bug fixed (was no-op)
- 8 hardcoded `/tmp/` test paths → `tempfile::TempDir`
- Local `shannon_byte_entropy` and `percentile` documented as domain-specific (not barraCuda candidates)

---

## Part 2: barraCuda Primitive Consumption (current state)

### CPU Primitives

| Category | Primitives |
|----------|-----------|
| **stats** | shannon, simpson, chao1, bray_curtis, pielou_evenness, rarefaction_curve, alpha_diversity, bootstrap_ci, mean, percentile, variance, welford |
| **numerical** | CooperationOde, rk45_solve, trapz, gradient_1d |
| **linalg** | ridge_regression, jacobi_eigh |
| **special** | erf, ln_gamma, regularized_gamma_p |

### GPU Primitives (150+)

| Category | Primitives |
|----------|-----------|
| **ops** | FusedMapReduceF64, BrayCurtisF64, GemmF64, AniBatchF64, DnDsBatchF64 |
| **bio** | QualityFilterGpu, HmmBatchForwardF64, PairwiseHammingGpu, PairwiseJaccardGpu |
| **spatial** | BatchFitnessGpu, LocusVarianceGpu, DiversityFusionGpu, BatchedMultinomialGpu |
| **ODE** | BatchedOdeRK4<S>::generate_shader() (5 systems) |
| **linalg** | KrigingF64, BatchedEighGpu, PCoA ordination |

### Unwired Primitives (candidates for future wiring)

- `SparseGemmF64` — Track 3 drug repurposing (when sparse matrices needed)
- `TranseScoreF64` — Knowledge graph embedding GPU acceleration
- `TopK` — Ranking acceleration for NMF/KG results

---

## Part 3: Patterns Worth Absorbing

### For barraCuda

1. **Tolerance module pattern**: 214 named constants across 10 submodules, each with
   scientific justification and validation provenance. No inline literals. Pattern:
   `pub const NAME: f64 = value; /// doc with derivation + validated: Exp###`.

2. **`#[expect(reason)]` migration**: All lint overrides use `#[expect()]` with mandatory
   reason strings. Zero bare `#[allow()]` in production code. Makes every suppression
   self-documenting and detectable when stale.

3. **Primal name constants**: `primal_names.rs` with `SELF`, `BIOMEOS`, `SONGBIRD`, etc.
   eliminates hardcoded strings. Runtime discovery only — zero compile-time coupling.

4. **Proptest integration**: 4 property-based tests for stochastic algorithms (Gillespie,
   bootstrap, rarefaction, cooperation). Fixed seeds with named tolerance ceilings.

### For toadStool

1. **Deploy graph `fallback = "skip"`**: Optional primals (ToadStool, Squirrel, NestGate,
   petalTongue) declared with `fallback = "skip"` — allows partial niche boot.

2. **Capability-based routing**: `capability_domains.rs` with 16 domains / 20 methods,
   `all_methods()` introspection, `registration_domains()` for Songbird. Semantic
   method naming: `{domain}.{operation}`.

3. **Niche self-knowledge**: `niche.rs` exports dependencies, cost estimates per method,
   ecology semantic mappings. `const fn deploy_graph_path()`.

---

## Part 4: Evolution Requests for Upstream

### barraCuda (P1 — high priority)

- **`u64` percentile API**: wetSpring has a local `fn percentile(data: &[u64], pct: usize) -> u64`
  for NPU timing diagnostics. If barraCuda adds a generic integer percentile, we can delegate.
- **Tolerance module documentation**: Consider adopting wetSpring's tolerance constant pattern
  (named, justified, provenance-traced) as a barraCuda standard for cross-spring consistency.

### barraCuda (P2 — nice to have)

- **SparseGemmF64**: Needed for Track 3 drug repurposing with sparse drug-disease matrices.
- **TranseScoreF64**: GPU-accelerated knowledge graph scoring for ROBOKOP embedding.
- **TopK kernel**: Ranking acceleration for NMF reconstruction scores.

### toadStool (P1)

- **`#[expect(reason)]` pattern**: Adopt across toadStool codebase for self-documenting
  lint suppressions. Remove stale `#[allow()]` attributes.
- **tempfile for test paths**: Evolve any `/tmp/` hardcoded test paths to `tempfile::TempDir`.

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| `cargo fmt --check` | Clean (0 diffs) |
| `cargo clippy --all-features` | **Zero warnings** |
| `cargo test --workspace` | **1,685 passed**, 0 failed, 2 ignored |
| Validation checks | 5,707+ |
| Validation binaries | 354 (332 barracuda + 22 forge) |
| Experiments | 376 |
| `unsafe` blocks | 0 (`#![forbid(unsafe_code)]`) |
| `#[allow()]` in production | 0 |
| TODO/FIXME markers | 0 |
| Inline tolerance literals | 0 |
| Hardcoded primal names | 0 |
| Local WGSL shaders | 0 |
| Named tolerance constants | 214 |
| Python baselines | 58 (SHA-256 verified) |
| barraCuda version | v0.3.5 (HEAD 03986ce) |
| toadStool | S155 |
| Rust edition | 2024 |
| MSRV | 1.87 |
