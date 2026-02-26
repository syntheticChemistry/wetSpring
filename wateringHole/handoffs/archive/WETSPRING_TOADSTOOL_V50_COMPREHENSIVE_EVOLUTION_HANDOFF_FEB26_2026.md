# wetSpring → ToadStool/BarraCuda V50 Comprehensive Evolution Handoff

**Date:** February 26, 2026
**Phase:** 50 (V50 — ODE derivative rewire + doc cleanup + comprehensive handoff)
**ToadStool pin:** `17932267` (S65)
**wetSpring status:** Fully lean — 0 local WGSL, 0 local derivative math, 9/9 P0-P3 DONE
**Tests:** 902 total (823 barracuda + 47 forge + 32 integration/doc)

---

## Purpose

This handoff captures the complete evolution state of wetSpring's relationship
with ToadStool/BarraCuda as of V50. It serves three audiences:

1. **ToadStool team** — What wetSpring contributed, what it consumes, and what
   patterns worked well for absorption.
2. **Other springs** — Lessons from completing the full Write → Absorb → Lean cycle.
3. **Future wetSpring maintainers** — What remains local and why.

---

## Part 1: What wetSpring Contributed to ToadStool

### Biological ODE Systems (5 systems, S54-S58)

wetSpring wrote, validated, and handed off 5 biological ODE systems that became
`barracuda::numerical::ode_bio`:

| System | Paper | State Dim | Unique Math |
|--------|-------|-----------|-------------|
| `QsBiofilmOde` | Waters 2008 | 6 | Hill activation, AHL-LuxR binding |
| `BistableOde` | Fernandez 2020 | 8 | Dual-signal bistable switch, c-di-GMP |
| `CooperationOde` | Bruger 2018 | 6 | Cooperative QS, mutualistic coupling |
| `MultiSignalOde` | Srivastava 2011 | 7 | Multi-signal integration, c-di-GMP |
| `PhageDefenseOde` | Hsueh 2022 | 5 | CRISPR/Monod + phage lysis |

These became the `OdeSystem` trait implementations that power both CPU
(`cpu_derivative`) and GPU (`wgsl_derivative`) paths. The trait-generated
WGSL shaders (`BatchedOdeRK4<S>::generate_shader()`) eliminated all local
WGSL and enabled batched GPU ODE sweeps.

**What worked:** Implementing in Rust first, validating against Python (scipy),
then extracting the derivative math into a trait. The struct-based param types
made the Rust code self-documenting; the flat-array `cpu_derivative` signature
made GPU mapping trivial.

### Diversity Statistics (11 functions, S64)

wetSpring's `bio::diversity` module (Shannon, Simpson, Chao1, Pielou, ACE,
Bray-Curtis, Jaccard, UniFrac weighted/unweighted, Faith PD, rarefaction)
was absorbed into `barracuda::stats::diversity`. These are numerically simple
but widely used — every spring that touches ecology or graph similarity
benefits.

### Diversity Fusion GPU (S63)

`diversity_fusion_f64.wgsl` — a fused GPU shader computing all diversity
metrics in a single dispatch — was absorbed as
`barracuda::ops::bio::diversity_fusion`. This was the last local WGSL file
in wetSpring.

### NMF, Ridge, Cosine Similarity (S58)

Drug repurposing primitives handed off from Track 3:
- `NmfGpu` — non-negative matrix factorization
- `RidgeRegressionGpu` — ridge regression with Cholesky solve
- `CosineSimilarityF64` — pairwise cosine similarity

### Linear Algebra Contributions

`dot` and `l2_norm` from `barracuda::stats` originated as wetSpring helpers
before absorption into S64.

---

## Part 2: What wetSpring Consumes (71 primitives)

### By Category

| Category | Count | Key Primitives |
|----------|------:|----------------|
| GPU bio ops | 15 | BatchedOdeRK4, DiversityFusionGpu, Dada2EStepGpu, GillespieGpu, etc. |
| GPU core (linalg) | 11 | GemmF64, pcoa_gpu, kriging, ESN, etc. |
| CPU special functions | 7 | erf, ln_gamma, gamma, beta, digamma, reg_gamma_{p,q} |
| CPU stats | 4 | norm_cdf, pearson_correlation, dot, l2_norm |
| CPU diversity | 11 | shannon, simpson, chao1, pielou, bray_curtis, etc. |
| Spectral / localization | 5 | anderson_3d, lanczos, level_spacing_ratio, GOE_R, POISSON_R |
| Cross-spring | 8 | find_w_c, anderson_sweep_averaged, hamming, jaccard, etc. |
| Linalg/NMF | 5 | NMF, ridge, cosine_similarity, cholesky_solve |
| BGL helpers | 2 | storage_bgl_entry, uniform_bgl_entry |
| ODE cpu_derivative | 5 | CapacitorOde, CooperationOde, MultiSignalOde, BistableOde, PhageDefenseOde |
| **Total** | **73** | (66 GPU/CPU + 2 BGL + 5 ODE derivative) |

### Usage Patterns

- **Heavy hitters** (5+ consumers): `GemmF64`, `FusedMapReduceF64`, `BatchedOdeRK4`
- **Single consumer**: `DiversityFusionGpu`, `Dada2EStepGpu`, `GillespieGpu`,
  `KrigingF64`, `SnpCallingF64`, `PeakDetectF64`, `PangenomeClassifyGpu`
- **Delegation only** (thin re-exports): `diversity_fusion_gpu`, `bio::diversity`,
  `special::{dot, l2_norm}`, 5 ODE `cpu_derivative`
- **Compose pattern**: kmd, merge_pairs, robinson_foulds, derep, NJ, reconciliation, molecular_clock

---

## Part 3: V50 ODE Derivative Rewire (What Changed)

### Before V50
wetSpring had local copies of all 5 ODE derivative functions (`hill()`,
`capacitor_rhs()`, `coop_rhs()`, etc.) — ~200 lines of math that duplicated
what barracuda already had in `OdeSystem::cpu_derivative`.

### After V50
All 5 local RHS functions deleted. Each `run_*()` function now calls
`barracuda::numerical::*Ode::cpu_derivative(t, state, &flat_params)`.

### c-di-GMP Guard (Retained Locally)
`multi_signal.rs` and `bistable.rs` wrap the barracuda derivative with a thin
guard: if c-di-GMP < `ODE_CDG_CONVERGENCE` and its derivative < 0, clamp to 0.
This prevents oscillation in fixed-step RK4 near zero. Barracuda's derivative
is correct for adaptive (RK45) and GPU batched solvers where post-step clamping
handles it. The guard is specific to wetSpring's fixed-step integration path.

**Recommendation for ToadStool:** If a future `rk4_fixed_step` API is added,
consider an optional convergence-guard callback.

### What Stays Local (By Design)

| Component | Lines | Why |
|-----------|------:|-----|
| `ode.rs` (RK4 integrator) | ~80 | Returns full trajectory; barracuda's batched solver returns final state only |
| `qs_biofilm.rs` (base model) | ~150 | Monostable Waters 2008 variant — not absorbed (bistable extension is) |
| `tolerances.rs` | ~200 | 77 domain-specific constants, complementary to `barracuda::tolerances` |
| `validation.rs` | ~100 | Simpler hotSpring-pattern validator (pass/fail/exit-code) |
| `ncbi/` module | ~300 | wetSpring-specific NCBI data pipeline |
| Param structs | ~400 | Named fields, Default impls, domain docs (flat arrays for barracuda) |

---

## Part 4: Evolution Lessons for ToadStool

### What Worked Well

1. **`OdeSystem` trait pattern.** The `cpu_derivative` / `wgsl_derivative` split
   is the single best design decision. Springs write Rust, get GPU for free.

2. **`barracuda::stats::diversity` absorption (S64).** Eleven functions, all
   trivial individually, but eliminating 11 local copies across every spring
   that touches ecology, graph similarity, or community analysis.

3. **BGL helpers.** `storage_bgl_entry` and `uniform_bgl_entry` saved ~258 lines
   of boilerplate across 6 files. Small utility, outsized impact.

4. **Cross-spring evolution.** hotSpring's special functions (erf, gamma) are
   now used by wetSpring's biology, neuralSpring's eigensolvers, and airSpring's
   Kriging. The biome model (springs don't import each other, ToadStool mediates)
   scales naturally.

5. **Named tolerances.** Central `tolerances.rs` with documented constants
   eliminates magic numbers and makes tolerance rationale grep-able.

### What Could Improve

1. **`QsBiofilmOde` gap.** The monostable Waters 2008 base model isn't in
   barracuda as a standalone `OdeSystem`. Only the bistable extension
   (Fernandez 2020) is. Adding `QsBiofilmOde` would complete the set and
   let wetSpring's `qs_biofilm.rs` fully lean.

2. **RK4 trajectory API.** barracuda's `BatchedOdeRK4::integrate_cpu` returns
   final states. A `integrate_cpu_trajectory` variant returning `Vec<Vec<f64>>`
   would let springs delegate integration entirely.

3. **Fixed-step convergence guards.** The c-di-GMP guard pattern (clamp
   derivative near zero to prevent oscillation) is generic. A
   `ConvergenceGuard` trait or callback in `rk4_fixed_step` would formalize it.

4. **`ComputeDispatch` builder adoption.** Available since S44 but wetSpring
   hasn't adopted it yet (no new GPU modules being written). Other springs
   doing Write-phase GPU work should use it from day one — eliminates 80 lines
   of bind-group/pipeline boilerplate per module.

---

## Part 5: Cross-Spring Evolution Value

### Provenance Map (What Came From Where)

```
Session  Origin        Primitive                  → Consumed By
───────  ──────────    ────────────────────────   ──────────────────
S39-40   hotSpring     erf, ln_gamma, gamma       wetSpring, neuralSpring, airSpring
S42      hotSpring     anderson_3d, lanczos       wetSpring (Track 4)
S44      hotSpring     norm_cdf                   wetSpring, airSpring
S50      hotSpring     level_spacing_ratio        wetSpring (Track 4)
S50      neuralSpring  pearson_correlation        wetSpring
S54-58   wetSpring     5 ODE systems, NMF, ridge  (shared via ToadStool)
S62      hotSpring     PeakDetectF64              wetSpring (signal_gpu)
S63      ToadStool     diversity_fusion absorb    wetSpring (lean)
S64      ToadStool     stats::diversity, metrics  wetSpring, neuralSpring
S65      ToadStool     smart refactoring          all springs (smaller crate)
```

### The Full Circle

wetSpring's ODE systems (written V24-V25) were absorbed into barracuda (S54-S58),
which generated GPU WGSL shaders used for batched parameter sweeps. In V50,
wetSpring's CPU code was rewired to call barracuda's `cpu_derivative` — the
same code it originally wrote. The Write → Absorb → Lean → Rewire cycle is
complete: wetSpring has zero local derivative math, and any improvement to
barracuda's ODE implementations (precision, performance) automatically flows
back to wetSpring's validation pipeline.

---

## Part 6: Absorption Candidates for ToadStool

### Non-blocking (future opportunities)

| Candidate | Origin | Value | Complexity |
|-----------|--------|-------|------------|
| `QsBiofilmOde` | wetSpring | Completes the 6-system ODE set | Low — follows existing `OdeSystem` pattern |
| `rk4_trajectory` | wetSpring | Returns full trajectory, not just final state | Low — ~40 lines wrapping existing integrator |
| `ConvergenceGuard` | wetSpring | Generic derivative-clamping for fixed-step solvers | Medium — API design needed |
| `hill()` public API | wetSpring | `fn hill(x, k, n) -> f64` — used everywhere in bio | Trivial — already exists as method, expose as free function |
| `monod()` public API | wetSpring | `fn monod(s, ks) -> f64` — Monod kinetics | Trivial |
| `mzML` streaming parser | wetSpring | Zero-copy LC-MS parser | Medium — currently wetSpring-specific |
| `FASTQ` streaming parser | wetSpring | Zero-copy FASTQ parser | Medium — currently wetSpring-specific |

None of these are blocking. wetSpring is fully functional and validated.

---

## Part 7: Verification Summary

| Check | Result |
|-------|--------|
| `cargo test --lib` (barracuda) | 823 pass, 0 fail, 1 ignored |
| `cargo test --lib` (forge) | 47 pass, 0 fail |
| `cargo fmt --check` | Clean (both crates) |
| `cargo clippy --lib -D warnings -W clippy::pedantic -W clippy::nursery` | 0 warnings (both crates) |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib` | 0 warnings (both crates) |
| ODE validators (6 systems) | ALL PASS |
| Cross-spring validators | Exp120 9/9, Exp169 12/12, Exp070 50/50, Exp163 27/27 |
| TODO/FIXME markers | 0 |
| Inline tolerance literals | 0 (all 77 use `tolerances::` constants) |
| Local WGSL files | 0 |
| Local ODE derivative math | 0 |

---

## Appendix: File Map

### Key barracuda Source Files

| Path | Purpose | Lines |
|------|---------|------:|
| `src/bio/capacitor.rs` | Capacitor ODE (lean on `CapacitorOde::cpu_derivative`) | ~120 |
| `src/bio/cooperation.rs` | Cooperation ODE (lean) | ~120 |
| `src/bio/multi_signal.rs` | Multi-signal ODE (lean + cdg guard) | ~140 |
| `src/bio/bistable.rs` | Bistable ODE (lean + cdg guard) | ~140 |
| `src/bio/phage_defense.rs` | Phage defense ODE (lean) | ~100 |
| `src/bio/qs_biofilm.rs` | QS biofilm base model (local — monostable) | ~200 |
| `src/bio/ode.rs` | RK4 integrator, OdeResult, trajectory | ~120 |
| `src/tolerances.rs` | 77 named tolerance constants | ~200 |
| `src/ncbi/` | NCBI data pipeline (4 submodules) | ~350 |
| `src/validation.rs` | hotSpring-pattern validator | ~100 |

### Handoff History

| Version | Date | Key Change |
|---------|------|------------|
| V7-V25 | Feb 21-22 | Write phase: ODE systems, parsers, diversity |
| V26-V34 | Feb 22-24 | Absorb phase: ToadStool S42-S62 |
| V35-V45 | Feb 24-25 | Lean phase: rewire to upstream, delete local |
| V47-V48 | Feb 25 | Track 4 + S65 rewire (fully lean) |
| V49 | Feb 25 | Evolution learnings handoff |
| V50 | Feb 26 | ODE derivative rewire (zero local math) |
