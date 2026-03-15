# wetSpring V111: Deep Debt Resolution + Build Health + Idiomatic Evolution

**Date:** 2026-03-14
**From:** wetSpring (life science + analytical chemistry validation)
**To:** barraCuda/toadStool team, ecoPrimals ecosystem

---

## Summary

V111 is a comprehensive build health restoration and deep debt elimination pass.
The workspace was broken (path dependency failures, feature gate errors) and
multiple clippy pedantic/nursery warnings had accumulated. All issues resolved.

### Key Changes

1. **Build restored** — workspace was uncompilable due to `akida-driver` path
   case mismatch and missing `bingocube-nautilus` crate
2. **bingocube-nautilus v0.1.0** — real implementation created at
   `primalTools/bingoCube/nautilus/` with evolutionary reservoir computing
   (board populations, LOO edge detection, concept edges)
3. **29 clippy errors eliminated** — `suboptimal_flops`, `cast_lossless`,
   `doc_markdown`, `single_match_else`, `redundant_clone`, `collection_is_never_read`
4. **Hot-path clones eliminated** — `msa.rs` profile/leaf clones replaced with
   ownership transfer (`std::mem::take`), Arc clones documented
5. **Flaky tests fixed** — forge nest tests: sleep → poll-based socket wait
6. **Provenance documentation** — 4 visualization validators received structured
   provenance tables; ad-hoc tolerances justified
7. **barraCuda v0.3.5** — dependency updated from v0.3.3

### Verification

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo clippy --workspace --all-targets -D warnings -W pedantic -W nursery` | PASS (0 warnings) |
| `cargo doc --workspace --no-deps` | PASS (0 warnings) |
| `cargo test --workspace` | 1,621 passed, 0 failed, 2 ignored |

---

## Absorption Opportunities for Upstream

### For barraCuda team

1. **`shaders::provenance` un-gating** — currently GPU-feature-gated. Shader
   provenance is metadata, not GPU compute. Un-gating would let CPU-only
   validation binaries (like `validate_barracuda_cpu_v27`) report provenance
   without requiring the GPU feature. Low risk, high convenience.

2. **`barracuda::stats::{dot, l2_norm}` f64 slice versions** — wetSpring has
   local `special::dot` and `special::l2_norm` because barracuda's versions
   are GPU tensor ops. Adding CPU `f64` slice variants would let springs
   eliminate local math.

3. **Gompertz/Monod/Haldane kinetics** — wetSpring has 12+ identical copies
   of biogas kinetics functions across validation binaries. These are
   reusable analytical functions that could live in `barracuda::numerical`
   or `barracuda::bio::kinetics`.

### For toadStool team

1. **`bingocube-nautilus` evolution** — the crate at `primalTools/bingoCube/nautilus/`
   is a minimal but functional implementation. hotSpring's brain architecture
   (4-layer concurrent, Gen2 36-head ESN) should inform the evolution of this
   shared tool. Key API: `NautilusBrain`, `BetaObservation`, `ShellConfig`,
   `ReservoirInput`.

2. **metalForge absorption** — `wetspring-forge` (probe, inventory, dispatch,
   streaming, bridge) is the absorption seam for toadStool. When toadStool
   absorbs forge, the bridge module becomes the integration point.

---

## Primitive Consumption (v0.3.5)

| Category | Primitives | Examples |
|----------|:----------:|---------|
| stats | 15+ | Shannon, Simpson, Chao1, mean, variance, correlation, norm_cdf, bootstrap, jackknife |
| linalg | 5+ | Laplacian, effective_rank, ridge, eigh |
| special | 3 | erf, ln_gamma, regularized_gamma_p |
| numerical | 3 | trapz, gradient_1d, rk45_integrate |
| GPU ops | 20+ | FusedMapReduceF64, BrayCurtisF64, GemmF64, BatchedOdeRK4, BatchedEighGpu |
| shaders::provenance | 1 | evolution_report, shader registry |

---

## What's Left (Future Work)

- **llvm-cov verification** — CI gates at 90% (barracuda) and 80% (forge),
  but not locally verified in this session
- **Python baseline rerun** — confirm no drift in baseline scripts
- **Tolerance tightening** — visualization validators use wide tolerances
  (0.2–0.5) that should narrow as synthetic baselines stabilize
- **Crate-level `cast_*` allows** — documented but not yet narrowed to sites
