# Deprecation & Migration Tracking

**Date:** February 22, 2026
**Pattern:** Following hotSpring's `DEPRECATION_MIGRATION.md`

When ToadStool absorbs a wetSpring shader or pattern, the local copy becomes
**deprecated**. We keep it for reference but use the upstream version in all
new code. This document tracks what's been absorbed and what's still local.

---

## Absorbed (Lean on Upstream)

These are primitives we wrote locally, ToadStool absorbed, and we now
consume from the barracuda crate. **Do not modify local copies.**

| Local Code | Upstream | Absorbed In | Migration |
|------------|----------|-------------|-----------|
| `alignment::score_batch` | `SmithWatermanGpu` | cce8fe7c (Feb 20) | Rewired in Exp044/045 |
| `gillespie::gillespie_ssa` | `GillespieGpu` | cce8fe7c (Feb 20) | Rewired (driver skip on Ada) |
| `decision_tree::predict_batch` | `TreeInferenceGpu` | cce8fe7c (Feb 20) | Rewired in Exp044/045 |
| `felsenstein::log_likelihood` | `FelsensteinGpu` | cce8fe7c (Feb 20) | Rewired in Exp046 |
| `gemm_cached` (include_str path) | `GemmF64::WGSL` constant | cce8fe7c (Feb 20) | Path eliminated in Exp045 |
| `hmm_forward_f64.wgsl` | `HmmBatchForwardF64` | ToadStool 31d (Feb 22) | Rewired in Exp077; local shader deleted |
| `ani_batch_f64.wgsl` | `AniBatchF64` | ToadStool 31d (Feb 22) | Rewired in Exp077; local shader deleted |
| `snp_calling_f64.wgsl` | `SnpCallingF64` | ToadStool 31d (Feb 22) | Rewired in Exp077; SNP binding layout fix |
| `dnds_batch_f64.wgsl` | `DnDsBatchF64` | ToadStool 31d (Feb 22) | Rewired in Exp077; local shader deleted |
| `pangenome_classify.wgsl` | `PangenomeClassifyGpu` | ToadStool 31d (Feb 22) | Rewired in Exp077; local shader deleted |
| `quality_filter.wgsl` | `QualityFilterGpu` | ToadStool 31d (Feb 22) | Rewired in Exp077; local shader deleted |
| `dada2_e_step.wgsl` | `Dada2EStepGpu` | ToadStool 31d (Feb 22) | Rewired in Exp077; local shader deleted |
| `rf_batch_inference.wgsl` | `RfBatchInferenceGpu` | ToadStool 31g (Feb 22) | Rewired in Exp077; local shader deleted |

---

## Active Local Extensions (Not Yet Absorbed)

These are wetSpring-specific GPU modules and shaders that ToadStool has not
yet absorbed. They remain active and maintained in wetSpring.

### Local WGSL Shaders (Tier A — absorption candidates)

| Shader | Module | GPU Checks | Since | Blocker |
|--------|--------|:----------:|-------|---------|
| `batched_qs_ode_rk4_f64.wgsl` | `ode_sweep_gpu` | 7 | Exp049 | Upstream `batched_ode_rk4.rs:209` uses `compile_shader` not `compile_shader_f64` |

### CPU-Only Modules (Stable)

| Module | Domain | Since |
|--------|--------|-------|
| `random_forest` | RF ensemble inference | Exp061 |
| `gbm` | GBM binary + multi-class | Exp062 |
| `ani` | Average Nucleotide Identity | Exp055 |
| `snp` | SNP calling | Exp055 |
| `dnds` | Nei-Gojobori dN/dS | Exp052 |
| `molecular_clock` | Strict/relaxed clock | Exp053 |
| `pangenome` | Gene clustering + enrichment | Exp056 |

### Shared Math (`crate::special`) — MIGRATED

Promoted from `bio::special` to top-level `crate::special` in Phase 23.
`bio::special` re-export shim removed in Phase 24.

| Function | Consumers | Location | Status |
|----------|-----------|----------|--------|
| `erf()` | `normal_cdf()` | `crate::special::erf` | ✅ Live — `mul_add` chains, tested |
| `normal_cdf()` | `bio::pangenome` | `crate::special::normal_cdf` | ✅ Live — thin wrapper |
| `ln_gamma()` | `regularized_gamma_lower()` | `crate::special::ln_gamma` | ✅ Live — Lanczos, tested |
| `regularized_gamma_lower()` | `bio::dada2` | `crate::special::regularized_gamma_lower` | ✅ Live — series expansion, 1e-15 |

**Upstream extraction** (when barracuda adds `[features] math = []`):
1. ~~Delete `bio/special.rs`~~ DONE (Phase 24)
2. Change `bio/dada2.rs`: `use barracuda::special::regularized_gamma_p`
3. Change `bio/pangenome.rs`: `use barracuda::special::normal_cdf`
4. Change `bio/eic.rs`: `use barracuda::numerical::trapz`
5. Update `EVOLUTION_READINESS.md`: mark as ✅ Absorbed

---

## Migration Steps (When ToadStool Absorbs)

Following hotSpring's pattern:

1. **ToadStool absorbs** the shader as a new primitive (e.g., `ops::bio::hmm_forward`)
2. **wetSpring verifies** — run the GPU validator against the upstream primitive
3. **wetSpring rewires** — change `include_str!` to upstream `use barracuda::HmmBatchForwardF64`
4. **Move local WGSL** to `shaders_toadstool_ref/` for divergence tracking
5. **Delete GPU module** if fully replaced (or simplify to thin wrapper)
6. **Update EVOLUTION_READINESS.md** — change tier from A → ✅ Absorbed
7. **Update this file** — move from Active to Absorbed table

---

## Deprecation Candidates (Pending Upstream)

0 local WGSL shaders. All 12 absorbed by ToadStool (sessions 31d/31g + 39-41).
Lean phase complete. ODE blocker resolved (S41 fixed `compile_shader_f64`).
