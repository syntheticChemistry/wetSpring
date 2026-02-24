# Deprecation & Migration Tracking

**Date:** February 23, 2026
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

### Local WGSL Shaders (5 — Write phase, absorption candidates)

| Shader | Module | GPU Checks | Since | Target |
|--------|--------|:----------:|-------|--------|
| `phage_defense_ode_rk4_f64.wgsl` | `phage_defense_gpu` | 4v/11p | Exp099 | `BatchedOdeRK4Generic<4,11>` |
| `bistable_ode_rk4_f64.wgsl` | `bistable_gpu` | 5v/21p | Exp100 | `BatchedOdeRK4Generic<5,21>` |
| `multi_signal_ode_rk4_f64.wgsl` | `multi_signal_gpu` | 7v/24p | Exp100 | `BatchedOdeRK4Generic<7,24>` |
| `cooperation_ode_rk4_f64.wgsl` | `cooperation_gpu` | 4v/13p | Exp101 | `BatchedOdeRK4Generic<4,13>` |
| `capacitor_ode_rk4_f64.wgsl` | `capacitor_gpu` | 6v/16p | Exp101 | `BatchedOdeRK4Generic<6,16>` |

All use `compile_shader_f64()` with `fmax`/`fclamp`/`fpow` polyfills.

### CPU-Only Modules (no GPU benefit)

| Module | Domain | Reason |
|--------|--------|--------|
| `phred` | Phred quality decode/encode | I/O-bound, no parallelism benefit |

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

5 local WGSL ODE shaders pending ToadStool absorption as `BatchedOdeRK4Generic<N_VARS, N_PARAMS>`.
All 12 non-ODE shaders absorbed by ToadStool (sessions 31d/31g + 39-41).
Write phase active. ODE blocker resolved (S41 fixed `compile_shader_f64`).
All other CPU modules now have GPU wrappers (Lean, Compose, or Passthrough).
