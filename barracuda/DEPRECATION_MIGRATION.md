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

### Local WGSL Shaders (0 — Lean COMPLETE)

All 5 ODE shaders **deleted** — replaced by `BatchedOdeRK4::<S>::generate_shader()`
via `OdeSystem` trait implementations. Absorption complete.

| Shader (deleted) | Module | Status |
|-------------------|--------|--------|
| ~~`phage_defense_ode_rk4_f64.wgsl`~~ | `phage_defense_gpu` | **Lean** — uses `generate_shader()` |
| ~~`bistable_ode_rk4_f64.wgsl`~~ | `bistable_gpu` | **Lean** — uses `generate_shader()` |
| ~~`multi_signal_ode_rk4_f64.wgsl`~~ | `multi_signal_gpu` | **Lean** — uses `generate_shader()` |
| ~~`cooperation_ode_rk4_f64.wgsl`~~ | `cooperation_gpu` | **Lean** — uses `generate_shader()` |
| ~~`capacitor_ode_rk4_f64.wgsl`~~ | `capacitor_gpu` | **Lean** — uses `generate_shader()` |

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
4. **Delete local WGSL** — upstream is the source of truth (no local shaders remain)
5. **Delete GPU module** if fully replaced (or simplify to thin wrapper)
6. **Update EVOLUTION_READINESS.md** — change tier from A → ✅ Absorbed
7. **Update this file** — move from Active to Absorbed table

---

## Deprecation Status (Resolved)

All deprecation candidates have been resolved as of Phase 41 (S59):

- **WGSL shaders:** Zero local shaders remain. All 17 absorbed by ToadStool or replaced by `generate_shader()`.
- **CPU math:** NMF (482 lines), ODE systems (715 lines), Anderson builder (~115 lines), ridge (~100 lines) — all deleted, leaning upstream.
- **Special functions:** `erf`, `ln_gamma`, `regularized_gamma_lower` delegate to `barracuda::special` on GPU; local fallback for no-GPU builds.
- **`shaders/` directory:** Removed (was empty after lean).

No pending deprecation candidates.
