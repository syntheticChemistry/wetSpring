# Deprecation & Migration Tracking

**Date:** February 20, 2026
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

---

## Active Local Extensions (Not Yet Absorbed)

These are wetSpring-specific GPU modules and shaders that ToadStool has not
yet absorbed. They remain active and maintained in wetSpring.

### Local WGSL Shaders (Tier A — absorption candidates)

| Shader | Module | GPU Checks | Since |
|--------|--------|:----------:|-------|
| `quality_filter.wgsl` | `quality_gpu` | 88 (pipeline) | Exp016 |
| `dada2_e_step.wgsl` | `dada2_gpu` | 88 (pipeline) | Exp016 |
| `hmm_forward_f64.wgsl` | `hmm_gpu` | 13 | Exp047 |
| `batched_qs_ode_rk4_f64.wgsl` | `ode_sweep_gpu` | 7 | Exp049 |
| `ani_batch_f64.wgsl` | `ani_gpu` | 7 | Exp058 |
| `snp_calling_f64.wgsl` | `snp_gpu` | 5 | Exp058 |
| `dnds_batch_f64.wgsl` | `dnds_gpu` | 9 | Exp058 |
| `pangenome_classify.wgsl` | `pangenome_gpu` | 6 | Exp058 |
| `rf_batch_inference.wgsl` | `random_forest_gpu` | 13 | Exp063 |

### CPU-Only Modules (New Since Last Absorption)

| Module | Domain | Since |
|--------|--------|-------|
| `random_forest` | RF ensemble inference | Exp061 |
| `gbm` | GBM binary + multi-class | Exp062 |
| `ani` | Average Nucleotide Identity | Exp055 |
| `snp` | SNP calling | Exp055 |
| `dnds` | Nei-Gojobori dN/dS | Exp052 |
| `molecular_clock` | Strict/relaxed clock | Exp053 |
| `pangenome` | Gene clustering + enrichment | Exp056 |

### Shared Math (bio::special — extraction candidate)

| Function | Consumers | barracuda Target | Extraction Status |
|----------|-----------|------------------|-------------------|
| `erf()` | `normal_cdf()` | `barracuda::special::erf` | Ready — `mul_add` chains, tested |
| `normal_cdf()` | `bio::pangenome` | `barracuda::special::normal_cdf` | Ready — thin wrapper |
| `ln_gamma()` | `regularized_gamma_lower()` | `barracuda::special::ln_gamma` | Ready — Lanczos, tested |
| `regularized_gamma_lower()` | `bio::dada2` | `barracuda::special::regularized_gamma_p` | Ready — series expansion, 1e-15 |

**Extraction plan**: When barracuda adds `[features] math = []`:
1. Delete `bio/special.rs`
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

When ToadStool next absorbs, these are the highest-value candidates:

| Priority | Shader | Why |
|----------|--------|-----|
| **P1** | `hmm_forward_f64.wgsl` | 13/13 validated, clean batch API, log-space numerics |
| **P1** | `batched_qs_ode_rk4_f64.wgsl` | Fixes upstream `enable f64;` bug |
| **P2** | `rf_batch_inference.wgsl` | SoA layout, (sample,tree) parallelism |
| **P2** | `ani_batch_f64.wgsl` | Simple integer counting + f64 division |
| **P2** | `snp_calling_f64.wgsl` | Per-position parallel, allele frequency |
| **P2** | `dnds_batch_f64.wgsl` | Complex (genetic code, `log()` polyfill) |
| **P3** | `pangenome_classify.wgsl` | Presence/absence counting |
| **P3** | `dada2_e_step.wgsl` | Pipeline-specific, needs generalization |
| **P3** | `quality_filter.wgsl` | Pipeline-specific, u32 only |
