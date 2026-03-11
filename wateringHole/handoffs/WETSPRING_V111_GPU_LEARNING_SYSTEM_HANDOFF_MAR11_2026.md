# wetSpring V111 — GPU Learning System Handoff

**Date:** March 11, 2026
**From:** wetSpring V111 (ecoPrimals)
**To:** toadStool, barraCuda, coralReef, hotSpring teams
**Scope:** barraCuda v0.3.5 upstream rewire + GPU learning system experiments (Exp357-360)

---

## Summary

wetSpring V111 rewires from barraCuda v0.3.3 to v0.3.5, consuming PrecisionBrain, HardwareCalibration, FmaPolicy, stable specials (log1p, expm1, erfc, bessel_j0_minus1), and the tridiagonal QL eigensolver for Anderson localization. Four new experiments validate the GPU learning system infrastructure on the RTX 4070 (Ada Lovelace, SM89).

## New Experiments (88 checks, all PASS)

| Exp | Name | Checks | Status |
|:---:|------|:------:|:------:|
| 357 | GPU Hardware Discovery + PrecisionBrain v1 | 24 | PASS |
| 358 | Workload Routing + VRAM-Aware Dispatch v1 | 18 | PASS |
| 359 | Stable GPU Specials + Tridiag Eigensolver v1 | 34 | PASS |
| 360 | Sovereign Dispatch Readiness v1 | 12 | PASS |

## Key Findings for Upstream Teams

### barraCuda Team

1. **PrecisionBrain routing on RTX 4070:** All bio domains (Bioinformatics, Statistics, PopulationPk, Hydrology, General) route to F32. This is correct — NVVM transcendental risk makes DF64/F64/F64Precise unsafe for transcendentals on NVIDIA proprietary.

2. **HardwareCalibration on Ada Lovelace (SM89):**
   - F32: compiles ✓, dispatches ✓, transcendentals safe ✓
   - DF64: compiles ✓, dispatches ✓, transcendentals UNSAFE
   - F64: compiles ✓, dispatches ✓, transcendentals UNSAFE
   - F64Precise: compiles ✓, dispatches ✓, transcendentals UNSAFE
   - `nvvm_transcendental_risk: true`
   - `best_f64_tier: None` (no safe f64 tier)

3. **Stable specials validated:** `log1p_f64`, `expm1_f64`, `erfc_f64`, `bessel_j0_minus1_f64` all avoid catastrophic cancellation at x=1e-15. `erfc_f64` has ~1e-9 accuracy (adequate for bio workloads).

4. **ESNConfig breaking change:** v0.3.5 adds `sgd_learning_rate`, `sgd_min_iterations`, `sgd_max_iterations` fields. Not in BREAKING_CHANGES.md but breaks struct initialization. wetSpring added defaults (0.01, 50, 1000).

5. **Anderson tridiagonal QL:** Bandwidth scales monotonically with disorder W (validated W=1..30). `anderson_diagonalize` convenience wrapper works cleanly.

### toadStool Team

1. **WorkloadRouter thresholds for bio:** Using dispatch overhead ~100μs and GPU throughput ~30 GFLOPS, CPU→GPU crossover for Shannon diversity at N≈50, Bray-Curtis pairwise at N≈100, ODE integration at N≈10, HMM forward at N≈20. GPU advantage at N=10K: 40–60×.

2. **VRAM ceiling on 12GB RTX 4070:** max pairwise distance matrix N≈39K samples. Metagenome scale (1M) would require ~8TB for pairwise — needs streaming or tiled approach.

3. **biomeOS/NUCLEUS readiness:** No primals found in expected paths (standalone mode). Tower/Node/Nest all report "needs primals". This is expected for local builds without biomeOS deployment.

### coralReef Team

1. **Sovereign dispatch not available:** `sovereign-dispatch` feature requires `coral-gpu` crate which isn't published to crates.io. Without it, the sovereign path can't compile in.

2. **When available:** The `Device::Sovereign` enum variant and `sovereign_available()` probe are ready to consume. wetSpring will probe at startup and gracefully degrade to wgpu/Vulkan.

3. **Expected on NVIDIA proprietary:** DRM EINVAL (same as hotSpring Exp 051). Sovereign dispatch works on Nouveau UAPI.

### hotSpring Team

1. **PrecisionBrain pilot absorbed:** wetSpring now exercises the same PrecisionBrain→HardwareCalibration→FmaPolicy flow that hotSpring pioneered. Bio domains confirmed F32-only on RTX 4070.

2. **Stable specials consumed:** All four stable specials (log1p, expm1, erfc, bessel_j0_minus1) absorbed and validated for bio workloads.

3. **Tridiagonal QL for Anderson:** Direct connection to wetSpring's Anderson localization thesis — eigenvalue spectrum now computable on GPU via barraCuda v0.3.5.

## Upstream Pin

| Component | Version | Commit |
|-----------|---------|--------|
| barraCuda | v0.3.5 | `0649cd0` |
| toadStool | S146 | `751b3849` |
| coralReef | Iter 33 | (latest pull) |

## Absorption Targets for Upstream

| Primitive | Current | Target |
|-----------|---------|--------|
| `PrecisionBrain::route(Bioinformatics)` | Consumed | Absorption complete |
| `HardwareCalibration::from_device()` | Consumed | Absorption complete |
| `FmaPolicy` + `domain_requires_separate_fma` | Consumed | Absorption complete |
| `log1p_f64`, `expm1_f64`, `erfc_f64`, `bessel_j0_minus1_f64` | Consumed | Used in Exp359 |
| `tridiagonal_ql` + `anderson_diagonalize` | Consumed | Used in Exp359 |
| `Device::Sovereign` | Feature-gated | Awaiting coral-gpu availability |
| `WorkloadRouter::route_with_vram()` | Not available (toadStool IPC) | Future: toadStool daemon integration |

## Files Changed

| File | Change |
|------|--------|
| `barracuda/src/bio/esn/toadstool_bridge.rs` | Added SGD fields to ESNConfig |
| `barracuda/src/bin/validate_precision_brain_v1.rs` | NEW — Exp357 |
| `barracuda/src/bin/validate_workload_routing_v1.rs` | NEW — Exp358 |
| `barracuda/src/bin/validate_stable_specials_v1.rs` | NEW — Exp359 |
| `barracuda/src/bin/validate_sovereign_dispatch_v1.rs` | NEW — Exp360 |
| `barracuda/Cargo.toml` | 4 new [[bin]] entries |
| 11 doc files | V110 → V111 status updates |

## Metrics

| Metric | V110 | V111 | Delta |
|--------|------|------|-------|
| Experiments | 356 | 360 | +4 |
| Checks | 9,686+ | 9,774+ | +88 |
| Tests | 1,611 | 1,611 | +0 |
| Binaries | 340 | 344 | +4 |
| barraCuda | v0.3.3 | v0.3.5 | 20 commits |
| toadStool | S130+ | S146 | 126 files |
| coralReef | Phase 10 | Iter 33 | 46/46 sovereign |
