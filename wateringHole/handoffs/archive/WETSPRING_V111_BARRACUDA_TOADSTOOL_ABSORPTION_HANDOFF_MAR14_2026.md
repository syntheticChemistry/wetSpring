# wetSpring V111: barraCuda + toadStool Absorption Handoff

**Date:** 2026-03-14
**From:** wetSpring V111 (life science + analytical chemistry)
**To:** barraCuda team (standalone math primal), toadStool team (hardware dispatch)

---

## Purpose

This handoff documents what wetSpring has learned, built, and validated that
should flow upstream into barraCuda and toadStool. It covers absorption
candidates, API gaps, evolution blockers, and cross-spring patterns.

---

## Absorption Candidates for barraCuda

### Tier 1: Direct Extraction (low effort, high value)

| Item | Location | Rationale |
|------|----------|-----------|
| Gompertz kinetics | 12+ validation bins (`fn gompertz()`) | Identical biogas kinetics function duplicated everywhere. Add to `barracuda::numerical::kinetics`. |
| First-order decay | 12+ validation bins (`fn first_order()`) | Same — `B_max * (1 - exp(-k*t))`. |
| Monod growth | 8+ validation bins (`fn monod()`) | Michaelis-Menten form. |
| Haldane inhibition | 6+ validation bins (`fn haldane()`) | Substrate inhibition kinetics. |
| `dot` / `l2_norm` (f64 slice) | `barracuda/src/special.rs` | wetSpring has CPU f64 slice versions; barracuda only has GPU tensor versions. |

### Tier 2: Module Promotion (medium effort)

| Item | Location | Rationale |
|------|----------|-----------|
| Streaming validation harness | `barracuda/src/validation/` | Generic hotSpring-pattern validator (check, check_pass, finish, exit codes). Multiple springs would benefit. |
| Tolerance hierarchy | `barracuda/src/tolerances/` | 180+ named constants with scientific justification. Cross-spring standard candidate. |
| Sovereign base64 | `barracuda/src/encoding.rs` | Zero-dependency RFC 4648 base64. Already duplicated in some springs. |

### Tier 3: Evolution Targets (requires design discussion)

| Item | Discussion |
|------|-----------|
| `shaders::provenance` un-gating | Currently GPU-feature-gated. Provenance is metadata — should be available in CPU-only builds. |
| Bio ODE trait system | wetSpring's 5 ODE systems use `BatchedOdeRK4<S>::generate_shader()` from toadStool. The trait pattern works well and could be documented as the standard for new ODE domains. |
| Visualization schema types | `DataChannel`, `EcologyScenario`, `StreamSession` — petalTongue types that could become a shared schema crate. |

---

## Absorption Candidates for toadStool

### metalForge → toadStool

The `wetspring-forge` crate (v0.3.0) is the absorption seam:

| Module | Purpose | Absorption Path |
|--------|---------|-----------------|
| `probe.rs` | GPU/CPU/NPU discovery (wgpu + /proc + /dev) | Merge into toadStool's existing probe infrastructure |
| `inventory.rs` | Unified substrate view | Align with toadStool's device inventory |
| `dispatch.rs` | Capability-based workload routing | Map to toadStool's `compute_dispatch` |
| `streaming.rs` | Multi-stage pipeline analysis | Map to toadStool's pipeline framework |
| `bridge.rs` | forge ↔ barracuda device bridge | This IS the integration point |
| `workloads/` | Preset bio workloads with `ShaderOrigin` | Feed into toadStool workload catalog |

### bingocube-nautilus Evolution

- Created at `primalTools/bingoCube/nautilus/` (v0.1.0)
- API: `NautilusBrain`, `BetaObservation`, `ShellConfig`, `ReservoirInput`
- Used by wetSpring (`BioNautilusBrain`) and hotSpring (physics brain)
- Needs evolution: proper board population genetics, tournament selection,
  LOO cross-validation with real residual analysis
- hotSpring's 4-layer brain architecture should inform the evolution

---

## GPU Promotion Readiness

All 47 GPU modules are operational. Tier mapping from `specs/BARRACUDA_REQUIREMENTS.md`:

| Tier | Modules | Status |
|------|:-------:|--------|
| A (Lean) | 22 | Ready — direct toadStool primitive use |
| B (Compose) | 11 | Ready — multiple primitives with wiring |
| C (Write) | 0 | Complete — all absorbed upstream |

**Local WGSL shaders: 0** (fully lean)

---

## API Gaps Discovered

1. **`barracuda::linalg::eigh` CPU-only** — gated behind GPU feature, but
   wetSpring's spectral analysis (`spectral/stats.rs` upstream) needs eigensolvers
   on CPU path too. Currently blocks `--no-default-features` builds.

2. **`FitResult` unnamed fields** — `barracuda::stats::FitResult` uses a params
   array instead of named fields. Springs must know the index convention
   (slope=0, intercept=1). Named fields would improve ergonomics.

3. **`graph_laplacian` flat-array API** — returns flat `Vec<f64>` instead of a
   matrix type. Springs reconstruct 2D layout manually.

---

## Cross-Spring Patterns Learned

1. **Gompertz is universal** — biogas, growth curves, dose-response (IC50),
   pharmacokinetics all use the same modified Gompertz. Should be in barracuda.

2. **Anderson localization applies to biology** — QS signaling in confined
   geometries (pores, biofilms, skin layers) exhibits Anderson-like localization.
   The mapping W → H' (disorder → diversity) is well-validated.

3. **Streaming > round-trip** — GPU streaming pipelines (441-837x over
   round-trip) are the correct abstraction for bio workloads that chain
   many small operations. toadStool's pipeline framework should be the
   standard path for all springs.

4. **Tolerance hierarchies save debugging time** — 180+ named constants with
   scientific justification + hierarchy test prevents tolerance creep.
   Recommend ecosystem-wide adoption.

---

## Validation Chain Position

```
Paper Math → Python → R Industry → BarraCuda CPU → GPU → Streaming → metalForge → NUCLEUS
   52 papers   71 scripts  3 tools     v27 (47 dom)   v14    v13          v19         v4
```

All tiers green. 5,707+ checks PASS. 1,621 tests PASS.
