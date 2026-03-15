# wetSpring V115 → BarraCUDA/ToadStool Deep Audit Evolution Handoff

**Date:** March 15, 2026
**From:** wetSpring V115 (375 experiments, 5,707+ checks, 1,662 tests, 354 binaries)
**To:** BarraCUDA/ToadStool team
**Authority:** wateringHole (ecoPrimals Core Standards)
**Supersedes:** V114 Deep Audit Handoff (Mar 12), V114 Niche Setup Guidance (Mar 15)

---

## What Changed in V115

wetSpring executed a comprehensive 12-finding deep audit covering code quality,
architecture, and ecosystem integration. All changes verified with zero clippy
warnings (pedantic + nursery) and 1,662 tests passing.

### 1. UniBin Binary Compliance

The `wetspring` binary now follows the ecoBin standard:

```
wetspring           # runs server (default)
wetspring server    # explicit server mode
wetspring status    # reports socket, Songbird, capabilities
wetspring version   # version string
wetspring help      # usage with capability list
```

**Impact on BarraCUDA:** If BarraCUDA exposes an IPC surface, adopt the same
`server|status|version` subcommand pattern. The `status` command should list
all registered capabilities dynamically from the capability domain module.

### 2. Capability Domain Architecture

New `ipc/capability_domains.rs` defines 19 capabilities across 4 domains:
- `ecology.*` — diversity, QS model, full pipeline
- `science.*` — kinetics, alignment, taxonomy, phylogenetics, NMF, timeseries
- `provenance.*` — begin, record, complete (rhizoCrypt/loamSpine/sweetGrass)
- `brain.*` — classify, predict, train

Machine-readable `capability_registry.toml` with per-capability metadata:
domain, description, GPU acceleration status, input/output schemas.

**Impact on BarraCUDA:** When BarraCUDA adds IPC capabilities, use the same
domain hierarchy pattern. `math.*` and `shader.*` are natural BarraCUDA domains.

### 3. Tolerance Centralization

All inline tolerance literals eliminated from handler code:
- `NMF_CONVERGENCE = 1e-6` (IPC handler iteration threshold)
- `NMF_CONVERGENCE_LOOSE = 1e-4` (relaxed convergence for large matrices)
- `MATRIX_EPS = 1e-12` (epsilon guard in NMF denominators)
- `STABLE_SPECIAL_TINY = 1e-28` (stable special function precision)

**Impact on BarraCUDA:** When BarraCUDA uses tolerance thresholds, centralize
them in a tolerances module with scientific justification. wetSpring's 180+
named constants follow this pattern. Inline `1e-10` should be `ANALYTICAL_LOOSE`.

### 4. XDG Path Resolution

All hardcoded `/tmp/` paths replaced with XDG-compliant resolution:
```
$SPECIFIC_ENV_VAR → $XDG_RUNTIME_DIR → std::env::temp_dir()
```

NestGate socket discovery removed its hardcoded `/run/nestgate/default.sock`
fallback. All socket paths are now capability-discovered.

**Impact on BarraCUDA:** Any hardcoded paths should follow this pattern.
Primals discover each other at runtime, never assume fixed locations.

### 5. Python Baseline Provenance

Baseline scripts now embed SHA-256 self-hash and git commit in output JSON.
`verify_baseline_outputs.sh` automates integrity and numeric drift checking.

**Impact on BarraCUDA:** When BarraCUDA adds validation baselines from external
tools, follow this provenance pattern.

### 6. metalForge Coverage Boost

CI coverage threshold raised to 90%. 12 new tests added:
- `forge/src/inventory/output.rs`: empty, CPU-only, GPU detail, mesh, mixed
- `forge/src/data.rs`: discover fallbacks, NestGate unreachable, key escaping

**Impact on BarraCUDA:** forge crate is the absorption seam for BarraCUDA → ToadStool.
Higher coverage in forge means less risk during absorption.

## GPU Primitive Opportunities (Updated from V114)

The V114 handoff identified 8 GPU primitive opportunities. These remain valid:

| Priority | Primitive | Source | Status |
|----------|-----------|--------|--------|
| P1 | `SparseGemmF64` | NMF multiplicative update | Open — NMF needs dense + sparse GEMM |
| P1 | `AdaptiveOdeGpu` | Kinetics IPC handler | Open — stiff systems need adaptive step |
| P2 | `SmithWatermanBatchGpu` | Alignment IPC handler | Open — batched pairs per request |
| P2 | `NaiveBayesBatchGpu` | Taxonomy IPC handler | Open — k-mer probability parallel |
| P3 | `RobinsonFouldsBatchGpu` | Phylogenetics handler | Open — tree distance matrix |
| P3 | `TimeSeriesStatsGpu` | Timeseries handler | Open — mean/var/trend bulk |
| P3 | `GompertzBatchGpu` | Kinetics handler | Open — parameter sweep |
| P3 | `NMFGpu` | NMF handler end-to-end | Open — full W/H update cycle |

## Key Files Changed

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | `[[bin]] name = "wetspring"` (was `wetspring_server`) |
| `barracuda/src/bin/wetspring.rs` | UniBin main with server/status/version/help |
| `barracuda/src/ipc/capability_domains.rs` | New — 19 capabilities, 4 domains |
| `barracuda/src/ipc/handlers/expanded.rs` | Inline tolerances → `tolerances::` constants |
| `barracuda/src/tolerances/mod.rs` | NMF_CONVERGENCE, MATRIX_EPS, STABLE_SPECIAL_TINY |
| `barracuda/src/bin/validate_primal_pipeline_v1.rs` | XDG path resolution |
| `barracuda/src/bin/validate_qs_gene_profiling_v1.rs` | QsType bitflag refactor |
| `barracuda/src/bin/validate_barracuda_cpu_v27.rs` | Baseline provenance table |
| `metalForge/forge/src/nest/discovery.rs` | XDG socket discovery |
| `metalForge/forge/src/inventory/output.rs` | 6 new coverage tests |
| `metalForge/forge/src/data.rs` | 6 new coverage tests |
| `capability_registry.toml` | New — machine-readable capability manifest |
| `scripts/verify_baseline_outputs.sh` | New — automated baseline verification |
| `.github/workflows/ci.yml` | metalForge coverage 80% → 90% |

## Metrics

| Metric | V114 | V115 | Delta |
|--------|------|------|-------|
| Tests | 1,621 | 1,662 | +41 |
| Experiments | 374 | 375 | +1 |
| Binaries | 340 | 354 | +14 (count correction) |
| Validation checks | 5,707+ | 5,707+ | — |
| Named tolerances | 180 | 180+ | +3 |
| Capabilities | 19 | 19 | — |
| clippy warnings | 0 | 0 | — |
