# wetSpring wateringHole

**Date:** March 24, 2026  
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V135** | `handoffs/WETSPRING_V135_BARRACUDA_TOADSTOOL_ABSORPTION_HANDOFF_MAR24_2026.md` | Mar 24 | **barraCuda + toadStool** — doc reconciliation, canonical metrics (1,891 tests, 355 binaries, 91.20% coverage), V134 audit themes (drug NMF delegation, validation harness refactor, 7-primal discovery, CI feature-matrix, SPDX), upstream asks (BatchReconcileGpu, DF64 GEMM, Jacobi eigen, BandwidthTier, BatchMergePairsGpu), evolution intelligence (harness decomposition pattern, primal discovery cascade, workspace forbid, `#[expect(reason)]`). |
| **V133** | `handoffs/WETSPRING_V133_DEEP_EVOLUTION_CROSS_ECOSYSTEM_ABSORPTION_HANDOFF_MAR23_2026.md` | Mar 23 | **Deep evolution (all springs / primals)** — `validate_all`, `GpuContext`/`TensorSession`, `check_relative`/`check_abs_or_rel`, zero-copy I/O, `performance_surface` complete, IPC refactors, feature-gate cleanup, release/coverage tooling. |
| | *Superseded → `handoffs/archive/`* | | V134, V132, V130 and earlier are archived (**144** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (784+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — **V134 and earlier** are archived (**144** files).  
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:  
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).  
No reverse dependencies.
