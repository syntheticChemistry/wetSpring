# wetSpring wateringHole

**Date:** March 24, 2026  
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V137** | `handoffs/WETSPRING_V137_PROVENANCE_TOLERANCE_IPC_HANDOFF_MAR24_2026.md` | Mar 24 | **barraCuda + toadStool + spring teams + primal teams** — full provenance headers (355/355 binaries), 8 new named tolerance constants (242 total), `ipc/connection.rs` extraction, evolution intelligence (provenance header pattern, tolerance naming convention, IPC decomposition), upstream asks (BatchReconcileGpu, DF64 GEMM, Jacobi eigen, BandwidthTier, BatchMergePairsGpu), primal coordination patterns. |
| | *Superseded → `handoffs/archive/`* | | V135, V134, V133 and earlier are archived (**146** files). |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `ECOSYSTEM_LEVERAGE_GUIDE.md` | What wetSpring absorbs from ecosystem and contributes back |
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (784+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — **V135 and earlier** are archived (**146** files).  
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:  
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).  
No reverse dependencies.
