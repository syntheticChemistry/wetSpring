# wetSpring wateringHole

**Date:** March 14, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V112** | `handoffs/WETSPRING_V112_STREAMING_PEDANTIC_CAPABILITY_HANDOFF_MAR14_2026.md` | Mar 14 | **Streaming-only I/O + capability discovery handoff** — deprecated buffering parsers removed, streaming iterator pattern recommended, capability-based runtime discovery (`$PATH`/`$XDG_RUNTIME_DIR`), zero clippy warnings, tolerance hierarchy, upstream requests (P0-P2), pre-existing failure documentation. |
| **V114** | `handoffs/WETSPRING_V114_DEEP_AUDIT_BARRACUDA_TOADSTOOL_HANDOFF_MAR12_2026.md` | Mar 12 | **Deep audit handoff** — 15 `required-features` gate fixes, 52 clippy warnings resolved, deprecated parsers migrated, inline tolerances eliminated, VRAM capability-based, code deduplicated. Absorption status (fully lean), upstream requests (P0-P2 carried forward), API observations, Python baseline audit, evolution roadmap. |
| **V113** | `handoffs/WETSPRING_V113_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR11_2026.md` | Mar 11 | **Upstream evolution handoff** — 150+ primitives, P0/P1/P2 absorption targets (BIOM parser, esearch_ids, Gompertz fit, hw-learn profile consumer, SRA prefetch, temporal W, regulon DB, distributed Anderson graph), API observations, hardware learning findings (RTX 4070 F32/DF64, nouveau status), science pipeline architecture, deployment feedback. |
| **V113** | `handoffs/WETSPRING_V113_PAPER_EXTENSION_ROADMAP_HANDOFF_MAR11_2026.md` | Mar 11 | Paper extension roadmap (Exp364-370, 67/67). EMP 28K atlas, Liao real data, KBS LTER temporal, QS gene profiling (FNR/ArcAB/Rex regulons), primal integration pipeline, P1 framework (cold seep/Tara/HMP/AMR/mycorrhizal), LAN mesh SRA atlas plan (5 towers, 96GB VRAM, 208 TFLOPS). |
| **V112** | `handoffs/WETSPRING_V112_NVIDIA_HARDWARE_LEARNING_HANDOFF_MAR11_2026.md` | Mar 11 | NVIDIA hardware learning prototype (Exp361-363, 45/45). Probe-calibrate-route-apply pattern. Dual-GPU discovery (RTX 4070 nvidia + Titan V nouveau), firmware inventory, nouveau diagnostic (VM_INIT OK, CHANNEL_ALLOC blocked on Volta), capability profile JSON, adaptive dispatch from profile. |
| **V111** | `handoffs/WETSPRING_V111_BARRACUDA_TOADSTOOL_ABSORPTION_HANDOFF_MAR14_2026.md` | Mar 14 | barraCuda/toadStool absorption handoff from V111 deep debt resolution. |
| **V111** | `handoffs/WETSPRING_V111_DEEP_DEBT_EVOLUTION_HANDOFF_MAR14_2026.md` | Mar 14 | V111 deep debt evolution handoff — build health, clippy pedantic, bingocube-nautilus. |
| | *V109 and earlier → `handoffs/archive/`* | | Fossil record: V7–V109 (89+ archived handoffs) |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V111 (89+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
