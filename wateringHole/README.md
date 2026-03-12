# wetSpring wateringHole

**Date:** March 12, 2026
**Purpose:** Spring-local handoff documents to `barraCuda`/`toadStool` and cross-spring provenance records.

---

## Active Handoffs

| Version | File | Date | Scope |
|---------|------|------|-------|
| **V114** | `handoffs/WETSPRING_V114_DEEP_AUDIT_BARRACUDA_TOADSTOOL_HANDOFF_MAR12_2026.md` | Mar 12 | **Deep audit handoff** — 15 `required-features` gate fixes, 52 clippy warnings resolved, deprecated parsers migrated, inline tolerances eliminated, VRAM capability-based, code deduplicated. Absorption status (fully lean), upstream requests (P0-P2 carried forward), API observations, Python baseline audit, evolution roadmap. |
| **V113** | `handoffs/WETSPRING_V113_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR11_2026.md` | Mar 11 | **Upstream evolution handoff** — 150+ primitives, P0/P1/P2 absorption targets (BIOM parser, esearch_ids, Gompertz fit, hw-learn profile consumer, SRA prefetch, temporal W, regulon DB, distributed Anderson graph), API observations, hardware learning findings (RTX 4070 F32/DF64, nouveau status), science pipeline architecture, deployment feedback. |
| **V113** | `handoffs/WETSPRING_V113_PAPER_EXTENSION_ROADMAP_HANDOFF_MAR11_2026.md` | Mar 11 | Paper extension roadmap (Exp364-370, 67/67). EMP 28K atlas, Liao real data, KBS LTER temporal, QS gene profiling (FNR/ArcAB/Rex regulons), primal integration pipeline, P1 framework (cold seep/Tara/HMP/AMR/mycorrhizal), LAN mesh SRA atlas plan (5 towers, 96GB VRAM, 208 TFLOPS). |
| **V112** | `handoffs/WETSPRING_V112_NVIDIA_HARDWARE_LEARNING_HANDOFF_MAR11_2026.md` | Mar 11 | NVIDIA hardware learning prototype (Exp361-363, 45/45). Probe-calibrate-route-apply pattern. Dual-GPU discovery (RTX 4070 nvidia + Titan V nouveau), firmware inventory, nouveau diagnostic (VM_INIT OK, CHANNEL_ALLOC blocked on Volta), capability profile JSON, adaptive dispatch from profile. |
| **V111** | `handoffs/WETSPRING_V111_GPU_LEARNING_SYSTEM_HANDOFF_MAR11_2026.md` | Mar 11 | barraCuda v0.3.5 upstream rewire + GPU learning system (Exp357-360, 88/88). PrecisionBrain bio routing (F32 on RTX 4070), HW calibration (NVVM risk), stable specials (log1p/expm1/erfc/bessel), tridiagonal QL eigensolver, sovereign dispatch probe. Upstream pins: barraCuda v0.3.5 `0649cd0`, toadStool S146, coralReef Iter 33. |
| **V110** | `handoffs/WETSPRING_V110_PETALTONGUE_VIZ_PIPELINE_HANDOFF_MAR10_2026.md` | Mar 10 | petalTongue visualization pipeline (Exp353-355, 93/93) + Anderson QS cross-environment validation (Exp356, 18/18). O₂-modulated W model (H3, r=0.851). `stream_ecology` module. 6 JSON scenario artifacts. biomeOS/NUCLEUS readiness probing. Anderson W evolution target for `barracuda::bio::anderson`. |
| **V109** | `handoffs/WETSPRING_V109_UPSTREAM_REWIRE_NUCLEUS_HANDOFF_MAR10_2026.md` | Mar 10 | Upstream rewire validation (SpringDomain SCREAMING_SNAKE_CASE, sync GPU diversity, DADA2 fix) + mixed hardware (NPU→GPU PCIe bypass, CPU fallback) + NUCLEUS atomics (Tower/Node/Nest via biomeOS graph). 6 experiments (Exp347-352), 145/145 PASS. Absorption: Gompertz/Monod/Haldane → `barracuda::bio::kinetics`, `variance` re-export, spectral CPU fallback. |
| **V107** | `handoffs/WETSPRING_V107_R_INDUSTRY_PARITY_BARRACUDA_TOADSTOOL_HANDOFF_MAR10_2026.md` | Mar 10 | R industry parity baselines: R/vegan + R/DADA2 + R/phyloseq gold-standard references, 53/53 PASS, `PhyloTree::patristic_distance()`, phyloseq trifurcation bug discovered, weighted UniFrac normalization analysis, 3 new upstream absorption opportunities (CopheneticMatrixGpu, dual-normalization weighted_unifrac, GPU DADA2 error model). |
| **V106** | `handoffs/WETSPRING_V106_DEEP_DEBT_CLEANUP_HANDOFF_MAR10_2026.md` | Mar 10 | Deep debt cleanup: 112+ stale `#[expect()]` removed, `#![forbid(unsafe_code)]` on all 320 crate roots, BIOM streaming parser, validates absorption fidelity. |
| **V105** | `handoffs/WETSPRING_V105_PETALTONGUE_EVOLUTION_HANDOFF_MAR10_2026.md` | Mar 10 | petalTongue visualization evolution: LivePipelineSession, Scatter3D, 33 scenario builders, 5 new domain scenarios (MSA, calibration, spectroscopy, basecalling, NJ), 3 sample profiles, IPC 64KB, scientific ranges. 1,288 lib + 219 integration tests. |
| **V105** | `handoffs/WETSPRING_V105_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR10_2026.md` | Mar 10 | Comprehensive upstream handoff: 150+ primitives consumed, 47 GPU modules, 6 upstream requests (BipartitionEncodeGpu, CPU Jacobi, merge pairs GPU, BatchReconcileGpu), API evolution observations, visualization layer evolution. |
| **V104** | `handoffs/WETSPRING_V104_DEEP_DEBT_EVOLUTION_HANDOFF_MAR09_2026.md` | Mar 9 | Deep debt: JCAMP-DX parser, Dorado basecaller, GPU peak integration, #[allow]→#[expect] migration, 56 stale suppressions removed, 8 clippy warnings resolved. |
| **V103** | `handoffs/WETSPRING_V103_UPSTREAM_REWIRE_HANDOFF_MAR10_2026.md` | Mar 10 | Upstream rewire: #[allow]→#[expect] across 209 files, 37 stale suppressions, /tmp/→temp_dir(). |
| **V102** | *(archived — V102 handoffs superseded by V105 petalTongue + evolution handoffs)* | Mar 9 | petalTongue V2 + barraCuda evolution (content folded into V105 handoffs) |
| **V101** | `handoffs/WETSPRING_V101_DEEP_DEBT_EVOLUTION_HANDOFF_MAR09_2026.md` | Mar 9 | Deep debt evolution: 179 tolerances (was 164), all bare literals promoted, 3 binaries refactored, clippy pedantic clean, absorption manifest corrected. 1,455 tests, 316 binaries, 9,060+ checks. |
| **V99** | `handoffs/WETSPRING_V99_BIOMEOS_NUCLEUS_FULL_CHAIN_HANDOFF_MAR08_2026.md` | Mar 8 | biomeOS/NUCLEUS integration + full V99 chain: Exp321-326 (166/166 PASS). Deploy graph, IPC server, cross-primal pipeline, CPU v25 (46), GPU v14 (27), metalForge v17 (29). ToadStool dispatch + NUCLEUS atomic probes. 300 experiments, 8,886+ checks. |
| **V98+** | `handoffs/WETSPRING_V98_CROSS_SPRING_EVOLUTION_HANDOFF_MAR08_2026.md` | Mar 8 | Cross-spring evolution: Exp319 (52/52) + Exp320 benchmark. All 5 springs exercised, provenance registry (28 shaders, 22 cross-spring). GPU FusedMapReduceF64 validated. |
| **V98+** | `handoffs/WETSPRING_V98_UPSTREAM_REWIRE_HANDOFF_MAR08_2026.md` | Mar 8 | Upstream rewire: barraCuda `a898dee`, toadStool S130+ `bfe7977b`, coralReef Iteration 10 `d29a734`. Zero API breakage. V98 chain 173/173 re-validated. |
| **V98** | `handoffs/WETSPRING_V98_BARRACUDA_TOADSTOOL_FULL_CHAIN_HANDOFF_MAR07_2026.md` | Mar 7 | V98 full-chain validation (173/173), primitive inventory, GPU Hybrid findings, DF64 shader gap, absorption targets. |
| **V97e** | `handoffs/WETSPRING_V97E_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR07_2026.md` | Mar 7 | Full evolution handoff: 150+ primitive inventory, cross-spring insights, re-export suggestions, absorption feedback, next-gen targets (P1: BatchedOdeRK45F64). |
| **V97e** | `handoffs/WETSPRING_V97E_PROVENANCE_REWIRE_HANDOFF_MAR07_2026.md` | Mar 7 | Provenance rewire: builder patterns (HMM, DADA2, Gillespie), PrecisionRoutingAdvice, shaders::provenance API, Exp312 (31/31). 1,346 tests, zero warnings. |
| **V97d+** | `handoffs/WETSPRING_V97D_ECOSYSTEM_SYNC_HANDOFF_MAR07_2026.md` | Mar 7 | Ecosystem sync: barraCuda 2a6c072, toadStool S130, coralReef Phase 10. Zero API breakage, 1,347 tests PASS. |
| **V97d** | `handoffs/WETSPRING_V97D_DEEP_AUDIT_EVOLUTION_HANDOFF_MAR07_2026.md` | Mar 7 | Deep audit: I/O deprecation, unwrap→expect evolution, doc accuracy, broken ref cleanup (Exp311, 125 items) |
| **V97c** | `handoffs/WETSPRING_V97C_FUSED_OPS_CHAIN_HANDOFF_MAR05_2026.md` | Mar 5 | Fused ops full chain: Exp306-310 (111 checks). DF64 dispatch routing confirmed wired; DF64 fused shaders produce zero on RTX 4070 (shader validation gap, not wiring). |
| **V97** | `handoffs/WETSPRING_V97_BARRACUDA_033_WGPU28_REWIRE_HANDOFF_MAR05_2026.md` | Mar 5 | barraCuda v0.3.3 + wgpu 28 rewire: 1,247 tests, zero clippy, chi_squared upstream fix |
| **V96** | `handoffs/WETSPRING_V96_DEEP_DEBT_CHUNA_HANDOFF_MAR05_2026.md` | Mar 5 | Deep debt: silent fallback elimination, capability IPC, Chuna papers queued |
| **V95** | `handoffs/WETSPRING_V95_CROSS_SPRING_EVOLUTION_COMPLETE_MAR04_2026.md` | Mar 4 | Cross-spring evolution complete: 6 GPU ops + 2 CPU delegations wired, Exp305 (59/59), full provenance table, benchmarks |
| **V94** | `handoffs/WETSPRING_V94_BARRACUDA_EVOLUTION_SYNC_MAR04_2026.md` | Mar 4 | barraCuda evolution sync: norm_ppf wiring, 50+ doc files cleaned (ToadStool → barraCuda), gap analysis |
| **V93+** | `handoffs/WETSPRING_V93_DEEP_DEBT_TOADSTOOL_HANDOFF_MAR04_2026.md` | Mar 4 | Deep debt round 3: 164 tolerances, test extraction, provenance complete |
| **V93** | `handoffs/WETSPRING_V93_BARRACUDA_EVOLUTION_FEEDBACK_MAR03_2026.md` | Mar 3 | Evolution feedback: 150+ primitives consumed, precision observations |
| | *V92 and earlier → `handoffs/archive/`* | | Fossil record: V7–V92F (86 archived handoffs) |

## Cross-Spring Documents

| File | Purpose |
|------|---------|
| `CROSS_SPRING_SHADER_EVOLUTION.md` | Cross-spring shader provenance map (767+ barraCuda WGSL shaders) |
| `TOADSTOOL_WETSPRING_GAP_ANALYSIS.md` | Gap analysis: barraCuda exports vs wetSpring usage |

## Archive

Superseded handoffs in `handoffs/archive/` — V7–V92J ToadStool-era handoffs (80+ files).
Preserved as fossil record of the evolution from ToadStool-embedded to standalone barraCuda.

## Convention

Following hotSpring's naming pattern:
`WETSPRING_{VERSION}_{TOPIC}_HANDOFF_{DATE}.md`

Handoffs flow: wetSpring → barraCuda (math) and wetSpring → toadStool (hardware).
No reverse dependencies.
