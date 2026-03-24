<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->

# NUCLEUS Local Deployment Plan — wetSpring V108+

**Date:** March 10, 2026
**Foundation:** wetSpring V108 (346 experiments, 9,430+ checks, 63 papers), biomeOS IPC validated (Exp203-208, 321 checks), NUCLEUS probes validated (Exp258, Exp266, Exp270)

---

## What Already Works in wetSpring

| Component | Status | Location |
|-----------|--------|----------|
| IPC server | Working | `barracuda/src/ipc/server.rs` |
| NestGate discovery | Working | `barracuda/src/ncbi/nestgate/discovery.rs` |
| NestGate storage | Working | `barracuda/src/ncbi/nestgate/storage.rs` |
| NCBI 3-tier fetch | Working | `barracuda/src/ncbi/nestgate/fetch.rs` |
| NUCLEUS probes | Working | `validate_nucleus_tower_node.rs` (Exp258) |
| biomeOS detection | Working | Binary discovery in PATH |
| JSON-RPC 2.0 handlers | Working | diversity, QS, Anderson, pipeline, brain |
| GPU dispatch via ToadStool | Working | 150+ primitives, Fp64Strategy |
| NPU inference via AKD1000 | Working | Exp193-195, 3 ESN classifiers |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Local Machine — Eastgate                │
│                                                           │
│  biomeOS (orchestrator)                                   │
│    ├── Tower Atomic                                       │
│    │     ├── BearDog (crypto, TLS 1.3, FIDO2)           │
│    │     └── Songbird (discovery, mesh routing)           │
│    ├── Node Atomic                                        │
│    │     ├── ToadStool (GPU dispatch, 844 WGSL shaders)  │
│    │     └── BarraCuda (math: CPU + GPU + NPU)           │
│    └── Nest Atomic                                        │
│          └── NestGate (storage, NCBI provider)           │
│                                                           │
│  wetSpring (science workloads)                            │
│    ├── 16S pipeline (DADA2, diversity, taxonomy)         │
│    ├── Anderson-QS framework                              │
│    ├── Track 6 kinetics (Gompertz/Monod/Haldane)         │
│    └── metalForge cross-substrate validation              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   LAN — Future (10G backbone)             │
│                                                           │
│  Northgate ── RTX 5090 (32GB), Node Atomic               │
│  Southgate ── RTX 3090 (24GB), Node Atomic               │
│  Strandgate ─ RTX 3090 + RX 6950 XT, Node Atomic        │
│  biomeGate ── RTX 3090 + Titan V, Node Atomic            │
│  Westgate ─── 76TB ZFS, Nest Atomic (cold storage)       │
│                                                           │
│  Plasmodium: multi-tower coordination layer               │
└─────────────────────────────────────────────────────────┘
```

---

## Phase A: Single-Machine NUCLEUS (Now)

### Deployment Steps

1. **Build biomeOS** from `phase2/biomeOS/` on Eastgate
2. **Start Tower Atomic** (BearDog + Songbird)
   - BearDog: local crypto, no TLS needed for single-machine
   - Songbird: local-only discovery (no mesh yet)
3. **Start Nest Atomic** (NestGate with NCBI provider)
   - Wire `WETSPRING_DATA_PROVIDER=nestgate`
   - NestGate handles ESearch/ESummary/EFetch with local cache
   - Content-addressed blob storage on local disk
4. **Start Node Atomic** (ToadStool GPU dispatch)
   - RTX 4070: primary GPU compute
   - AKD1000: NPU inference (4× Akida on Eastgate)
5. **Wire wetSpring data pipeline through NUCLEUS**
   - `data.ncbi_search` → NestGate JSON-RPC
   - `data.ncbi_fetch` → NestGate with sovereign HTTP fallback
   - GPU dispatch → ToadStool IPC
6. **Validate**: Run Exp184 (real NCBI sovereign pipeline) through NUCLEUS stack

### What This Enables

- Orchestrated EMP 30K sample download + processing
- NestGate-cached NCBI lookups (no re-download on retry)
- biomeOS graph execution for multi-step pipelines
- Foundation for LAN expansion (same NUCLEUS model, just add towers)

### Compute Budget: Phase A

| Workload | CPU Time | GPU Time | Storage |
|----------|----------|----------|---------|
| Liao group supplementary data | minutes | — | <1GB |
| EMP 30K OTU tables | 4h diversity | 20 min Anderson | <10GB |
| Cold seep 170 metagenomes | 2h diversity | 15 min Anderson | ~5GB |
| KBS LTER time series | 2h DADA2 | 30 min Anderson | ~200GB raw, ~5GB processed |
| Track 6 on real NCBI communities | minutes | minutes | <1GB |
| **Total Phase A** | **~9h** | **~1h** | **~220GB** |

---

## Phase B: Multi-Tower LAN (After 10G Cables)

### Prerequisites

- 10G Cat6a cables between towers (~$50)
- 10G switch (acquired)
- 10G NICs (installed in all towers)

### Deployment Steps

1. Install 10G cables: Northgate ↔ Switch ↔ Eastgate ↔ Strandgate ↔ Westgate
2. Start NUCLEUS on each tower:
   - `biomeos nucleus start --mode node` on Northgate, Strandgate, biomeGate
   - `biomeos nucleus start --mode nest` on Westgate
   - `biomeos nucleus start --mode full` on Eastgate (hub)
3. Songbird mesh discovery → Plasmodium coordination
4. Capability-based routing:
   - "RTX 5090 with 32GB" → Northgate for large ODE sweeps
   - "RTX 3090 with 24GB" → Strandgate/biomeGate for batch Anderson
   - "EPYC 64-core" → Strandgate for DADA2 parallelism
   - "76TB ZFS" → Westgate for SRA bulk storage
5. NestGate blob replication across towers (Westgate primary, local caches)

### Compute Budget: Phase B

| Workload | Single GPU (RTX 4070) | LAN (176GB VRAM) | Bottleneck |
|----------|-----------------------|-------------------|------------|
| EMP 30K full pipeline | ~4h | ~30 min | Download |
| Tara Oceans 243 stations | ~30 min | ~5 min | Download |
| HMP 4,700 samples | ~1h | ~10 min | Download |
| SRA 2000 BioProjects | Multi-day | ~12h | Download |
| Full cold seep metagenomes | ~8h | ~1h | Download (170GB) |

### Hardware Inventory

| Tower | CPU | GPU | RAM | Storage | Role |
|-------|-----|-----|-----|---------|------|
| Eastgate | — | RTX 4070 (12GB), 4× AKD1000 NPU | — | Local SSD | Hub + NPU inference |
| Northgate | — | RTX 5090 (32GB) | — | Local SSD | Flagship GPU compute |
| Southgate | — | RTX 3090 (24GB) | — | Local SSD | GPU compute |
| Strandgate | EPYC 64-core | RTX 3090 (24GB) + RX 6950 XT (16GB) | — | Local SSD | Heavy CPU + dual GPU |
| biomeGate | Threadripper 32-core | RTX 3090 (24GB) + Titan V (12GB) | — | Local SSD | Precision oracle (Titan V DF64) |
| Westgate | — | — | — | 76TB ZFS | Cold storage nest |

**Total VRAM:** ~176GB across 7 GPUs + 4 NPUs

---

## Key Insight: Data-Hungry, Not Compute-Hungry

The entire P0+P1 extension scope fits on Eastgate alone in a weekend. The LAN
mesh unlocks parallelism for the SRA atlas (Tier 2-3), not because individual
jobs are large, but because we want to run thousands of small jobs concurrently.

The true bottleneck is **data acquisition speed** — SRA downloads are
rate-limited and network-bound. NestGate's content-addressed caching ensures
we never re-download, and Westgate's 76TB ZFS provides the cold archive for
the full SRA atlas.

---

## Validation Experiments

| Experiment | What It Proves | Prerequisite |
|------------|---------------|--------------|
| Exp184 through NUCLEUS | Real NCBI sovereign pipeline via orchestrated NUCLEUS | Phase A |
| EMP 30K Anderson atlas | Paper 01 at 30K real samples | Phase A |
| KBS LTER W(t) | Paper 06 with real 30-year time series | Phase A |
| SRA atlas distributed | 2000 BioProjects across LAN mesh | Phase B |
| Multi-GPU Anderson batch | Large-lattice DF64 on Titan V + 5090 | Phase B |
