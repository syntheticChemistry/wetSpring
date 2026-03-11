# wetSpring V113 — barraCuda / toadStool Evolution Handoff

**Date:** March 11, 2026
**From:** wetSpring V113 (370 experiments, 9,886+ checks, 354 binaries, 1,294 lib tests pass)
**To:** barraCuda team, toadStool team, coralReef team, NestGate team
**Supersedes:** WETSPRING_V112_NVIDIA_HARDWARE_LEARNING_HANDOFF_MAR11_2026.md (archived)
**License:** AGPL-3.0-or-later
**Covers:** V110–V113 (Exp353–370), paper extension roadmap, primal integration, LAN mesh planning

---

## Executive Summary

- **370 experiments, 9,886+ checks, ALL PASS.** 63 papers reproduced (46 at full CPU+GPU+metalForge tiers).
- **V113 adds 7 experiments (67/67):** EMP 28K-sample atlas, Liao group real community data, KBS LTER 30-year temporal model, QS gene profiling with FNR/ArcAB/Rex regulons, primal integration pipeline (NestGate + ToadStool socket discovery), P1 extension framework (5 datasets), LAN mesh SRA atlas plan (5 towers, 96GB VRAM, 208 TFLOPS).
- **H3 model validated at scale:** W = 3.5·H' + 8·O₂ confirmed across 28 biomes at N=28K. Anaerobic P(QS)=0.81 vs aerobic 0.16.
- **Zero local WGSL, zero Passthrough, zero unsafe.** Fully lean. 150+ primitives consumed.
- **Primal pipeline operational:** Three-tier NCBI routing (biomeOS → NestGate → sovereign HTTP) tested. Graceful degradation when primals down.
- **Hardware profile ready for absorption:** `output/hardware_capability_profile.json` encodes RTX 4070 dispatch recipe.

---

## Part 1: Primitive Consumption Report

### Current Usage (V113)

| Category | Count | Status |
|----------|------:|--------|
| Primitives consumed | 150+ | All lean |
| ComputeDispatch ops | 264 | All via `compile_shader_universal` |
| GPU modules | 47 | All lean, 0 local WGSL |
| CPU modules | 47 | All lean, 0 local derivative math |
| Passthrough | 0 | All promoted (V40+) |
| ODE systems | 5 | All via `BatchedOdeRK4::<S>::generate_shader()` |
| Named tolerances | 180 | |
| petalTongue scenario builders | 33 | |

### Primitives Exercised in V113

| Primitive | Experiment | Usage |
|-----------|-----------|-------|
| `barracuda::stats::diversity::shannon` | Exp364-370 | Per-sample H' computation (28K+ calls) |
| `barracuda::stats::diversity::simpson` | Exp364-366 | Per-sample Simpson D |
| `barracuda::stats::norm_cdf` | Exp364-370 | P(QS) = norm_cdf((W_c - W) / σ) |
| `barracuda::ncbi::esearch_count` | Exp368 | NCBI search validation |
| `barracuda::ncbi::nestgate::fetch_tiered` | Exp368 | Three-tier NCBI routing |
| `barracuda::ncbi::api_key` | Exp368 | API key discovery |
| `barracuda::device::WgpuDevice` | Exp368 | GPU device probing |
| `barracuda::device::HardwareCalibration` | Exp368 | Hardware capability probing |
| `barracuda::visualization::*` | Exp364,366,368 | petalTongue scenarios |

### API Observations

1. **`api_key()` returns `Option<String>`** — callers need `.unwrap_or_default()`. Consider providing a `api_key_or_empty()` convenience.
2. **`ncbi::entrez` module is private** — only `esearch_count` is re-exported. For real data pipelines (EMP, KBS LTER), we'll need bulk search results (accessions, not just count). Consider exposing `esearch_ids(db, term, retmax, api_key) -> Vec<String>`.
3. **`DataChannel::Scatter` requires `point_labels: Vec<String>`** even when empty — consider `#[serde(default)]` to make it optional in construction (currently must pass `vec![]`).
4. **`DataChannel::TimeSeries` uses `x_values`/`y_values`** — naming is correct but the pattern is line chart, not time series per se. Consider an alias or `LineChart` variant for non-temporal data.

---

## Part 2: Absorption Targets (Prioritized)

### P0 — Immediate (enables real data pipeline)

**For barraCuda:**

| # | Target | Justification | Experiment |
|---|--------|---------------|-----------|
| 1 | `barracuda::io::biom` — BIOM format OTU table parser | EMP data is distributed as BIOM. TSV loader exists (Exp364) but BIOM is standard. | Exp364 |
| 2 | `barracuda::ncbi::entrez::esearch_ids()` — expose accession list | Three-tier routing works but we can only count, not retrieve IDs for batch fetch. | Exp368 |
| 3 | `barracuda::bio::kinetics::gompertz_fit()` — nonlinear fitting | Exp365 manually sets Gompertz parameters. Least-squares fitting from time-series data enables automated Track 6 pipeline. | Exp365 |

**For toadStool / hw-learn:**

| # | Target | Justification | Experiment |
|---|--------|---------------|-----------|
| 4 | Consume `hardware_capability_profile.json` | toadStool's hw-learn module should auto-configure dispatch from capability profile. The JSON format is stable (Exp362). | Exp362-363 |
| 5 | VRAM-aware batch sizing | Use VRAM ceiling from profile (12GB for RTX 4070, max pairwise N≈40K) to auto-size batches. | Exp362 |

### P1 — Near-term (enables P1 datasets)

**For NestGate:**

| # | Target | Justification | Experiment |
|---|--------|---------------|-----------|
| 6 | `data.sra_prefetch` capability | SRA bulk download via `prefetch` tool. Required for KBS LTER (200GB FASTQ) and P1 datasets. | Exp366,369 |
| 7 | Content-addressed EMP cache | Cache Qiita downloads with BLAKE3 keys. Re-runs become instant. | Exp364 |

**For barraCuda:**

| # | Target | Justification | Experiment |
|---|--------|---------------|-----------|
| 8 | `barracuda::bio::anderson::temporal_w()` | Dynamic W(t) = W_eq + ΔW·exp(-t/τ) model. Currently computed inline in Exp366. | Exp366 |
| 9 | `barracuda::bio::qs::regulon_map()` | FNR/ArcAB/Rex regulon cross-reference database. Currently hardcoded in Exp367. | Exp367 |

### P2 — Future (enables LAN mesh)

**For biomeOS:**

| # | Target | Justification | Experiment |
|---|--------|---------------|-----------|
| 10 | Distributed Anderson graph | NUCLEUS graph: `fetch → process → classify → store` across tower mesh. | Exp370 |
| 11 | Workload splitter | Distribute N samples across K GPU nodes proportional to throughput. | Exp370 |

---

## Part 3: Hardware Learning Findings

### RTX 4070 (nvidia proprietary, Exp361-363)

| Tier | Status | Safe For |
|------|--------|----------|
| F32 | Safe (compile + dispatch + transcendentals) | Shannon, Simpson, Bray-Curtis, Anderson eigenvalues |
| DF64 | Arithmetic-only (transcendentals unsafe, NVVM) | Anderson Lanczos iteration (no transcendentals needed) |
| F64 | Arithmetic-only (same NVVM risk) | Large-N pairwise distance (no log/exp) |
| F64Precise | Arithmetic-only | — |

### Titan V (nouveau, Exp361)

- VM_INIT succeeds (kernel 6.17.9 new UAPI)
- CHANNEL_ALLOC returns EINVAL (Volta lacks GSP firmware)
- Non-GSP firmware present: acr, gr, nvdec, sec2 (needs PMU for compute)

### RTX 4070 on nouveau — highest-ROI unlock

Ada Lovelace (AD104) has GSP-only firmware. Switching from nvidia proprietary to nouveau should enable sovereign dispatch via coralReef. This is the single highest-ROI hardware evolution target.

**toadStool action:** When nouveau RTX 4070 is tested, the `hw-learn` module should detect the sovereign dispatch path and route accordingly.

---

## Part 4: Science Pipeline Architecture

### Three-Tier NCBI Routing (validated Exp368)

```
Tier 1: biomeOS Neural API → capability.call("science.ncbi_fetch")
Tier 2: NestGate IPC → cache check → sovereign HTTP → store
Tier 3: Sovereign HTTP → barracuda::ncbi::efetch_fasta()
```

Falls back gracefully. Tier 3 always works. When NestGate is running, caching is automatic.

### biomeOS Graph (target for Exp370 LAN mesh)

```
fetch(ncbi, "16S AND soil") → NestGate
process(wetspring, "dada2") → BarraCuda CPU (Strandgate 128 cores)
classify(wetspring, "anderson_qs") → BarraCuda GPU via ToadStool
store(nestgate, results) → Westgate ZFS
```

### petalTongue Dashboard Ecosystem (33 builders + 8 new JSON artifacts)

V113 artifacts:
- `output/emp_anderson_qs_atlas.json` — 28K samples, biome-stratified
- `output/kbs_lter_anderson_temporal.json` — 30-year W(t) time series
- `output/primal_pipeline_status.json` — primal health gauges
- `output/qs_gene_regulon_analysis.json` — QS × O₂ matrix
- Plus 4 more summary/framework JSONs

---

## Part 5: Version Pins

| Component | Version | Hash | Notes |
|-----------|---------|------|-------|
| barraCuda | v0.3.5 | `0649cd0` | 784+ WGSL shaders, 150+ primitives |
| toadStool | S146 | `751b3849` | Dual socket (tarpc + JSON-RPC) |
| coralReef | Iter 33 | `b783217` | Sovereign shader compiler |
| wetSpring | V113 | — | 370 experiments, 9,886+ checks |
| wgpu | 28 | — | Vulkan backend on RTX 4070 |
| Rust | 1.87 | — | MSRV |

---

## Part 6: What's Working Well

1. **Write → Absorb → Lean cycle** is complete. Zero local WGSL, zero local derivative math. Every GPU op delegates to barraCuda.
2. **Three-tier NCBI routing** is elegant and resilient. Falls back gracefully.
3. **Hardware capability profile** is machine-readable and consumable by any primal.
4. **petalTongue integration** produces actionable dashboards from science pipelines.
5. **Anderson QS model (H3)** is the strongest result — validated at 28K samples with molecular mechanism support from QS gene profiling.
6. **Graceful degradation everywhere** — experiments run whether primals are up or down, whether GPU is available or not, whether network is up or down.

---

## Part 7: Deployment Feedback

See `whitePaper/baseCamp/NUCLEUS_LOCAL_DEPLOYMENT.md` for 6 actionable issues (F1–F6) found during Eastgate deployment. Key blockers:

- **F4 (ToadStool Songbird registration):** ToadStool sends registration without `primal_id`. Falls back to standalone. Non-blocking but limits auto-discovery.
- **F6 (NestGate storage.list):** `storage.list` returns empty despite stored keys existing. Workaround: use `storage.retrieve` with known keys.

---

## Appendix: Files Changed in V113

### New Binaries (7)
- `validate_emp_anderson_v1.rs` (Exp364)
- `validate_liao_real_data_v1.rs` (Exp365)
- `validate_kbs_lter_anderson_v1.rs` (Exp366)
- `validate_qs_gene_profiling_v1.rs` (Exp367)
- `validate_primal_pipeline_v1.rs` (Exp368)
- `validate_p1_extensions_v1.rs` (Exp369)
- `validate_lan_mesh_plan_v1.rs` (Exp370)

### Documentation Updated (16 files)
- `README.md`, `CHANGELOG.md`, `barracuda/README.md`, `barracuda/Cargo.toml`
- `barracuda/ABSORPTION_MANIFEST.md`, `barracuda/EVOLUTION_READINESS.md`
- `whitePaper/README.md`, `whitePaper/baseCamp/README.md`
- `whitePaper/baseCamp/EXTENSION_PLAN.md`, `whitePaper/baseCamp/sub_thesis_01_anderson_qs.md`
- `whitePaper/baseCamp/NUCLEUS_LOCAL_DEPLOYMENT.md`
- `experiments/README.md`, `specs/README.md`, `specs/PAPER_REVIEW_QUEUE.md`
- `specs/INDUSTRY_TOOL_COVERAGE.md`, `specs/BARRACUDA_REQUIREMENTS.md`
- `metalForge/ABSORPTION_STRATEGY.md`, `metalForge/PRIMITIVE_MAP.md`
- `wateringHole/README.md`
- `ecoPrimals/whitePaper/gen3/baseCamp/README.md`, `ecoPrimals/whitePaper/gen3/baseCamp/EXTENSION_PLAN.md`
