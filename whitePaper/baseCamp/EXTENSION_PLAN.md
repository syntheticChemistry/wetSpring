# baseCamp Extension Plan: From Validated Math to Real-World Science

**Date:** March 3, 2026 (updated)
**Author:** wetSpring Phase 93 (ecoPrimals)
**Foundation:** 280 experiments, 8,241+ checks, 52 papers (6 proposed), 144 primitives (standalone barraCuda v0.3.1), Python parity proven, cross-spring provenance validated (Exp271: 73/73, 6 springs + Exp272: 64/64, 7 springs), unified discovery, handler refactoring, 1,044 lib tests, Paper 12: full three-tier immunological Anderson (Exp273-279: 157/157 checks — science, CPU parity, GPU, streaming, metalForge)

---

## What We Have vs What We Need

The V84 pipeline buildout proved the math is correct (Python parity), portable
(CPU → GPU → streaming), and fast (10-1000× vs interpreted). The next phase
shifts from **validating compute** to **doing science with it**.

```
V1-V84: Can we compute correctly?          ✅ PROVEN
V85-V89: Cross-spring evolution validated? ✅ PROVEN (Exp271: 6 springs → ToadStool S79 → wetSpring)
V90:    Bio brain cross-spring ingest?     ✅ PROVEN (Exp272: 7 springs, 64/64 — brain + Nautilus + 36-head)
V91:    Deep debt + idiomatic evolution?   ✅ PROVEN (unified discovery, handler refactoring, 1,088 tests)
V92C:   Immunological Anderson (Paper 12)? ✅ PROVEN (Exp273-279: 157/157 — science + CPU parity + GPU + streaming + metalForge)
V92b:   Gonzales paper reproductions? ✅ PROVEN (Exp280-286: 202/202 — IC50 dose-response + PK decay + IL-31 serum + CPU parity + GPU + streaming + metalForge)
V93+:   What can we discover with it?      ← HERE
```

### New Since V84: Cross-Spring Evolution (V89–V90)

The V89 deep rewire proved that all cross-spring evolved ToadStool primitives
work end-to-end in wetSpring. V90 completes the brain architecture ingest.

- **`MultiHeadBioEsn`** (V89): hotSpring 36-head concept → bio 5-head (diversity,
  taxonomy, AMR, bloom, disorder) with per-head training and head disagreement
  uncertainty — ready for real multi-domain classifiers
- **`SpectralAnalysis` IPC** (V89): Anderson handler returns bandwidth, condition
  number, spectral phase, and Marchenko-Pastur bound in one call
- **Bio Brain** (V90): hotSpring 4-layer brain architecture → `BioBrain` adapter
  with 36-head `BioHeadGroupDisagreement`, `AttentionState` (Healthy/Alert/Critical),
  smoothed urgency window, and `DiversityUpdate` sentinel channel
- **`BioNautilusBrain`** (V90): `bingoCube/nautilus` evolutionary reservoir computing
  for concept edge detection (community phase boundaries), drift monitoring,
  and adaptive sampling — bio observation → Nautilus physics mapping
- **3 new IPC methods** (V90): `brain.observe`, `brain.attention`, `brain.urgency`
- **Cross-spring provenance**: hotSpring (spectral), neuralSpring (phase),
  wetSpring (diversity/bio), groundSpring (evolution/jackknife), airSpring
  (regression/hydrology), wateringHole (Boltzmann) — all validated

---

## Extension Axis 1: Real Data Through the Pipeline

### 1A. Earth Microbiome Project (EMP) — Anderson at Scale

**What:** Run the Anderson-QS classifier on 30,000+ real 16S samples from the
EMP (Thompson et al. 2017). Tests whether the 28-biome atlas holds at N=30K.

**Data requirements:**
- EMP 16S OTU/ASV tables: ~2GB processed (Qiita study 10317)
- Sample metadata (biome type, coordinates, environmental): ~50MB
- Acquisition: direct HTTP download from Qiita/EMP portal, no SRA needed

**Compute requirements:**
- Per sample: Shannon/Simpson/Bray-Curtis (~1ms) + Anderson 3D L=8 (~500ms)
- 30K samples: ~4 hours single-thread Eastgate, ~20 min GPU batch on RTX 4070
- Storage: <10GB total (OTU tables + results)
- **Verdict: trivial — runs on Eastgate alone in an afternoon**

**Primal integration:**
- NestGate: HTTP fetch of EMP data, content-addressed storage on Westgate
- ToadStool: GPU batch diversity + Anderson spectral on Eastgate/Strandgate
- No NUCLEUS needed yet — single-node job

**Science value:** Turns Paper 01 from "28 synthetic biomes" into "30,000 real
samples confirm the model." Highest impact, lowest risk.

### 1B. KBS LTER Soil Time Series — No-Till Paper 06

**What:** MSU's Kellogg Biological Station Long-Term Ecological Research has
30+ years of soil microbiome data under different tillage regimes. Apply the
dynamic W(t) model from Exp186 to real temporal data.

**Data requirements:**
- KBS LTER 16S amplicon data: ~50-200GB raw FASTQ (SRA BioProject PRJNA####)
- Treatment metadata (tillage type, crop rotation, fertilizer): ~10MB
- Weather/soil moisture (airSpring coupling): available from KBS portal
- Acquisition: NestGate EFetch from SRA, or `sra-toolkit` prefetch

**Compute requirements:**
- DADA2 processing: ~2 hours per run on Eastgate CPU (962 tests prove pipeline)
- Diversity + Anderson per time point: ~1 min GPU batch
- Full time series (30 years × 4 seasons × 8 treatments = ~960 samples): ~4 hours
- Storage: ~200GB raw + ~5GB processed
- **Verdict: moderate — Eastgate handles it, Strandgate is faster**

**Primal integration:**
- NestGate: SRA bulk download (needs EFetch + `prefetch` integration)
- ToadStool: GPU Anderson spectral for each time point
- airSpring: soil moisture data coupling (already validated Exp045)

**Science value:** First physics-based temporal model of no-till soil health
using the actual KBS data. MSU's own LTER site.

### 1C. SRA Longitudinal 16S Atlas — Community Dynamics at Scale

**What:** Systematic pull of all longitudinal 16S studies from SRA. Compute
W(t) trajectories for perturbation → recovery dynamics.

**Data requirements:**
- Search: "16S AND time series" on SRA → ~500-2000 BioProjects
- Raw data: 1-50TB depending on scope (most BioProjects are 1-10GB)
- Acquisition: NestGate ESearch → EFetch → `prefetch` → FASTQ dump
- **This is where Westgate's 76TB ZFS matters**

**Compute requirements:**
- DADA2: ~2 hours per BioProject (CPU-bound, parallelizable across towers)
- Anderson classification: minutes per BioProject on GPU
- Full atlas (1000 BioProjects): ~80 tower-hours CPU, ~8 hours GPU
- **Verdict: LAN-scale — distribute DADA2 across towers, centralize GPU**

**Primal integration:**
- NestGate: bulk SRA acquisition → Westgate cold storage
- biomeOS NUCLEUS: distribute DADA2 workloads across Node Atomics
- ToadStool: GPU Anderson + diversity on Eastgate/Strandgate/biomeGate
- Plasmodium: aggregate results across gates

**Science value:** Comprehensive atlas of Anderson regime dynamics across
hundreds of environmental perturbation studies. No one has done this.

### 1D. Hospital/Wastewater AMR Surveillance

**What:** Apply the sentinel framework (Paper 04) to published AMR
surveillance data from wastewater nanopore studies.

**Data requirements:**
- SRA AMR studies: ~10-50GB per study, 20-50 studies available
- Resistance gene databases (CARD, ResFinder): ~500MB
- Acquisition: NestGate EFetch

**Compute requirements:**
- K-mer + Anderson classification: minutes per sample on GPU
- Full analysis: hours on single tower
- **Verdict: trivial on single node**

---

## Extension Axis 2: New Scientific Questions

### 2A. Mycorrhizal Network Anderson Model

**What:** Apply Anderson localization to underground mycorrhizal (fungal)
hyphal networks. Hyphae form literal 3D lattices in soil. The Anderson
model predicts which soil conditions support long-range nutrient/signal
relay and which don't.

**Data needs:** Published mycorrhizal network imaging data (micro-CT soil
scans), plus NCBI ITS amplicon data for mycorrhizal community composition.
~10GB processed.

**Compute:** Same Anderson pipeline. Trivial.

**Science:** Extends Paper 03 (bioag) and Paper 06 (no-till) with the fungal
dimension. First Anderson model of mycorrhizal signaling geometry.

### 2B. Plasmid Transfer Through Anderson Lens

**What:** Model horizontal gene transfer of AMR plasmids as wave propagation
through a disordered community lattice. Anderson predicts: plasmid spread
should be geometry-dependent (efficient in 3D biofilm, suppressed on 2D
surfaces).

**Data needs:** NCBI plasmid databases + published conjugation rate data.
~5GB. Available via NestGate.

**Compute:** Anderson spectral + birth-death-transfer ODE. Trivial.

**Science:** First physics-based model of AMR spread. Connects Papers 04
(sentinels) and 09 (field genomics).

### 2C. Coupled Nutrient Cycle Anderson Models

**What:** Carbon, nitrogen, phosphorus cycles are each mediated by QS-active
microbial guilds. A multi-disorder Anderson model (coupled lattices with
different W values per nutrient guild) predicts which cycles activate under
which soil conditions.

**Data needs:** NCBI functional gene databases (NifH for N-fixation, phoD for
P-cycling, etc.) + soil chemistry data from LTER sites. ~20GB.

**Compute:** Multi-lattice Anderson spectral. Moderate (3× single lattice).

**Science:** Predictive model for nutrient cycling in managed soils. Directly
actionable for precision agriculture (Paper 03).

---

## Extension Axis 3: Hardware + Primal Integration

### 3A. NUCLEUS Local Deployment (Phase 1: Single-Node)

**What:** Run `biomeos nucleus start --mode node` on Eastgate. ToadStool
handles GPU compute, NestGate handles data storage, Songbird handles
discovery. All over Unix sockets.

**What exists:**
- biomeOS: Pure Rust launcher, lifecycle manager, Neural API (124 capabilities)
- ToadStool: A++ socket-standardized, Node Atomic ready
- NestGate: NCBI provider operational (ESearch/ESummary/EFetch)
- BearDog: Crypto/trust layer operational

**What's needed:**
- Wire NestGate NCBI → wetSpring BarraCuda diversity pipeline via IPC
- Wire ToadStool GPU dispatch → wetSpring Anderson spectral via IPC
- Already prototyped in Exp203-208 (321 IPC checks)

**Compute cost:** Zero additional — same hardware, better orchestration.

### 3B. NUCLEUS LAN Mesh (Phase 2: Multi-Gate)

**What:** Cable the 10G backbone (switch acquired, NICs installed, Cat6a
pending). Run NUCLEUS across Eastgate + Strandgate + Northgate + Westgate.

**Architecture:**
```
Eastgate (Node: GPU + NPU)     ──┐
Strandgate (Node: 2×GPU + NPU) ──┤── 10G Switch ──── Westgate (Nest: 76TB ZFS)
Northgate (Node: RTX 5090)     ──┘
```

**Workload distribution:**
- Westgate (Nest): NestGate cold storage, SRA archive, content-addressed blobs
- Eastgate (Node): Primary GPU compute (RTX 4070), NPU inference (AKD1000)
- Strandgate (Node): Heavy GPU (RTX 3090 + RX 6950 XT), EPYC CPU parallelism
- Northgate (Node): Flagship GPU (RTX 5090), large-batch Anderson spectral

**What unlocks:**
- SRA atlas (Axis 1C) becomes practical — distribute DADA2 across 3 CPU nodes
- Titan V on biomeGate as precision oracle for DF64 validation
- 4× Akida mesh for distributed NPU sentinel pipeline

**Investment:** ~$50 for Cat6a cables. Everything else is acquired.

### 3C. MinION Integration (Phase 3: Field Genomics)

**What:** Purchase MinION Mk1D (~$1K). Wire `io::nanopore` (FAST5/POD5
reader) and `bio::basecall` (signal → base) into BarraCuda.

**What exists:** All downstream compute (16S → diversity → Anderson → NPU).

**What's new:** Raw signal processing + basecalling in Rust.

**Science:** Paper 09 becomes hardware-validated end-to-end.

---

## Data and Compute Budget Summary

| Extension | Data Size | Compute Time | Hardware | Priority |
|-----------|-----------|--------------|----------|----------|
| EMP 30K samples | 2GB | 20 min GPU | Eastgate alone | **P0 — do now** |
| KBS LTER time series | 200GB | 4 hours | Eastgate alone | **P0 — do now** |
| SRA longitudinal atlas | 1-50TB | 80 tower-hours | LAN mesh | P1 — after 10G cables |
| AMR surveillance | 50GB | hours | Eastgate alone | P1 |
| Mycorrhizal Anderson | 10GB | hours | Eastgate alone | P1 |
| Plasmid transfer model | 5GB | hours | Eastgate alone | P2 |
| Coupled nutrient cycles | 20GB | moderate | LAN mesh | P2 |
| MinION integration | 20GB/run | 30 min/run | Eastgate + MinION | P2 — ~$1K investment |

**Total storage appetite:** ~50TB at full SRA atlas scale. Westgate has 76TB.
**Total compute appetite:** ~100 tower-hours for the full atlas. LAN mesh
handles it in a weekend. Eastgate alone handles everything else.

---

## Primal Integration Sequence

```
Phase 1 (Now — single node):
  biomeos nucleus start --mode node     # on Eastgate
  NestGate: EMP + KBS LTER download     # HTTP + SRA
  ToadStool: GPU Anderson + diversity   # RTX 4070
  BarraCuda: 16S → diversity → W(t)     # wetSpring pipeline

Phase 2 (After 10G cables — LAN mesh):
  biomeos nucleus start --mode full     # Eastgate hub
  biomeos nucleus start --mode node     # Strandgate, Northgate
  biomeos nucleus start --mode nest     # Westgate
  Plasmodium: distribute DADA2 + aggregate results
  NestGate: SRA bulk → Westgate cold storage
  ToadStool: multi-GPU Anderson (3090 + 5090 + 4070)

Phase 3 (After MinION — field genomics):
  MinION → Eastgate (basecall) → AKD1000 (classify) → NestGate (store)
  biomeGate: mobile field station (Threadripper + 3090 + Titan V + Akida)
```

---

## What This Produces

By the end of Phase 2, the baseCamp papers have:

| Paper | Current State | After Extensions |
|-------|--------------|-----------------|
| 01 Anderson QS | 28 synthetic biomes | 30,000+ EMP real samples |
| 02 LTEE | Predictions only | Still predictions (needs LTEE data access) |
| 03 Bioag | Framework validated | Mycorrhizal Anderson + nutrient coupling model |
| 04 Sentinels | Live NPU | AMR surveillance real data |
| 05 Cross-Species | 170 metagenomes | Plasmid transfer model |
| 06 No-Till | 9 papers reproduced | KBS LTER 30-year real time series |
| 07 WDM | Paper parity | (hotSpring leads) |
| 08 NPU IoT | Live hardware | Multi-gate NPU mesh |
| 09 Field Genomics | Architecture defined | MinION hardware-validated |
| 10 coralForge | GPU pipeline proven | (neuralSpring leads) |

The infrastructure is built. The math is proven. The data is public. The
hardware is in the basement. The primals know how to talk to each other.
Now we do the science.

---

## Extension Axis 4: Primal Integration Strategy (V89+)

The extensions above assume single-primal execution. The next evolution
brings multiple primals into the pipeline, coordinated through biomeOS.

### What Exists Now (wetSpring-side)

| Module | Status | Role |
|--------|--------|------|
| `ncbi/` (barracuda) | Operational | ESearch, EFetch, SRA download, cache |
| `ncbi/nestgate/` | Operational (optional) | JSON-RPC to NestGate provider |
| IPC handlers | Operational | JSON-RPC 2.0 for diversity, QS, Anderson, pipeline |
| `MultiHeadBioEsn` | New (V89) | 5-head bio classifier with uncertainty |
| NUCLEUS atomics model | Validated (Exp266, 270) | Tower → Node → Nest + Vault |

### What Exists Externally

| Primal | Location | Role | Key Capabilities |
|--------|----------|------|-----------------|
| **biomeOS** | phase2/biomeOS/ | Orchestrator | Graph execution, capability registry, Neural API (124+ capabilities), lifecycle management |
| **NestGate** | Referenced in ncbi/nestgate/ | Data layer | Content-addressed storage, NCBI provider, SRA, ZFS cold storage (Westgate 76TB) |
| **ToadStool** | phase1/toadstool/ | Compute | 844 WGSL shaders, MultiHeadEsn, spectral, 93+ primitives consumed by wetSpring |
| **Songbird** | Referenced in IPC | Discovery | Primal discovery, mesh routing, HTTPS stack |
| **BearDog** | Referenced in IPC | Trust | Crypto, TLS 1.3, key hierarchy, SoloKey FIDO2 anchors |

### Primal Integration Phases (Revised)

```
Phase 0 (Now — wetSpring standalone on Eastgate):
  wetSpring consumes ToadStool as a Rust dependency (path dep)
  NCBI via sovereign HTTP (no NestGate needed)
  GPU via wgpu (RTX 4070), NPU via akida-driver (AKD1000)
  IPC: local JSON-RPC 2.0 over Unix sockets
  ✅ Exp271 validates 93+ primitives, 73/73 checks

Phase 1 (Next — biomeOS single-node on Eastgate):
  biomeos nucleus start --mode node
  Wire: NestGate NCBI → wetSpring diversity pipeline via IPC
  Wire: ToadStool GPU dispatch → wetSpring Anderson spectral via IPC
  Benefit: EMP 30K samples (Axis 1A) via orchestrated pipeline
  Benefit: KBS LTER (Axis 1B) via SRA + NestGate cache
  Benefit: MultiHeadBioEsn real training on NCBI data
  Compute: trivial — same hardware, better data flow

Phase 2 (After 10G cables — LAN mesh):
  Tower (Eastgate) + Node (Strandgate, Northgate) + Nest (Westgate)
  biomeOS coordinates workload placement by capability
  NestGate on Westgate: 76TB ZFS cold archive for SRA atlas
  ToadStool on Northgate: RTX 5090 for large-batch Anderson
  ToadStool on Strandgate: RTX 3090 + EPYC 64-core for DADA2
  wetSpring on Eastgate: orchestration + NPU inference (AKD1000)
  Benefit: SRA atlas (Axis 1C) becomes weekend job not month job
  Benefit: 4× Akida NPU mesh for distributed sentinel pipeline
  Investment: ~$50 Cat6a cables (everything else acquired)

Phase 3 (After MinION — field genomics):
  MinION Mk1D → Eastgate basecall → AKD1000 classify → NestGate store
  biomeGate as mobile field station (32C Threadripper + 3090 + Titan V + Akida)
  Paper 09 end-to-end validated on hardware
  Investment: ~$1K MinION Mk1D
```

### Data Hunger Analysis

| Extension | Raw Data | Processed | Source | Acquisition |
|-----------|----------|-----------|--------|-------------|
| EMP 30K (P0) | 2GB | <1GB | Qiita HTTP | Direct download |
| KBS LTER (P0) | 200GB FASTQ | 5GB | SRA | NestGate EFetch |
| SRA Atlas (P1) | 1-50TB | 50GB | SRA bulk | NestGate → Westgate ZFS |
| AMR Surveillance (P1) | 50GB | 5GB | SRA | NestGate EFetch |
| Mycorrhizal (P1) | 10GB | 1GB | NCBI ITS + micro-CT | NestGate + HTTP |
| Plasmid Transfer (P2) | 5GB | 500MB | NCBI plasmid DB | NestGate |
| Coupled Nutrients (P2) | 20GB | 2GB | NCBI func. genes + LTER | NestGate + HTTP |
| MinION Field (P2) | 20GB/run | 2GB/run | Local sequencer | Direct USB |
| **Total at full scope** | **~55TB** | **~65GB** | | Westgate has 76TB |

### Compute Hunger Analysis

| Extension | CPU Hours | GPU Hours | NPU Hours | Best Hardware |
|-----------|-----------|-----------|-----------|---------------|
| EMP 30K (P0) | 4h diversity | 0.3h Anderson batch | — | Eastgate alone |
| KBS LTER (P0) | 2h DADA2 | 0.5h Anderson | — | Eastgate alone |
| SRA Atlas (P1) | 80h DADA2 | 8h Anderson | — | LAN mesh (3 nodes) |
| AMR Surveillance (P1) | 2h | 0.5h | 0.1h NPU sentinel | Eastgate alone |
| MultiHead ESN training (P1) | 0.5h | 1h reservoir update | 0.01h deploy | Eastgate |
| DF64 Anderson L=20 (P1) | — | 4h DF64 spectral | — | Strandgate (3090) |
| SRA+Anderson full atlas (P2) | 200h | 20h | 4h NPU classify | Full LAN mesh |
| MinION real-time (P2) | continuous | continuous classify | continuous sentinel | Eastgate + MinION |

### Key Insight: We're Not Compute-Hungry

The entire P0+P1 scope fits on Eastgate alone in a weekend. The LAN mesh
unlocks parallelism for the SRA atlas (P1-P2), not because single jobs are
large, but because we want to run thousands of small jobs. The bottleneck
is **data acquisition** (SRA download speed), not compute.

The `MultiHeadBioEsn` training on real NCBI data (diversity/taxonomy/AMR/
bloom/disorder heads) is the most scientifically valuable P1 item because
it turns the validated math into a trained bio classifier with quantified
uncertainty. That's a 1-hour GPU job on Eastgate with 30K EMP samples.

### Where Other Primals Help

| Primal | What It Adds | When Needed |
|--------|-------------|-------------|
| **biomeOS** | Orchestration: graph-based pipeline execution, capability routing | Phase 1+ (but P0 works without it) |
| **NestGate** | Data caching: SRA prefetch, content-addressed blobs, cold archive | Phase 1+ (P0 uses sovereign HTTP) |
| **Songbird** | Discovery: automatic primal finding on LAN, mesh routing | Phase 2 (multi-gate coordination) |
| **BearDog** | Trust: encrypted NestGate blobs, inter-gate TLS, FIDO2 attestation | Phase 2 (multi-gate security) |
| **ToadStool** | Compute: already consumed as dependency, IPC for distributed dispatch | Phase 1+ (direct dispatch) |

biomeOS is the natural orchestrator because it already has NUCLEUS lifecycle
management and a capability registry (124+ capabilities). The pattern is:

```
biomeOS graph:
  fetch(ncbi, "16S AND soil") → NestGate
  process(wetspring, "dada2") → BarraCUDA CPU
  classify(wetspring, "anderson_qs") → BarraCUDA GPU (ToadStool)
  deploy(wetspring, "multi_head_esn") → AKD1000 NPU
  store(nestgate, results) → Westgate ZFS
```

This is the graph execution model that biomeOS already supports. wetSpring's
IPC handlers (diversity, QS, Anderson, pipeline) map directly to biomeOS
graph nodes. The `MultiHeadBioEsn` export → NPU deployment path is
validated end-to-end (Exp193-195).
