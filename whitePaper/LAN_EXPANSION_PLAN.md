# wetSpring LAN Expansion Plan — Metagenomics at Scale

**Date:** February 28, 2026
**Status:** Planning (Phase F of Data Extensions plan)
**Prerequisites:** 10G cables + Tower atomic wiring (completed V68)

---

## 1. Hardware Available

| Gate | CPU | RAM | GPU | NPU | Storage | Role |
|------|-----|-----|-----|-----|---------|------|
| **Eastgate** | i9-12900K (16c/24t) | 32 GB | RTX 4070 | 1x AKD1000 | 2TB + 2x2TB NVMe | Local dev, NPU sentinel |
| **Strandgate** | 2x EPYC 7452 (64c/128t) | 256 GB ECC | RTX 3090 + RX 6950 XT | 2x AKD1000 | 2x2TB + 4x4TB NVMe | **Metagenome assembly** |
| **biomeGate** | Threadripper 3970X (32c/64t) | 256 GB | RTX 3090 + Titan V | 1x AKD1000 | 1TB + 2x2TB NVMe | Heavy compute, portable |
| **Northgate** | i9-14900K | 192 GB | RTX 5090 | — | 1TB + 4TB | AI/LLM, top GPU |
| **Southgate** | 5800X3D | 128 GB | RTX 3090 | — | 1TB + 4TB | Gaming + compute |
| **Westgate** | i7-4771 | 32 GB | RTX 2070 Super | — | 76TB HDD (ZFS) + 2TB NVMe | **Cold storage archive** |

**10G backbone**: Switch acquired, NICs installed (4 towers). Cables pending purchase.

---

## 2. NestGate SRA Evolution

### Current state

NestGate's `NCBILiveProvider` handles:
- `esearch`, `esummary`, `efetch` (single sequences)
- Storage via `storage.store` / `storage.retrieve` (BLAKE3 content-addressed)

### Missing for SRA bulk

| Gap | Solution | Priority |
|-----|----------|----------|
| `fasterq-dump` integration | Subprocess wrapper in NestGate (Rust `Command`) | P0 |
| Resumable downloads | Content-range HTTP + partial file tracking | P1 |
| Download queue | Async task queue with priority + rate limiting | P1 |
| Provenance chain | accession -> SHA-256 -> NestGate blob key | P0 |
| Cross-gate replication | Plasmodium data announcements | P2 |

### Evolution path

```
wetSpring scripts (Python: fetch_ncbi_phase35.py, ncbi_bulk_download.sh)
  -> NestGate Rust provider (absorb Python NCBI + SRA logic)
    -> Nest atomic (biomeOS orchestrates download/storage/provenance)
      -> Plasmodium data distribution (multi-gate, 10G backbone)
```

### SRA download targets

| Dataset | Accession | Est. Size | Target Gate | Experiments |
|---------|-----------|-----------|-------------|-------------|
| Algae 16S | PRJNA488170 | ~5 GB | Eastgate | Exp012 |
| Phytoplankton 16S | PRJNA1114688 | ~3 GB | Eastgate | Exp002, 017 |
| Deep-sea vents | PRJNA283159 + PRJEB5293 | ~100 GB | Strandgate | Exp051-056 |
| Cold seep QS | 170 metagenomes | ~500 GB | Strandgate | Exp144-145 |
| Vibrio assemblies | NCBI Assembly | 257 MB | Eastgate (done) | Exp121 |
| Campylobacterota | NCBI Assembly | 90 MB | Eastgate (done) | Exp125 |

---

## 3. Plasmodium Data Distribution

### What Plasmodium provides

When multiple NUCLEUS gates join the mesh, they form a Plasmodium — an emergent
collective that can distribute data and compute across the LAN.

### Data distribution strategy

```
         Westgate (76 TB cold)
            │ 10G
    ┌───────┼───────┐
    │       │       │
Strandgate  │   biomeGate
(256 GB)    │   (256 GB)
    │       │       │
    │   Eastgate    │
    │   (32 GB)     │
    │       │       │
    └───────┼───────┘
         Northgate
         (192 GB)
```

| Data tier | Location | Strategy |
|-----------|----------|----------|
| **Hot** (active experiments) | Eastgate + Strandgate NVMe | NestGate local cache |
| **Warm** (recent downloads) | Strandgate 20TB NVMe pool | NestGate replication |
| **Cold** (archive) | Westgate 76TB ZFS | NestGate cold storage + ZFS snapshots |

### Replication protocol

1. NestGate on gate A downloads dataset (SRA FASTQ)
2. NestGate announces availability via Songbird: `data.announce(accession, sha256, size)`
3. Gates needing the data request via `data.request(accession)`
4. Sender streams over 10G backbone (direct socket, no HTTP overhead)
5. Receiver verifies SHA-256, stores in local NestGate cache

---

## 4. Strandgate Metagenomics Workloads

### Why Strandgate

Metagenome assembly requires:
- **RAM**: 32-128 GB per sample (MEGAHIT/MetaSPAdes)
- **Threads**: More = faster (embarrassingly parallel k-mer counting)
- **Storage**: Intermediate files can be 10x input size

Strandgate has 256 GB ECC RAM, 128 threads, and 20TB NVMe — purpose-built for this.

### Workload plan

| Phase | Workload | Samples | Est. Time | Gate |
|-------|----------|---------|-----------|------|
| F1 | 16S amplicon (algae + phytoplankton) | ~50 | ~2 hrs | Eastgate |
| F2 | PFAS spectral match (real mzML) | 10-50 | ~5 min | Eastgate |
| F3 | Vibrio/Campy landscape analysis | 357 assemblies | ~30 min | Eastgate |
| F4 | Deep-sea vent metagenome assembly | ~50 samples | ~2-5 days | **Strandgate** |
| F5 | Cold seep 170 metagenomes | 170 samples | ~1-2 weeks | **Strandgate** |
| F6 | Pangenome analysis (post-assembly) | all | ~1 day | biomeGate |

### metalForge integration

With Tower wiring (V68), metalForge's `discover_with_tower()` will see
Strandgate's substrates when the mesh is online:

```
Inventory: 4 local (Eastgate), 6 mesh (Strandgate GPU+CPU, biomeGate GPU+CPU, ...)
```

Dispatch can then route:
- Diversity metrics → Eastgate GPU (RTX 4070, low latency)
- Metagenome assembly → Strandgate CPU (128 threads, 256 GB)
- Anderson spectral → biomeGate Titan V (HBM2, full f64)
- NPU sentinel → Eastgate AKD1000 (real-time, coin-cell power)

---

## 5. Prerequisites and Timeline

### Immediate (can start now)

- [x] Tower atomic wiring in metalForge (V68, done)
- [x] Priority-1 data downloaded (Vibrio, Campy, PFAS, SILVA)
- [x] Data resolution chain (env -> NestGate -> synthetic)
- [ ] Purchase 10G Cat6a cables (4 needed, ~$50)

### Short-term (1-2 weeks)

- [ ] Cable 10G backbone (Eastgate <-> Strandgate <-> Westgate <-> Northgate)
- [ ] Deploy Tower atomic on Strandgate (BearDog + Songbird)
- [ ] Deploy Nest atomic on Strandgate (Tower + NestGate)
- [ ] NestGate SRA evolution: `fasterq-dump` wrapper + download queue
- [ ] Test Plasmodium data distribution (Eastgate -> Strandgate)

### Medium-term (2-4 weeks)

- [ ] Download deep-sea vent metagenomes to Strandgate (~100 GB)
- [ ] Run first metagenome assembly on Strandgate (MEGAHIT)
- [ ] Deploy Nest atomic on Westgate for cold archive
- [ ] Cold seep dataset download planning (500 GB)

### Long-term (1-2 months)

- [ ] Cold seep 170 metagenome downloads + assembly
- [ ] Pangenome analysis pipeline on biomeGate
- [ ] Full Plasmodium with all gates online
- [ ] Cross-gate metalForge dispatch validation (Exp222+)

---

## 6. Key Decision Points

- **10G cables**: Required for metagenomics. 1G works for 16S amplicon but
  streaming 100 GB datasets at 1 Gbps takes ~15 minutes vs ~80 seconds at 10G.

- **Cold seep scale**: 170 metagenomes at ~3 GB each = ~500 GB raw. Assembly
  intermediates could be 2-5 TB. This is why Strandgate's 20 TB NVMe pool and
  Westgate's 76 TB ZFS are essential.

- **NestGate SRA**: The absorption of Python SRA scripts into NestGate Rust
  follows the exact same pattern as ToadStool absorbing BarraCuda primitives.
  The Python scripts serve as baselines; the Rust implementation validates parity.
