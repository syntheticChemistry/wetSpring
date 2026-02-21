# metalForge — PCIe Topology & Mixed Hardware Routing

**Date:** February 21, 2026
**Purpose:** Document the physical interconnect topology that enables mixed
hardware dispatch (GPU↔NPU↔CPU) without unnecessary round-trips.

---

## PCIe Device Map

```
Root Complex: Intel Alder Lake (0000:00)
│
├── Root Port 01.0 ─── Bus [01] ─── NVIDIA RTX 4070 (01:00.0)
│                                    PCIe Gen4 x16, 32 GB/s each direction
│                                    Primary GPU — ToadStool dispatch target
│
├── Root Port 06.0 ─── Bus [02] ─── Samsung NVMe SSD
│                                    PCIe Gen4 x4, 8 GB/s
│
├── Root Port 1b.4 ─── Bus [05] ─── NVIDIA Titan V (05:00.0)
│                                    PCIe Gen3 x16, 16 GB/s each direction
│                                    Secondary GPU — NVK, Volta full-rate f64
│
├── Root Port 1c.2 ─── Bus [07] ─── Realtek RTL8125 2.5GbE
│                                    PCIe Gen2 x1
│
└── Root Port 1d.0 ─── Bus [08] ─── BrainChip AKD1000 NPU (08:00.0)
                                     PCIe 2.0 x1, 500 MB/s each direction
                                     Neural inference — taxonomy/anomaly
```

---

## P2P DMA Paths

### GPU → NPU (bypass CPU)

```
RTX 4070 (01:00.0) ──→ Root Complex ──→ AKD1000 (08:00.0)
                        ↑
                        No CPU round-trip needed
                        Data stays on PCIe fabric
```

**Bandwidth**: limited by NPU's PCIe 2.0 x1 = **500 MB/s**.
For taxonomy inference (256-element k-mer vector × f32 = 1 KB per read),
this supports **~500K classifications/sec** at full PCIe bandwidth.
In practice, NPU inference latency (~650µs round-trip) is the bottleneck,
not bandwidth.

**Key insight**: GPU compute results (diversity metrics, abundance vectors)
can be routed directly to NPU for classification without copying through
host DRAM. ToadStool's dispatch system manages this via buffer sharing.

### GPU → GPU (multi-GPU)

```
RTX 4070 (01:00.0) ──→ Root Complex ──→ Titan V (05:00.0)
```

**Bandwidth**: limited by the slower link (Gen3 x16 = 16 GB/s).
Both GPUs can run the same WGSL shaders via wgpu. The Titan V avoids
the NVVM f64 transcendental bug (Volta has native f64 exp/log/pow).

### CPU → GPU (standard ToadStool path)

```
i9-12900K ──→ Host DRAM ──→ PCIe RC ──→ RTX 4070
```

**Bandwidth**: PCIe Gen4 x16 = 32 GB/s.
This is the current ToadStool dispatch path. Data is prepared on CPU,
uploaded to GPU via buffer mapping, computed, and results read back.

### CPU → NPU (standard AKD1000 path)

```
i9-12900K ──→ Host DRAM ──→ PCIe RC ──→ AKD1000
```

**Bandwidth**: PCIe 2.0 x1 = 500 MB/s.
Current path via `/dev/akida0`. Direct weight injection bypasses the
Keras/QuantizeML pipeline (hotSpring finding).

---

## Mixed Hardware Pipeline

The wetSpring production pipeline routes workloads across all three substrates:

```
                    ┌─────────────────────────┐
                    │ FASTQ → DADA2 → ASVs    │ CPU (sequential, branching)
                    │ Chimera → Derep → Merge  │
                    └──────────┬──────────────┘
                               │ abundance vectors
                               ▼
                    ┌─────────────────────────┐
                    │ Diversity (FMR)          │
                    │ Spectral match (GEMM)    │ GPU (batch-parallel, f64)
                    │ ANI/SNP/dN/dS/Pan       │
                    │ HMM/ODE/RF/Felsenstein  │
                    └──────────┬──────────────┘
                               │ feature vectors
                               ▼
              ┌────────────────┴────────────────┐
              │                                  │
    ┌─────────┴─────────┐            ┌──────────┴──────────┐
    │  NPU (via PCIe)   │            │  CPU fallback       │
    │  Taxonomy FC      │            │  If NPU unavailable │
    │  Anomaly detect   │            │  Full-precision f64 │
    │  PFAS screening   │            │  Reference truth    │
    │  ~1mW per infer   │            │                     │
    └───────────────────┘            └─────────────────────┘
```

### Routing Rules

| Condition | Route | Rationale |
|-----------|-------|-----------|
| Batch size > dispatch breakeven | GPU | GPU parallelism beats CPU |
| Batch size < dispatch breakeven | CPU | Dispatch overhead > compute savings |
| Classification (taxonomy, anomaly) | NPU | Ultra-low power, field deployment |
| f64 transcendental on Ada Lovelace | GPU + polyfill | NVVM workaround needed |
| NPU unavailable | CPU fallback | Always correct, just slower |
| GPU unavailable | CPU only | 1,291 CPU checks prove correctness |

### Dispatch Overhead Budget

| Transfer | Latency | Bandwidth | Note |
|----------|---------|-----------|------|
| CPU → GPU buffer upload | ~50-200µs | 32 GB/s | Depends on buffer size |
| GPU shader dispatch | ~100-500µs | — | Includes compilation cache |
| GPU → CPU readback | ~100-500µs | 32 GB/s | Synchronous wait |
| CPU → NPU inference | ~650µs | 500 MB/s | PCIe 2.0 x1 dominated |
| **GPU → NPU (P2P)** | **~100-300µs** | **500 MB/s** | **Bypass CPU DRAM** |

---

## ToadStool Unidirectional Streaming

ToadStool's streaming dispatch eliminates per-stage round-trips:

```
Traditional:  CPU → GPU → CPU → GPU → CPU → GPU → CPU
                 ↑ dispatch  ↑ readback  ↑ dispatch  (6 PCIe transfers)

Streaming:    CPU → GPU ───→ GPU ───→ GPU → CPU
                 ↑ single upload         ↑ single readback (2 PCIe transfers)
```

For a 4-stage GPU pipeline (e.g., quality → diversity → spectral → ANI),
streaming reduces PCIe transfers from 8 to 2 — a **4× reduction** in
dispatch overhead.

---

## Verification

- Exp064: GPU math matches CPU (26/26 PASS)
- Exp065: Substrate-independence proven (35/35 PASS)
- Exp066: Scaling benchmark characterizes GPU crossover
- Exp072: Streaming pipeline proof — 1.27x speedup, zero CPU round-trips (17/17 PASS)
- Exp073: Dispatch overhead quantified — streaming beats individual at all batch sizes (21/21 PASS)
- Exp074: Substrate router — GPU↔NPU↔CPU routing, PCIe topology-aware (20/20 PASS)
- Exp075: Pure GPU 5-stage pipeline — 0.1% pipeline overhead (31/31 PASS)
- Exp076: Cross-substrate pipeline — GPU→NPU→CPU latency profiled (17/17 PASS)
- PCIe topology validated via `lspci -tv` on Eastgate system
