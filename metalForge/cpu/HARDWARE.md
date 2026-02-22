# CPU Characterization — Intel i9-12900K

**Purpose**: Document CPU as the reference substrate and baseline for all
GPU/NPU comparisons. Identify workloads that should stay CPU-only.

---

## Hardware

| Property | Value |
|----------|-------|
| Architecture | Alder Lake (12th Gen) |
| P-cores | 8 (Raptor Cove, HT → 16 threads) |
| E-cores | 8 (Gracemont, no HT → 8 threads) |
| Total threads | 24 |
| L3 Cache | 30 MB shared |
| Max clock | P: 5.2 GHz, E: 3.9 GHz |
| TDP | 125W (PBP), 241W (MTP) |
| Memory | DDR5 (assumed, based on platform) |

---

## CPU-Only Workloads in wetSpring

These algorithms should remain CPU-only due to branching, recursion, or
sequential dependencies that make GPU dispatch overhead exceed compute savings:

| Algorithm | Module | Why CPU | Measured Performance |
|-----------|--------|---------|---------------------|
| Newick parsing | `bio::unifrac` | Recursive descent parser | <1ms for 1000-tip trees |
| DADA2 denoising (outer loop) | `bio::dada2` | Complex branching, EM iteration | GPU E-step accelerated |
| Chimera detection | `bio::chimera` | K-mer hash lookups, voting | Hash-heavy |
| Robinson-Foulds distance | `bio::robinson_foulds` | Bipartition set operations | <1ms for 100-tip trees |
| File I/O | `io::fastq`, `io::mzml`, `io::ms2` | Streaming parse, zero-copy | I/O bound, not compute |
| Cooperation game theory | `bio::cooperation` | 4-var ODE, trivial compute | <10ms per scenario |

---

## CPU as Validation Reference

All GPU and NPU results are validated against CPU (f64) implementations.
The CPU is the mathematical ground truth:

```
Python baseline → Rust CPU (reference) → GPU/NPU (must match CPU)
```

### BarraCUDA CPU Parity — All 25 Domains (Exp035–Exp062)

205/205 checks pass across 25 algorithmic domains + 6 ODE flat modules,
proving Rust CPU math matches Python across every validated module:

**v1: 21 checks (9 domains)**

| Domain | Checks | Tolerance | Notes |
|--------|:------:|-----------|-------|
| ODE integration | 3 | 1e-2 (RK4 vs odeint) | Fixed vs adaptive step |
| Stochastic simulation | 1 | 5% (ensemble statistics) | Inherent stochasticity |
| HMM | 8 | 1e-6 | Log-space arithmetic |
| Smith-Waterman | 3 | Exact | Integer scoring |
| Felsenstein | 1 | 1e-8 | f64 transcendentals |
| Diversity | 2 | 1e-10 | ln() precision |
| Signal processing | 1 | 0 | Exact peak positions |
| Game theory | 1 | 0 | ODE steady states |
| Tree distance | 1 | 0 | Integer metric |

**v2: 18 checks (batch/flat APIs)**

| Domain | Checks | Notes |
|--------|:------:|-------|
| FlatTree Felsenstein | 3 | GPU-ready data layout |
| Batch HMM | 4 | Parallel over sequences |
| Batch Smith-Waterman | 3 | Score batch API |
| Neighbor-Joining | 4 | NJ tree construction |
| DTL Reconciliation | 4 | Host-microbe coevolution |

**v3: 45 checks (9 new domains)**

| Domain | Checks | Notes |
|--------|:------:|-------|
| Multi-signal QS | 3 | Srivastava 2011 |
| Phage defense | 3 | Hsueh 2022 |
| Bootstrap resampling | 4 | Wang 2021 |
| Phylogenetic placement | 5 | Alamin & Liu 2024 |
| Decision tree | 8 | sklearn parity |
| Spectral matching | 5 | Cosine similarity |
| Extended diversity | 10 | Pielou, BC, Chao1 |
| K-mer counting | 4 | 2-bit encoding |
| Integrated pipeline | 3 | End-to-end |

### Rust vs Python Timing (Exp043)

| Platform | Total (µs) | Notes |
|----------|-----------|-------|
| Rust (release) | ~84,500 | v1 ~60,000 + v3 ~24,500 |
| Python (CPython) | ~1,749,000 | Pure Python, no numpy for ODE |
| **Speedup** | **~20x** | Compiled native vs interpreted |

ODE-heavy domains show 3-5x speedup. SSA shows 4-8x from memory layout.

---

## P-Core vs E-Core Considerations

The i9-12900K's hybrid architecture presents an optimization opportunity:

| Core Type | Best For | wetSpring Usage |
|-----------|----------|-----------------|
| P-core (5.2 GHz) | Latency-sensitive, single-threaded | ODE integration, DADA2, parsing |
| E-core (3.9 GHz) | Throughput, background | Batch file I/O, validation suite |

Currently we don't pin threads to specific cores. For production deployment,
the orchestrator should:
1. Pin DADA2/ODE/parsing to P-cores (latency-sensitive)
2. Run validation/benchmark suite on E-cores (throughput)
3. Leave 2 P-cores free for GPU dispatch (wgpu poll) and NPU I/O

---

## CPU Performance Baseline

| Workload | Time (release) | Notes |
|----------|---------------|-------|
| All 650 Rust tests | ~0.8s | 587 lib + 50 integration + 13 doc |
| All 48 CPU validation binaries | ~5s (total) | Sequential execution |
| Full 16S pipeline (10 samples) | ~2.1s | CPU path |
| BarraCUDA CPU v1 (21 checks) | ~60ms | 9 domains (release) |
| BarraCUDA CPU v3 (45 checks) | ~25ms | 9 new domains (release) |
| BarraCUDA CPU v4 (44 checks) | ~1.3ms | 5 new Track 1c domains (release) |
| BarraCUDA CPU v5 (29 checks) | ~62µs | RF + GBM (release) |
| All CPU parity (157 checks) | ~87ms | 25 domains total |
| Spectral cosine (2048×2048) | ~4.8s | CPU baseline (926× slower than GPU) |

The 16S pipeline is where CPU→GPU transition provides the most user-visible
improvement: from 2.1s to 0.86s for real-world sample counts.
