# NVIDIA GPU Characterization — Life Science Workloads

**Devices**: RTX 4070 (primary) + Titan V (secondary)
**Purpose**: Document GPU capabilities for wetSpring's bioinformatics algorithms

---

## Local Hardware

### RTX 4070 (Primary)

| Property | Value |
|----------|-------|
| Architecture | Ada Lovelace (AD104) |
| CUDA Cores | 5,888 |
| VRAM | 12 GB GDDR6X |
| Memory BW | 504 GB/s |
| L2 Cache | 36 MB |
| TDP | 200W |
| PCIe Slot | `01:00.0` |
| f64 throughput | **1:2 via wgpu** (BarraCUDA discovery) |

### Titan V (Secondary — NVK)

| Property | Value |
|----------|-------|
| Architecture | Volta (GV100) |
| CUDA Cores | 5,120 |
| VRAM | 12 GB HBM2 |
| Memory BW | 653 GB/s |
| L2 Cache | 4.5 MB |
| TDP | 250W |
| PCIe Slot | `05:00.0` |
| f64 throughput | Native 1:2 (Volta full-rate) |

---

## Validated GPU Performance (wetSpring)

| Workload | Speedup | Parity | Experiment |
|----------|---------|--------|------------|
| Spectral cosine (2048×2048) | 926× | ≤1e-10 | Exp016 |
| Full 16S pipeline (10 samples) | 2.45× | 88/88 checks | Exp016 |
| Shannon/Simpson diversity | 15-25× | ≤1e-6 | Exp016 |
| Bray-Curtis distance matrix | High | ≤1e-10 | Exp016 |

---

## Life Science GPU Promotion Targets

### Tier 1: High-Impact (validated CPU, ready for GPU)

| Algorithm | Expected Parallelism | Memory Pattern | Estimated Speedup |
|-----------|---------------------|----------------|-------------------|
| **Felsenstein pruning** | Sites × edges | Regular, column-major | 100-500× |
| **Phylogenetic placement** | Edges × queries | Independent insertions | 50-200× |
| **Bootstrap resampling** | N replicates | Column-major + likelihood | 100-1000× |
| **Smith-Waterman** | Anti-diagonals | Wavefront, O(mn) cells | 50-100× |

### Tier 2: Medium-Impact

| Algorithm | Expected Parallelism | Notes |
|-----------|---------------------|-------|
| HMM forward/backward | States per step | Sequential over time, parallel over states |
| ODE parameter sweeps | Parameters | Each (cost, efficiency, defense) independent |
| Bifurcation scanning | Parameter grid | ~100 ODE solves per scan |

### Tier 3: Exploration

| Algorithm | Challenge | Notes |
|-----------|-----------|-------|
| Gillespie SSA | Inherently sequential per trajectory | Parallel across trajectories |
| Game theory equilibria | Small systems, CPU-fast | Only useful at population scale |

---

## Cache Considerations for Life Science

The RTX 4070's 36MB L2 cache is relevant for bioinformatics:

| Data Structure | Typical Size | Fits in L2? |
|----------------|-------------|-------------|
| HMM transition matrix (4×4 f64) | 128 bytes | Trivially |
| SW scoring matrix (1000×1000 i32) | 4 MB | Yes |
| Felsenstein partials (1000 sites × 4 states × f64) | 32 KB | Trivially |
| Bootstrap alignment (1000 sites × 100 taxa × 1B) | 100 KB | Trivially |
| Placement: all-edge partials | ~200 KB | Trivially |
| Spectral cosine (2048 spectra × 2048 bins × f64) | 33 MB | Borderline |

Most phylogenetic workloads fit comfortably in L2. This means GPU memory
bandwidth is not the bottleneck — compute throughput is. The 1:2 f64
throughput becomes the dominant factor.

---

## RTX 4070 vs Titan V for Life Science

| Factor | RTX 4070 | Titan V | Winner |
|--------|----------|---------|--------|
| f64 throughput | 1:2 (via wgpu) | 1:2 (native) | Tie |
| Memory BW | 504 GB/s | 653 GB/s | Titan V |
| L2 Cache | 36 MB | 4.5 MB | RTX 4070 |
| HBM2 latency | N/A | Low | Titan V |
| Power | 200W | 250W | RTX 4070 |
| Price (used) | ~$500 | ~$300 | Titan V |
| Driver | Proprietary 580.x | NVK (open source) | Depends on needs |

For wetSpring's phylogenetic workloads (small working sets, compute-bound),
the RTX 4070's larger L2 likely wins. For large-scale alignment (memory-bound),
the Titan V's HBM2 bandwidth helps.

**Both produce identical math** — validated via BarraCUDA's wgpu path.

---

## Next Steps

1. Run Felsenstein pruning on GPU via ToadStool dispatch
2. Measure per-site throughput (target: 1M sites/sec on RTX 4070)
3. Profile Smith-Waterman wavefront on both GPUs
4. Compare bootstrap replicate throughput: GPU vs 24-thread CPU
