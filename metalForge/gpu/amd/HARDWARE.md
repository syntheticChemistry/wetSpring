# AMD GPU Characterization — Future Target

**Status**: No AMD GPU available locally. This documents the opportunity.

---

## Target: RX 7900 XTX (RDNA3)

| Property | Value |
|----------|-------|
| Architecture | RDNA 3 (Navi 31) |
| Compute Units | 96 |
| VRAM | 24 GB GDDR6 |
| Memory BW | 960 GB/s |
| **Infinity Cache** | **96 MB** |
| TDP | 355W |
| Estimated Price | ~$700 |

---

## Why AMD Matters for Life Science

### The Infinity Cache Advantage

AMD RDNA3's 96MB Infinity Cache is 2.7× larger than NVIDIA RTX 4070's 36MB L2.
For bioinformatics workloads where the entire working set fits in cache:

| Workload | Working Set | Fits in 96MB? | Impact |
|----------|-------------|---------------|--------|
| Smith-Waterman (10k × 10k) | 400 MB | No — streams | Bandwidth-limited |
| Smith-Waterman (1k × 1k) | 4 MB | **Yes** | Compute-limited → fast |
| HMM (1M observations × 4 states) | 32 MB | **Yes** | All forward/backward cache-resident |
| Felsenstein (100k sites × 4 states) | 3.2 MB | **Yes** | Trivially cache-resident |
| Pairwise alignment batch (100 pairs × 1k×1k) | 400 MB | No | Stream from VRAM |
| Diversity (10k samples × 10k OTUs) | 800 MB | No | GEMM → bandwidth |

The sweet spot: moderate-scale phylogenetic workloads (100-100k sites) where
the entire computation fits in Infinity Cache. Zero DRAM pressure.

### wgpu Compatibility

BarraCUDA/ToadStool already supports AMD via Vulkan backend. No code changes
needed — only characterization and tuning.

---

## Budget Option: RX 7800 XT

| Property | Value |
|----------|-------|
| Infinity Cache | 64 MB |
| VRAM | 16 GB |
| Price | ~$450 |

Still 1.8× NVIDIA's L2. Sufficient for most phylogenetic workloads.

---

## Required Experiments (When Hardware Available)

1. Run existing GPU validation binaries (`validate_diversity_gpu`, `validate_16s_pipeline_gpu`)
2. Measure Felsenstein per-site throughput vs NVIDIA
3. Profile HMM forward/backward with Infinity Cache monitoring
4. Compare Smith-Waterman wavefront: cache-resident vs streaming
5. Document any wgpu backend differences (shader compilation time, dispatch latency)
