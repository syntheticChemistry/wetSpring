# Experiment 046: GPU Phylogenetic Composition

**Date:** February 20, 2026
**Status:** COMPLETE — 15/15 PASS
**Track:** Cross-cutting (GPU)
**Binary:** `validate_gpu_phylo_compose` (requires `--features gpu`)

---

## Objective

Demonstrate that ToadStool's FelsensteinGpu can serve as a **drop-in GPU inner loop** for bootstrap resampling and phylogenetic placement. Convert CPU TreeNode to ToadStool's PhyloTree format, run FelsensteinGpu, and compare with CPU recursive and flat implementations.

## Method

1. **TreeNode → PhyloTree conversion** — Convert BarraCUDA's CPU TreeNode representation to ToadStool's PhyloTree format (level-order layout for GPU dispatch)
2. **FelsensteinGpu execution** — Run ToadStool's FelsensteinGpu primitive as the inner likelihood kernel
3. **Parity comparison** — Compare GPU results against CPU recursive (`felsenstein::prune`) and flat (`felsenstein::FlatTree`) implementations across three validation sections

## Results

### Section 1: CPU↔GPU Felsenstein Parity
| Case | CPU Result | GPU Result | Diff |
|------|------------|------------|------|
| 3-taxon tree | log-likelihood | log-likelihood | 0.00e0 |
| 5-taxon tree | log-likelihood | log-likelihood | 0.00e0 |

**Status:** CPU ≈ GPU with 0.00e0 diff

### Section 2: GPU Bootstrap RAWR
| Metric | CPU | GPU | Status |
|--------|-----|-----|--------|
| Replicates | 20 | 20 | PASS |
| max \|CPU−GPU\| | — | 0.00e0 | PASS |
| Mean match | ✓ | ✓ | PASS |
| Variance match | ✓ | ✓ | PASS |

**Status:** 20 replicates, max \|CPU−GPU\| = 0.00e0, mean and variance match

### Section 3: GPU Placement
| Metric | CPU | GPU | Status |
|--------|-----|-----|--------|
| Edges scanned | 5 | 5 | PASS |
| max \|CPU−GPU\| | — | 0.00e0 | PASS |
| Best edge agreement | ✓ | ✓ | PASS |

**Status:** 5 edges scanned, max \|CPU−GPU\| = 0.00e0, best edge agreement

## Key Findings

1. **ToadStool FelsensteinGpu is a drop-in replacement** for the CPU inner loop in higher-level phylogenetic workflows (bootstrap, placement)
2. **Zero tolerance violations** — CPU and GPU produce identical log-likelihoods at machine precision
3. **Composition validated** — Bootstrap resampling (RAWR) and phylogenetic placement both correctly compose FelsensteinGpu as the inner likelihood kernel

## References

- Exp029: Felsenstein pruning (CPU baseline)
- Exp031: Wang 2021 RAWR bootstrap
- Exp032: Alamin & Liu 2024 placement
- Exp045: ToadStool bio absorption (FelsensteinGpu)

## Files Changed

| File | Purpose |
|------|---------|
| `barracuda/src/bin/validate_gpu_phylo_compose.rs` | GPU phylo composition validator (15 checks) |

## Run

```bash
cargo run --bin validate_gpu_phylo_compose --features gpu
```
