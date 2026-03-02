# Experiment 303: Mixed Hardware NUCLEUS Orchestration — V92G

**Date:** March 2, 2026
**Status:** DONE
**Phase:** V92H
**Objective:** End-to-end mixed hardware pipeline with NUCLEUS atomics and biomeOS coordination

---

## Sections

| Section | Coverage | Checks | Status |
|---------|----------|:------:|--------|
| S1 | Multi-GPU dispatch — 3 GPUs, bandwidth tiers, diversity routing | 11 | PASS |
| S2 | GPU→NPU→CPU interleaved — 6 topology patterns | 8 | PASS |
| S3 | Topology decision matrix — all substrate pair transitions | 4 | PASS |
| S4 | Workload routing — all 54 workloads, standard + BW-aware | 107 | PASS |
| S5 | NUCLEUS coordination — Tower/Node/Nest, evolution tracking | 5 | PASS |
| S6 | Bandwidth decision matrix — 6 data sizes + 8 workloads | 14 | PASS |
| **Total** | **6 sections** | **147** | **ALL PASS** |

## Pipeline Topologies Validated

| Pattern | Stages | Chained | Round-trips | Streamable |
|---------|--------|---------|-------------|------------|
| GPU-only (4 stages) | 4 | 3 | 0 | yes |
| GPU→NPU (P2P bypass) | 3 | 2 | 0 | yes |
| GPU→CPU→GPU (roundtrip) | 3 | 0 | 2 | no |
| CPU→GPU→NPU→CPU | 4 | 1 | 2 | no |
| NPU→GPU→GPU→CPU | 4 | 1 | 2 | no |
| GPU→GPU→CPU→CPU→GPU | 5 | 1 | 3 | no |

## Command

```bash
cargo run -p wetspring-forge --release --bin validate_mixed_nucleus_v92g
```
