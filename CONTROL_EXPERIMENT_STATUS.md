# Control Experiment Status — wetSpring

**Last updated:** V180 (2026-05-19)
**Lib tests:** 1,962 passed, 0 failed, 2 ignored (pre-existing upstream module visibility)
**Experiment binaries:** 386 indexed (385 completed, 1 in progress)
**Clippy:** zero warnings (`--features ipc --lib -- -W clippy::pedantic -W clippy::nursery`)

## Experiment Summary

| Category | Count | Status |
|----------|-------|--------|
| CPU validation (`validate_barracuda_cpu_*`) | 28 | All green |
| GPU validation (`validate_barracuda_gpu_*`) | 14 | All green (requires GPU) |
| CPU vs GPU parity (`validate_cpu_vs_gpu_*`) | 9 | All green (requires GPU) |
| Cross-spring validation | 12 | All green (requires GPU) |
| Cross-spring benchmarks | 8 | All green (requires GPU) |
| ToadStool dispatch validation | 5 | All green (requires GPU) |
| Paper math controls (`validate_paper_math_control_*`) | 6 | All green |
| metalForge mixed hardware | 22 | All green |
| Hormesis chain (Exp377–379) | 3 | All green (V174) |
| Total experiment binaries | 345+ | Green |

## Control Chains

### Chain 1: Python → Rust CPU → Rust GPU

```
scripts/benchmark_python_baseline.py  (Tier 1: Python/numpy/scipy)
    ↓
validate_barracuda_cpu_v1..v27       (Tier 2: Rust CPU parity)
    ↓
validate_barracuda_gpu_v1..v14       (Tier 3: Rust GPU parity via wgpu)
    ↓
validate_cpu_vs_gpu_all_domains      (Tier 3: 16-domain head-to-head)
```

### Chain 2: Cross-Spring Evolution (ToadStool)

```
validate_cross_spring_s57 → s62 → s65 → s68 → s70 → s79 → s86 → s87 → s93 → v98
```

Sessions covered: S57, S62, S65, S68, S70, S79, S86, S87, S93, S130+

### Chain 3: NUCLEUS Atomics (metalForge)

```
validate_dispatch_routing              → Tower + Node + Nest discovery
validate_pcie_bypass_mixed_hw          → GPU→NPU bypass, GPU→GPU, CPU fallback
validate_mixed_hw_dispatch             → NUCLEUS + PCIe, 8-stage mixed pipeline
validate_nucleus_biomeos_v92g          → Tower/Nest/Node + biomeOS DAG, 53 workloads
validate_mixed_nucleus_v92g            → GPU→NPU→CPU→GPU interleaving
```

### Chain 4: Hormesis → Anderson → Colonization (Exp377–379)

```
validate_hormesis_biphasic   (17/17 PASS) → bio::hormesis + dose_to_disorder + Anderson W_c
validate_trophic_cascade     (10/10 PASS) → anderson_spectral::sweep + diversity under pesticide
validate_colonization_resistance (30/30 PASS) → binding_landscape + resistance_surface_sweep
```

### Chain 5: Paper Review Queue

63/63 papers reviewed. LTEE GuideStone queue: B7 TIER 2 COMPLETE, 9 queued.

## Pending Work

| Item | Blocked By | Priority |
|------|------------|----------|
| LTEE B1–B6, B8, E1, E5 queue | lithoSpore modules | Medium |
| Field genomics Exp197–202 | Field hardware | Low |
| EPA UCMR5 + PFOS datasets | Download + parse | Medium |
| GPU parity for NPU candidates | AKD1000 firmware | Low |

## Hardware Matrix

| Substrate | Validated | Coverage |
|-----------|-----------|----------|
| CPU (i9-12900K) | 252 lib + 28 CPU binaries | Full |
| GPU (RTX 4070 Ada) | 14 GPU + 9 CPU-vs-GPU binaries | Full |
| GPU (Titan V) | Cross-spring validators | Partial (NVK) |
| NPU (AKD1000) | metalForge dispatch routing | Routing only |
