# Experiment 298: Full 5-Tier Validation Chain (V92F)

**Date:** March 2, 2026
**Status:** DONE
**Phase:** V92F
**Objective:** Confirm the complete paper-math → CPU → GPU → streaming → metalForge chain passes after ToadStool S86 rewire and cross-spring validation

---

## The Question

After the ToadStool S79→S86 rewire (fixing feature-gate bugs, adding 51 new
ComputeDispatch ops, and rewiring `BatchedMultinomialGpu::sample`), does the
full 5-tier validation chain still pass? This proves that every level of
abstraction preserves mathematical correctness from published paper equations
through to cross-substrate hardware dispatch.

## The Chain

```
Published equations (52 papers, 6 tracks)
  → BarraCuda CPU (pure Rust math, 10-1000× vs Python)
    → BarraCuda GPU (same answer on GPU, portable via ToadStool)
      → Pure GPU streaming (unidirectional, zero round-trips)
        → metalForge mixed hardware (GPU → NPU → CPU, substrate-independent)
```

## Results

### Tier 1: Paper Math Controls (Exp251/291)

| Binary | Checks | Result |
|--------|--------|--------|
| `validate_paper_math_control_v3` | 27/27 | PASS |

Validates core equations from 32 papers directly through BarraCuda CPU.
Open data: all papers use public datasets (NCBI BioProjects, EPA PFAS,
PDB structures, Zenodo repositories).

### Tier 2: BarraCuda CPU (Exp252-253/292)

| Binary | Checks | Result | Notes |
|--------|--------|--------|-------|
| `validate_barracuda_cpu_v19` | 42/42 | PASS | 7 domains pure Rust |
| `validate_barracuda_cpu_v20` | 37/37 | PASS | Vault + DF64 + cross-domain |
| `benchmark_python_vs_rust_v3` | 35/35 | PASS | 10-1000× speedup proven |
| `validate_cross_spring_s86` | 64/64 | PASS | S86 ungated CPU modules |

**Key result:** Pure Rust math matches published equations AND outperforms
Python/scipy/BioPython by 10-1000× with zero interpreter overhead.

### Tier 3: BarraCuda GPU (Exp254/264/268/293/297)

| Binary | Checks | Result | Notes |
|--------|--------|--------|-------|
| `validate_barracuda_gpu_v11` | 25/25 | PASS | Domain GPU portability |
| `validate_cpu_vs_gpu_v7` | 22/22 | PASS | 27-domain parity |
| `validate_cpu_vs_gpu_pure_math` | 38/38 | PASS | ToadStool primitive parity |
| `validate_cross_spring_modern_s86` | 46/46 | PASS | 264 ComputeDispatch ops |

**Key result:** Same math answer on GPU as CPU. DF64 Hybrid on RTX 4070
gives max error 1.75e-5 for GEMM (expected for double-float on FP32 cores).
ToadStool's `compile_shader_universal()` handles precision transparently.

### Tier 4: Pure GPU Streaming (Exp255/294)

| Binary | Checks | Result | Notes |
|--------|--------|--------|-------|
| `validate_pure_gpu_streaming_v8` | 43/43 | PASS | Unidirectional pipeline |

**Key result:** ToadStool unidirectional streaming eliminates N-1 round-trips.
Data stays on device through the entire pipeline: diversity → Bray-Curtis →
NMF → Anderson spectral → stats. 441-837× speedup vs round-trip dispatch.

### Tier 5: metalForge Mixed Hardware (Exp269-270/295)

| Binary | Checks | Result | Notes |
|--------|--------|--------|-------|
| `validate_mixed_hw_dispatch` | 91/91 | PASS | Cross-system NUCLEUS |
| `validate_biomeos_graph` | 29/29 | PASS | biomeOS coordination |

**Key result:** Same mathematical answer regardless of which hardware executes:
CPU, GPU, or NPU. metalForge routes computation to the optimal substrate based
on capability discovery, not hardcoded paths.

## Summary

| Tier | What It Proves | Total Checks | Status |
|------|---------------|:------------:|--------|
| Paper math | Equations match published papers | 27 | PASS |
| BarraCuda CPU | Pure Rust faster than Python | 178 | PASS |
| BarraCuda GPU | GPU = CPU (math portable) | 131 | PASS |
| Pure GPU streaming | Zero round-trips, data on-device | 43 | PASS |
| metalForge mixed | CPU = GPU = NPU (substrate portable) | 120 | PASS |
| **Total** | **End-to-end correctness** | **499** | **ALL PASS** |

## Hardware

- CPU: AMD (system default)
- GPU: NVIDIA GeForce RTX 4070 (Fp64Strategy::Hybrid, Precision::Df64)
- ToadStool: S86 (`2fee1969`), 264 ComputeDispatch ops
- No NPU hardware available — NPU paths validated via CPU emulation

## Open Data Confirmation

All 52 papers in the queue use publicly available data:
- NCBI BioProjects (PRJNA382322, PRJNA488170, PRJNA1195978, etc.)
- EPA PFAS surface water monitoring (Michigan DEQ, 3,719 records)
- MassBank spectral library (175 PFAS reference spectra)
- Zenodo (Jones Lab PFAS dataset 14341321)
- PDB/AlphaFold structures (Track 3 drug repurposing)
- EMP Atlas (30K samples, Anderson disorder mapping)
- SRA metagenomic data (Anderson deep-sea vents)

No proprietary data. No restricted access. Full reproducibility.
