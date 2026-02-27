# Exp210 — Modern Cross-Spring Evolution Benchmark (ToadStool S68+)

**Date**: 2026-02-27
**Status**: PASS (24/24)
**Binary**: `benchmark_cross_spring_modern_s68plus`
**Features**: `gpu`

## Purpose

Validates and benchmarks the full cross-spring evolution at S68+ scale, with
provenance tracking showing how primitives flow between springs.

## What's New (V64 rewiring)

| Capability | Source | Wiring |
|-----------|--------|--------|
| `GpuF64::fp64_strategy()` | hotSpring S58 → ToadStool S67 | Surfaces `GpuDriverProfile::fp64_strategy()` |
| `GpuF64::optimal_precision()` | ToadStool S68 universal | Returns `F64` (compute) or `Df64` (consumer) |
| `submit_and_poll()` | ToadStool S68+ | Replaces raw `q.submit()` + `d.poll()` in 5 ODE + 1 GEMM module |
| `DispatchSemaphore` | ToadStool S68+ | Transparent via `submit_and_poll` (dGPU=8 permits) |
| Device-lost resilience | ToadStool S68+ | `catch_unwind` + `is_lost()` flag via `submit_and_poll` |

## Sections Validated

| § | Domain | Provenance | Checks |
|---|--------|-----------|--------|
| 0 | GPU Init + Modern Capabilities | ToadStool S68+ | 4 |
| 1 | CPU Diversity (500 taxa) | wetSpring → S64 Absorb | 3 |
| 2 | GPU DiversityFusion (5×10k) | wetSpring → S63 → GPU | 2 |
| 3 | hotSpring Special Functions + Anderson | hotSpring → S59 | 4 |
| 4 | airSpring/groundSpring Stats | S64/S66 | 6 |
| 5 | GPU GEMM (Precision-Aware) | wetSpring → S62 → S68 | 3 |
| 6 | NMF + Ridge Linalg | wetSpring → S58 | 2 |
| **Total** | | | **24** |

## Cross-Spring Evolution Map

| Spring | Contribution | Sessions |
|--------|-------------|----------|
| hotSpring | Fp64Strategy, DF64 core-streaming, Anderson spectral, lattice QCD, Validation harness, NVK/RADV workarounds | S58-S68 |
| wetSpring | Bio ODE ×5, diversity (Shannon/Simpson/Chao1/Bray-Curtis), DiversityFusion, GEMM cached, smith-waterman, Felsenstein, Gillespie, NMF, ridge, ESN, nanopore, NCBI pipeline | S41-S68 |
| neuralSpring | Pairwise ops (L2/Hamming/Jaccard), graph Laplacian, spatial payoff, swarm NN, spectral IPR, KL divergence | S54-S56 |
| airSpring | Regression (linear/quad/exp/log), hydrology (Hargreaves ET0, FAO-56), kriging, Richards PDE, moving window | S64-S66 |
| groundSpring | Bootstrap (RAWR Dirichlet), batched multinomial, percentile, mean | S64-S66 |
| ToadStool | 700 WGSL shaders, universal precision (F16/F32/F64/DF64), sovereign compiler, device-lost resilience, dispatch semaphore | S39-S68+ |

## Architecture Summary (RTX 4070)

| Property | Value |
|----------|-------|
| ToadStool alignment | S68+ (`e96576ee`) |
| Fp64Strategy | Hybrid (consumer GPU) |
| Optimal precision | Df64 (DF64 core-streaming) |
| Dispatch threshold | 10,000 elements |
| Local WGSL | 0 (fully lean) |
| Primitives consumed | 79 |
| Device-lost resilience | Active (submit_and_poll) |

## Timing Results

| Benchmark | Provenance | Time |
|-----------|-----------|------|
| CPU diversity (500 taxa) | wetSpring→S64 | 0.013ms |
| CPU DiversityFusion (5×10k) | wetSpring→S63 | 0.395ms |
| GPU DiversityFusion (5×10k) | wetSpring→S63→GPU | 52.687ms |
| Anderson 3D (L=8) | hotSpring→S59 | 2.589ms |
| trapz(x², 1000 pts) | cross-spring | 0.002ms |
| GEMM cold 256×256 | wetSpring→S62→S68 | 2.493ms |
| GEMM cached avg | wetSpring→S62→S68+ | 2.104ms |
| NMF 20×10 KL k=3 | wetSpring→S58 | 0.221ms |
| Ridge 20×5→2 | wetSpring→S59 | 0.002ms |

## Notes

- GPU DiversityFusion slower than CPU at 5×10k due to upload overhead on small workload.
  At production scale (100×100k+), GPU dominates via FusedMapReduceF64.
- GEMM cached dispatch (2.1ms/call) stable with submit_and_poll semaphore overhead.
- DF64 GEMM path available for ~10× throughput on consumer GPUs once
  `BatchedOdeRK4::generate_shader()` emits universal precision code.
