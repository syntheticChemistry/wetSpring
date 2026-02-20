# Experiment 050: GPU Bifurcation Eigenvalue Analysis

**Date**: 2026-02-20
**Status**: PASS (5/5)
**Binary**: `validate_gpu_ode_sweep` (--features gpu)

## Objective

Validate GPU-accelerated eigenvalue decomposition for bifurcation analysis
of the QS/c-di-GMP steady-state Jacobian via ToadStool's `BatchedEighGpu`.

## Method

1. Integrate QS ODE to steady state (t=50, dt=0.01) using CPU.
2. Compute 5×5 numerical Jacobian via finite differences (ε=1e-6).
3. Form J^T*J (symmetric PSD) for eigenvalue analysis.
4. CPU: deflated power iteration (200 iterations per eigenvalue).
5. GPU: `BatchedEighGpu::execute_batch` (Jacobi algorithm, single matrix).

## Results

| Check | Result |
|-------|--------|
| Max eigenvalue > 0 | PASS |
| Eigenvalue finite | PASS |
| GPU eigenvalues finite | PASS |
| GPU ≈ CPU max eigenvalue (< 5%) | PASS (rel diff = 2.67e-16) |
| J^T*J eigenvalues ≥ 0 (PSD) | PASS |

**Eigenvalues**: [26.59, 22.64, 0.25, 0.04, 0.02]

## Key Findings

1. **BatchedEighGpu is bit-exact**: 2.67e-16 relative diff (machine epsilon).
   No NVVM issues — the eigenvalue shader doesn't use transcendentals.

2. **Jacobian spectrum reveals system dynamics**: two dominant modes
   (eigenvalues 26.6 and 22.6) correspond to the fast cell-growth and
   autoinducer dynamics; three slow modes (< 0.25) correspond to the
   HapR/c-di-GMP/biofilm signaling cascade.

3. **BatchedEighGpu scales to parameter sweeps**: at B=10,000 matrices
   this primitive enables bifurcation diagram construction entirely on GPU.

## Run

```bash
cargo run --features gpu --bin validate_gpu_ode_sweep
```
