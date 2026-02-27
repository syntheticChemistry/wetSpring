# Exp207: BarraCuda GPU Parity v4 — IPC Science Capabilities on GPU

**Date:** 2026-02-27
**Status:** PASS (54/54 checks)
**Track:** cross/GPU/IPC
**Binary:** `cargo run --features gpu,ipc --release --bin validate_barracuda_gpu_v4`

## Hypothesis

Science capabilities dispatched through the IPC layer produce GPU↔CPU parity
when ToadStool compute dispatch is available. The math is truly portable:
CPU call == GPU call == IPC dispatch for the same input data.

## Method

Six validation domains:

| Domain | Checks | What it proves |
|--------|:------:|----------------|
| G01: GPU Diversity via Dispatch | 20 | 5 communities × 4 metrics — all `GPU_VS_CPU_F64` |
| G02: GPU Bray-Curtis | 3 | Large community pair + identical + range check |
| G03: GPU QS ODE | 8 | 4 scenarios × 2 checks (t_end, peak) — GPU↔CPU parity |
| G04: GPU Anderson Spectral | 7 | Dispatch availability + 2 lattice sizes × 3 disorder values |
| G05: Full Pipeline GPU Streaming | 6 | Pipeline completion + Shannon parity + Anderson available |
| G06: ToadStool Dispatch Model | 10 | 6+ capabilities, version, all 6 expected methods registered |

## Results

- 54/54 checks passed
- GPU diversity, QS ODE, and Anderson spectral all at GPU↔CPU parity
- ToadStool unidirectional streaming eliminates CPU round-trips
- Total wall-clock: 6.8s (debug, includes GPU device creation)

## Key Finding

The same math that runs on CPU also runs on GPU with parity preserved through
the IPC dispatch layer. ToadStool's compute dispatch makes the substrate
transparent — the IPC layer doesn't need to know whether the backend is CPU
or GPU.

## Modules Validated

`ipc::dispatch`, `bio::diversity`, `bio::qs_biofilm`, `barracuda::spectral`,
ToadStool GPU primitives (FusedMapReduceF64, BatchedOdeRK4, Lanczos)
