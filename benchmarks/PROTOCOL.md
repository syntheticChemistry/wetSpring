# wetSpring Benchmark Protocol

## Overview

Three-tier benchmark comparing identical scientific workloads across substrates:

| Tier | Substrate | Binary / Script |
|------|-----------|----------------|
| 1 | Python (numpy/scipy) | `scripts/benchmark_python_baseline.py` |
| 2 | Rust CPU | `barracuda/src/bin/benchmark_pipeline.rs` |
| 3 | Rust GPU (BarraCUDA/ToadStool) | `barracuda/src/bin/benchmark_cpu_gpu.rs` |

## How to Run

```bash
# Full three-tier run (recommended)
./scripts/run_three_tier_benchmark.sh

# Individual tiers
python3 scripts/benchmark_python_baseline.py          # Tier 1
cargo run --release --bin benchmark_pipeline           # Tier 2
cargo run --release --features gpu --bin benchmark_cpu_gpu  # Tier 3
```

## Prerequisites

- **Python 3** with `numpy`, `scipy` (`pip install numpy scipy`)
- **Rust** toolchain (`rustup`, edition 2021, MSRV 1.82)
- **NVIDIA GPU** + `nvidia-smi` for Tier 3
- **Intel RAPL** access for CPU energy (optional but recommended):
  ```bash
  sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
  ```

## Measurement Methodology

### Timing

- **Warmup**: 3 iterations discarded before measurement.
- **Calibration**: Iteration count auto-scaled until total elapsed > 10 ms or 1000 iterations reached.
- **Clock**: `std::time::Instant` (Rust), `time.perf_counter()` (Python).
- **Metric**: microseconds per evaluation (`per_eval_us`), wall time (`wall_time_s`).

### Energy

- **CPU**: Intel RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`), delta in Joules. Handles counter wrap via `max_energy_range_uj`.
- **GPU**: nvidia-smi continuous polling at 100 ms interval. Power integrated via trapezoidal rule. Reports average/peak watts, peak temperature, peak VRAM.
- **Fallback**: If RAPL unavailable, TDP-based estimate (CPU TDP × wall time).

### Memory

- **Peak RSS**: Read from `/proc/self/status` (`VmHWM` field), reported in MB.
- **GPU VRAM**: Peak `memory.used` from nvidia-smi samples.

## JSON Schema

All tiers emit JSON matching this schema. Files are saved to `benchmarks/results/`.

```json
{
  "timestamp": "2026-02-19T12:00:00",
  "hardware": {
    "gate_name": "string",
    "cpu_model": "string",
    "cpu_cores": 0,
    "cpu_threads": 0,
    "cpu_cache_kb": 0,
    "ram_total_mb": 0,
    "gpu_name": "string",
    "gpu_vram_mb": 0,
    "gpu_driver": "string",
    "gpu_compute_cap": "string",
    "os_kernel": "string",
    "rust_version": "string"
  },
  "phases": [
    {
      "phase": "Shannon entropy N=1000000",
      "substrate": "Python (numpy/scipy) | Rust CPU | BarraCUDA GPU",
      "wall_time_s": 0.007,
      "per_eval_us": 7000.0,
      "n_evals": 142,
      "energy": {
        "cpu_joules": 0.85,
        "gpu_joules": 0.0,
        "gpu_watts_avg": 0.0,
        "gpu_watts_peak": 0.0,
        "gpu_temp_peak_c": 0.0,
        "gpu_vram_peak_mib": 0.0,
        "gpu_samples": 0
      },
      "peak_rss_mb": 32.0,
      "notes": ""
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `phase` | string | Workload name with parameter (e.g. "Shannon entropy N=1000000") |
| `substrate` | string | Execution substrate identifier |
| `wall_time_s` | float | Wall-clock time for one evaluation (seconds) |
| `per_eval_us` | float | Time per evaluation (microseconds) |
| `n_evals` | int | Number of evaluations in the timed region |
| `energy.cpu_joules` | float | CPU energy from RAPL (Joules), 0 if unavailable |
| `energy.gpu_joules` | float | GPU energy integrated from nvidia-smi (Joules) |
| `energy.gpu_watts_avg` | float | Average GPU power draw (Watts) |
| `energy.gpu_watts_peak` | float | Peak GPU power draw (Watts) |
| `energy.gpu_temp_peak_c` | float | Peak GPU temperature (Celsius) |
| `energy.gpu_vram_peak_mib` | float | Peak GPU VRAM usage (MiB) |
| `energy.gpu_samples` | int | Number of nvidia-smi samples collected |
| `peak_rss_mb` | float | Peak process RSS (MB) |
| `notes` | string | Free-text annotation |

## Workloads

### Single-Vector Reductions

| Workload | Sizes | Algorithm |
|----------|-------|-----------|
| Shannon entropy | 1K, 10K, 100K, 1M | −Σ pᵢ ln(pᵢ) |
| Simpson diversity | 1K, 10K, 100K, 1M | 1 − Σ pᵢ² |
| Variance | 1K, 10K, 100K, 1M | Σ(xᵢ − x̄)²/n |
| Dot product | 1K, 10K, 100K, 1M | Σ aᵢbᵢ |

### Pairwise N×N

| Workload | Sizes | Algorithm |
|----------|-------|-----------|
| Bray-Curtis matrix | 10², 20², 50², 100² samples × 500 species | Σ|aᵢ−bᵢ| / Σ(aᵢ+bᵢ) |
| Spectral cosine | 10², 50², 100², 200² spectra × 500 bins | a·b / (‖a‖·‖b‖) |

### Matrix Algebra

| Workload | Sizes | Algorithm |
|----------|-------|-----------|
| PCoA | 10², 20², 30² | Classical MDS via eigendecomposition |

### Pipeline (Tier 2 only)

Full 16S metagenomics pipeline on real NCBI public data:
FASTQ → Quality filter → Dereplicate → DADA2 → Chimera → Taxonomy → Diversity

## Hardware Gate

Results are only comparable when collected on the same hardware. The `hardware` section of the JSON captures the exact configuration. Gate name should include the machine identifier.

## Reproduction

```bash
git clone https://github.com/syntheticChemistry/wetSpring
cd wetSpring
pip install numpy scipy
./scripts/run_three_tier_benchmark.sh
ls benchmarks/results/
```
