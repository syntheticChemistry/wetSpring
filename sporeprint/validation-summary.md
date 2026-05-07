+++
title = "wetSpring Validation Summary"
description = "16S metagenomics, LC-MS, PFAS screening — 5,707+ checks across 376 experiments, 63/63 papers reproduced"
date = 2026-05-06

[taxonomies]
primals = ["barracuda", "toadstool", "biomeos", "nestgate", "squirrel", "coralreef"]
springs = ["wetspring", "hotspring", "neuralspring", "groundspring"]
+++

## Status

- **5,707+ checks** across 376 experiments — all passing
- **63/63 papers** reproduced (Waters, Liu, Cahill/Smallwood, Jones, Anderson)
- **1,077x GPU speedup** for spectral cosine matching
- **30 sovereign bio modules**, 1 runtime dependency (flate2)

## Key Validation Binaries

<!-- TODO: List your actual validation binary names here -->
- `validate_16s_pipeline` — full 16S from FASTQ to taxonomy
- `validate_diversity` — Shannon, Simpson, UniFrac
- `validate_gonzales_cpu_parity` — 43 cross-validated checks
- `validate_algae_16s` — real NCBI data (11.9M reads)

## Workload TOMLs

Available in `projectNUCLEUS/workloads/wetspring/` (11 workloads).

## See Also

- [wetSpring Science Hub](https://primals.eco/lab/springs/wetspring/) on primals.eco
- [baseCamp Papers 01, 03, 04, 05, 06](https://primals.eco/science/)
