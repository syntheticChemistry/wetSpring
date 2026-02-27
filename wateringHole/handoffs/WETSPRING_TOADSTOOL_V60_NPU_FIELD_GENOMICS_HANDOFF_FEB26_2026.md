# toadStool / barracuda — V60 NPU Live + Field Genomics Handoff

**Date:** February 26, 2026
**From:** wetSpring
**To:** toadStool / barracuda core team
**Covers:** V60 NPU live hardware validation + field genomics architecture + data type profiling
**ToadStool Pin:** S68 (`f0feb226`)
**License:** AGPL-3.0-only

---

## Executive Summary

wetSpring V60 completes two major milestones and defines the next evolution
challenge:

1. **NPU Live on Real AKD1000** — 3 ESN classifiers (QS/Bloom/Disorder)
   validated on real neuromorphic hardware via the pure Rust `akida-driver`.
   60 new NPU validation checks. Online evolution, PUF fingerprinting,
   temporal streaming, and cross-reservoir crosstalk all validated.

2. **Field Genomics Architecture** — Sub-thesis 06 defines a MinION nanopore
   sequencer + AKD1000 NPU architecture for autonomous field genomics.
   metalForge extends to a fourth substrate class (Sequencer). 7 new
   experiments planned (Exp196-202).

3. **Data Type Profiling for NestGate** — Comprehensive catalog of all
   biological data types flowing through wetSpring (FASTQ, FASTA, FAST5,
   POD5, mzML, Newick, OTU tables, taxonomy, etc.), profiling what NestGate
   needs to evolve from generic blob storage to a biology-aware data primal.

---

## Part 1: How wetSpring Uses BarraCUDA (V60 State)

### Upstream Dependencies (lean — we use barracuda for this)

| barracuda Module | wetSpring Usage | Validation |
|------------------|-----------------|------------|
| `ops::linalg::lu_solve` | ESN reservoir weight solving (Cholesky) | 1,008 tests |
| `ops::linalg::BatchedEighGpu` | Anderson eigenvalues, PCoA, bifurcation | 1,578 GPU checks |
| `spectral::*` | Anderson 1D/2D/3D, Lanczos, level spacing ratio | 3,400+ Anderson checks |
| `ops::bio::*` | All 42 GPU bio modules (DADA2, chimera, taxonomy, diversity, etc.) | Lean — 0 local WGSL |
| `device::WgpuDevice` | GPU adapter, shader compilation | All GPU binaries |
| `pipeline::*` | FusedMapReduce, streaming, dispatch | 152 streaming checks |
| `linalg::eigh_f64` | CPU eigensolve for Anderson, PCoA | All spectral experiments |
| `numerical::{trapz, gradient_1d}` | ODE helpers, integral validation | 6 ODE models |

### Local Implementation (wetSpring-specific)

| wetSpring Module | Why Local | Absorption Candidate? |
|------------------|-----------|:---------------------:|
| `bio::esn` | Echo state network — bio-specific reservoir computing | **Yes** — generalizable to any domain |
| `bio::anderson_qs` | QS-disorder mapping — wetSpring domain science | No — stays local |
| `io::fastq`, `io::mzml`, `io::ms2` | File format I/O — consumer-side parsing | No — stays local (NestGate candidate) |
| `npu` module | NPU inference bridge (DMA, quantization) | **Partial** — `npu_infer_i8` generalizable |
| `tolerances` | 86 named constants — wetSpring-specific | No — stays local |
| `validation` | Validator harness — hotSpring pattern | **Yes** — shared pattern |

### Evolution Principle

> wetSpring owns domain biology. barracuda owns compute primitives.
> Local code that becomes a reusable compute primitive gets absorbed.
> Local code that encodes wetSpring-specific biology stays local.

---

## Part 2: What to Absorb

### 2a. ESN Reservoir Computing (`bio::esn`)

**Validated:** Exp114-119, 123, 194-195 (CPU sim + NPU live)
**Generalization:** ESN is domain-agnostic reservoir computing — used by
hotSpring (WDM surrogates), neuralSpring (time series), airSpring (crop
stress), and wetSpring (bloom/QS classifiers). Moving ESN to barracuda
enables all Springs to share the implementation.

**API:**
```rust
pub struct EsnConfig { input_size, reservoir_size, output_size, spectral_radius, leak_rate, seed }
pub struct Esn { w_in, w_res, w_out, config, state }
impl Esn {
    pub fn new(config: &EsnConfig) -> Self;
    pub fn train(config: &EsnConfig, inputs: &[Vec<f64>], targets: &[Vec<f64>], ridge: f64) -> Self;
    pub fn step(&mut self, input: &[f64]) -> Vec<f64>;
    pub fn classify(&mut self, input: &[f64]) -> usize;
    pub fn w_in(&self) -> &[f64];  // direct weight access for NPU bridge
    pub fn w_res(&self) -> &[f64];
    pub fn w_out(&self) -> &[f64];
    pub fn w_out_mut(&mut self) -> &mut [f64];  // mutable for online evolution
    pub const fn config(&self) -> &EsnConfig;
}
```

**Why absorb:** 4 Springs use ESN. NPU deployment requires weight access.
Online evolution (`w_out_mut`) is a pattern other Springs will need.

### 2b. NPU Inference Bridge (`npu` module)

**Validated:** Exp193-195 (real AKD1000 hardware)
**What's generalizable:**
- `npu_infer_i8` — single int8 inference via DMA
- `load_reservoir_weights` — f64 → f32 → SRAM
- `load_readout_weights` — online weight switching (86 µs)
- `npu_batch_infer` — batch inference with metrics
- `NpuInferResult`, `ReservoirLoadResult`, `NpuBatchResult`

**What stays local:** NPU-specific quantization schemes tied to wetSpring's
ESN classifiers. The DMA primitives themselves are generalizable.

### 2c. Validator Harness Pattern

Both hotSpring and wetSpring independently converged on the same validation
pattern: hardcoded expected values, named tolerance constants, explicit
pass/fail with exit code 0/1. This should be a shared barracuda primitive.

**Key types:**
- `Validator` — accumulates checks, reports, exits
- Named tolerance constants with scientific provenance
- `expect!` macro equivalent with tolerance + message

### 2d. metalForge Sequencer Substrate (Future)

When field genomics experiments begin, metalForge will need a `Sequencer`
substrate type alongside CPU/GPU/NPU. This is not ready for absorption yet
but should be tracked as an evolution target.

---

## Part 3: V60 NPU Results (Absorption-Relevant)

### Hardware Validation (Exp193-195)

| Metric | Value | Relevance to ToadStool |
|--------|-------|----------------------|
| DMA throughput | 37 MB/s (164 KB in 4.5 ms) | Sets baseline for NPU data transfer |
| Inference latency | 48 µs mean, 76 µs p99 | Sub-ms classification enables real-time feedback |
| Batch throughput | 20.7K infer/sec (8-wide) | NPU can serve multiple consumers |
| Power | 1.4 µJ/infer, 30 mW active | Coin-cell deployment viable |
| Weight switching | 86 µs for 3 hot swaps | Online model adaptation works |
| Evolution speed | 136 gen/sec (1+1)-ES | On-device learning practical |
| Streaming | 12.9K Hz sustained | Real-time time series processing |
| PUF entropy | 6.34 bits | Hardware-derived device identity |
| Crosstalk | 12.8K switch/sec, no bleed | Multi-classifier safe |

### New ESN Accessors (for absorption)

Added `w_in()`, `w_res()`, `w_out()`, `w_out_mut()`, `config()` to `Esn`
for direct weight manipulation by NPU bridge and online evolution.

---

## Part 4: Field Genomics Architecture (Evolution Preview)

### New BarraCUDA Modules Needed

| Module | What | Priority | Absorption? |
|--------|------|:--------:|:-----------:|
| `io::nanopore` | FAST5/POD5 raw signal reader | P0 | NestGate candidate |
| `bio::basecall` | Signal → base conversion | P1 | Yes — general compute |
| `io::minknow` | MinKNOW gRPC client | P2 | No — consumer-side |

### metalForge Evolution

```rust
pub enum SubstrateKind {
    Cpu,
    Gpu,
    Npu,
    Sequencer,  // NEW — sensing substrate, not compute
}
```

The dispatch loop becomes cyclic: SEQ → GPU → NPU → SEQ (adaptive sampling
feedback). This is a new pattern for metalForge — current dispatch is linear.

### Data Type Profiling

`specs/DATA_TYPES.md` catalogs every biological data type in wetSpring.
Key finding for barracuda: the library handles 15+ file formats and 20+
biological entity types. Most of the I/O should live in consumers (wetSpring,
NestGate) — barracuda should focus on compute primitives.

---

## Part 5: Quality State

| Metric | V60 Value |
|--------|-----------|
| `cargo fmt --check` | Clean |
| `cargo clippy --pedantic` | 0 warnings |
| `cargo doc --no-deps` | 0 warnings |
| `#![deny(unsafe_code)]` | Enforced |
| Named tolerances | 86 (all provenanced) |
| External C dependencies | 0 |
| TODO/FIXME/HACK | 0 |
| Tests | 1,008 |
| Coverage | 96.67% |
| Experiments | 200 completed, 7 planned |
| Checks | 4,748+ (1,578 GPU, 60 NPU) |
| Binaries | 186 |
| .rs files | 315 |

---

## Part 6: Absorption Roadmap

| Priority | What | From | To |
|:--------:|------|------|-----|
| **P0** | ESN reservoir computing | `wetspring::bio::esn` | `barracuda::ops::bio::esn` |
| **P0** | Validator harness | `wetspring::validation` + `hotspring::validation` | `barracuda::validation` |
| **P1** | NPU inference primitives | `wetspring::npu` | `barracuda::ops::npu` |
| **P1** | Int8 affine quantization | `wetspring::npu` (inline) | `barracuda::ops::quantize` |
| **P2** | Sequencer substrate type | `wetspring::metalForge` (planned) | `barracuda::substrate` |
| **P2** | POD5/Arrow reader | `wetspring::io::nanopore` (planned) | NestGate provider |
| **P3** | Basecalling GPU shader | `wetspring::bio::basecall` (planned) | `barracuda::ops::bio::basecall` |
