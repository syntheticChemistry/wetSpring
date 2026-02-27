# Field Genomics Requirements — Sub-thesis 06 Systems

**Last Updated:** February 26, 2026
**Purpose:** Define the systems, modules, and infrastructure needed to execute
the field genomics research programs (Sub-thesis 06 / Gen3 Paper 09).

---

## 1. New BarraCUDA Modules

### 1.1 `io::nanopore` — Raw Signal Reader

**Priority:** P0 (blocks all field genomics experiments)

**What:** Read Oxford Nanopore raw signal files (FAST5 and POD5 formats) and
expose them as streaming iterators for downstream processing.

**FAST5 format:**
- HDF5 container (binary)
- Groups per read: raw signal (int16 array), channel info, context tags
- Multi-read FAST5: multiple reads per file
- Dependency: HDF5 reader (consider `hdf5-rust` crate or minimal sovereign parser)

**POD5 format:**
- Apache Arrow IPC (binary)
- Read table: read_id, signal (int16), channel, pore type, calibration
- Dependency: Arrow reader (consider `arrow-rs` or minimal parser)

**API sketch:**
```rust
pub struct NanoporeRead {
    pub read_id: String,
    pub signal: Vec<i16>,
    pub channel: u32,
    pub sample_rate: f64,
    pub calibration_offset: f64,
    pub calibration_scale: f64,
}

pub struct NanoporeIter<R> { /* streaming */ }

pub fn stream_fast5(path: &Path) -> Result<NanoporeIter<...>, Error>;
pub fn stream_pod5(path: &Path) -> Result<NanoporeIter<...>, Error>;
```

**Validation pattern:** Synthetic signal → known basecall → round-trip check.
hotSpring pattern: hardcoded expected values, explicit pass/fail.

**Dependencies to evaluate:**
- `hdf5` crate (C binding via HDF5 lib) — violates sovereign constraint
- Minimal HDF5 parser in Rust — significant effort but sovereign
- POD5 via `arrow-rs` — pure Rust, aligns with ecosystem

**Decision:** Start with POD5 (pure Rust via Arrow). FAST5 support as Phase 2
if needed for legacy data. POD5 is Oxford Nanopore's forward direction.

### 1.2 `bio::basecall` — Signal to Base Conversion

**Priority:** P1 (can initially delegate to Dorado and consume FASTQ output)

**What:** Convert raw nanopore signal (int16 ionic current) to DNA bases.
This is a neural network inference task — the signal is processed by a
trained model to produce base calls and quality scores.

**Options:**
1. **Delegate to Dorado** (ONT's basecaller) — call as external process,
   consume FASTQ output. Fastest path to working pipeline. Not sovereign.
2. **Port Dorado models to BarraCUDA** — re-implement basecalling in Rust.
   Sovereign but significant effort (Dorado uses CTC/CRF models).
3. **NPU basecalling** — run basecalling model on AKD1000. Novel research.
   Requires model conversion to int8 + Akida-compatible architecture.
4. **GPU basecalling via BarraCUDA** — WGSL shader for CTC/CRF inference.
   Leverages existing ToadStool GPU infrastructure.

**Recommended path:** Option 1 first (Dorado as subprocess), then Option 4
(GPU basecalling shader) as a ToadStool Write → Absorb target.

**Validation:** Compare basecalled FASTQ from our pipeline vs Dorado output
on the same raw data. Per-read accuracy, per-base accuracy, Q-score
distribution.

### 1.3 `io::minknow` — MinKNOW API Client

**Priority:** P2 (needed for adaptive sampling, not for basic pipeline)

**What:** Client for Oxford Nanopore's MinKNOW gRPC API. Enables real-time
control of the sequencer: start/stop sequencing, read adaptive sampling
decisions, query run metrics.

**Key endpoints:**
- `acquisition` — start, stop, get current run info
- `data` — get live reads (signal chunks as they're sequenced)
- `analysis_configuration` — set adaptive sampling targets
- `device` — get device info, flow cell position, temperature

**For adaptive sampling:**
- Receive partial read signal as it translocates through the pore
- BarraCUDA processes partial signal (k-mer classification or partial basecall)
- NPU classifies: target (keep) or non-target (reject)
- Send reject signal back to MinKNOW to eject the read

**Dependency:** gRPC client. Consider `tonic` (pure Rust gRPC).

---

## 2. Experiment Plan

| Exp | Name | Module(s) | Prerequisite | Validation |
|-----|------|-----------|-------------|------------|
| 196 | Nanopore Signal Bridge | `io::nanopore` | POD5 test files | Synthetic signal → basecall round-trip |
| 197 | NPU Adaptive Sampling | `io::minknow`, `npu` | MinION hardware | NPU classification latency < MinKNOW decision window |
| 198 | Field Bloom Sentinel E2E | All new + existing 16S | MinION hardware | Full pipeline: sample → sequencing → classification → alert |
| 199 | Soil 16S Field Pipeline | `io::nanopore`, `bio::taxonomy` | MinION hardware, soil samples | MinION 16S → Anderson W matches Illumina reference |
| 200 | Soil Health NPU Classifier | `bio::esn`, `npu` | Exp199 data | NPU classifies soil health from MinION 16S |
| 201 | AMR Gene Detection | `bio::alignment`, `io::nanopore` | MinION hardware, wastewater | Long-read → resistance gene identification accuracy |
| 202 | AMR Threat NPU Classifier | `bio::esn`, `npu` | Exp201 data | NPU classifies AMR profile severity |

### Pre-Hardware Experiments (Can Start Now)

These experiments use simulated nanopore data to validate pipeline components
before hardware arrives:

| Exp | Name | What It Proves |
|-----|------|---------------|
| 196a | POD5 Parser Validation | `io::nanopore` reads synthetic POD5, round-trips signal |
| 196b | Simulated Long-Read 16S | Synthetic nanopore-length 16S reads through BarraCUDA pipeline |
| 196c | Int8 Quantization from Noisy Reads | Nanopore-quality reads → ESN features → NPU classification |

---

## 3. Hardware Requirements

### Minimum Viable Kit

| Item | Purpose | Estimated Cost |
|------|---------|:--------------:|
| MinION Mk1D | Nanopore sequencer | ~$1,000 (starter pack) |
| Flow cells (R10.4.1) × 3 | Consumable sequencing substrate | ~$2,700 |
| Rapid Barcoding Kit (SQK-RBK114) | 16S amplicon library prep, 10 min | ~$600 |
| 16S Barcoding Kit (SQK-16S114) | Full-length 16S amplicon | ~$600 |
| DNA extraction kit (Zymo Quick-DNA) | Environmental sample prep | ~$200 |
| **Total** | | **~$5,100** |

### Enhanced Kit (Mk1C for Edge Compute)

| Item | Purpose | Estimated Cost |
|------|---------|:--------------:|
| MinION Mk1C | Sequencer with onboard ARM+GPU | ~$4,900 |
| Flow cells × 5 | Extended experiments | ~$4,500 |
| Rapid + 16S + Ligation kits | Multiple library prep options | ~$1,800 |
| **Total** | | **~$11,200** |

### Already Available (No Additional Cost)

| Item | Purpose |
|------|---------|
| AKD1000 NPU (PCIe) | Edge classification — live, validated |
| RTX 4070 GPU | Basecalling acceleration, batch Anderson |
| i9-12900K CPU | General compute, fallback |
| BarraCUDA + ToadStool | 200 experiments of validated math |
| NestGate v4.1.0 | Content-addressed storage |
| Songbird | Network telemetry |

---

## 4. NestGate Integration Points

Field genomics creates the highest-bandwidth data path NestGate has faced.
A single MinION flow cell generates 5-50 GB of raw signal + 1-5 GB of FASTQ.

| Integration | NestGate Role | Priority |
|------------|--------------|:--------:|
| Raw signal archive | Store POD5 files, content-addressed | P1 |
| Basecalled reads | Store FASTQ, link to raw signal | P1 |
| Reference databases | Serve SILVA, NCBI taxonomy to pipeline | P0 |
| OTU/ASV tables | Store community profiles with metadata | P1 |
| NPU weight deployment | Serve ESN weights to AKD1000 | P2 |
| Provenance chain | Sample → extraction → sequencing → basecall → classification | P1 |
| Multi-station sync | Sync data between field units via Songbird | P2 |

---

## 5. metalForge Extension

### New Substrate Type: Sequencer

```rust
pub enum SubstrateKind {
    Cpu,
    Gpu,
    Npu,
    Sequencer,  // NEW
}

pub struct SequencerCapability {
    pub device: SequencerDevice,  // Mk1D, Mk1C, PromethION
    pub flow_cell: Option<FlowCellInfo>,
    pub pore_count: u32,
    pub basecalling: BasecallCapability,  // Onboard (Mk1C) or External
    pub adaptive_sampling: bool,
}
```

### Dispatch Extension

The current metalForge dispatch routes compute workloads:
`GPU > NPU > CPU` (by capability preference).

Field genomics adds a feedback loop:
`SEQ → GPU (basecall) → NPU (classify) → SEQ (adaptive sampling)`

This is not a linear pipeline — it is a closed loop where the NPU's
classification decision feeds back to the sequencer to influence what
data is generated next. metalForge needs to model this as a cyclic
dispatch graph, not just a linear chain.

---

## 6. Quality Gates

All new modules follow wetSpring's established quality standards:

| Gate | Requirement |
|------|-------------|
| `cargo fmt --check` | Clean |
| `cargo clippy --pedantic` | Zero warnings |
| `cargo doc --no-deps` | Zero warnings |
| `#![deny(unsafe_code)]` | Enforced |
| `#![deny(clippy::expect_used, clippy::unwrap_used)]` | Enforced |
| Named tolerance constants | All thresholds scientifically justified |
| hotSpring validation pattern | Hardcoded expected, explicit pass/fail, exit 0/1 |
| Three-tier control | CPU → GPU → metalForge for all new domains |
| Provenance header | Every validation binary documents its experiment |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|-----------|
| FAST5/POD5 parsing complexity | Medium | High | Start with POD5 (Arrow-based, pure Rust). Defer FAST5. |
| Basecalling accuracy gap | High | Medium | Delegate to Dorado initially. Build sovereign basecaller incrementally. |
| Flow cell cost per experiment | High | Medium | Maximize reads per flow cell. Use adaptive sampling to focus on targets. |
| MinKNOW API changes | Low | Medium | Abstract behind `io::minknow` trait. Version-pin API. |
| Nanopore read quality vs Illumina | Known | Low | Already validated: Anderson regime detection is robust to noise (Exp051 rare biosphere). |
| NestGate bandwidth for streaming | Medium | Medium | Content-address at chunk level, not file level. Stream processing via iterator. |
