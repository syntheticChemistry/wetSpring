# metalForge — Hardware Exploration for Life Science Compute

**Parent**: ecoPrimals/wetSpring
**Purpose**: Characterize available hardware substrates and guide Rust
implementations for optimal absorption by ToadStool/BarraCUDA.

---

## Philosophy

wetSpring validates that life science and analytical chemistry algorithms
can be faithfully ported from Python to Rust CPU, then promoted to GPU via
ToadStool/BarraCUDA. metalForge characterizes the hardware those algorithms
run on — what it can do, what the vendor SDK hides, and how to shape our
Rust implementations so they absorb smoothly into ToadStool's multi-substrate
dispatch system.

Following hotSpring's precedent: vendor SDKs present a limited view of
hardware capabilities. Going lower finds what they don't advertise.
hotSpring proved this with NVIDIA f64 (1:2 via wgpu vs CUDA's 1:32
artificial limit) and AKD1000 NPU (direct weight injection bypassing
Keras/QuantizeML/CNN2SNN). wetSpring inherits those findings and extends
them to life science workloads.

---

## Local Hardware Inventory

| Substrate | Device | PCIe Slot | Key Spec | Status |
|-----------|--------|-----------|----------|--------|
| **GPU (primary)** | NVIDIA RTX 4070 | `01:00.0` | 12GB GDDR6X, Ada, f64 1:2 via wgpu | Active — ToadStool backend |
| **GPU (secondary)** | NVIDIA Titan V | `05:00.0` | 12GB HBM2, Volta GV100, NVK | Validated identical results |
| **NPU** | BrainChip AKD1000 | `08:00.0` | 80 NPs, 8MB SRAM, ~30mW | Driver loaded, `/dev/akida0` |
| **CPU** | Intel i9-12900K | — | 16C/24T, 30MB L3, P+E cores | Reference substrate |

### Future Targets

| Substrate | Target | Why |
|-----------|--------|-----|
| AMD RDNA3/4 GPU | RX 7900 XTX | 96MB Infinity Cache — SW/HMM matrices cache-resident |
| Intel Arc GPU | A770 | Xe HPG + XMX, alternative wgpu backend |

---

## Directory Structure

```
metalForge/
├── README.md              ← this file
├── PRIMITIVE_MAP.md       ← Rust module ↔ ToadStool primitive mapping
├── ABSORPTION_STRATEGY.md ← Write → Absorb → Lean methodology
├── PCIE_TOPOLOGY.md       ← PCIe topology and device mapping
├── forge/                 ← Rust crate: wetspring-forge v0.2.0 (discovery + dispatch)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs         ← crate root
│   │   ├── substrate.rs   ← Substrate, Capability, Identity types
│   │   ├── probe.rs       ← GPU (wgpu), CPU (/proc), NPU (/dev/akida*)
│   │   ├── inventory.rs   ← unified discovery
│   │   ├── dispatch.rs    ← capability-based workload routing
│   │   ├── streaming.rs   ← multi-stage GPU pipeline analysis (v0.2.0)
│   │   └── bridge.rs      ← forge ↔ barracuda device bridge
│   ├── examples/
│   │   └── inventory.rs   ← discover + print + dispatch demo
│   └── src/bin/
│       └── validate_dispatch_routing.rs ← Exp080 validation
├── gpu/
│   ├── nvidia/
│   │   └── HARDWARE.md    ← RTX 4070 + Titan V for life science workloads
│   └── amd/
│       └── HARDWARE.md    ← future: RDNA cache advantage for bioinformatics
├── npu/
│   └── akida/
│       └── HARDWARE.md    ← AKD1000 for taxonomy/anomaly detection
├── cpu/
│   └── HARDWARE.md        ← i9-12900K characterization, P-core vs E-core
└── benchmarks/
    ├── CROSS_SYSTEM_STATUS.md ← algorithm × substrate matrix
    ├── cpu_baseline.json      ← CPU performance baselines
    └── profile_gpu_candidates.py ← GPU candidate profiling
```

### Forge Crate (v0.2.0)

The `forge/` directory is a standalone Rust crate (`wetspring-forge` v0.2.0)
that discovers compute substrates at runtime and routes workloads by capability.
It follows hotSpring's `metalForge/forge/` pattern — same substrate types,
same capability-based dispatch, same wgpu GPU probing — adapted for life
science workloads. **32 unit tests**, clippy clean, `forbid(unsafe_code)`.

**v0.2.0 additions:**
- `streaming` module — multi-stage GPU pipeline analysis
- `CpuCompute` capability variant
- `StreamingSession` for pipeline topology analysis

**Absorption seam:** when ToadStool absorbs forge, the `bridge` module
becomes the integration point. `substrate_from_device` wraps an existing
barracuda `WgpuDevice` as a forge `Substrate`; `create_device` goes the
other direction. The streaming module's `StreamingSession` maps to
ToadStool's unidirectional pipeline model.

```bash
cd metalForge/forge
cargo test          # 32 tests
cargo run --example inventory  # discover + dispatch demo
```

---

## wetSpring Workload → Hardware Mapping

The key insight: different life science workloads have different hardware
affinities. metalForge maps each validated algorithm to its optimal substrate.

### GPU-Optimal Workloads (batch-parallel, f64)

| Algorithm | Module | Why GPU | ToadStool Primitive |
|-----------|--------|---------|---------------------|
| Smith-Waterman alignment | `bio::alignment` | Anti-diagonal parallelism, O(mn) independent cells | `smith_waterman_f64.wgsl` |
| Felsenstein pruning | `bio::felsenstein` | Site-parallel: each alignment column independent | `felsenstein_f64.wgsl` |
| Phylogenetic placement | `bio::placement` | Edge-parallel: each candidate insertion independent | `placement_f64.wgsl` |
| Bootstrap resampling | `bio::bootstrap` | Replicate-parallel: each bootstrap is independent | `batch_likelihood.wgsl` |
| ODE parameter sweeps | `bio::ode` + domain models | Parameter-parallel: each (cost, efficiency) pair | `rk_stage` (ToadStool) |
| HMM forward/backward | `bio::hmm` | Observation-parallel, state-parallel per step | `logsumexp_wgsl` (ToadStool) |
| Spectral matching | `bio::spectral_match` | Pairwise cosine → GEMM | `GemmF64` (ToadStool) |
| Diversity metrics | `bio::diversity` | Map-reduce over abundance vectors | `FusedMapReduceF64` (ToadStool) |

### NPU-Optimal Workloads (low-power inference)

| Algorithm | Module | Why NPU | Key Constraint |
|-----------|--------|---------|----------------|
| Taxonomy classification | `bio::taxonomy` | Pre-trained Naive Bayes → FC inference | int8 quantization |
| Anomaly detection (pond crash) | future | Time-series classification | Event-based, streaming |
| PFAS presence/absence | `bio::decision_tree` | Binary classification at field sensor | Ultra-low power |

### CPU-Optimal Workloads (branching, sequential)

| Algorithm | Module | Why CPU | Notes |
|-----------|--------|---------|-------|
| Newick parsing | `bio::unifrac` | Tree-structured, recursive | Keep CPU-only |
| DADA2 denoising | `bio::dada2` | Complex branching, state-dependent | GPU E-step helps |
| Chimera detection | `bio::chimera` | K-mer hash + voting | Hash-heavy |
| File I/O (FASTQ/mzML) | `io::*` | Streaming parse, branching | Zero-copy already |

---

## Evolution Strategy

wetSpring follows hotSpring's pattern for ToadStool absorption:

```
1. Validate in Rust CPU (barracuda/)          ← DONE: 41 modules, 730 tests, 96.21% coverage, 83 experiments, 1,835 checks. 38% dispatch overhead reduction via pipeline caching (Exp068).
2. Characterize hardware (metalForge/)         ← THIS DIRECTORY
3. Write Rust in GPU-friendly patterns         ← 8 absorbed by ToadStool, 4 local WGSL (ODE, kmer, unifrac, taxonomy)
4. ToadStool absorbs as shared primitives      ← unidirectional handoff via archive/handoffs/
5. wetSpring consumes ToadStool primitives     ← 15 consumed, pipeline closure
```

### What "GPU-Friendly Patterns" Means for Life Science

From hotSpring's experience, the patterns that absorb best into ToadStool:

1. **Flat arrays, not trees**: Linearize tree structures before dispatch
   (e.g., Felsenstein: pre-order traversal → flat partial arrays)
2. **Batched operations**: Group independent instances for dispatch
   (e.g., 1000 bootstrap replicates → single dispatch)
3. **No allocation in hot path**: Pre-allocate all buffers, reuse across calls
4. **Deterministic math**: Same inputs → same outputs (bit-exact for f64)
5. **Documented tolerances**: When CPU≠GPU (f64 transcendentals), justify ε

These patterns are already present in our validated Rust modules. metalForge
documents how each pattern maps to specific hardware capabilities.

---

## Relationship to hotSpring metalForge

| Aspect | hotSpring metalForge | wetSpring metalForge |
|--------|---------------------|---------------------|
| Domain | Computational physics (MD, HFB, lattice QCD) | Life science (metagenomics, phylogenetics, PFAS) |
| GPU workloads | Force kernels, eigensolve, HMC | SW alignment, Felsenstein, HMM, spectral match |
| NPU workloads | Phase classification (ESN→lattice QCD) | Taxonomy, anomaly detection, PFAS screening |
| Shared finding | f64 at 1:2 via wgpu, NPU direct weight injection | Inherited from hotSpring |
| New contribution | Lattice QCD phase structure on consumer GPU | Phylogenetic likelihood on consumer GPU |

Both metalForges feed into the same ToadStool dispatch system. Findings
from either inform the other's hardware utilization strategy.

---

## Current Status (Feb 22, 2026)

### CPU: 25 Domains Validated (Exp001–063)
All 25 algorithmic domains proven correct in pure Rust CPU:
- **205/205 CPU parity checks** across v1 (21) + v2 (18) + v3 (45) + v4 (44) + v5 (29) + v6 (48)
- **1,291 total CPU checks** across 50 CPU validation binaries
- **5 Track 1c domains**: ANI, SNP, dN/dS, molecular clock, pangenomics
- **2 ML ensemble domains**: Random Forest, Gradient Boosting Machine
- **22.5× overall speedup** over Python (Exp059, peak 625× for SW)

### GPU: 451 Checks Passing (Exp044–076)
GPU math portability proven across all promoted domains:
- **451/451 GPU checks** across 18 validation binaries
- **Absorbed** (Lean): SW, Gillespie, DT, Felsenstein, GEMM, diversity, PCoA, spectral
- **Absorbed** (Lean, Feb 22): HMM, DADA2, quality, ANI, SNP, dN/dS, pangenome, RF
- **Local WGSL** (4 shaders): ODE, kmer, unifrac, taxonomy (ODE blocked: upstream `compile_shader` needs `compile_shader_f64`)
- **Composed**: Bootstrap (15/15), placement (15/15), bifurcation eigenvalues (5/5)
- **GPU Parity v1** (Exp064): 8 domains consolidated in single binary
- **metalForge Full** (Exp065): CPU↔GPU substrate-independence for full portfolio

### Track 1c GPU Promotion — COMPLETE (Exp058)
All 4 GPU-eligible Track 1c modules promoted with local WGSL shaders:
- **ANI**: `ani_batch_f64.wgsl` — 7/7 GPU checks
- **SNP**: `snp_calling_f64.wgsl` — 5/5 GPU checks
- **dN/dS**: `dnds_batch_f64.wgsl` — 9/9 GPU checks, `log()` polyfill
- **Pangenome**: `pangenome_classify.wgsl` — 6/6 GPU checks
- **Molecular clock**: CPU-only (sequential tree traversal, too small for GPU)

### ML Ensembles — GPU-Promoted (Exp063)
- **Random Forest**: `rf_batch_inference.wgsl` — 13/13 GPU checks, SoA layout
- **GBM**: CPU-only (sequential boosting across rounds)

### Cross-System Vision
```
CPU: 1,291 checks (25 domains, reference ground truth)
GPU: 451 checks (23 ToadStool primitives + 4 local WGSL + 6 composed + consolidated)
NPU: Characterized (AKD1000, taxonomy/anomaly/RF inference at ~30mW)

Proven: substrate-independence across 8 domains (Exp065 — metalForge Full)
        CPU or GPU → same answer → router can dispatch to cheapest substrate

Pipeline: FASTQ parse (CPU) → DADA2 (CPU+GPU) → Diversity (GPU) → Taxonomy (NPU)
         → ANI/SNP (GPU batch) → Pangenome (GPU reduce) → dN/dS (GPU codon batch)
         → RF/GBM (GPU/CPU) → metalForge substrate routing
```

Full mapping: `PRIMITIVE_MAP.md` | Absorption tracking: `../barracuda/EVOLUTION_READINESS.md`

---

## Write → Absorb → Lean (hotSpring Pattern)

Following hotSpring's proven methodology for ToadStool absorption:

### Phase A: Write Locally
Implement the algorithm in Rust with GPU-friendly patterns (flat arrays,
batch APIs, `#[repr(C)]` structs, deterministic math). Write the WGSL shader
locally with `include_str!` or inline `pub const`. Validate against CPU
baseline with documented tolerances.

### Phase B: Hand Off
Submit to ToadStool via unidirectional handoff: shader source, binding layout,
dispatch geometry, CPU reference values, tolerance rationale. ToadStool team
absorbs the shader into `ops::bio::` (or other domain).

### Phase C: Lean on Upstream
Once ToadStool absorbs, rewire wetSpring to consume the upstream primitive.
Delete local shader copy. Use the crate-level re-export
(e.g., `barracuda::SmithWatermanGpu`). Local extensions that still need work
remain in wetSpring until absorption.

### Current Status (Feb 22)

| Phase | Count | Items |
|-------|:-----:|-------|
| **Absorbed** (Lean) | 11 modules | SW, Gillespie, DT, Felsenstein, GEMM, diversity, PCoA, spectral, stats, EIC, rarefaction |
| **Absorbed** (Lean, Feb 22) | 8 shaders | HMM, DADA2, QF, ANI, SNP, dN/dS, pangenome, RF |
| **Local WGSL** (Write) | 4 shaders | ODE, kmer, unifrac, taxonomy |
| **CPU math** (barracuda overlap) | 4 functions | erf, ln_gamma, regularized_gamma, integrate_peak |
| **Blocked** | 3 modules | kmer (lock-free hash), UniFrac (tree traversal), taxonomy (NPU) |

Active handoff: `../../wateringHole/handoffs/WETSPRING_TOADSTOOL_TIER_A_SHADERS_FEB21_2026.md`

---

## BEYOND_SDK Findings

Following hotSpring's approach of testing SDK assumptions against silicon:

### GPU: f64 is 1:2, not 1:32
NVIDIA's CUDA SDK artificially limits f64 throughput on consumer GPUs to 1:64.
Via wgpu/Vulkan, the RTX 4070 delivers f64 at 1:2 of f32 peak (measured by
hotSpring, confirmed by wetSpring on diversity + spectral workloads).

### GPU: Gillespie SSA NVVM Limit
Complex f64 WGSL shaders (Gillespie SSA with inline xoshiro PRNG, 6 reaction
channels, Exp-distribution sampling) exceed NVVM's shader complexity limit on
the RTX 4070 (Ada Lovelace, driver 580.x). The same shader should compile on
Titan V (Volta, simpler instruction set). This is a **driver** limit, not a
**hardware** limit.

### NPU: Direct Weight Injection
AKD1000 accepts raw weights via `/dev/akida0` without going through
Keras→QuantizeML→CNN2SNN. Bypasses the SDK's quantization pipeline
entirely (hotSpring finding, inherited).

### CPU: P-core vs E-core Affinity
Intel i9-12900K: P-cores (8, SMT) are 2-3x faster than E-cores (8, no SMT)
for single-threaded ODE integration. E-cores saturate first under `rayon`
parallel workloads. Pin latency-sensitive stages (DADA2 error model,
Felsenstein per-site) to P-cores.

### GPU: NVVM f64 Transcendental Limitation (RTX 4070)
**CRITICAL FINDING (Feb 20)**: NVIDIA RTX 4070 (Ada Lovelace) NVVM compiler
cannot compile native f64 `exp()`, `log()`, or `pow()` in WGSL shaders.
Driver profile reports `needs_f64_exp_log_workaround() = false`, so
ToadStool's automatic polyfill doesn't trigger.

**Workaround**: Force polyfill via `ShaderTemplate::for_driver_auto(source, true)`.
This injects `exp_f64`, `log_f64`, `pow_f64` as pure-arithmetic WGSL
polyfills that bypass the NVVM compiler entirely.

**Impact**: All shaders using f64 transcendentals must be compiled with
forced polyfill on Ada Lovelace GPUs. Affects: HMM forward, ODE sweep,
Gillespie SSA. Does NOT affect: Felsenstein, GEMM, eigenvalues, Smith-Waterman.

### GPU: naga f64 Literal Promotion
naga (wgpu 22) does not auto-promote `f32` literals to `f64` in builtin
calls. `max(x, 0.0)` where `x: f64` fails validation. Fix: use `f64(0.0)`.
Also: `enable f64;` directive is not supported by naga — shaders must omit
it and rely on `compile_shader_f64` or `ShaderTemplate` preprocessing.

---

## Completed GPU Wiring (Feb 20)

| Domain | Method | Status |
|--------|--------|--------|
| Felsenstein pruning | `FelsensteinGpu` (ToadStool) | Exp046: 15/15 |
| Bootstrap resampling | Compose FelsensteinGpu per replicate | Exp046: 15/15 |
| Phylogenetic placement | Compose FelsensteinGpu per edge | Exp046: 15/15 |
| HMM batch forward | Local WGSL shader (forced polyfill) | Exp047: 13/13 |
| ODE parameter sweep | Local WGSL shader (pow_f64 polyfill) | Exp049: 7/7 |
| Bifurcation eigenvalues | `BatchedEighGpu` (ToadStool) | Exp050: 5/5 |

## ToadStool Absorption Candidates

1. **`hmm_forward_f64.wgsl`** — batch HMM forward in log-space. 13/13 checks.
2. **`batched_qs_ode_rk4_f64.wgsl`** — local fix: upstream `batched_ode_rk4.rs:209`
   uses `compile_shader` not `compile_shader_f64`; wetSpring uses
   `compile_shader_f64()` (ToadStool removed `enable f64;` session 39).
3. **NVVM driver profile fix** — `needs_f64_exp_log_workaround()` should
   return `true` for Ada Lovelace (RTX 4070/4080/4090).

---

## Absorption Engineering (Phase 17)

wetSpring is now shaping all validated Rust implementations for
ToadStool/BarraCUDA team absorption. Following hotSpring's pattern:

### Ready for Absorption

| Component | Status | Absorption Path |
|-----------|--------|-----------------|
| 8 WGSL shaders (absorbed Feb 22) | Lean — delegating to ToadStool | `ops::bio::*` |
| 4 WGSL shaders (ODE, kmer, unifrac, taxonomy) | Local — ODE blocked: upstream uses `compile_shader` not `compile_shader_f64` | `ops::bio::*` |
| `bio::special` (erf, ln_gamma, regularized_gamma) | Consolidated, `mul_add`-optimized | `barracuda::math` feature |
| `bio::eic::integrate_peak` | Validated against Python | `barracuda::numerical::trapz` |
| 32 tolerance constants | Hierarchy-tested | `barracuda::tolerances` cross-Spring standard |
| SoA/batch patterns | GPU-friendly, `#[repr(C)]` | Direct shader translation |

### Blocked

| Blocker | Impact | Resolution |
|---------|--------|------------|
| `barracuda::math` feature gate | 4 CPU math functions can't lean upstream | Propose `[features] math = []` |
| NVVM driver profile bug | f64 transcendentals crash on Ada Lovelace | Fix `needs_f64_exp_log_workaround()` |
| Lock-free hash GPU primitive | kmer module can't promote | ToadStool new primitive |
| Tree traversal GPU primitive | UniFrac can't promote | ToadStool new primitive |

### Next Steps

1. **ToadStool absorption**: 9 Tier A shaders submitted via handoff doc.
   Priority: HMM forward (compose `logsumexp_wgsl`), ODE sweep (fix
   upstream `BatchedOdeRK4F64`), RF inference (extend `TreeInferenceGpu`
   to ensemble/SoA).

2. **barracuda `math` feature**: Propose CPU-only feature gate for
   `numerical`, `special`, `stats` modules. Unblocks switching 4 local
   math functions to upstream primitives.

3. **GPU**: Scale ODE sweep to B=10,000 with ToadStool unidirectional
   streaming. Current: 64 batches in 2s. Target: 10,000 in <1s.

4. **NPU**: Train taxonomy FC model from Naive Bayes weights, quantize int8,
   validate classification accuracy NPU vs CPU.

5. **CPU**: Profile P-core vs E-core for ODE-heavy workloads. Consider
   P-core pinning for latency-sensitive stages (DADA2, parsing).

6. **GPU**: Remaining P3 items (K-mer counting, UniFrac) require new
   ToadStool primitives — wait for upstream absorption cycle.
