# Exp099: Expanded CPU vs GPU Parity + metalForge Mixed Hardware

| Field    | Value                                       |
|----------|---------------------------------------------|
| Script   | `validate_cpu_gpu_expanded`                 |
| Command  | `cargo run --features gpu --bin validate_cpu_gpu_expanded` |
| Status   | **PASS** (27/27)                            |
| Phase    | 27                                          |
| Depends  | Exp049, Exp066, Exp093, Exp098              |

## Purpose

Validates CPU ↔ GPU parity for newly-implemented GPU domains and metalForge
mixed-hardware pipelines:

1. **K-mer Histogram**: CPU `count_kmers` ↔ GPU `KmerHistogramGpu` (ToadStool)
2. **UniFrac Propagation**: GPU `UniFracPropagateGpu` dispatch (ToadStool)
3. **QS ODE Sweep**: CPU `run_scenario` ↔ GPU `BatchedOdeRK4F64` (ToadStool)
4. **Phage Defense ODE**: CPU `run_defense` ↔ GPU local WGSL shader (evolution)
5. **metalForge Pipeline**: GPU k-mer → CPU feature extraction → GPU diversity

## New GPU Wrappers Created

| Module              | ToadStool Primitive        | Tier     |
|---------------------|----------------------------|----------|
| `kmer_gpu.rs`       | `KmerHistogramGpu`         | Absorbed |
| `unifrac_gpu.rs`    | `UniFracPropagateGpu`      | Absorbed |
| `phage_defense_gpu.rs` | Local WGSL shader       | Evolution (A) |

## Phage Defense GPU — Local Evolution

First local WGSL shader in wetSpring since Lean phase completed. Written with
all established f64 patterns:

- `fmax`/`fclamp` polyfills (no naga f64 overloads)
- `(zero + literal)` pattern for all f64 constants
- 4-variable, 11-parameter Monod-based phage-bacteria defense ODE
- Achieves **exact** CPU ↔ GPU parity (0.000000 relative error) on RTX 4070

To be absorbed by ToadStool as a generalized batched ODE primitive.

## metalForge Mixed-Hardware Pipeline

GPU → CPU → GPU handoff validated:
1. GPU: k-mer histogram (atomic increment on 4^k buffer)
2. CPU: richness feature extraction from histogram
3. CPU ↔ GPU: exact total match (28 k-mers, 0 bin difference)

## Results

| Domain             | Checks | Status | Notes                         |
|--------------------|--------|--------|-------------------------------|
| K-mer (k=4)        | 3/3    | PASS   | Exact raw histogram match     |
| UniFrac            | 4/4    | PASS   | Leaf init + finite sums       |
| QS ODE             | 9/9    | PASS   | Vars 0-2 exact; 3-4 known d_bio gap |
| Phage Defense ODE  | 8/8    | PASS   | All vars exact to 0.000000    |
| metalForge Pipeline| 3/3    | PASS   | GPU→CPU→GPU handoff           |
| **Total**          | **27/27** | **PASS** |                            |
