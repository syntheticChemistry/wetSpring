# Exp382: Sovereign Rust Resequencing — Barrick 2009 Parity

| Field | Value |
|-------|-------|
| Status | IN PROGRESS |
| Binary | `validate_sovereign_resequencing` |
| Features | CPU: `(default)`, GPU: `--features gpu` |
| Paper | Barrick et al. *Nature* 461, 1243–1247 (2009) |
| Baseline | Exp381 breseq output (cached) |

## Purpose

Cross-tier parity proof: run the sovereign Rust resequencing pipeline
on the same Barrick 2009 FASTQ data that Exp381 processes with breseq,
and compare mutation calls position-by-position.

When built with `--features gpu`, the pipeline dispatches computationally
intensive steps to barraCuda GPU primitives while keeping seeding and I/O
on CPU — the same math, different substrate.

## Pipeline

```
FASTQ → FM-index seed (CPU)
      → Smith-Waterman extend (CPU | SmithWatermanGpu)
      → SAM
      → pileup (CPU | Tensor::scan GPU prefix sum)
      → variant caller (CPU | SnpCallingF64 GPU column-parallel)
      → output.gd
```

### Sovereign Modules + GPU Composition

| Module | Role | GPU Primitive (--features gpu) |
|--------|------|-------------------------------|
| `io::fasta` | Reference genome loading | — |
| `io::fastq` | Read input | — |
| `bio::ref_index` | FM-index for seeding | — (CPU only, no upstream equivalent) |
| `bio::read_mapper` | Seed-and-extend orchestration | `SmithWatermanGpu` — banded SW on GPU |
| `io::sam` | Alignment interchange | — |
| `bio::pileup` | Depth accumulation | `Tensor::scan` — inclusive prefix sum |
| `bio::variant_caller` | Mutation identification | `SnpCallingF64` — column-parallel SNP calling |

### Dispatch Model

The binary auto-detects GPU availability at startup:
- **GPU available (f64 native)**: full GPU dispatch for SW, pileup scan, and SNP calling
- **GPU available (no f64)**: GPU dispatch with df64 fallback where available
- **No GPU**: pure CPU fallback (identical math, zero GPU dependency)

Each GPU call has an inline CPU fallback — if any individual GPU dispatch
fails, that pair/column falls back to the CPU path transparently.

## Validation

| Module | Validates Against | Tolerance |
|--------|------------------|-----------|
| `io::fasta` | REL606.gbk (4,629,812 bp) | Exact sequence |
| `bio::ref_index` | bowtie2 seed positions | Exact k-mer locations |
| `bio::read_mapper` | breseq step 02 SAM | MAPQ within 5, position within 3bp |
| `io::sam` | samtools roundtrip | Field-exact |
| `bio::pileup` | samtools depth | Exact per-position |
| `bio::variant_caller` | breseq output.gd | Position-exact, type-match |

## Data

Uses cached Exp381 workspace on 4TB NVMe:
- Reference: `/mnt/4tb-work/.../barrick_2009/reference/REL606.fasta`
- FASTQs: `/mnt/4tb-work/.../barrick_2009/fastq/SRR03237*.fastq`
- breseq baseline: `/mnt/4tb-work/.../barrick_2009/breseq_output/*/output/output.gd`

## Run

```bash
# CPU-only
cargo run --bin validate_sovereign_resequencing

# GPU-accelerated (SmithWatermanGpu + SnpCallingF64 + Tensor::scan)
cargo run --features gpu --bin validate_sovereign_resequencing
```

## Results

Pending — will be populated after first run.

## Notes

- Initial run uses 10k read subsample per clone for development speed.
  Full-depth validation will follow once the pipeline is tuned.
- The FM-index is the one module with no barraCuda equivalent — SA-IS
  construction is inherently sequential and runs on CPU.
- GPU composition uses `#[cfg(feature = "gpu")]` gating — zero GPU code
  in the default build, zero runtime overhead when GPU is absent.
