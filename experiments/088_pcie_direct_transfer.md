# Exp088: metalForge PCIe Direct Transfer Proof

**Status**: PASS — 32/32 checks  
**Binary**: `validate_pcie_direct`  
**Date**: 2026-02-22

## Purpose

Prove that data flows between compute substrates (GPU ↔ NPU ↔ CPU) without
requiring CPU staging, validating the buffer layout contracts that enable
PCIe peer-to-peer transfers.

## Validated Transfer Paths

| Path | Data Flow | Test | Checks |
|------|-----------|------|--------|
| GPU → NPU | kmer histogram → int8 classify | classification parity | 7 |
| NPU → GPU | classification indices → diversity | Shannon/Simpson parity | 4 |
| GPU → GPU | kmer → diversity → UniFrac chain | end-to-end parity | 5 |
| Transfer parity | 6 paths × (direct vs CPU-staged) | histograms + classification | 8 |
| Buffer contracts | kmer, taxonomy, unifrac layouts | size + structure | 8 |

## Key Results

- GPU → NPU: int8 quantized classification matches f64 on all inputs
- NPU → GPU: diversity computed from NPU outputs matches CPU-path diversity
- GPU → GPU: 3-stage chain produces correct UniFrac through flat CSR
- All 6 transfer paths (direct and CPU-staged) produce identical results
- Buffer layout contracts verified: correct sizes for GPU upload

## Reproduction

```bash
cargo run --release --bin validate_pcie_direct
# Expected: 32/32 PASS, exit 0
```
