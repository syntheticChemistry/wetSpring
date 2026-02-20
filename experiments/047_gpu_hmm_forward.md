# Experiment 047: GPU HMM Batch Forward

**Date:** February 20, 2026
**Status:** COMPLETE — 13/13 PASS
**Track:** Cross-cutting (GPU)
**Binary:** `validate_gpu_hmm_forward` (requires `--features gpu`)

---

## Objective

Validate the local WGSL shader for GPU-accelerated HMM batch forward algorithm. This is a **Write → Absorb → Lean** candidate: implement the forward algorithm in WGSL, validate parity with CPU, then absorb into ToadStool as `HmmBatchForwardF64`.

## Method

1. **New local shader** — `hmm_forward_f64.wgsl` implements batch forward algorithm
2. **Thread model** — One thread per sequence, sequential over time steps
3. **NVVM workaround** — Use forced exp/log polyfill via `ShaderTemplate::for_driver_auto` to work around RTX 4070 NVVM driver bug (native f64 transcendentals fail to compile)
4. **Parity validation** — Compare GPU log-likelihood and alpha values against CPU `hmm::forward_batch`

## Results

### Section 1: 2-State HMM
| Metric | CPU | GPU | Status |
|--------|-----|-----|--------|
| Log-likelihood | finite | CPU ≈ GPU | PASS |
| Max alpha diff | — | 4.26e-14 | PASS |

### Section 2: 3-State HMM
| Metric | Expected | Result |
|--------|----------|--------|
| Log-likelihood parity | CPU ≈ GPU | PASS |
| Viterbi consistency | Confirmed | PASS |

### Section 3: Batch (64 sequences)
| Metric | Value | Status |
|--------|-------|--------|
| Sequences | 64 | PASS |
| max \|CPU−GPU\| | 2.81e-10 | PASS |

### Section 4: Forward-Backward Consistency
| Metric | Value | Status |
|--------|-------|--------|
| Max FB diff | 5.33e-14 | PASS |

## Key Findings

1. **Local WGSL shader achieves near-bit-exact parity** with CPU HMM forward algorithm
2. **Shader is a ToadStool absorption candidate** — `HmmBatchForwardF64` primitive
3. **NVVM workaround** — Force exp/log polyfill since RTX 4070 NVVM cannot compile native f64 transcendentals
4. **Forward-backward consistency** — Max diff 5.33e-14 validates numerical stability

## References

- Exp026: Liu 2014 HMM (CPU baseline)
- Exp037: PhyloHMM discordance
- Exp045: ToadStool bio absorption pattern

## Files Changed

| File | Purpose |
|------|---------|
| `barracuda/src/shaders/hmm_forward_f64.wgsl` | Local WGSL shader for HMM batch forward |
| `barracuda/src/bio/hmm_gpu.rs` | GPU dispatch + ShaderTemplate::for_driver_auto |
| `barracuda/src/bin/validate_gpu_hmm_forward.rs` | GPU HMM forward validator (13 checks) |

## Run

```bash
cargo run --bin validate_gpu_hmm_forward --features gpu
```
