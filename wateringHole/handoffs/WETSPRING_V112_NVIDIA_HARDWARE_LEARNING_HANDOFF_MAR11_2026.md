# wetSpring V112: NVIDIA Hardware Learning Prototype Handoff

**Date:** March 11, 2026
**From:** wetSpring V112
**To:** barraCuda, toadStool, coralReef, biomeOS teams
**Upstream pins:** barraCuda v0.3.5 `0649cd0`, toadStool S146 `751b3849`, coralReef Iter 33 `b783217`

---

## Summary

3 new experiments (Exp361-363) implement the **probe-calibrate-route-apply** hardware learning pattern, producing a machine-readable **capability profile** from GPU hardware discovery. This is the prototype for toadStool's planned `hw-learn` module.

| Exp | Name | Checks | Status |
|:---:|------|:------:|:------:|
| 361 | Nouveau RTX 4070 Diagnostic v1 | 15 | PASS |
| 362 | Hardware Learning Prototype v1 | 13 | PASS |
| 363 | Adaptive Dispatch from Profile v1 | 17 | PASS |

**Total: 45/45 PASS.** 363 experiments, 347 binaries, 9,819+ checks cumulative.

---

## Key Findings

### Dual-GPU Rig Discovery (Exp361)

The eastgate rig has two NVIDIA GPUs on different drivers:

| GPU | PCI Device | Driver | Firmware | Status |
|-----|-----------|--------|----------|--------|
| RTX 4070 (AD104, Ada) | 0x2786 | nvidia proprietary | GSP-only | wgpu/Vulkan working |
| Titan V (GV100, Volta) | 0x1d81 | nouveau | ACR/GR/NVDEC/SEC2, no GSP | VM_INIT OK, CHANNEL_ALLOC blocked |

**Nouveau dispatch on Titan V:**
- VM_INIT: **OK** — kernel 6.17.9 supports new UAPI
- CHANNEL_ALLOC: **EINVAL (22)** — Volta lacks PMU firmware
- All compute classes tested (VOLTA 0xC3C0, TURING 0xC5C0, AMPERE 0xC6C0): EINVAL
- Full NvDevice::open_path: FAILED (EINVAL from CHANNEL_ALLOC)
- GEM_NEW, VM_BIND, EXEC: not reached

**RTX 4070 on nouveau:** UNTESTED — nvidia proprietary driver active. This is the **highest-ROI next step**. Ada uses GSP-only firmware (present at `/lib/firmware/nvidia/ad104/gsp`), which should bypass the Volta PMU blocker.

### Firmware Inventory (Exp361)

| Architecture | Chip | Firmware Components | GSP? |
|-------------|------|---------------------|------|
| Ada Lovelace | ad102-ad107 | gsp only | Yes |
| Ampere | ga100-ga107 | acr, gr, gsp, nvdec, sec2 | Yes |
| Volta | gv100 | acr, gr, nvdec, sec2 | No |

Ada's GSP-only firmware is the key differentiator. No PMU/ACR/GR/SEC2 needed — simpler dispatch path.

### HardwareCalibration (Exp362)

RTX 4070 via wgpu/Vulkan:

| Tier | Compiles | Dispatches | Transcendentals | Arith Only | Safe |
|------|:--------:|:----------:|:---------------:|:----------:|:----:|
| F32 | true | true | true | false | **true** |
| DF64 | true | true | false | true | false |
| F64 | true | true | false | true | false |
| F64Precise | true | true | false | true | false |

All non-F32 tiers: compile and dispatch work, but transcendentals are unsafe (NVVM poisoning risk). Arithmetic-only DF64 shaders (add, mul, FMA — no exp/log/sqrt) are safe.

### PrecisionBrain 12-Domain Routing (Exp362)

All 12 PhysicsDomain variants route to **F32** on RTX 4070 (nvidia proprietary):

- LatticeQcd, GradientFlow, Dielectric, KineticFluid, Eigensolve, MolecularDynamics, NuclearEos, PopulationPk, Bioinformatics, Hydrology, Statistics, General → **F32**

This is correct: F32 is the only fully safe tier. Bio-relevant domains (Bioinformatics, Statistics, Hydrology) run at F32 screening precision. F64 precision requires CPU fallback or sovereign dispatch (coralReef bypass).

### Bio Workload Thresholds (Exp362)

| Workload | CPU/GPU Crossover N | GPU Advantage at 10K | Max Pairwise N (12GB) |
|----------|:-------------------:|:--------------------:|:---------------------:|
| Shannon diversity | ~50K | 46× | 40,132 |
| Bray-Curtis pairwise | ~25K | 52× | 40,132 |
| ODE integration | ~5K | 58× | 40,132 |
| HMM forward | ~10K | 57× | 40,132 |
| Anderson eigenvalue | ~6K | 58× | 40,132 |

Below the crossover N, CPU is faster due to dispatch overhead (~100μs).

### Adaptive Dispatch (Exp363)

Given the capability profile from Exp362, Exp363 demonstrates correct adaptation:

| Workload Category | Decision | Reason |
|-------------------|----------|--------|
| F32 bio (Shannon, Simpson) | **EXECUTED** | Always safe |
| DF64 arithmetic (Anderson QL) | **EXECUTED** | f64 hardware present |
| F64 transcendentals (erfc, log1p) | **SKIPPED** | NVVM risk detected |
| Large-N pairwise (Bray-Curtis 1K) | **EXECUTED** | 12GB VRAM sufficient |
| Sovereign dispatch | **SKIPPED** | Not available |

VRAM scaling recommendation:
- N ≤ 1K: CPU (dispatch overhead dominates)
- 1K < N ≤ 40K: GPU (within single-pass VRAM)
- N > 40K: tiled/streaming (exceeds VRAM)

---

## Absorption Targets

### For toadStool: hw-learn Module

The capability profile JSON format (`output/hardware_capability_profile.json`) is the absorption target for toadStool's planned `hw-learn` module. Key fields:

```json
{
  "schema_version": "1.0",
  "adapter_name": "NVIDIA GeForce RTX 4070",
  "has_any_f64": true,
  "df64_safe": true,
  "nvvm_transcendental_risk": true,
  "precision_tiers": [...],
  "domain_routing": {...},
  "bio_workload_thresholds": {...},
  "firmware_inventory": {...},
  "sovereign_dispatch": {...}
}
```

The `hw-learn` module pattern: **observer → distiller → knowledge → applicator → brain_ext** maps to:
- observer: `HardwareCalibration::from_device()` + firmware scan
- distiller: `PrecisionBrain::from_device()` + domain routing
- knowledge: capability profile JSON
- applicator: adaptive dispatch logic (Exp363 pattern)
- brain_ext: `PrecisionBrain` extensions for new domains

### For coralReef: Nouveau RTX 4070 Diagnostic

The critical next step is testing `diag_ioctl` on the RTX 4070 with nouveau loaded (currently nvidia proprietary). Ada's GSP-only firmware should allow CHANNEL_ALLOC to succeed where Volta's missing PMU blocks it.

Steps to test:
1. Blacklist nvidia, modprobe nouveau
2. Run `cargo run --example diag_ioctl -p coral-driver`
3. If CHANNEL_ALLOC succeeds: GEM_NEW + VM_BIND + EXEC pipeline unlocks
4. If succeeds: every GSP-equipped NVIDIA GPU (Turing+) unlocks

### For barraCuda: Profile-Driven Routing

The capability profile enables profile-driven routing:
- Read `hardware_capability_profile.json` at startup
- Select workloads based on tier safety (not just tier availability)
- VRAM-aware problem sizing
- FMA policy per domain

### For biomeOS: compute.hardware.* Capabilities

The capability profile maps directly to biomeOS NUCLEUS capabilities:
- `compute.hardware.f32_safe: true`
- `compute.hardware.df64_arith: true`
- `compute.hardware.f64_transcendentals: false`
- `compute.hardware.vram_gb: 12`
- `compute.hardware.sovereign: false`

---

## Artifacts

| File | Size | Purpose |
|------|------|---------|
| `output/hardware_capability_profile.json` | 4.3KB | Machine-readable dispatch recipe |
| `output/dispatch_pipeline_status.json` | 2.0KB | petalTongue dispatch dashboard |

---

## Test Suite

1,294 lib tests pass, 3 known GPU f32 parity failures (pre-existing: hamming_gpu, jaccard_gpu, spatial_payoff_gpu). No regressions from V112 changes.

---

## Next Steps

1. **Nouveau on RTX 4070** — switch driver, run diag_ioctl, test Ada GSP dispatch
2. **hw-learn module prototype** — toadStool absorbs capability profile format
3. **Profile-driven PrecisionBrain** — barraCuda reads profile at startup
4. **biomeOS compute.hardware capabilities** — NUCLEUS capability advertisement
5. **Sovereign bypass benchmark** — once coralReef dispatch works on RTX 4070
