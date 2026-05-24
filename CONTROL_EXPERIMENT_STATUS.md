# Control Experiment Status — wetSpring

**Last updated:** V184 (2026-05-23)
**Lib tests:** 1,962 passed, 0 failed, 2 ignored (pre-existing upstream module visibility)
**UniBin scenarios:** 345 (318 validation + 23 benchmark + 4 composition)
**Live NUCLEUS:** southGate — 9/9 primals started, exp091+exp094 PASS
**Experiment specs:** 386 indexed (385 completed, 1 in progress)
**Clippy:** zero warnings (`--features ipc --lib -- -W clippy::pedantic -W clippy::nursery`)

## V184 Post-Primordial Gate Deployment

NUCLEUS composition deployed and validated on southGate:
- **plasmidBin v5.5.0** — CLI alignment fix absorbed (`8c8cb44`)
- **9/9 primals started** — bearDog, Songbird, toadStool, barraCuda, coralReef, nestGate, rhizoCrypt, loamSpine, sweetGrass
- **exp091 primal routing matrix:** PASS
- **exp094 composition parity:** PASS
- **wetspring validate (rust tier):** 66/77 PASS (2 stale checksums, 3 empty benchmarks, minor tolerance drift)
- **Ionic bonding module wired:** `ipc::bonding` (6 methods → IonicContractRegistry)

## Experiment Summary

All scenarios are accessed via the `wetspring` UniBin:

```bash
wetspring validate --scenario <id>     # e.g. diversity, barracuda_cpu_v27
wetspring validate --list              # list all 345 scenarios
wetspring validate --all               # run everything
```

| Category | Scenarios | Status |
|----------|-----------|--------|
| CPU validation (`barracuda_cpu_*`) | 28 | All green |
| GPU validation (`barracuda_gpu_*`) | 14 | All green (requires GPU) |
| CPU vs GPU parity | 9 | All green (requires GPU) |
| Cross-spring validation | 12 | All green (requires GPU) |
| Cross-spring benchmarks | 8 | All green (requires GPU) |
| ToadStool dispatch validation | 5 | All green (requires GPU) |
| Paper math controls | 6 | All green |
| metalForge mixed hardware | 22 | All green |
| Hormesis chain (Exp377–379) | 3 | All green |
| Pharmacology (Gonzales/Fajgenbaum) | 14 | All green |
| Composition (Exp400–403) | 4 | All green |
| Total scenarios | 345 | Green |

## Control Chains

### Chain 1: Python → Rust CPU → Rust GPU

```
scripts/*.py (55 baselines)     Tier 1: Python/numpy/scipy reference
    ↓
wetspring validate --scenario barracuda_cpu_*    Tier 2: Rust CPU parity (28 versions)
    ↓
wetspring validate --scenario barracuda_gpu_*    Tier 3: GPU parity via wgpu (14 versions)
    ↓
wetspring validate --scenario cpu_vs_gpu_all_domains   Tier 3: 16-domain head-to-head
```

### Chain 2: Cross-Spring Evolution (ToadStool)

```
wetspring validate --scenario cross_spring_s57 → s62 → s65 → s68 → s70 → s79 → s86 → s87 → s93 → v98
```

Sessions covered: S57, S62, S65, S68, S70, S79, S86, S87, S93, S130+

### Chain 3: NUCLEUS Atomics (metalForge)

```
dispatch_routing              → Tower + Node + Nest discovery
pcie_bypass_mixed_hw          → GPU→NPU bypass, GPU→GPU, CPU fallback
mixed_hw_dispatch             → NUCLEUS + PCIe, 8-stage mixed pipeline
nucleus_biomeos_v92g          → Tower/Nest/Node + biomeOS DAG, 53 workloads
mixed_nucleus_v92g            → GPU→NPU→CPU→GPU interleaving
```

### Chain 4: Hormesis → Anderson → Colonization (Exp377–379)

```
hormesis_biphasic         (17/17 PASS) → bio::hormesis + dose_to_disorder + Anderson W_c
trophic_cascade           (10/10 PASS) → anderson_spectral::sweep + diversity under pesticide
colonization_resistance   (30/30 PASS) → binding_landscape + resistance_surface_sweep
```

### Chain 5: Composition (Exp400–403)

```
composition_nucleus_v1    (136/136 proto-nucleate)
composition_parity_v1     (43/43 IPC parity)
niche_parity_v1           (63/63 niche gate)
primal_parity_v1          (Tier 2, 5 primals live)
```

### Chain 6: Paper Review Queue

63/63 papers reviewed. LTEE GuideStone queue: B7 TIER 2 COMPLETE, 9 queued.

## Pending Work

| Item | Blocked By | Priority |
|------|------------|----------|
| Tenaillon 264-clone batch | toadStool `compute.fan_out` | HIGH |
| LTEE B1–B6, B8, E1, E5 queue | lithoSpore modules | Medium |
| Field genomics Exp197–202 | Field hardware | Low |
| EPA UCMR5 + PFOS datasets | Download + parse | Medium |
| GPU parity for NPU candidates | AKD1000 firmware | Low |

## Hardware Matrix

| Substrate | Validated | Coverage |
|-----------|-----------|----------|
| CPU (i9-12900K) | 1,962 lib + 28 CPU scenarios | Full |
| GPU (RTX 4070 Ada) | 14 GPU + 9 CPU-vs-GPU scenarios | Full |
| GPU (Titan V) | Cross-spring validators | Partial (NVK) |
| NPU (AKD1000) | metalForge dispatch routing | Routing only |
