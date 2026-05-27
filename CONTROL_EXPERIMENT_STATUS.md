# Control Experiment Status — wetSpring

**Last updated:** V188 (2026-05-26)
**Lib tests:** 1,962 passed, 0 failed, 2 ignored (pre-existing upstream module visibility)
**UniBin scenarios:** 345 (318 validation + 23 benchmark + 4 composition)
**Live NUCLEUS:** southGate — 8/13 health-responding (Wave 53 hardened, Songbird stable)
**Experiment specs:** 386 indexed (385 completed, 1 in progress)
**Clippy:** zero warnings (`--features ipc --lib -- -W clippy::pedantic -W clippy::nursery`)
**Dispatch methods:** 45 (was 38; +lifecycle.status, +6 bonding.*)
**Binary discovery:** plasmidBin-only (v2026.05.27 release, Wave 53)

## V188 Wave 55 — southGate Redeploy + PG Verification

Wave 53 hardened binaries redeployed on southGate:
- **Songbird:** Stable. Socket hardening fix confirmed. No more stale socket crashes.
- **loamSpine:** Alive. Tokio runtime-in-runtime panic FIXED.
- **NUCLEUS:** 8/13 health-responding (biomeOS, BearDog, Songbird, coralReef, NestGate, loamSpine, sweetGrass, rhizoCrypt). ToadStool socket but no health response. barraCuda crashed. Squirrel/petalTongue/skunkBat socket issues.
- **PG-02 VERIFIED:** spine.create + braid.create live IPC roundtrip. Provenance trio operational.
- **PG-04 VERIFIED:** NestGate alive with 66 methods. BTSP-auth gates storage correctly.
- **Mesh seeded:** SONGBIRD_PEERS=192.168.1.144:7700, mesh.init called. 0 peers (eastGate offline).
- **All PG gaps closed:** 22/22 resolved.

## V187 Wave 50 Post-Primordial + Covalent HPC

Wave 49 post-primordial compliance + Wave 50 covalent readiness:
- **Post-primordial:** Zero `target/release/` for primal binaries. Shared `primal_binary` module.
- **NUCLEUS:** 7/13 health-responding on southGate (Songbird, ToadStool, barraCuda, coralReef, NestGate, sweetGrass, petalTongue). loamSpine Tokio panic, BearDog/Squirrel/biomeOS socket timeout.
- **Cross-subnet:** southGate (192.168.4.29) → eastGate (192.168.1.144:7700) reachable at 4ms. No TURN relay needed.
- **Federation:** Songbird TCP on 0.0.0.0:7700 (LAN-reachable, not loopback).
- **Covalent targets:** breseq pipeline validation, WS-2 nest.sync, WS-11 re-measurement via remote compute.

## V185 Wave 48 Covalent Mesh

Deployment conformance + debt resolution:
- **`health.liveness`:** Returns `{"status":"alive"}` per DEPLOYMENT_BEHAVIOR_STANDARD v1.0
- **`lifecycle.status`:** New handler — primal name, version, status, uptime_s
- **`bonding.*` dispatched:** 6 handlers wired into JSON-RPC dispatch (propose, accept, reject, status, terminate, list)
- **`wetspring serve`:** `--socket`, `--port`, `--family-id` flags + `server` alias
- **Songbird TCP federation:** Port 7700 confirmed, `discovery.peers` responding
- **Cell deployment:** `wetspring_cell.toml` validated (13 nodes, domain overlay order 12)
- **primalSpring v0.9.28 absorbed:** 52 scenarios, 458/458 methods (100% coverage), 787 tests
- **22 domains, 50 niche capabilities, 59 consumed capabilities**

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

| Substrate | Gate | Validated | Coverage |
|-----------|------|-----------|----------|
| CPU (5800X3D 8c) | southGate | 1,962 lib + 28 CPU scenarios | Full |
| GPU (RTX 4060) | southGate | 14 GPU + 9 CPU-vs-GPU scenarios | Full |
| CPU (i9-12900K) | eastGate | Lib tests, 28 CPU scenarios | Full |
| GPU (RTX 4070 Ada) | eastGate | 14 GPU + 9 CPU-vs-GPU | Full |
| GPU (Titan V) | biomeGate | Cross-spring validators | Partial (NVK) |
| NPU (AKD1000) | ironGate | metalForge dispatch routing | Routing only |
