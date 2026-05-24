# wetSpring Gate Deployment Status — southGate

**Date:** 2026-05-23
**From:** wetSpring (southGate)
**To:** primalSpring (coordination), all delta springs
**Gate:** southGate
**Co-resident:** healthSpring
**Hardware:** AMD Ryzen 7 5800X3D 8-Core, 128GB DDR4
**Priority:** Gate deployment status — post-primordial covalent validation

---

## Gate Assignment Confirmed

| Field | Value |
|-------|-------|
| Gate | **southGate** |
| Springs | wetSpring, healthSpring |
| Secondary | **strandGate** (Dual EPYC 64-core, 256GB ECC, RTX 3090 + RX 6950 XT) |
| primalSpring | v0.9.27 (458 methods, 49 scenarios) |
| plasmidBin | v5.5.0 (13/13 primals) |
| wetSpring | V184 (345 scenarios, 49 niche, 52 consumed) |

---

## Proto-Nucleate Composition

Per `primalSpring/graphs/downstream/downstream_manifest.toml`, wetSpring's
life science NUCLEUS requires:

| Fragment | Primals |
|----------|---------|
| tower_atomic | bearDog, Songbird |
| node_atomic | barraCuda, coralReef, toadStool |
| nest_atomic | NestGate |
| meta_tier | Squirrel, petalTongue |

**Particle profile:** balanced
**Total primals needed:** 8 (beardog, songbird, coralreef, toadstool, barracuda, nestgate, squirrel, petaltongue)

### Validation Capabilities Required (IPC)

```
tensor.matmul, tensor.create, stats.mean, stats.std_dev, stats.variance,
stats.correlation, linalg.solve, linalg.eigenvalues, spectral.fft,
spectral.power_spectrum, compute.dispatch, storage.store, storage.retrieve,
inference.complete, crypto.hash
```

15 methods across 7 capability domains.

---

## Deployment Status

### What is ready

| Component | Status | Notes |
|-----------|--------|-------|
| wetSpring UniBin | READY | V184, 345 scenarios, compiles clean |
| Registry sync | DONE | 458 methods, zero drift |
| Ionic bonding | WIRED | `ipc::bonding` → IonicContractRegistry (6 methods) |
| Niche capabilities | 49 | Including bonding.* |
| Proto-nucleate graph | IDENTIFIED | `downstream_manifest.toml` wetspring entry |
| `fetch_primals.sh` | AVAILABLE | primalSpring/tools/fetch_primals.sh |
| `nucleus_launcher.sh` | AVAILABLE | primalSpring/tools/ + plasmidBin/ |
| Hardware | CONFIRMED | 5800X3D, 128GB DDR4 (southGate) |

### What needs execution

| Step | Status | Blocker |
|------|--------|---------|
| `fetch_primals.sh --all` | NOT YET RUN | Need GitHub release v2026.05.23 live |
| biomeOS start | PENDING | Needs fetched binaries |
| `primalspring validate` (49 scenarios) | PENDING | Needs live NUCLEUS |
| `wetspring validate --all` (345 scenarios vs live primals) | PENDING | Needs live NUCLEUS |
| Multi-domain validation with healthSpring | PENDING | healthSpring gate readiness |
| Socket conflict assessment | PENDING | Co-residency test |

### Known gaps

| Gap | Impact | Resolution |
|-----|--------|------------|
| No GPU on southGate | GPU scenarios skip (RTX needed for strandGate) | strandGate for GPU science |
| `guidestone_binary` in manifest says `wetspring_guidestone` | Stale (V184: `wetspring certify`) | Upstream fix in primalSpring manifest |
| `guidestone_readiness = 3` in manifest | Stale (wetSpring self-reports Level 5) | Upstream fix in primalSpring manifest |
| BTSP auth not yet wired | bearDog Ed25519 needed for live bonding | Deferred to bearDog readiness |

---

## strandGate Readiness (secondary)

strandGate (Dual EPYC, RTX 3090 + RX 6950 XT) is wetSpring's GPU gate.
GPU-heavy scenarios (44 GPU modules, CPU-vs-GPU parity, metalForge dispatch)
will validate there. Co-resident with airSpring.

| Component | Status |
|-----------|--------|
| Remote access | PENDING (cellMembrane) |
| GPU scenarios | 14 GPU + 9 CPU-vs-GPU + cross-spring |
| Tenaillon 264-clone batch | strandGate target (590 GB, `compute.fan_out`) |

---

## Deployment Plan

```bash
# 1. Fetch primal binaries
cd /home/southgate/Development/ecoPrimals/springs/primalSpring
./tools/fetch_primals.sh --all

# 2. Start NUCLEUS composition
./tools/nucleus_launcher.sh

# 3. Validate primalSpring surface (49 scenarios)
primalspring validate

# 4. Validate wetSpring against live primals
cd /home/southgate/Development/ecoPrimals/springs/wetSpring
wetspring validate --all

# 5. Report composition parity results
# → Update this handoff with pass/fail/skip counts
```

---

## For primalSpring

1. `downstream_manifest.toml` wetspring entry has stale `guidestone_binary = "wetspring_guidestone"` — should be `"wetspring"` (UniBin subcommand: `wetspring certify`)
2. `guidestone_readiness = 3` should be `5` (Level 5 since V179)
3. Confirm plasmidBin v2026.05.23 release is live for `fetch_primals.sh`
4. southGate has no GPU — confirm GPU scenarios can `check_skip` gracefully

---

## Next Steps

1. Execute deployment plan above once plasmidBin release is confirmed
2. Report socket layout for healthSpring co-residency
3. Document any capability collisions
4. Update this handoff with live validation results
