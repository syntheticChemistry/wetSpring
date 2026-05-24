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

### Execution Results (2026-05-23 21:00 EDT)

| Step | Status | Result |
|------|--------|--------|
| `fetch_primals.sh` | DONE (local) | GitHub release has `checksums.toml` only — no binary assets uploaded. Used local `plasmidBin` repo (12/13 binaries, skunkBat missing) |
| `nucleus_launcher.sh start` | DONE | 7/12 primals socket-live: biomeOS, BearDog, Songbird, ToadStool, barraCuda, NestGate, sweetGrass |
| `primalspring validate` | DONE | **1043/1172 passed** (89%), 146 skipped, 129 failures |
| `wetspring validate` | DONE | **66/77 scenarios PASS**, 9 FAIL (see below), runner exited early on missing data |
| Multi-domain validation with healthSpring | PENDING | healthSpring gate readiness |
| Socket conflict assessment | NOT YET | Awaiting healthSpring co-residency |

### Primals that did NOT come online

| Primal | Phase | Failure Reason |
|--------|-------|----------------|
| Songbird | 1 | CLI arg mismatch: binary expects `--beardog-socket`, launcher sends `--security-socket` |
| coralReef | 2 | Socket never appeared (timeout) |
| NestGate | 2 | **FIXED** — needed `NESTGATE_JWT_SECRET` (32+ bytes) |
| Squirrel | 2 | Socket never appeared (Ollama not running) |
| rhizoCrypt | 3 | Socket never appeared (timeout) |
| loamSpine | 3 | Tokio runtime nesting bug in infant_discovery.rs |
| petalTongue | 4 | CLI arg mismatch: binary has no `--socket` flag |

### wetSpring Validation Failures (9)

| Scenario | Result | Cause |
|----------|--------|-------|
| checksum:niche.rs | FAIL | Expected hash stale after V184 niche changes |
| checksum:Cargo.toml | FAIL | Expected hash stale after feature work |
| benchmark_23_domain_timing | 0/0 | Empty benchmark (hardware-specific) |
| benchmark_cross_spring_modern | 0/0 | Empty benchmark (hardware-specific) |
| benchmark_pipeline | 0/0 | Empty benchmark (hardware-specific) |
| barracuda_cpu_v12 | 54/55 | Single tolerance assertion (2.0 vs expected 1.0) |
| biomeos_nucleus_v98 | 39/40 | biomeOS binary not found for local build check |
| composition_nucleus_v1 | 135/136 | IPC routing gap (capability not wired to live primal) |
| features | SKIP(exit 2) | Missing mzML data (`data/exp005_asari/MT02`) — not a failure |

### Known gaps (updated post-deployment)

| Gap | Impact | Resolution |
|-----|--------|------------|
| No GPU on southGate | GPU/benchmark scenarios skip | strandGate for GPU science |
| `guidestone_binary` in manifest says `wetspring_guidestone` | Stale (V184: `wetspring certify`) | Upstream fix in primalSpring manifest |
| `guidestone_readiness = 3` in manifest | Stale (wetSpring self-reports Level 5) | Upstream fix in primalSpring manifest |
| BTSP auth not yet wired | bearDog Ed25519 needed for live bonding | Deferred to bearDog readiness |
| Songbird `--security-socket` → `--beardog-socket` | Tower Phase 1 partial | Upstream `nucleus_launcher.sh` flag fix |
| loamSpine Tokio nesting panic | Provenance trio incomplete | Upstream runtime fix in loamSpine |
| petalTongue no `--socket` CLI | Interface phase incomplete | Upstream CLI alignment |
| plasmidBin v2026.05.23 no binary assets | Cannot use `fetch_primals.sh` from GitHub | Upstream build/upload needed |
| Checksum hashes stale | 2 FAIL in self-integrity check | Regenerate `gen_checksums` after V184 |

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
# 1. Fetch primal binaries (use local plasmidBin repo until GH release has assets)
export NUCLEUS_BIN_DIR=/home/southgate/Development/ecoPrimals/infra/plasmidBin/primals

# 2. Start NUCLEUS composition (requires NODE_ID + JWT)
export NODE_ID=southGate
export NESTGATE_JWT_SECRET=$(openssl rand -base64 48)
cd /home/southgate/Development/ecoPrimals/springs/primalSpring
./tools/nucleus_launcher.sh start

# 3. Validate primalSpring surface (49 scenarios)
cargo run --release --bin primalspring_unibin -- validate

# 4. Validate wetSpring against live primals
cd /home/southgate/Development/ecoPrimals/springs/wetSpring
cargo run --release --bin wetspring --features guidestone -- validate
```

---

## For primalSpring

1. `downstream_manifest.toml` wetspring entry has stale `guidestone_binary = "wetspring_guidestone"` — should be `"wetspring"` (UniBin subcommand: `wetspring certify`)
2. `guidestone_readiness = 3` should be `5` (Level 5 since V179)
3. **plasmidBin v2026.05.23 release has no binary assets** — only `checksums.toml`. Upload builds or document local repo workaround.
4. `nucleus_launcher.sh` passes `--security-socket` to Songbird but binary expects `--beardog-socket` — flag rename drift
5. `nucleus_launcher.sh` passes `--socket` to petalTongue but binary has no such flag
6. loamSpine `infant_discovery.rs:233` panics with Tokio runtime nesting — `block_on` inside existing runtime
7. southGate has no GPU — confirm GPU scenarios can `check_skip` gracefully

---

## Next Steps

1. Regenerate checksums (`gen_checksums`) to clear 2 stale hash FAILs
2. Fix the single tolerance assertion in `barracuda_cpu_v12` (2.0 vs 1.0)
3. Re-run once Songbird, loamSpine, petalTongue upstream fixes land
4. Deploy on strandGate for GPU scenarios once remote access is available
5. Report socket layout for healthSpring co-residency
6. Document any capability collisions with healthSpring
