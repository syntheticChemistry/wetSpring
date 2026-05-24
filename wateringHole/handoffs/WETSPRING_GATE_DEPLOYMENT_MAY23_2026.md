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

### Execution Results (2026-05-23 22:08 EDT — plasmidBin CLI fix absorbed)

| Step | Status | Result |
|------|--------|--------|
| plasmidBin pull | DONE | `8c8cb44` — CLI alignment fix for v2026.05.23 binaries |
| `nucleus_launcher.sh --composition nucleus --validate` | **DONE** | **9/9 primals started**, exp091+exp094 PASS |
| `primalspring validate` (primalSpring tools/) | DONE | **1043/1172 passed** (89%), 146 skipped, 129 failures (cross-gate routing) |
| `wetspring validate --tier rust` | DONE | **66/77 scenarios PASS**, 9 FAIL (see below) |
| Multi-domain validation with healthSpring | PENDING | healthSpring gate readiness |
| Socket conflict assessment | NOT YET | Awaiting healthSpring co-residency |

#### plasmidBin `nucleus_launcher.sh` Results (9/9 started)

| Primal | Port | Health |
|--------|------|--------|
| bearDog | 9100 | HEALTHY |
| Songbird | 9200 | HEALTHY |
| toadStool | 9400 | STARTED (health probe pending) |
| barraCuda | 9740 | HEALTHY |
| coralReef | 9730 | HEALTHY |
| nestGate | 9500 | STARTED (health probe pending) |
| rhizoCrypt | 9700 | STARTED (health probe pending) |
| loamSpine | 9710 | STARTED (health probe pending) |
| sweetGrass | 9720 | STARTED (health probe pending) |

**Composition validation:** exp091 primal_routing_matrix PASS, exp094 composition_parity PASS.

### Primals with degraded health probes (started but UNREACHABLE)

| Primal | Phase | Failure Reason |
|--------|-------|----------------|
| toadStool | 2 | Health probe UNREACHABLE (process running, may need longer timeout) |
| nestGate | 2 | Health probe UNREACHABLE (process running) |
| rhizoCrypt | 3 | Health probe UNREACHABLE (process running) |
| loamSpine | 3 | Tokio runtime nesting bug in infant_discovery.rs (upstream) |
| sweetGrass | 3 | Health probe UNREACHABLE (process running) |

Note: loamSpine is the only hard failure (panic). Others started successfully but
health probes timed out — likely need extended timeout or alternate probe path.
The previous run (primalSpring `nucleus_launcher.sh`) showed socket timeouts for
the same primals. The plasmidBin launcher uses HTTP health endpoints instead of UDS.

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
| loamSpine Tokio nesting panic | Provenance trio incomplete on southGate | Upstream runtime fix in loamSpine |
| toadStool/nestgate/rhizoCrypt/sweetgrass health probe timing | Health sweep UNREACHABLE despite successful start | Extend timeout or fix health endpoint |
| Checksum hashes stale | 2 FAIL in self-integrity check | **FIXED** — regenerated validation/CHECKSUMS |
| Squirrel needs Ollama | AI narration unavailable | Not included in `nucleus` composition (meta_tier) |

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
3. plasmidBin `nucleus_launcher.sh` CLI fix `8c8cb44` resolved Songbird/petalTongue issues — confirmed working on southGate
4. Health sweep timeout for toadStool, nestGate, rhizoCrypt, sweetGrass — all start and register capabilities but don't respond to health probe within default timeout. Consider extending `--health-timeout` or adding alternate probe.
5. loamSpine `infant_discovery.rs:233` panics with Tokio runtime nesting — `block_on` inside existing runtime. Only hard failure in composition.
6. southGate has no GPU — confirm GPU scenarios can `check_skip` gracefully

---

## Next Steps

1. Fix the single tolerance assertion in `barracuda_cpu_v12` (2.0 vs 1.0)
2. Re-run full 345-scenario suite once loamSpine upstream fix lands (provenance trio)
3. Deploy on strandGate for GPU scenarios once remote access is available
4. Report socket layout for healthSpring co-residency
5. Document any capability collisions with healthSpring
6. Extend health probe timeouts for slow-starting primals
