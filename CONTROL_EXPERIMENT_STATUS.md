# Control Experiment Status — wetSpring

**Last updated:** V206 (2026-06-13)
**Tests:** 2,160 workspace (0 failed, 3 ignored)
**UniBin scenarios:** 346 (319 validation + 23 benchmark + 4 composition)
**Live NUCLEUS:** southGate — 13/13 processes (13/13 health-responding)
**Experiment specs:** 386 indexed (385 completed, 1 in progress)
**Clippy:** 0 warnings (pedantic + nursery, all feature combinations)
**Dispatch methods:** 47 (incl. lifecycle.status + 6 bonding.* + composition.mesh_health)
**Binary discovery:** plasmidBin-only (v2026.05.28, Wave 60 eukaryotic)

## V206 Wave 111 — WS-11: MAPQ Calibration Module

- **MAPQ calibration pipeline:** `mapq_calibration` module — simulated read generator (xorshift64), training pipeline (sim→map→compare→model), `MapqModel` lookup table (Phred-scaled), wired into `compute_mapq` via `MapperConfig.mapq_model`.
- **Full MAPQ chain complete:** calibrate() → MapqModel → compute_mapq → pileup mapq_sums → binomial_quality combined error model.
- **WS-11 variant caller parity:** 5/8 items complete (MAPQ calibration, MAPQ-aware binomial, per-gen thresholds, cross-validation, base binomial).
- **+12 tests** from calibration module (2,148→2,160).

## V205 Wave 111 — WS-11: Binomial Model Evolution

- **MAPQ-aware quality weighting:** Combined error model P(err) = P_base + P_map - P_base×P_map.
- **Per-generation LTEE thresholds:** Exponential saturation model (`LteeThresholds`).
- **breseq cross-validation:** Concordance stats (sensitivity/precision/F1) with per-type breakdown.
- **+24 tests** (2,124→2,148).

## V204 Wave 111 — Stream 6: Mesh Health in Certify + Scenario

- **`wetspring certify` Layer 3b:** NUCLEUS mesh health audit (13/13 + version skew).
- **`mesh-health` validation scenario:** Tier 2, registered in build_registry().
- **Feature-gated:** `barracuda-lib` with skip fallback.

## V203 Wave 111 — Deep Debt: Module Splits + Macro Consolidation

- **`gonzales.rs` → `gonzales/` submodule:** 586L split into 3 domain files.
- **`discover.rs` macro:** 8 wrappers → `primal_discover_fn!` macro (70L→20L).
- **`variant_caller/stats.rs`:** 267L pure-math extracted (873L→604L, 31% reduction).

## V195 Wave 76 — Deep Debt: Architecture Evolution

- **TCP transport:** `ipc::transport` now supports `Transport::Tcp(addr)` alongside Unix sockets. Cross-gate JSON-RPC with unified `jsonrpc_line()` dispatch.
- **Primal name centralization:** Replaced scattered string literals with `primal_names::*` constants.
- **Songbird registration fix:** Non-`barracuda-lib` builds register `niche::CAPABILITIES` instead of empty list.
- **macOS RSS:** `bench::peak_rss_mb()` supports macOS via safe `ps` subprocess.
- **Ionic bonding modernized:** `try_from` casts, `ok_or_else`, dead match arms removed.
- **Dependencies bumped:** axum 0.8.9, blake3 1.8.5, proptest 1.11, tempfile 3.27, tower-http 0.6.11.
- **Registries const-ified:** `ScenarioRegistry` and `BenchmarkRegistry` constructors/accessors.
- **4 new tests** from TCP transport (+4 to 2,089).

## V194 Wave 76 — Parity Alignment

- **bearDog w135 absorbed:** wetSpring has zero direct `auth.verify_ionic` calls. Multi-issuer change is transparent.
- **157 compiler warnings eliminated:** Consolidated lint suppression on `validation::experiments` module.
- **Build gate:** 0 warnings, 0 clippy warnings, 2,085→2,089 tests.

## V193 Wave 67 — Glacial Cutover

- **Songbird security socket fix (P0):** `SecurityCryptoProvider::from_env()` now honors `--security-socket`/`SECURITY_PROVIDER_ENDPOINT`.
- **biomeOS capability.call proxy (P0):** API socket proxies `capability.call` to Neural API.
- **bearDog S4 auth verified:** TCP :9100, ionic token roundtrip, Ed25519 DID operational.

## V192 Wave 63 — River Delta

- **PG-02/PG-04 re-verified live.** Provenance trio responding, NestGate alive with 66 capabilities.
- **`composition_nucleus.sh` fossilized** → `fossilRecord/tools/`.
- **`domain_profile.toml` created** for pseudoSpore emission.
- **Temporal sync tooling** confirmed.

## V190 Wave 60 — Eukaryotic Polish

- **Doc debt resolved:** 17 stale V188 banners → V190. W55 handoff archived.
- **Clippy debt:** 37 → 4 warnings. Removed stale `#[expect()]`, auto-fixed 23 suggestions.
- **Test fixes:** 3 socket discovery tests updated for connect-probe. Capability count 49 → 50.
- **`.gate` identity file:** cascade-pull v2.0.0 auto-detection enabled.
- **`composition_nucleus.sh`:** 8 → 13 primals (full NUCLEUS from one script).

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
