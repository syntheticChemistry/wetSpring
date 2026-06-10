# Context — wetSpring

## What This Is

wetSpring is a pure Rust scientific validation spring that reproduces published
results in metagenomics, analytical chemistry (LC-MS, PFAS), and mathematical
biology using the barraCuda GPU compute library. It is part of the ecoPrimals
sovereign computing ecosystem — components coordinate via JSON-RPC 2.0 over
Unix sockets, with zero compile-time coupling.

**Current release — V200:** Wave 107 upstream absorption. Topology-aware mesh routing — `capability.resolve` RPC fix (`domain`→`capability`), `Transport::MeshRelay` variant, 4-tier socket discovery cascade (`BIOMEOS_SOCKET_DIR`), `to_transport_or_relay()`. **Clippy zero warnings** (all feature combinations). 2,107 tests (0 failures). 345 scenarios, 51 niche, 59 consumed, 46 dispatch, 22 domains. 0 PG gaps open.

## Role in the Ecosystem

wetSpring validates that Python scientific baselines (diversity indices, ODE
solvers, phylogenetics, spectral matching) can be faithfully ported to Rust and
then promoted to GPU acceleration via barraCuda WGSL shaders. It is the primary
consumer of barraCuda's bio-domain GPU primitives and the upstream contributor
of statistical and ODE-related shader requirements. Other springs (hotSpring,
airSpring, groundSpring) cross-validate shared primitives through wetSpring's
evolution pipeline.

## Technical Facts

- **Language:** 100% Rust, zero C dependencies (wgpu optional for GPU)
- **Architecture:** 2 library crates + 1 UniBin (`wetspring`, 345 scenarios)
- **Communication:** JSON-RPC 2.0 over Unix sockets, 51 niche capabilities, 59 consumed (33 barraCuda canonical + 15 legacy + 7 bonding/lifecycle + 4 Wave 17/20), 46 dispatch methods, 22 domains, Wire Standard L2+L3
- **License:** AGPL-3.0-or-later
- **Tests:** 2,107 workspace (0 failed)
- **Validation checks:** 5,967+ across 345 scenarios (UniBin)
- **Composition:** 136/136 proto-nucleate (Exp400), Exp401 IPC parity (43/43), Exp402 niche gate (63/63), Exp403 primal parity (Tier 2, 5 primals), `wetspring certify` (Level 5, NUCLEUS 38/38, 4 skip), 9 niche deps (5 required + 4 optional)
- **Deploy graphs:** 7 (all canonical `[[graph.nodes]]` schema, bonding + fragments metadata, validated by `graph_validate.rs`)
- **MSRV:** 1.87 (Rust edition 2024)
- **Crate count:** 2 workspace crates (wetspring-barracuda, wetspring-forge)
- **Clippy:** zero warnings (pedantic + nursery)
- **Unsafe code:** zero — `forbid(unsafe_code)` at workspace level + per-crate roots
- **Primal gaps:** 0 open (`docs/PRIMAL_GAPS.md`) — PG-01 through PG-22, PG-06 locally wired (V185), PG-02/PG-04 VERIFIED (V188), 22 resolved/closed. Zero wetSpring-internal gaps.
- **Coverage:** 91.20% line / 90.30% function (gated at 90%)

## Key Capabilities

- **16S rRNA pipeline:** FASTQ QC, merge, derep, DADA2 denoise, chimera, taxonomy
- **Diversity:** Shannon, Simpson, Chao1, Bray-Curtis, UniFrac, PCoA, rarefaction
- **Phylogenetics:** Felsenstein pruning, Robinson-Foulds, HMM, NJ, placement
- **LC-MS:** EIC extraction, peak detection, feature tables, spectral matching, KMD
- **Math biology:** ODE systems (QS, bistable, cooperation, phage, capacitor)
- **Anderson physics:** hormesis, binding landscapes, disorder mapping
- **Drug repurposing:** NMF, TransE knowledge graph embedding, drug-target scoring
- **GPU acceleration:** 44 GPU modules via barraCuda v0.4.0, 150+ primitives consumed
- **IPC:** 45 JSON-RPC methods, 50 niche capabilities, 22 domains, 1 live composition health handler (science_health — runtime probing), 6 bonding.* methods (IonicContractRegistry), 8 MCP tools, Wire Standard L2+L3
- **Ecosystem wiring:** sweetGrass braids, toadStool performance surface, StreamItem NDJSON
- **Primal discovery:** coralReef, toadStool, petalTongue, Squirrel, sweetGrass, rhizoCrypt, loamSpine

## What This Does NOT Do

- Does not compile WGSL shaders (that is barraCuda/coralReef)
- Does not manage hardware discovery or dispatch routing (that is toadStool)
- Does not run deployment orchestration (that is biomeOS)
- Does not perform cryptographic operations (that is BearDog)
- Does not generate visualizations directly (renders scenarios for petalTongue)

## Related Repositories

- [wateringHole](https://github.com/ecoPrimals/wateringHole) — ecosystem standards and registry
- [barraCuda](https://github.com/ecoPrimals/barraCuda) — GPU compute library (math primal)
- [toadStool](https://github.com/ecoPrimals/toadStool) — hardware discovery and dispatch
- [coralReef](https://github.com/ecoPrimals/coralReef) — WGSL shader compilation to native

## Evolution Path

```
          ┌──────────────────────────────────────────────────┐
          │  wetspring UniBin (1 binary, 345 scenarios)      │
          │  certify · validate · benchmark · serve · status │
          └──────────────────────────────────────────────────┘

Tier 1: Python baseline → Rust CPU parity → GPU validation
           (55 scripts)    (1,962 tests)     (44 GPU modules)
Tier 2: UniBin scenarios → NUCLEUS composition patterns
           (345 scenarios)  (136/136 proto-nucleate, 7 deploy graphs)
Tier 3: Composition      → IPC parity → Niche gate
           (L2+L3)         (Exp401)     (Exp402)
Tier 4: Primal proof     → Live NUCLEUS IPC (Exp403) → ecoBin harvest
           (59 consumed)    (5 primals, check_skip)    (plasmidBin)
Tier 5: guideStone       → Self-validating NUCLEUS node (Level 5)
           (wetspring certify) (38/38 live NUCLEUS, v0.9.17 manifest)
```

## Gate Deployment

| Field | Value |
|-------|-------|
| **Gate** | southGate |
| **Hardware** | AMD Ryzen 7 5800X3D 8-Core, 128GB DDR4, NVIDIA RTX 4060 |
| **Composition** | Full NUCLEUS (13/13 processes, 11/13 health-responding, 2 BTSP-gated) |
| **NUCLEUS status** | **eukaryotic** (plasmidBin v2026.05.28, biomeOS 1725 caps / 21 surfaces) |
| **Songbird federation** | 0.0.0.0:7700 (LAN-reachable, cross-subnet confirmed) |
| **LAN mesh** | ready — covalent linking via Songbird TCP |
| **Cell graph** | `plasmidBin/cells/wetspring_cell.toml` |
| **Secondary gate** | strandGate (Dual EPYC 64-core, 256GB ECC, RTX 3090 + RX 6950 XT) — GPU science |
| **Launch** | `plasmidBin/nucleus_launcher.sh --family-id nucleus01` (core primals) + manual `--port` starts for sweetGrass/rhizoCrypt/skunkBat (launcher `--socket` bug) |
| **Forgejo** | Remotes configured via `cascade-pull.sh --ensure-remotes`; push **blocked** pending SSH key registration |
| **Gate identity** | `$ECOPRIMALS_ROOT/.gate` contains `southGate` (cascade-pull auto-detection) |

## Design Philosophy

Primals are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling.
