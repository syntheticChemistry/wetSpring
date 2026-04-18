# Context — wetSpring

## What This Is

wetSpring is a pure Rust scientific validation spring that reproduces published
results in metagenomics, analytical chemistry (LC-MS, PFAS), and mathematical
biology using the barraCuda GPU compute library. It is part of the ecoPrimals
sovereign computing ecosystem — a collection of self-contained binaries that
coordinate via JSON-RPC 2.0 over Unix sockets, with zero compile-time coupling
between components.

**Current release — V145:** Primal proof Tier 2 (IPC-WIRED) — Exp403 live NUCLEUS IPC vs local Rust (barraCuda, NestGate, Squirrel, BearDog, toadStool over UDS with check_skip). 22 barraCuda consumed capabilities in niche.rs. PG-09 IPC evaporation surface documented. V144: composition validation tier (Exp400-402, 18 IPC roundtrips). V143: deploy graph canonical. V142: Wire Standard L2+L3.

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
- **Architecture:** 2 library crates + 341 validation/benchmark binaries
- **Communication:** JSON-RPC 2.0 over Unix sockets, 42 niche capabilities, 22 consumed (barraCuda IPC), 37 dispatch methods, 21 domains, Wire Standard L2+L3
- **License:** AGPL-3.0-or-later
- **Tests:** 1,592 lib + 18 IPC roundtrip + integration, 0 failed
- **Validation checks:** 5,900+ across 341 binaries
- **Composition:** 136/136 proto-nucleate (Exp400), Exp401 IPC parity (43/43), Exp402 niche gate (63/63), Exp403 primal parity (Tier 2, 5 primals), 9 niche deps (5 required + 4 optional)
- **Deploy graphs:** 7 (all canonical `[[graph.nodes]]` schema, bonding + fragments metadata, validated by `graph_validate.rs`)
- **MSRV:** 1.87 (Rust edition 2024)
- **Crate count:** 2 workspace crates (wetspring-barracuda, wetspring-forge)
- **Clippy:** zero errors (pedantic + nursery)
- **Unsafe code:** zero — `forbid(unsafe_code)` at workspace level + per-crate roots
- **Primal gaps:** 7 open (`docs/PRIMAL_GAPS.md`)
- **Coverage:** 91.20% line / 90.30% function (gated at 90%)

## Key Capabilities

- **16S rRNA pipeline:** FASTQ QC, merge, derep, DADA2 denoise, chimera, taxonomy
- **Diversity:** Shannon, Simpson, Chao1, Bray-Curtis, UniFrac, PCoA, rarefaction
- **Phylogenetics:** Felsenstein pruning, Robinson-Foulds, HMM, NJ, placement
- **LC-MS:** EIC extraction, peak detection, feature tables, spectral matching, KMD
- **Math biology:** ODE systems (QS, bistable, cooperation, phage, capacitor)
- **Anderson physics:** hormesis, binding landscapes, disorder mapping
- **Drug repurposing:** NMF, TransE knowledge graph embedding, molecular docking
- **GPU acceleration:** 44 GPU modules via barraCuda v0.3.12, 150+ primitives consumed
- **IPC:** 37 JSON-RPC methods, 42 niche capabilities, 21 domains, 1 composition health handler (science_health), 8 MCP tools, Wire Standard L2+L3
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
Tier 1: Python baseline  → Rust CPU parity  → GPU validation
           (71 scripts)     (1,592+ tests)     (47 GPU modules)
Tier 2: Rust validation   → NUCLEUS composition patterns
           (340 binaries)   (136/136 proto-nucleate, 7 deploy graphs)
Tier 3: Composition       → IPC parity → Niche gate → ecoBin harvest
           (L2+L3)          (Exp401)     (Exp402)     (plasmidBin)
```

## Design Philosophy

These binaries are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling.
