# Context — wetSpring

## What This Is

wetSpring is a pure Rust scientific validation spring that reproduces published
results in metagenomics, analytical chemistry (LC-MS, PFAS), and mathematical
biology using the barraCuda GPU compute library. It is part of the ecoPrimals
sovereign computing ecosystem — a collection of self-contained binaries that
coordinate via JSON-RPC 2.0 over Unix sockets, with zero compile-time coupling
between components.

**Current release — V144:** Composition evolution — biomeOS v3.04 alignment. Fixed `akida-driver` path case mismatch. Removed universal `composition.*_health` methods (biomeOS owns these per `COMPOSITION_HEALTH_STANDARD.md`). Retained `composition.science_health` (spring-specific). Capabilities: 42 niche, 37 dispatch. V143: Deploy graph canonical migration, D07 composition validation (141/141 Exp400). V142: Wire Standard L2+L3, `identity.get`, 22 consumed capabilities. V141: audit remediation. V140: ecosystem audit. V139: NUCLEUS composition.

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
- **Architecture:** 2 library crates + 360 validation/benchmark binaries (338 barracuda + 22 forge)
- **Communication:** JSON-RPC 2.0 over Unix sockets, 42 niche capabilities, 37 dispatch methods, 21 domains, Wire Standard L2+L3
- **License:** AGPL-3.0-or-later
- **Tests:** 1,949 (unit + integration + property + doc), 0 failed
- **Validation checks:** 5,800+ across 356 binaries
- **Composition:** 141/141 proto-nucleate alignment (Exp400 D01–D07), 9 niche dependencies (5 required + 4 optional)
- **Deploy graphs:** 7 (all canonical `[[graph.nodes]]` schema, bonding + fragments metadata, validated by `graph_validate.rs`)
- **MSRV:** 1.87 (Rust edition 2024)
- **Crate count:** 2 workspace crates (wetspring-barracuda, wetspring-forge)
- **Clippy:** zero errors (pedantic + nursery)
- **Unsafe code:** zero — `forbid(unsafe_code)` at workspace level + per-crate roots
- **Coverage:** 91.20% line / 90.30% function (gated at 90%)

## Key Capabilities

- **16S rRNA pipeline:** FASTQ QC, merge, derep, DADA2 denoise, chimera, taxonomy
- **Diversity:** Shannon, Simpson, Chao1, Bray-Curtis, UniFrac, PCoA, rarefaction
- **Phylogenetics:** Felsenstein pruning, Robinson-Foulds, HMM, NJ, placement
- **LC-MS:** EIC extraction, peak detection, feature tables, spectral matching, KMD
- **Math biology:** ODE systems (QS, bistable, cooperation, phage, capacitor)
- **Anderson physics:** hormesis, binding landscapes, disorder mapping
- **Drug repurposing:** NMF, TransE knowledge graph embedding, molecular docking
- **GPU acceleration:** 44 GPU modules via barraCuda v0.3.11, 150+ primitives consumed
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
           (58 scripts)     (1,950 tests)      (44 GPU modules)
Tier 2: Rust validation   → NUCLEUS composition patterns
           (360 binaries)   (141/141 proto-nucleate, 7 deploy graphs)
Tier 3: Composition       → Wire Standard compliance → ecoBin harvest
           (L2+L3)          (22 consumed caps)          (plasmidBin)
```

## Design Philosophy

These binaries are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling.
