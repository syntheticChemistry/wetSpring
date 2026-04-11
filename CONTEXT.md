# Context — wetSpring

## What This Is

wetSpring is a pure Rust scientific validation spring that reproduces published
results in metagenomics, analytical chemistry (LC-MS, PFAS), and mathematical
biology using the barraCuda GPU compute library. It is part of the ecoPrimals
sovereign computing ecosystem — a collection of self-contained binaries that
coordinate via JSON-RPC 2.0 over Unix sockets, with zero compile-time coupling
between components.

**Current release — V140:** Composition validation evolution — full ecosystem audit, deploy graph canonicalization (all 7 `[[graph.node]]`), tolerance provenance trail, zero clippy warnings, cargo-deny clean, CI orchestrator. V139: NUCLEUS composition validation (97/97 proto-nucleate, Exp400). V138: primal composition patterns. V137: provenance headers, tolerance centralization.

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
- **Architecture:** 2 library crates + 356 validation/benchmark binaries (334 barracuda + 22 forge)
- **Communication:** JSON-RPC 2.0 over Unix sockets, 45 IPC capabilities, 37 dispatch methods
- **License:** AGPL-3.0-or-later
- **Tests:** 1,942 (unit + integration + property + doc), 0 failed
- **Validation checks:** 5,800+ across 356 binaries
- **Composition:** 97/97 proto-nucleate alignment (Exp400), 9 niche dependencies (5 required + 4 optional)
- **Deploy graphs:** 7 (all canonical `[[graph.node]]` schema, validated by `graph_validate.rs`)
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
- **GPU acceleration:** 44 GPU modules via barraCuda v0.3.7, 150+ primitives consumed
- **IPC:** 37 JSON-RPC methods, 45 advertised capabilities, 5 composition health handlers, 8 MCP tools
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
           (58 scripts)     (1,942 tests)      (44 GPU modules)
Tier 2: Rust validation   → NUCLEUS composition patterns
           (356 binaries)   (97/97 proto-nucleate, 7 deploy graphs)
Tier 3: Composition       → ecoBin harvest to plasmidBin
```

## Design Philosophy

These binaries are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling.
