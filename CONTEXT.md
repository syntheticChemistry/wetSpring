# Context — wetSpring

## What This Is

wetSpring is a pure Rust scientific validation spring that reproduces published
results in metagenomics, analytical chemistry (LC-MS, PFAS), and mathematical
biology using the barraCuda GPU compute library. It is part of the ecoPrimals
sovereign computing ecosystem — a collection of self-contained binaries that
coordinate via JSON-RPC 2.0 over Unix sockets, with zero compile-time coupling
between components.

**Current release — V137:** Full provenance + tolerance centralization + IPC modularization — `//! Provenance:` headers on all 355 binaries, 8 new named tolerance constants (242 total), `ipc/connection.rs` extraction, GPU buffer renames, doc link fixes. V136: `thiserror` derives, named cast helpers (~60 casts), upstream contract pinning, determinism tests, CI pin, zero hardcoded primal strings, CONTRIBUTING.md + SECURITY.md.

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
- **Architecture:** 2 library crates + 355 validation/benchmark binaries (333 barracuda + 22 forge)
- **Communication:** JSON-RPC 2.0 over Unix sockets (IPC feature-gated)
- **License:** AGPL-3.0-or-later
- **Tests:** 1,902 total (unit + integration + property + doc); 1,205+ lib (default), 1,569+ lib (ipc+vault+json)
- **Validation checks:** 5,700+ across 355 binaries
- **MSRV:** 1.87 (Rust edition 2024)
- **Crate count:** 2 workspace crates (wetspring-barracuda, wetspring-forge)
- **Clippy:** zero warnings (pedantic + nursery, all features, `-D warnings`)
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
- **IPC:** 23 JSON-RPC methods, 8 MCP tools, Songbird discovery, capability-based primal coordination
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
Python baseline → Rust CPU parity → GPU validation → sovereign pipeline
     (58 scripts)    (1,902 tests)     (44 GPU modules)   (barraCuda/coralReef)
```

## Design Philosophy

These binaries are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling.
