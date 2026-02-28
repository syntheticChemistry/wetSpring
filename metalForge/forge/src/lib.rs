// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]
#![deny(clippy::expect_used, clippy::unwrap_used)]

//! wetSpring Forge v0.3.0 — hardware discovery, dispatch, and absorption tracking.
//!
//! Forge discovers what compute substrates exist on THIS machine at runtime
//! and routes life science workloads to the best capable substrate. It leans
//! on `ToadStool`/barracuda for GPU discovery and device management, and adds
//! NPU probing, cross-substrate orchestration, streaming pipeline topology
//! analysis, and shader origin tracking locally.
//!
//! # Design Principle
//!
//! wetSpring is a biome. `ToadStool` (barracuda) is the fungus — it lives in
//! every biome. We lean on it for what it already provides (GPU enumeration,
//! shader dispatch, buffer management), and evolve new capabilities locally
//! (NPU probing, cross-substrate routing, ODE shaders). `ToadStool` absorbs
//! what works, then all Springs benefit.
//!
//! Springs don't reference each other. hotSpring doesn't import wetSpring.
//! Both lean on `ToadStool` independently — hotSpring evolves physics shaders,
//! wetSpring evolves bio shaders, and `ToadStool` absorbs both.
//!
//! # Architecture
//!
//! ```text
//!    ┌─────────────────────────────┐
//!    │  probe (barracuda + local)  │  wgpu adapters + /dev + /proc
//!    └──────────┬──────────────────┘
//!               │ Vec<Substrate>
//!    ┌──────────▼──────────────────┐
//!    │       inventory             │  unified view of all substrates
//!    └──────────┬──────────────────┘
//!               │ &Substrate
//!    ┌──────────▼──────────────────┐
//!    │       dispatch              │  capability-based routing
//!    └──────────┬──────────────────┘
//!               │ BioWorkload
//!    ┌──────────▼──────────────────┐
//!    │       workloads             │  preset domains + shader origin tracking
//!    └─────────────────────────────┘
//! ```
//!
//! # Life Science Workloads
//!
//! wetSpring routes bioinformatics and analytical chemistry workloads:
//! - **GPU (absorbed, 28)**: Felsenstein, HMM, spectral cosine, diversity,
//!   ANI/SNP/dN/dS, RF batch, QS ODE, k-mer, `UniFrac`, 5 ODE bio systems
//!   (via `BatchedOdeRK4` trait), DADA2, GBM, `Robinson-Foulds`, chimera,
//!   diversity fusion (absorbed by `ToadStool` S63)
//! - **NPU-optimal**: Taxonomy classification, anomaly detection, PFAS screening
//! - **CPU-optimal**: FASTQ/mzML parsing, tree traversal
//!
//! # Write → Absorb → Lean Tracking
//!
//! The [`workloads`] module tracks shader origin for every domain:
//! - **Absorbed** (28): lean on `ToadStool` primitives (incl. 5 ODE via trait)
//! - **Local** (0): zero local WGSL — Full Lean achieved (S63)
//! - **CPU-only** (1): I/O-bound domain
//!
//! When `ToadStool` absorbs a local shader, update origin from `Local` to
//! `Absorbed` and rewire. This is the Lean step.
//!
//! # Extension Pattern (following hotSpring)
//!
//! New bio-specific GPU shaders are written as local extensions with:
//! 1. WGSL shader in `barracuda/src/bio/shaders/` (`include_str!`)
//! 2. CPU reference implementation with analytical validation
//! 3. Binding layout table in doc comments
//! 4. Dispatch geometry documented (workgroup size, grid dims)
//! 5. Handoff document in `wateringHole/handoffs/`

pub mod bridge;
pub mod data;
pub mod dispatch;
pub mod inventory;
pub mod ncbi;
pub mod nest;
pub mod node;
pub mod probe;
pub mod streaming;
pub mod substrate;
pub mod workloads;
