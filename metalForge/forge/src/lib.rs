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
//! - **GPU (absorbed)**: Felsenstein, HMM, spectral cosine, diversity,
//!   ANI/SNP/dN/dS, RF batch, QS ODE, k-mer, `UniFrac`
//! - **GPU (local WGSL)**: Phage defense ODE, bistable QS ODE, multi-signal ODE
//! - **NPU-optimal**: Taxonomy classification, anomaly detection, PFAS screening
//! - **CPU-optimal**: FASTQ/mzML parsing, chimera detection, tree traversal
//!
//! # Write → Absorb → Lean Tracking
//!
//! The [`workloads`] module tracks shader origin for every domain:
//! - **Absorbed** (8): lean on `ToadStool` primitives
//! - **Local** (3): local WGSL shaders pending absorption
//! - **CPU-only** (2): no GPU path planned
//!
//! When `ToadStool` absorbs a local shader, update origin from `Local` to
//! `Absorbed` and rewire. This is the Lean step.

pub mod bridge;
pub mod dispatch;
pub mod inventory;
pub mod probe;
pub mod streaming;
pub mod substrate;
pub mod workloads;
