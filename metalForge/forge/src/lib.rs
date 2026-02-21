// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]
#![deny(clippy::expect_used, clippy::unwrap_used)]

//! wetSpring Forge — local hardware discovery and cross-substrate dispatch.
//!
//! Forge discovers what compute substrates exist on THIS machine at runtime
//! and routes life science workloads to the best capable substrate. It leans
//! on toadstool/barracuda for GPU discovery and device management, and adds
//! NPU probing and cross-substrate orchestration locally.
//!
//! # Design Principle
//!
//! wetSpring is a biome. ToadStool (barracuda) is the fungus — it lives in
//! every biome. We lean on it for what it already provides (GPU enumeration,
//! shader dispatch, buffer management), and evolve new capabilities locally
//! (NPU probing, cross-substrate routing). ToadStool absorbs what works,
//! then all Springs benefit.
//!
//! Springs don't reference each other. hotSpring doesn't import wetSpring.
//! Both lean on ToadStool independently — hotSpring evolves physics shaders,
//! wetSpring evolves bio shaders, and ToadStool absorbs both.
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
//!    └─────────────────────────────┘
//! ```
//!
//! # Life Science Workloads
//!
//! wetSpring routes bioinformatics and analytical chemistry workloads:
//! - **GPU-optimal**: Felsenstein pruning, HMM forward, spectral cosine,
//!   diversity map-reduce, ANI/SNP/dN/dS batch, RF batch inference
//! - **NPU-optimal**: Taxonomy classification, anomaly detection, PFAS screening
//! - **CPU-optimal**: FASTQ/mzML parsing, chimera detection, tree traversal

pub mod dispatch;
pub mod inventory;
pub mod probe;
pub mod substrate;
