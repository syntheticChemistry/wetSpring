// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bio brain: concurrent monitoring pipeline adapted from hotSpring's 4-layer
//! brain architecture.
//!
//! # Cross-spring provenance
//!
//! - 4-layer brain concept: hotSpring v0.6.15 (`BIOMEGATE_BRAIN_ARCHITECTURE.md`)
//! - Attention state machine: hotSpring `AttentionState` (Green/Yellow/Red)
//! - Head-group disagreement: hotSpring `HeadGroupDisagreement` (reservoir.rs)
//! - Bio observation mapping: wetSpring V89 (`CROSS_SPRING_EVOLUTION.md`)
//!
//! # Architecture (adapted from physics → bio)
//!
//! | Layer       | Physics (hotSpring)           | Bio (wetSpring)                           |
//! |-------------|-------------------------------|-------------------------------------------|
//! | Cerebellum  | NPU AKD1000 real-time monitor | ESN sentinel on diversity stream           |
//! | Motor       | RTX 3090 CG solver / HMC      | GPU Anderson spectral on community matrix  |
//! | Pre-motor   | Titan V speculative pre-comp   | Pre-computed rarefaction / ordination       |
//! | Prefrontal  | CPU planning, ESN retrain      | CPU Nautilus training, edge detection       |

mod observation;

#[cfg(feature = "nautilus")]
pub mod nautilus_bridge;

pub use observation::{BioBrain, BioObservation, BrainStatus, DiversityUpdate};

#[cfg(feature = "nautilus")]
pub use nautilus_bridge::BioNautilusBrain;

#[cfg(test)]
mod tests;
