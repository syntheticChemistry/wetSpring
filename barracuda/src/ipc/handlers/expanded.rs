// SPDX-License-Identifier: AGPL-3.0-or-later
//! Expanded science IPC handlers — kinetics, alignment, taxonomy, phylogenetics, NMF.
//!
//! Each handler wraps existing barracuda library functions, exposing them
//! as biomeOS `capability.call` targets. No math is duplicated.
//!
//! # Sub-modules
//!
//! | Module         | Handlers                          |
//! |-----------------|-----------------------------------|
//! | `kinetics`      | `science.kinetics`                |
//! | `alignment`     | `science.alignment`               |
//! | `taxonomy`      | `science.taxonomy`                |
//! | `phylogenetics` | `science.phylogenetics`           |
//! | `drug`          | `science.nmf` (drug repurposing) |
//! | `anderson`      | (reserved for Anderson handlers)  |

pub use super::alignment::handle_alignment;
pub use super::drug::handle_nmf;
pub use super::kinetics::handle_kinetics;
pub use super::phylogenetics::handle_phylogenetics;
pub use super::taxonomy::handle_taxonomy;
