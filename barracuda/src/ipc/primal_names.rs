// SPDX-License-Identifier: AGPL-3.0-or-later
//! Re-exports of primal name constants for IPC code.
//!
//! The canonical definitions live in [`crate::primal_names`] (unconditionally
//! available). This module re-exports them so existing `use crate::ipc::primal_names::*`
//! call sites continue to compile without changes.

pub use crate::primal_names::BEARDOG;
pub use crate::primal_names::BIOMEOS;
pub use crate::primal_names::LOAMSPINE;
pub use crate::primal_names::NESTGATE;
pub use crate::primal_names::PETALTONGUE;
pub use crate::primal_names::RHIZOCRYPT;
pub use crate::primal_names::SELF_NAME as SELF;
pub use crate::primal_names::SONGBIRD;
pub use crate::primal_names::SQUIRREL;
pub use crate::primal_names::SWEETGRASS;
pub use crate::primal_names::TOADSTOOL;
