// SPDX-License-Identifier: AGPL-3.0-or-later
//! Re-export of [`crate::special`] for backward compatibility.
//!
//! The canonical location is now [`crate::special`]. This re-export
//! keeps `bio::special::*` paths working during the migration.

pub use crate::special::*;
