// SPDX-License-Identifier: AGPL-3.0-or-later
//! Primal name constants — single source of truth for IPC identifiers.
//!
//! All primal names used in IPC discovery, registration, and capability
//! routing are defined here. No hardcoded primal name strings elsewhere
//! in library code.

/// This primal's canonical identifier.
pub const SELF: &str = "wetspring";

/// biomeOS orchestrator.
pub const BIOMEOS: &str = "biomeos";

/// Songbird discovery mesh.
pub const SONGBIRD: &str = "songbird";

/// `NestGate` content-addressed storage.
pub const NESTGATE: &str = "nestgate";

/// `BearDog` security foundation.
pub const BEARDOG: &str = "beardog";

/// `ToadStool` compute orchestrator.
pub const TOADSTOOL: &str = "toadstool";

/// petalTongue visualization.
pub const PETALTONGUE: &str = "petaltongue";

/// rhizoCrypt DAG.
pub const RHIZOCRYPT: &str = "rhizocrypt";

/// loamSpine commit.
pub const LOAMSPINE: &str = "loamspine";

/// sweetGrass provenance.
pub const SWEETGRASS: &str = "sweetgrass";

/// Squirrel AI assistant.
pub const SQUIRREL: &str = "squirrel";
