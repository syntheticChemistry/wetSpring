// SPDX-License-Identifier: AGPL-3.0-or-later
//! Primal name constants — single source of truth for primal identifiers.
//!
//! All primal names used in IPC discovery, registration, capability routing,
//! and niche dependency manifests are defined here. Unconditionally available
//! (no feature gates) so that `niche.rs` and other non-IPC code can reference
//! primal names without hardcoding strings.

/// This primal's canonical identifier.
pub const SELF_NAME: &str = "wetspring";

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

/// coralReef sovereign shader compiler (WGSL → native ISA).
pub const CORALREEF: &str = "coralreef";

/// petalTongue visualization.
pub const PETALTONGUE: &str = "petaltongue";

/// rhizoCrypt derivation DAG — tracks content lineage and license metadata.
pub const RHIZOCRYPT: &str = "rhizocrypt";

/// loamSpine immutable ledger — certificate storage for provenance proofs.
pub const LOAMSPINE: &str = "loamspine";

/// sweetGrass provenance — W3C PROV-O attribution braids.
pub const SWEETGRASS: &str = "sweetgrass";

/// Squirrel AI assistant.
pub const SQUIRREL: &str = "squirrel";

/// Legacy JSON-RPC method prefix (`{SELF_NAME}.`) on the wire — must match [`SELF_NAME`] + `.`.
pub const LEGACY_SELF_METHOD_PREFIX: &str = "wetspring.";

/// Vault encryption / KDF context string (stable; do not change — existing ciphertext depends on it).
pub const VAULT_KEY_CONTEXT: &str = "wetspring-vault-encryption-v1";

/// Default niche deploy graph path (relative to repo / packaging layout).
pub const DEPLOY_GRAPH_REL_PATH: &str = "graphs/wetspring_deploy.toml";
