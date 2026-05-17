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

/// barraCuda standalone math primal.
pub const BARRACUDA: &str = "barracuda";

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

/// skunkBat audit logging — cross-primal event instrumentation.
pub const SKUNKBAT: &str = "skunkbat";

// ── Display names (human-readable, for JSON gap reports / UX) ──

/// Human-readable display name for this primal.
pub const SELF_DISPLAY: &str = "wetSpring";

/// `biomeOS` display name (for gap reports and composition JSON).
pub const BIOMEOS_DISPLAY: &str = "biomeOS";

/// `NestGate` display name.
pub const NESTGATE_DISPLAY: &str = "NestGate";

/// `ToadStool` display name.
pub const TOADSTOOL_DISPLAY: &str = "ToadStool";

/// `rhizoCrypt` display name.
pub const RHIZOCRYPT_DISPLAY: &str = "rhizoCrypt";

/// `loamSpine` display name.
pub const LOAMSPINE_DISPLAY: &str = "loamSpine";

/// `sweetGrass` display name.
pub const SWEETGRASS_DISPLAY: &str = "sweetGrass";

/// `skunkBat` display name.
pub const SKUNKBAT_DISPLAY: &str = "skunkBat";

/// `BearDog` display name.
pub const BEARDOG_DISPLAY: &str = "BearDog";

/// `Songbird` display name.
pub const SONGBIRD_DISPLAY: &str = "Songbird";

/// `petalTongue` display name.
pub const PETALTONGUE_DISPLAY: &str = "petalTongue";

/// `Squirrel` display name.
pub const SQUIRREL_DISPLAY: &str = "Squirrel";

/// `coralReef` display name.
pub const CORALREEF_DISPLAY: &str = "coralReef";

/// Legacy JSON-RPC method prefix (`{SELF_NAME}.`) on the wire — must match [`SELF_NAME`] + `.`.
pub const LEGACY_SELF_METHOD_PREFIX: &str = "wetspring.";

/// Vault encryption / KDF context string (stable; do not change — existing ciphertext depends on it).
pub const VAULT_KEY_CONTEXT: &str = "wetspring-vault-encryption-v1";

/// Default niche deploy graph path (relative to repo / packaging layout).
pub const DEPLOY_GRAPH_REL_PATH: &str = "graphs/wetspring_deploy.toml";
