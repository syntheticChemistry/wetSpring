# wetSpring → ToadStool/BarraCuda V85 Vault & NUCLEUS Evolution Handoff

**Date:** March 1, 2026
**From:** wetSpring V85 (Exp256-259, 6,656+ checks, 1,223 Rust tests, 975 barracuda lib)
**To:** ToadStool/BarraCuda team
**Supersedes:** V84 Pipeline Buildout Handoff (archived)

---

## What Happened in V85

V85 extended the V84 science pipeline (32 papers, 26 CPU / 21 GPU domains, Python
parity proven, 6-stage streaming) with four new experiments:

| Exp | Name | Checks | Result |
|-----|------|--------|--------|
| 256 | EMP-Scale Anderson Atlas | 35 | PASS — 30,002 samples, 14 biomes, 55ms CPU |
| 257 | NUCLEUS Data Pipeline | 9 | PASS — three-tier routing (NestGate → Songbird → sovereign) |
| 258 | NUCLEUS Tower-Node | 13 | PASS — all 6 primals READY, IPC 3.2× overhead, bit-identical |
| 259 | Genomic Vault | 30 | PASS — organ model (consent, encrypted storage, Merkle provenance) |

**Totals:** 260 experiments, 6,656+ validation checks, 1,223 tests (975 barracuda lib),
223+ binaries, clippy pedantic CLEAN (both crates, ZERO warnings).

---

## New Vault Module

wetSpring V85 introduced `barracuda/src/vault/` — a genomic data sovereignty module
that treats genomic data like a personal organ (it belongs to the individual).

### Architecture

```
vault/
├── mod.rs          — module root, primal integration targets
├── consent.rs      — ConsentTicket protocol (scope, sensitivity, expiry, revocation)
├── provenance.rs   — ProvenanceChain (append-only Merkle-linked audit log)
└── storage.rs      — VaultStore (encrypted blobs, content-addressed, consent-gated)
```

### Key Types

| Type | Purpose | Absorb Target |
|------|---------|---------------|
| `ConsentTicket` | Time-bounded, scope-limited, revocable authorization | Songbird `ConsentManager` |
| `ConsentScope` | Sensitivity hierarchy (DiversityAnalysis < FullPipeline < ReadRawSequences) | biomeOS Neural API routing |
| `ProvenanceChain` | Append-only Merkle chain — who, what, when, where, under what consent | NestGate CAS |
| `ProvenanceEntry` | Single audit record (hash, parent, timestamp, actor, consent_ticket_id, signature) | BearDog signing |
| `VaultBlob` | Encrypted data with content-addressed hash | NestGate storage |
| `VaultStore` | In-memory vault combining consent, encryption, provenance | biomeOS NUCLEUS orchestration |

### Test Coverage

- 20 unit tests in `barracuda/src/vault/` (consent: 6, provenance: 5, storage: 9)
- 30 experiment checks in Exp259 (`validate_genomic_vault`)
- All clippy pedantic CLEAN

---

## Absorption Targets — What ToadStool/BarraCuda Needs

### 1. Sovereign Cryptography → BearDog

The vault currently uses sovereign XOR-rotate cipher and simple hash functions.
These are placeholders that must absorb BearDog primitives:

| Current Sovereign | BearDog Target | Capability |
|-------------------|---------------|------------|
| XOR-rotate cipher | ChaCha20-Poly1305 AEAD | `crypto.encrypt` / `crypto.decrypt` |
| Simple hash (multiply-XOR) | BLAKE3 | `crypto.hash` |
| 64-byte zero signature | Ed25519 | `crypto.sign` / `crypto.verify` |
| In-memory key | X25519 key exchange + HKDF | `crypto.key_exchange` |

The vault also has a future path to post-quantum: ML-KEM for key encapsulation,
ML-DSA for signatures (BearDog already implements both).

### 2. Neural API Capabilities — New Routes Needed

| Capability | Method | Description |
|------------|--------|-------------|
| `vault.store` | Store encrypted blob with consent ticket | VaultStore::store |
| `vault.retrieve` | Consent-gated retrieval with provenance recording | VaultStore::retrieve |
| `vault.consent` | Issue/revoke consent tickets | ConsentTicket::new / revoke |
| `vault.provenance` | Query provenance chain for a blob | ProvenanceChain::entries_for |
| `science.diversity_batch` | Array of count vectors → array of results | Batch dispatch for EMP-scale |

### 3. Storage Evolution → NestGate

| Current | NestGate Target | Notes |
|---------|----------------|-------|
| In-memory `HashMap` | Content-addressed blob store | Westgate ZFS backend |
| Content hash as key | BLAKE3 CAS with BearDog signing | Tamper-evident |
| Per-node provenance | Replicated provenance across gates | Merkle chain replication |

### 4. Batch Science Dispatch → ToadStool GPU

EMP Atlas (Exp256) processes 30,002 samples. The current CPU path takes 55ms.
ToadStool can accelerate this:

| Workload | Current | ToadStool Target |
|----------|---------|-----------------|
| Shannon H' × 30K | CPU sequential | GPU fused diversity kernel |
| Anderson spectral × 30K | CPU per-sample | GPU batched (L, W, seed) tuples |
| Diversity → J → W → Anderson | CPU round-trips | Unidirectional streaming (Exp255 pattern) |

---

## IPC Dispatch Findings

Exp258 measured the cost of JSON-RPC 2.0 IPC dispatch vs direct function calls:

| Metric | Direct | IPC | Overhead |
|--------|--------|-----|----------|
| Shannon H' call | 0.86µs | 2.74µs | 3.2× |
| Math error | 0 | 0 | bit-identical |
| Anderson spectral | ~500ms | ~500ms | negligible |

Conclusion: IPC overhead is acceptable for all science workloads. The serialization
boundary adds zero numerical error.

---

## NUCLEUS Deployment Status

All six primals confirmed READY on Eastgate (Exp258):

| Primal | Binary | Version | Status |
|--------|--------|---------|--------|
| biomeOS | `biomeos` | v0.1.0 | Built, NUCLEUS modes work |
| BearDog | `beardog` | installed | `~/.local/bin/` |
| Songbird | `songbird` | installed | `~/.local/bin/` |
| ToadStool | `toadstool` | installed | `~/.local/bin/`, S70+++ |
| NestGate | `nestgate` | installed | `~/.local/bin/` |
| Squirrel | `squirrel` | installed | `~/.local/bin/` |

Four NUCLEUS deployment modes validated: Tower, Node, Nest, Full.

---

## Files Created/Modified in V85

| File | Purpose |
|------|---------|
| `barracuda/src/vault/mod.rs` | Vault module root |
| `barracuda/src/vault/consent.rs` | ConsentTicket protocol |
| `barracuda/src/vault/provenance.rs` | ProvenanceChain (Merkle audit) |
| `barracuda/src/vault/storage.rs` | VaultStore (encrypted blobs) |
| `barracuda/src/bin/validate_emp_anderson_atlas.rs` | Exp256 binary |
| `barracuda/src/bin/validate_nucleus_data_pipeline.rs` | Exp257 binary |
| `barracuda/src/bin/validate_nucleus_tower_node.rs` | Exp258 binary |
| `barracuda/src/bin/validate_genomic_vault.rs` | Exp259 binary |
| `experiments/256_emp_anderson_atlas.md` | Exp256 protocol |
| `experiments/257_nucleus_data_pipeline.md` | Exp257 protocol |
| `experiments/258_nucleus_tower_node.md` | Exp258 protocol |
| `experiments/259_genomic_vault.md` | Exp259 protocol |

---

## Evolution Priority

1. **BearDog absorption** (highest) — replace sovereign cipher/hash/signing with real
   cryptography. The vault API is designed for drop-in replacement.
2. **Neural API vault routes** — register `vault.*` capabilities in biomeOS routing.
3. **NestGate CAS backend** — move from in-memory to persistent content-addressed storage.
4. **Batch GPU dispatch** — fuse 30K-sample workloads into single ToadStool launches.
5. **Songbird consent integration** — wire ConsentTicket issuance through Songbird's
   existing ConsentManager HTTP API.

---

## Reproduction

```bash
cargo test --lib -p wetspring-barracuda                                     # 975 lib tests
cargo clippy --lib -p wetspring-barracuda -- -W clippy::pedantic            # ZERO warnings
cargo run --release --bin validate_emp_anderson_atlas                        # 35 checks (Exp256)
cargo run --release --bin validate_nucleus_data_pipeline                     # 9  checks (Exp257)
cargo run --release --features ipc --bin validate_nucleus_tower_node         # 13 checks (Exp258)
cargo run --release --bin validate_genomic_vault                             # 30 checks (Exp259)
```
