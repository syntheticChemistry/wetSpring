# Exp270: biomeOS Graph Coordination — Tower/Node/Nest + Vault

| Field | Value |
|-------|-------|
| Binary | `validate_biomeos_graph` |
| Date | 2026-03-01 |
| Crate | `wetspring-forge` (metalForge) |
| Command | `cargo run -p wetspring-forge --bin validate_biomeos_graph` |
| Checks | 29/29 PASS |

## Purpose

Validates the full biomeOS coordination layer: socket discovery, primal orchestration,
vault-aware routing, and sovereign fallback. Proves that NUCLEUS atomics (Tower, Node,
Nest) correctly orchestrate mixed hardware dispatch while maintaining isolation.

## Sections

| Section | Focus | Checks |
|---------|-------|--------|
| S1 | Tower graph — substrate capability map (6 capabilities) | 6 |
| S2 | Node dispatch DAG — workload→substrate routing | 5 |
| S3 | Nest coordination — storage protocol, sovereign fallback | 3 |
| S4 | Cross-substrate pipeline topologies (3 topologies) | 8 |
| S5 | Write→Absorb→Lean evolution graph | 2 |
| S6 | Sovereign mode — zero external dependencies | 4 |

## Pipeline Topologies Validated

1. **GPU-only**: 6 stages (DADA2→Chimera→Diversity→PCoA→K-mer→KMD), 5 chained, fully streamable
2. **Mixed GPU→NPU→CPU**: 3 stages, 1 GPU transition, CPU round-trip
3. **CPU-only**: 2 stages (FASTQ parse→Assembly), 0 GPU chained

## Key Results

- S1: 4 substrates discovered with 6 unique capability types
- S2: 45/45 GPU-capable workloads routed (2 CPU-only excluded correctly)
- S3: Sovereign fallback when NestGate not running
- S5: 96% ToadStool absorption rate
- S6: Fully sovereign — wetSpring discovers, dispatches, and validates without any external primal
