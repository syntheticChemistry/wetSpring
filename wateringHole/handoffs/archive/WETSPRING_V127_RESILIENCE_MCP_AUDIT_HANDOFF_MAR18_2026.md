# wetSpring V127 Handoff — IPC Resilience, MCP Tools, Audit Debt Resolution

**Date:** 2026-03-18
**From:** wetSpring V127
**To:** toadStool, Squirrel, biomeOS, petalTongue, barraCuda, all springs

---

## What Changed

### 1. IPC Resilience (`ipc::resilience`)

New `RetryPolicy` and `CircuitBreaker` following sweetGrass IPC resilience patterns:

- `RetryPolicy::quick()` — 3 attempts, 100ms initial backoff, exponential growth
- `RetryPolicy::standard()` — 5 attempts, 500ms initial backoff
- Non-retriable errors (`Codec`, `RpcReject`, `SocketPath`) short-circuit immediately
- `CircuitBreaker::execute()` — failure threshold, half-open probing, cooldown
- Prevents retry storms against unhealthy primals

### 2. MCP Tool Definitions (`ipc::mcp`)

Typed MCP (Model Context Protocol) tool schemas for Squirrel AI integration:

| Tool | Maps to | Input |
|------|---------|-------|
| `wetspring_diversity` | `science.diversity` | `abundances: [f64]` |
| `wetspring_anderson` | `science.anderson` | `disorder: f64, lattice_size?: u32` |
| `wetspring_qs_model` | `science.qs_model` | `scenario: enum, dt?: f64` |
| `wetspring_alignment` | `science.alignment` | `query: str, reference: str` |
| `wetspring_kinetics` | `science.kinetics` | `model: enum, parameters: obj` |
| `wetspring_ecology_interpret` | `ai.ecology_interpret` | `query: str, context?: obj` |
| `wetspring_nmf` | `science.nmf` | `matrix: [f64], n_samples, n_features, rank` |
| `wetspring_phylogenetics` | `science.phylogenetics` | `tree1: str, tree2: str` |

- `list_tools()` — builds `tools/list` response per MCP spec
- `tool_to_method()` — maps MCP tool name → JSON-RPC method
- All tools verified against capability domain registry

### 3. Centralized Python Baseline Provenance (`provenance.rs`)

New structured provenance registry for Python-to-Rust baseline traceability:

- `PythonBaseline` struct: `binary`, `script`, `commit`, `date`, `category`
- `BaselineCategory` enum: `PythonParity`, `GpuParity`, `Analytical`, `Published`, `Visualization`
- `commits` module: canonical epoch hashes (`PYTHON_PARITY_V1`, `GPU_PARITY_V1`, `VIZ_V1`)
- `python_baselines()` — static registry of all baseline provenance records

### 4. Tolerance Constant Promotion

15 magic-number tolerances promoted to named, documented constants:

| Constant | Value | Location | Used By |
|----------|-------|----------|---------|
| `ODE_CARRYING_CAPACITY_REL` | 0.3 | `tolerances/bio/ode.rs` | `validate_soil_qs_cpu_parity` |
| `VIZ_KINETICS_WIDE` | 50.0 | `tolerances/bio/ode.rs` | `validate_petaltongue_biogas_v1` |
| `VIZ_MONOD_RATE` | 0.1 | `tolerances/bio/ode.rs` | `validate_petaltongue_biogas_v1` |
| `PHYLO_GPU_RELATIVE` | 0.10 | `tolerances/bio/phylogeny.rs` | `validate_streaming_ode_phylo` |
| `COLD_SEEP_EXTENDED_MIN_FRACTION` | 0.60 | `tolerances/bio/diversity.rs` | `validate_cold_seep_pipeline` |
| `DIVERSITY_SPECTRAL_SPEARMAN_MIN` | 0.1 | `tolerances/bio/diversity.rs` | `validate_cold_seep_pipeline` |
| `SPECTRAL_R_PIPELINE_MARGIN` | 0.05 | `tolerances/spectral.rs` | `validate_real_ncbi_pipeline` |

### 5. Math Delegation to barraCuda

- `bio::numerics::kahan_sum()` now delegates to `barracuda::shaders::precision::cpu::kahan_sum()`
- Zero local math duplication — all compensated summation uses canonical upstream

### 6. Dependency Policy Tightening

- `unlicensed = "deny"` added to `deny.toml` — no unlicensed deps allowed
- NPU device path centralized to `niche::NPU_DEFAULT_DEVICE`
- All validation binaries use the constant instead of hardcoded string

### 7. 4-Format Capability Parsing

- Extended `extract_capabilities()` to parse Format C (`method_info`) and Format D (`semantic_mappings`)
- Following airSpring/sweetGrass standard

---

## What's New for Cross-Primal Coordination

### For Squirrel

- MCP `tools/list` endpoint returns 8 typed tool definitions with JSON Schema
- `tool_to_method()` maps MCP tool calls to existing JSON-RPC dispatch
- `ai.ecology_interpret` handler unchanged — MCP adds discoverability, not behavior

### For toadStool

- `RetryPolicy` + `CircuitBreaker` prevent retry storms against unhealthy toadStool
- `DispatchOutcome<T>` (V126) fully integrated with resilience module
- `submit_outcome()` + retry + circuit breaker = production-grade dispatch

### For biomeOS

- 24 capabilities across 16 domains (unchanged)
- `health.liveness`/`health.readiness` probes remain stable
- MCP layer is additive — existing JSON-RPC contract unchanged

---

## Test Summary

- 1,443+ tests, 0 failures on non-GPU
- New: 5 MCP tests (schema validation, method mapping, capability cross-check)
- New: 4 provenance tests (baseline registry, date format, commit format)
- New: 4 IPC integration tests (diversity round-trip, readiness probes, capability list, QS model)
- Zero warnings, zero unsafe, `#![forbid(unsafe_code)]` on all crates

---

## Breaking Changes

None. All changes are additive.

---

## File Summary

| File | Change |
|------|--------|
| `barracuda/src/ipc/mcp.rs` | NEW — MCP tool definitions (8 tools) |
| `barracuda/src/ipc/mod.rs` | Added `mcp` module |
| `barracuda/src/provenance.rs` | Added Python baseline registry |
| `barracuda/src/tolerances/bio/ode.rs` | 3 new constants |
| `barracuda/src/tolerances/bio/phylogeny.rs` | 1 new constant |
| `barracuda/src/tolerances/bio/diversity.rs` | 2 new constants |
| `barracuda/src/tolerances/spectral.rs` | 1 new constant |
| `barracuda/src/bio/numerics.rs` | Delegated `kahan_sum` to barraCuda |
| `barracuda/src/niche.rs` | Added `NPU_DEFAULT_DEVICE` constant |
| `deny.toml` | Added `unlicensed = "deny"` |
| `barracuda/tests/ipc_roundtrip.rs` | NEW — IPC integration tests |
| 6 validation binaries | Updated to use named tolerance constants |
| 3 validation binaries | Updated to use `niche::NPU_DEFAULT_DEVICE` |
