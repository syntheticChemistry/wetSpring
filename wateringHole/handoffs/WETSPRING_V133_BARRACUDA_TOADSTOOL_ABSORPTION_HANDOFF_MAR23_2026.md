# wetSpring V133 — barraCuda / toadStool Absorption Handoff

| Field | Value |
|-------|--------|
| **Date** | 2026-03-23 |
| **From** | wetSpring **V133** |
| **To** | barraCuda + toadStool teams |
| **License** | AGPL-3.0-or-later |
| **Companion** | `WETSPRING_V133_DEEP_EVOLUTION_CROSS_ECOSYSTEM_ABSORPTION_HANDOFF_MAR23_2026.md` |

---

## What wetSpring V133 Did (Relevant to Upstream)

- **`GpuContext`** wraps `Arc<WgpuDevice>` and delegates **`TensorSession::with_device()`** — shared device, multiple tensor sessions (`barracuda/src/gpu/context.rs`).
- **`performance_surface` client** — implements report/query paths toward **toadStool `compute.performance_surface.query`** (and report); uses shared socket discovery; degrades gracefully offline.
- **`Validator`:** **`check_relative`** / **`check_abs_or_rel`** for tolerance checks aligned with groundSpring V120 patterns.
- **Zero-copy I/O:** `Arc<Path>` in MS2 parser; XML pool **`Arc<str>`** — less cloning on hot paths.
- **Feature-gate cleanup:** **7** binaries behind GPU/heavy features; **`--no-default-features`** compiles cleanly for lean CI.

---

## What wetSpring Needs from barraCuda

| Ask | Why |
|-----|-----|
| **`BatchReconcileGpu`** | `reconciliation_gpu` is still CPU passthrough; need batched GPU reconciliation to match validation story. |
| **DF64 GEMM** | Spectral cosine and related dot/GEMM workloads — double-float pipeline on FP32-heavy hardware. |
| **`BandwidthTier` in metalForge substrate** | PCIe-aware routing for cross-system dispatch (`validate_metalforge_*` paths). |
| **`ComputeDispatch` → new BGL model** | Migration guidance as bind-group layouts evolve — reduce churn in lean GPU modules. |
| **CPU Jacobi eigensolve** | `pcoa.rs` local core — parity story vs `BatchedEighGpu`; document or provide shared CPU kernel. |

---

## What wetSpring Needs from toadStool

| Ask | Why |
|-----|-----|
| **Wire `compute.performance_surface.query`** (and report) | Client is ready; need live RPCs for routing hints and measured throughput integration. |
| **`compute.route.multi_unit`** | Large bio pipelines — multi-unit routing for throughput and back-pressure. |
| **`DeviceCapabilities::latency_model()` refinements** | Better precision/routing advice for `GpuContext` / `GpuF64` decisions. |

---

## What wetSpring Can Contribute Back

- **`validate_all` meta-runner** pattern — aggregate validation entrypoint (primalSpring-style).
- **`GpuContext` / `TensorSession` wrapper** — thin bridge over `WgpuDevice` for ML-style session batching.
- **`ValidationSink` + `check_relative` / `check_abs_or_rel`** — reusable validation ergonomics.
- **`PROVENANCE_REGISTRY` pattern** — **307** validation binaries, SHA-256 discipline.
- **Smart refactor patterns** — extract modules by **domain cohesion** (e.g. `ipc/message` vs `ipc/dispatch_strategy`), not arbitrary file size.
- **Zero-copy I/O patterns** — `Arc<Path>`, `Arc<str>` pools in parsers.

---

## API Stability Notes (wetSpring → barraCuda)

wetSpring **pins** barraCuda via path dependency (`barracuda/Cargo.toml`). Stable integration surfaces in active use include:

- **`barracuda::device`:** `WgpuDevice`, `TensorContext`, **`Fp64Strategy`**, **`DeviceCapabilities`**, **`PrecisionRoutingAdvice`**, latency helpers.
- **`barracuda::session`:** **`TensorSession`**, especially **`with_device(Arc<WgpuDevice>)`**.
- **`barracuda::device::compute_pipeline`:** **`ComputeDispatch`** and related pipeline/BGL helpers across GPU bio modules.
- **`barracuda::ops`:** GPU ops — e.g. **`GemmF64`**, **`FusedMapReduceF64`**, **`BrayCurtisF64`**, ODE/`OdeSystem` codegen, **`TopK`**, diversity/signal stacks as consumed in `bio/*_gpu.rs`.
- **`barracuda::stats`**, **`barracuda::linalg`**, **`barracuda::spectral`**, **`barracuda::sample`:** heavy use in full-domain and streaming validation binaries.
- **`barracuda::shaders::Precision`** — routing with `GpuF64::optimal_precision()` / DF64 paths.

Breaking changes to the above should be staged with **semver** and a short migration note in barraCuda changelog; wetSpring will absorb via path pin + validation suite.

---

*End of V133 barraCuda / toadStool absorption handoff — 2026-03-23.*
