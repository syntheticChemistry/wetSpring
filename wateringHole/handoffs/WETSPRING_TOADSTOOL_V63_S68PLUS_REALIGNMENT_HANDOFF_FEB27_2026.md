# wetSpring → ToadStool/BarraCuda Handoff V63

**Date:** February 27, 2026
**From:** wetSpring (life science biome)
**To:** ToadStool/BarraCuda evolution team
**Supersedes:** V62 (archived)
**ToadStool pin:** `e96576ee` (S68+: device-lost resilience + dispatch semaphore)

---

## Executive Summary

wetSpring completed V63 — a ToadStool S68+ realignment. Three commits landed
in ToadStool since our S68 pin (`f0feb226`): CPU feature-gate fix, root doc
cleanup, and GPU device-lost resilience. wetSpring rewired `GpuF64::is_lost()`
to surface the new `WgpuDevice::is_lost()` API and updated IPC dispatch to
automatically fall back to CPU when the GPU device is lost. All 6 key
validation binaries pass against the new ToadStool HEAD. GEMM benchmark
matrix size increased from 64×64 to 256×256 to absorb `submit_and_poll`
per-dispatch overhead introduced by the device-lost resilience layer.

**209 experiments, 5,021+ checks, 1,103 tests, all PASS, clippy clean.**

---

## Part 1: ToadStool S68+ Changes Acknowledged

### `89356efa` — CPU Feature-Gate Regression Fix

- `#[cfg(feature = "gpu")]` added to `wgsl_hessian_column()` in
  `numerical/mod.rs` and `WGSL_BOOTSTRAP_MEAN_F64`/`WGSL_HISTOGRAM` in
  `stats/mod.rs`
- **Impact on wetSpring:** None — wetSpring does not reference these WGSL
  constants directly

### `92679172` — Root Docs Cleaned

- 5 stale scripts archived to `ecoPrimals/fossil/`
- 4 old docs deleted (biomeOS integration guide, error code guides)
- **Impact on wetSpring:** None

### `e96576ee` — GPU Device-Lost Resilience (Major)

New `WgpuDevice` infrastructure:

| Addition | Purpose |
|----------|---------|
| `lost: Arc<AtomicBool>` | Flag set when GPU reports device-lost |
| `gpu_lock: Arc<Mutex<()>>` | Serializes submit+poll across threads |
| `DispatchSemaphore` | Concurrency budget per device type (CPU=2, iGPU=4, dGPU=8) |
| `install_error_handler()` | Device-lost → flag + warning (no panic) |
| `submit_and_poll()` | Centralized submit with `catch_unwind` for device-lost |
| `is_lost()` | Public: check if device is lost |
| `lock()` | Public: manual GPU serialization |
| `acquire_dispatch()` | Public: acquire concurrency permit |
| `max_concurrent_dispatches()` | Public: query concurrency budget |

**589 files** migrated from `queue.submit + device.poll` to `submit_and_poll`.
`from_existing_simple()` deprecated in favor of `from_existing()` with real
`AdapterInfo`.

---

## Part 2: wetSpring Rewiring

### `GpuF64::is_lost()`

New method in `barracuda/src/gpu.rs` delegates to `WgpuDevice::is_lost()`:

```rust
pub fn is_lost(&self) -> bool {
    self.wgpu_device.is_lost()
}
```

### IPC Dispatch Device-Lost Awareness

`barracuda/src/ipc/dispatch.rs`:

- `try_gpu()` now filters out lost GPU contexts via `.filter(|g| !g.is_lost())`
- When GPU is initialized but lost, `health.check` reports `"substrate": "gpu_lost"`
- All GPU-dispatched methods (diversity, Anderson) automatically fall back to
  CPU when device is lost

### Benchmark GEMM Fix

`barracuda/src/bin/benchmark_cross_spring_s68.rs`:

- Matrix size increased from 64×64 to 256×256 so compute dominates over
  `submit_and_poll` synchronization overhead
- 5-iteration warm-up added before timing loop to amortize GPU clock ramp
  and dispatch semaphore initialization

---

## Part 3: Revalidation Results

All key binaries green against ToadStool `e96576ee`:

| Binary | Experiment | Checks | Result |
|--------|-----------|--------|--------|
| `validate_barracuda_cpu_v11` | Exp206 | 64/64 | PASS |
| `validate_barracuda_gpu_v4` | Exp207 | 54/54 | PASS |
| `validate_metalforge_v7_mixed` | Exp208 | 75/75 | PASS |
| `validate_cold_seep_pipeline` | Exp185 | 10/10 | PASS |
| `benchmark_cross_spring_s68` | Exp189 | 28/28 | PASS |
| `validate_pure_gpu_pipeline` | Exp075 | 31/31 | PASS |
| `cargo test --release` | Unit tests | 20/20 | PASS |
| `cargo clippy --features gpu,ipc` | Lint | 0 warnings | CLEAN |

`submit_and_poll` executes transparently inside all ToadStool GPU ops used by
wetSpring (FusedMapReduceF64, BrayCurtisF64, BatchedOdeRK4F64, GemmF64, etc.).
No changes needed to any bio module — the resilience layer is internal to
ToadStool.

---

## Part 4: Recommendations (Carried Forward from V62)

### High Priority

1. **GPU Lanczos kernel** — `barracuda::spectral::lanczos` is still the primary
   CPU bottleneck in GPU validation. A WGSL Lanczos tridiagonalization would
   eliminate the ~457ms CPU cost for Anderson spectral analysis. wetSpring
   and hotSpring both use this heavily.

2. **Document `erf` precision** — `barracuda::special::erf` achieves ~5e-7
   (Abramowitz & Stegun polynomial), not machine epsilon. Add a doc comment
   noting the accuracy bound. Springs should use `tolerances::ERF_PARITY`
   not `tolerances::ANALYTICAL_F64`.

3. **ESN absorption** — `bio::esn` is stable, well-tested, and used by both
   wetSpring (6 classifiers) and neuralSpring. Single-file extraction.

### Medium Priority

4. **Dispatch threshold API** — `DispatchSemaphore` (S68+) manages concurrency
   budget per device type. A complementary workload-size threshold API would
   help Springs decide GPU vs CPU per method call (wetSpring's
   `GpuF64::dispatch_threshold()` pattern).

5. **`OnceLock<GpuF64>` pattern** — wetSpring's lazy GPU init pattern could
   be upstreamed as a `ToadStool::lazy_context()` helper. Springs that
   intermittently need GPU would benefit.

6. **NPU DMA bridge** — `npu.rs` is hardware-validated (AKD1000) and could
   serve as the reference NPU integration for other Springs.

### Low Priority

7. **BatchReconcileGpu** — reconciliation_gpu compose wrapper depends on
   `TreeInferenceGpu` stability. Not blocking.

8. **`io::nanopore` upstream** — waiting for MinION hardware validation.
   POD5/NRS parser is sovereign and dependency-free.

### New Observations (V63)

9. **`submit_and_poll` overhead** — Per-dispatch overhead is measurable on
   small workloads (64×64 GEMM: cached dispatch 2.5ms vs first dispatch
   0.6ms). For small matrices, the `catch_unwind` + `gpu_lock` + semaphore
   acquisition cost can exceed the compute. Springs doing many small GPU
   dispatches may want a batched submission API that amortizes the overhead.

10. **`from_existing_simple` deprecation** — wetSpring already uses
    `from_existing()` with real `AdapterInfo`. No migration needed, but
    other Springs should be notified.

---

## Part 5: Code Quality Summary

| Check | Status |
|-------|--------|
| Clippy (pedantic + nursery) | CLEAN (0 warnings, all features, all targets) |
| Tests | 1,103 (977 lib + 60 integration + 19 doc + 47 forge) |
| Coverage | 95.46% line / 93.54% fn / 94.99% branch |
| Unsafe code | 0 production |
| TODO/FIXME | 0 |
| Named tolerances | 92 (all scientifically justified) |
| Local WGSL | 0 (fully lean) |
| Passthrough | 0 (all promoted) |
| ToadStool alignment | S68+ (`e96576ee`) |

---

## Appendix: Files Changed in V63

```
barracuda/src/gpu.rs                    — GpuF64::is_lost() surfaced
barracuda/src/ipc/dispatch.rs           — device-lost awareness in try_gpu()/health
barracuda/src/bin/benchmark_cross_spring_s68.rs — GEMM 64→256, warm-up added
```
