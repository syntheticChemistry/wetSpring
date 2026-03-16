// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU vs CPU parity tolerances.
//!
//! Tolerances for comparing GPU (WGSL f64/f32) and CPU results.
//! GPU instruction reordering, FMA behavior, and parallel reduction
//! order cause small numerical differences vs sequential CPU code.

/// GPU f64 vs CPU f64 for exact arithmetic (add, mul, comparison).
///
/// GPU `SHADER_F64` uses IEEE 754 f64, but different instruction ordering
/// and FMA behavior can introduce small differences. Typical max observed
/// diff ~8e-8 for `exp()` on current hardware (hotSpring Exp001 §4.4).
///
/// For simple add/mul chains (diversity metrics), expect < 1e-10.
pub const GPU_VS_CPU_F64: f64 = 1e-6;

/// GPU f64 vs CPU f64 for log/exp operations (transcendentals).
///
/// Native WGSL `log(f64)` on NVIDIA matches CPU to ~1e-14 per call.
/// Allow 1e-10 for single transcendental evaluations (one log or exp).
/// For chained Shannon/Simpson/cosine, use `GPU_LOG_POLYFILL` (1e-7).
pub const GPU_VS_CPU_TRANSCENDENTAL: f64 = 1e-10;

/// GPU f64 log polyfill precision (software `log_f64` shader).
///
/// When native WGSL `log(f64)` is unavailable, barraCuda uses a
/// polynomial `log_f64` polyfill with ~1e-8 absolute precision.
/// Allow 1e-7 for single evaluations; accumulated chains (Shannon
/// over N species) may reach ~1e-6 covered by [`GPU_VS_CPU_F64`].
pub const GPU_LOG_POLYFILL: f64 = 1e-7;

/// GPU Bray-Curtis vs CPU: per-pair tolerance.
///
/// Each pair involves N additions and a division. For N=2000 features,
/// rounding differs by at most a few ULP per addition.
pub const GPU_VS_CPU_BRAY_CURTIS: f64 = 1e-10;

/// GPU vs CPU for stochastic ensemble statistics (bootstrap mean/var).
///
/// Parallel reduction order differs on GPU; for ensemble averages
/// over ~1000 replicates, accumulated rounding yields ~1e-4 drift.
pub const GPU_VS_CPU_ENSEMBLE: f64 = 1e-4;

/// GPU vs CPU for HMM batch forward log-likelihoods.
///
/// 256 sequences × 100 steps × 3 states: log-space additions
/// across the Forward lattice accumulate rounding differently on GPU
/// (warp-level parallel reduction) vs CPU (sequential). 1e-3 covers
/// the worst observed per-sequence drift across the batch.
/// Validated: Exp048, `benchmark_phylo_hmm_gpu`, commit `e4358c5`.
pub const GPU_VS_CPU_HMM_BATCH: f64 = 1e-3;

/// GPU f32 vs CPU f64 for integer-derived results (Hamming, Jaccard).
///
/// f32 has ~7 significant digits; operations on integer-derived values
/// (count / total) yield results exact to ~1e-6. Allow 1e-5 to cover
/// accumulated rounding in pairwise summation.
pub const GPU_F32_PARITY: f64 = 1e-5;

/// GPU f32 spatial computation tolerance (payoff, fitness, variance).
///
/// f32 grid operations (neighbor sums, dot products) accumulate
/// rounding proportional to neighborhood size. For 8-neighbor grids,
/// 1e-4 covers the worst-case f32 summation error.
pub const GPU_F32_SPATIAL: f64 = 1e-4;

/// GEMM streaming max absolute error (CPU vs GPU over full matrix).
///
/// Large GEMM (256×128×256) accumulates rounding across many dot products;
/// 1e-5 covers worst-case element-wise drift in streaming pipeline.
/// Validated: Exp227 (Pure GPU Streaming v4).
pub const GEMM_GPU_MAX_ERR: f64 = 1e-5;

/// GPU f32 `PairwiseL2` element-wise parity (f32 kernel).
///
/// `PairwiseL2` runs in f32 on GPU; element-wise max error vs CPU f64
/// reference is bounded by f32 precision (~7 digits). 1e-3 covers
/// accumulated rounding in the L2 norm reduction across feature vectors.
/// Validated: Exp230 (`BarraCuda` GPU v7), `PairwiseL2` domain.
pub const GPU_F32_PAIRWISE_L2: f64 = 1e-3;

/// metalForge transfer-time parity tolerance.
///
/// Dispatch and transfer timing metadata must agree within 1e-6 seconds
/// between observed and recorded values. Covers timer resolution and
/// scheduling jitter in the NUCLEUS pipeline.
/// Validated: metalForge mixed-hardware dispatch binaries.
pub const TRANSFER_TIME_PARITY: f64 = 1e-6;

/// ODE landscape GPU vs CPU tolerance for QS parameter sweeps.
///
/// Multi-parameter ODE landscape comparisons (e.g., Vibrio QS parameter
/// grid) produce max element-wise differences up to ~1.5 due to GPU
/// instruction reordering in stiff systems with exponential sensitivity.
/// 2.0 covers worst-case chaotic-regime drift.
/// Validated: Exp113 (Vibrio QS landscape), Exp293 (GPU v9).
pub const ODE_GPU_LANDSCAPE_PARITY: f64 = 2.0;
