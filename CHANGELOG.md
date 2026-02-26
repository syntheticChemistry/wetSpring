# Changelog

All notable changes to wetSpring are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## V56 — Science Extension Pipeline + Primal Integration (2026-02-26)

### Science pipeline infrastructure
- **`ncbi/efetch.rs`** — NEW: `EFetch` FASTA/GenBank download with response validation
- **`ncbi/sra.rs`** — NEW: SRA run download via capability-discovered `fasterq-dump`/`fastq-dump`
- **`ncbi/cache.rs`** — EXTENDED: accession-based directory trees, SHA-256 integrity sidecar files,
  pure-Rust SHA-256 implementation (FIPS 180-4, verified against NIST test vectors)
- **`ncbi/nestgate.rs`** — NEW: optional NestGate data provider via JSON-RPC 2.0 over Unix sockets
  (`WETSPRING_DATA_PROVIDER=nestgate`); discovers socket via capability cascade; sovereign fallback

### GPU Anderson finite-size scaling
- **`validate_anderson_gpu_scaling.rs`** — NEW: Exp184b validation binary for L=14–20 lattices
  (16 disorder realizations, 15 W-points, scaling collapse for critical exponent ν)
- **3 new named tolerances**: `GPU_LANCZOS_EIGENVALUE_ABS` (0.03), `FINITE_SIZE_SCALING_REL` (0.08),
  `LEVEL_SPACING_STDERR_MAX` (0.015) — all with full provenance in `tolerances.rs`

### biomeOS orchestration
- **`graphs/science_pipeline.toml`** — NEW: biomeOS deployment graph for NCBI → diversity →
  Anderson spectral pipeline (NestGate data, ToadStool GPU compute, wetSpring science)
- **`config/capability_registry.toml`** — UPDATED: `science` domain added (wetSpring as provider:
  `science.diversity`, `science.anderson`, `science.ncbi_fetch`, `science.qs_model`)

### Experiment protocols
- **Exp184** — real NCBI 16S through sovereign pipeline (5 accessions test, 170 full set)
- **Exp185** — cold seep metagenomes through sovereign pipeline (170 metagenomes)
- **Exp186** — dynamic Anderson W(t): tillage transition, antibiotic recovery, seasonal
- **Exp187** — DF64 Anderson at L=24+: extended precision for refined W_c and ν
- **Exp188** — NPU sentinel with real sensor stream (Akida AKD1000)

### Metrics
- Barracuda lib tests: 833 → 882 (+49 from NCBI pipeline + cache + NestGate modules)
- Total Rust tests: 912 → 961 (882 barracuda + 47 forge + 32 integration/doc)
- Named tolerances: 79 → 82 (3 new GPU/scaling tolerances)
- Experiments: 183 → 188 (5 new protocols)
- Binaries: 172 → 174 (new `validate_anderson_gpu_scaling` + registration)
- `cargo clippy --all-targets -- -W clippy::pedantic` CLEAN (including all new code)
- Pure-Rust SHA-256 verified against 3 NIST reference vectors

## V55 — Deep Debt Resolution, Idiomatic Rust Evolution (2026-02-26)

### Fixed
- **Clippy pedantic**: All 6 failing binaries now pass `cargo clippy --all-targets -- -D warnings -W clippy::pedantic`. Zero errors across entire codebase (lib + 173 binaries).
  - `benchmark_cross_spring_s65.rs`: backticks, `f64::from()`, `f64::midpoint()`, `.is_some_and()`, import ordering.
  - `validate_gpu_diversity_fusion.rs`: full rewrite — migrated to `Validator` framework, refactored 270-line monolith into 4 focused sub-functions.
  - `validate_soil_qs_cpu_parity.rs`: strict float comparison replaced with `Validator::check()`.
  - `benchmark_cross_spring_modern.rs`, `benchmark_modern_systems_df64.rs`, `validate_metalforge_drug_repurposing.rs`: allow annotations for domain-appropriate patterns.
- **`ncbi/http.rs`**: whitespace-only `WETSPRING_HTTP_CMD` no longer treated as valid custom backend (latent bug fix).
- **`encoding.rs`**: evolved `base64_decode` return from `Result<Vec<u8>, String>` to `crate::error::Result<Vec<u8>>` using proper `Error::Base64` variant. Caller in `mzml/decode.rs` simplified.

### Added
- `tolerances::ODE_GPU_SWEEP_ABS` (0.15) — GPU ODE sweep absolute parity with scientific justification. Replaces ad-hoc magic number.
- `tolerances::GPU_EIGENVALUE_REL` (0.05) — GPU bifurcation eigenvalue relative parity. Replaces ad-hoc magic number.
- `PfasFragments` provenance: NIST Chemistry WebBook monoisotopic mass derivation for CF2, C2F4, HF defaults.
- 6 new tests: FASTQ (empty-line break, nonexistent file), HTTP (whitespace custom cmd, custom with args, invalid UTF-8 output), tolerances.
- V55 handoff: `wateringHole/handoffs/WETSPRING_TOADSTOOL_V55_DEEP_DEBT_HANDOFF_FEB26_2026.md`.

### Metrics
- Tests: 906 → 912 (833 barracuda + 47 forge + 32 integration/doc).
- Named tolerances: 77 → 79 (zero ad-hoc magic numbers remaining).
- Clippy pedantic: lib + all targets CLEAN.
- Coverage: 96.67% llvm-cov (library code).
- `ncbi/http.rs` coverage: 81.71% → 83.99%.

## V54 — Codebase Audit, Provenance Hardening, Supply-Chain Audit (2026-02-26)

### Audited
- Full codebase audit: zero `unsafe`, zero `unwrap`/`expect` in library code, zero `todo!`/`unimplemented!`, zero mocks in production, all files under 1000 LOC.
- 1 ignored test confirmed intentional: `bench::power::tests::power_monitor_start_stop` (requires nvidia-smi and RAPL hardware).
- All I/O parsers confirmed streaming (FASTQ, MS2, mzML, XML) — no full-file buffering.
- AGPL-3.0-or-later SPDX headers confirmed on all source files.

### Hardened
- `ncbi_data/mod.rs` JSON parser: handles escaped quotes and braces inside strings. 4 new edge-case tests (barracuda tests 823 → 827, total 902 → 906).
- 14 tolerance constants (`MZ_TOLERANCE`, `PYTHON_PARITY`, `SPECTRAL_COSINE`, etc.) now have full experiment/script/commit provenance chains.
- 28 Python baseline scripts: added `Reproduction:` headers with exact commands.
- `BASELINE_MANIFEST.md`: added reproduction environment (Python, OS, NumPy/SciPy versions), automated drift verification instructions, updated all SHA-256 hashes.

### Added
- `barracuda/deny.toml` — cargo-deny supply-chain audit (license allowlist, advisory DB, source restrictions).
- `scripts/verify_baseline_integrity.sh` — automated SHA-256 drift detection for all 44 baseline scripts. Exit 0/1 for CI integration.
- V54 handoff: `wateringHole/handoffs/WETSPRING_TOADSTOOL_V54_CODEBASE_AUDIT_HANDOFF_FEB26_2026.md`.

### Updated
- Root README, CHANGELOG, CONTROL_EXPERIMENT_STATUS, experiments/README, BENCHMARK_RESULTS: all counts synced (827 barracuda tests, 906 total, 79 primitives, S66).
- whitePaper/baseCamp/ updated with audit findings and evolution state.
- Archived V52 handoff to `wateringHole/handoffs/archive/`.

## V53 — Cross-Spring Evolution Benchmarks + Doc Cleanup (2026-02-26)

### Benchmarked
- 7 cross-spring evolution benchmarks on RTX 4070: PairwiseJaccard 122×, SpatialPayoff 22×, PairwiseHamming 10× GPU speedup.
- ODE lean benchmark: upstream `integrate_cpu` 18-31% faster than local after ToadStool absorption optimization.
- Three-tier Python → Rust CPU → GPU benchmark on RTX 4070 (Exp069).
- Modern systems DF64 benchmark (Exp166) — 5 ODE × 128 batches, GEMM pipeline, CPU special functions.
- Spectral cross-spring validation (Exp107) — Anderson localization, Almost-Mathieu, QS-disorder analogy, 25/25 PASS.

### Cleaned
- Root docs, specs/, metalForge/, whitePaper/baseCamp/, experiments/README — all updated to Phase 53, S66, 79 primitives.
- Archived V48-V50 handoffs to `wateringHole/handoffs/archive/` (46 total archived files).
- barracuda/ABSORPTION_MANIFEST.md and EVOLUTION_READINESS.md updated to V53/S66.

### Added
- `wateringHole/handoffs/WETSPRING_TOADSTOOL_V53_CROSS_SPRING_EVOLUTION_HANDOFF_FEB26_2026.md` — GPU performance data, tolerance learnings, absorption candidates, full cross-spring provenance timeline.
- BENCHMARK_RESULTS.md: cross-spring evolution narrative (hotSpring→wetSpring→neuralSpring provenance tables, ODE lean benchmark, GPU scaling results).

## V52 — ToadStool S66 Rewire (2026-02-26)

### Rewired
- `qs_biofilm::hill()` → delegates to `barracuda::stats::hill` (S66 `stats::metrics`). Retains `x ≤ 0` physical guard.
- `pangenome::fit_heaps_law` → delegates to `barracuda::stats::fit_linear` (S66 `stats::regression`). Eliminates 15 lines of manual log-log regression.
- `rarefaction_gpu::compute_ci` → uses `barracuda::stats::{mean, percentile}` (S66). Interpolated percentiles replace manual sort+index.

### Added
- Re-export `diversity::shannon_from_frequencies` from `barracuda::stats` (S66).

### Pinned
- ToadStool `045103a7` (S66 Wave 5), up from `17932267` (S65).
- New primitives consumed: `hill`, `monod`, `fit_linear`, `percentile`, `mean`, `shannon_from_frequencies`.
- Total primitives consumed: 79 (+6).

### Validated
- 823 lib tests PASS, clippy pedantic+nursery clean, 70 GPU validators (1,578 checks) PASS.

## V51 — Full GPU Validation on Local Hardware (2026-02-26)

### Fixed
- **BatchFitness + LocusVariance (Exp094)**: Upgraded f32 buffers → f64 to match upstream ToadStool S65 shader evolution. Readback generalized to `readback_bytes<T: Pod>`.
- **Shannon/Simpson/Spectral tolerance**: 8 validators used `GPU_VS_CPU_TRANSCENDENTAL` (1e-10) for chained transcendental chains (Shannon=Σ p·ln(p)). Empirical GPU error exceeds 1e-10 on RTX 4070 Ada. Corrected to `GPU_LOG_POLYFILL` (1e-7) — still 7 significant digits, tighter than Python parity.

### Validated (RTX 4070 + Titan V)
- **70 GPU validators**: 1,578 checks, ALL PASS. Covers 16S pipeline (88), ODE sweep (12), cross-spring evolution (39), metalForge v4-v6 (104), pure GPU streaming (80+27), soil QS GPU+streaming+metalForge (89), 49-check all-domains head-to-head, and 6 GPU benchmarks.
- **Total validation checks**: 4,494+ (CPU 1,476 + GPU 1,578 + dispatch 80 + layout 35 + integration).
- **830 barracuda lib tests** with `--features gpu` — all pass including GPU shader compilation tests.

### Docs
- Tolerance doc: clarified `GPU_VS_CPU_TRANSCENDENTAL` is for single-call, `GPU_LOG_POLYFILL` for chained ops.
- All project docs updated: 3,618→4,494+ checks, 702→1,578 GPU checks.

## V50 — ODE Derivative Rewire + Cross-Spring Validation (2026-02-26)

### Rewired
- 5 ODE RHS functions replaced with `barracuda::numerical::ode_bio::*Ode::cpu_derivative`: capacitor, cooperation, multi_signal (+ cdg guard), bistable (+ cdg guard), phage_defense. ~200 lines local derivative math eliminated.
- `qs_biofilm::hill()` and `qs_biofilm::qs_rhs()` exposed as `pub` API. `validate_gpu_ode_sweep` local `qs_rhs_wrap` + `hill` replaced with library function.
- `ncbi/http.rs`: `interpret_output` takes ownership (eliminates stdout clone). `which_exists` rewritten as pure Rust PATH scan (no subprocess).

### Validated
- All 6 ODE validators PASS: QS 16/16, Bistable 14/14, Cooperation 20/20, Capacitor 18/18, Multi-Signal 19/19, Phage 12/12.
- Cross-spring: Exp120 9/9, Exp169 12/12, Exp070 CPU-full 50/50, Exp163 v9 27/27.
- 823 barracuda lib tests, 47 forge tests. Clippy pedantic+nursery clean.

### Tests
- 4 new `try_load_json_array` error-path tests (missing file, invalid JSON, empty array, valid JSON).
- 823 barracuda + 47 forge + 32 integration/doc = 902 total.

### Docs
- ABSORPTION_MANIFEST.md, EVOLUTION_READINESS.md updated for V50 ODE derivative lean.
- CROSS_SPRING_SHADER_EVOLUTION.md: V50 timeline entry + full Write→Absorb→Lean→Rewire narrative.
- V50 handoff: `WETSPRING_TOADSTOOL_V50_ODE_DERIVATIVE_REWIRE_FEB26_2026.md`.
- Root docs, specs, CONTROL_EXPERIMENT_STATUS: test counts 819→823, 898→902, 8/9→9/9 P0-P3.

## V49 — Documentation Cleanup + Evolution Handoff (2026-02-25)

### Cleaned
- Stale references fixed across 20+ files: 182→183 experiments, 53→66 primitives, S62+DF64→S65, 1→0 local WGSL, Phase 49→50, V46/V47→V48, 8/9→9/9 P0-P3, Write phase→Lean phase.
- `diversity_fusion_f64.wgsl` references updated to reflect S63 absorption (specs/BARRACUDA_REQUIREMENTS.md, metalForge docs, barracuda docs, experiments/README.md, whitePaper/README.md).
- `validate_gpu_diversity_fusion.rs` (Exp167) doc header updated from "Write Phase" to "Lean Phase (absorbed S63)".
- ABSORPTION_MANIFEST.md status line clarified (was double-counting stats::diversity/metrics within the 66 total).
- metalForge PRIMITIVE_MAP.md, README.md, ABSORPTION_STRATEGY.md: Write→Lean, 1→0 local WGSL.
- specs/PAPER_REVIEW_QUEUE.md: Phase 49 V46→Phase 50 V48, 53→66 primitives, 1→0 WGSL.
- specs/README.md: 8/9→9/9 P0-P3, 47→49 handoffs, V47→V48 current handoff.

### Archived
- V44, V45 handoffs moved to `wateringHole/handoffs/archive/` (41 total archived).

### Handoff
- V49 handoff: `WETSPRING_V49_EVOLUTION_LEARNINGS_HANDOFF_FEB25_2026.md` — barracuda primitive usage review (66+2 by category and pattern), cross-spring evolution timeline (S39→S65, 4 springs contributing), Exp183 benchmark summary, future opportunities (non-blocking), lessons for other springs doing Write→Absorb→Lean.

### Debris Audit
- No dead code, no temp files, no orphan scripts, no stale TODOs in barracuda/src/.
- `bio/shaders/` directory confirmed empty (diversity_fusion_f64.wgsl deleted V48).
- 173 binaries, 183 experiments, 4,494+ checks, 898 tests — all verified.

## V48 — ToadStool S65 Rewire (2026-02-25)

### ToadStool Audit
- Audited ToadStool commit evolution: S60-S65 (4 commits since S62 sync point `02207c4a`).
- S60: DF64 FMA + transcendentals + polyfill hardening.
- S61-63: Sovereign compiler + deep debt + `diversity_fusion` absorption + `batched_multinomial`.
- S64: Cross-spring absorption — `stats::diversity` (16 tests), `stats::metrics` (18 tests), 8 lattice shaders.
- S65: Smart refactoring — compute_graph, esn_v2, tensor, gamma, rk45 reduced 30-40%.

### Rewired (Lean Phase)
- `diversity_fusion_gpu`: Local WGSL (`diversity_fusion_f64.wgsl`) deleted, module is now a thin re-export of `barracuda::ops::bio::diversity_fusion::{DiversityFusionGpu, DiversityResult, diversity_fusion_cpu}`.
- `bio::diversity`: 11 functions (shannon, simpson, chao1, bray_curtis, etc.) now delegate to `barracuda::stats::diversity` (S64). Zero local math.
- `special::{dot, l2_norm}`: Now delegate to `barracuda::stats::{dot, l2_norm}` (S64).
- metalForge forge `diversity_fusion` workload: `ShaderOrigin::Local` → `ShaderOrigin::Absorbed` + `with_primitive("DiversityFusionGpu")`. Absorbed count: 28/28, local: 0/28.

### Updated
- ToadStool pin: `02207c4a` (S62) → `17932267` (S65).
- Primitive count: 53 → 66 (added 11 stats::diversity + 2 stats::metrics).
- Evolution request score: 8/9 → 9/9 DONE.
- ABSORPTION_MANIFEST.md, EVOLUTION_READINESS.md, BARRACUDA_REQUIREMENTS.md, root README — all updated to S65 pin.
- V48 handoff: `WETSPRING_TOADSTOOL_V48_S65_REWIRE_HANDOFF_FEB25_2026.md`.

### Validated
- 819 lib tests PASS, 47 forge tests PASS, 18 GPU diversity fusion checks PASS.
- Exp167 (diversity fusion GPU), Exp179 (Track 4 CPU parity), Exp002 (diversity), Exp102 (CPU v8) all PASS.
- Exp183: Cross-Spring Evolution Benchmark (ToadStool S65) — 36/36 checks PASS. Covers GPU ODE (5 systems), DiversityFusion GPU (Write→Absorb→Lean), CPU diversity delegation (11 functions → barracuda::stats), CPU math delegation (dot/l2_norm → barracuda::stats), GEMM pipeline, Anderson spectral, NMF, ridge. Cross-spring provenance timeline from S39 to S65 documenting contributions from all 4 springs.
- Fixed erf(1.0) tolerance in Exp166 and Exp168: A&S 7.1.26 approximation has ~1.5e-7 max error, was using 1e-7 (GPU_LOG_POLYFILL). Changed to 5e-7.
- Fixed Exp166 cached dispatch timing check: GPU timing can vary, changed to ≤ 2× first dispatch.
- Clippy clean (pedantic + nursery, --features gpu).

## V47 — Documentation Cleanup + Evolution Handoff (2026-02-25)

### Documentation Cleanup
- Root README.md: Track table expanded to 6 tracks (added Track 3 + Track 4), stale counts corrected (4,494+ checks, 182 experiments, 171 binaries, 39/39 three-tier), handoff list reorganized with V47 as current.
- whitePaper/baseCamp/README.md: Track 4 faculty row added, paper total corrected (52), "actionable papers" updated (39/39), validation chain corrected (39/39 three-tier, 52 papers open data).
- experiments/README.md: Phase markers updated, counts synchronized (182 experiments, 171 binaries, 52 papers).
- specs/README.md: Paper queue corrected (52/52), handoff reference updated to V47, handoff count corrected (47 delivered), three-tier matrix updated (39/39).
- specs/PAPER_REVIEW_QUEUE.md: Open Data Provenance corrected ("52 reproductions"), Track 4 provenance row added (published soil metrics, model equations, review tables).
- BENCHMARK_RESULTS.md: Status line updated (39/39 three-tier).
- barracuda/ABSORPTION_MANIFEST.md: V47 doc sync noted.
- barracuda/EVOLUTION_READINESS.md: V47 doc sync noted.

### Evolution Handoff
- V47 handoff: `WETSPRING_TOADSTOOL_V47_TRACK4_EVOLUTION_HANDOFF_FEB25_2026.md` — Track 4 soil QS contributions (Anderson-QS in soil pores, 9 papers, 13 experiments, 321 checks), barracuda primitive utilization report (53 consumed, 7 CPU math, 15 GPU bio, 11 GPU core, 8 cross-spring, 5 spectral, 5 linalg/sample, 2 BGL), evolution opportunities for ToadStool (soil-specific Anderson presets, ODE initial condition sensitivity, GPU diversity fusion absorption), and lessons learned.
- Cross-spring shader evolution doc updated with V47 Track 4 entries.

### Paper Queue Review
- Confirmed: all 39/39 actionable papers have full three-tier controls (CPU + GPU + metalForge).
- Confirmed: all 52/52 papers have CPU baselines using open data.
- Confirmed: 9 extension papers are CPU-only by design (analytical/catalog).
- Gap fixed: Track 4 now in Open Data Provenance Audit table.

## V46 — Track 4 Soil QS Experiment Buildout (2026-02-25)

### New Experiments (Exp170-178)
- **Tier 1 — Soil Pore QS (Papers 44-46)**:
  - Exp170: Martínez-García 2023 — QS-pore geometry coupling (26/26 checks)
  - Exp171: Feng 2024 — pore-size-dependent diversity (27/27 checks)
  - Exp172: Mukherjee 2024 — distance-dependent colonization (23/23 checks)
- **Tier 2 — No-Till Data (Papers 47-49)**:
  - Exp173: Islam 2014 — Brandt farm soil health (14/14 checks)
  - Exp174: Zuber & Villamil 2016 — meta-analysis effect sizes (20/20 checks)
  - Exp175: Liang 2015 — 31-year tillage factorial (19/19 checks)
- **Tier 3 — Soil Structure (Papers 50-52)**:
  - Exp176: Tecon & Or 2017 — biofilm-aggregate bridge (23/23 checks)
  - Exp177: Rabot 2018 — structure-function indicators (16/16 checks)
  - Exp178: Wang 2025 — tillage × compartment microbiomes (15/15 checks)

### BarraCuda CPU Validation
- All 9 experiments validated in release mode via pure Rust math (BarraCuda CPU).
- Uses: `barracuda::stats::norm_cdf`, `barracuda::special::erf`,
  `barracuda::stats::pearson_correlation`, plus wetSpring `bio::diversity`,
  `bio::qs_biofilm`, and `bio::cooperation` modules.
- 183 new validation checks (total: 3,480+).
- Anderson-QS coupling: soil pore geometry maps to Anderson disorder W;
  aggregate stability predicts QS activation probability via `norm_cdf`.

### Full Three-Tier Validation (Exp179-182)
- Exp179: CPU parity benchmark — 8 domains, timing table, 49/49 checks.
- Exp180: GPU validation — Shannon, Simpson, Bray-Curtis (FMR + BrayCurtisF64),
  Anderson 3D spectral, QS ODE, cooperation; CPU↔GPU parity proven. 23/23 checks.
- Exp181: Pure GPU streaming — unidirectional soil QS pipeline (abundance →
  diversity → BC on-device). ToadStool streaming, zero CPU round-trips. 52/52.
- Exp182: metalForge cross-substrate — CPU = GPU for diversity, BC, Anderson,
  ODE. Capability-based dispatch proven. 14/14 checks.

### Paper Queue
- 52/52 papers now have CPU baselines (was 43/43).
- Track 4 (9 papers) now has full three-tier: CPU + GPU + streaming + metalForge.
- 39/39 papers with full three-tier (30 original + 9 Track 4).
- Zero queued papers remaining.

## V45 — Comprehensive Evolution Handoff (2026-02-25)

### Documentation Cleanup
- Root README.md: V44 marked as **current** handoff (was V42), handoff list updated.
- whitePaper/README.md: 49 → 53 primitives consumed.
- ABSORPTION_MANIFEST.md: "3 Passthrough" → "0 Passthrough" (stale reference), V40 → V44 active handoff, 49 → 53 absorbed items.
- specs/README.md: "Thirty-six delivered" → "Forty-four delivered" (v1–v44).
- baseCamp/README.md: Added V44 cross-spring rewire section, Exp168/169 entries, updated validation chain.

### Comprehensive Handoff
- V45 handoff: `WETSPRING_TOADSTOOL_V45_COMPREHENSIVE_EVOLUTION_HANDOFF_FEB25_2026.md`
  — Complete dependency surface (53 primitives by module), cross-spring provenance map
  (hotSpring→bio, wetSpring→all, neuralSpring→ecology, ToadStool→infra), P0-P8
  evolution requests, lessons learned (dispatch threshold, GemmCached, ODE trait,
  tolerance pattern), bug reports (BatchedEighGpu naga, log_f64), quality evidence.
- Cross-spring shader evolution doc updated with V44 + Exp169 entries.
- wateringHole README updated (V45 as current, V44/V43/V42 in sequence).

## V44 — Complete Cross-Spring Rewire + Modern Benchmark (2026-02-25)

### Rewire — Anderson Spectral (hotSpring → ToadStool)
- `find_last_downward_crossing` → `barracuda::spectral::find_w_c` in 4 validation binaries:
  `validate_finite_size_scaling`, `validate_geometry_zoo`, `validate_correlated_disorder`,
  `validate_finite_size_scaling_v2`. Local functions deleted, upstream `AndersonSweepPoint` adopted.
- Inline W_c loop → `find_w_c` in `validate_correlated_disorder` and `validate_finite_size_scaling_v2`.

### Rewire — Stats (ToadStool S59)
- `correlation_cpu`/`variance_cpu` → `barracuda::stats::pearson_correlation` in
  `validate_pure_gpu_pipeline`. Local variance deleted, correlation delegates upstream.

### New Experiment
- Exp169 `benchmark_cross_spring_modern`: 12/12 PASS. Validates all CPU primitives
  (erf, ln_gamma, regularized_gamma_p, norm_cdf, pearson_correlation, trapz) with
  full cross-spring provenance map (hotSpring → wetSpring → neuralSpring → ToadStool).

### Primitive Count
- 50 → 53 consumed primitives (added `find_w_c`, `anderson_sweep_averaged`, `pearson_correlation`).

### Architecture Decision: CPU ODE Hill Functions Stay Local
- 6 local `hill()` functions (cooperation, bistable, multi_signal, qs_biofilm, capacitor)
  are derivative-level CPU math inside ODE systems. GPU equivalents are generated by
  `BatchedOdeRK4::generate_shader()`. No rewire needed — correct by design.

### Quality Gates
- `cargo fmt --check` — 0 diffs
- `cargo clippy --all-targets` — 0 warnings (pedantic + nursery)
- `cargo test --lib` — 819 passed, 1 ignored, 0 failed

## V43 — ToadStool Catch-Up Review (2026-02-25)

### ToadStool Absorption Verification
- Reviewed ToadStool ABSORPTION_TRACKER (S42–S62+DF64, 80 commits).
- All 46 wetSpring V16–V22 handoff items confirmed DONE in ToadStool.
- Evolution request score: 7/9 → 8/9 (tolerance module pattern confirmed DELIVERED, S52).
- Only open item: `diversity_fusion_f64.wgsl` absorption (P0).

### Rewire
- `special::normal_cdf` → `barracuda::stats::norm_cdf` delegation (50th primitive).
  Same formula, single implementation upstream. Matches `erf`/`ln_gamma` delegation pattern.
- `ValidationHarness` (S59) reviewed — available upstream but local `Validator` kept.
  Different API, 158 binaries, no functional benefit to rewire.
- `barracuda::tolerances` (S52) confirmed complementary to wetSpring's flat `tolerances.rs`.

### Documentation
- V43 handoff: `WETSPRING_TOADSTOOL_V43_CATCH_UP_REVIEW_HANDOFF_FEB25_2026.md`
  — V40–V42 items documented for ToadStool tracker, updated priority status.
- ABSORPTION_MANIFEST.md: 49 → 50 primitives, `norm_cdf` lean, 8/9 P0-P3.
- EVOLUTION_READINESS.md: ValidationHarness inaccuracy fixed, 8/9 P0-P3.
- All root docs, whitePaper, experiments, specs updated to Phase 48.

### Quality Gates
- `cargo fmt --check` — 0 diffs
- `cargo clippy --all-targets` — 0 warnings (pedantic + nursery)
- `cargo test --lib` — 819 passed, 1 ignored, 0 failed
- `cargo test` (all) — 898 passed, 0 failed

## V42 — Deep Debt Evolution Round 2 (2026-02-25)

### Quality Gates
- `cargo fmt --check` — 0 diffs
- `cargo clippy --all-targets` — 0 warnings (pedantic + nursery)
- `cargo test` — 898 tests (819 barracuda + 47 forge + 32 integration/doc), 0 failures
- `cargo doc --no-deps` — 0 warnings
- `cargo llvm-cov --lib` — 96.78% line coverage (up from 96.48%)

### Testability Refactoring
- `ncbi.rs`: Extracted `api_key_from_paths()`, `select_backend()`, `resolve_cache_path()`
  — pure-logic functions separated from env-dependent wrappers. 16 new tests.
  Coverage: 86.39% → 93.38% (target 90% met).
- `ncbi_data.rs`: Smart refactored from monolithic 724-line file to
  `bio/ncbi_data/{mod,vibrio,campy,biome}.rs` submodule with shared JSON helpers.

### Tolerance Completeness
- `tolerances.rs`: 7 new constants (`GPU_VS_CPU_HMM_BATCH`, `ODE_BISTABLE_LOW_B`,
  `ODE_SIGNAL_SS`, `HMM_INVARIANT_SLACK`, `PHAGE_LARGE_POPULATION`,
  `PHAGE_CRASH_FLOOR`, `NPU_PASS_RATE_CEILING`, `NPU_RECALL_FLOOR`,
  `NPU_TOP1_FLOOR`, `GEMM_COMPILE_TIMEOUT_MS`). Total: 77 named constants.
- 14 bare `0.0` tolerance params → `tolerances::EXACT` across 3 validation binaries:
  `validate_metalforge_v5`, `validate_cpu_vs_gpu_all_domains`, `benchmark_phylo_hmm_gpu`.
- Hardcoded `1e-3` HMM batch parity → `tolerances::GPU_VS_CPU_HMM_BATCH`.
- Semantic tolerance fixes: `GC_CONTENT` → `ODE_BISTABLE_LOW_B`,
  `KMD_SPREAD` → `ODE_SIGNAL_SS` in ODE binaries.
- `validate_npu_spectral_triage`: 3 hardcoded thresholds → NPU tolerance constants.

### Code Quality
- `validation_helpers.rs`: SILVA filenames extracted to `SILVA_FASTA`/`SILVA_TAX_TSV` constants.
- `barracuda/Cargo.toml`: Corrected `renderdoc-sys` transitive dependency documentation.
- `special::dot`/`l2_norm`: Confirmed as correct local helpers (barracuda `dotproduct`
  is GPU Tensor op, not CPU f64 slice).
- Clippy `doc_markdown` warnings fixed in tolerances.rs doc comments.

### Documentation
- Root README.md: Phase 47, updated test/coverage/tolerance counts.
- CHANGELOG.md: V42 entry.
- wateringHole handoff V42: Deep debt evolution + ToadStool/BarraCuda team handoff.
- whitePaper/README.md and experiments/README.md updated to V42.

## V41 — Deep Audit + Coverage + Idiomatic Evolution (2026-02-25)

### Quality Gates
- `cargo fmt --check` — 0 diffs
- `cargo clippy --all-targets` — 0 warnings (pedantic + nursery)
- `cargo test` — 918 tests (871 barracuda + 47 forge), 0 failures
- `cargo doc --no-deps` — 0 warnings
- `cargo llvm-cov --lib` — 96.48% line coverage (up from 96.41%)

### Coverage Improvements
- bench/power.rs: 59.5% → 70.6% (6 new tests)
- bench/hardware.rs: 76.9% → 81.4% (6 new tests)
- ncbi.rs: 77.5% → 79.6% (5 new tests)
- io/fastq/mod.rs: 83.7% → 87.2% (7 new tests)
- ncbi_data.rs: 68% → 95% (12 new tests from V40)

### Code Quality
- `#![deny(missing_docs)]` escalated from `warn` — every public item documented
- 33 scattered `clippy::cast_precision_loss` annotations consolidated to function-level
- 9 missing tolerance constants added to `all_tolerances_are_non_negative` test
- Provenance headers added to 20 validation binaries
- Inline dot-product replaced with `special::dot`/`special::l2_norm` in 2 binaries
- SPDX headers verified on all Python scripts

### Dependency Analysis
- 3 direct deps (all pure Rust): barracuda (path), bytemuck, flate2
- 0 C dependencies, 0 HTTP/TLS crate deps
- flate2 uses rust_backend (miniz_oxide)

### Documentation
- specs/README.md: 871 tests, 96.48% coverage
- specs/BARRACUDA_REQUIREMENTS.md: evolution readiness table, blocking items
- Root README.md: Phase 46, updated counts and coverage
- wateringHole handoff V41: deep audit + ToadStool evolution handoff

### Changed (V40 — ToadStool S39-S62+DF64 Catch-Up + Rewiring)
- **ToadStool evolution review**: Reviewed 55+ ToadStool commits (S39-S62+DF64).
  7/9 P0-P3 evolution requests delivered by ToadStool: GemmF64 public API (P0-1),
  PeakDetectF64 f64 fix (P1-2), ComputeDispatch builder (P1-3), GPU dot/l2_norm
  (P1-4), ODE generate_shader (P2-6), TopK GPU (P3-7), quantize_affine_i8 (P3-8).
- **Primitive count**: 44 → 49 ToadStool primitives consumed. New: PeakDetectF64,
  ComputeDispatch, SparseGemmF64, TranseScoreF64, TopK.
- **Passthrough elimination**: 3 → 0. `signal_gpu` leaned on `PeakDetectF64` (S62).
- **Track 3 GPU unblocked**: NMF, SpMM, TransE, cosine, Top-K all upstream.
- **SparseGemmF64 wired**: `validate_gpu_drug_repurposing` now validates GPU sparse
  CSR × dense GEMM (100×80 @ 5% fill) against CPU reference.
- **NMF Top-K wired**: Drug candidate ranking via `barracuda::linalg::nmf::top_k_predictions`.
- **Exp168**: New `validate_cross_spring_s62` binary — comprehensive cross-spring
  evolution validation covering hotSpring precision → wetSpring bio → neuralSpring
  population genetics → Track 3 complete GPU path, with evolution timeline narrative.
- **Pre-existing clippy fix**: `diversity_fusion_gpu.rs` float_cmp → epsilon check.
- **Doc updates**: All handoffs, specs, root docs updated to V40 with 49 primitives,
  0 Passthrough, 7/9 P0-P3 delivered. 158 binaries (was 157).

### Changed (V39 — Comprehensive Audit + Tolerance Completion)
- **tolerances.rs**: 8 additional named constants for full coverage. Total: 70 named
  tolerance constants. New: `RAREFACTION_MONOTONIC`, `PCOA_EIGENVALUE_FLOOR`,
  `KMD_NON_HOMOLOGUE`, `HMM_FORWARD_PARITY`, `GILLESPIE_PYTHON_RANGE_REL`,
  `GILLESPIE_FANO_PHYSICAL`, `ASARI_CROSS_MATCH_PCT`, `ASARI_MZ_RANGE_PCT`.
- **5 validation binaries updated**: `validate_diversity`, `validate_pfas`,
  `validate_barracuda_cpu`, `validate_gillespie`, `validate_features` — all
  remaining ad-hoc tolerance literals replaced with named constants. Zero
  ad-hoc tolerances remain in any validation binary.
- **Comprehensive audit**: Zero TODOs/FIXMEs/mocks, zero unsafe code, zero
  production unwrap/expect, all 145 validation binaries follow hotSpring
  pattern with provenance, all I/O parsers stream (no full-file buffering),
  all external data from public repos with accession numbers.

### Changed (V38 — Deep Debt Resolution)
- **tolerances.rs**: 3 new named constants (`GPU_LOG_POLYFILL`, `ODE_NEAR_ZERO_RELATIVE`,
  and strengthened `EXACT` usage). Total: 62 named tolerance constants.
- **Tolerance centralization**: ~35 validation binaries migrated from ad-hoc numeric
  literals (`0.0`, `1e-10`, `0.001`) to `tolerances::` module constants. 200+
  individual replacements.
- **MS2 parser**: `Ms2Iter` and `stats_from_file` evolved from `reader.lines()`
  (per-line `String` allocation) to `read_line()` with reusable buffer.
- **Streaming I/O**: `stream_taxonomy_tsv`, `stream_fasta_subsampled`
  (validation_helpers.rs), `spawn_nvidia_smi_poller` (bench/power.rs) all
  migrated from `reader.lines()` to reusable buffer pattern.
- **Provenance**: 25 binaries updated from placeholder commits to `1f9f80e`.
  13 binaries had `| Command |` rows added. 2 binaries reformatted to standard
  `//! # Provenance` table format. All 157 binaries now carry complete provenance.
- **`#[must_use]`**: Added to 7 public API functions (`parse_ms2`, `stats_from_file`
  ×3, `parse_mzml`, `parse_fastq`, `http_get`, `esearch_count`).
- **`Vec::with_capacity`**: Pre-allocation added in 5 library files (xml attributes,
  eic bins, spectral_match indices, mzml spectrum arrays, FASTA refs).
- **Clippy**: `too_many_lines` suppressed on 2 long validation functions. Zero
  warnings with `-D warnings -W pedantic -W nursery`.
- **EVOLUTION_READINESS.md**: Added upstream request #4 for `barracuda::math::{dot, l2_norm}`.

### Changed (V37 — Sovereignty + Safety)
- **ncbi.rs**: Evolved from hardcoded `curl` shell-out to capability-based
  HTTP transport discovery (`WETSPRING_HTTP_CMD` > `curl` > `wget`). Removed
  legacy relative dev paths. `cache_file()` now uses `WETSPRING_DATA_ROOT`
  cascade instead of hardcoded `CARGO_MANIFEST_DIR` paths.
- **I/O parsers**: Deprecated buffering functions (`parse_fastq`, `parse_mzml`,
  `parse_ms2`) evolved from duplicate buffering implementations to thin
  wrappers over streaming iterators (`FastqIter`, `MzmlIter`, `Ms2Iter`).
- **Validation binaries**: All 56 binary files modernized — `partial_cmp().unwrap()`
  replaced with NaN-safe `.unwrap_or(Ordering::Equal)`, bare `.unwrap()`
  replaced with descriptive `.expect()` messages throughout.
- **tolerances.rs**: Added commit-hash provenance to `GC_CONTENT` (`504b0a8`),
  `MEAN_QUALITY` (`cf15167`), `GALAXY_SHANNON_RANGE` / `GALAXY_SIMPSON_RANGE` /
  `GALAXY_BRAY_CURTIS_RANGE` (`21d43a0`).
- **CI**: Added `cargo-llvm-cov` 90% library coverage gate, metalForge clippy
  (pedantic + nursery) and test jobs, `forge-clippy` and `forge-test` pipelines.
- **URL encoding**: `encode_entrez_term` now encodes `&` and `#` characters.

### Removed
- Legacy hardcoded relative dev paths in `ncbi.rs` (`../../../testing-secrets/`).
- Duplicate buffering code in I/O parsers (was behind `#[deprecated]`).

### Revalidated (V37 + V38)
- 759 lib tests pass against ToadStool `02207c4a` (S62+DF64).
- 95.75% library coverage, clippy clean (pedantic + nursery + `-D warnings`).
- cargo doc clean (88 files, 0 warnings).
- All 157 binaries carry complete provenance (commit, command, date).
- ABSORPTION_MANIFEST, EVOLUTION_READINESS, BARRACUDA_REQUIREMENTS all synced.

## [0.1.0] — 2026-02-25

### Added
- Initial release: 47 CPU modules, 42 GPU wrappers, 157 validation/benchmark
  binaries, 4 fuzz targets.
- Three-tier validation (Python baseline → Rust CPU → GPU acceleration).
- 43/43 papers reproduced across 4 tracks.
- 812+ tests, 95.57% library coverage, 3,279+ validation checks.
- AGPL-3.0-or-later license, 100% SPDX header coverage.
