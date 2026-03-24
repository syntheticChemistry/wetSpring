#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commit: 48fb787
# Date: 2026-02-23
"""
Shared tolerance constants for wetSpring Python baseline scripts.

Mirrors the Rust tolerance module at ``barracuda/src/tolerances/`` so that
Python baselines and Rust validation binaries use identical thresholds.

Usage::

    from tolerances import ANALYTICAL_F64, ODE_DIVISION_GUARD

    assert abs(computed - expected) < ANALYTICAL_F64

When adding new tolerances, define them in the Rust module first, then
mirror them here with the same name and value.
"""

# ═══════════════════════════════════════════════════════════════════
# Machine-precision tolerances (IEEE 754 f64)
# ═══════════════════════════════════════════════════════════════════

EXACT: float = 0.0
ANALYTICAL_F64: float = 1e-12
ANALYTICAL_LOOSE: float = 1e-10
LIMIT_CONVERGENCE: float = 1e-8
VARIANCE_EXACT: float = 1e-20

# NMF
NMF_SPARSITY_THRESHOLD: float = 1e-8
NMF_CONVERGENCE: float = 1e-6
NMF_CONVERGENCE_LOOSE: float = 1e-4

# Special functions
ERF_PARITY: float = 5e-7
NORM_CDF_PARITY: float = 1e-3
NORM_CDF_TAIL: float = 1e-4
NORM_PPF_KNOWN: float = 0.01

# Jacobi / PCoA
JACOBI_CONVERGENCE: float = 1e-24
JACOBI_ELEMENT_SKIP: float = 1e-15

# Numerical stability guards
MATRIX_EPS: float = 1e-15
BOX_MULLER_U1_FLOOR: float = 1e-15
ODE_DIVISION_GUARD: float = 1e-30
GAMMA_SERIES_CONVERGENCE: float = 1e-15

# ODE integration
ODE_DEFAULT_DT: float = 0.001

# DF64 / streaming
DF64_ROUNDTRIP: float = 1e-13

# Pharmacological
PHARMACOKINETIC_PARITY: float = 0.1
IC50_RESPONSE_TOL: float = 0.01
REGRESSION_FIT_PARITY: float = 0.01

# Bootstrap
RAREFACTION_BOOTSTRAP_SHANNON: float = 0.5

# ═══════════════════════════════════════════════════════════════════
# Instrument tolerances
# ═══════════════════════════════════════════════════════════════════

GC_CONTENT: float = 0.005
MEAN_QUALITY: float = 0.5
MZ_TOLERANCE: float = 0.01
MZ_FRAGMENT: float = 0.001
SPECTRAL_MZ_WINDOW: float = 0.5
PPM_FACTOR: float = 1e-6
KMD_GROUPING: float = 0.01
KMD_SPREAD: float = 0.02
KMD_NON_HOMOLOGUE: float = 0.005
EIC_TRAPEZOID: float = 0.01
PEAK_HEIGHT_REL: float = 0.01
PEAK_MIN_PROMINENCE: float = 0.05
MZ_SEARCH_EXACT: float = 1e-6
MZ_SEARCH_RELAXED: float = 1e-4
CF2_SPACING_TOL: float = 0.01
PFSA_HOMOLOGUE_WINDOW: float = 60.0
RETENTION_INDEX_MATCH: float = 0.1

# ═══════════════════════════════════════════════════════════════════
# GPU vs CPU parity
# ═══════════════════════════════════════════════════════════════════

GPU_VS_CPU_F64: float = 1e-6
GPU_VS_CPU_TRANSCENDENTAL: float = 1e-10
GPU_LOG_POLYFILL: float = 1e-7
GPU_VS_CPU_BRAY_CURTIS: float = 1e-10
GPU_VS_CPU_ENSEMBLE: float = 1e-4
GPU_VS_CPU_HMM_BATCH: float = 1e-3
GPU_F32_PARITY: float = 1e-5
GPU_F32_SPATIAL: float = 1e-4
GEMM_GPU_MAX_ERR: float = 1e-5
ODE_GPU_LANDSCAPE_PARITY: float = 2.0

# ═══════════════════════════════════════════════════════════════════
# Bio tolerances
# ═══════════════════════════════════════════════════════════════════

# Parity
EXACT_F64: float = 1e-15
PYTHON_PARITY: float = 1e-10
PYTHON_PARITY_TIGHT: float = 1e-14
PYTHON_PVALUE: float = 1e-5
ML_PREDICTION: float = 1e-10
ML_F1_SCORE: float = 1e-4

# ODE
ODE_CDG_CONVERGENCE: float = 1e-12
ODE_METHOD_PARITY: float = 1e-3
ODE_GPU_PARITY: float = 1e-6
RK45_DEFAULT_REL_TOL: float = 1e-8
RK45_DEFAULT_ABS_TOL: float = 1e-10

# Diversity
DIVERSITY_EVENNESS_TOL: float = 0.01
SHANNON_RECOVERY_TOL: float = 0.1
BRAY_CURTIS_SYMMETRY: float = 1e-15
RAREFACTION_MONOTONIC: float = 1e-10
NANOPORE_DIVERSITY_TOLERANCE: float = 0.3

# Misc bio
DADA2_ERR_CONVERGENCE: float = 1e-6
DADA2_ERROR_MODEL_PARITY: float = 1e-7
PCOA_EIGENVALUE_FLOOR: float = 1e-10
NANOPORE_CALIBRATION: float = 1e-12
NANOPORE_SIGNAL_STATS: float = 1e-10
BOUNDED_METRIC_GUARD: float = 1e-10
REGRESSION_EXACT_FIT: float = 1e-10
LAPLACIAN_ROW_SUM: float = 1e-10
DISTRIBUTION_SUM_TO_ONE: float = 1e-10
RAREFACTION_CI_GUARD: float = 1e-10
NMF_CONVERGENCE_KL: float = 1e-4
NMF_CONVERGENCE_EUCLIDEAN: float = 1e-6
NMF_CONVERGENCE_RANK_SEARCH: float = 1e-5
RIDGE_REGULARIZATION_SMALL: float = 1e-6
RIDGE_TEST_TOL: float = 1e-4
NUMERICAL_HESSIAN_EPSILON: float = 1e-5
HESSIAN_TEST_TOL: float = 1e-4
EMBEDDING_NORM_FLOOR: float = 1e-12
ODE_COOPERATOR_PERSIST_THRESHOLD: float = 0.001
ODE_CELL_GROWTH_THRESHOLD: float = 0.01
LOG_PROB_FLOOR: float = 1e-300

# Phylogeny
DNDS_OMEGA_GUARD: float = 1e-10
PHYLO_LIKELIHOOD: float = 1e-8
JC69_PROBABILITY: float = 1e-6

# Alignment
ANI_CROSS_SPECIES: float = 1e-4
EVOLUTIONARY_DISTANCE: float = 1e-3
SPECTRAL_COSINE: float = 1e-3
HMM_FORWARD_PARITY: float = 1e-6
HMM_INVARIANT_SLACK: float = 1e-10

# ESN
ESN_REGULARIZATION: float = 1e-6
ESN_REGULARIZATION_TIGHT: float = 1e-5

# Brain
BRAIN_DISAGREEMENT_ANALYTICAL: float = 1e-10
BRAIN_URGENCY_TOL: float = 0.01

# Spectral
TRAPZ_COARSE: float = 1e-6
TRAPZ_101: float = 1e-4
CROSS_SPRING_NUMERICAL: float = 1e-3
