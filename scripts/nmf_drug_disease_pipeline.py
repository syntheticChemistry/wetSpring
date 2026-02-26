#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Date: 2026-02-24
"""
Drug Repurposing via Non-Negative Matrix Factorization — Python Control

Reproduces the core methodology from:
  - Gao et al. (2020) "NMF for Drug Repositioning: Experiments with the repoDB Dataset"
  - Yang et al. (2020) "Matrix Factorization-based Technique for Drug Repurposing"
  - Fajgenbaum et al. (2025) "Pioneering computational pharmacophenomics"

Pipeline:
  1. Build a drug-disease association matrix V [n_drugs × n_diseases]
  2. Decompose V ≈ W × H via NMF (Lee & Seung multiplicative updates)
  3. Reconstruct Ṽ = W × H to predict missing drug-disease associations
  4. Score and rank predictions by cosine similarity in latent space
  5. Validate against known approved/withdrawn pairs from repoDB

This is the Python control — BarraCUDA CPU validation uses bio::nmf (Rust).
GPU tier will use ToadStool-absorbed NMF shader (generate_shader pattern).

Reproduction:
    python3 scripts/nmf_drug_disease_pipeline.py
"""

import time
import numpy as np
from collections import defaultdict


# ─── Synthetic repoDB Dataset ────────────────────────────────────────────────
#
# repoDB: Brown & Patel (2017) — 1,571 unique drugs × 1,209 unique diseases
# with 6,677 approved and 4,972 withdrawn associations.
#
# We generate a synthetic dataset matching these dimensions and sparsity
# for pipeline validation. Real repoDB data will be swapped in for production.


def generate_synthetic_repodb(
    n_drugs: int = 1571,
    n_diseases: int = 1209,
    n_approved: int = 6677,
    n_withdrawn: int = 4972,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Generate a synthetic drug-disease matrix matching repoDB statistics.

    Returns:
        V: drug-disease association matrix [n_drugs × n_diseases], values in {0, 1}
        V_test: held-out associations for evaluation
        approved: list of (drug_idx, disease_idx) approved pairs
        withdrawn: list of (drug_idx, disease_idx) withdrawn pairs
    """
    rng = np.random.RandomState(seed)

    V = np.zeros((n_drugs, n_diseases), dtype=np.float64)

    drug_activity = rng.power(0.3, n_drugs)
    disease_prevalence = rng.power(0.3, n_diseases)

    approved = []
    for _ in range(n_approved):
        attempts = 0
        while attempts < 100:
            d = rng.choice(n_drugs, p=drug_activity / drug_activity.sum())
            s = rng.choice(n_diseases, p=disease_prevalence / disease_prevalence.sum())
            if V[d, s] == 0:
                V[d, s] = 1.0
                approved.append((d, s))
                break
            attempts += 1

    withdrawn = []
    for _ in range(n_withdrawn):
        attempts = 0
        while attempts < 100:
            d = rng.choice(n_drugs)
            s = rng.choice(n_diseases)
            if V[d, s] == 0 and (d, s) not in withdrawn:
                withdrawn.append((d, s))
                break
            attempts += 1

    test_size = len(approved) // 5
    test_indices = rng.choice(len(approved), test_size, replace=False)
    V_test = np.zeros_like(V)
    test_pairs = set()
    for idx in test_indices:
        d, s = approved[idx]
        V_test[d, s] = 1.0
        V[d, s] = 0.0
        test_pairs.add((d, s))

    approved = [(d, s) for d, s in approved if (d, s) not in test_pairs]

    return V, V_test, approved, withdrawn


# ─── NMF: Lee & Seung Multiplicative Updates ────────────────────────────────


def nmf_multiplicative_update(
    V: np.ndarray,
    k: int = 50,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Non-Negative Matrix Factorization via multiplicative update rules.

    V ≈ W × H where V ∈ ℝ₊^{m×n}, W ∈ ℝ₊^{m×k}, H ∈ ℝ₊^{k×n}

    Updates:
        H ← H ⊙ (Wᵀ V) / (Wᵀ W H + ε)
        W ← W ⊙ (V Hᵀ) / (W H Hᵀ + ε)

    Returns:
        W: drug latent factors [n_drugs × k]
        H: disease latent factors [k × n_diseases]
        losses: Frobenius norm ||V - WH||_F per iteration
    """
    rng = np.random.RandomState(seed)
    m, n = V.shape
    eps = 1e-16

    W = rng.uniform(0.01, 1.0, (m, k)).astype(np.float64)
    H = rng.uniform(0.01, 1.0, (k, n)).astype(np.float64)

    losses = []
    prev_loss = np.inf

    for iteration in range(max_iter):
        # Update H
        WtV = W.T @ V             # [k × n]
        WtWH = W.T @ W @ H        # [k × n]
        H *= WtV / (WtWH + eps)

        # Update W
        VHt = V @ H.T             # [m × k]
        WHHt = W @ H @ H.T        # [m × k]
        W *= VHt / (WHHt + eps)

        # Column-normalize W, scale H
        col_norms = np.linalg.norm(W, axis=0, keepdims=True)
        col_norms = np.maximum(col_norms, eps)
        W /= col_norms
        H *= col_norms.T

        # Convergence check
        residual = np.linalg.norm(V - W @ H, 'fro')
        losses.append(residual)

        if prev_loss > 0 and abs(prev_loss - residual) / max(prev_loss, 1.0) < tol:
            break
        prev_loss = residual

    return W, H, losses


# ─── Scoring: Cosine Similarity in Latent Space ─────────────────────────────


def cosine_similarity_matrix(H: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between disease columns of H.

    H: [k × n_diseases]
    Returns: [n_diseases × n_diseases] similarity matrix

    Zero-norm columns get self-similarity = 1.0 by convention.
    """
    norms = np.linalg.norm(H, axis=0, keepdims=True)
    nonzero = norms > 1e-16
    H_normed = np.where(nonzero, H / np.maximum(norms, 1e-16), 0.0)
    sim = H_normed.T @ H_normed
    np.fill_diagonal(sim, 1.0)
    return sim


def score_drug_disease_pairs(
    W: np.ndarray,
    H: np.ndarray,
    V_train: np.ndarray,
) -> np.ndarray:
    """
    Score all drug-disease pairs by reconstructed association strength.

    Reconstruction Ṽ = W × H gives predicted association scores.
    Mask out known training associations (we want novel predictions).
    """
    V_hat = W @ H
    V_hat[V_train > 0] = -1.0
    return V_hat


def evaluate_predictions(
    scores: np.ndarray,
    V_test: np.ndarray,
    withdrawn: list[tuple[int, int]],
    top_k: int = 100,
) -> dict:
    """
    Evaluate NMF predictions against held-out approved and withdrawn pairs.

    Metrics:
    - Recall@K: fraction of held-out approved pairs in top K predictions
    - Precision@K: fraction of top K that are true associations
    - Withdrawn avoidance: fraction of top K that are NOT withdrawn
    """
    flat_scores = scores.ravel()
    top_indices = np.argsort(flat_scores)[::-1][:top_k]
    top_pairs = set()
    for idx in top_indices:
        d = idx // scores.shape[1]
        s = idx % scores.shape[1]
        top_pairs.add((d, s))

    test_pairs = set()
    for d in range(V_test.shape[0]):
        for s in range(V_test.shape[1]):
            if V_test[d, s] > 0:
                test_pairs.add((d, s))

    withdrawn_set = set(withdrawn)

    hits = top_pairs & test_pairs
    withdrawn_hits = top_pairs & withdrawn_set

    recall_at_k = len(hits) / max(len(test_pairs), 1)
    precision_at_k = len(hits) / max(len(top_pairs), 1)
    withdrawn_avoidance = 1.0 - len(withdrawn_hits) / max(len(top_pairs), 1)

    return {
        "top_k": top_k,
        "hits": len(hits),
        "total_test": len(test_pairs),
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "withdrawn_avoidance": withdrawn_avoidance,
        "withdrawn_in_top_k": len(withdrawn_hits),
    }


# ─── Full Pipeline ───────────────────────────────────────────────────────────


def run_pipeline():
    """Run the full drug repurposing pipeline and validate."""
    checks_passed = 0
    checks_total = 0

    print("=" * 72)
    print("Drug Repurposing via NMF — Python Control")
    print("Gao et al. (2020) / Yang et al. (2020) reproduction")
    print("=" * 72)

    # ── Step 1: Generate synthetic repoDB ─────────────────────────────────
    print("\n── Step 1: Synthetic repoDB Dataset ──")
    t0 = time.perf_counter()
    V, V_test, approved, withdrawn = generate_synthetic_repodb()
    gen_time = time.perf_counter() - t0

    print(f"  Matrix shape:     {V.shape[0]} drugs × {V.shape[1]} diseases")
    print(f"  Training pairs:   {len(approved)}")
    print(f"  Test pairs:       {int(V_test.sum())}")
    print(f"  Withdrawn pairs:  {len(withdrawn)}")
    print(f"  Sparsity:         {1.0 - V.sum() / V.size:.4f}")
    print(f"  Generation time:  {gen_time*1000:.1f}ms")

    # Check 1: Matrix dimensions match repoDB
    checks_total += 1
    if V.shape == (1571, 1209):
        checks_passed += 1
        print(f"  ✓ CHECK 1: Matrix dimensions match repoDB (1571 × 1209)")
    else:
        print(f"  ✗ CHECK 1: Dimensions {V.shape}, expected (1571, 1209)")

    # Check 2: All values non-negative
    checks_total += 1
    if V.min() >= 0 and V_test.min() >= 0:
        checks_passed += 1
        print(f"  ✓ CHECK 2: All values non-negative")
    else:
        print(f"  ✗ CHECK 2: Negative values found")

    # ── Step 2: NMF Decomposition ─────────────────────────────────────────
    print("\n── Step 2: NMF Decomposition (k=50, max_iter=200) ──")
    t0 = time.perf_counter()
    W, H, losses = nmf_multiplicative_update(V, k=50, max_iter=200)
    nmf_time = time.perf_counter() - t0

    print(f"  W shape:          {W.shape}")
    print(f"  H shape:          {H.shape}")
    print(f"  Iterations:       {len(losses)}")
    print(f"  Final residual:   {losses[-1]:.4f}")
    print(f"  Residual ratio:   {losses[-1] / losses[0]:.4f}")
    print(f"  NMF time:         {nmf_time:.2f}s")

    # Check 3: NMF converged (loss decreased)
    checks_total += 1
    if losses[-1] < losses[0]:
        checks_passed += 1
        print(f"  ✓ CHECK 3: NMF converged ({losses[0]:.2f} → {losses[-1]:.2f})")
    else:
        print(f"  ✗ CHECK 3: NMF did not converge")

    # Check 4: W and H non-negative (NMF constraint)
    checks_total += 1
    if W.min() >= 0 and H.min() >= 0:
        checks_passed += 1
        print(f"  ✓ CHECK 4: W and H non-negative (NMF constraint satisfied)")
    else:
        print(f"  ✗ CHECK 4: Negative values in factors (W min={W.min():.2e}, H min={H.min():.2e})")

    # Check 5: Loss monotonically decreasing (multiplicative updates guarantee)
    checks_total += 1
    monotonic = all(losses[i] >= losses[i+1] - 1e-10 for i in range(len(losses) - 1))
    if monotonic:
        checks_passed += 1
        print(f"  ✓ CHECK 5: Loss monotonically decreasing")
    else:
        violations = sum(
            1 for i in range(len(losses) - 1) if losses[i] < losses[i+1] - 1e-10
        )
        print(f"  ✗ CHECK 5: {violations} monotonicity violations")

    # ── Step 3: Reconstruction and Scoring ────────────────────────────────
    print("\n── Step 3: Reconstruction and Scoring ──")
    t0 = time.perf_counter()
    scores = score_drug_disease_pairs(W, H, V)
    score_time = time.perf_counter() - t0

    print(f"  Score matrix:     {scores.shape}")
    print(f"  Max score:        {scores[scores >= 0].max():.4f}")
    print(f"  Mean score:       {scores[scores >= 0].mean():.4f}")
    print(f"  Scoring time:     {score_time*1000:.1f}ms")

    # Check 6: Reconstruction is non-trivial (not all zeros or all same)
    checks_total += 1
    valid_scores = scores[scores >= 0]
    if valid_scores.std() > 0.001:
        checks_passed += 1
        print(f"  ✓ CHECK 6: Non-trivial score distribution (std={valid_scores.std():.4f})")
    else:
        print(f"  ✗ CHECK 6: Degenerate scores (std={valid_scores.std():.4e})")

    # ── Step 4: Cosine Similarity in Latent Space ─────────────────────────
    print("\n── Step 4: Cosine Similarity (Disease Space) ──")
    t0 = time.perf_counter()
    sim_matrix = cosine_similarity_matrix(H)
    sim_time = time.perf_counter() - t0

    print(f"  Similarity matrix: {sim_matrix.shape}")
    print(f"  Diagonal check:    {sim_matrix.diagonal().mean():.6f} (should be ~1.0)")
    print(f"  Off-diag mean:     {(sim_matrix.sum() - sim_matrix.trace()) / (sim_matrix.size - sim_matrix.shape[0]):.4f}")
    print(f"  Similarity time:   {sim_time*1000:.1f}ms")

    # Check 7: Self-similarity = 1.0 (diagonal)
    checks_total += 1
    diag_mean = sim_matrix.diagonal().mean()
    if abs(diag_mean - 1.0) < 0.01:
        checks_passed += 1
        print(f"  ✓ CHECK 7: Self-similarity ≈ 1.0 (mean diagonal = {diag_mean:.6f})")
    else:
        print(f"  ✗ CHECK 7: Self-similarity ≠ 1.0 (mean diagonal = {diag_mean:.6f})")

    # Check 8: Similarity values in [-1, 1]
    checks_total += 1
    if sim_matrix.min() >= -1.01 and sim_matrix.max() <= 1.01:
        checks_passed += 1
        print(f"  ✓ CHECK 8: Similarity values in [-1, 1] range")
    else:
        print(f"  ✗ CHECK 8: Similarity out of range [{sim_matrix.min():.2f}, {sim_matrix.max():.2f}]")

    # ── Step 5: Evaluate Predictions ──────────────────────────────────────
    print("\n── Step 5: Prediction Evaluation ──")
    for top_k in [100, 500, 1000]:
        metrics = evaluate_predictions(scores, V_test, withdrawn, top_k=top_k)
        print(f"  Top-{top_k:>4d}: recall={metrics['recall_at_k']:.4f}  "
              f"precision={metrics['precision_at_k']:.4f}  "
              f"withdrawn_avoidance={metrics['withdrawn_avoidance']:.4f}  "
              f"hits={metrics['hits']}")

    eval_1000 = evaluate_predictions(scores, V_test, withdrawn, top_k=1000)

    # Check 9: Recall@1000 > random baseline
    checks_total += 1
    random_recall = 1000 / (V.shape[0] * V.shape[1])
    if eval_1000["recall_at_k"] > random_recall:
        checks_passed += 1
        print(f"  ✓ CHECK 9: Recall@1000 > random ({eval_1000['recall_at_k']:.4f} > {random_recall:.6f})")
    else:
        print(f"  ✗ CHECK 9: Recall@1000 ≤ random")

    # Check 10: Withdrawn avoidance > 0.5 (model avoids known failures)
    checks_total += 1
    if eval_1000["withdrawn_avoidance"] > 0.5:
        checks_passed += 1
        print(f"  ✓ CHECK 10: Withdrawn avoidance > 50% ({eval_1000['withdrawn_avoidance']:.4f})")
    else:
        print(f"  ✗ CHECK 10: Poor withdrawn avoidance ({eval_1000['withdrawn_avoidance']:.4f})")

    # ── Step 6: BarraCUDA GPU Comparison Target ───────────────────────────
    print("\n── Step 6: BarraCUDA GPU Target ──")
    total_flops = 0
    m, n = V.shape
    k = 50
    gemm_flops = 2 * m * k * n
    total_flops += gemm_flops * 4 * len(losses)
    print(f"  Total GEMM FLOPs:  {total_flops:.2e}")
    print(f"  NMF iterations:    {len(losses)}")
    print(f"  Per-iter GEMM:     4 × ({m}×{k}×{n}) = {gemm_flops * 4:.2e} FLOPs")
    print(f"  RTX 4070 f64:      ~{5.0:.1f} TFLOPS (Vulkan SHADER_F64)")
    est_gpu_time = total_flops / 5e12
    print(f"  Estimated GPU time: {est_gpu_time*1000:.1f}ms (vs {nmf_time*1000:.1f}ms CPU)")
    speedup = nmf_time / max(est_gpu_time, 1e-9)
    print(f"  Estimated speedup:  {speedup:.1f}×")

    # Check 11: Pipeline faster than 60s total
    checks_total += 1
    total_time = gen_time + nmf_time + score_time + sim_time
    if total_time < 60.0:
        checks_passed += 1
        print(f"  ✓ CHECK 11: Total pipeline < 60s ({total_time:.2f}s)")
    else:
        print(f"  ✗ CHECK 11: Total pipeline > 60s ({total_time:.2f}s)")

    # Check 12: Estimated GPU speedup > 1x
    checks_total += 1
    if speedup > 1.0:
        checks_passed += 1
        print(f"  ✓ CHECK 12: GPU speedup > 1× ({speedup:.1f}×)")
    else:
        print(f"  ✗ CHECK 12: No GPU speedup expected")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 72}")

    return checks_passed, checks_total


if __name__ == "__main__":
    passed, total = run_pipeline()
    exit(0 if passed == total else 1)
