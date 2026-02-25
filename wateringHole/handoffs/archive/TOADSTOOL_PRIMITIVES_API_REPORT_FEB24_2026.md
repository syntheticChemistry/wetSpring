# Toadstool Barracuda Primitives API Report — Feb 24, 2026

**Purpose**: API signatures and usage patterns for primitives that wetSpring could wire into.  
**Source**: `/home/eastgate/Development/ecoPrimals/phase1/toadstool/crates/barracuda/`

---

## 1. `disordered_laplacian`

### Module path
```text
barracuda::linalg::disordered_laplacian
```
Re-exported from `barracuda::linalg::graph`.

### Function signature
```rust
pub fn disordered_laplacian(
    laplacian: &[f64],
    n: usize,
    heterogeneity: &[f64],
    disorder_strength: f64,
) -> Vec<f64>
```

### What it computes
Anderson-type diagonal disorder on a graph Laplacian:

- **Formula**: `L' = L + W * diag(heterogeneity - mean)`
- `heterogeneity[i]` = disorder value at node `i`
- Mean is subtracted so the disorder is centered (sum of diagonal additions is zero)
- Only diagonal entries change; off-diagonals are unchanged

### Input/output types
| Input | Type | Description |
|-------|------|-------------|
| `laplacian` | `&[f64]` | Flat row-major Laplacian (n×n) |
| `n` | `usize` | Matrix dimension |
| `heterogeneity` | `&[f64]` | Disorder values per node (at least `n` elements) |
| `disorder_strength` | `f64` | Scaling factor W |
| **Output** | `Vec<f64>` | Modified Laplacian (n×n) |

### Example usage (from tests)
```rust
// Zero disorder strength -> equals input laplacian
let l = vec![1.0, -1.0, -1.0, 1.0];
let h = vec![0.5, 1.5];
let result = disordered_laplacian(&l, 2, &h, 0.0);
assert_eq!(result, l);

// Disorder only affects diagonal
let l = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
let h = vec![0.1, 0.2, 0.3];
let result = disordered_laplacian(&l, 3, &h, 1.0);
```

### GPU (WGSL) version
**None.** CPU-only. No WGSL shader for disordered Laplacian.

---

## 2. `graph_laplacian`

### Module path
```text
barracuda::linalg::graph_laplacian
```
Re-exported from `barracuda::linalg::graph`.

### Function signature
```rust
pub fn graph_laplacian(adjacency: &[f64], n: usize) -> Vec<f64>
```

### What it computes
Graph Laplacian from adjacency matrix:

- **Formula**: `L = D - A`
- `D` = degree matrix (diagonal = row sums of A)
- `L[i,j] = degree(i)` if `i == j`, else `-A[i,j]`
- Row sums of L are zero

### Input/output types
| Input | Type | Description |
|-------|------|-------------|
| `adjacency` | `&[f64]` | Flat row-major adjacency (n×n) |
| `n` | `usize` | Matrix dimension |
| **Output** | `Vec<f64>` | Laplacian (n×n) |

### Example usage (from tests)
```rust
// 3-node path: 0--1--2
let adj = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
let l = graph_laplacian(&adj, 3);

// 2x2 complete graph
let adj = vec![0.0, 1.0, 1.0, 0.0];
let l = graph_laplacian(&adj, 2);
// L = [[1,-1],[-1,1]]
```

### GPU (WGSL) version
**Yes.** `crates/barracuda/src/shaders/linalg/laplacian.wgsl`:

- Entry point: `graph_laplacian`
- Uses `f32` (CPU uses `f64`)
- Bindings: adjacency (read), laplacian (read_write), params (uniform with `n`)
- **Not wired into high-level API** — `WGSL_LAPLACIAN` in `ops/linalg/mod.rs` is `#[allow(dead_code)]`; no GPU dispatch layer calls it.

---

## 3. `boltzmann_sampling`

### Module path
```text
barracuda::sample::boltzmann_sampling
```
Re-exported from `barracuda::sample::metropolis`.

### Function signature
```rust
pub fn boltzmann_sampling(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    initial_params: &[f64],
    temperature: f64,
    step_size: f64,
    n_steps: usize,
    seed: u64,
) -> BoltzmannResult
```

### Return type
```rust
pub struct BoltzmannResult {
    pub losses: Vec<f64>,
    pub acceptance_rate: f64,
    pub final_params: Vec<f64>,
}
```

### What it computes
Metropolis–Hastings MCMC with Boltzmann acceptance:

- Target: `exp(-loss/temperature)`
- Proposals: `x' = x + step_size * Normal(0,1)` (Box–Muller)
- Accept with `min(1, exp((current_loss - proposed_loss) / temperature))`
- Uses Xoshiro256** PRNG (seeded via splitmix64)

### Input/output types
| Input | Type | Description |
|-------|------|-------------|
| `loss_fn` | `&dyn Fn(&[f64]) -> f64` | Loss/energy function |
| `initial_params` | `&[f64]` | Starting point |
| `temperature` | `f64` | Temperature T |
| `step_size` | `f64` | Proposal scale |
| `n_steps` | `usize` | MCMC steps |
| `seed` | `u64` | RNG seed |
| **Output** | `BoltzmannResult` | `losses`, `acceptance_rate`, `final_params` |

### Example usage (from tests)
```rust
let loss_fn = |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>();
let result = boltzmann_sampling(&loss_fn, &[5.0, 5.0], 1.0, 0.5, 2000, 42);

// Lower temperature -> lower acceptance rate
let result_high_t = boltzmann_sampling(&loss_fn, &[1.0], 2.0, 0.5, 500, 123);
let result_low_t = boltzmann_sampling(&loss_fn, &[1.0], 0.1, 0.5, 500, 123);
assert!(result_high_t.acceptance_rate > result_low_t.acceptance_rate);
```

### GPU (WGSL) version
**Yes.** `crates/barracuda/src/shaders/sample/metropolis.wgsl`:

- Entry point: `metropolis_step`
- Uses `f32`
- Different design: one chain per thread, pre-computed `log_target_current`, `log_target_proposed`, `proposed` buffers
- **Not wired into high-level API** — `WGSL_METROPOLIS` in `sample/mod.rs` is referenced as “complementing” the CPU implementation; no GPU dispatch.

---

## 4. `effective_rank`

### Module path
```text
barracuda::linalg::effective_rank
```
Re-exported from `barracuda::linalg::graph`.

### Function signature
```rust
pub fn effective_rank(eigenvalues: &[f64]) -> f64
```

### What it computes
Effective rank from eigenvalue spectrum:

- **Formula**: `rank_eff = exp(H)` where `H = -sum(p_i * log(p_i))`
- `p_i = |λ_i| / sum(|λ_j|)` (normalized absolute eigenvalues)
- Shannon entropy of the spectrum

### Input/output types
| Input | Type | Description |
|-------|------|-------------|
| `eigenvalues` | `&[f64]` | Eigenvalues (or singular values) |
| **Output** | `f64` | Effective rank |

### Example usage (from tests)
```rust
// Equal eigenvalues -> full rank
let ev = vec![1.0, 1.0, 1.0, 1.0];
let r = effective_rank(&ev);
assert!((r - 4.0).abs() < 1e-10);

// One nonzero -> rank 1
let ev = vec![1.0, 0.0, 0.0];
let r = effective_rank(&ev);
assert!((r - 1.0).abs() < 1e-10);
```

### GPU (WGSL) version
**None.** CPU-only. No WGSL shader.

---

## 5. `numerical_hessian`

### Module path
```text
barracuda::numerical::numerical_hessian
```
Re-exported from `barracuda::numerical::hessian`.

### Function signature
```rust
pub fn numerical_hessian(
    loss_fn: &dyn Fn(&[f64]) -> f64,
    params: &[f64],
    epsilon: f64,
) -> Vec<f64>
```

### What it computes
Numerical Hessian via central finite differences:

- **Formula**: `H(i,j) = (f(x+e_i+e_j) - f(x+e_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)) / (4*ε²)`
- Symmetric by construction
- Uses `4n²` function evaluations (optimized to `(n²+n)/2` unique evals)

### Input/output types
| Input | Type | Description |
|-------|------|-------------|
| `loss_fn` | `&dyn Fn(&[f64]) -> f64` | Scalar function |
| `params` | `&[f64]` | Point at which to evaluate |
| `epsilon` | `f64` | Finite-difference step |
| **Output** | `Vec<f64>` | Hessian (n×n, row-major) |

### Example usage (from tests)
```rust
// f(x) = sum(x_i^2) -> H = 2I
let f = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
let params = vec![1.0, 2.0, 3.0];
let h = numerical_hessian(&f, &params, 1e-5);

// Rosenbrock at (1,1)
let f = |x: &[f64]| {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    a * a + 100.0 * b * b
};
let h = numerical_hessian(&f, &[1.0, 1.0], 1e-5);
```

### GPU (WGSL) version
**Yes.** `crates/barracuda/src/shaders/numerical/hessian_column.wgsl`:

- Entry point: `hessian_column`
- Uses `f32`
- Computes one column `j` of H; needs pre-computed buffers `f_pp`, `f_pm`, `f_mp`, `f_mm` (4 function evals per row)
- **Not wired into high-level API** — `WGSL_HESSIAN_COLUMN` in `numerical/mod.rs` is `#[allow(dead_code)]`; no GPU dispatch.

---

## 6. `belief_propagation_chain`

### Module path
```text
barracuda::linalg::belief_propagation_chain
```
Re-exported from `barracuda::linalg::graph`.

### Function signature
```rust
pub fn belief_propagation_chain(
    input_dist: &[f64],
    transition_matrices: &[&[f64]],
    layer_dims: &[usize],
) -> Vec<Vec<f64>>
```

### What it computes
Chain belief propagation (HMM-like forward pass):

- **Formula**: `P(layer_k) = normalize(transition_k * P(layer_{k-1}))`
- Returns distributions at each layer (including input)
- Equivalent to HMM forward algorithm for a chain PGM

### Input/output types
| Input | Type | Description |
|-------|------|-------------|
| `input_dist` | `&[f64]` | Initial distribution (sum = 1) |
| `transition_matrices` | `&[&[f64]]` | One matrix per layer (row-major) |
| `layer_dims` | `&[usize]` | Output dimension per layer |
| **Output** | `Vec<Vec<f64>>` | Distributions at each layer (len = 1 + num_transitions) |

### Example usage (from tests)
```rust
// Identity transition preserves distribution
let input = vec![0.5, 0.3, 0.2];
let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
let trans = [identity.as_slice()];
let layer_dims = [3];
let dists = belief_propagation_chain(&input, &trans, &layer_dims);

// 2-layer chain
let input = vec![1.0, 0.0];
let t1 = vec![0.5, 0.5, 0.5, 0.5];
let t2 = vec![1.0, 0.0, 0.0, 1.0];
let trans = [t1.as_slice(), t2.as_slice()];
let layer_dims = [2, 2];
let dists = belief_propagation_chain(&input, &trans, &layer_dims);
```

### GPU (WGSL) version
**None.** CPU-only. No WGSL shader.

---

## WGSL Shader Summary

| Shader | Path | Entry point | Precision | Wired to API? |
|--------|------|-------------|------------|---------------|
| `laplacian.wgsl` | `shaders/linalg/laplacian.wgsl` | `graph_laplacian` | f32 | No |
| `symmetrize.wgsl` | `shaders/linalg/symmetrize.wgsl` | `symmetrize` | f32 | No |
| `histogram.wgsl` | `shaders/stats/histogram.wgsl` | `histogram` | f32 | No |
| `metropolis.wgsl` | `shaders/sample/metropolis.wgsl` | `metropolis_step` | f32 | No |
| `hessian_column.wgsl` | `shaders/numerical/hessian_column.wgsl` | `hessian_column` | f32 | No |

- **symmetrize.wgsl**: `out[i,j] = (A[i,j] + A[j,i]) / 2` — adjacency, covariance, Hessians.
- **histogram.wgsl**: Atomic binning; CPU normalizes counts. Used for spectral density, EDA.
- **metropolis.wgsl**: Batch chains, pre-computed log-target buffers; different interface than CPU `boltzmann_sampling`.
- **hessian_column.wgsl**: One column per dispatch; needs pre-computed f_pp, f_pm, f_mp, f_mm buffers.

---

## Re-exports (lib.rs / prelude)

These primitives are **not** in the prelude. Import paths:

```rust
use barracuda::linalg::{graph_laplacian, disordered_laplacian, effective_rank, belief_propagation_chain};
use barracuda::sample::boltzmann_sampling;
use barracuda::numerical::numerical_hessian;
```

---

## wetSpring wiring notes

1. **CPU-only primitives** (no WGSL): `disordered_laplacian`, `effective_rank`, `belief_propagation_chain`.
2. **CPU + WGSL (unused)**: `graph_laplacian`, `boltzmann_sampling`, `numerical_hessian` — WGSL exists but no dispatch layer.
3. **Precision**: CPU uses `f64`; WGSL shaders use `f32`. GPU versions would need f64 shaders or explicit conversion.
4. **Typical pipeline**: `graph_laplacian` → `disordered_laplacian` → eigendecomposition → `effective_rank` on eigenvalues.
5. **boltzmann_sampling**: Needs a callable loss; wetSpring would pass a closure or a function pointer.
