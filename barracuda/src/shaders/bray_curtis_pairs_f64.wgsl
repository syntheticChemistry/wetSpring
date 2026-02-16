// ============================================================================
// bray_curtis_pairs_f64.wgsl — All-pairs Bray-Curtis distance matrix (f64)
// ============================================================================
//
// Computes the condensed Bray-Curtis distance matrix for N samples,
// each of dimension D. One thread per pair (i, j) where i > j.
//
// BC(a, b) = sum(|a_k - b_k|) / sum(a_k + b_k)
//
// Output: N*(N-1)/2 distances in condensed form:
//   (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
//
// Requires: SHADER_F64 (wgpu::Features::SHADER_F64)
//
// This is the primary GPU acceleration target for diversity analysis:
// O(N^2 * D) work, embarrassingly parallel across N*(N-1)/2 pairs.
// For N=1000, D=2000: 500K independent tasks × 2000 features each.
//
// Binding layout:
//   @binding(0) samples: array<f64>     — flat [N*D], row-major
//   @binding(1) output: array<f64>      — condensed [N*(N-1)/2]
//   @binding(2) params: BcParams        — { n_samples, n_features }
//
// Dispatch: (ceil(n_pairs / 256), 1, 1)
//   where n_pairs = N*(N-1)/2
// ============================================================================

struct BcParams {
    n_samples: u32,
    n_features: u32,
    n_pairs: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> samples: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: BcParams;

// Map linear pair index to (i, j) where i > j.
// pair_idx follows condensed order: (1,0), (2,0), (2,1), ...
// i = floor((1 + sqrt(1 + 8*pair_idx)) / 2)
// j = pair_idx - i*(i-1)/2
fn pair_to_ij(pair_idx: u32) -> vec2<u32> {
    // Integer square root via Newton's method for robustness
    let k = pair_idx;
    var i = u32(sqrt(f32(2u * k + 1u)));
    // Adjust: ensure i*(i-1)/2 <= k < i*(i+1)/2
    if (i * (i - 1u) / 2u > k) {
        i = i - 1u;
    }
    if ((i + 1u) * i / 2u <= k) {
        i = i + 1u;
    }
    let j = k - i * (i - 1u) / 2u;
    return vec2<u32>(i, j);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if (pair_idx >= params.n_pairs) {
        return;
    }

    let ij = pair_to_ij(pair_idx);
    let i = ij.x;
    let j = ij.y;
    let d = params.n_features;

    var sum_diff = f64(0.0);
    var sum_sum = f64(0.0);

    let base_i = i * d;
    let base_j = j * d;

    for (var k = 0u; k < d; k = k + 1u) {
        let a = samples[base_i + k];
        let b = samples[base_j + k];

        // |a - b|
        var diff = a - b;
        if (diff < f64(0.0)) {
            diff = -diff;
        }

        sum_diff = sum_diff + diff;
        sum_sum = sum_sum + a + b;
    }

    if (sum_sum > f64(0.0)) {
        output[pair_idx] = sum_diff / sum_sum;
    } else {
        output[pair_idx] = f64(0.0);
    }
}
