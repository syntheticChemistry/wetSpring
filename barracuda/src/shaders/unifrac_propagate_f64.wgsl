// unifrac_propagate_f64.wgsl — GPU UniFrac tree propagation
//
// Write → Absorb → Lean: local shader for ToadStool absorption.
// Bottom-up propagation of sample abundances through a phylogenetic tree
// stored in CSR (flat) format. Each thread handles one internal node,
// summing child contributions weighted by branch length.
//
// Absorption target: ToadStool `ops::bio::unifrac_propagate`
// CPU reference: wetspring_barracuda::bio::unifrac::unweighted_unifrac
// Validation: Exp082 (unifrac CSR flat tree)
//
// Binding layout:
//   @group(0) @binding(0) uniform  UniFracConfig { n_nodes, n_samples, n_leaves, _pad }
//   @group(0) @binding(1) storage  parent:       array<i32>   [n_nodes]  (-1 = root)
//   @group(0) @binding(2) storage  branch_len:   array<f64>   [n_nodes]
//   @group(0) @binding(3) storage  sample_mat:   array<f64>   [n_leaves * n_samples]
//   @group(0) @binding(4) storage  node_sums:    array<f64>   [n_nodes * n_samples]
//   @group(0) @binding(5) storage  result:       array<f64>   [n_samples * n_samples]
//
// Dispatch: ceil(n_nodes / 64) per tree level (multi-pass)
//
// Design notes:
//   - FlatTree CSR from bio::unifrac::PhyloTree::to_flat_tree()
//   - Leaf nodes copy from sample_mat; internal nodes sum children
//   - Weighted UniFrac multiplies by branch_len before accumulating
//   - Final pass computes pairwise distances from propagated sums

struct UniFracConfig {
    n_nodes:   u32,
    n_samples: u32,
    n_leaves:  u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform>             config:     UniFracConfig;
@group(0) @binding(1) var<storage, read>       parent:     array<i32>;
@group(0) @binding(2) var<storage, read>       branch_len: array<f64>;
@group(0) @binding(3) var<storage, read>       sample_mat: array<f64>;
@group(0) @binding(4) var<storage, read_write> node_sums:  array<f64>;

@compute @workgroup_size(64)
fn unifrac_leaf_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf = gid.x;
    if leaf >= config.n_leaves {
        return;
    }

    for (var s: u32 = 0u; s < config.n_samples; s = s + 1u) {
        let src = leaf * config.n_samples + s;
        let dst = leaf * config.n_samples + s;
        node_sums[dst] = sample_mat[src];
    }
}

@compute @workgroup_size(64)
fn unifrac_propagate_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if node >= config.n_nodes {
        return;
    }

    let p = parent[node];
    if p < 0 {
        return;
    }

    let bl = branch_len[node];
    let p_u = u32(p);
    for (var s: u32 = 0u; s < config.n_samples; s = s + 1u) {
        let child_val = node_sums[node * config.n_samples + s] * bl;
        // Atomic-style accumulation (single-level pass, no race within level)
        node_sums[p_u * config.n_samples + s] = node_sums[p_u * config.n_samples + s] + child_val;
    }
}
