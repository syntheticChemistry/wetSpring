// kmer_histogram_f64.wgsl — GPU k-mer histogram accumulation
//
// Write → Absorb → Lean: local shader for ToadStool absorption.
// Computes 4^k histogram from encoded k-mer sequences.
// One thread per k-mer, atomic add into histogram buffer.
//
// Absorption target: ToadStool `ops::bio::kmer_histogram`
// CPU reference: wetspring_barracuda::bio::kmer::count_kmers
// Validation: Exp081 (kmer GPU histogram)
//
// Binding layout:
//   @group(0) @binding(0) uniform  KmerConfig { n_kmers, k, _pad0, _pad1 }
//   @group(0) @binding(1) storage  encoded_kmers: array<u32>   [n_kmers]
//   @group(0) @binding(2) storage  histogram:     array<atomic<u32>> [4^k]
//
// Dispatch: ceil(n_kmers / 256)

struct KmerConfig {
    n_kmers: u32,
    k:       u32,
    _pad0:   u32,
    _pad1:   u32,
}

@group(0) @binding(0) var<uniform>             config:   KmerConfig;
@group(0) @binding(1) var<storage, read>       kmers:    array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn kmer_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= config.n_kmers {
        return;
    }

    let kmer_hash = kmers[idx];
    atomicAdd(&histogram[kmer_hash], 1u);
}
