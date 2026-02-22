// taxonomy_fc_f64.wgsl — GPU taxonomy Naive Bayes scoring (fully-connected)
//
// Write → Absorb → Lean: local shader for ToadStool absorption.
// Computes log-posterior = log_prior + sum(log_prob[feature]) for each taxon.
// One thread per (query, taxon) pair. GEMM-like but with log-space accumulation.
//
// Absorption target: ToadStool `ops::bio::taxonomy_fc`
// CPU reference: wetspring_barracuda::bio::taxonomy::NaiveBayesClassifier::classify
// Validation: Exp083 (taxonomy int8 quantization)
//
// Binding layout:
//   @group(0) @binding(0) uniform  TaxConfig { n_queries, n_taxa, n_features, _pad }
//   @group(0) @binding(1) storage  log_probs:  array<f64>  [n_taxa * n_features]
//   @group(0) @binding(2) storage  log_priors: array<f64>  [n_taxa]
//   @group(0) @binding(3) storage  features:   array<u32>  [n_queries * n_features]
//   @group(0) @binding(4) storage  scores:     array<f64>  [n_queries * n_taxa]
//
// Dispatch: ceil(n_queries / 16) × ceil(n_taxa / 16)
//
// Int8 variant (NPU path):
//   Replace f64 log_probs with i8 quantized weights.
//   Scale factor stored in config. NPU FC dispatch via metalForge.
//   CPU reference: bio::taxonomy::classify_quantized

struct TaxConfig {
    n_queries:  u32,
    n_taxa:     u32,
    n_features: u32,
    _pad:       u32,
}

@group(0) @binding(0) var<uniform>             config:     TaxConfig;
@group(0) @binding(1) var<storage, read>       log_probs:  array<f64>;
@group(0) @binding(2) var<storage, read>       log_priors: array<f64>;
@group(0) @binding(3) var<storage, read>       features:   array<u32>;
@group(0) @binding(4) var<storage, read_write> scores:     array<f64>;

@compute @workgroup_size(16, 16)
fn taxonomy_fc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query = gid.x;
    let taxon = gid.y;

    if query >= config.n_queries || taxon >= config.n_taxa {
        return;
    }

    var score = log_priors[taxon];

    let feat_base = query * config.n_features;
    let prob_base = taxon * config.n_features;

    for (var f: u32 = 0u; f < config.n_features; f = f + 1u) {
        let feat_present = features[feat_base + f];
        if feat_present != 0u {
            score = score + log_probs[prob_base + f];
        }
    }

    scores[query * config.n_taxa + taxon] = score;
}
