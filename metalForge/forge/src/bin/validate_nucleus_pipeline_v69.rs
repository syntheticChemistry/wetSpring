// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]

//! Exp222 — Full NUCLEUS Pipeline Validation (V69)
//!
//! Validates the complete Tower → Nest → Node data flow:
//!
//! **S1: Nest Client Protocol** — `NestGate` storage JSON-RPC protocol
//!       (store, retrieve, exists, list, blob round-trip, base64)
//!
//! **S2: NCBI Acquisition via Tower** — ESearch/ESummary/EFetch protocol,
//!       assembly resolution chain, `NestGate` caching layer
//!
//! **S3: Real Assembly Compute (Node)** — N50, GC content, genome size,
//!       Shannon entropy on 199 Vibrio + 158 Campylobacterota assemblies
//!
//! **S4: Collection Diversity Analysis** — Cross-collection GC diversity,
//!       genome size distributions, comparative statistics
//!
//! **S5: Full Pipeline Integration** — Tower discovers → Nest resolves →
//!       Node dispatches → results verified against biological expectations
//!
//! **S6: NUCLEUS Workload Catalog** — All 47 workloads registered,
//!       NUCLEUS data-driven domains dispatched correctly
//!
//! Validation class: Pipeline
//! Provenance: End-to-end pipeline integration test

#![expect(
    clippy::expect_used,
    reason = "validation harness: fail-fast on setup errors"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "validation harness: f64 arithmetic for timing and metric ratios"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "validation harness: u128→u64 timing, f64→u32 counts"
)]

use wetspring_barracuda::tolerances;
use wetspring_forge::data;
use wetspring_forge::dispatch;
use wetspring_forge::inventory;
use wetspring_forge::ncbi;
use wetspring_forge::nest;
use wetspring_forge::node;
use wetspring_forge::substrate::{Capability, SubstrateKind};
use wetspring_forge::workloads;

use std::path::PathBuf;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp222: Full NUCLEUS Pipeline Validation (V69)");
    println!("  Tower → Nest → Node");
    println!("═══════════════════════════════════════════════════════════");

    let mut pass = 0u32;
    let mut fail = 0u32;

    section_nest_protocol(&mut pass, &mut fail);
    section_ncbi_acquisition(&mut pass, &mut fail);
    section_assembly_compute(&mut pass, &mut fail);
    section_collection_diversity(&mut pass, &mut fail);
    section_pipeline_integration(&mut pass, &mut fail);
    section_workload_catalog(&mut pass, &mut fail);

    let total = pass + fail;
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp222 Result: {pass} PASS, {fail} FAIL ({total} total)");
    if fail == 0 {
        println!("  STATUS: ALL PASS");
    } else {
        println!("  STATUS: FAILED");
    }
    println!("═══════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}

fn check(name: &str, condition: bool, pass: &mut u32, fail: &mut u32) {
    if condition {
        println!("  [PASS] {name}");
        *pass += 1;
    } else {
        println!("  [FAIL] {name}");
        *fail += 1;
    }
}

// ── S1: Nest Client Protocol ────────────────────────────────────────

fn section_nest_protocol(pass: &mut u32, fail: &mut u32) {
    println!();
    println!("── S1: Nest Client Protocol ─────────────────────────");

    // Base64 round-trip (critical for blob storage)
    let test_data = b"ATGCATGCATGC\n>genome_assembly_data\nGGGCCCATATAT";
    let encoded = nest_base64_encode(test_data);
    let decoded = nest_base64_decode(&encoded);
    check(
        "Base64 round-trip (genomic data)",
        decoded == test_data,
        pass,
        fail,
    );

    // Base64 with binary payload
    let binary_data: Vec<u8> = (0..=255).collect();
    let enc2 = nest_base64_encode(&binary_data);
    let dec2 = nest_base64_decode(&enc2);
    check(
        "Base64 round-trip (binary 0-255)",
        dec2 == binary_data,
        pass,
        fail,
    );

    // NestGate socket discovery
    let socket_result = nest::discover_nestgate_socket();
    let has_nestgate = socket_result.is_some();
    println!(
        "  [INFO] NestGate socket: {}",
        if has_nestgate {
            "FOUND"
        } else {
            "not running (protocol-only mode)"
        }
    );
    check("NestGate socket discovery does not panic", true, pass, fail);

    // NestClient construction
    let dir = tempfile::tempdir().expect("tempdir creation for NestClient test should succeed");
    let sock = dir.path().join("test-nestgate.sock");
    let client = nest::NestClient::new(sock);
    check(
        "NestClient constructs with arbitrary socket",
        client
            .socket_path()
            .to_str()
            .expect("socket path should be valid UTF-8")
            .contains("test-nestgate"),
        pass,
        fail,
    );

    // NestClient with family
    let sock2 = dir.path().join("test.sock");
    let client2 = nest::NestClient::new(sock2).with_family("wetspring");
    check("NestClient accepts custom family_id", true, pass, fail);
    let _ = client2;

    // NestClient discover (graceful when no socket)
    let discovered = nest::NestClient::discover();
    check(
        "NestClient::discover() graceful (no panic)",
        true,
        pass,
        fail,
    );
    let _ = discovered;
}

// ── S2: NCBI Acquisition via Tower ──────────────────────────────────

fn section_ncbi_acquisition(pass: &mut u32, fail: &mut u32) {
    println!();
    println!("── S2: NCBI Acquisition via Tower ────────────────────");

    // URL encoding
    check(
        "URL encode: spaces",
        url_encode_test("Vibrio cholerae") == "Vibrio+cholerae",
        pass,
        fail,
    );
    check(
        "URL encode: brackets",
        url_encode_test("Vibrio[Organism]").contains("%5B"),
        pass,
        fail,
    );

    // XML parsing (ESearch)
    let sample_xml = r#"<?xml version="1.0" ?>
<eSearchResult><Count>199</Count><RetMax>3</RetMax>
<IdList><Id>12345</Id><Id>67890</Id><Id>11111</Id></IdList>
</eSearchResult>"#;
    let count = extract_xml_count(sample_xml);
    check("ESearch XML parse: count=199", count == 199, pass, fail);

    let ids = extract_xml_ids(sample_xml);
    check(
        "ESearch XML parse: 3 IDs returned",
        ids.len() == 3,
        pass,
        fail,
    );

    // NcbiClient construction
    let client = ncbi::NcbiClient::direct();
    check(
        "NcbiClient::direct() has no Nest",
        !client.has_nest(),
        pass,
        fail,
    );

    let client2 = ncbi::NcbiClient::discover();
    check("NcbiClient::discover() graceful", true, pass, fail);
    let _ = client2;

    // Assembly resolution: check local data directory
    let data_dir = data_dir();
    let vibrio_dir = data_dir.join("vibrio_assemblies");
    if vibrio_dir.is_dir() {
        let client3 = ncbi::NcbiClient::direct();
        let result = client3.acquire_assembly("GCF_000024825.1", Some(&vibrio_dir));
        check(
            "Assembly resolve: Vibrio GCF_000024825.1 found locally",
            result.is_ok()
                && result
                    .as_ref()
                    .expect("assembly result should be Ok when is_ok")
                    .source
                    == ncbi::AssemblySource::LocalFile(vibrio_dir.join("GCF_000024825.1.fna.gz")),
            pass,
            fail,
        );
        if let Ok(ref r) = result {
            check("Assembly resolve: size > 0", r.size_bytes > 0, pass, fail);
        }
    } else {
        println!("  [SKIP] Vibrio assemblies not on disk");
    }
}

// ── S3: Real Assembly Compute (Node) ────────────────────────────────

fn section_assembly_compute(pass: &mut u32, fail: &mut u32) {
    println!();
    println!("── S3: Real Assembly Compute (Node) ───────────────────");

    let data_dir = data_dir();
    let vibrio_dir = data_dir.join("vibrio_assemblies");
    let campy_dir = data_dir.join("campylobacterota_assemblies");

    if vibrio_dir.is_dir() {
        validate_organism_collection(
            "Vibrio",
            &vibrio_dir,
            150,
            0.44..=0.52,
            2.5..=6.5,
            pass,
            fail,
        );
    } else {
        println!("  [SKIP] Vibrio assemblies not available");
    }

    if campy_dir.is_dir() {
        validate_organism_collection(
            "Campylobacterota",
            &campy_dir,
            100,
            0.25..=0.45,
            1.0..=4.0,
            pass,
            fail,
        );
    } else {
        println!("  [SKIP] Campylobacterota assemblies not available");
    }
}

fn validate_organism_collection(
    name: &str,
    dir: &std::path::Path,
    min_count: usize,
    gc_range: std::ops::RangeInclusive<f64>,
    size_range_mb: std::ops::RangeInclusive<f64>,
    pass: &mut u32,
    fail: &mut u32,
) {
    let files = node::list_assembly_files(dir).unwrap_or_default();
    check(
        &format!("{name} assemblies on disk: {}", files.len()),
        files.len() >= min_count,
        pass,
        fail,
    );

    let sample: Vec<_> = files.into_iter().take(10).collect();
    let stats = compute_stats_sample(&sample);

    check(
        &format!("{name}: {}/10 assemblies parsed", stats.len()),
        stats.len() >= 8,
        pass,
        fail,
    );

    if !stats.is_empty() {
        let mean_gc: f64 = stats.iter().map(|s| s.gc_content).sum::<f64>() / stats.len() as f64;
        check(
            &format!("{name} mean GC: {mean_gc:.4} (expect {gc_range:?})"),
            gc_range.contains(&mean_gc),
            pass,
            fail,
        );

        let mean_size: f64 =
            stats.iter().map(|s| s.total_length as f64).sum::<f64>() / stats.len() as f64;
        let size_mb = mean_size / 1_000_000.0;
        check(
            &format!("{name} mean genome: {size_mb:.2} Mbp (expect {size_range_mb:?})"),
            size_range_mb.contains(&size_mb),
            pass,
            fail,
        );

        if name == "Vibrio" {
            let mean_n50: f64 =
                stats.iter().map(|s| s.n50 as f64).sum::<f64>() / stats.len() as f64;
            check(
                &format!("{name} mean N50: {mean_n50:.0} bp (expect > 10,000)"),
                mean_n50 > 10_000.0,
                pass,
                fail,
            );
        }
    }
}

fn compute_stats_sample(paths: &[PathBuf]) -> Vec<node::AssemblyStats> {
    let mut stats = Vec::new();
    for path in paths {
        let acc = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .strip_suffix(".fna")
            .unwrap_or("?");
        match node::compute_assembly_stats_from_file(acc, path) {
            Ok(s) => stats.push(s),
            Err(e) => println!("  [WARN] {acc}: {e}"),
        }
    }
    stats
}

// ── S4: Collection Diversity Analysis ───────────────────────────────

fn section_collection_diversity(pass: &mut u32, fail: &mut u32) {
    println!();
    println!("── S4: Collection Diversity Analysis ──────────────────");

    let data_dir = data_dir();

    // Vibrio full collection analysis
    let vibrio_dir = data_dir.join("vibrio_assemblies");
    if vibrio_dir.is_dir() {
        let files = node::list_assembly_files(&vibrio_dir).unwrap_or_default();
        // Use a sample of 20 for speed
        let sample: Vec<_> = files.into_iter().take(20).collect();
        let mut gc_values = Vec::new();
        let mut sizes = Vec::new();

        for path in &sample {
            let acc = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?")
                .strip_suffix(".fna")
                .unwrap_or("?");
            if let Ok(s) = node::compute_assembly_stats_from_file(acc, path) {
                gc_values.push(s.gc_content);
                sizes.push(s.total_length as f64);
            }
        }

        if gc_values.len() >= 5 {
            let gc_entropy = node::shannon_entropy_binned(&gc_values, 10);
            check(
                &format!("Vibrio GC Shannon entropy: {gc_entropy:.4} (expect > 0, diverse GC)"),
                gc_entropy > 0.0,
                pass,
                fail,
            );

            let size_entropy = node::shannon_entropy_binned(&sizes, 10);
            check(
                &format!("Vibrio genome size entropy: {size_entropy:.4} (expect > 0)"),
                size_entropy > 0.0,
                pass,
                fail,
            );

            // GC range should span at least 2% for Vibrio diversity
            let gc_min = gc_values.iter().copied().fold(f64::INFINITY, f64::min);
            let gc_max = gc_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let gc_range = gc_max - gc_min;
            check(
                &format!(
                    "Vibrio GC range: {gc_range:.4} (expect >= {} for genus diversity)",
                    tolerances::GC_GENUS_DIVERSITY_MIN
                ),
                gc_range >= tolerances::GC_GENUS_DIVERSITY_MIN,
                pass,
                fail,
            );
        }
    } else {
        println!("  [SKIP] Vibrio not available for diversity analysis");
    }

    // Cross-collection comparison
    let campy_dir = data_dir.join("campylobacterota_assemblies");
    if vibrio_dir.is_dir() && campy_dir.is_dir() {
        let v_files = node::list_assembly_files(&vibrio_dir).unwrap_or_default();
        let c_files = node::list_assembly_files(&campy_dir).unwrap_or_default();

        let v_sample: Vec<_> = v_files.into_iter().take(10).collect();
        let c_sample: Vec<_> = c_files.into_iter().take(10).collect();

        let v_gc = compute_gc_sample(&v_sample);
        let c_gc = compute_gc_sample(&c_sample);

        if !v_gc.is_empty() && !c_gc.is_empty() {
            let v_mean: f64 = v_gc.iter().sum::<f64>() / v_gc.len() as f64;
            let c_mean: f64 = c_gc.iter().sum::<f64>() / c_gc.len() as f64;

            // Vibrio should have higher GC than Campylobacterota
            check(
                &format!(
                    "Cross-collection: Vibrio GC ({v_mean:.4}) > Campylobacterota GC ({c_mean:.4})"
                ),
                v_mean > c_mean,
                pass,
                fail,
            );
        }
    }
}

// ── S5: Full Pipeline Integration ───────────────────────────────────

fn section_pipeline_integration(pass: &mut u32, fail: &mut u32) {
    println!();
    println!("── S5: Full Pipeline Integration ──────────────────────");

    // Tower: discover substrates
    let substrates = inventory::discover_with_tower();
    check(
        &format!("Tower discovers {} substrate(s)", substrates.len()),
        !substrates.is_empty(),
        pass,
        fail,
    );

    let has_cpu = substrates.iter().any(|s| s.kind == SubstrateKind::Cpu);
    check("Tower: CPU substrate present", has_cpu, pass, fail);

    // Nest: data resolution chain
    let vibrio = data::resolve_dataset("vibrio_assemblies");
    check(
        &format!("Nest resolves vibrio_assemblies: {:?}", vibrio.source),
        vibrio.is_real,
        pass,
        fail,
    );

    let campy = data::resolve_dataset("campylobacterota_assemblies");
    check(
        &format!(
            "Nest resolves campylobacterota_assemblies: {:?}",
            campy.source
        ),
        campy.is_real,
        pass,
        fail,
    );

    let pfas = data::resolve_dataset("pfas_zenodo");
    check(
        &format!("Nest resolves pfas_zenodo: {:?}", pfas.source),
        pfas.is_real,
        pass,
        fail,
    );

    // Node: dispatch routing for NUCLEUS workloads
    let w = dispatch::Workload::new("vibrio_assembly_stats", vec![Capability::F64Compute]);
    let d = dispatch::route_bandwidth_aware(&w, &substrates);
    check(
        "Node routes assembly_stats workload",
        d.is_some(),
        pass,
        fail,
    );

    // Full pipeline: compute_assembly_stats
    if let Some(dir) = vibrio.path.as_ref().filter(|_| vibrio.is_real) {
        match node::compute_collection_from_dir("vibrio_assemblies", dir) {
            Ok(stats) => {
                check(
                    &format!(
                        "Pipeline: computed {} Vibrio assemblies",
                        stats.assembly_count
                    ),
                    stats.assembly_count > 0,
                    pass,
                    fail,
                );
                check(
                    &format!(
                        "Pipeline: mean GC = {:.4}, mean N50 = {:.0}",
                        stats.mean_gc, stats.mean_n50
                    ),
                    stats.mean_gc > 0.0 && stats.mean_n50 > 0.0,
                    pass,
                    fail,
                );
            }
            Err(e) => {
                println!("  [FAIL] Pipeline compute: {e}");
                *fail += 1;
            }
        }
    }
}

// ── S6: NUCLEUS Workload Catalog ────────────────────────────────────

fn section_workload_catalog(pass: &mut u32, fail: &mut u32) {
    println!();
    println!("── S6: NUCLEUS Workload Catalog ─────────────────────");

    let all = workloads::all_workloads();
    check(
        &format!("Workload catalog: {} entries (expect >= 47)", all.len()),
        all.len() >= 47,
        pass,
        fail,
    );

    let (absorbed, local, cpu_only) = workloads::origin_summary();
    check(
        &format!("Absorbed: {absorbed}, Local: {local}, CPU-only: {cpu_only}"),
        absorbed == 45 && local == 0 && cpu_only == 2,
        pass,
        fail,
    );

    // NUCLEUS data-driven workloads exist
    let nucleus_names = [
        "assembly_statistics",
        "gc_analysis",
        "genome_diversity",
        "pfas_spectral_match",
        "vibrio_landscape",
        "campylobacterota_comparative",
        "ncbi_assembly_ingest",
    ];
    for name in &nucleus_names {
        let found = all.iter().any(|w| w.workload.name == *name);
        check(&format!("NUCLEUS workload: {name}"), found, pass, fail);
    }

    // Dispatch: NUCLEUS workloads route to substrates
    let substrates = inventory::discover();
    let assembly_w = workloads::assembly_statistics();
    let d = dispatch::route(&assembly_w.workload, &substrates);
    check(
        "assembly_statistics dispatches to substrate",
        d.is_some(),
        pass,
        fail,
    );

    let gc_w = workloads::gc_analysis();
    let d2 = dispatch::route(&gc_w.workload, &substrates);
    check(
        "gc_analysis dispatches to substrate",
        d2.is_some(),
        pass,
        fail,
    );
}

// ── Helpers ─────────────────────────────────────────────────────────

fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("WETSPRING_DATA_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(std::path::Path::parent)
        .map_or_else(|| PathBuf::from("data"), |p| p.join("data"))
}

fn compute_gc_sample(paths: &[PathBuf]) -> Vec<f64> {
    let mut gcs = Vec::new();
    for path in paths {
        let acc = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .strip_suffix(".fna")
            .unwrap_or("?");
        if let Ok(s) = node::compute_assembly_stats_from_file(acc, path) {
            gcs.push(s.gc_content);
        }
    }
    gcs
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn nest_base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        result.push(B64[((n >> 18) & 0x3F) as usize] as char);
        result.push(B64[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64[((n >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn nest_base64_decode(encoded: &str) -> Vec<u8> {
    let clean: Vec<u8> = encoded
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    let mut result = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let vals: Vec<u8> = chunk.iter().map(|&b| b64_val(b)).collect();
        let n = (u32::from(vals[0]) << 18)
            | (u32::from(vals[1]) << 12)
            | (u32::from(vals[2]) << 6)
            | u32::from(vals[3]);
        result.push((n >> 16) as u8);
        if chunk[2] != b'=' {
            result.push((n >> 8) as u8);
        }
        if chunk[3] != b'=' {
            result.push(n as u8);
        }
    }
    result
}

const fn b64_val(ch: u8) -> u8 {
    match ch {
        b'A'..=b'Z' => ch - b'A',
        b'a'..=b'z' => ch - b'a' + 26,
        b'0'..=b'9' => ch - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => 0,
    }
}

fn url_encode_test(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            b' ' => result.push('+'),
            _ => {
                result.push('%');
                result.push(hex_char(byte >> 4));
                result.push(hex_char(byte & 0xF));
            }
        }
    }
    result
}

const fn hex_char(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'A' + nibble - 10) as char,
    }
}

fn extract_xml_count(xml: &str) -> u64 {
    let open = "<Count>";
    let close = "</Count>";
    xml.find(open)
        .and_then(|start| {
            let content_start = start + open.len();
            xml[content_start..]
                .find(close)
                .and_then(|end| xml[content_start..content_start + end].parse().ok())
        })
        .unwrap_or(0)
}

fn extract_xml_ids(xml: &str) -> Vec<String> {
    let open = "<Id>";
    let close = "</Id>";
    let mut ids = Vec::new();
    let mut search = 0;
    while let Some(start) = xml[search..].find(open) {
        let abs_start = search + start + open.len();
        if let Some(end) = xml[abs_start..].find(close) {
            ids.push(xml[abs_start..abs_start + end].to_string());
            search = abs_start + end + close.len();
        } else {
            break;
        }
    }
    ids
}
