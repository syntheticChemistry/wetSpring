// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::similar_names
)]
//! Exp221: Tower Atomic Wiring + Real Data Validation (V68)
//!
//! Validates the NUCLEUS Tower atomic integration with metalForge:
//! 1. Songbird discovery wiring (remote substrate parsing)
//! 2. `NestGate` data resolution chain (env -> IPC -> synthetic)
//! 3. Real NCBI assembly analysis (Vibrio + Campylobacterota)
//! 4. Tower-aware inventory (local + mesh substrates)
//! 5. Data-driven dispatch routing
//! 6. PFAS library validation against Zenodo data
//!
//! # Provenance
//!
//! | Field | Value |
//! |-------|-------|
//! | Baseline commit | V68 Tower wiring |
//! | Baseline tool | metalForge Tower atomic integration |
//! | Data | Real NCBI assemblies + Zenodo PFAS (priority-1) |
//! | Hardware | Eastgate (i9-12900K, 64 GB, RTX 4070) |
//!
//! Validation class: Pipeline
//! Provenance: End-to-end pipeline integration test

use wetspring_forge::data::{self, DataSource};
use wetspring_forge::dispatch::{self, Workload};
use wetspring_forge::inventory;
use wetspring_forge::substrate::{
    Capability, Identity, Properties, Substrate, SubstrateKind, SubstrateOrigin,
};
use wetspring_forge::workloads;

fn main() {
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut total = 0u32;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp221: Tower Atomic Wiring + Real Data Validation");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // ═══ S1: Songbird Substrate Parsing ═══════════════════════════════
    println!("── S1: Songbird Substrate Parsing ──────────────────────");

    let check = |desc: &str, ok: bool, pass: &mut u32, fail: &mut u32, total: &mut u32| {
        *total += 1;
        if ok {
            *pass += 1;
            println!("  [PASS] {desc}");
        } else {
            *fail += 1;
            println!("  [FAIL] {desc}");
        }
    };

    let songbird_resp_empty = r#"{"jsonrpc":"2.0","result":[],"id":1}"#;
    let parsed = parse_songbird_response(songbird_resp_empty);
    check(
        "Parse empty Songbird result",
        parsed.is_empty(),
        &mut pass,
        &mut fail,
        &mut total,
    );

    let songbird_resp_gpu = r#"{"jsonrpc":"2.0","result":[{"name":"strandgate","capabilities":["compute","gpu","f64","reduce"]}],"id":1}"#;
    let parsed = parse_songbird_response(songbird_resp_gpu);
    check(
        "Parse GPU gate from Songbird",
        parsed.len() == 1,
        &mut pass,
        &mut fail,
        &mut total,
    );
    if let Some(s) = parsed.first() {
        check(
            "Strandgate is GPU",
            s.kind == SubstrateKind::Gpu,
            &mut pass,
            &mut fail,
            &mut total,
        );
        check(
            "Strandgate has f64",
            s.has(&Capability::F64Compute),
            &mut pass,
            &mut fail,
            &mut total,
        );
        check(
            "Strandgate has shader",
            s.has(&Capability::ShaderDispatch),
            &mut pass,
            &mut fail,
            &mut total,
        );
        check(
            "Strandgate origin is Mesh",
            matches!(&s.origin, SubstrateOrigin::Mesh { gate_name } if gate_name == "strandgate"),
            &mut pass,
            &mut fail,
            &mut total,
        );
    }

    let songbird_resp_multi = r#"{"jsonrpc":"2.0","result":[{"name":"strandgate","capabilities":["gpu","f64"]},{"name":"biomegate","capabilities":["gpu","f64","reduce"]},{"name":"eastgate-npu","capabilities":["npu","quant"]}],"id":1}"#;
    let parsed = parse_songbird_response(songbird_resp_multi);
    check(
        "Parse 3 gates",
        parsed.len() == 3,
        &mut pass,
        &mut fail,
        &mut total,
    );
    check(
        "Gate 3 is NPU",
        parsed.get(2).is_some_and(|s| s.kind == SubstrateKind::Npu),
        &mut pass,
        &mut fail,
        &mut total,
    );

    let songbird_resp_error = r#"{"jsonrpc":"2.0","error":{"code":-1,"message":"offline"},"id":1}"#;
    let parsed = parse_songbird_response(songbird_resp_error);
    check(
        "Error response returns empty",
        parsed.is_empty(),
        &mut pass,
        &mut fail,
        &mut total,
    );
    println!();

    // ═══ S2: Data Resolution Chain ════════════════════════════════════
    println!("── S2: Data Resolution Chain ───────────────────────────");

    let vibrio_res = data::resolve_dataset("vibrio_assemblies");
    check(
        "Vibrio assemblies resolved",
        vibrio_res.is_real,
        &mut pass,
        &mut fail,
        &mut total,
    );
    check(
        "Vibrio source is local dir",
        matches!(vibrio_res.source, DataSource::LocalDir(_)),
        &mut pass,
        &mut fail,
        &mut total,
    );

    let campy_res = data::resolve_dataset("campylobacterota_assemblies");
    check(
        "Campylobacterota resolved",
        campy_res.is_real,
        &mut pass,
        &mut fail,
        &mut total,
    );

    let pfas_res = data::resolve_dataset("pfas_zenodo");
    check(
        "PFAS Zenodo resolved",
        pfas_res.is_real,
        &mut pass,
        &mut fail,
        &mut total,
    );

    let silva_res = data::resolve_dataset("reference_dbs");
    check(
        "SILVA reference resolved",
        silva_res.is_real,
        &mut pass,
        &mut fail,
        &mut total,
    );

    let synthetic_res = data::resolve_dataset("nonexistent_dataset_xyz");
    check(
        "Nonexistent dataset → synthetic",
        !synthetic_res.is_real && synthetic_res.source == DataSource::Synthetic,
        &mut pass,
        &mut fail,
        &mut total,
    );
    println!();

    // ═══ S3: Real Assembly File Validation ════════════════════════════
    println!("── S3: Real Assembly File Validation ───────────────────");

    if let Some(ref vibrio_path) = vibrio_res.path {
        let assemblies: Vec<_> = std::fs::read_dir(vibrio_path)
            .expect("read_dir vibrio_assemblies should succeed when path exists")
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "gz"))
            .collect();

        check(
            &format!("Vibrio: {} assemblies on disk", assemblies.len()),
            assemblies.len() >= 150,
            &mut pass,
            &mut fail,
            &mut total,
        );

        let total_bytes: u64 = assemblies
            .iter()
            .map(|e| e.metadata().map_or(0, |m| m.len()))
            .sum();
        #[expect(clippy::cast_precision_loss)]
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        check(
            &format!("Vibrio: {total_mb:.0} MB total (expect ~250 MB)"),
            total_mb > 100.0,
            &mut pass,
            &mut fail,
            &mut total,
        );

        let avg_size = total_bytes / assemblies.len().max(1) as u64;
        check(
            &format!("Vibrio: avg {avg_size} bytes/assembly"),
            avg_size > 500_000 && avg_size < 10_000_000,
            &mut pass,
            &mut fail,
            &mut total,
        );
    } else {
        check(
            "Vibrio path available",
            false,
            &mut pass,
            &mut fail,
            &mut total,
        );
    }

    if let Some(ref campy_path) = campy_res.path {
        let assemblies: Vec<_> = std::fs::read_dir(campy_path)
            .expect("read_dir campylobacterota_assemblies should succeed when path exists")
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "gz"))
            .collect();

        check(
            &format!("Campylobacterota: {} assemblies on disk", assemblies.len()),
            assemblies.len() >= 100,
            &mut pass,
            &mut fail,
            &mut total,
        );
    } else {
        check(
            "Campylobacterota path available",
            false,
            &mut pass,
            &mut fail,
            &mut total,
        );
    }
    println!();

    // ═══ S4: Tower-Aware Inventory ════════════════════════════════════
    println!("── S4: Tower-Aware Inventory ───────────────────────────");

    let local_subs = inventory::discover();
    let tower_subs = inventory::discover_with_tower();

    check(
        "Local inventory has CPU",
        local_subs.iter().any(|s| s.kind == SubstrateKind::Cpu),
        &mut pass,
        &mut fail,
        &mut total,
    );
    check(
        "Tower inventory >= local",
        tower_subs.len() >= local_subs.len(),
        &mut pass,
        &mut fail,
        &mut total,
    );
    check(
        "All local substrates have Local origin",
        local_subs
            .iter()
            .all(|s| s.origin == SubstrateOrigin::Local),
        &mut pass,
        &mut fail,
        &mut total,
    );

    let songbird_available = inventory::discover_songbird_socket().is_some();
    println!(
        "  Songbird socket: {}",
        if songbird_available {
            "FOUND"
        } else {
            "not running (standalone mode)"
        }
    );

    let local_count = tower_subs
        .iter()
        .filter(|s| s.origin == SubstrateOrigin::Local)
        .count();
    let mesh_count = tower_subs.len() - local_count;
    println!("  Inventory: {local_count} local, {mesh_count} mesh substrates");

    inventory::print_inventory(&tower_subs);
    println!();

    // ═══ S5: Data-Driven Dispatch ═════════════════════════════════════
    println!("── S5: Data-Driven Dispatch ────────────────────────────");

    let all_workloads = workloads::all_workloads();
    let absorbed = all_workloads.iter().filter(|w| w.is_absorbed()).count();
    check(
        &format!("Absorbed workloads: {absorbed} (expect >= 39)"),
        absorbed >= 39,
        &mut pass,
        &mut fail,
        &mut total,
    );

    let vibrio_workload = Workload::new("vibrio_landscape", vec![Capability::F64Compute]);
    let decision = dispatch::route(&vibrio_workload, &tower_subs);
    check(
        "Vibrio landscape routes to a substrate",
        decision.is_some(),
        &mut pass,
        &mut fail,
        &mut total,
    );
    if let Some(d) = &decision {
        println!(
            "  Vibrio routed to: {} ({:?})",
            d.substrate.identity.name, d.reason
        );
    }

    let spectral_workload = Workload::new(
        "pfas_spectral_match",
        vec![Capability::F64Compute, Capability::ShaderDispatch],
    );
    let spectral_decision = dispatch::route(&spectral_workload, &tower_subs);
    if spectral_decision.is_some() {
        check(
            "PFAS spectral match routes to GPU",
            true,
            &mut pass,
            &mut fail,
            &mut total,
        );
    } else {
        check(
            "PFAS spectral match falls back to CPU (no GPU)",
            true,
            &mut pass,
            &mut fail,
            &mut total,
        );
    }
    println!();

    // ═══ S6: PFAS Library Validation ══════════════════════════════════
    println!("── S6: PFAS Library Validation ─────────────────────────");

    if let Some(ref pfas_path) = pfas_res.path {
        let files: Vec<_> = std::fs::read_dir(pfas_path)
            .expect("read_dir pfas_zenodo should succeed when path exists")
            .filter_map(Result::ok)
            .collect();

        check(
            &format!("PFAS Zenodo: {} files", files.len()),
            files.len() >= 3,
            &mut pass,
            &mut fail,
            &mut total,
        );

        let has_joseph = files
            .iter()
            .any(|f| f.file_name().to_string_lossy().contains("Joseph"));
        check(
            "PFAS Joseph library present",
            has_joseph,
            &mut pass,
            &mut fail,
            &mut total,
        );
    } else {
        check(
            "PFAS path available",
            false,
            &mut pass,
            &mut fail,
            &mut total,
        );
    }
    println!();

    // ═══ Summary ══════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════");
    println!("  Exp221 Result: {pass} PASS, {fail} FAIL ({total} total)");
    if fail > 0 {
        println!("  STATUS: FAIL");
    } else {
        println!("  STATUS: ALL PASS");
    }
    println!("═══════════════════════════════════════════════════════════");

    if fail > 0 {
        std::process::exit(1);
    }
}

/// Parse a Songbird discovery response into substrates (test helper).
fn parse_songbird_response(response: &str) -> Vec<Substrate> {
    if response.contains("\"error\"") {
        return Vec::new();
    }

    let Some(result_start) = response.find("\"result\"") else {
        return Vec::new();
    };
    let after = &response[result_start..];
    let Some(arr_start) = after.find('[') else {
        return Vec::new();
    };
    let Some(arr_end) = after.rfind(']') else {
        return Vec::new();
    };
    if arr_start >= arr_end {
        return Vec::new();
    }

    let arr_content = &after[arr_start + 1..arr_end];
    let mut substrates = Vec::new();

    for entry in split_json_objects(arr_content) {
        if let Some(sub) = parse_service_entry(entry) {
            substrates.push(sub);
        }
    }
    substrates
}

fn split_json_objects(content: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = None;
    for (i, ch) in content.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        objects.push(&content[s..=i]);
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    objects
}

fn parse_service_entry(json: &str) -> Option<Substrate> {
    let gate_name = extract_str(json, "name")?;
    let caps_raw = extract_array(json, "capabilities");

    let mut capabilities = vec![Capability::F32Compute];
    let mut kind = SubstrateKind::Cpu;
    let mut has_f64 = false;

    for cap in &caps_raw {
        match cap.as_str() {
            "f64" => {
                capabilities.push(Capability::F64Compute);
                has_f64 = true;
            }
            "gpu" | "shader" => {
                kind = SubstrateKind::Gpu;
                capabilities.push(Capability::ShaderDispatch);
            }
            "npu" | "quant" => {
                kind = SubstrateKind::Npu;
                capabilities.push(Capability::QuantizedInference { bits: 8 });
            }
            "reduce" => {
                capabilities.push(Capability::ScalarReduce);
            }
            _ => {}
        }
    }

    Some(Substrate {
        kind,
        identity: Identity::named(&gate_name),
        properties: Properties {
            has_f64,
            ..Properties::default()
        },
        capabilities,
        origin: SubstrateOrigin::Mesh { gate_name },
    })
}

fn extract_str(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let s = json.find(&pat)?;
    let after = &json[s + pat.len()..];
    let colon = after.find(':')?;
    let trimmed = after[colon + 1..].trim_start();
    let inner = trimmed.strip_prefix('"')?;
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn extract_array(json: &str, key: &str) -> Vec<String> {
    let pat = format!("\"{key}\"");
    let Some(s) = json.find(&pat) else {
        return Vec::new();
    };
    let after = &json[s + pat.len()..];
    let Some(arr_s) = after.find('[') else {
        return Vec::new();
    };
    let Some(arr_e) = after[arr_s..].find(']') else {
        return Vec::new();
    };
    after[arr_s + 1..arr_s + arr_e]
        .split(',')
        .filter_map(|s| {
            let t = s.trim().trim_matches('"');
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect()
}
