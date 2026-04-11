// SPDX-License-Identifier: AGPL-3.0-or-later
//! petalTongue grammar rendering path.
//!
//! Constructs `GrammarExpr` structures from wetSpring science data and sends
//! them to petalTongue's `visualization.render.grammar` RPC for server-side
//! SVG rendering. This replaces client-side Plotly.js when the toggle is active.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

use serde_json::{Value, json};

/// Build a `GrammarRenderRequest` for petalTongue from a grammar expression and data rows.
fn grammar_request(session_id: &str, grammar: &Value, data: &[Value], domain: &str) -> Value {
    json!({
        "session_id": session_id,
        "grammar": grammar,
        "data": data,
        "modality": "svg",
        "validate_tufte": true,
        "domain": domain,
    })
}

/// Construct a GrammarExpr for dose-response timeseries.
pub fn dose_response_grammar(ipc_result: &Value) -> (Value, Vec<Value>) {
    let grammar = json!({
        "data_source": "gonzales_dose_response",
        "variables": [
            { "name": "dose", "field": "dose", "role": "X" },
            { "name": "response", "field": "response", "role": "Y" },
            { "name": "pathway", "field": "pathway", "role": "Color" },
        ],
        "geometry": "Line",
        "scales": [
            { "variable": "dose", "scale_type": "Linear" },
            { "variable": "response", "scale_type": "Linear" },
        ],
        "coordinate": "Cartesian",
        "aesthetics": [
            { "Stroke": "pathway" },
        ],
        "title": "IC50 Dose-Response — Gonzales Cytokine Pathways",
        "domain": "health",
    });

    let mut rows = Vec::new();
    let doses = ipc_result.get("doses").and_then(Value::as_array);
    let curves = ipc_result.get("curves").and_then(Value::as_array);

    if let (Some(doses), Some(curves)) = (doses, curves) {
        for curve in curves {
            let pathway = curve.get("pathway").and_then(Value::as_str).unwrap_or("?");
            let responses = curve.get("responses").and_then(Value::as_array);
            if let Some(responses) = responses {
                for (i, dose) in doses.iter().enumerate() {
                    if let Some(resp) = responses.get(i) {
                        rows.push(json!({
                            "dose": dose,
                            "response": resp,
                            "pathway": pathway,
                        }));
                    }
                }
            }
        }
    }

    (grammar, rows)
}

/// Construct a GrammarExpr for PK decay timeseries.
pub fn pk_decay_grammar(ipc_result: &Value) -> (Value, Vec<Value>) {
    let grammar = json!({
        "data_source": "gonzales_pk_decay",
        "variables": [
            { "name": "time", "field": "time_days", "role": "X" },
            { "name": "efficacy", "field": "efficacy", "role": "Y" },
            { "name": "dose", "field": "dose_label", "role": "Color" },
        ],
        "geometry": "Line",
        "scales": [
            { "variable": "time", "scale_type": "Linear" },
            { "variable": "efficacy", "scale_type": "Linear" },
        ],
        "coordinate": "Cartesian",
        "aesthetics": [
            { "Stroke": "dose" },
        ],
        "title": "Lokivetmab PK Decay — Dose-Duration Relationship",
        "domain": "health",
    });

    let mut rows = Vec::new();
    let times = ipc_result.get("times_days").and_then(Value::as_array);
    let profiles = ipc_result.get("dose_profiles").and_then(Value::as_array);

    if let (Some(times), Some(profiles)) = (times, profiles) {
        for profile in profiles {
            let dose = profile
                .get("dose_mg_kg")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let label = format!("{dose} mg/kg");
            let efficacy = profile.get("efficacy").and_then(Value::as_array);
            if let Some(efficacy) = efficacy {
                for (i, t) in times.iter().enumerate() {
                    if let Some(e) = efficacy.get(i) {
                        rows.push(json!({
                            "time_days": t,
                            "efficacy": e,
                            "dose_label": label,
                        }));
                    }
                }
            }
        }
    }

    (grammar, rows)
}

/// Construct a GrammarExpr for tissue lattice bar chart.
pub fn tissue_lattice_grammar(ipc_result: &Value) -> (Value, Vec<Value>) {
    let grammar = json!({
        "data_source": "tissue_lattice",
        "variables": [
            { "name": "profile", "field": "profile", "role": "X" },
            { "name": "shannon", "field": "shannon", "role": "Y" },
        ],
        "geometry": "Bar",
        "scales": [
            { "variable": "profile", "scale_type": "Categorical" },
            { "variable": "shannon", "scale_type": "Linear" },
        ],
        "coordinate": "Cartesian",
        "aesthetics": [
            { "Fill": "profile" },
        ],
        "title": "Shannon Diversity by AD Severity Profile",
        "domain": "health",
    });

    let mut rows = Vec::new();
    if let Some(profiles) = ipc_result.get("profiles").and_then(Value::as_array) {
        for p in profiles {
            rows.push(json!({
                "profile": p.get("profile"),
                "shannon": p.get("shannon"),
                "pielou": p.get("pielou"),
                "anderson_w": p.get("anderson_w"),
            }));
        }
    }

    (grammar, rows)
}

/// Construct a GrammarExpr for cross-species scatter.
pub fn cross_species_grammar(ipc_result: &Value) -> (Value, Vec<Value>) {
    let grammar = json!({
        "data_source": "cross_species",
        "variables": [
            { "name": "d_eff", "field": "d_eff", "role": "X" },
            { "name": "barrier_w", "field": "barrier_w", "role": "Y" },
            { "name": "species", "field": "name", "role": "Label" },
        ],
        "geometry": "Point",
        "scales": [
            { "variable": "d_eff", "scale_type": "Linear" },
            { "variable": "barrier_w", "scale_type": "Linear" },
        ],
        "coordinate": "Cartesian",
        "aesthetics": [
            { "Fill": "species" },
            { "Size": "epidermis_um" },
        ],
        "title": "Cross-Species Tissue Geometry",
        "domain": "health",
    });

    let mut rows = Vec::new();
    if let Some(species) = ipc_result.get("species").and_then(Value::as_array) {
        for s in species {
            rows.push(s.clone());
        }
    }

    (grammar, rows)
}

/// Construct a GrammarExpr for hormesis response curve.
pub fn hormesis_grammar(ipc_result: &Value) -> (Value, Vec<Value>) {
    let grammar = json!({
        "data_source": "hormesis",
        "variables": [
            { "name": "dose", "field": "dose", "role": "X" },
            { "name": "response", "field": "response", "role": "Y" },
            { "name": "component", "field": "component", "role": "Color" },
        ],
        "geometry": "Line",
        "scales": [
            { "variable": "dose", "scale_type": "Linear" },
            { "variable": "response", "scale_type": "Linear" },
        ],
        "coordinate": "Cartesian",
        "aesthetics": [
            { "Stroke": "component" },
        ],
        "title": "Hormesis — Biphasic Dose-Response",
        "domain": "health",
    });

    let mut rows = Vec::new();
    let doses = ipc_result.get("doses").and_then(Value::as_array);
    let responses = ipc_result.get("responses").and_then(Value::as_array);
    let stim = ipc_result
        .get("stimulatory_component")
        .and_then(Value::as_array);
    let surv = ipc_result
        .get("survival_component")
        .and_then(Value::as_array);

    if let Some(doses) = doses {
        for (i, dose) in doses.iter().enumerate() {
            if let Some(r) = responses.and_then(|a| a.get(i)) {
                rows.push(json!({ "dose": dose, "response": r, "component": "Total" }));
            }
            if let Some(s) = stim.and_then(|a| a.get(i)) {
                rows.push(json!({ "dose": dose, "response": s, "component": "Stimulatory" }));
            }
            if let Some(s) = surv.and_then(|a| a.get(i)) {
                rows.push(json!({ "dose": dose, "response": s, "component": "Survival" }));
            }
        }
    }

    (grammar, rows)
}

/// Call petalTongue's `visualization.render.grammar` RPC via the Neural API socket.
///
/// Returns the SVG string on success, or `None` if petalTongue is unreachable.
pub fn render_grammar(grammar: &Value, data: &[Value], domain: &str) -> Option<Value> {
    let session_id = format!(
        "facade-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );

    let request = grammar_request(&session_id, grammar, data, domain);

    let neural_socket = {
        let family_id = std::env::var("FAMILY_ID").ok()?;
        let runtime = std::env::var("XDG_RUNTIME_DIR")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        let path = std::path::PathBuf::from(runtime)
            .join("biomeos")
            .join(format!("neural-api-{family_id}.sock"));
        if path.exists() {
            path
        } else {
            return None;
        }
    };

    let mut stream = UnixStream::connect(&neural_socket).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(15)))
        .ok();

    let rpc = json!({
        "jsonrpc": "2.0",
        "method": "visualization.render.grammar",
        "params": request,
        "id": 1,
    });
    let mut line = serde_json::to_string(&rpc).ok()?;
    line.push('\n');
    stream.write_all(line.as_bytes()).ok()?;
    stream.flush().ok()?;

    let mut reader = BufReader::new(stream);
    let mut resp_line = String::new();
    reader.read_line(&mut resp_line).ok()?;

    let resp: Value = serde_json::from_str(resp_line.trim()).ok()?;
    resp.get("result").cloned()
}
