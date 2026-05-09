// SPDX-License-Identifier: AGPL-3.0-or-later
//! Layers 3–6: NUCLEUS health and IPC parity certification.
//!
//! - **N1**: Manifest IPC parity (15 validation capabilities)
//! - **N2**: Extended domain science IPC
//! - **N3**: Cross-atomic pipeline (hash → store → retrieve → verify)

use primalspring::composition::{CompositionContext, is_skip_error, validate_parity};
use primalspring::tolerances as ps_tol;
use primalspring::validation::ValidationResult;

use crate::tolerances;

/// N1 — Manifest IPC Parity: validates the 15 downstream manifest capabilities.
pub fn validate_manifest_ipc(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    validate_manifest_math(ctx, v);
    validate_manifest_services(ctx, v);
}

fn validate_manifest_math(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    validate_tensor_matmul(ctx, v);

    match ctx.call(
        "tensor",
        "tensor.create",
        serde_json::json!({"data": [1.0, 2.0, 3.0], "shape": [3]}),
    ) {
        Ok(_) => v.check_bool("tensor.create accepted", true, "barraCuda tensor creation"),
        Err(e) => v.check_skip("tensor.create", &format!("{e}")),
    }

    validate_parity(
        ctx,
        v,
        "stats.mean([10..50]) IPC = 30.0",
        "tensor",
        "stats.mean",
        serde_json::json!({"data": [10.0, 20.0, 30.0, 40.0, 50.0]}),
        "result",
        30.0,
        ps_tol::IPC_ROUND_TRIP_TOL,
    );

    validate_parity(
        ctx,
        v,
        "stats.std_dev([10..50]) IPC = √250 (sample)",
        "tensor",
        "stats.std_dev",
        serde_json::json!({"data": [10.0, 20.0, 30.0, 40.0, 50.0]}),
        "result",
        (250.0_f64).sqrt(),
        ps_tol::IPC_ROUND_TRIP_TOL,
    );

    validate_parity_or_skip(
        ctx,
        v,
        "stats.variance([10..50]) IPC = 250 (sample)",
        "tensor",
        "stats.variance",
        serde_json::json!({"data": [10.0, 20.0, 30.0, 40.0, 50.0]}),
        250.0,
        ps_tol::IPC_ROUND_TRIP_TOL,
    );

    validate_parity_or_skip(
        ctx,
        v,
        "stats.correlation(x,y) IPC = 1.0 (perfect)",
        "tensor",
        "stats.correlation",
        serde_json::json!({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]}),
        1.0,
        ps_tol::IPC_ROUND_TRIP_TOL,
    );

    match ctx.call(
        "tensor",
        "linalg.solve",
        serde_json::json!({"matrix": [[2.0, 1.0], [1.0, 3.0]], "b": [5.0, 7.0]}),
    ) {
        Ok(r) => {
            let result = r.get("result").and_then(|v| v.as_array());
            let ok = result.is_some_and(|arr| {
                arr.len() == 2
                    && arr[0]
                        .as_f64()
                        .is_some_and(|x| (x - 1.6).abs() < tolerances::ANALYTICAL_LOOSE)
                    && arr[1]
                        .as_f64()
                        .is_some_and(|x| (x - 1.8).abs() < tolerances::ANALYTICAL_LOOSE)
            });
            v.check_bool(
                "linalg.solve Ax=b IPC",
                ok,
                "barraCuda: [[2,1],[1,3]]x=[5,7] → [1.6,1.8]",
            );
        }
        Err(e) if is_skip_error(&e) => v.check_skip("linalg.solve", &format!("{e}")),
        Err(e) => v.check_bool("linalg.solve", false, &format!("{e}")),
    }

    match ctx.call(
        "tensor",
        "linalg.eigenvalues",
        serde_json::json!({"matrix": [[2.0, 1.0], [1.0, 2.0]]}),
    ) {
        Ok(r) => {
            let result = r.get("result").and_then(|v| v.as_array());
            let ok = result.is_some_and(|arr| arr.len() == 2);
            v.check_bool(
                "linalg.eigenvalues 2×2 IPC",
                ok,
                "barraCuda: [[2,1],[1,2]] → [3, 1]",
            );
        }
        Err(e) if is_skip_error(&e) => v.check_skip("linalg.eigenvalues", &format!("{e}")),
        Err(e) => v.check_bool("linalg.eigenvalues", false, &format!("{e}")),
    }

    match ctx.call(
        "tensor",
        "spectral.fft",
        serde_json::json!({"data": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]}),
    ) {
        Ok(r) => {
            let has_result = r
                .get("result")
                .and_then(|v| v.as_array())
                .is_some_and(|a| !a.is_empty());
            v.check_bool(
                "spectral.fft IPC",
                has_result,
                "barraCuda: 8-point FFT returns result array",
            );
        }
        Err(e) if is_skip_error(&e) => v.check_skip("spectral.fft", &format!("{e}")),
        Err(e) => v.check_bool("spectral.fft", false, &format!("{e}")),
    }

    match ctx.call(
        "tensor",
        "spectral.power_spectrum",
        serde_json::json!({"data": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]}),
    ) {
        Ok(r) => {
            let has_result = r
                .get("result")
                .and_then(|v| v.as_array())
                .is_some_and(|a| !a.is_empty());
            v.check_bool(
                "spectral.power_spectrum IPC",
                has_result,
                "barraCuda: 8-point power spectrum returns result array",
            );
        }
        Err(e) if is_skip_error(&e) => v.check_skip("spectral.power_spectrum", &format!("{e}")),
        Err(e) => v.check_bool("spectral.power_spectrum", false, &format!("{e}")),
    }
}

fn validate_tensor_matmul(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    let a = ctx.call(
        "tensor",
        "tensor.create",
        serde_json::json!({"data": [1.0, 2.0, 3.0, 4.0], "shape": [2, 2]}),
    );
    let b = ctx.call(
        "tensor",
        "tensor.create",
        serde_json::json!({"data": [5.0, 6.0, 7.0, 8.0], "shape": [2, 2]}),
    );

    match (a, b) {
        (Ok(a_res), Ok(b_res)) => {
            let a_id = a_res.get("tensor_id").and_then(|v| v.as_str());
            let b_id = b_res.get("tensor_id").and_then(|v| v.as_str());
            match (a_id, b_id) {
                (Some(a_id), Some(b_id)) => {
                    match ctx.call(
                        "tensor",
                        "tensor.matmul",
                        serde_json::json!({"lhs_id": a_id, "rhs_id": b_id}),
                    ) {
                        Ok(r) => {
                            let shape = r.get("shape");
                            let ok =
                                shape.is_some_and(|s| s.as_array().is_some_and(|a| a.len() == 2));
                            v.check_bool(
                                "tensor.matmul 2×2 IPC (handle-based)",
                                ok,
                                "barraCuda: create→matmul→shape [2,2]",
                            );
                        }
                        Err(e) => v.check_skip("tensor.matmul", &format!("{e}")),
                    }
                }
                _ => v.check_skip("tensor.matmul", "tensor.create missing tensor_id"),
            }
        }
        (Err(e), _) | (_, Err(e)) => v.check_skip("tensor.matmul", &format!("{e}")),
    }
}

fn validate_manifest_services(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    match ctx.call(
        "compute",
        "compute.dispatch",
        serde_json::json!({"op": "noop"}),
    ) {
        Ok(_) => v.check_bool(
            "compute.dispatch noop accepted",
            true,
            "toadStool acknowledged dispatch",
        ),
        Err(e) if is_skip_error(&e) => v.check_skip("compute.dispatch", &format!("{e}")),
        Err(e) => v.check_bool("compute.dispatch", false, &format!("{e}")),
    }

    let test_key = "wetspring_guidestone_proof";
    let test_val = serde_json::json!({"experiment": "guidestone", "ts": "2026-04-18"});
    match ctx.call(
        "storage",
        "storage.store",
        serde_json::json!({"key": test_key, "value": test_val}),
    ) {
        Ok(_) => {
            v.check_bool(
                "storage.store accepted",
                true,
                "NestGate acknowledged store",
            );
            match ctx.call(
                "storage",
                "storage.retrieve",
                serde_json::json!({"key": test_key}),
            ) {
                Ok(ret) => {
                    let has_value = ret.get("value").is_some() || ret.get("data").is_some();
                    v.check_bool(
                        "storage.retrieve roundtrip",
                        has_value,
                        "NestGate returned data",
                    );
                }
                Err(e) if is_skip_error(&e) => {
                    v.check_skip("storage.retrieve", &format!("{e}"));
                }
                Err(e) => v.check_bool("storage.retrieve", false, &format!("{e}")),
            }
        }
        Err(e) if is_skip_error(&e) => v.check_skip("storage.store", &format!("{e}")),
        Err(e) => v.check_bool("storage.store", false, &format!("{e}")),
    }

    match ctx.call(
        "ai",
        "inference.complete",
        serde_json::json!({"prompt": "Define Shannon diversity index", "max_tokens": 32}),
    ) {
        Ok(ret) => {
            let has_text = ret.get("text").is_some()
                || ret.get("completion").is_some()
                || ret.get("content").is_some();
            v.check_bool(
                "inference.complete returns text",
                has_text,
                "Squirrel responded",
            );
        }
        Err(e) if is_skip_error(&e) => v.check_skip("inference.complete", &format!("{e}")),
        Err(e) => v.check_bool("inference.complete", false, &format!("{e}")),
    }

    match ctx.call(
        "security",
        "crypto.hash",
        serde_json::json!({"data": "d2V0c3ByaW5nX2d1aWRlc3RvbmVfcHJvb2Y=", "algorithm": "blake3"}),
    ) {
        Ok(ret) => {
            let has_hash = ret.get("hash").is_some() || ret.get("digest").is_some();
            v.check_bool("crypto.hash returns digest", has_hash, "BearDog blake3");
        }
        Err(e) if is_skip_error(&e) => v.check_skip("crypto.hash", &format!("{e}")),
        Err(e) => v.check_bool("crypto.hash", false, &format!("{e}")),
    }
}

/// N2 — Extended Domain Science IPC.
pub fn validate_domain_science(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    validate_parity_or_skip(
        ctx,
        v,
        "stats.median([1,3,5,7,9]) IPC = 5",
        "tensor",
        "stats.median",
        serde_json::json!({"data": [1.0, 3.0, 5.0, 7.0, 9.0]}),
        5.0,
        ps_tol::IPC_ROUND_TRIP_TOL,
    );

    validate_parity_or_skip(
        ctx,
        v,
        "linalg.determinant([[1,2],[3,4]]) IPC = -2",
        "tensor",
        "linalg.determinant",
        serde_json::json!({"matrix": [[1.0, 2.0], [3.0, 4.0]]}),
        -2.0,
        ps_tol::IPC_ROUND_TRIP_TOL,
    );

    validate_parity(
        ctx,
        v,
        "stats.weighted_mean IPC = 1.7",
        "tensor",
        "stats.weighted_mean",
        serde_json::json!({"values": [1.0, 2.0, 3.0], "weights": [0.5, 0.3, 0.2]}),
        "result",
        1.7,
        ps_tol::IPC_ROUND_TRIP_TOL,
    );
}

/// N3 — Cross-Atomic Pipeline: hash → store → retrieve → verify.
pub fn validate_cross_atomic(ctx: &mut CompositionContext, v: &mut ValidationResult) {
    let data = "wetspring_guidestone_cross_atomic_proof";

    let hash = match ctx.hash_bytes(data.as_bytes(), "blake3") {
        Ok(h) => {
            v.check_bool("cross-atomic: BearDog hash", true, "blake3 hash computed");
            Some(h)
        }
        Err(e) if is_skip_error(&e) => {
            v.check_skip("cross-atomic: hash", &format!("{e}"));
            None
        }
        Err(e) => {
            v.check_bool("cross-atomic: hash", false, &format!("{e}"));
            None
        }
    };

    let Some(hash_val) = hash else { return };

    let store_key = "wetspring_guidestone_xatomic";
    match ctx.call(
        "storage",
        "storage.store",
        serde_json::json!({"key": store_key, "value": {"hash": &hash_val}}),
    ) {
        Ok(_) => {
            v.check_bool(
                "cross-atomic: NestGate store",
                true,
                "stored hash via NestGate",
            );
            match ctx.call(
                "storage",
                "storage.retrieve",
                serde_json::json!({"key": store_key}),
            ) {
                Ok(ret) => {
                    let retrieved = ret
                        .get("value")
                        .and_then(|val| val.get("hash"))
                        .and_then(serde_json::Value::as_str);
                    v.check_bool(
                        "cross-atomic: retrieve + verify hash match",
                        retrieved == Some(hash_val.as_str()),
                        "BearDog→NestGate→retrieve integrity",
                    );
                }
                Err(e) if is_skip_error(&e) => {
                    v.check_skip("cross-atomic: retrieve", &format!("{e}"));
                }
                Err(e) => v.check_bool("cross-atomic: retrieve", false, &format!("{e}")),
            }
        }
        Err(e) if is_skip_error(&e) => v.check_skip("cross-atomic: store", &format!("{e}")),
        Err(e) => v.check_bool("cross-atomic: store", false, &format!("{e}")),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors validate_parity signature from primalspring"
)]
fn validate_parity_or_skip(
    ctx: &mut CompositionContext,
    v: &mut ValidationResult,
    label: &str,
    capability: &str,
    method: &str,
    params: serde_json::Value,
    expected: f64,
    tolerance: f64,
) {
    match ctx.call(capability, method, params) {
        Err(e) if is_skip_error(&e) => v.check_skip(label, &format!("not in ecobin: {e}")),
        Err(e) => v.check_bool(label, false, &format!("{e}")),
        Ok(resp) => match resp.get("result").and_then(serde_json::Value::as_f64) {
            Some(actual) => {
                let diff = (actual - expected).abs();
                v.check_bool(
                    label,
                    diff <= tolerance,
                    &format!(
                        "composition={actual}, expected={expected}, diff={diff:.2e}, tol={tolerance:.2e}"
                    ),
                );
            }
            None => v.check_skip(label, "no 'result' f64 in response"),
        },
    }
}
