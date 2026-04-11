// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dark Forest gate middleware for the science facade.
//!
//! Mirrors the biomeOS `dark_forest_gate_middleware` pattern: every request
//! must present a valid `X-Dark-Forest-Token` header, verified via
//! `birdsong.decrypt` on the Neural API socket. Health and `.well-known/`
//! paths are exempt.
//!
//! Disable with `FACADE_DARK_FOREST=false` for local development.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use serde_json::{Value, json};
use tokio::sync::RwLock;

const BYPASS_PREFIXES: &[&str] = &["/.well-known/"];

const BARE_OK_PATHS: &[&str] = &[
    "/health",
    "/api/v1/health",
    "/api/v1/health/ready",
    "/api/v1/health/live",
];

const TOKEN_HEADER: &str = "x-dark-forest-token";

const CACHE_TTL_SECS: u64 = 300;

/// Gate configuration resolved from environment variables.
#[derive(Debug, Clone)]
pub struct DarkForestConfig {
    /// Whether the gate is enabled. Default `true`.
    pub enabled: bool,
    /// Family ID for `birdsong.decrypt` context.
    pub family_id: String,
    /// Neural API socket path (discovered at startup).
    pub neural_api_socket: Option<String>,
}

impl DarkForestConfig {
    /// Build config from environment.
    pub fn from_env() -> Self {
        let enabled = std::env::var("FACADE_DARK_FOREST")
            .map(|v| !matches!(v.as_str(), "false" | "0" | "no"))
            .unwrap_or(true);

        let family_id = std::env::var("FAMILY_ID").unwrap_or_else(|_| "default".into());

        let neural_api_socket = {
            let runtime = std::env::var("XDG_RUNTIME_DIR")
                .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
            let path = std::path::PathBuf::from(runtime)
                .join("biomeos")
                .join(format!("neural-api-{family_id}.sock"));
            if path.exists() {
                Some(path.to_string_lossy().into_owned())
            } else {
                None
            }
        };

        Self {
            enabled,
            family_id,
            neural_api_socket,
        }
    }
}

/// Shared state for the Dark Forest middleware.
#[derive(Debug, Clone)]
pub struct DarkForestState {
    /// Gate configuration.
    pub config: DarkForestConfig,
    /// Recently verified tokens: token → expiry timestamp.
    verified_cache: Arc<RwLock<HashMap<String, u64>>>,
}

impl DarkForestState {
    /// Create a new gate state from config.
    #[must_use]
    pub fn new(config: DarkForestConfig) -> Self {
        Self {
            config,
            verified_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn verify_token(&self, token: &str) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        {
            let cache = self.verified_cache.read().await;
            if let Some(&expiry) = cache.get(token) {
                if now < expiry {
                    return true;
                }
            }
        }

        let verified = verify_via_neural_api(
            self.config.neural_api_socket.as_deref(),
            &self.config.family_id,
            token,
        )
        .await;

        if verified {
            let mut cache = self.verified_cache.write().await;
            cache.insert(token.to_string(), now + CACHE_TTL_SECS);
            cache.retain(|_, &mut exp| exp > now);
        }

        verified
    }
}

/// Axum middleware that enforces Dark Forest token verification.
pub async fn dark_forest_middleware(
    State(gate): State<DarkForestState>,
    request: Request<Body>,
    next: Next,
) -> Response<Body> {
    if !gate.config.enabled {
        return next.run(request).await;
    }

    let path = request.uri().path();

    for bare in BARE_OK_PATHS {
        if path == *bare || path.starts_with(&format!("{bare}/")) {
            return Response::builder()
                .status(StatusCode::OK)
                .body(Body::empty())
                .unwrap_or_else(|_| Response::new(Body::empty()));
        }
    }

    for prefix in BYPASS_PREFIXES {
        if path.starts_with(prefix) {
            return next.run(request).await;
        }
    }

    let token = request
        .headers()
        .get(TOKEN_HEADER)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if token.is_empty() || !gate.verify_token(token).await {
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(Body::empty())
            .unwrap_or_else(|_| Response::new(Body::empty()));
    }

    next.run(request).await
}

async fn verify_via_neural_api(socket_path: Option<&str>, family_id: &str, token: &str) -> bool {
    let Some(socket) = socket_path else {
        tracing::warn!("Dark Forest: no Neural API socket — rejecting token");
        return false;
    };

    let params = json!({
        "family_id": family_id,
        "ciphertext": token,
    });

    call_neural_async(socket, "birdsong.decrypt", &params)
        .await
        .map_or_else(
            || {
                tracing::warn!("Dark Forest: birdsong.decrypt call failed");
                false
            },
            |result| parse_decrypt_result(&result),
        )
}

fn parse_decrypt_result(value: &Value) -> bool {
    let success = value
        .get("success")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let plaintext = value.get("plaintext").and_then(Value::as_str).unwrap_or("");
    success && !plaintext.is_empty()
}

async fn call_neural_async(socket_path: &str, method: &str, params: &Value) -> Option<Value> {
    let path = socket_path.to_string();
    let request = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    });
    let payload = serde_json::to_string(&request).ok()?;

    tokio::task::spawn_blocking(move || {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixStream;

        let mut stream = UnixStream::connect(&path).ok()?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .ok();

        let mut line = payload;
        line.push('\n');
        stream.write_all(line.as_bytes()).ok()?;
        stream.flush().ok()?;

        let mut reader = BufReader::new(stream);
        let mut resp_line = String::new();
        reader.read_line(&mut resp_line).ok()?;

        let resp: Value = serde_json::from_str(resp_line.trim()).ok()?;
        resp.get("result").cloned()
    })
    .await
    .ok()?
}
