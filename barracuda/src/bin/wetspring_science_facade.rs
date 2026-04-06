// SPDX-License-Identifier: AGPL-3.0-or-later
#![forbid(unsafe_code)]
//! wetspring-science-facade — HTTP gateway to wetSpring IPC for primals.eco.
//!
//! Thin Axum service that translates browser REST calls into wetSpring
//! JSON-RPC, shapes responses into petalTongue `DataChannel`-compatible
//! JSON, and attaches progressive provenance metadata.
//!
//! # Usage
//!
//! ```text
//! wetspring_science_facade [--bind 0.0.0.0:3100] [--cors https://primals.eco]
//! ```
//!
//! # Environment
//!
//! - `WETSPRING_SOCKET` — Override wetSpring IPC socket path
//! - `FACADE_BIND` — HTTP bind address (default: `127.0.0.1:3100`)
//! - `FACADE_CORS_ORIGIN` — Allowed CORS origin (default: `https://primals.eco`)
//! - `FAMILY_ID` — biomeOS family ID (for Neural API provenance trio access)

use axum::{Router, routing::get};
use tower_http::cors::{Any, CorsLayer};

use wetspring_barracuda::facade::{dark_forest, routes};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cors_origin = std::env::var("FACADE_CORS_ORIGIN")
        .unwrap_or_else(|_| "https://primals.eco".to_string());

    let cors = if cors_origin == "*" {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        let origin: axum::http::HeaderValue = cors_origin.parse().unwrap_or_else(|_| {
            "https://primals.eco".parse().expect("valid header")
        });
        CorsLayer::new()
            .allow_origin(origin)
            .allow_methods(Any)
            .allow_headers(Any)
    };

    let gate_config = dark_forest::DarkForestConfig::from_env();
    let gate_state = dark_forest::DarkForestState::new(gate_config.clone());

    tracing::info!(
        "Dark Forest gate: {}",
        if gate_config.enabled { "ENABLED" } else { "DISABLED" }
    );
    if gate_config.enabled {
        tracing::info!(
            "Dark Forest neural socket: {}",
            gate_config.neural_api_socket.as_deref().unwrap_or("not found")
        );
    }

    let app = Router::new()
        .route("/api/v1/health", get(routes::health))
        .route(
            "/api/v1/science/gonzales/dose-response",
            get(routes::dose_response),
        )
        .route(
            "/api/v1/science/gonzales/pk-decay",
            get(routes::pk_decay),
        )
        .route(
            "/api/v1/science/gonzales/tissue-lattice",
            get(routes::tissue_lattice),
        )
        .route(
            "/api/v1/science/anderson/hormesis",
            get(routes::hormesis),
        )
        .route(
            "/api/v1/science/anderson/cross-species",
            get(routes::cross_species),
        )
        .route(
            "/api/v1/science/anderson/biome-atlas",
            get(routes::biome_atlas),
        )
        .route(
            "/api/v1/science/anderson/disorder-sweep",
            get(routes::disorder_sweep),
        )
        .route(
            "/api/v1/science/gonzales/full",
            get(routes::full_dashboard),
        )
        .route(
            "/api/v1/render/gonzales/dose-response",
            get(routes::grammar_dose_response),
        )
        .route(
            "/api/v1/render/gonzales/pk-decay",
            get(routes::grammar_pk_decay),
        )
        .route(
            "/api/v1/render/gonzales/tissue-lattice",
            get(routes::grammar_tissue_lattice),
        )
        .route(
            "/api/v1/render/anderson/hormesis",
            get(routes::grammar_hormesis),
        )
        .route(
            "/api/v1/render/anderson/cross-species",
            get(routes::grammar_cross_species),
        )
        .route(
            "/api/v1/provenance/{result_id}",
            get(routes::provenance_query),
        )
        .route(
            "/api/v1/validation/chain/{paper_id}",
            get(routes::validation_chain),
        )
        .route(
            "/api/v1/system/composition",
            get(routes::system_composition),
        )
        .layer(axum::middleware::from_fn_with_state(
            gate_state,
            dark_forest::dark_forest_middleware,
        ))
        .layer(cors);

    let bind = std::env::var("FACADE_BIND").unwrap_or_else(|_| "127.0.0.1:3100".to_string());
    let listener = tokio::net::TcpListener::bind(&bind).await.unwrap();
    tracing::info!("wetspring-science-facade listening on {bind}");
    tracing::info!(
        "wetspring IPC socket: {}",
        wetspring_barracuda::facade::ipc_client::socket_path().display()
    );

    axum::serve(listener, app).await.unwrap();
}
