// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generate the CHECKSUMS manifest for wetspring_guidestone P3 verification.
//!
//! Usage: `cargo run --features guidestone --example gen_checksums > validation/CHECKSUMS`

fn main() {
    let files: &[&str] = &[
        "barracuda/src/bin/wetspring_guidestone.rs",
        "barracuda/src/niche.rs",
        "barracuda/src/tolerances/mod.rs",
        "barracuda/src/lib.rs",
        "barracuda/Cargo.toml",
        "Cargo.toml",
    ];

    let root = std::path::Path::new(".");
    let manifest = primalspring::checksums::generate_manifest(root, files);
    println!("# wetSpring guideStone CHECKSUMS — BLAKE3");
    println!("# Generated: {}", chrono_free_date());
    println!("# Files: {}", files.len());
    println!("#");
    println!("# Verify: primalspring::checksums::verify_manifest()");
    println!("{manifest}");
}

fn chrono_free_date() -> String {
    let output = std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_owned());
    output.trim().to_owned()
}
