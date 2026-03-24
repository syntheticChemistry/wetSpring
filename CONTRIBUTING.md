# Contributing to wetSpring

Thank you for your interest in wetSpring — the life science and PFAS
analytical chemistry validation spring in the ecoPrimals ecosystem.

## Development Setup

```bash
# Clone (SSH)
git clone git@github.com:syntheticChemistry/wetSpring.git
cd wetSpring

# CPU-only build
cargo build

# Full build (GPU + IPC + vault)
cargo build --all-features

# Run all tests
cargo test --all-features
```

## Code Standards

- **Rust edition 2024**, MSRV 1.87
- `#![forbid(unsafe_code)]` — workspace-wide, no exceptions
- All `as` casts use `crate::cast::*` named helpers
- Error types derive `thiserror::Error`
- `clippy::nursery` enabled; zero warnings policy
- Every public function has `# Errors` / `# Panics` doc sections where applicable

## Testing

wetSpring validates against Python/Galaxy/QIIME2/R baselines.
Every algorithm has a corresponding `validate_*` binary.

```bash
# Run CPU test suite
cargo test

# Run a specific validator
cargo run --bin validate_diversity

# Run GPU tests (requires Vulkan-capable GPU)
cargo test --features gpu
```

## Pull Request Process

1. Branch from `main`
2. Keep commits focused and atomic
3. Run `cargo test --all-features` before pushing
4. Run `cargo clippy --all-features -- -D warnings`
5. Update CHANGELOG.md with your changes

## Architecture

- **Primal code has only self-knowledge** — discovers other primals at runtime
- **No hardcoded primal paths** — use `primal_names::*` constants + capability discovery
- **Mocks are test-only** — production code uses real implementations
- **Zero external C dependencies** in default build (ecoBin compliant)

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) for details.
