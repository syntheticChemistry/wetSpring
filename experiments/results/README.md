# Experiment Results Directory

**Layout:** Each subdirectory corresponds to an experiment by number and short name.

## Directory Structure

```
results/
├── NNN_experiment_name/     Numbered experiment directories
│   └── *_python_baseline.json   Python reference output (frozen baselines)
├── paper_benchmarks/        Paper data extracted for validation targets
├── python_16s_controls/     16S pipeline control baselines
├── qs_ode_baseline/         QS ODE time-series baselines
├── ncbi_dataset_search/     NCBI dataset search results
└── track2_validation_report.json   Track 2 (LC-MS/PFAS) top-level report
```

## Provenance

Every `*_python_baseline.json` traces to a Python script in `scripts/` and a Rust
validation binary in `barracuda/src/bin/validate_*.rs`. The mapping is documented in
`scripts/BASELINE_MANIFEST.md`.

### Reproduction

```bash
python3 scripts/<script>.py              # regenerate baseline
diff <(python3 scripts/<script>.py) experiments/results/<exp>/baseline.json
cargo run --bin validate_<experiment>     # Rust must match within tolerance
```

### Freeze

Baselines were frozen at commit `48fb787` (2026-02-23). SPDX headers and provenance
metadata were added in Phase 61 (2026-02-27). Content hashes in `BASELINE_MANIFEST.md`
reflect the post-SPDX state.

Drift verification: `./scripts/verify_baseline_integrity.sh`

## File Conventions

| Pattern | Purpose |
|---------|---------|
| `*_python_baseline.json` | Frozen Python reference output |
| `*_report.json` | Validation summary (pass/fail, metrics) |
| `*.tsv` | Tabular data (manifests, paper tables) |
| `*.dat` | Signal/peak test vectors |

## Paper Benchmarks

`paper_benchmarks/` contains data extracted from published papers for use as
validation targets. See `paper_benchmarks/README.md` for details on provenance
and data availability per paper.
