# Public Notebook Pattern — wetSpring Exemplar

How to create public-facing notebooks for your spring. This directory
is the reference implementation that other springs absorb.

## Directory Convention

```
your-spring/
  notebooks/
    NOTEBOOK_PATTERN.md          ← this file (copy to your spring)
    01-domain-validation.ipynb   ← flagship validation story
    02-benchmark-comparison.ipynb← Python vs Rust vs GPU
    03-paper-reproductions.ipynb ← per-researcher evidence
    04-cross-spring.ipynb        ← ecosystem connections
    05-domain-deep-dive.ipynb    ← your most compelling discovery
```

## Cell Structure

Every notebook follows the same structure:

1. **Title cell** (markdown): Title, one-paragraph context, data sources, "for other springs" adaptation note
2. **Imports + data loading** (code): Load from `../experiments/results/*.json`
3. **Domain-specific cells** (code + markdown): Visualization and analysis
4. **Summary cell** (markdown): Validation table, provenance note, links to primals.eco

## Data Loading Pattern

```python
import json
from pathlib import Path

RESULTS = Path('..') / 'experiments' / 'results'

def load(path):
    with open(RESULTS / path) as f:
        return json.load(f)

data = load('001_experiment/validation_report.json')
```

Notebooks load **frozen data** (committed JSON artifacts), not live API responses.
This means they work without primals running. When Tier 2 JSON-RPC APIs are
available, notebooks can also call primals directly (see Tier 2 stubs).

## Visualization Standards

- Use `matplotlib` (available everywhere, renders to static PNG)
- Save figures to `/tmp/<spring>_<notebook>_<chart>.png`
- Use `matplotlib.use('Agg')` for headless rendering
- Color palette: `#2ecc71` (pass/ok), `#e74c3c` (fail), `#3498db` (info)
- Always include chart titles with key numbers

## Publishing Pipeline

1. **Git**: Notebooks committed to `your-spring/notebooks/`
2. **JupyterHub**: Symlinked into `shared/abg/commons/<spring>-public/notebooks/`
3. **sporePrint**: `render_notebooks.sh` converts to Zola pages at `/lab/`
4. **primals.eco**: Static HTML rendered from notebook output

## What Makes a Good Notebook

- **Tells a story**: Not just charts — narrative flow from question to evidence
- **Uses frozen data**: Reproducible without live infrastructure
- **Links to primals.eco**: Cross-references other lab pages and science pages
- **Includes adaptation note**: Header tells other springs how to customize
- **Shows provenance**: Final cell links to provenance pipeline

## Adapting for Your Spring

1. Copy this directory structure
2. Replace data paths with your `experiments/results/` JSONs
3. Update the narrative for your domain
4. Keep the cell structure (title → load → analyze → summary)
5. Add your spring to `shared/abg/commons/<spring>-public/notebooks/` symlink
