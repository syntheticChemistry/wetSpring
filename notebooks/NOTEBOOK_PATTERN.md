# Public Notebook Pattern — wetSpring

How to create public-facing notebooks for wetSpring (and other springs).
Adapted from the primalSpring exemplar pattern with wetSpring's
three-tier evolution model.

## Evolution Model

Notebooks evolve through three tiers — all three coexist in the same file:

- **Tier 1 (frozen):** Load JSON from `experiments/results/`. Works offline,
  in CI, and on GitHub Pages. Ships now.
- **Tier 2 (live IPC):** When `WETSPRING_IPC_SOCKET` is set, call barracuda
  JSON-RPC handlers directly. Assert parity against frozen baselines.
- **Tier 3 (full composition):** biomeOS `capability.call` routing, provenance
  trio wrapping, petalTongue server-side rendering. The notebook becomes a
  live gAIa artifact.

## Directory Convention

```
your-spring/
  notebooks/
    NOTEBOOK_PATTERN.md          ← this file (copy to your spring)
    01-domain-validation.ipynb   ← flagship validation story
    02-benchmark-comparison.ipynb← Python vs Rust vs GPU
    03-domain-deep-dive.ipynb    ← your most compelling science
    04-cross-spring.ipynb        ← ecosystem connections
    05-composition-patterns.ipynb← primal composition evidence
```

## Cell Structure

Every notebook follows the same structure:

1. **Title cell** (markdown): Title, one-paragraph context, data sources,
   "for other springs" adaptation note
2. **Tier detection cell** (code): Detect `WETSPRING_IPC_SOCKET`, set `TIER`,
   define `ipc_call()` helper
3. **Imports + data loading** (code): Load from `../experiments/results/*.json`
4. **Domain-specific cells** (code + markdown): Visualization and analysis;
   Tier 2 branches with parity assertions
5. **Summary cell** (markdown): Validation table, provenance note, links to primals.eco

## Data Loading Pattern

```python
import json
from pathlib import Path

RESULTS = Path('..') / 'experiments' / 'results'

def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)

data = load('science_validation.json')
```

## Tier Detection Pattern

```python
import os, json, socket, struct

TIER = "frozen"
IPC_SOCKET = os.environ.get("WETSPRING_IPC_SOCKET")

def ipc_call(method, params=None):
    """JSON-RPC call to barracuda IPC."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(IPC_SOCKET)
    req = json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}, "id": 1})
    payload = req.encode()
    sock.sendall(struct.pack("<I", len(payload)) + payload)
    length = struct.unpack("<I", sock.recv(4))[0]
    data = sock.recv(length)
    sock.close()
    return json.loads(data)["result"]

if IPC_SOCKET and os.path.exists(IPC_SOCKET):
    try:
        ipc_call("health.check")
        TIER = "live_ipc"
        print(f"Tier 2 ACTIVE — live IPC via {IPC_SOCKET}")
    except Exception:
        print("Tier 2 socket found but not responding — using frozen data")
else:
    print(f"Tier 1 — frozen data (no IPC socket)")
```

## Frozen Data for wetSpring

| File | Contents |
|------|----------|
| `science_validation.json` | Test counts, science methods, validation chain, gap reports |
| `benchmark_timing.json` | 23-domain Python timing, Rust pipeline, energy estimates |
| `gonzales_domain.json` | IC50 Table 1, PK dose tiers, tissue lattice, ChEMBL hashes |
| `experiment_catalog.json` | 56 experiment dirs categorized by 6 domain tracks |
| `primal_composition.json` | Composition routing, provenance wire names, deploy graph |
| `cross_spring_matrix.json` | Path deps, primal consumption, cross-spring data exchange |

## Visualization Standards

- Use `matplotlib` (available everywhere, renders to static PNG)
- Save figures to `/tmp/wetspring_<notebook>_<chart>.png`
- Color palette: `#2ecc71` (pass/ok), `#e74c3c` (fail), `#3498db` (info)
- Always include chart titles with key numbers

## Adapting for Your Spring

1. Copy this directory structure
2. Replace data paths with your `experiments/results/` JSONs
3. Update the narrative for your domain
4. Keep the cell structure (title → tier detect → load → analyze → summary)
5. Include Tier 2 stubs with your IPC methods — the same notebook validates
   both frozen correctness and live composition parity
6. Add your spring to `shared/abg/commons/<spring>-public/notebooks/` symlink
