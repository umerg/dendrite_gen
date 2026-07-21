# Local external metric checkouts

This directory holds ignored local checkouts of external research metrics. Its
contents are not committed as embedded repositories.

## Elastic SRVFT audit record

- Expected checkout: `metrics/external/elastic_srvft/`
- Upstream: `https://github.com/martinalex000/complexTrees_distanceMetric`
- Audited revision: `903a82c8ae9ec8692fea85ea57803ba727b438a1`
- Python code: `python_distance/`
- Requirements: `python_distance/requirements.txt`
- Project adapter: `metrics/adapters/elastic_srvft.py`
- Declared license: CC BY-NC 4.0 in the upstream README; no standalone license
  file was present at the audited revision

The clone is ignored and therefore not pinned by this repository. The adapter
records its actual Git HEAD and dirty state in every result. The upstream has
no `setup.py` or `pyproject.toml`; install only its requirements into `trees`.
The adapter imports the exact checkout with `importlib`, rejects conflicting
`python_distance` modules, and never modifies `sys.path`.

The audited Python path performs elastic curve reparameterization and branch
assignment but no rotational alignment. The project adapter supplies only the
required relative SO(2) minimum around `z`; it never invokes full SO(3). The
reported scalar is the raw upstream alignment energy `E`, not an established
metric. The backend has a fixed four-branch-layer representation, radius does
not enter the current energy, and one full-neuron evaluation can take tens of
seconds. See `metrics/README.md` for the adapter's safety policies and full
interpretation notes.

The upstream README asks users to contact its authors for commercial uses or
derivatives. Do not vendor, redistribute, or use the code commercially without
resolving those terms.
