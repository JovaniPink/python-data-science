# Python Data Science Experiments

Small, reviewable experiments using a current Python data-science stack and
explicit source, provenance, validation, and claim boundaries.

## Current experiment

The first experiment recreates the sibling Elixir analysis in Python: it asks
whether K-means separates recurring combinations of observed U.S. CPI
inflation and unemployment in a fixed 2006–2025 sample.

The runtime retrieves first-party BLS data, derives 12-month CPI-U inflation
with Polars, standardizes both features in a scikit-learn pipeline, fits
K-means with a fixed seed and 20 starts, and renders Vega-Lite charts through
Vega-Altair. The marimo notebook is reactive, executable Python rather than a
stateful JSON notebook.

The result is descriptive and ex post. Cluster IDs are arbitrary; they are not
objective economic regimes, causal explanations, recession predictions,
trading signals, or financial advice.

## Why this is a 2026 Python stack

- Python 3.14.6 is the pinned default; CI also covers the maintained Python
  3.13 line.
- uv manages the project, standardized dependency groups, and a committed
  cross-platform lockfile.
- Polars supplies typed columnar transformations without introducing pandas as
  an implicit interchange layer.
- scikit-learn uses a `Pipeline` so standardization and K-means remain one
  auditable fitted object.
- Sensitivity diagnostics compare `k=2..6` over five seeds using silhouette
  and pairwise adjusted Rand index (ARI). ARI is label-invariant, which matters
  because K-means IDs can be permuted between runs.
- Vega-Altair 6 emits Vega-Lite 6 specifications, and marimo keeps the notebook
  reactive, Git-friendly, and checkable in CI.
- Ruff, mypy, pytest, synthetic BLS-shaped fixtures, and a two-version CI
  matrix test the code without making live network access a merge gate.

See the [dated Python ecosystem brief](docs/research/python-data-science-2026.md)
for the version evidence and design choices.
The [documentation guide](docs/README.md) provides a review path through the
research brief, source record, executed run, and sibling Elixir replication.

## Data source and rights boundary

The source is the [BLS Public Data API](https://www.bls.gov/developers/) using
CPI-U series `CUUR0000SA0` and the seasonally adjusted civilian unemployment
rate series `LNS14000000`. The
[source record](docs/data-sources/bls-public-data-api.md) documents access
limits, terms, attribution, retrieval behavior, transformations, and analysis
limitations.

FRED is not the training-data source. Its current legal terms prohibit using
FRED services or content in connection with developing or training machine-
learning systems, so this experiment retrieves the series directly from BLS
under the reviewed BLS terms.

No raw source response, model artifact, generated chart, or credential is
committed.

## Setup

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv python install 3.14.6
uv sync --locked
```

The broad supported range lives in `pyproject.toml`; `.python-version` pins
the default interpreter and `uv.lock` pins the complete dependency graph.

## Validate

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest
uv run --locked marimo check --strict notebooks/bls_macro_clustering.py
```

Automated tests use only synthetic BLS-shaped responses. A BLS outage or
revision therefore cannot silently change CI.

## Run the live analysis

```bash
uv run --locked bls-macro-clustering
```

The command makes two anonymous BLS requests, records the UTC retrieval time,
reports unavailable source records, prints bounded cluster profiles, and runs
the `k=2..6` sensitivity comparison. It writes no data file. Use
`--no-diagnostics` only when you deliberately want the direct three-cluster
reproduction without the robustness table.

The [dated run record](docs/experiments/bls-macro-clustering.md) separates
verified output from the code and live notebook. The August 16, 2026 Python
run produced 227 observations, retained the missing October 2025 records, and
reproduced the Elixir profile groups up to arbitrary cluster-label permutation.

## Open the reactive notebook

```bash
uv run --locked marimo edit notebooks/bls_macro_clustering.py
```

The notebook retrieves data at runtime, exposes the cluster count as an
interactive control, displays Polars tables, and renders linked Vega-Lite
views. It preserves the same source and claim boundaries as the CLI.

## Repository contents

- `src/python_data_science/` — BLS retrieval, Polars transformations,
  scikit-learn analysis, diagnostics, chart builders, and CLI.
- `tests/` — synthetic source fixtures and deterministic unit tests.
- `notebooks/bls_macro_clustering.py` — reactive marimo analysis.
- `docs/data-sources/` — source terms, provenance, and claim boundaries.
- `docs/experiments/` — dated executions and bounded interpretations.
- `docs/research/` — dated Python ecosystem and technique choices.
- `docs/README.md` — documentation map and cross-language replication boundary.
- `.github/workflows/ci.yml` — locked Python 3.13/3.14 quality matrix.

## License

Repository-authored material is available under the [MIT License](LICENSE).
Third-party data and dependencies remain subject to their own terms.
