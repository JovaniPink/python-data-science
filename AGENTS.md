# Repository guidance

## Purpose

This repository contains small, reviewable Python data-science experiments.
The first experiment uses first-party BLS economic data, Polars dataframes,
scikit-learn clustering, Vega-Altair views, and a marimo notebook.

## Working rules

- Keep supported and tested Python versions, `pyproject.toml`, `uv.lock`, tests,
  and exact run instructions synchronized.
- Do not commit datasets until their source, license, retrieval date, and
  permitted use are documented.
- Keep observed source data, repository-derived transformations, model output,
  and interpretation visibly distinct.
- Do not turn descriptive clusters into causal, predictive, recession, or
  financial claims without a separately reviewed validation design.
- Prefer deterministic pipelines, explicit seeds, label-invariant diagnostics,
  and synthetic fixtures in automated tests.
- Keep generated outputs, local environments, and credentials out of Git.
- Stage explicit files and preserve unrelated work.

## Quality gate

Run the locked repository contract before review:

```bash
uv sync --locked
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy src
uv run --locked pytest
uv run --locked marimo check --strict notebooks/bls_macro_clustering.py
uv build --no-build-isolation --clear
uv export --quiet --locked --no-dev --no-emit-project --format requirements-txt \
  --output-file /tmp/python-data-science-runtime.txt
uv run --locked pip-audit --disable-pip --require-hashes \
  -r /tmp/python-data-science-runtime.txt
uv export --quiet --locked --all-groups --no-emit-project \
  --format requirements-txt --output-file /tmp/python-data-science-all.txt
uv run --locked pip-audit --disable-pip --require-hashes \
  -r /tmp/python-data-science-all.txt
```

The build uses the synchronized, locked `uv_build` backend. The dependency audits
cover the exact exported runtime graph and the complete development/build graph.
They are not a source-code, operating-system, container, or live-data security
assessment.
