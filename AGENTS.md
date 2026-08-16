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
