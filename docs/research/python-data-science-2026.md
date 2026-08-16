# Python data science and machine learning in 2026

_Verified August 16, 2026._

Python remains the broadest general-purpose data-science and machine-learning
ecosystem. This repository uses that breadth selectively: the smallest current
stack that can retrieve, transform, test, model, diagnose, visualize, and
reproduce this economic analysis without hiding source or claim boundaries.

## Selected stack

| Need | Selected tool | Verified current line | Reason here |
|---|---|---:|---|
| Runtime | [Python](https://www.python.org/downloads/release/python-3146/) | 3.14.6 stable | Current stable CPython; 3.15 is still pre-release |
| Project and lock | [uv](https://docs.astral.sh/uv/concepts/projects/) | 0.12.3 | Fast environment management, PEP 735 dependency groups, cross-platform lock |
| Arrays | [NumPy](https://numpy.org/) | 2.5.2 | Interchange layer for scikit-learn's numerical estimators |
| Dataframes | [Polars](https://docs.pola.rs/) | 1.43.2 | Typed columnar expressions and predictable joins |
| Classical ML | [scikit-learn](https://scikit-learn.org/stable/) | 1.9.0 | Pipelines, K-means, silhouette, and adjusted Rand index |
| Visualization | [Vega-Altair](https://altair-viz.github.io/) | 6.2.2 | Declarative Vega-Lite 6 specifications |
| Notebook | [marimo](https://docs.marimo.io/) | 0.23.16 | Reactive, pure-Python, Git-friendly, executable notebook |
| HTTP | [HTTPX](https://www.python-httpx.org/) | 0.28.1 | Timeouts, typed client API, explicit status handling |
| Quality | [Ruff](https://docs.astral.sh/ruff/), [mypy](https://mypy.readthedocs.io/), [pytest](https://docs.pytest.org/) | 0.16.3 / 2.3.1 / 9.1.1 | Formatting/linting, strict static analysis, synthetic tests |

Exact resolved versions and hashes belong to `uv.lock`; compatible lower and
upper bounds belong to `pyproject.toml`. This avoids presenting an August 2026
research snapshot as a timeless package claim.

## Technique choices

### One fitted preprocessing-and-model pipeline

`StandardScaler` and `KMeans` live in one scikit-learn `Pipeline`. That prevents
training on one transformation while reporting another and keeps learned
means, scales, centroids, inertia, and assignments together.

### Repeated starts, not a single lucky initialization

The primary fit uses k-means++ initialization, 20 starts, a fixed seed, and the
Lloyd algorithm. The choices are explicit for cross-run review. Repeated starts
reduce sensitivity to one initialization; they do not validate `k=3`.

### Label-invariant sensitivity evidence

The repository compares `k=2..6` across five seeds. It reports silhouette for
geometric separation and pairwise adjusted Rand index for assignment stability.
ARI is appropriate here because it is invariant to arbitrary cluster-label
permutations. The table is evidence about this sample and feature space, not an
automatic regime-discovery oracle.

### Time structure remains visible

The notebook shows time series beside the feature-space scatter. No shuffled
cross-validation is claimed: this is an unsupervised ex-post description, and
monthly observations have serial dependence plus overlapping 12-month CPI
windows. A predictive follow-up would require a separate time-aware validation
and leakage design.

### Reactive notebooks without hidden execution order

marimo represents the notebook as executable Python and derives cell order from
dependencies. Changing the cluster-count control recomputes downstream model,
profiles, diagnostics, and charts. Retrieval metadata and the source disclaimer
remain visible in the same artifact.

### Network-closed automated validation

Tests inject deterministic synthetic BLS-shaped responses. Live retrieval is a
separate documented verification step, so API availability, source revisions,
or future missing months do not become nondeterministic CI behavior.

## Deliberate exclusions

- Deep learning, boosted trees, feature stores, distributed compute, and GPU
  frameworks add no analytical value to a two-feature, 227-row descriptive fit.
- pandas is not required merely for ecosystem familiarity; Polars provides all
  dataframe operations used here and Altair receives explicit inline records.
- Free-threaded Python 3.14 is supported by CPython, but this project uses the
  standard build. Its numerical work already executes in native libraries, and
  a free-threaded deployment should be adopted only after dependency-specific
  testing.
- October 2025 is not interpolated. Preserving the official source gap is more
  faithful than manufacturing a value for an illustrative clustering exercise.
- No causal labels, recession labels, forecasting target, trading strategy, or
  financial recommendation is derived from the clusters.
