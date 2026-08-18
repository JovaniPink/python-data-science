# Documentation guide

This directory separates dated ecosystem research, source rights and
provenance, and executed experiment evidence. Start here when reviewing the
repository without installing the Python environment.

## Reading paths

| Goal | Document | What it establishes |
|---|---|---|
| Understand the selected Python stack | [Python data science and machine learning in 2026](research/python-data-science-2026.md) | Dated version evidence, technique choices, deliberate exclusions, and validation boundaries |
| Review the source boundary | [BLS Public Data API source record](data-sources/bls-public-data-api.md) | Series definitions, terms, retrieval behavior, transformations, missing-source treatment, and claim limits |
| Review the observed result | [Python BLS macro clustering run record](experiments/bls-macro-clustering.md) | The exact executed configuration, observed profiles, sensitivity diagnostics, and bounded interpretation |
| Reproduce interactively | [BLS macro clustering marimo notebook](../notebooks/bls_macro_clustering.py) | The reactive, executable notebook backed by the repository lockfile |

The root [README](../README.md) remains the operational entry point for setup,
validation, CLI execution, and opening marimo.

## Cross-language replication

The sibling
[Elixir data-science repository](https://github.com/JovaniPink/elixir-data-science)
independently implements the same bounded question with Explorer, Scholar,
VegaLite, and Livebook. Its
[documentation guide](https://github.com/JovaniPink/elixir-data-science/blob/main/docs/README.md)
links the Elixir ecosystem, source, and run records.

The two implementations intentionally share:

- first-party BLS series `CUUR0000SA0` and `LNS14000000`;
- the January 2006 through December 2025 requested sample;
- 12-month CPI-U change and unemployment-rate level as features;
- explicit exclusion, without imputation, of unavailable October 2025 data;
- population standardization, three K-means clusters, seed 42, and 20 starts;
- neutral profile reporting and the same non-causal, non-predictive claim
  boundary.

They do not share runtime code, dataframe libraries, clustering
implementations, or notebook systems. The observed profile groups agree up to
arbitrary cluster-label permutation. That is an implementation cross-check,
not evidence that the clusters are causal, optimal, stable under new data, or
economically real.

The Python sensitivity table compares `k=2..6` across five seeds using
silhouette and pairwise adjusted Rand index. It is additional evidence about
this locked sample and implementation, not a claim that the Elixir run
performed the same diagnostic or that three clusters are universally correct.

The broader
[Awesome Economic Data catalog](https://github.com/JovaniPink/awesome-economic-data)
is useful for discovering candidate sources. Inclusion in that catalog is not
permission to retrieve, train on, redistribute, or publish a source.

## Documentation contract

- Keep research snapshots dated and distinguish observed package state from
  permanent compatibility claims.
- Record source terms, access limits, retrieval dates, transformations, missing
  values, and permitted use before adding a dataset.
- Keep source observations, repository-derived transformations, model output,
  and interpretation visibly separate.
- Add a new run record rather than silently replacing historical output when a
  source revision changes results.
- Treat links to sibling implementations as comparison context, not as shared
  validation or evidence of causal meaning.
