# Python BLS macro clustering run record

_Executed August 16, 2026 at 20:12:38 UTC._

## Question and claim boundary

Can K-means separate recurring combinations of observed U.S. CPI inflation and
unemployment in a fixed 2006–2025 sample?

This is an ex-post descriptive exercise. The run does not establish causal
economic regimes, classify recessions, forecast future data, or produce a
trading or financial-advice signal.

## Run manifest

| Item | Value |
|---|---|
| Runtime | CPython 3.14.6 |
| Source | BLS Public Data API |
| Source series | `CUUR0000SA0`, `LNS14000000` |
| Requested years | 2006–2025 |
| Anonymous API windows | 2006–2015, 2016–2025 |
| Derived features | 12-month CPI-U change, unemployment-rate level |
| Standardization | scikit-learn `StandardScaler` population mean and scale |
| K-means | scikit-learn 1.9.0, k-means++, Lloyd, 3 clusters, seed 42, 20 starts |
| Aligned observations | 227 months |
| Missing source month | October 2025 in both series; retained and not imputed |
| Inertia | 129.7056 in standardized feature space |
| Silhouette | 0.5236 |

BLS's official [CPI notice](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)
and [Current Population Survey notice](https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm)
attribute the October 2025 gap to the 2025 lapse in appropriations. The Python
client retained both API records and footnotes and excluded the month through
an explicit inner alignment. It performed no interpolation.

## Observed profile summaries

Cluster numbers are implementation labels, not ordered economic categories.

| Python cluster ID | Months | Mean 12-month CPI inflation | Mean unemployment rate |
|---|---:|---:|---:|
| 2 | 74 | 1.48% | 8.63% |
| 0 | 125 | 2.20% | 4.54% |
| 1 | 28 | 6.64% | 4.31% |

The profile groups and inertia reproduce the independently executed Elixir
experiment. The middle and higher-inflation profile IDs are permuted between
implementations, which is expected: raw K-means labels carry no identity.
Cross-language agreement here is a useful implementation check, not evidence
that the clusters are causal, optimal, stable under new data, or economically
real.

The sibling
[Elixir run record](https://github.com/JovaniPink/elixir-data-science/blob/main/docs/experiments/bls-macro-clustering.md)
contains the independently executed comparison and its Elixir runtime manifest.

## Sensitivity diagnostics

Each row repeats k-means++ with 20 starts under seeds `0`, `1`, `2`, `42`, and
`99`. Pairwise adjusted Rand index (ARI) compares assignments without assuming
that cluster numbers match between runs.

| k | Mean silhouette | Silhouette range | Mean pairwise ARI | Minimum pairwise ARI | Mean inertia |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.4535 | 0.4535–0.4535 | 1.0000 | 1.0000 | 241.1428 |
| 3 | 0.5236 | 0.5236–0.5236 | 1.0000 | 1.0000 | 129.7056 |
| 4 | 0.4537 | 0.4537–0.4537 | 1.0000 | 1.0000 | 97.9655 |
| 5 | 0.4787 | 0.4784–0.4791 | 0.9965 | 0.9941 | 74.4813 |
| 6 | 0.4955 | 0.4945–0.4970 | 0.9868 | 0.9780 | 55.6608 |

In this bounded comparison, `k=3` has the highest mean silhouette and identical
assignments across the tested seeds. That supports using three clusters as a
stable illustration for this locked sample and implementation. It does not
establish three as the true number of economic regimes: the candidate range is
small, observations are serially dependent, and the feature space contains
only inflation and unemployment.

Inertia falls as k increases by construction, so its decline is not independent
validation. The profile means also hide within-cluster dispersion, time order,
measurement revisions, and changing economic institutions.

## Reproduction

```bash
uv run --locked bls-macro-clustering
```

Source data can be revised, so a later run may differ. Every rerun must report
its own UTC timestamp, unavailable records, model configuration, and diagnostic
results rather than overwriting this record without review.

> BLS.gov cannot vouch for the data or analyses derived from these data after
> the data have been retrieved from BLS.gov.
