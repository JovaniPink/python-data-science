"""Reactive marimo notebook for the first-party BLS macro experiment."""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from python_data_science.analysis import (
        cluster_observations,
        derive_observations,
        diagnose_cluster_choices,
    )
    from python_data_science.bls import BLS_DISCLAIMER, fetch_bls_dataset
    from python_data_science.charts import cluster_scatter_chart, timeline_chart

    return (
        BLS_DISCLAIMER,
        cluster_observations,
        cluster_scatter_chart,
        derive_observations,
        diagnose_cluster_choices,
        fetch_bls_dataset,
        mo,
        timeline_chart,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # BLS inflation and unemployment clustering

    Can K-means separate recurring combinations of observed U.S. CPI inflation
    and unemployment in a fixed 2006-2025 sample?

    This is an **ex-post descriptive exercise**. It does not identify true
    economic regimes, explain causality, predict recessions, or produce an
    investment signal. Cluster IDs are arbitrary.
    """)
    return


@app.cell
def _(mo):
    cluster_count = mo.ui.slider(2, 6, value=3, step=1, label="Illustrative clusters")
    mo.hstack([cluster_count], justify="start")
    return (cluster_count,)


@app.cell
def _(derive_observations, fetch_bls_dataset):
    dataset = fetch_bls_dataset(2006, 2025)
    observations = derive_observations(dataset)
    return dataset, observations


@app.cell
def _(BLS_DISCLAIMER, dataset, mo, observations):
    unavailable_count = sum(len(points) for points in dataset.unavailable.values())
    mo.md(
        f"""
        ## Retrieval manifest

        - Source: `{dataset.source_url}`
        - Series: `CUUR0000SA0`, `LNS14000000`
        - Retrieved (UTC): `{dataset.retrieved_at.isoformat()}`
        - Anonymous request windows: `{dataset.request_windows}`
        - Aligned observations: `{observations.height}`
        - Unavailable source records retained: `{unavailable_count}`
        - API messages: `{dataset.messages}`

        > {BLS_DISCLAIMER}
        """
    )
    return


@app.cell
def _(dataset, mo):
    unavailable_rows = [
        {
            "series_id": series_id,
            "date": point.date.isoformat(),
            "source_value": point.value,
            "source_footnotes": " | ".join(point.footnotes),
        }
        for series_id, points in dataset.unavailable.items()
        for point in points
    ]
    mo.vstack(
        [
            mo.md("### Unavailable source records (retained; not imputed)"),
            mo.ui.table(unavailable_rows),
        ]
    )
    return


@app.cell
def _(mo, observations):
    mo.vstack(
        [
            mo.md("## Repository-derived observations"),
            mo.ui.table(observations.head(12)),
        ]
    )
    return


@app.cell
def _(
    cluster_count,
    cluster_observations,
    diagnose_cluster_choices,
    observations,
):
    analysis = cluster_observations(
        observations, n_clusters=cluster_count.value, seed=42, n_init=20
    )
    diagnostics = diagnose_cluster_choices(observations, n_init=20)
    return analysis, diagnostics


@app.cell
def _(analysis, diagnostics, mo):
    output_note = mo.md(
        f"""
## Descriptive model output

Fixed seed: `{analysis.metrics.seed}` · repeated starts:
`{analysis.metrics.n_init}` · inertia: `{analysis.metrics.inertia:.4f}` ·
silhouette: `{analysis.metrics.silhouette:.4f}`

The diagnostics compare separation and label-invariant stability. They do
not automatically select an economically true cluster count.
"""
    )
    mo.vstack([output_note, mo.ui.table(analysis.profiles), mo.ui.table(diagnostics)])
    return


@app.cell
def _(analysis, cluster_scatter_chart, mo, timeline_chart):
    mo.vstack(
        [
            mo.md("## Vega-Lite views"),
            mo.ui.altair_chart(timeline_chart(analysis.observations)),
            mo.ui.altair_chart(cluster_scatter_chart(analysis.observations)),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation checklist

    1. Show the retrieval timestamp, API messages, and unavailable source values.
    2. Do not impute the missing October 2025 values in this experiment.
    3. Report sample, features, standardization, cluster count, seed, and starts.
    4. Treat silhouette and adjusted Rand index as diagnostics, not economic truth.
    5. Acknowledge serial correlation, overlapping CPI windows, and revisions.
    6. Make no causal, recession-prediction, trading, or financial-advice claim.
    """)
    return


if __name__ == "__main__":
    app.run()
