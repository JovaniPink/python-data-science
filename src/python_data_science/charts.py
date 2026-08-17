"""Declarative Vega-Lite charts built with Vega-Altair."""

from datetime import date
from typing import Any, cast

import altair as alt
import polars as pl


def timeline_chart(observations: pl.DataFrame) -> alt.VConcatChart:
    """Build vertically aligned inflation and unemployment time-series views."""
    data = alt.Data(values=_chart_rows(observations))  # type: ignore[no-untyped-call]
    inflation = (
        alt.Chart(data)
        .mark_line(color="#7c3aed")
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("inflation_yoy:Q", title="CPI-U inflation (12-month %)"),
            tooltip=[
                alt.Tooltip("date:T", title="Month"),
                alt.Tooltip("inflation_yoy:Q", title="Inflation", format=".2f"),
            ],
        )
        .properties(width=760, height=220)
    )
    unemployment = (
        alt.Chart(data)
        .mark_line(color="#0f766e")
        .encode(
            x=alt.X("date:T", title="Month"),
            y=alt.Y("unemployment_rate:Q", title="Unemployment rate (%)"),
            tooltip=[
                alt.Tooltip("date:T", title="Month"),
                alt.Tooltip("unemployment_rate:Q", title="Unemployment", format=".1f"),
            ],
        )
        .properties(width=760, height=220)
    )
    chart = (
        alt.vconcat(inflation, unemployment)
        .resolve_scale(x="shared")
        .properties(title="Observed BLS macro indicators by month")
    )
    return cast(alt.VConcatChart, chart)


def cluster_scatter_chart(observations: pl.DataFrame) -> alt.Chart:
    """Build a scatterplot of the two clustering inputs and arbitrary labels."""
    if "cluster" not in observations.columns:
        raise ValueError("cluster scatter requires labeled observations")
    data = alt.Data(values=_chart_rows(observations))  # type: ignore[no-untyped-call]
    chart = (
        alt.Chart(data)
        .mark_point(filled=True, size=55, opacity=0.78)
        .encode(
            x=alt.X("inflation_yoy:Q", title="CPI-U inflation (12-month %)"),
            y=alt.Y("unemployment_rate:Q", title="Civilian unemployment rate (%)"),
            color=alt.Color("cluster:N", title="Cluster ID"),
            tooltip=[
                alt.Tooltip("date:T", title="Month"),
                alt.Tooltip("inflation_yoy:Q", title="Inflation", format=".2f"),
                alt.Tooltip("unemployment_rate:Q", title="Unemployment", format=".1f"),
                alt.Tooltip("cluster:N", title="Cluster ID"),
            ],
        )
        .properties(
            title="Descriptive K-means clusters (IDs are arbitrary)",
            width=700,
            height=460,
        )
    )
    return cast(alt.Chart, chart)


def _chart_rows(observations: pl.DataFrame) -> list[dict[str, Any]]:
    rows = observations.to_dicts()
    for row in rows:
        observed_on = row.get("date")
        if isinstance(observed_on, date):
            row["date"] = observed_on.isoformat()
        if "cluster" in row:
            row["cluster"] = str(row["cluster"])
    return rows
