"""Tests for transformations, reproducibility, sensitivity, and charts."""

from datetime import UTC, date, datetime

import polars as pl
import pytest

from python_data_science.analysis import (
    cluster_observations,
    derive_observations,
    diagnose_cluster_choices,
)
from python_data_science.bls import fetch_bls_dataset
from python_data_science.charts import cluster_scatter_chart, timeline_chart
from tests.fixtures import bls_response


def synthetic_observations() -> pl.DataFrame:
    def request(payload: dict[str, object]) -> dict[str, object]:
        return bls_response(int(payload["startyear"]), int(payload["endyear"]))

    dataset = fetch_bls_dataset(
        2000,
        2005,
        request=request,
        retrieved_at=datetime(2026, 8, 16, tzinfo=UTC),
    )
    return derive_observations(dataset)


def test_derives_aligned_year_over_year_inflation() -> None:
    observations = synthetic_observations()
    assert observations.height == 60
    assert observations["date"][0] == date(2001, 1, 1)
    assert observations["date"][-1] == date(2005, 12, 1)
    assert observations["inflation_yoy"][0] == pytest.approx(3.0416, abs=0.01)


def test_fits_reproducible_descriptive_clusters() -> None:
    observations = synthetic_observations()
    first = cluster_observations(observations, n_clusters=3, seed=42)
    second = cluster_observations(observations, n_clusters=3, seed=42)

    assert first.observations["cluster"].to_list() == second.observations["cluster"].to_list()
    assert first.profiles.height == 3
    assert first.profiles["months"].sum() == observations.height
    assert 0.0 < first.metrics.silhouette < 1.0


def test_sensitivity_diagnostics_are_label_invariant() -> None:
    diagnostics = diagnose_cluster_choices(
        synthetic_observations(), cluster_counts=(2, 3), seeds=(1, 42), n_init=5
    )
    assert diagnostics["clusters"].to_list() == [2, 3]
    assert diagnostics["seed_runs"].to_list() == [2, 2]
    assert diagnostics["min_pairwise_ari"].min() >= -1.0
    assert diagnostics["mean_silhouette"].is_not_null().all()


def test_builds_valid_vega_lite_specs() -> None:
    observations = synthetic_observations()
    result = cluster_observations(observations, n_clusters=3, seed=42)
    timeline = timeline_chart(result.observations).to_dict()
    scatter = cluster_scatter_chart(result.observations).to_dict()

    assert len(timeline["vconcat"]) == 2
    assert scatter["mark"]["type"] == "point"
    assert scatter["encoding"]["color"]["field"] == "cluster"


def test_rejects_constant_features() -> None:
    observations = pl.DataFrame(
        {
            "date": [date(2025, month, 1) for month in range(1, 5)],
            "inflation_yoy": [2.0] * 4,
            "unemployment_rate": [4.0] * 4,
            "preliminary": [False] * 4,
        }
    )
    with pytest.raises(ValueError, match="constant"):
        cluster_observations(observations, n_clusters=2)
