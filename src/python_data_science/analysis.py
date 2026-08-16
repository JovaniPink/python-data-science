"""Polars transformations and descriptive scikit-learn clustering."""

from dataclasses import dataclass
from itertools import combinations
from statistics import fmean
from typing import Final

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from python_data_science.bls import CPI_SERIES_ID, UNEMPLOYMENT_SERIES_ID, BLSDataset

FEATURE_COLUMNS: Final = ("inflation_yoy", "unemployment_rate")
DEFAULT_SEEDS: Final = (0, 1, 2, 42, 99)


@dataclass(frozen=True, slots=True)
class ClusterMetrics:
    """Diagnostics for one fitted descriptive clustering pipeline."""

    inertia: float
    silhouette: float
    n_clusters: int
    seed: int
    n_init: int


@dataclass(frozen=True, slots=True)
class Standardization:
    """Population mean and scale learned by StandardScaler."""

    inflation_mean: float
    inflation_scale: float
    unemployment_mean: float
    unemployment_scale: float


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Labeled observations plus summaries and the fitted sklearn pipeline."""

    observations: pl.DataFrame
    profiles: pl.DataFrame
    metrics: ClusterMetrics
    standardization: Standardization
    pipeline: Pipeline


def derive_observations(dataset: BLSDataset) -> pl.DataFrame:
    """Align both source series and derive the 12-month CPI change."""
    cpi_points = dataset.series.get(CPI_SERIES_ID, ())
    unemployment_points = dataset.series.get(UNEMPLOYMENT_SERIES_ID, ())
    if not cpi_points or not unemployment_points:
        raise ValueError("both selected BLS series must contain numeric observations")

    cpi = pl.DataFrame(
        {
            "date": [point.date for point in cpi_points],
            "cpi": [point.value for point in cpi_points],
            "cpi_preliminary": [point.preliminary for point in cpi_points],
        },
        schema={"date": pl.Date, "cpi": pl.Float64, "cpi_preliminary": pl.Boolean},
    )
    unemployment = pl.DataFrame(
        {
            "date": [point.date for point in unemployment_points],
            "unemployment_rate": [point.value for point in unemployment_points],
            "unemployment_preliminary": [point.preliminary for point in unemployment_points],
        },
        schema={
            "date": pl.Date,
            "unemployment_rate": pl.Float64,
            "unemployment_preliminary": pl.Boolean,
        },
    )
    prior_cpi = cpi.select(
        pl.col("date").dt.offset_by("1y").alias("date"),
        pl.col("cpi").alias("prior_cpi"),
        pl.col("cpi_preliminary").alias("prior_cpi_preliminary"),
    )

    return (
        cpi.join(prior_cpi, on="date", how="inner", validate="1:1")
        .join(unemployment, on="date", how="inner", validate="1:1")
        .with_columns(
            ((pl.col("cpi") / pl.col("prior_cpi") - 1.0) * 100.0).alias("inflation_yoy"),
            (
                pl.col("cpi_preliminary")
                | pl.col("prior_cpi_preliminary")
                | pl.col("unemployment_preliminary")
            ).alias("preliminary"),
        )
        .select("date", "inflation_yoy", "unemployment_rate", "preliminary")
        .sort("date")
    )


def cluster_observations(
    observations: pl.DataFrame,
    *,
    n_clusters: int = 3,
    seed: int = 42,
    n_init: int = 20,
) -> ClusterResult:
    """Standardize both features and fit a deterministic repeated K-means."""
    _validate_cluster_request(observations, n_clusters, n_init)
    matrix = observations.select(FEATURE_COLUMNS).to_numpy()
    pipeline = _pipeline(n_clusters=n_clusters, seed=seed, n_init=n_init)
    labels = pipeline.fit_predict(matrix)
    labeled = observations.with_columns(pl.Series("cluster", labels, dtype=pl.Int32))

    scaler = pipeline.named_steps["standardize"]
    model = pipeline.named_steps["kmeans"]
    if not isinstance(scaler, StandardScaler) or not isinstance(model, KMeans):
        raise TypeError("unexpected pipeline steps")
    standardized = scaler.transform(matrix)

    profiles = (
        labeled.group_by("cluster")
        .agg(
            pl.len().alias("months"),
            pl.col("inflation_yoy").mean().alias("mean_inflation_yoy"),
            pl.col("unemployment_rate").mean().alias("mean_unemployment_rate"),
            pl.col("date").min().alias("first_month"),
            pl.col("date").max().alias("last_month"),
        )
        .sort("cluster")
    )
    means = np.asarray(scaler.mean_, dtype=np.float64)
    scales = np.asarray(scaler.scale_, dtype=np.float64)

    return ClusterResult(
        observations=labeled,
        profiles=profiles,
        metrics=ClusterMetrics(
            inertia=float(model.inertia_),
            silhouette=float(silhouette_score(standardized, labels)),
            n_clusters=n_clusters,
            seed=seed,
            n_init=n_init,
        ),
        standardization=Standardization(
            inflation_mean=float(means[0]),
            inflation_scale=float(scales[0]),
            unemployment_mean=float(means[1]),
            unemployment_scale=float(scales[1]),
        ),
        pipeline=pipeline,
    )


def diagnose_cluster_choices(
    observations: pl.DataFrame,
    *,
    cluster_counts: tuple[int, ...] = (2, 3, 4, 5, 6),
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_init: int = 20,
) -> pl.DataFrame:
    """Compare separation and label-invariant seed stability across choices.

    These diagnostics are sensitivity evidence, not an automatic selector for
    an economically true cluster count. Pairwise adjusted Rand index is used
    because raw K-means cluster identifiers can be permuted between fits.
    """
    if not seeds:
        raise ValueError("at least one diagnostic seed is required")
    if not cluster_counts:
        raise ValueError("at least one cluster count is required")

    matrix = observations.select(FEATURE_COLUMNS).to_numpy()
    standardized = StandardScaler().fit_transform(matrix)
    rows: list[dict[str, float | int]] = []

    for n_clusters in cluster_counts:
        _validate_cluster_request(observations, n_clusters, n_init)
        labels_by_seed: list[np.ndarray] = []
        silhouettes: list[float] = []
        inertias: list[float] = []
        for seed in seeds:
            model = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=n_init,
                random_state=seed,
                algorithm="lloyd",
            ).fit(standardized)
            labels_by_seed.append(np.asarray(model.labels_, dtype=np.int32))
            silhouettes.append(float(silhouette_score(standardized, model.labels_)))
            inertias.append(float(model.inertia_))

        pairwise_ari = [
            float(adjusted_rand_score(left, right))
            for left, right in combinations(labels_by_seed, 2)
        ] or [1.0]
        rows.append(
            {
                "clusters": n_clusters,
                "seed_runs": len(seeds),
                "mean_silhouette": fmean(silhouettes),
                "min_silhouette": min(silhouettes),
                "max_silhouette": max(silhouettes),
                "mean_pairwise_ari": fmean(pairwise_ari),
                "min_pairwise_ari": min(pairwise_ari),
                "mean_inertia": fmean(inertias),
            }
        )

    return pl.DataFrame(rows).sort("clusters")


def _pipeline(*, n_clusters: int, seed: int, n_init: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=n_init,
                    random_state=seed,
                    algorithm="lloyd",
                ),
            ),
        ]
    )


def _validate_cluster_request(observations: pl.DataFrame, n_clusters: int, n_init: int) -> None:
    missing = set(FEATURE_COLUMNS) - set(observations.columns)
    if missing:
        raise ValueError(f"observations are missing features: {sorted(missing)!r}")
    if n_clusters < 2 or n_clusters >= observations.height:
        raise ValueError("n_clusters must be at least 2 and smaller than the sample")
    if n_init < 1:
        raise ValueError("n_init must be positive")
    if observations.select(pl.col(FEATURE_COLUMNS).null_count()).row(0) != (0, 0):
        raise ValueError("clustering features must not contain nulls")
    if any(observations.get_column(column).n_unique() < 2 for column in FEATURE_COLUMNS):
        raise ValueError("clustering features must not be constant")
