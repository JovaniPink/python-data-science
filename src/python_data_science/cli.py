"""Command-line runner for the live BLS macro-clustering experiment."""

import argparse
from collections.abc import Sequence

import polars as pl

from python_data_science.analysis import (
    cluster_observations,
    derive_observations,
    diagnose_cluster_choices,
)
from python_data_science.bls import BLS_DISCLAIMER, fetch_bls_dataset


def run(argv: Sequence[str] | None = None) -> int:
    """Execute one non-persisting live analysis and print its audit manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare k=2..6 across five seeds (enabled by default)",
    )
    args = parser.parse_args(argv)

    dataset = fetch_bls_dataset(args.start_year, args.end_year)
    observations = derive_observations(dataset)
    result = cluster_observations(
        observations,
        n_clusters=args.clusters,
        seed=args.seed,
        n_init=args.n_init,
    )

    print(f"BLS macro clustering: {args.start_year}-{args.end_year}")
    print(f"Retrieved at: {dataset.retrieved_at.isoformat()}")
    print(f"Anonymous request windows: {dataset.request_windows}")
    print(f"Aligned observations: {observations.height}")
    print(f"Inertia: {result.metrics.inertia:.4f}")
    print(f"Silhouette: {result.metrics.silhouette:.4f}")

    for series_id, points in dataset.unavailable.items():
        for point in points:
            print(
                "Unavailable source value: "
                f"series={series_id} month={point.date.isoformat()} "
                f"footnotes={' | '.join(point.footnotes)}"
            )

    profiles = result.profiles.sort("mean_inflation_yoy")
    for row in profiles.iter_rows(named=True):
        print(
            f"cluster={row['cluster']} months={row['months']} "
            f"mean_inflation={row['mean_inflation_yoy']:.2f}% "
            f"mean_unemployment={row['mean_unemployment_rate']:.2f}%"
        )

    if args.diagnostics:
        diagnostics = diagnose_cluster_choices(observations, n_init=args.n_init)
        with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
            print("\nSensitivity diagnostics (not an automatic model selector):")
            print(diagnostics)

    print(
        "\nCluster IDs are arbitrary and descriptive; this is not a forecast or investment advice."
    )
    print(BLS_DISCLAIMER)
    return 0


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
