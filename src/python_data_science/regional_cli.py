"""Offline runner for the point-in-time regional expert ensemble."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from python_data_science.regional import run_experiment


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/regional-ensemble/python/v1")
    )
    args = parser.parse_args(argv)
    manifest = run_experiment(args.source_bundle, args.output_dir)
    metrics = manifest["metrics"]
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    print("Regional expert ensemble: point-in-time historical backtest")
    print("Population: 50 states plus Washington, DC; territories excluded")
    print("Forecast origins: 2020Q1 through 2025Q3")
    print("Target: next-quarter final QCEW third-month employment year-over-year log growth")
    print(f"Contract SHA-256: {manifest['contract_sha256']}")
    print(f"Overall metrics: {overall}")
    print(f"Artifacts: {args.output_dir}")
    print("This is not a causal, recession, trading, or financial-advice claim.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
