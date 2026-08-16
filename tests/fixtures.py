"""Synthetic BLS-shaped responses with no external data dependency."""

from collections.abc import Callable
from math import pow
from typing import Any

from python_data_science.bls import CPI_SERIES_ID, UNEMPLOYMENT_SERIES_ID


def bls_response(start_year: int, end_year: int) -> dict[str, Any]:
    """Return deterministic monthly values in the public BLS response shape."""
    return {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {
            "series": [
                _series(CPI_SERIES_ID, start_year, end_year, _cpi_value),
                _series(
                    UNEMPLOYMENT_SERIES_ID,
                    start_year,
                    end_year,
                    _unemployment_value,
                ),
            ]
        },
    }


def _series(
    series_id: str,
    start_year: int,
    end_year: int,
    value_function: Callable[[int, int], float],
) -> dict[str, Any]:
    data = [
        {
            "year": str(year),
            "period": f"M{month:02d}",
            "value": f"{value_function(year, month):.3f}",
            "footnotes": [],
        }
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]
    return {"seriesID": series_id, "data": list(reversed(data))}


def _cpi_value(year: int, month: int) -> float:
    elapsed_months = (year - 2000) * 12 + month - 1
    return 100.0 * pow(1.0025, elapsed_months)


def _unemployment_value(year: int, month: int) -> float:
    phase = ((year - 2000) * 12 + month - 1) % 36
    if phase < 12:
        return 4.0 + month / 100.0
    if phase < 24:
        return 6.5 + month / 100.0
    return 9.0 + month / 100.0
