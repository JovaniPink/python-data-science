"""Tests for bounded requests and provenance-preserving BLS parsing."""

from datetime import UTC, date, datetime
from typing import Any

import pytest

from python_data_science.bls import (
    CPI_SERIES_ID,
    ENDPOINT,
    BLSResponseError,
    fetch_bls_dataset,
    parse_bls_response,
    year_windows,
)
from tests.fixtures import bls_response


def test_year_windows_obey_anonymous_ten_year_limit() -> None:
    assert year_windows(2006, 2025) == ((2006, 2015), (2016, 2025))
    assert year_windows(2025, 2026) == ((2025, 2026),)
    with pytest.raises(ValueError):
        year_windows(2025, 2024)


def test_fetch_merges_sorts_and_records_provenance_without_a_key() -> None:
    payloads: list[dict[str, object]] = []

    def request(payload: Any) -> dict[str, Any]:
        payloads.append(dict(payload))
        return bls_response(int(payload["startyear"]), int(payload["endyear"]))

    timestamp = datetime(2026, 8, 16, 19, 24, 8, tzinfo=UTC)
    dataset = fetch_bls_dataset(2006, 2025, request=request, retrieved_at=timestamp)

    assert dataset.source_url == ENDPOINT
    assert dataset.retrieved_at == timestamp
    assert dataset.request_windows == ((2006, 2015), (2016, 2025))
    assert len(dataset.series[CPI_SERIES_ID]) == 240
    assert dataset.series[CPI_SERIES_ID][0].date == date(2006, 1, 1)
    assert dataset.series[CPI_SERIES_ID][-1].date == date(2025, 12, 1)
    assert all("registrationkey" not in payload for payload in payloads)


def test_parse_retains_unavailable_month_and_footnote() -> None:
    parsed = parse_bls_response(
        {
            "status": "REQUEST_SUCCEEDED",
            "message": [],
            "Results": {
                "series": [
                    {
                        "seriesID": CPI_SERIES_ID,
                        "data": [
                            {
                                "year": "2025",
                                "period": "M10",
                                "value": "-",
                                "footnotes": [
                                    {
                                        "text": (
                                            "Data unavailable due to the 2025 lapse "
                                            "in appropriations."
                                        )
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        }
    )

    assert parsed.series[CPI_SERIES_ID] == ()
    point = parsed.unavailable[CPI_SERIES_ID][0]
    assert point.date == date(2025, 10, 1)
    assert point.value == "-"
    assert "lapse in appropriations" in point.footnotes[0]


def test_parse_rejects_unsuccessful_status() -> None:
    with pytest.raises(BLSResponseError, match="REQUEST_FAILED"):
        parse_bls_response({"status": "REQUEST_FAILED", "message": ["bad request"]})
