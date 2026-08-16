"""Minimal, provenance-preserving client for the BLS Public Data API."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Final, Protocol, cast

import httpx

ENDPOINT: Final = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CPI_SERIES_ID: Final = "CUUR0000SA0"
UNEMPLOYMENT_SERIES_ID: Final = "LNS14000000"
SERIES_IDS: Final = (CPI_SERIES_ID, UNEMPLOYMENT_SERIES_ID)
MAX_ANONYMOUS_YEARS: Final = 10
BLS_DISCLAIMER: Final = (
    "BLS.gov cannot vouch for the data or analyses derived from these data "
    "after the data have been retrieved from BLS.gov."
)

type JsonObject = Mapping[str, Any]
type RequestPayload = Mapping[str, object]
type RequestFunction = Callable[[RequestPayload], JsonObject]


class _Dated(Protocol):
    @property
    def date(self) -> date:
        """Date used as the merge key."""
        ...


class BLSResponseError(RuntimeError):
    """Raised when BLS or the transport returns an unusable response."""


@dataclass(frozen=True, slots=True)
class BLSPoint:
    """One numeric monthly value observed in a BLS response."""

    date: date
    value: float
    preliminary: bool = False


@dataclass(frozen=True, slots=True)
class BLSUnavailablePoint:
    """One monthly BLS record whose published value is not numeric."""

    date: date
    value: str
    footnotes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedBLSResponse:
    """Normalized contents of one successful BLS response."""

    series: dict[str, tuple[BLSPoint, ...]]
    unavailable: dict[str, tuple[BLSUnavailablePoint, ...]]
    messages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BLSDataset:
    """Merged BLS retrieval plus enough provenance to audit the request."""

    source_url: str
    retrieved_at: datetime
    start_year: int
    end_year: int
    request_windows: tuple[tuple[int, int], ...]
    series: dict[str, tuple[BLSPoint, ...]]
    unavailable: dict[str, tuple[BLSUnavailablePoint, ...]]
    messages: tuple[str, ...]


def year_windows(start_year: int, end_year: int) -> tuple[tuple[int, int], ...]:
    """Split an inclusive range into windows allowed for anonymous BLS use."""
    if start_year < 1 or end_year < 1 or start_year > end_year:
        raise ValueError("years must be positive and start_year must not exceed end_year")

    windows: list[tuple[int, int]] = []
    window_start = start_year
    while window_start <= end_year:
        window_end = min(window_start + MAX_ANONYMOUS_YEARS - 1, end_year)
        windows.append((window_start, window_end))
        window_start = window_end + 1
    return tuple(windows)


def fetch_bls_dataset(
    start_year: int,
    end_year: int,
    *,
    request: RequestFunction | None = None,
    registration_key: str | None = None,
    retrieved_at: datetime | None = None,
) -> BLSDataset:
    """Retrieve and merge the two source series without persisting raw data."""
    windows = year_windows(start_year, end_year)
    series_accumulator: dict[str, dict[date, BLSPoint]] = {key: {} for key in SERIES_IDS}
    unavailable_accumulator: dict[str, dict[date, BLSUnavailablePoint]] = {
        key: {} for key in SERIES_IDS
    }
    messages: list[str] = []

    client = None if request is not None else httpx.Client(timeout=30.0)
    try:
        for window_start, window_end in windows:
            payload: dict[str, object] = {
                "seriesid": list(SERIES_IDS),
                "startyear": str(window_start),
                "endyear": str(window_end),
            }
            if registration_key:
                payload["registrationkey"] = registration_key

            if request is None:
                if client is None:  # pragma: no cover - defensive invariant
                    raise AssertionError("HTTP client was not initialized")
                response = client.post(ENDPOINT, json=payload)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise BLSResponseError(f"BLS HTTP status {response.status_code}") from exc
                body = cast(JsonObject, response.json())
            else:
                body = request(payload)

            parsed = parse_bls_response(body)
            _merge_points(series_accumulator, parsed.series)
            _merge_points(unavailable_accumulator, parsed.unavailable)
            messages.extend(parsed.messages)
    except httpx.HTTPError as exc:
        raise BLSResponseError(f"BLS request failed: {exc}") from exc
    finally:
        if client is not None:
            client.close()

    timestamp = retrieved_at or datetime.now(UTC).replace(microsecond=0)
    if timestamp.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")

    return BLSDataset(
        source_url=ENDPOINT,
        retrieved_at=timestamp.astimezone(UTC),
        start_year=start_year,
        end_year=end_year,
        request_windows=windows,
        series={
            key: tuple(sorted(values.values(), key=lambda point: point.date))
            for key, values in series_accumulator.items()
        },
        unavailable={
            key: tuple(sorted(values.values(), key=lambda point: point.date))
            for key, values in unavailable_accumulator.items()
        },
        messages=tuple(dict.fromkeys(messages)),
    )


def parse_bls_response(body: JsonObject) -> ParsedBLSResponse:
    """Normalize one BLS v2 response while retaining unavailable records."""
    status = body.get("status")
    messages = _messages(body.get("message"))
    if status != "REQUEST_SUCCEEDED":
        raise BLSResponseError(f"BLS request status {status!r}: {list(messages)!r}")

    entries = _series_entries(body.get("Results"))
    series: dict[str, tuple[BLSPoint, ...]] = {}
    unavailable: dict[str, tuple[BLSUnavailablePoint, ...]] = {}

    for entry in entries:
        series_id = entry.get("seriesID")
        data = entry.get("data")
        if not isinstance(series_id, str) or not isinstance(data, list):
            raise BLSResponseError("BLS series entry is missing seriesID or data")

        points: list[BLSPoint] = []
        unavailable_points: list[BLSUnavailablePoint] = []
        for raw_entry in data:
            if not isinstance(raw_entry, Mapping):
                continue
            parsed = _parse_monthly_entry(cast(JsonObject, raw_entry))
            if isinstance(parsed, BLSPoint):
                points.append(parsed)
            elif isinstance(parsed, BLSUnavailablePoint):
                unavailable_points.append(parsed)

        series[series_id] = tuple(sorted(points, key=lambda point: point.date))
        unavailable[series_id] = tuple(sorted(unavailable_points, key=lambda point: point.date))

    return ParsedBLSResponse(series=series, unavailable=unavailable, messages=messages)


def _series_entries(results: object) -> list[JsonObject]:
    if isinstance(results, Mapping):
        series_entries = results.get("series")
        if isinstance(series_entries, list):
            return [cast(JsonObject, item) for item in series_entries if isinstance(item, Mapping)]
    if isinstance(results, list):
        flattened_entries: list[JsonObject] = []
        for result in results:
            if isinstance(result, Mapping) and isinstance(result.get("series"), list):
                flattened_entries.extend(
                    cast(JsonObject, item) for item in result["series"] if isinstance(item, Mapping)
                )
        return flattened_entries
    raise BLSResponseError("BLS response does not contain a series result")


def _parse_monthly_entry(entry: JsonObject) -> BLSPoint | BLSUnavailablePoint | None:
    year_text = entry.get("year")
    period = entry.get("period")
    raw_value = entry.get("value")
    if not (
        isinstance(year_text, str)
        and isinstance(period, str)
        and len(period) == 3
        and period.startswith("M")
        and isinstance(raw_value, str)
    ):
        return None

    try:
        month = int(period[1:])
        if month not in range(1, 13):
            return None
        observed_on = date(int(year_text), month, 1)
    except ValueError:
        return None

    footnotes = _footnotes(entry.get("footnotes"))
    try:
        value = float(raw_value.replace(",", ""))
    except ValueError:
        return BLSUnavailablePoint(observed_on, raw_value, footnotes)

    preliminary = any(
        footnote.get("code") == "P" or "preliminary" in str(footnote.get("text", "")).casefold()
        for footnote in _footnote_objects(entry.get("footnotes"))
    )
    return BLSPoint(observed_on, value, preliminary)


def _footnote_objects(value: object) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(cast(JsonObject, item) for item in value if isinstance(item, Mapping))


def _footnotes(value: object) -> tuple[str, ...]:
    texts = [item.get("text") for item in _footnote_objects(value)]
    return tuple(dict.fromkeys(text for text in texts if isinstance(text, str) and text))


def _messages(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    messages = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return tuple(dict.fromkeys(messages))


def _merge_points[T: _Dated](
    destination: dict[str, dict[date, T]], source: Mapping[str, tuple[T, ...]]
) -> None:
    for series_id, points in source.items():
        by_date = destination.setdefault(series_id, {})
        for point in points:
            by_date[point.date] = point
