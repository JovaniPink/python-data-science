"""Typed, fail-closed boundaries for regional-expert-ensemble.v2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypedDict, cast

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type SourceId = Literal[
    "qcew",
    "qcew_industry",
    "bea",
    "fhfa",
    "bfs",
    "building_permits",
    "treasury",
    "eia",
    "qwi",
    "fdic",
]
type ExpertId = Literal[
    "labor",
    "qcew_business",
    "industry",
    "formation",
    "construction",
    "growth",
    "housing",
    "energy",
    "labor_flows",
    "credit",
]
type GateContextId = Literal["treasury"]

V1_SHA256: Final = "c1693dbe606629fcc1f63eb7a915f14219c7b2bc580ea85afc72371957f651c9"
V2_SHA256: Final = "f318803c442f53f7fe43e0a730244beaf7729ef866cf670ed5dcb4f173baef1a"
KNOWN_EXPERTS: Final[frozenset[str]] = frozenset(
    {
        "labor",
        "qcew_business",
        "industry",
        "formation",
        "construction",
        "growth",
        "housing",
        "energy",
        "labor_flows",
        "credit",
    }
)


class ArtifactReceiptDict(TypedDict):
    """Serialized integrity metadata for one normalized artifact."""

    path: str
    sha256: str
    byte_count: int
    row_count: int


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Integrity metadata for one declared normalized artifact."""

    path: str
    sha256: str
    byte_count: int
    row_count: int


@dataclass(frozen=True, slots=True)
class SourceReceipt:
    """Publisher and custody metadata for one source object."""

    source_id: SourceId
    publisher_url: str
    release_date: date
    retrieved_at: str
    media_type: str
    sha256: str
    byte_count: int
    terms_url: str
    vintage_status: str
    extraction_tools: dict[str, str] = field(default_factory=dict)
    manual_checks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PublishedObservation:
    """A normalized observation with an explicit public release date."""

    source_id: SourceId
    state_fips: str
    observation_period: str
    release_date: date
    vintage: str
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class Profile:
    """An admitted set of experts and gate-only context sources."""

    profile_id: str
    active: bool
    experts: tuple[ExpertId, ...]
    gate_context: tuple[GateContextId, ...]


@dataclass(frozen=True, slots=True)
class PanelRow:
    """An in-memory v2 panel row before canonical CSV serialization."""

    forecast_origin: str
    state_fips: str
    target_quarter: str
    target_quarter_number: int
    census_division: str
    evaluation_origin: bool
    features: dict[str, float]
    source_release_dates: dict[SourceId, date]
    target: float
    outcome_available_date: date
    target_vintage: str


@dataclass(frozen=True, slots=True)
class Prediction:
    """A model prediction and interval for one state and origin."""

    forecast_origin: str
    state_fips: str
    model_id: str
    prediction: float
    final_outcome: float
    error: float
    interval_lower_80: float
    interval_upper_80: float
    weights: dict[ExpertId, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResult:
    """A typed result for one trained expert or combiner."""

    model_id: str
    predictions: tuple[Prediction, ...]
    settings: dict[str, JsonValue]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class StackResult:
    """Selected experts and exact full-width convex weights."""

    selected_experts: tuple[ExpertId, ...]
    weights: dict[ExpertId, float]
    mse: float


def repository_root() -> Path:
    """Return the repository root containing the shared v2 contract."""
    return Path(__file__).resolve().parents[2]


def contract_path() -> Path:
    """Return the committed v2 contract path."""
    return repository_root() / "contracts" / "regional-expert-ensemble.v2.json"


def contract_sha256() -> str:
    """Hash the exact committed v2 contract bytes."""
    return hashlib.sha256(contract_path().read_bytes()).hexdigest()


def load_contract() -> JsonObject:
    """Load and minimally identify the v2 contract."""
    value: object = json.loads(contract_path().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "regional-expert-ensemble.v2":
        raise ValueError("contract.schema_version: expected regional-expert-ensemble.v2")
    return cast(JsonObject, value)


def profile(profile_id: str) -> Profile:
    """Load one active admission profile and reject conditional profiles."""
    profiles = _object(load_contract().get("profiles"), "profiles")
    raw = _object(profiles.get(profile_id), f"profiles.{profile_id}")
    active = raw.get("active", True)
    if not isinstance(active, bool):
        raise ValueError(f"profiles.{profile_id}.active: expected boolean")
    if not active:
        raise ValueError(f"profiles.{profile_id}: inactive profile")

    expert_values = _string_list(raw.get("experts"), f"profiles.{profile_id}.experts")
    if any(item not in KNOWN_EXPERTS for item in expert_values):
        raise ValueError(f"profiles.{profile_id}.experts: unknown expert")
    context_values = _string_list(
        raw.get("gate_context", []), f"profiles.{profile_id}.gate_context"
    )
    if any(item != "treasury" for item in context_values):
        raise ValueError(f"profiles.{profile_id}.gate_context: unknown context")

    return Profile(
        profile_id=profile_id,
        active=active,
        experts=cast(tuple[ExpertId, ...], tuple(expert_values)),
        gate_context=cast(tuple[GateContextId, ...], tuple(context_values)),
    )


def artifact_receipt(root: Path, relative_path: str, row_count: int) -> ArtifactReceipt:
    """Create a receipt for an already normalized local artifact."""
    data = (root / relative_path).read_bytes()
    return ArtifactReceipt(
        path=relative_path,
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        row_count=row_count,
    )


def validate_artifact(root: Path, receipt: ArtifactReceipt) -> None:
    """Validate a declared regular file without following a symlink."""
    relative = PurePosixPath(receipt.path)
    if relative.is_absolute() or ".." in relative.parts or not receipt.path:
        raise ValueError(f"artifact.path: unsafe relative path: {receipt.path}")
    path = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"artifact.path: symlink rejected: {receipt.path}")
    if not path.is_file():
        raise ValueError(f"artifact.path: expected regular file: {receipt.path}")
    data = path.read_bytes()
    if len(data) != receipt.byte_count or hashlib.sha256(data).hexdigest() != receipt.sha256:
        raise ValueError(f"artifact.receipt: mismatch: {receipt.path}")


def first_complete_quarter_origin(
    observations: list[PublishedObservation],
    states: tuple[str, ...],
    first: str,
    last: str,
    lags: tuple[int, ...],
) -> str:
    """Derive the first origin with all required point-in-time lag quarters."""
    for origin in _quarters(first, last):
        if _origin_complete(observations, states, origin, lags):
            return origin
    raise ValueError("no complete point-in-time feature origin")


def require_complete_quarter_origin(
    observations: list[PublishedObservation],
    states: tuple[str, ...],
    origin: str,
    lags: tuple[int, ...],
) -> None:
    """Reject an origin missing any state or required lag quarter."""
    if not _origin_complete(observations, states, origin, lags):
        raise ValueError(f"incomplete point-in-time feature origin: {origin}")


def first_eligible_outer_origin(rows: list[PanelRow], minimum_quarters: int) -> str:
    """Derive the first origin with enough fully published prior labels."""
    origins = sorted({row.forecast_origin for row in rows if row.evaluation_origin})
    for outer_origin in origins:
        outer_end = quarter_end(outer_origin)
        eligible = {
            row.forecast_origin
            for row in rows
            if row.forecast_origin < outer_origin and row.outcome_available_date <= outer_end
        }
        if len(eligible) >= minimum_quarters:
            return outer_origin
    raise ValueError("no eligible outer origin")


def screened_convex_stack(
    experts: tuple[ExpertId, ...],
    trailing_mae: dict[ExpertId, float],
    predictions: list[list[float]],
    outcomes: list[float],
) -> StackResult:
    """Fit the exact 0.05 convex grid over the four best prior-MAE experts."""
    if (
        not experts
        or len(set(experts)) != len(experts)
        or len(predictions) != len(outcomes)
        or not outcomes
        or any(len(row) != len(experts) for row in predictions)
        or any(not math.isfinite(value) for row in predictions for value in row)
        or any(not math.isfinite(value) for value in outcomes)
        or any(
            expert not in trailing_mae
            or not math.isfinite(trailing_mae[expert])
            or trailing_mae[expert] < 0.0
            for expert in experts
        )
    ):
        raise ValueError("invalid screened stack inputs")

    contract_order = _string_list(load_contract().get("expert_order"), "expert_order")
    order = {expert: index for index, expert in enumerate(contract_order)}
    selected = tuple(sorted(experts, key=lambda expert: (trailing_mae[expert], order[expert]))[:4])
    indices = tuple(experts.index(expert) for expert in selected)
    selected_predictions = [[row[index] for index in indices] for row in predictions]
    selected_weights, mse = _search_simplex(selected_predictions, outcomes)
    weights: dict[ExpertId, float] = dict.fromkeys(experts, 0.0)
    for expert, weight in zip(selected, selected_weights, strict=True):
        weights[expert] = weight
    return StackResult(selected, weights, mse)


def quarter_add(quarter: str, offset: int) -> str:
    """Add calendar quarters to a YYYYQn identifier."""
    year = int(quarter[:4])
    number = int(quarter[5])
    absolute = year * 4 + number - 1 + offset
    return f"{absolute // 4:04d}Q{absolute % 4 + 1}"


def quarter_end(quarter: str) -> date:
    """Return the calendar quarter-end date."""
    year = int(quarter[:4])
    number = int(quarter[5])
    return date(year, number * 3, (31, 30, 30, 31)[number - 1])


def _origin_complete(
    observations: list[PublishedObservation],
    states: tuple[str, ...],
    origin: str,
    lags: tuple[int, ...],
) -> bool:
    origin_end = quarter_end(origin)
    for state in states:
        snapshot: dict[str, PublishedObservation] = {}
        for row in observations:
            if row.state_fips != state or row.release_date > origin_end:
                continue
            prior = snapshot.get(row.observation_period)
            if prior is None or prior.release_date < row.release_date:
                snapshot[row.observation_period] = row
        eligible = [period for period in snapshot if period <= origin]
        if not eligible:
            return False
        latest = max(eligible)
        if any(quarter_add(latest, -lag) not in snapshot for lag in lags):
            return False
    return True


def _quarters(first: str, last: str) -> list[str]:
    values: list[str] = []
    current = first
    while current <= last:
        values.append(current)
        current = quarter_add(current, 1)
    return values


def _object(value: JsonValue | None, path: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _string_list(value: JsonValue, path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: expected array of strings")
    return cast(list[str], value)


def _search_simplex(
    predictions: list[list[float]], outcomes: list[float]
) -> tuple[list[float], float]:
    width = len(predictions[0])
    best_weights: list[float] | None = None
    best_mse = math.inf
    for integers in _integer_compositions(20, width):
        weights = [value / 20.0 for value in integers]
        mse = sum(
            (sum(value * weight for value, weight in zip(row, weights, strict=True)) - outcome) ** 2
            for row, outcome in zip(predictions, outcomes, strict=True)
        ) / len(outcomes)
        if mse < best_mse - 1.0e-15:
            best_weights = weights
            best_mse = mse
    if best_weights is None:  # pragma: no cover
        raise RuntimeError("screened stack search produced no candidate")
    return best_weights, best_mse


def _integer_compositions(total: int, width: int) -> list[tuple[int, ...]]:
    if width == 1:
        return [(total,)]
    return [
        (first, *rest)
        for first in range(total + 1)
        for rest in _integer_compositions(total - first, width - 1)
    ]
