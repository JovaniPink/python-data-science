"""Point-in-time regional economic panel, ensemble, and evidence contracts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, TypedDict, cast
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import sklearn
from sklearn.linear_model import Ridge

EXPERT_IDS: Final[tuple[str, str, str, str]] = ("labor", "business", "growth", "housing")
EXPERT_SOURCE_IDS: Final[tuple[str, str, str]] = ("qcew", "bea", "fhfa")
PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "forecast_origin",
    "state_fips",
    "target_quarter",
    "target_quarter_number",
    "census_division",
    "evaluation_origin",
    "qcew_employment_yoy",
    "qcew_employment_qoq",
    "qcew_employment_yoy_lag1",
    "qcew_establishments_yoy",
    "qcew_total_wages_yoy",
    "bea_real_gdp_yoy",
    "bea_real_gdp_qoq",
    "bea_personal_income_yoy",
    "bea_personal_income_qoq",
    "fhfa_hpi_qoq",
    "fhfa_hpi_yoy",
    "qcew_release_date",
    "bea_release_date",
    "fhfa_release_date",
    "target_employment_growth_yoy",
    "outcome_available_date",
    "target_vintage",
)
PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "forecast_origin",
    "state_fips",
    "target_quarter",
    "census_division",
    "model_id",
    "prediction",
    "final_outcome",
    "error",
    "interval_lower_80",
    "interval_upper_80",
    "weight_labor",
    "weight_business",
    "weight_growth",
    "weight_housing",
    "contribution_labor",
    "contribution_business",
    "contribution_growth",
    "contribution_housing",
    "alpha",
)

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type Json = dict[str, JsonValue]
type RowValue = bool | int | float | str
type Row = dict[str, RowValue]
type WeightVector = tuple[float, float, float, float]
FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int_]


class ArtifactReceipt(TypedDict):
    """Integrity receipt for one generated artifact."""

    sha256: str
    byte_count: int
    row_count: int


@dataclass(frozen=True, slots=True)
class NeuralGateResult:
    """Predictions, softmax expert weights, and bounded training evidence."""

    predictions: FloatArray
    weights: FloatArray
    epochs: int
    validation_mse: float
    feature_mean: FloatArray
    feature_scale: FloatArray
    w1: FloatArray
    b1: FloatArray
    w2: FloatArray
    b2: FloatArray
    validation_index: IndexArray


@dataclass(frozen=True, slots=True)
class OofResult:
    """Historical predictions that were out of fold for every target row."""

    rows: list[Row]
    expert_predictions: FloatArray
    pooled_predictions: FloatArray
    targets: FloatArray


@dataclass(frozen=True, slots=True)
class ModelForecast:
    """Predictions and calibration state for one model at one outer origin."""

    model_id: str
    predictions: FloatArray
    weights: WeightVector | FloatArray | None
    alpha: float | str
    radius: float


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def contract_path() -> Path:
    return repository_root() / "contracts" / "regional-expert-ensemble.v1.json"


def load_contract(path: Path | None = None) -> Json:
    """Load the versioned shared contract without mutating it."""
    value = json.loads((path or contract_path()).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "regional-expert-ensemble.v1":
        raise ValueError("contract.schema_version: expected regional-expert-ensemble.v1")
    return cast(Json, value)


def contract_sha256(path: Path | None = None) -> str:
    return hashlib.sha256((path or contract_path()).read_bytes()).hexdigest()


def quarter_add(value: str, amount: int) -> str:
    """Add calendar quarters to a canonical YYYYQn identifier."""
    if len(value) != 6 or value[4] != "Q" or value[5] not in "1234":
        raise ValueError(f"invalid quarter: {value!r}")
    serial = int(value[:4]) * 4 + int(value[5]) - 1 + amount
    year, zero_quarter = divmod(serial, 4)
    return f"{year:04d}Q{zero_quarter + 1}"


def quarter_end(value: str) -> date:
    if len(value) != 6 or value[4] != "Q" or value[5] not in "1234" or not value[:4].isdigit():
        raise ValueError(f"invalid quarter: {value!r}")
    year = int(value[:4])
    quarter = int(value[5])
    return (date(year, 3, 31), date(year, 6, 30), date(year, 9, 30), date(year, 12, 31))[
        quarter - 1
    ]


def _quarters(start: str, end: str) -> list[str]:
    values: list[str] = []
    current = start
    while current <= end:
        values.append(current)
        current = quarter_add(current, 1)
    return values


def validate_source_bundle(bundle: Json, *, contract: Json | None = None) -> None:
    """Fail closed unless receipts and normalized observations satisfy v1."""
    contract = contract or load_contract()
    if bundle.get("schema_version") != "regional-source-bundle.v1":
        raise ValueError("schema_version: expected regional-source-bundle.v1")
    if bundle.get("contract_sha256") != contract_sha256():
        raise ValueError("contract_sha256: does not match committed contract bytes")
    if bundle.get("research_cutoff") != contract["research_cutoff"]:
        raise ValueError("research_cutoff: does not match contract")

    cutoff = date.fromisoformat(cast(str, contract["research_cutoff"]))
    sources_contract = cast(dict[str, Json], contract["sources"])
    sources = _list_of_dicts(bundle.get("sources"), "sources")
    seen_source_ids: set[str] = set()
    for index, receipt in enumerate(sources):
        path = f"sources[{index}]"
        source_id = _required_text(receipt, "source_id", path)
        if source_id not in sources_contract:
            raise ValueError(f"{path}.source_id: unsupported source {source_id!r}")
        seen_source_ids.add(source_id)
        publisher_url = _required_text(receipt, "publisher_url", path)
        host = (urlparse(publisher_url).hostname or "").lower()
        required_host = cast(str, sources_contract[source_id]["required_host"])
        if not (host == required_host or host.endswith(f".{required_host}")):
            raise ValueError(f"{path}.publisher_url: host must be {required_host}")
        release_date = date.fromisoformat(_required_text(receipt, "release_date", path))
        if release_date > cutoff:
            raise ValueError(f"{path}.release_date: exceeds research cutoff")
        for field in ("retrieved_at", "media_type", "terms_url", "vintage_status"):
            _required_text(receipt, field, path)
        expected_hash = _required_text(receipt, "sha256", path)
        expected_count = receipt.get("byte_count")
        if not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError(f"{path}.byte_count: expected positive integer")
        cache_path = Path(_required_text(receipt, "cache_path", path))
        try:
            payload = cache_path.read_bytes()
        except OSError as error:
            raise ValueError(f"{path}.cache_path: unreadable: {error}") from error
        if len(payload) != expected_count:
            raise ValueError(f"{path}.byte_count: cache byte count mismatch")
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"{path}.sha256: cache hash mismatch")
    if seen_source_ids != set(sources_contract):
        raise ValueError("sources: receipts for qcew, bea, and fhfa are required")

    observations = bundle.get("observations")
    if not isinstance(observations, dict):
        raise ValueError("observations: expected object")
    state_fips = cast(list[str], cast(Json, contract["population"])["state_fips"])
    for source_id in EXPERT_SOURCE_IDS:
        rows = _list_of_dicts(observations.get(source_id), f"observations.{source_id}")
        _validate_observations(source_id, rows, set(state_fips), cutoff)
    _validate_fhfa_layout_checks(bundle, observations, cutoff)


def _validate_fhfa_layout_checks(
    bundle: Json, observations: dict[str, JsonValue], cutoff: date
) -> None:
    """Require one successful layout and manual-sample receipt per FHFA report."""
    extraction_tools = bundle.get("extraction_tools")
    if not isinstance(extraction_tools, dict):
        raise ValueError("extraction_tools: expected object")
    pdftotext = extraction_tools.get("pdftotext")
    if not isinstance(pdftotext, str) or not pdftotext:
        raise ValueError("extraction_tools.pdftotext: expected nonempty version")

    checks = _list_of_dicts(bundle.get("fhfa_layout_checks"), "fhfa_layout_checks")
    observed_reports = {
        _required_text(row, "report_url", "observations.fhfa")
        for row in _list_of_dicts(observations.get("fhfa"), "observations.fhfa")
    }
    checked_reports: set[str] = set()
    for index, check in enumerate(checks):
        path = f"fhfa_layout_checks[{index}]"
        report_url = _required_text(check, "report_url", path)
        if report_url in checked_reports:
            raise ValueError(f"{path}.report_url: duplicate layout check")
        host = (urlparse(report_url).hostname or "").lower()
        if not (host == "fhfa.gov" or host.endswith(".fhfa.gov")):
            raise ValueError(f"{path}.report_url: host must be fhfa.gov")
        release_date = date.fromisoformat(_required_text(check, "release_date", path))
        if release_date > cutoff:
            raise ValueError(f"{path}.release_date: exceeds research cutoff")
        _required_text(check, "layout_era", path)
        if check.get("pdftotext_version") != pdftotext:
            raise ValueError(f"{path}.pdftotext_version: does not match extraction tool")
        if check.get("row_count") != 51:
            raise ValueError(f"{path}.row_count: expected 51")
        for flag in (
            "expected_headings",
            "numeric_values",
            "warning_text_preserved",
            "manual_samples_verified",
        ):
            if check.get(flag) is not True:
                raise ValueError(f"{path}.{flag}: expected true")
        checked_reports.add(report_url)
    if checked_reports != observed_reports:
        raise ValueError("fhfa_layout_checks: must cover every admitted FHFA report")


def _validate_observations(
    source_id: str, rows: list[Row], state_set: set[str], cutoff: date
) -> None:
    seen: set[tuple[object, ...]] = set()
    group_states: dict[tuple[object, ...], set[str]] = {}
    for index, row in enumerate(rows):
        path = f"observations.{source_id}[{index}]"
        state = _required_text(row, "state_fips", path)
        if state not in state_set:
            raise ValueError(f"{path}.state_fips: outside state/DC universe")
        observation_quarter = _required_text(row, "observation_quarter", path)
        quarter_end(observation_quarter)
        release_text = _required_text(row, "release_date", path)
        if date.fromisoformat(release_text) > cutoff:
            raise ValueError(f"{path}.release_date: exceeds research cutoff")
        discriminator = (
            row.get("status") if source_id == "qcew" else row.get("vintage", row.get("report_url"))
        )
        key = (state, observation_quarter, release_text, discriminator)
        if key in seen:
            raise ValueError(f"{path}: duplicate vintage key")
        seen.add(key)
        group_key = (observation_quarter, release_text, discriminator)
        group_states.setdefault(group_key, set()).add(state)
        numeric_fields = {
            "qcew": ("employment", "establishments", "total_wages"),
            "bea": ("real_gdp", "personal_income"),
            "fhfa": ("hpi_qoq", "hpi_yoy"),
        }[source_id]
        for field in numeric_fields:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{path}.{field}: expected finite number")
            if source_id != "fhfa" and float(value) <= 0.0:
                raise ValueError(f"{path}.{field}: expected positive level")
        if source_id == "qcew" and row.get("status") not in {"preliminary", "final"}:
            raise ValueError(f"{path}.status: expected preliminary or final")
    incomplete = [key for key, states in group_states.items() if states != state_set]
    if incomplete:
        raise ValueError(
            f"observations.{source_id}: incomplete 51-state/DC vintage {incomplete[0]!r}"
        )


def _list_of_dicts(value: object, path: str) -> list[Row]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, dict) for item in value)
    ):
        raise ValueError(f"{path}: expected nonempty array of objects")
    return cast(list[Row], value)


def _required_text(row: Row, field: str, path: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{field}: expected nonempty string")
    return value


def build_panel(bundle: Json, *, contract: Json | None = None) -> list[Row]:
    """Derive the canonical state-quarter panel from admitted source observations."""
    contract = contract or load_contract()
    validate_source_bundle(bundle, contract=contract)
    observations = cast(dict[str, list[Row]], bundle["observations"])
    population = cast(Json, contract["population"])
    time = cast(Json, contract["time"])
    states = cast(list[str], population["state_fips"])
    divisions = cast(dict[str, str], population["census_divisions"])
    origins = _quarters(cast(str, time["source_start"]), cast(str, time["last_forecast_origin"]))
    first_evaluation = cast(str, time["first_forecast_origin"])
    last_evaluation = cast(str, time["last_forecast_origin"])
    by_source_state = {
        source: _group_by_state(observations[source]) for source in EXPERT_SOURCE_IDS
    }
    panel: list[Row] = []
    for origin in origins:
        origin_end = quarter_end(origin)
        target_quarter = quarter_add(origin, 1)
        for state in states:
            qcew_rows = by_source_state["qcew"][state]
            bea_rows = by_source_state["bea"][state]
            fhfa_rows = by_source_state["fhfa"][state]
            qcew = _as_of_snapshot(qcew_rows, origin_end)
            bea = _as_of_snapshot(bea_rows, origin_end)
            fhfa = _as_of_snapshot(fhfa_rows, origin_end)
            q_latest = _latest_quarter(qcew, origin)
            b_latest = _latest_quarter(bea, origin)
            h_latest = _latest_quarter(fhfa, origin)
            q0 = _at(qcew, q_latest, "qcew", state, origin)
            q1 = _at(qcew, quarter_add(q_latest, -1), "qcew", state, origin)
            q4 = _at(qcew, quarter_add(q_latest, -4), "qcew", state, origin)
            q5 = _at(qcew, quarter_add(q_latest, -5), "qcew", state, origin)
            b0 = _at(bea, b_latest, "bea", state, origin)
            b1 = _at(bea, quarter_add(b_latest, -1), "bea", state, origin)
            b4 = _at(bea, quarter_add(b_latest, -4), "bea", state, origin)
            h0 = _at(fhfa, h_latest, "fhfa", state, origin)
            target_current = _final_qcew(qcew_rows, target_quarter, state)
            target_prior = _final_qcew(qcew_rows, quarter_add(target_quarter, -4), state)
            values: Row = {
                "forecast_origin": origin,
                "state_fips": state,
                "target_quarter": target_quarter,
                "target_quarter_number": int(target_quarter[5]),
                "census_division": divisions[state],
                "evaluation_origin": first_evaluation <= origin <= last_evaluation,
                "qcew_employment_yoy": _log_growth(q0, q4, "employment"),
                "qcew_employment_qoq": _log_growth(q0, q1, "employment"),
                "qcew_employment_yoy_lag1": _log_growth(q1, q5, "employment"),
                "qcew_establishments_yoy": _log_growth(q0, q4, "establishments"),
                "qcew_total_wages_yoy": _log_growth(q0, q4, "total_wages"),
                "bea_real_gdp_yoy": _log_growth(b0, b4, "real_gdp"),
                "bea_real_gdp_qoq": _log_growth(b0, b1, "real_gdp"),
                "bea_personal_income_yoy": _log_growth(b0, b4, "personal_income"),
                "bea_personal_income_qoq": _log_growth(b0, b1, "personal_income"),
                "fhfa_hpi_qoq": float(cast(float, h0["hpi_qoq"])),
                "fhfa_hpi_yoy": float(cast(float, h0["hpi_yoy"])),
                "qcew_release_date": max(
                    str(q0["release_date"]),
                    str(q1["release_date"]),
                    str(q4["release_date"]),
                    str(q5["release_date"]),
                ),
                "bea_release_date": max(
                    str(b0["release_date"]), str(b1["release_date"]), str(b4["release_date"])
                ),
                "fhfa_release_date": str(h0["release_date"]),
                "target_employment_growth_yoy": _log_growth(
                    target_current, target_prior, "employment"
                ),
                "outcome_available_date": max(
                    str(target_current["release_date"]), str(target_prior["release_date"])
                ),
                "target_vintage": str(target_current["vintage"]),
            }
            for source in EXPERT_SOURCE_IDS:
                release = date.fromisoformat(cast(str, values[f"{source}_release_date"]))
                if release > origin_end:
                    raise ValueError(f"panel.{origin}.{state}.{source}: post-origin feature")
            panel.append({column: values[column] for column in PANEL_COLUMNS})
    return panel


def _group_by_state(rows: list[Row]) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(cast(str, row["state_fips"]), []).append(row)
    return grouped


def _as_of_snapshot(rows: list[Row], origin_end: date) -> dict[str, Row]:
    selected: dict[str, Row] = {}
    for row in rows:
        if date.fromisoformat(cast(str, row["release_date"])) <= origin_end:
            quarter = cast(str, row["observation_quarter"])
            prior = selected.get(quarter)
            if prior is None or cast(str, prior["release_date"]) < cast(str, row["release_date"]):
                selected[quarter] = row
    return selected


def _latest_quarter(snapshot: dict[str, Row], origin: str) -> str:
    eligible = [quarter for quarter in snapshot if quarter <= origin]
    if not eligible:
        raise ValueError(f"panel.{origin}: no admitted source observation")
    return max(eligible)


def _at(snapshot: dict[str, Row], quarter: str, source: str, state: str, origin: str) -> Row:
    try:
        return snapshot[quarter]
    except KeyError as error:
        raise ValueError(
            f"panel.{origin}.{state}.{source}: missing required quarter {quarter}"
        ) from error


def _final_qcew(rows: list[Row], quarter: str, state: str) -> Row:
    candidates = [
        row
        for row in rows
        if row["observation_quarter"] == quarter and row.get("status") == "final"
    ]
    if not candidates:
        raise ValueError(f"target.{quarter}.{state}: missing final QCEW outcome")
    return max(candidates, key=lambda row: cast(str, row["release_date"]))


def _log_growth(current: Row, prior: Row, field: str) -> float:
    current_value = float(cast(float, current[field]))
    prior_value = float(cast(float, prior[field]))
    if current_value <= 0.0 or prior_value <= 0.0:
        raise ValueError(f"{field}: logarithm requires positive levels")
    return 100.0 * math.log(current_value / prior_value)


def build_folds(panel: list[Row], *, contract: Json | None = None) -> list[Row]:
    """Emit exact expanding outer-fold membership with label availability."""
    contract = contract or load_contract()
    minimum = cast(int, cast(Json, contract["time"])["minimum_training_quarters"])
    evaluation_origins = sorted(
        {cast(str, row["forecast_origin"]) for row in panel if row["evaluation_origin"] is True}
    )
    rows: list[Row] = []
    for outer_origin in evaluation_origins:
        outer_end = quarter_end(outer_origin)
        eligible = [
            row
            for row in panel
            if cast(str, row["forecast_origin"]) < outer_origin
            and date.fromisoformat(cast(str, row["outcome_available_date"])) <= outer_end
        ]
        if len({cast(str, row["forecast_origin"]) for row in eligible}) < minimum:
            continue
        for row in eligible:
            rows.append(
                {
                    "outer_origin": outer_origin,
                    "membership": "train",
                    "row_origin": row["forecast_origin"],
                    "state_fips": row["state_fips"],
                    "target_quarter": row["target_quarter"],
                    "outcome_available_date": row["outcome_available_date"],
                }
            )
        for row in panel:
            if row["forecast_origin"] == outer_origin:
                rows.append(
                    {
                        "outer_origin": outer_origin,
                        "membership": "forecast",
                        "row_origin": row["forecast_origin"],
                        "state_fips": row["state_fips"],
                        "target_quarter": row["target_quarter"],
                        "outcome_available_date": row["outcome_available_date"],
                    }
                )
    membership_order = {"train": 0, "forecast": 1}
    return sorted(
        rows,
        key=lambda row: (
            cast(str, row["outer_origin"]),
            membership_order[cast(str, row["membership"])],
            cast(str, row["row_origin"]),
            cast(str, row["state_fips"]),
        ),
    )


def canonical_csv(rows: list[Row], *, columns: tuple[str, ...] | None = None) -> str:
    """Serialize rows using the cross-language LF and float contract."""
    if not rows:
        raise ValueError("canonical_csv: rows must not be empty")
    ordered_columns = columns or tuple(rows[0])
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(ordered_columns)
    for row in rows:
        writer.writerow(_canonical_value(row[column]) for column in ordered_columns)
    return output.getvalue()


def _canonical_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical_csv: non-finite float")
        return f"{value:.10f}"
    return value


def search_convex_stack(expert_predictions: FloatArray, outcome: FloatArray) -> tuple[float, ...]:
    """Exhaustively search the 0.05 simplex; first tuple wins exact ties."""
    predictions = np.asarray(expert_predictions, dtype=np.float64)
    targets = np.asarray(outcome, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != len(EXPERT_IDS):
        raise ValueError("expert_predictions must have four columns")
    if targets.shape != (predictions.shape[0],):
        raise ValueError("outcome length must match predictions")
    if (
        predictions.shape[0] == 0
        or not np.isfinite(predictions).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("stack inputs must be nonempty and finite")
    best_weights: tuple[float, ...] | None = None
    best_mse = math.inf
    for first in range(21):
        for second in range(21 - first):
            for third in range(21 - first - second):
                fourth = 20 - first - second - third
                weights = (first / 20.0, second / 20.0, third / 20.0, fourth / 20.0)
                mse = float(np.mean(np.square(predictions @ np.asarray(weights) - targets)))
                if mse < best_mse - 1.0e-15:
                    best_mse = mse
                    best_weights = weights
    if best_weights is None:  # pragma: no cover
        raise RuntimeError("stack search produced no candidate")
    return best_weights


def fit_neural_gate(
    expert_predictions: FloatArray,
    trailing_mae: FloatArray,
    context: FloatArray,
    outcome: FloatArray,
    *,
    quarter_ids: list[str] | None = None,
) -> NeuralGateResult:
    """Fit the v1 tanh/softmax gate with deterministic NumPy Adam."""
    experts = np.asarray(expert_predictions, dtype=np.float64)
    maes = np.asarray(trailing_mae, dtype=np.float64)
    context_values = np.asarray(context, dtype=np.float64)
    targets = np.asarray(outcome, dtype=np.float64)
    if experts.ndim != 2 or experts.shape[1] != 4 or maes.shape != experts.shape:
        raise ValueError("neural gate requires matching four-column expert and MAE matrices")
    if context_values.ndim != 2 or context_values.shape[0] != experts.shape[0]:
        raise ValueError("neural gate context rows must match expert rows")
    if targets.shape != (experts.shape[0],) or experts.shape[0] < 8:
        raise ValueError("neural gate requires at least eight labeled rows")
    if not all(np.isfinite(value).all() for value in (experts, maes, context_values, targets)):
        raise ValueError("neural gate inputs must be finite")
    features = np.concatenate((experts, maes, context_values), axis=1)
    train_index, validation_index = _neural_split(features.shape[0], quarter_ids)
    mean = features[train_index].mean(axis=0)
    scale = features[train_index].std(axis=0)
    scale[scale == 0.0] = 1.0
    features = (features - mean) / scale
    rng = np.random.default_rng(42)
    w1 = rng.normal(0.0, 0.05, size=(features.shape[1], 16))
    b1 = np.zeros(16, dtype=np.float64)
    w2 = rng.normal(0.0, 0.05, size=(16, 4))
    b2 = np.zeros(4, dtype=np.float64)
    parameters = [w1, b1, w2, b2]
    first_moment = [np.zeros_like(parameter) for parameter in parameters]
    second_moment = [np.zeros_like(parameter) for parameter in parameters]
    best = [parameter.copy() for parameter in parameters]
    best_mse = math.inf
    stale_epochs = 0
    epoch = 0
    for epoch in range(1, 501):
        weights, hidden = _gate_forward(features[train_index], w1, b1, w2, b2)
        predictions = np.sum(weights * experts[train_index], axis=1)
        error = predictions - targets[train_index]
        d_logits = (
            (2.0 / len(train_index))
            * error[:, None]
            * weights
            * (experts[train_index] - predictions[:, None])
        )
        grad_w2 = hidden.T @ d_logits
        grad_b2 = d_logits.sum(axis=0)
        d_hidden = d_logits @ w2.T
        d_pre_hidden = d_hidden * (1.0 - np.square(hidden))
        gradients = [
            features[train_index].T @ d_pre_hidden,
            d_pre_hidden.sum(axis=0),
            grad_w2,
            grad_b2,
        ]
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
            first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient
            second_moment[index] = 0.999 * second_moment[index] + 0.001 * np.square(gradient)
            corrected_first = first_moment[index] / (1.0 - 0.9**epoch)
            corrected_second = second_moment[index] / (1.0 - 0.999**epoch)
            parameter -= 0.01 * corrected_first / (np.sqrt(corrected_second) + 1.0e-8)
        validation_weights, _ = _gate_forward(features[validation_index], w1, b1, w2, b2)
        validation_predictions = np.sum(validation_weights * experts[validation_index], axis=1)
        validation_mse = float(
            np.mean(np.square(validation_predictions - targets[validation_index]))
        )
        if validation_mse < best_mse - 1.0e-12:
            best_mse = validation_mse
            best = [parameter.copy() for parameter in parameters]
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 30:
                break
    final_weights, _ = _gate_forward(features, *best)
    final_predictions = np.sum(final_weights * experts, axis=1)
    return NeuralGateResult(
        final_predictions,
        final_weights,
        epoch,
        best_mse,
        mean,
        scale,
        best[0],
        best[1],
        best[2],
        best[3],
        validation_index,
    )


def predict_neural_gate(
    result: NeuralGateResult,
    expert_predictions: FloatArray,
    trailing_mae: FloatArray,
    context: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply a fitted neural gate to rows that were not used for fitting."""
    experts = np.asarray(expert_predictions, dtype=np.float64)
    maes = np.asarray(trailing_mae, dtype=np.float64)
    context_values = np.asarray(context, dtype=np.float64)
    if experts.ndim != 2 or experts.shape[1] != 4 or maes.shape != experts.shape:
        raise ValueError("neural prediction requires matching four-column expert and MAE matrices")
    if context_values.ndim != 2 or context_values.shape[0] != experts.shape[0]:
        raise ValueError("neural prediction context rows must match expert rows")
    if not all(np.isfinite(value).all() for value in (experts, maes, context_values)):
        raise ValueError("neural prediction inputs must be finite")
    features = np.concatenate((experts, maes, context_values), axis=1)
    if features.shape[1] != result.feature_mean.shape[0]:
        raise ValueError("neural prediction feature width does not match fitted gate")
    features = (features - result.feature_mean) / result.feature_scale
    weights, _hidden = _gate_forward(features, result.w1, result.b1, result.w2, result.b2)
    return np.sum(weights * experts, axis=1), weights


def run_experiment(bundle_path: Path, output_dir: Path) -> Json:
    """Run the offline point-in-time backtest and write ignored v1 artifacts."""
    bundle_value = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle_value, dict):
        raise ValueError("source bundle must be a JSON object")
    bundle = cast(Json, bundle_value)
    panel = build_panel(bundle)
    folds = build_folds(panel)
    predictions = _backtest(panel, folds)
    if not predictions:
        raise ValueError("backtest produced no eligible forecasts")
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_bytes = canonical_csv(panel, columns=PANEL_COLUMNS).encode()
    fold_columns = tuple(folds[0])
    fold_bytes = canonical_csv(folds, columns=fold_columns).encode()
    prediction_bytes = canonical_csv(predictions, columns=PREDICTION_COLUMNS).encode()
    paths = {
        "regional-panel.v1.csv": panel_bytes,
        "regional-folds.v1.csv": fold_bytes,
        "regional-predictions.v1.csv": prediction_bytes,
    }
    for name, payload in paths.items():
        (output_dir / name).write_bytes(payload)
    row_counts = {
        "regional-panel.v1.csv": len(panel),
        "regional-folds.v1.csv": len(folds),
        "regional-predictions.v1.csv": len(predictions),
    }
    contract = load_contract()
    manifest: Json = {
        "schema_version": "regional-run-manifest.v1",
        "contract_sha256": contract_sha256(),
        "source_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "artifacts": {
            name: cast(
                Json,
                ArtifactReceipt(
                    sha256=hashlib.sha256(payload).hexdigest(),
                    byte_count=len(payload),
                    row_count=row_counts[name],
                ),
            )
            for name, payload in paths.items()
        },
        "environment": {
            "implementation": "python",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "git": _git_state(repository_root()),
        "settings": {
            "ridge": contract["ridge"],
            "stack": contract["stack"],
            "neural_gate": contract["neural_gate"],
            "intervals": contract["intervals"],
        },
        "metrics": _metrics(predictions),
        "exclusions": contract["excluded_v1"],
        "claims": contract["claims"],
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (output_dir / "regional-run-manifest.v1.json").write_bytes(manifest_bytes)
    return manifest


def _git_state(root: Path) -> Json:
    """Read the exact local Git head and whether tracked or untracked files differ."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"head": head, "dirty": bool(status), "executable": sys.executable}


def _backtest(panel: list[Row], folds: list[Row]) -> list[Row]:
    fold_origins = sorted({cast(str, row["outer_origin"]) for row in folds})
    all_oof = _oof_predictions(panel)
    panel_by_key = {
        (cast(str, row["forecast_origin"]), cast(str, row["state_fips"])): row for row in panel
    }
    output: list[Row] = []
    for outer_origin in fold_origins:
        fold = [row for row in folds if row["outer_origin"] == outer_origin]
        train = [
            panel_by_key[(cast(str, row["row_origin"]), cast(str, row["state_fips"]))]
            for row in fold
            if row["membership"] == "train"
        ]
        forecast = [
            panel_by_key[(outer_origin, cast(str, row["state_fips"]))]
            for row in fold
            if row["membership"] == "forecast"
        ]
        expert_forecasts: list[FloatArray] = []
        selected_alphas: dict[str, float] = {}
        for expert in cast(list[Json], load_contract()["experts"]):
            expert_id = cast(str, expert["id"])
            features = cast(list[str], expert["numeric_features"])
            alpha = _select_alpha(train, features)
            selected_alphas[expert_id] = alpha
            expert_forecasts.append(_ridge_predict(train, forecast, features, alpha))
        forecast_matrix = np.column_stack(expert_forecasts)
        pooled_features = [
            feature
            for expert in cast(list[Json], load_contract()["experts"])
            for feature in cast(list[str], expert["numeric_features"])
        ]
        pooled_alpha = _select_alpha(train, pooled_features)
        pooled = _ridge_predict(train, forecast, pooled_features, pooled_alpha)
        oof_indexes = [
            index
            for index, row in enumerate(all_oof.rows)
            if cast(str, row["forecast_origin"]) < outer_origin
            and date.fromisoformat(cast(str, row["outcome_available_date"]))
            <= quarter_end(outer_origin)
        ]
        oof_rows = [all_oof.rows[index] for index in oof_indexes]
        oof_matrix = all_oof.expert_predictions[oof_indexes]
        oof_pooled = all_oof.pooled_predictions[oof_indexes]
        oof_targets = all_oof.targets[oof_indexes]
        stack_weights: WeightVector
        inverse_weights: WeightVector
        if len(oof_rows) == 0:
            stack_weights = (0.25, 0.25, 0.25, 0.25)
            inverse_weights = stack_weights
            reporting = cast(Json, load_contract()["reporting"])
            radii = dict.fromkeys(cast(list[str], reporting["models"]), 0.0)
        else:
            stack_weights = cast(WeightVector, search_convex_stack(oof_matrix, oof_targets))
            expert_mae = np.mean(np.abs(oof_matrix - oof_targets[:, None]), axis=0)
            inverse = 1.0 / np.maximum(expert_mae, 1.0e-12)
            inverse_weights = cast(
                WeightVector, tuple(float(value) for value in inverse / inverse.sum())
            )
            radii = {
                expert_id: _empirical_radius(oof_matrix[:, index], oof_targets)
                for index, expert_id in enumerate(EXPERT_IDS)
            }
            radii.update(
                {
                    "zero": _empirical_radius(np.zeros(len(oof_targets)), oof_targets),
                    "latest_qcew_yoy": _empirical_radius(
                        np.asarray(
                            [float(cast(float, row["qcew_employment_yoy"])) for row in oof_rows]
                        ),
                        oof_targets,
                    ),
                    "pooled_ridge": _empirical_radius(oof_pooled, oof_targets),
                    "equal_weight": _empirical_radius(oof_matrix.mean(axis=1), oof_targets),
                    "inverse_mae": _empirical_radius(
                        oof_matrix @ np.asarray(inverse_weights), oof_targets
                    ),
                    "convex_stack": _empirical_radius(
                        oof_matrix @ np.asarray(stack_weights), oof_targets
                    ),
                }
            )
        equal = forecast_matrix.mean(axis=1)
        inverse_prediction = forecast_matrix @ np.asarray(inverse_weights)
        stack_prediction = forecast_matrix @ np.asarray(stack_weights)
        model_values = [
            ModelForecast(
                expert_id,
                expert_forecasts[index],
                _one_hot(index),
                selected_alphas[expert_id],
                radii[expert_id],
            )
            for index, expert_id in enumerate(EXPERT_IDS)
        ]
        model_values.extend(
            [
                ModelForecast("zero", np.zeros(len(forecast)), None, "", radii["zero"]),
                ModelForecast(
                    "latest_qcew_yoy",
                    np.asarray(
                        [float(cast(float, row["qcew_employment_yoy"])) for row in forecast]
                    ),
                    None,
                    "",
                    radii["latest_qcew_yoy"],
                ),
                ModelForecast("pooled_ridge", pooled, None, pooled_alpha, radii["pooled_ridge"]),
                ModelForecast(
                    "equal_weight", equal, (0.25, 0.25, 0.25, 0.25), "", radii["equal_weight"]
                ),
                ModelForecast(
                    "inverse_mae",
                    inverse_prediction,
                    inverse_weights,
                    "",
                    radii["inverse_mae"],
                ),
                ModelForecast(
                    "convex_stack", stack_prediction, stack_weights, "", radii["convex_stack"]
                ),
            ]
        )
        if len({cast(str, row["forecast_origin"]) for row in oof_rows}) >= 8:
            quarter_ids = [cast(str, row["forecast_origin"]) for row in oof_rows]
            trailing = _trailing_mae_matrix(oof_matrix, oof_targets, quarter_ids)
            oof_context = _context_matrix(oof_rows)
            neural = fit_neural_gate(
                oof_matrix,
                trailing,
                oof_context,
                oof_targets,
                quarter_ids=quarter_ids,
            )
            forecast_mae = np.broadcast_to(
                np.mean(np.abs(oof_matrix - oof_targets[:, None]), axis=0),
                forecast_matrix.shape,
            ).copy()
            neural_prediction, neural_weights = predict_neural_gate(
                neural, forecast_matrix, forecast_mae, _context_matrix(forecast)
            )
            neural_radius = _empirical_radius(
                neural.predictions[neural.validation_index],
                oof_targets[neural.validation_index],
            )
            model_values.append(
                ModelForecast("neural_gate", neural_prediction, neural_weights, "", neural_radius)
            )
        for row_index, row in enumerate(forecast):
            outcome = float(cast(float, row["target_employment_growth_yoy"]))
            for model in model_values:
                prediction = float(model.predictions[row_index])
                selected_weights: WeightVector | None
                if isinstance(model.weights, np.ndarray):
                    selected_weights = cast(
                        WeightVector, tuple(float(value) for value in model.weights[row_index])
                    )
                else:
                    selected_weights = model.weights
                contributions = (
                    cast(
                        WeightVector,
                        tuple(
                            float(forecast_matrix[row_index, index] * selected_weights[index])
                            for index in range(4)
                        ),
                    )
                    if selected_weights is not None
                    else None
                )
                output.append(
                    {
                        "forecast_origin": outer_origin,
                        "state_fips": row["state_fips"],
                        "target_quarter": row["target_quarter"],
                        "census_division": row["census_division"],
                        "model_id": model.model_id,
                        "prediction": prediction,
                        "final_outcome": outcome,
                        "error": prediction - outcome,
                        "interval_lower_80": prediction - model.radius,
                        "interval_upper_80": prediction + model.radius,
                        "weight_labor": selected_weights[0] if selected_weights else "",
                        "weight_business": selected_weights[1] if selected_weights else "",
                        "weight_growth": selected_weights[2] if selected_weights else "",
                        "weight_housing": selected_weights[3] if selected_weights else "",
                        "contribution_labor": contributions[0] if contributions else "",
                        "contribution_business": contributions[1] if contributions else "",
                        "contribution_growth": contributions[2] if contributions else "",
                        "contribution_housing": contributions[3] if contributions else "",
                        "alpha": model.alpha,
                    }
                )
    return sorted(
        output,
        key=lambda row: (
            cast(str, row["forecast_origin"]),
            cast(str, row["state_fips"]),
            cast(str, row["model_id"]),
        ),
    )


def _one_hot(index: int) -> WeightVector:
    return cast(WeightVector, tuple(1.0 if position == index else 0.0 for position in range(4)))


def _empirical_radius(predictions: FloatArray, outcomes: FloatArray) -> float:
    residuals = np.sort(np.abs(np.asarray(predictions) - np.asarray(outcomes)))
    if len(residuals) == 0:
        return 0.0
    index = max(0, math.ceil(0.8 * len(residuals)) - 1)
    return float(residuals[index])


def _oof_predictions(train: list[Row]) -> OofResult:
    rows: list[Row] = []
    predictions: list[list[float]] = []
    pooled_predictions: list[float] = []
    origins = sorted({cast(str, row["forecast_origin"]) for row in train})
    experts = cast(list[Json], load_contract()["experts"])
    for origin in origins:
        prior = [
            row
            for row in train
            if cast(str, row["forecast_origin"]) < origin
            and date.fromisoformat(cast(str, row["outcome_available_date"])) <= quarter_end(origin)
        ]
        if len({cast(str, row["forecast_origin"]) for row in prior}) < 8:
            continue
        validation = [row for row in train if row["forecast_origin"] == origin]
        if not validation:
            continue
        expert_values: list[FloatArray] = []
        for expert in experts:
            features = cast(list[str], expert["numeric_features"])
            expert_values.append(
                _ridge_predict(prior, validation, features, _select_alpha(prior, features))
            )
        matrix = np.column_stack(expert_values)
        pooled_features = [
            feature for expert in experts for feature in cast(list[str], expert["numeric_features"])
        ]
        pooled_predictions.extend(
            _ridge_predict(
                prior,
                validation,
                pooled_features,
                _select_alpha(prior, pooled_features),
            ).tolist()
        )
        rows.extend(validation)
        predictions.extend(matrix.tolist())
    if not rows:
        return OofResult(
            [],
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    targets = np.asarray([float(cast(float, row["target_employment_growth_yoy"])) for row in rows])
    return OofResult(
        rows,
        np.asarray(predictions, dtype=np.float64),
        np.asarray(pooled_predictions, dtype=np.float64),
        targets,
    )


def _select_alpha(rows: list[Row], features: list[str]) -> float:
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
    origins = sorted({cast(str, row["forecast_origin"]) for row in rows})
    scores: dict[float, list[float]] = {alpha: [] for alpha in alphas}
    for validation_origin in origins[4:][-4:]:
        inner_train = [
            row
            for row in rows
            if cast(str, row["forecast_origin"]) < validation_origin
            and date.fromisoformat(cast(str, row["outcome_available_date"]))
            <= quarter_end(validation_origin)
        ]
        validation = [row for row in rows if row["forecast_origin"] == validation_origin]
        if len({cast(str, row["forecast_origin"]) for row in inner_train}) < 4:
            continue
        targets = np.asarray(
            [float(cast(float, row["target_employment_growth_yoy"])) for row in validation]
        )
        for alpha in alphas:
            prediction = _ridge_predict(inner_train, validation, features, alpha)
            scores[alpha].append(float(np.mean(np.square(prediction - targets))))
    ranked = [
        (float(np.mean(values)) if values else math.inf, -alpha, alpha)
        for alpha, values in scores.items()
    ]
    return min(ranked)[2] if any(values for values in scores.values()) else 100.0


def _ridge_predict(
    train: list[Row], forecast: list[Row], features: list[str], alpha: float
) -> FloatArray:
    x_train, x_forecast = _design_matrices(train, forecast, features)
    targets = np.asarray([float(cast(float, row["target_employment_growth_yoy"])) for row in train])
    model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
    model.fit(x_train, targets)
    return np.asarray(model.predict(x_forecast), dtype=np.float64)


def _design_matrices(
    train: list[Row], forecast: list[Row], features: list[str]
) -> tuple[FloatArray, FloatArray]:
    states = cast(list[str], cast(Json, load_contract()["population"])["state_fips"])[1:]

    def matrix(rows: list[Row]) -> FloatArray:
        return np.asarray(
            [
                [float(cast(float, row[feature])) for feature in features]
                + [1.0 if row["state_fips"] == state else 0.0 for state in states]
                + [1.0 if row["target_quarter_number"] == quarter else 0.0 for quarter in (2, 3, 4)]
                for row in rows
            ],
            dtype=np.float64,
        )

    train_matrix = matrix(train)
    forecast_matrix = matrix(forecast)
    numeric_width = len(features)
    mean = train_matrix[:, :numeric_width].mean(axis=0)
    scale = train_matrix[:, :numeric_width].std(axis=0)
    scale[scale == 0.0] = 1.0
    train_matrix[:, :numeric_width] = (train_matrix[:, :numeric_width] - mean) / scale
    forecast_matrix[:, :numeric_width] = (forecast_matrix[:, :numeric_width] - mean) / scale
    return train_matrix, forecast_matrix


def _context_matrix(rows: list[Row]) -> FloatArray:
    divisions = sorted(
        set(
            cast(
                dict[str, str], cast(Json, load_contract()["population"])["census_divisions"]
            ).values()
        )
    )
    return np.asarray(
        [
            [1.0 if row["target_quarter_number"] == quarter else 0.0 for quarter in (1, 2, 3, 4)]
            + [1.0 if row["census_division"] == division else 0.0 for division in divisions]
            for row in rows
        ],
        dtype=np.float64,
    )


def _trailing_mae_matrix(
    predictions: FloatArray, outcomes: FloatArray, quarter_ids: list[str]
) -> FloatArray:
    """Return MAEs using prior quarters only, never earlier states in the same quarter."""
    if len(predictions) != len(outcomes) or len(predictions) != len(quarter_ids):
        raise ValueError("trailing MAE inputs must have equal row counts")
    result = np.empty_like(predictions)
    running = np.ones(4, dtype=np.float64)
    historical_indexes: list[int] = []
    for quarter in sorted(set(quarter_ids)):
        current_indexes = [index for index, value in enumerate(quarter_ids) if value == quarter]
        result[current_indexes] = running
        historical_indexes.extend(current_indexes)
        running = np.mean(
            np.abs(
                predictions[historical_indexes]
                - outcomes[np.asarray(historical_indexes, dtype=np.int_), None]
            ),
            axis=0,
        )
    return result


def _metrics(predictions: list[Row]) -> Json:
    """Report overall and required grouped metrics without changing model selection."""
    return {
        "overall": _metric_group(predictions),
        "by_forecast_origin": _grouped_metrics(predictions, "forecast_origin"),
        "by_state": _grouped_metrics(predictions, "state_fips"),
        "by_census_division": _grouped_metrics(predictions, "census_division"),
    }


def _grouped_metrics(predictions: list[Row], field: str) -> Json:
    return {
        value: _metric_group([row for row in predictions if row[field] == value])
        for value in sorted({cast(str, row[field]) for row in predictions})
    }


def _metric_group(predictions: list[Row]) -> Json:
    result: Json = {}
    for model_id in sorted({cast(str, row["model_id"]) for row in predictions}):
        rows = [row for row in predictions if row["model_id"] == model_id]
        errors = np.asarray([float(cast(float, row["error"])) for row in rows])
        coverage = np.mean(
            [
                float(cast(float, row["interval_lower_80"]))
                <= float(cast(float, row["final_outcome"]))
                <= float(cast(float, row["interval_upper_80"]))
                for row in rows
            ]
        )
        result[model_id] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "interval_80_coverage": float(coverage),
            "row_count": len(rows),
        }
    return result


def _neural_split(row_count: int, quarter_ids: list[str] | None) -> tuple[IndexArray, IndexArray]:
    if quarter_ids is not None:
        if len(quarter_ids) != row_count:
            raise ValueError("quarter_ids length must match neural rows")
        unique = sorted(set(quarter_ids))
        if len(unique) < 8:
            raise ValueError("neural gate requires at least eight out-of-fold quarters")
        validation_quarters = set(unique[-4:])
        validation: IndexArray = np.asarray(
            [index for index, quarter in enumerate(quarter_ids) if quarter in validation_quarters],
            dtype=np.int_,
        )
        train: IndexArray = np.asarray(
            [
                index
                for index, quarter in enumerate(quarter_ids)
                if quarter not in validation_quarters
            ],
            dtype=np.int_,
        )
    else:
        validation_size = max(1, min(4, row_count // 5))
        train = np.arange(0, row_count - validation_size, dtype=np.int_)
        validation = np.arange(row_count - validation_size, row_count, dtype=np.int_)
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("neural gate train and validation partitions must be nonempty")
    return train, validation


def _gate_forward(
    features: FloatArray, w1: FloatArray, b1: FloatArray, w2: FloatArray, b2: FloatArray
) -> tuple[FloatArray, FloatArray]:
    hidden = np.tanh(features @ w1 + b1)
    logits = hidden @ w2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=1, keepdims=True), hidden
