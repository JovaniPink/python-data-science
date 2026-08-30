"""Point-in-time regional ensemble contract and modeling tests."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from python_data_science.regional import (
    EXPERT_IDS,
    _metrics,
    _trailing_mae_matrix,
    build_folds,
    build_panel,
    canonical_csv,
    contract_sha256,
    fit_neural_gate,
    load_contract,
    predict_neural_gate,
    quarter_add,
    quarter_end,
    search_convex_stack,
    validate_source_bundle,
)


def _quarters(start: str, end: str) -> list[str]:
    values: list[str] = []
    current = start
    while current <= end:
        values.append(current)
        current = quarter_add(current, 1)
    return values


def synthetic_bundle(tmp_path: Path) -> dict[str, object]:
    contract = load_contract()
    state_fips = contract["population"]["state_fips"]
    assert isinstance(state_fips, list)
    sources: list[dict[str, object]] = []
    for source_id, host in (("qcew", "bls.gov"), ("bea", "bea.gov"), ("fhfa", "fhfa.gov")):
        payload = f"synthetic-{source_id}\n".encode()
        path = tmp_path / f"{source_id}.source"
        path.write_bytes(payload)
        sources.append(
            {
                "source_id": source_id,
                "publisher_url": f"https://www.{host}/synthetic/{source_id}",
                "release_date": "2026-08-01",
                "retrieved_at": "2026-08-29T12:00:00Z",
                "media_type": "text/csv" if source_id != "fhfa" else "application/pdf",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
                "terms_url": f"https://www.{host}/terms",
                "vintage_status": "synthetic_test_fixture",
                "cache_path": str(path),
            }
        )

    qcew: list[dict[str, object]] = []
    bea: list[dict[str, object]] = []
    fhfa: list[dict[str, object]] = []
    fhfa_layout_checks: list[dict[str, object]] = []
    for quarter_index, quarter in enumerate(_quarters("2015Q1", "2025Q4")):
        end = quarter_end(quarter)
        report_url = f"https://www.fhfa.gov/reports/house-price-index/{quarter}"
        fhfa_release = (end + timedelta(days=60)).isoformat()
        fhfa_layout_checks.append(
            {
                "report_url": report_url,
                "release_date": fhfa_release,
                "layout_era": "synthetic_v1",
                "pdftotext_version": "pdftotext -layout synthetic 1.0",
                "row_count": 51,
                "expected_headings": True,
                "numeric_values": True,
                "warning_text_preserved": True,
                "manual_samples_verified": True,
            }
        )
        for state_index, state in enumerate(state_fips):
            scale = 100_000.0 + state_index * 1_000.0
            trend = 1.0 + quarter_index * 0.008 + state_index * 0.0001
            prelim_release = end + timedelta(days=75)
            final_release = end + timedelta(days=165)
            for status, release, adjustment in (
                ("preliminary", prelim_release, 0.997),
                ("final", final_release, 1.0),
            ):
                qcew.append(
                    {
                        "state_fips": state,
                        "observation_quarter": quarter,
                        "release_date": release.isoformat(),
                        "vintage": f"{quarter}-{status}",
                        "status": status,
                        "employment": scale * trend * adjustment,
                        "establishments": (scale / 20.0)
                        * (1.0 + quarter_index * 0.004)
                        * adjustment,
                        "total_wages": scale * 2_000.0 * (1.0 + quarter_index * 0.012) * adjustment,
                    }
                )
            bea.append(
                {
                    "state_fips": state,
                    "observation_quarter": quarter,
                    "release_date": (end + timedelta(days=80)).isoformat(),
                    "vintage": f"bea-{quarter}",
                    "real_gdp": scale * 10.0 * (1.0 + quarter_index * 0.01),
                    "personal_income": scale * 8.0 * (1.0 + quarter_index * 0.009),
                }
            )
            fhfa.append(
                {
                    "state_fips": state,
                    "observation_quarter": quarter,
                    "release_date": fhfa_release,
                    "report_url": report_url,
                    "hpi_qoq": 0.5 + state_index * 0.001 + quarter_index * 0.002,
                    "hpi_yoy": 2.0 + state_index * 0.002 + quarter_index * 0.004,
                }
            )

    return {
        "schema_version": "regional-source-bundle.v1",
        "contract_sha256": contract_sha256(),
        "research_cutoff": "2026-08-29",
        "extraction_tools": {"pdftotext": "pdftotext -layout synthetic 1.0"},
        "fhfa_layout_checks": fhfa_layout_checks,
        "sources": sources,
        "observations": {"qcew": qcew, "bea": bea, "fhfa": fhfa},
    }


def test_contract_hash_and_source_receipts_fail_closed(tmp_path: Path) -> None:
    assert contract_sha256() == "c1693dbe606629fcc1f63eb7a915f14219c7b2bc580ea85afc72371957f651c9"
    bundle = synthetic_bundle(tmp_path)
    validate_source_bundle(bundle)

    broken = json.loads(json.dumps(bundle))
    broken["sources"][0]["publisher_url"] = "https://example.com/not-authoritative"
    with pytest.raises(ValueError, match=r"sources\[0\]\.publisher_url"):
        validate_source_bundle(broken)

    broken = json.loads(json.dumps(bundle))
    broken["sources"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"sources\[1\]\.sha256"):
        validate_source_bundle(broken)

    broken = json.loads(json.dumps(bundle))
    broken["fhfa_layout_checks"][0]["warning_text_preserved"] = False
    with pytest.raises(ValueError, match="warning_text_preserved"):
        validate_source_bundle(broken)


def test_panel_uses_only_as_of_vintages_and_final_values_for_scoring(tmp_path: Path) -> None:
    bundle = synthetic_bundle(tmp_path)
    panel = build_panel(bundle)
    first = next(
        row for row in panel if row["forecast_origin"] == "2020Q1" and row["state_fips"] == "01"
    )

    assert first["target_quarter"] == "2020Q2"
    assert first["evaluation_origin"] is True
    assert date.fromisoformat(str(first["qcew_release_date"])) <= quarter_end("2020Q1")
    assert date.fromisoformat(str(first["bea_release_date"])) <= quarter_end("2020Q1")
    assert date.fromisoformat(str(first["fhfa_release_date"])) <= quarter_end("2020Q1")
    assert first["outcome_available_date"] > quarter_end("2020Q2").isoformat()
    assert math.isfinite(float(first["target_employment_growth_yoy"]))

    future = json.loads(json.dumps(bundle))
    future["observations"]["fhfa"][0]["release_date"] = "2027-01-01"
    with pytest.raises(ValueError, match="research cutoff"):
        validate_source_bundle(future)


def test_panel_and_folds_are_canonical_and_time_ordered(tmp_path: Path) -> None:
    panel = build_panel(synthetic_bundle(tmp_path))
    assert panel == sorted(
        panel, key=lambda row: (str(row["forecast_origin"]), str(row["state_fips"]))
    )
    assert len([row for row in panel if row["forecast_origin"] == "2020Q1"]) == 51

    panel_csv = canonical_csv(panel)
    assert panel_csv.endswith("\n")
    assert panel_csv.splitlines()[1].startswith("2017Q1,01,")

    folds = build_folds(panel)
    assert folds
    assert all(
        row["row_origin"] < row["outer_origin"] for row in folds if row["membership"] == "train"
    )
    assert all(
        date.fromisoformat(str(row["outcome_available_date"]))
        <= quarter_end(str(row["outer_origin"]))
        for row in folds
        if row["membership"] == "train"
    )
    assert all(row["membership"] in {"train", "forecast"} for row in folds)


def test_stack_grid_constraints_and_lexicographic_tie_break() -> None:
    predictions = np.array(
        [[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0], [3.0, 3.0, 4.0, 4.0]],
        dtype=np.float64,
    )
    outcome = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    weights = search_convex_stack(predictions, outcome)

    assert tuple(weights) == (0.0, 1.0, 0.0, 0.0)
    assert sum(weights) == pytest.approx(1.0)
    assert all(0.0 <= weight <= 1.0 for weight in weights)


def test_neural_gate_outputs_structurally_valid_weights() -> None:
    rng = np.random.default_rng(42)
    expert_predictions = rng.normal(size=(40, 4))
    trailing_mae = np.abs(rng.normal(size=(40, 4)))
    context = np.zeros((40, 13), dtype=np.float64)
    context[:, 0] = 1.0
    outcome = expert_predictions[:, 0] * 0.7 + expert_predictions[:, 1] * 0.3

    result = fit_neural_gate(expert_predictions, trailing_mae, context, outcome)

    assert result.weights.shape == (40, len(EXPERT_IDS))
    assert np.isfinite(result.predictions).all()
    assert np.all((result.weights >= 0.0) & (result.weights <= 1.0))
    assert np.allclose(result.weights.sum(axis=1), 1.0, atol=1.0e-6)
    assert result.epochs <= 500

    with pytest.raises(ValueError, match="four-column"):
        predict_neural_gate(result, expert_predictions[:, :3], trailing_mae, context)


def test_trailing_mae_never_uses_other_states_from_the_same_quarter() -> None:
    predictions = np.asarray(
        [[2.0, 4.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0], [9.0, 9.0, 9.0, 9.0]],
        dtype=np.float64,
    )
    outcomes = np.asarray([1.0, 3.0, 8.0], dtype=np.float64)
    trailing = _trailing_mae_matrix(predictions, outcomes, ["2020Q1", "2020Q1", "2020Q2"])

    assert np.array_equal(trailing[0], np.ones(4))
    assert np.array_equal(trailing[1], np.ones(4))
    assert np.allclose(trailing[2], np.asarray([1.0, 3.0, 5.0, 7.0]))


def test_metrics_report_every_required_group_and_even_median() -> None:
    rows = [
        {
            "forecast_origin": "2020Q1",
            "state_fips": state,
            "census_division": "new_england",
            "model_id": "zero",
            "error": error,
            "final_outcome": 1.0,
            "interval_lower_80": 0.0,
            "interval_upper_80": 2.0,
        }
        for state, error in (("09", 1.0), ("25", 3.0))
    ]
    metrics = _metrics(rows)

    assert set(metrics) == {
        "overall",
        "by_forecast_origin",
        "by_state",
        "by_census_division",
    }
    assert metrics["overall"]["zero"]["median_absolute_error"] == 2.0
