"""Tests for typed regional-expert-ensemble.v2 boundaries."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from python_data_science.regional import contract_sha256 as v1_contract_sha256
from python_data_science.regional_v2 import (
    V1_SHA256,
    V2_SHA256,
    ArtifactReceipt,
    PanelRow,
    PublishedObservation,
    artifact_receipt,
    contract_sha256,
    first_complete_quarter_origin,
    first_eligible_outer_origin,
    load_contract,
    profile,
    quarter_add,
    quarter_end,
    require_complete_quarter_origin,
    screened_convex_stack,
    validate_artifact,
)


def test_shared_v2_contract_does_not_change_v1() -> None:
    assert load_contract()["schema_version"] == "regional-expert-ensemble.v2"
    assert contract_sha256() == V2_SHA256
    assert v1_contract_sha256() == V1_SHA256


def test_first_complete_qcew_origin_comes_from_release_dates() -> None:
    states = ("01", "02")
    releases = (
        ("2017Q1", date(2017, 9, 6)),
        ("2017Q2", date(2017, 12, 7)),
        ("2017Q3", date(2018, 3, 8)),
        ("2017Q4", date(2018, 6, 7)),
        ("2018Q1", date(2018, 9, 6)),
        ("2018Q2", date(2018, 12, 6)),
    )
    observations = [
        PublishedObservation(
            source_id="qcew",
            state_fips=state_fips,
            observation_period=quarter,
            release_date=release,
            vintage=release.isoformat(),
            values={"employment": 100.0},
        )
        for state_fips in states
        for quarter, release in releases
    ]

    assert (
        first_complete_quarter_origin(observations, states, "2017Q1", "2020Q1", (0, 1, 4, 5))
        == "2018Q4"
    )
    with pytest.raises(ValueError, match="incomplete point-in-time feature origin: 2017Q1"):
        require_complete_quarter_origin(observations, states, "2017Q1", (0, 1, 4, 5))


def test_first_outer_origin_requires_eight_published_label_quarters() -> None:
    rows = [
        PanelRow(
            forecast_origin=(origin := quarter_add("2018Q4", index)),
            state_fips="01",
            target_quarter=quarter_add(origin, 1),
            target_quarter_number=int(origin[5]),
            census_division="east_south_central",
            evaluation_origin=True,
            features={},
            source_release_dates={},
            target=1.0,
            outcome_available_date=quarter_end(origin) + timedelta(days=180),
            target_vintage="final",
        )
        for index in range(10)
    ]
    assert first_eligible_outer_origin(rows, 8) == "2021Q1"


def test_artifact_paths_fail_closed(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    source = normalized / "qcew.csv"
    source.write_text("state_fips\n01\n", encoding="ascii")
    receipt = artifact_receipt(tmp_path, "normalized/qcew.csv", 1)
    validate_artifact(tmp_path, receipt)

    with pytest.raises(ValueError, match="unsafe relative path"):
        validate_artifact(
            tmp_path,
            ArtifactReceipt("/tmp/qcew.csv", receipt.sha256, receipt.byte_count, 1),
        )
    with pytest.raises(ValueError, match="unsafe relative path"):
        validate_artifact(
            tmp_path,
            ArtifactReceipt("../qcew.csv", receipt.sha256, receipt.byte_count, 1),
        )

    link = normalized / "qcew-link.csv"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink rejected"):
        validate_artifact(
            tmp_path,
            ArtifactReceipt("normalized/qcew-link.csv", receipt.sha256, receipt.byte_count, 1),
        )

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "qcew.csv").write_text("state_fips\n01\n", encoding="ascii")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink rejected"):
        validate_artifact(
            tmp_path,
            ArtifactReceipt("linked/qcew.csv", receipt.sha256, receipt.byte_count, 1),
        )


def test_profiles_are_explicit_and_conditional_profiles_remain_inactive() -> None:
    leading = profile("leading_signals")
    assert leading.experts == (
        "labor",
        "qcew_business",
        "industry",
        "formation",
        "construction",
        "growth",
        "housing",
    )
    assert leading.gate_context == ("treasury",)
    with pytest.raises(ValueError, match="inactive profile"):
        profile("energy_prospective")


def test_screened_stack_ranks_by_prior_mae_and_emits_exact_zero_weights() -> None:
    experts = ("labor", "qcew_business", "industry", "formation", "construction", "growth")
    maes = {
        "labor": 2.0,
        "qcew_business": 1.0,
        "industry": 1.0,
        "formation": 3.0,
        "construction": 0.5,
        "growth": 4.0,
    }
    predictions = [
        [1.0, 1.0, 1.0, 10.0, 1.0, 10.0],
        [2.0, 2.0, 2.0, 10.0, 2.0, 10.0],
        [3.0, 3.0, 3.0, 10.0, 3.0, 10.0],
    ]
    result = screened_convex_stack(experts, maes, predictions, [1.0, 2.0, 3.0])
    assert result.selected_experts == ("construction", "qcew_business", "industry", "labor")
    assert set(result.weights) == set(experts)
    assert result.weights["formation"] == 0.0
    assert result.weights["growth"] == 0.0
    assert sum(result.weights.values()) == pytest.approx(1.0)
