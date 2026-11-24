"""Unit — AN-1 experiment helpers (Spec 026 §5.1 T4). Red phase first (Article III)."""

from __future__ import annotations

import pytest

from cuas_sim.analysis.experiment import (
    _delta_rows,
    _design_point_to_config,
    factors_from_names,
)
from cuas_sim.config import MissionConfig


def test_paired_delta_synthetic() -> None:
    rows = [
        {"design_point_id": 0, "rep": 0, "seed": 111, "lambda_per_h": 6.0,
         "uas_archetype_p_mr": 0.5, "config": "baseline", "aor": "ALL",
         "metric": "MOS-1", "value": 0.10},
        {"design_point_id": 0, "rep": 0, "seed": 111, "lambda_per_h": 6.0,
         "uas_archetype_p_mr": 0.5, "config": "modernized", "aor": "ALL",
         "metric": "MOS-1", "value": 0.34},
    ]
    deltas = [
        r for r in _delta_rows(rows)
        if r["config"] == "delta" and r["metric"] == "MOS-1" and r["aor"] == "ALL"
    ]
    assert len(deltas) == 1
    assert deltas[0]["value"] == pytest.approx(0.24, abs=1e-12)


def test_design_point_to_config() -> None:
    base = MissionConfig()
    row = {"design_point_id": 0, "lambda_per_h": 12.0, "uas_archetype_p_mr": 0.3}
    cfg = _design_point_to_config(base, row)
    assert cfg.lambda_per_sec == pytest.approx(12.0 / 3600.0)
    assert cfg.uas_archetype_p_mr == pytest.approx(0.3)


def test_factors_from_names() -> None:
    flist = factors_from_names(["lambda", "p_mr"])
    assert [f.name for f in flist] == ["lambda_per_h", "uas_archetype_p_mr"]
    with pytest.raises(ValueError):
        factors_from_names(["nope"])
