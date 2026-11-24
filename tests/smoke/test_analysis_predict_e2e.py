"""Smoke / integration oracles for `analysis predict` (Spec 028 §5.3 S1, S2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cuas_sim.analysis.prediction import build_surrogate_report
from cuas_sim.analysis.report import write_surrogate

pytestmark = pytest.mark.smoke


def _make_datafarm(seed: int = 0, n_points: int = 8, n_reps: int = 6) -> pd.DataFrame:
    """Minimal synthetic AN-1 datafarm (tidy/long) — no simulation run."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dp in range(n_points):
        lam = 2.0 + dp * 2.0
        pmr = 0.05 + dp * 0.1
        for rep in range(n_reps):
            bsl = float(rng.uniform(0.05, 0.20))
            mod = bsl + 0.01 * lam + float(rng.uniform(0.0, 0.05))
            base = {"design_point_id": dp, "lambda_per_h": lam,
                    "uas_archetype_p_mr": pmr, "rep": rep, "seed": 1000 * dp + rep,
                    "aor": "ALL"}
            for cfg, val in (("baseline", bsl), ("modernized", mod), ("delta", mod - bsl)):
                rows.append({**base, "config": cfg, "metric": "MOS-1", "value": val})
            rows.append({**base, "config": "modernized",
                         "metric": "rockets_per_engagement", "value": float(rng.uniform(1, 5))})
    return pd.DataFrame(rows)


def test_predict_read_only_valid(tmp_path) -> None:
    df = _make_datafarm(seed=0)

    report, grid = build_surrogate_report(df, response="MOS-1", model="rf", model_seed=42)
    write_surrogate(report, tmp_path, grid, with_figures=False)

    sj = tmp_path / "surrogate.json"
    assert sj.exists()
    data = json.loads(sj.read_text(encoding="utf-8"))
    assert data["field_version"] == "1.0"
    for key in ("study", "model", "cv", "feature_importances", "grid_prediction", "n_train"):
        assert key in data
    # dense grid persisted alongside, not inflating the JSON
    assert (tmp_path / "grid_prediction.csv").exists()
    # read-only: the layer must not create a simulation run dir
    assert not (tmp_path / "runs").exists()


def test_surrogate_reproducible(tmp_path) -> None:
    df = _make_datafarm(seed=0)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out in (out_a, out_b):
        report, grid = build_surrogate_report(df, response="MOS-1", model="rf", model_seed=42)
        write_surrogate(report, out, grid, with_figures=False)

    assert (out_a / "surrogate.json").read_bytes() == (out_b / "surrogate.json").read_bytes()
