"""Smoke / integration oracles for `analysis stats` (Spec 027 §5.3 S1, S2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cuas_sim.analysis.report import build_report, write_report

pytestmark = pytest.mark.smoke


def _make_datafarm(seed: int = 0, n_points: int = 4, n_reps: int = 10) -> pd.DataFrame:
    """Minimal synthetic AN-1 datafarm (tidy/long) — no simulation run."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dp in range(n_points):
        lam = 2.0 + dp * 4.0
        pmr = 0.1 + dp * 0.2
        for rep in range(n_reps):
            sd = 1000 * dp + rep
            bsl = float(rng.uniform(0.05, 0.20))
            mod = bsl + float(rng.uniform(0.05, 0.30))
            base = {"design_point_id": dp, "lambda_per_h": lam,
                    "uas_archetype_p_mr": pmr, "rep": rep, "seed": sd, "aor": "ALL"}
            for cfg, val in (("baseline", bsl), ("modernized", mod), ("delta", mod - bsl)):
                rows.append({**base, "config": cfg, "metric": "MOS-1", "value": val})
            rows.append({**base, "config": "modernized",
                         "metric": "rockets_per_engagement", "value": float(rng.uniform(1, 5))})
    return pd.DataFrame(rows)


def test_stats_read_only_valid(tmp_path) -> None:
    df = _make_datafarm(seed=0)

    report = build_report(df, response="MOS-1", bootstrap_seed=7, n_resamples=499)
    write_report(report, tmp_path, with_figures=False)

    rj = tmp_path / "report.json"
    assert rj.exists()
    data = json.loads(rj.read_text(encoding="utf-8"))
    assert data["field_version"] == "1.0"
    for key in ("study", "hypothesis_tests", "feature_scores", "pareto_front", "eda"):
        assert key in data
    # read-only: the layer must not create a simulation run dir
    assert not (tmp_path / "runs").exists()


def test_report_reproducible(tmp_path) -> None:
    df = _make_datafarm(seed=0)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out in (out_a, out_b):
        report = build_report(df, response="MOS-1", bootstrap_seed=7, n_resamples=499)
        write_report(report, out, with_figures=False)

    assert (out_a / "report.json").read_bytes() == (out_b / "report.json").read_bytes()
