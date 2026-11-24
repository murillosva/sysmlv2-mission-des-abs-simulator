"""Smoke / integration oracles for `analysis report` (Spec 029 §5.3 S1, S2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cuas_sim.cli import app

pytestmark = pytest.mark.smoke

runner = CliRunner()


def _make_datafarm(
    seed: int = 0, n_points: int = 10, n_reps: int = 6,
    responses: tuple[str, ...] = ("MOS-1", "MOE-1", "MOE-2", "MOE-3"),
) -> pd.DataFrame:
    """Synthetic AN-1 datafarm (tidy/long) — no simulation run."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dp in range(n_points):
        lam = 2.0 + 16.0 * dp / (n_points - 1)
        pmr = dp / (n_points - 1)
        for rep in range(n_reps):
            sd = 1000 * dp + rep
            base = {"design_point_id": dp, "lambda_per_h": lam,
                    "uas_archetype_p_mr": pmr, "rep": rep, "seed": sd}
            for resp in responses:
                bsl = float(rng.uniform(0.05, 0.20))
                gain = 0.25 * pmr + 0.004 * lam + float(rng.normal(0.0, 0.01))
                mod = bsl + gain
                for cfg, val in (("baseline", bsl), ("modernized", mod),
                                 ("delta", mod - bsl)):
                    rows.append({**base, "config": cfg, "aor": "ALL",
                                 "metric": resp, "value": val})
            rows.append({**base, "config": "modernized", "aor": "ALL",
                         "metric": "rockets_per_engagement",
                         "value": float(rng.uniform(1, 5))})
    return pd.DataFrame(rows)


def _write_datafarm(tmp_path) -> str:
    df = _make_datafarm(seed=0)
    path = tmp_path / "datafarm.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    meta = {"field_version": "1.0", "base_seed": 42, "n_reps": 6,
            "max_arrivals": 20, "design_ref": "synthetic"}
    (tmp_path / "datafarm.parquet.meta.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8"
    )
    return str(path)


def test_report_read_only_valid(tmp_path) -> None:
    datafarm = _write_datafarm(tmp_path)
    out = tmp_path / "report"

    result = runner.invoke(app, [
        "analysis", "report", "--datafarm", datafarm,
        "--responses", "MOS-1,MOE-1,MOE-2,MOE-3", "--aor", "ALL",
        "--format", "html,json", "--bootstrap-seed", "20260701",
        "--model-seed", "20260701", "--no-figures", "-o", str(out),
    ])
    assert result.exit_code == 0, result.output

    rj = out / "report.json"
    assert rj.exists()
    data = json.loads(rj.read_text(encoding="utf-8"))
    assert data["field_version"] == "1.0"
    for key in ("study", "validity", "decision_summary", "master_table",
                "responses", "relationships", "anova_by_levels", "decision_index",
                "figures"):
        assert key in data, key
    assert (out / "report.html").exists()
    # read-only: the layer must not create a simulation run dir
    assert not (tmp_path / "runs").exists()
    assert not (out / "runs").exists()


def test_three_layers_byte_identical(tmp_path) -> None:
    datafarm = _write_datafarm(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out in (out_a, out_b):
        result = runner.invoke(app, [
            "analysis", "report", "--datafarm", datafarm,
            "--responses", "MOS-1,MOE-1", "--aor", "ALL",
            "--format", "html,json,md", "--bootstrap-seed", "20260701",
            "--model-seed", "20260701", "--no-figures", "-o", str(out),
        ])
        assert result.exit_code == 0, result.output

    for name in ("report.json", "report.html", "report.md"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes(), name
