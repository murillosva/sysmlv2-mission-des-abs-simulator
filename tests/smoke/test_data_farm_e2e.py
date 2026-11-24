"""Smoke — AN-1 data farm end-to-end (Spec 026 §5.3). Real SimPy (Article VIII)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cuas_sim.analysis.experiment import (
    _design_point_to_config,
    run_data_farm,
    write_datafarm,
)
from cuas_sim.config import MissionConfig
from cuas_sim.optimization.loop_controller import _crn_seeds
from cuas_sim.simulation.run import run_paired

pytestmark = pytest.mark.smoke


@pytest.fixture
def base_cfg() -> MissionConfig:
    return MissionConfig(lambda_per_sec=12.0 / 3600.0, skip_pre_pea=True)


def _single_point_design() -> pd.DataFrame:
    return pd.DataFrame(
        [{"design_point_id": 0, "lambda_per_h": 12.0, "uas_archetype_p_mr": 0.5}]
    )


def test_read_only_consistency(base_cfg: MissionConfig) -> None:
    design = _single_point_design()
    df = run_data_farm(
        base_cfg, design, n_reps=2, base_seed=42, max_arrivals=20, workers=1,
    )
    seeds = _crn_seeds(42, 0, 2)
    for rep, seed in enumerate(seeds):
        cfg = _design_point_to_config(base_cfg, design.iloc[0].to_dict())
        res = run_paired(cfg, seed=seed, max_arrivals=20, persist=False)
        sub = res.metrics[
            (res.metrics["config_name"] == "modernized")
            & (res.metrics["metric"] == "MOS-1")
            & (res.metrics["aor"] == "ALL")
        ]
        expected = float(sub["value"].iloc[0])
        got = df[
            (df["design_point_id"] == 0)
            & (df["rep"] == rep)
            & (df["config"] == "modernized")
            & (df["metric"] == "MOS-1")
            & (df["aor"] == "ALL")
        ]["value"]
        assert float(got.iloc[0]) == expected


def test_datafarm_reproducible(base_cfg: MissionConfig, tmp_path: Path) -> None:
    design = _single_point_design()
    df1 = run_data_farm(
        base_cfg, design, n_reps=2, base_seed=7, max_arrivals=20, workers=1,
    )
    df2 = run_data_farm(
        base_cfg, design, n_reps=2, base_seed=7, max_arrivals=20, workers=1,
    )
    assert df1.equals(df2)

    p1 = tmp_path / "a" / "datafarm.parquet"
    p2 = tmp_path / "b" / "datafarm.parquet"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    write_datafarm(df1, p1, base_seed=7, n_reps=2, max_arrivals=20, design_ref="design.parquet")
    write_datafarm(df2, p2, base_seed=7, n_reps=2, max_arrivals=20, design_ref="design.parquet")
    assert p1.read_bytes() == p2.read_bytes()
