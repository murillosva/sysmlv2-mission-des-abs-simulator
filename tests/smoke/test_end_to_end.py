"""End-to-end smoke test — runs the paired simulator and checks outputs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.simulation.run import run_paired


@pytest.fixture
def smoke_cfg() -> MissionConfig:
    """A quick configuration: λ=12/h, no pre-PEA, capped horizon."""
    return MissionConfig(
        lambda_per_sec=12.0 / 3600.0,
        skip_pre_pea=True,
    )


def test_paired_run_produces_all_artifacts(
    smoke_cfg: MissionConfig, tmp_path: Path,
) -> None:
    """One paired run with persistence should produce the documented artifacts."""
    res = run_paired(
        smoke_cfg, seed=42, outputs_root=tmp_path,
        max_arrivals=20, persist=True, mos1_target=0.80,
    )

    # The directory exists with the documented files
    runs_root = tmp_path / "runs"
    runs = list(runs_root.iterdir())
    assert len(runs) == 1
    rd = runs[0]
    for fname in ("run.yaml", "engagements.parquet", "metrics.parquet",
                  "leaks.parquet", "events_baseline.jsonl.gz",
                  "events_modernized.jsonl.gz"):
        assert (rd / fname).exists(), f"Missing {fname}"

    # engagements.parquet has the expected schema (subset)
    df = pd.read_parquet(rd / "engagements.parquet")
    expected_cols = {
        "config_name", "aor_name", "sector_index", "is_neutralized",
        "is_neutralized_before_zida", "moe1_engagement_time_s",
        "moe2_mean_per_engagement", "moe3_p_hit_expected",
    }
    assert expected_cols.issubset(set(df.columns))

    # metrics.parquet has the long format
    metrics = pd.read_parquet(rd / "metrics.parquet")
    expected_metrics = {"MOS-1", "MOS-2", "MOE-1", "MOE-2", "MOE-3",
                        "MOP-1", "MOP-2", "MOP-3"}
    assert expected_metrics.issubset(set(metrics["metric"].unique()))

    # events_*.jsonl.gz is decodable and non-empty
    with gzip.open(rd / "events_baseline.jsonl.gz", "rt") as fh:
        lines = fh.readlines()
    assert lines, "No baseline events recorded"
    for ln in lines[:3]:
        obj = json.loads(ln)
        for key in ("config", "a29", "t_s", "old", "new"):
            assert key in obj

    # MOS-1 sanity: 0 ≤ MOS-1 ≤ 1 for every row
    mos1 = metrics[metrics["metric"] == "MOS-1"]
    assert (mos1["value"] >= 0).all()
    assert (mos1["value"] <= 1.0).all()

    # Replanning signal: present iff triggered.
    sig_path = rd / "replanning_signal.json"
    if res.replanning_signal.triggered:
        assert sig_path.exists()
        payload = json.loads(sig_path.read_text())
        assert payload["triggered"] is True
        assert "leaks_per_aor" in payload
    else:
        assert not sig_path.exists()


def test_paired_run_is_deterministic_under_crn(
    smoke_cfg: MissionConfig, tmp_path: Path,
) -> None:
    """Re-running with the same seed yields the same metrics."""
    res1 = run_paired(smoke_cfg, seed=99, outputs_root=tmp_path / "a",
                      max_arrivals=15, persist=True)
    res2 = run_paired(smoke_cfg, seed=99, outputs_root=tmp_path / "b",
                      max_arrivals=15, persist=True)
    # MOS-1 ALL value identical
    def _mos1_all(df: pd.DataFrame, cfg_name: str) -> float:
        sub = df[(df["metric"] == "MOS-1") & (df["aor"] == "ALL")
                 & (df["config_name"] == cfg_name)]
        return float(sub["value"].iloc[0])
    assert _mos1_all(res1.metrics, "baseline") == _mos1_all(res2.metrics, "baseline")
    assert _mos1_all(res1.metrics, "modernized") == _mos1_all(res2.metrics, "modernized")
