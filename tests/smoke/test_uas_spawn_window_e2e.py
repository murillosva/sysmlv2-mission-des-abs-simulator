"""Smoke E2E test for Spec 018 — paired run with real SimPy (Article VIII).

Covers §5.1 (smoke) and §5.3 (regression oracles AC-6, AC-8):
- RTB_FUEL_LIMIT appears in both events_baseline.jsonl.gz and events_modernized.jsonl.gz.
- DEPARTING_CAP and RTB_BINGO do NOT appear when flag=True.
- outputs/runs/<ts>/window.json is parseable and has expected reason.
- *_first_bingo is not null for either configuration.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.simulation.run import run_paired

pytestmark = pytest.mark.smoke


def _events_states(run_dir: Path, config: str) -> set[str]:
    """Collect the set of ``new`` state values emitted in events_<config>.jsonl.gz."""
    states: set[str] = set()
    with gzip.open(run_dir / f"events_{config}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            new = obj.get("new")
            if isinstance(new, str):
                states.add(new)
    return states


def test_paired_run_emits_rtb_fuel_limit_in_both_configs(tmp_path: Path) -> None:
    base = MissionConfig(
        lambda_per_sec=12.0 / 3600.0,
        skip_pre_pea=True,
        uas_spawn_window_enabled=True,
        a29_endurance_s=600.0,  # short endurance forces bingo within smoke horizon
    )

    res = run_paired(
        base, seed=42, outputs_root=tmp_path,
        max_arrivals=20, persist=True, mos1_target=0.80,
    )
    assert res is not None

    runs_root = tmp_path / "runs"
    runs = list(runs_root.iterdir())
    assert len(runs) == 1
    rd = runs[0]

    # (1) RTB_FUEL_LIMIT appears in both event streams.
    states_b = _events_states(rd, "baseline")
    states_m = _events_states(rd, "modernized")
    assert "RTB_FUEL_LIMIT" in states_b, f"baseline states: {sorted(states_b)}"
    assert "RTB_FUEL_LIMIT" in states_m, f"modernized states: {sorted(states_m)}"

    # (2) Legacy terminals MUST NOT appear with flag=True.
    assert "DEPARTING_CAP" not in states_b, (
        "DEPARTING_CAP must not appear in baseline when flag=True"
    )
    assert "DEPARTING_CAP" not in states_m
    assert "RTB_BINGO" not in states_b, (
        "RTB_BINGO must not appear in baseline when flag=True"
    )
    assert "RTB_BINGO" not in states_m

    # (3) window.json exists, is parseable, and ends via all_first_bingo.
    window_path = rd / "window.json"
    assert window_path.exists(), "window.json must be persisted for every run"
    window = json.loads(window_path.read_text(encoding="utf-8"))
    assert window["field_version"] == "1.0"
    assert window["t_window_end_reason"] == "all_first_bingo", (
        f"expected all_first_bingo, got {window['t_window_end_reason']!r}"
    )

    # (4) Each configuration produced at least one first bingo.
    assert window["baseline_first_bingo"] is not None
    assert window["modernized_first_bingo"] is not None
    assert "a29_id" in window["baseline_first_bingo"]
    assert "t_s" in window["modernized_first_bingo"]
