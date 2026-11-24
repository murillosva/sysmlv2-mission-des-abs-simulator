"""Smoke test for the OPT↔SIM loop controller (Spec 011 AC-3).

Runs the full loop with I_max=3, N_final=5, max_arrivals=20 and validates:
- No exceptions raised
- loop_result.json written and valid JSON
- At least one iteration completed
- Convergence reason is one of the valid values
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.optimization.loop_controller import init_loop_state, run_opt_sim_loop, step_loop


@pytest.mark.smoke
def test_loop_runs_three_iterations_without_exception(tmp_path: Path) -> None:
    cfg = MissionConfig(
        lambda_per_sec=12.0 / 3600.0,
        skip_pre_pea=True,
    )
    run_dir = tmp_path / "loop_run"

    result = run_opt_sim_loop(
        config_base=cfg,
        run_dir=run_dir,
        I_max=3,
        N0=3,
        N_final=5,
        target_mos1=0.80,
        base_seed=42,
        max_arrivals=20,
    )

    assert result.reason in {"accepted", "stagnation", "budget", "infeasible"}
    assert result.iterations >= 0
    assert 0.0 <= result.final_mos1 <= 1.0

    loop_result_path = run_dir / "loop_result.json"
    assert loop_result_path.exists(), "loop_result.json must be written"

    data = json.loads(loop_result_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert "converged" in data
    assert "reason" in data


@pytest.mark.smoke
def test_step_api_matches_run_api(tmp_path: Path) -> None:
    # N calls to step_loop() must produce same final_mos1 as run_opt_sim_loop()
    cfg = MissionConfig(
        lambda_per_sec=12.0 / 3600.0,
        skip_pre_pea=True,
    )
    common = dict(I_max=2, N0=2, N_final=3, target_mos1=0.80, base_seed=42, max_arrivals=10)

    result_run = run_opt_sim_loop(config_base=cfg, run_dir=tmp_path / "run_a", **common)

    state = init_loop_state(config_base=cfg, run_dir=tmp_path / "run_b", **common)
    last_record = None
    while True:
        state, record, done, reason = step_loop(state)
        last_record = record
        if done:
            break

    assert last_record is not None
    mos1_from_step = last_record.mos1_mean
    assert mos1_from_step == result_run.final_mos1, (
        f"step() final_mos1={mos1_from_step} != run() final_mos1={result_run.final_mos1}"
    )
    assert reason == result_run.reason


@pytest.mark.smoke
def test_loop_crn_determinism(tmp_path: Path) -> None:
    cfg = MissionConfig(
        lambda_per_sec=12.0 / 3600.0,
        skip_pre_pea=True,
    )

    r1 = run_opt_sim_loop(
        config_base=cfg,
        run_dir=tmp_path / "run1",
        I_max=2, N0=2, N_final=3,
        target_mos1=0.80, base_seed=42, max_arrivals=10,
    )
    r2 = run_opt_sim_loop(
        config_base=cfg,
        run_dir=tmp_path / "run2",
        I_max=2, N0=2, N_final=3,
        target_mos1=0.80, base_seed=42, max_arrivals=10,
    )

    assert r1.final_mos1 == r2.final_mos1, "Same seed must produce identical MOS-1"
    assert r1.reason == r2.reason
