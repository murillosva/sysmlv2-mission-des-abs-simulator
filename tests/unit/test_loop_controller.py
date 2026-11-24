"""Unit tests for the OPT↔SIM loop controller and SCP solver.

Pure-function tests only — no SimPy, no filesystem I/O.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cuas_sim.optimization.loop_controller import (
    LoopResult,
    LoopState,
    _compute_ci,
    _rinott_n,
    init_loop_state,
    step_loop,
)
from cuas_sim.optimization.scp_solver import compute_coverage_matrix, compute_demand, solve

# ─── SCP — demand vector (spec §5.1 T4) ──────────────────────────────────────


def test_demand_vector_with_critical_aor() -> None:
    mos1_by_aor = {"AoR-1": 0.52, "AoR-2": 0.89}
    aor_names = ["AoR-1", "AoR-2"]
    demand = compute_demand(mos1_by_aor, aor_names, target=0.80)
    # AoR-1 gap = 0.28 → ceil(1 + 0.28/0.3) = ceil(1.933) = 2
    # AoR-2 gap = 0 → ceil(1 + 0/0.3) = 1
    assert demand[0] == 2
    assert demand[1] == 1


def test_demand_all_above_target() -> None:
    mos1_by_aor = {"AoR-1": 0.85, "AoR-2": 0.90, "AoR-3": 0.95}
    aor_names = ["AoR-1", "AoR-2", "AoR-3"]
    demand = compute_demand(mos1_by_aor, aor_names, target=0.80)
    assert all(d == 1 for d in demand), "All AoRs above target → demand=1 (standard cover)"


def test_demand_at_target_equals_one() -> None:
    demand = compute_demand({"AoR-1": 0.80}, ["AoR-1"], target=0.80)
    assert demand[0] == 1  # gap=0 → ceil(1+0) = 1


def test_demand_missing_aor_defaults_zero() -> None:
    demand = compute_demand({}, ["AoR-X"], target=0.80)
    # Missing AoR treated as mos1=0.0 → gap=0.80 → ceil(1+0.80/0.3)=ceil(3.67)=4
    assert demand[0] == 4


# ─── SCP — coverage matrix (spec §5.1 T6) ────────────────────────────────────


def test_coverage_matrix_shape_and_values() -> None:
    from cuas_sim.geometry.coordinates import GeoPoint
    # FARP very close to CAP → should cover
    farp_near = GeoPoint(lat=-5.0, lon=-68.0, name="NEAR")
    # FARP 5000 km away → should NOT cover (t_v < 0)
    farp_far = GeoPoint(lat=40.0, lon=-68.0, name="FAR")
    cap = GeoPoint(lat=-5.0, lon=-68.0, name="CAP-1")

    farps = {"NEAR": farp_near, "FAR": farp_far}
    caps = {"CAP-1": cap}
    endurance_s = 5.2 * 3600.0
    speed_mps = 144.0

    matrix = compute_coverage_matrix(farps, caps, endurance_s, speed_mps)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == 1  # NEAR covers CAP-1
    assert matrix[1, 0] == 0  # FAR does not cover CAP-1


# ─── SCP — solve (spec §5.1 T6 / §5.2 P2) ───────────────────────────────────


def test_scp_solve_selects_minimum_farps() -> None:
    # FARP-A covers both anchor points; FARP-B covers only anchor 0
    coverage = np.array([[1, 1], [1, 0]])  # shape (2 FARPs, 2 anchors)
    demand = np.array([1, 1])
    active, obj = solve(coverage, demand, ["FARP-A", "FARP-B"])
    # Optimal: select only FARP-A (covers both)
    assert active == ["FARP-A"]
    assert obj == pytest.approx(1.0, abs=1e-6)


def test_scp_solve_infeasible_raises() -> None:
    # No FARP covers anchor point 1
    coverage = np.array([[1, 0]])  # 1 FARP, 2 anchors
    demand = np.array([1, 1])
    with pytest.raises(ValueError, match="infeasible"):
        solve(coverage, demand, ["FARP-A"])


def test_scp_classic_matches_minimum_cover() -> None:
    # Standard SCP (demand=1): matches minimum cover solution
    # FARP-A covers [0,1], FARP-B covers [1,2], FARP-C covers [0,2]
    coverage = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    demand = np.ones(3, dtype=int)
    active, _ = solve(coverage, demand, ["A", "B", "C"])
    # Any 2-FARP solution that covers all 3 anchors is optimal
    assert len(active) == 2


# ─── Rinott correction (spec §5.1 T5 / §5.2 P3) ─────────────────────────────


def test_rinott_n_returns_at_least_n0() -> None:
    pilot = [0.7, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.71, 0.70, 0.72]
    n = _rinott_n(pilot, k=6, alpha=0.05, n0=10, delta=0.05)
    assert n >= 10


def test_rinott_n_low_variance_returns_n0() -> None:
    # Very low variance → Rinott correction ≤ n0 → clamp to n0
    pilot = [0.80] * 10  # zero variance
    n = _rinott_n(pilot, k=6, alpha=0.05, n0=10, delta=0.05)
    assert n == 10


def test_rinott_n_high_variance_exceeds_n0() -> None:
    # High variance → Rinott requires more replications than n0
    pilot = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # variance=0.25
    n = _rinott_n(pilot, k=6, alpha=0.05, n0=10, delta=0.05)
    assert n > 10


# ─── Confidence interval (spec §5.1 T5 / §5.2 P3) ───────────────────────────


def test_ci_contains_mean() -> None:
    values = [0.72, 0.74, 0.70, 0.73, 0.71, 0.75, 0.72, 0.73, 0.70, 0.74]
    lower, upper = _compute_ci(values)
    mean = sum(values) / len(values)
    assert lower <= mean <= upper


def test_ci_single_value_returns_value() -> None:
    lower, upper = _compute_ci([0.75])
    assert lower == pytest.approx(0.75)
    assert upper == pytest.approx(0.75)


def test_ci_narrows_with_more_samples() -> None:
    rng = np.random.default_rng(42)
    small = list(rng.normal(0.75, 0.05, 5))
    large = list(rng.normal(0.75, 0.05, 30))
    lo5, hi5 = _compute_ci(small)
    lo30, hi30 = _compute_ci(large)
    assert (hi30 - lo30) < (hi5 - lo5)


# ─── LoopResult (spec §5.1 T1–T3) ────────────────────────────────────────────


def test_loop_result_accepted_reason() -> None:
    r = LoopResult(converged=True, reason="accepted", iterations=2,
                   final_mos1=0.83, final_ci_lower=0.81, active_farps=["SBPV"],
                   last_signal_path=None, history=[])
    assert r.converged is True
    assert r.reason == "accepted"


def test_loop_result_stagnation_reason() -> None:
    r = LoopResult(converged=False, reason="stagnation", iterations=3,
                   final_mos1=0.72, final_ci_lower=0.69, active_farps=["SBPV"],
                   last_signal_path=None, history=[])
    assert r.converged is False
    assert r.reason == "stagnation"


def test_loop_result_budget_reason() -> None:
    r = LoopResult(converged=False, reason="budget", iterations=8,
                   final_mos1=0.65, final_ci_lower=0.62, active_farps=[],
                   last_signal_path=None, history=[])
    assert r.reason == "budget"
    assert r.iterations == 8


# ─── step() API (spec §5.1 T7–T9) ────────────────────────────────────────────


def test_init_loop_state_returns_iteration_1(tmp_path: Path) -> None:
    from cuas_sim.config import MissionConfig

    cfg = MissionConfig(lambda_per_sec=12.0 / 3600.0, skip_pre_pea=True)
    state = init_loop_state(
        config_base=cfg,
        run_dir=tmp_path / "state_run",
        I_max=3,
        N0=2,
        N_final=5,
        max_arrivals=5,
    )
    assert isinstance(state, LoopState)
    assert state.iteration == 1
    assert state.history == []
    assert state.I_max == 3
    J = len(cfg.farps)
    K = len(cfg.caps)
    assert state.coverage_matrix.shape == (J, K)


def test_step_loop_advances_iteration(tmp_path: Path) -> None:
    from cuas_sim.config import MissionConfig
    from cuas_sim.optimization.loop_controller import IterationRecord

    cfg = MissionConfig(lambda_per_sec=12.0 / 3600.0, skip_pre_pea=True)
    state = init_loop_state(
        config_base=cfg,
        run_dir=tmp_path / "step_run",
        I_max=3,
        N0=2,
        N_final=5,
        max_arrivals=5,
    )
    new_state, record, _done, reason = step_loop(state)

    assert new_state.iteration == 2
    assert isinstance(record, IterationRecord)
    assert record.iteration == 1
    assert 0.0 <= record.mos1_mean <= 1.0
    assert record.ci_lower <= record.mos1_mean <= record.ci_upper
    assert len(new_state.history) == 1
    # reason="running" when not done; terminal reasons when done
    assert reason in {"accepted", "stagnation", "budget", "infeasible", "running"}


def test_step_loop_raises_when_exhausted(tmp_path: Path) -> None:
    import dataclasses

    from cuas_sim.config import MissionConfig

    cfg = MissionConfig(lambda_per_sec=12.0 / 3600.0, skip_pre_pea=True)
    state = init_loop_state(
        config_base=cfg,
        run_dir=tmp_path / "exhausted_run",
        I_max=2,
        N0=2,
        N_final=5,
        max_arrivals=5,
    )
    # Force iteration past I_max
    exhausted = dataclasses.replace(state, iteration=state.I_max + 1)
    with pytest.raises(RuntimeError):
        step_loop(exhausted)
