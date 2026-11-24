"""Integration smoke for the Spec 019 Phase 3 missile-flyout wiring.

Verifies: Spec 019 §3.1 FR-5, FR-6

Drives :class:`cuas_sim.viz.animate.MissionAnimator` against a minimal,
schema-correct run directory (spawns + engagements parquet only — ``_load_data``
tolerates absent event logs) and asserts that:

* FR-5 — ``_load_data`` pre-computes ``_missile_events[config]`` as a list of
  :class:`~cuas_sim.viz.missile_flyout.MissileEvent`, with finite (lon, lat)
  endpoints on the map.
* FR-6 — scheduling matches the outcome: a **neutralised** multi-shot
  engagement spreads its ``n_shots`` impacts so the last lands exactly on
  ``t_kill_s`` (killing missile on the kill flash); a **miss** keeps the fixed
  ~13 s cadence from ``t_fire_s``.

Smoke (not unit) because it exercises the real parquet I/O + the full
``MissionAnimator`` construction path. Matplotlib runs on the headless ``Agg``
backend; ``animate()`` (the GUI window) is never opened.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.policies.weapon import make_weapon_policy
from cuas_sim.viz.animate import MissionAnimator
from cuas_sim.viz.missile_flyout import _SHOT_SPACING_S

matplotlib.use("Agg")

pytestmark = pytest.mark.viz

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "zida_shield_default.yaml"

_T_SPAWN = 9_200.0
_T_PURSUIT = 9_300.0
_T_ENV_CEILING = _T_PURSUIT + 600.0
_T_FIRE = _T_ENV_CEILING + 1.0
_FIRING_DISTANCE_M = 1_500.0


def _build_run_dir(
    run_dir: Path, cfg: MissionConfig, *, n_shots: int, is_neutralized: bool
) -> tuple[str, float]:
    """Write spawns + engagements parquet for one baseline engagement.

    Returns ``(a29_id, t_kill_s)`` — ``t_kill_s`` is NaN for a miss.
    """
    proj = LocalProjection(cfg.aoo_centroid_lat, cfg.aoo_centroid_lon)
    a29_id = sorted(cfg.a29_schedules)[0]
    schedule = cfg.a29_schedules[a29_id]
    cap_geo = cfg.caps[f"CAP-{schedule.cap_index}"]
    cap_xy = proj.project(cap_geo)

    spawn_xy = XY(cap_xy.x, cap_xy.y + 25_000.0)
    spawn_geo = proj.unproject(spawn_xy)
    uas_heading = -math.pi / 2.0
    uas_speed = 50.0

    pd.DataFrame([{
        "uas_id": 1,
        "spawn_time_s": _T_SPAWN,
        "lat": spawn_geo.lat,
        "lon": spawn_geo.lon,
        "heading_rad": uas_heading,
        "speed_mps": uas_speed,
        "archetype_name": "fixed-wing",
        "sector_index": 22,
    }]).to_parquet(run_dir / "spawns.parquet", index=False)

    # Neutralised: kill well after the first shot so the even spread is non-trivial.
    t_kill = _T_FIRE + (n_shots - 1) * 18.0 + 4.0 if is_neutralized else float("nan")
    env_max = make_weapon_policy(cfg, "baseline").envelope_max_m
    pd.DataFrame([{
        "config_name": "baseline",
        "a29_id": a29_id,
        "uas_id": 1,
        "t_pursuit_start_s": _T_PURSUIT,
        "t_envelope_entry_s": _T_ENV_CEILING,
        "t_fire_s": _T_FIRE,
        "t_kill_s": t_kill,
        "firing_distance_m": _FIRING_DISTANCE_M,
        "mop3_envelope_max_m": env_max,
        "is_neutralized": is_neutralized,
        "n_shots": n_shots,
        "feasible": True,
    }]).to_parquet(run_dir / "engagements.parquet", index=False)

    return a29_id, t_kill


def test_load_data_precomputes_missile_events(tmp_path: Path) -> None:
    """Smoke 1 — single-shot kill yields one flyout event impacting at t_kill."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    _, t_kill = _build_run_dir(tmp_path, cfg, n_shots=1, is_neutralized=True)

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)

    assert hasattr(animator, "_missile_events"), (
        "FR-5: _load_data must populate the _missile_events cache."
    )
    baseline = animator._missile_events["baseline"]
    assert len(baseline) == 1, f"single-shot → one event, got {len(baseline)}"
    ev = baseline[0]
    assert ev.is_neutralized is True
    assert ev.t_impact_s == pytest.approx(t_kill, abs=1e-6), "killing missile lands on t_kill"
    assert ev.t_impact_s - ev.t_fire_s > 1e-9, "must have a visible (positive-span) flight"
    for lon, lat in (ev.shooter_xy, ev.target_xy):
        assert math.isfinite(lon) and math.isfinite(lat)
        assert -90.0 <= lon <= 0.0 and -30.0 <= lat <= 15.0, "endpoint off-map"
    # Modernized panel had no engagement → no events.
    assert animator._missile_events["modernized"] == []


def test_neutralized_multishot_lands_at_kill(tmp_path: Path) -> None:
    """Smoke 2 — a neutralised 4-shot engagement spreads impacts ending at t_kill."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    _, t_kill = _build_run_dir(tmp_path, cfg, n_shots=4, is_neutralized=True)

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)
    events = sorted(animator._missile_events["baseline"], key=lambda e: e.shot_index)
    assert len(events) == 4, f"4-shot engagement → four events, got {len(events)}"

    # Monotonic fire times; every missile has a visible flight.
    for k in range(1, len(events)):
        assert events[k].t_fire_s > events[k - 1].t_fire_s, "fire times not increasing"
    for ev in events:
        assert ev.t_impact_s - ev.t_fire_s > 1e-9, "zero-span missile"
        assert ev.t_impact_s <= t_kill + 1e-9

    # Only the last shot is the killing round, and it impacts exactly at t_kill.
    assert [e.is_neutralized for e in events] == [False, False, False, True]
    assert events[-1].t_impact_s == pytest.approx(t_kill, abs=1e-6)


def test_miss_multishot_spaced_13s(tmp_path: Path) -> None:
    """Smoke 3 — a non-neutralised 3-shot engagement keeps the ~13 s cadence."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    _build_run_dir(tmp_path, cfg, n_shots=3, is_neutralized=False)

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)
    events = sorted(animator._missile_events["baseline"], key=lambda e: e.shot_index)
    assert len(events) == 3, f"3-shot miss → three events, got {len(events)}"

    fire_times = [ev.t_fire_s for ev in events]
    assert fire_times[0] == pytest.approx(_T_FIRE, abs=1e-6)
    for k in range(1, len(fire_times)):
        gap = fire_times[k] - fire_times[k - 1]
        assert gap == pytest.approx(_SHOT_SPACING_S, abs=1e-6), f"spacing {gap} != 13 s"
    assert all(ev.is_neutralized is False for ev in events), "a miss kills nothing"
