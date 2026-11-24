"""Integration smoke for the Spec 019 Phase 4 HUD wiring.

Verifies: Spec 019 §3.1 FR-11, FR-12 and §4.3

Drives :class:`cuas_sim.viz.animate.MissionAnimator` against a minimal,
schema-correct run directory (spawns + engagements parquet + an
``events_baseline.jsonl.gz`` state log) and asserts that:

* FR-11 — ``_load_data`` pre-computes the per-A-29 HUD inputs
  (``_a29_eng_windows`` / ``_a29_shots``) and ``_a29_hud_fields`` returns a
  coherent (state, rockets, fuel∈[0,1], target, marker_lonlat) tuple while the
  aircraft is PURSUING / ENGAGING.
* FR-12 — the HUD is suppressed for AT_BASE / MISSION_COMPLETE
  (``should_render_hud``), and rendered (Annotation) for an active A-29.
* §4.3 — ``render_a29_hud`` returns a ``matplotlib.text.Annotation``.

Smoke (not unit) because it exercises the real parquet + gzip I/O and the full
``MissionAnimator`` construction path. Matplotlib runs on the headless ``Agg``
backend; ``animate()`` (the GUI window) is never opened.

Red phase (Article III): FAILS until Phase 4 lands ``src/cuas_sim/viz/hud.py``
and the animator's HUD caches + ``_a29_hud_fields``.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from cuas_sim.config import MissionConfig
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.policies.weapon import make_weapon_policy
from cuas_sim.states import A29State
from cuas_sim.viz.animate import MissionAnimator

matplotlib.use("Agg")

pytestmark = pytest.mark.viz

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "zida_shield_default.yaml"

_T_REPOSITION = 3_000.0     # FARP→CAP departure → fuel-clock anchor
_T_SPAWN = 9_200.0
_T_PURSUIT = 9_300.0
_T_ENV = _T_PURSUIT + 600.0
_T_FIRE = _T_ENV + 1.0
_N_SHOTS = 4
_T_KILL = _T_FIRE + (_N_SHOTS - 1) * 18.0 + 4.0


def _build_run_dir(run_dir: Path, cfg: MissionConfig) -> str:
    """Write spawns + engagements + events for one neutralised baseline engagement."""
    proj = LocalProjection(cfg.aoo_centroid_lat, cfg.aoo_centroid_lon)
    a29_id = sorted(cfg.a29_schedules)[0]
    schedule = cfg.a29_schedules[a29_id]
    cap_geo = cfg.caps[f"CAP-{schedule.cap_index}"]
    cap_xy = proj.project(cap_geo)

    spawn_xy = XY(cap_xy.x, cap_xy.y + 25_000.0)
    spawn_geo = proj.unproject(spawn_xy)

    pd.DataFrame([{
        "uas_id": 1,
        "spawn_time_s": _T_SPAWN,
        "lat": spawn_geo.lat,
        "lon": spawn_geo.lon,
        "heading_rad": -math.pi / 2.0,
        "speed_mps": 50.0,
        "archetype_name": "fixed-wing",
        "sector_index": 22,
    }]).to_parquet(run_dir / "spawns.parquet", index=False)

    env_max = make_weapon_policy(cfg, "baseline").envelope_max_m
    pd.DataFrame([{
        "config_name": "baseline",
        "a29_id": a29_id,
        "uas_id": 1,
        "t_pursuit_start_s": _T_PURSUIT,
        "t_envelope_entry_s": _T_ENV,
        "t_fire_s": _T_FIRE,
        "t_kill_s": _T_KILL,
        "firing_distance_m": 1_500.0,
        "mop3_envelope_max_m": env_max,
        "is_neutralized": True,
        "n_shots": _N_SHOTS,
        "rockets_remaining": cfg.rocket_budget_per_sortie - _N_SHOTS,
        "feasible": True,
    }]).to_parquet(run_dir / "engagements.parquet", index=False)

    # State log for the engaged A-29 (the others stay AT_BASE for this run).
    transitions = [
        (_T_REPOSITION, A29State.REPOSITIONING_TO_CAP),
        (cfg.t_pea_start_s, A29State.AT_CAP_PEA),
        (_T_PURSUIT, A29State.PURSUING),
        (_T_ENV, A29State.ENGAGING),
        (_T_KILL, A29State.AT_CAP_PEA),
    ]
    with gzip.open(run_dir / "events_baseline.jsonl.gz", "wt", encoding="utf-8") as fh:
        prev = A29State.AT_BASE
        for t_s, st in transitions:
            fh.write(json.dumps({
                "config": "baseline", "a29": a29_id, "t_s": float(t_s),
                "old": prev.value, "new": st.value,
            }) + "\n")
            prev = st
    # Empty modernized log so _load_data sees both configs.
    with gzip.open(run_dir / "events_modernized.jsonl.gz", "wt", encoding="utf-8"):
        pass

    return a29_id


def test_load_data_precomputes_hud_inputs(tmp_path: Path) -> None:
    """Smoke 1 — the per-A-29 engagement windows + shots are cached."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id = _build_run_dir(tmp_path, cfg)

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)

    assert hasattr(animator, "_a29_eng_windows"), "FR-11: missing engagement-window cache"
    assert hasattr(animator, "_a29_shot_times"), "FR-11: missing per-shot fire-times cache"

    windows = animator._a29_eng_windows["baseline"][a29_id]
    assert len(windows) == 1
    t_start, t_end, uid = windows[0]
    assert t_start == pytest.approx(_T_PURSUIT)
    assert t_end == pytest.approx(_T_KILL)   # neutralised → window ends at t_kill
    assert uid == "1"

    # Rockets are tallied at the per-shot FIRE instants (≥ t_fire), not at the
    # pursuit start — n_shots fire times spread across [t_fire, t_kill].
    fire_times = animator._a29_shot_times["baseline"][a29_id]
    assert len(fire_times) == _N_SHOTS
    assert fire_times[0] == pytest.approx(_T_FIRE)
    assert fire_times[-1] == pytest.approx(_T_KILL)
    assert all(_T_FIRE <= t <= _T_KILL for t in fire_times)
    assert fire_times == sorted(fire_times)


def test_hud_fields_active_during_engagement(tmp_path: Path) -> None:
    """Smoke 2 — mid-engagement the HUD fields are coherent (state/rockets/fuel/target)."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id = _build_run_dir(tmp_path, cfg)

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)
    t_mid = (_T_ENV + _T_KILL) / 2.0
    state, rockets, fuel_pct, target, marker = animator._a29_hud_fields(
        "baseline", a29_id, t_mid
    )

    assert state == A29State.ENGAGING.value
    # Mid-engagement only some of the n_shots have fired → strictly between full
    # and fully-depleted (the rockets tick down with the missiles, FR-11).
    budget = cfg.rocket_budget_per_sortie
    assert budget - _N_SHOTS < rockets < budget
    assert 0.0 <= fuel_pct <= 1.0
    # Fuel anchored at the CAP departure (3000 s); ~6.7 ks elapsed of 18.72 ks.
    assert fuel_pct == pytest.approx(
        1.0 - (t_mid - _T_REPOSITION) / cfg.a29_endurance_s, abs=1e-6
    )
    assert target == "1"
    lon, lat = marker
    assert math.isfinite(lon) and math.isfinite(lat)
    assert -90.0 <= lon <= 0.0 and -30.0 <= lat <= 15.0


def test_rockets_tick_down_at_fire_not_pursuit(tmp_path: Path) -> None:
    """Smoke 2b — rockets stay full through PURSUING and only deplete from t_fire on.

    Regression for the reported bug: the count must NOT drop at the pursuit start
    (≈12 min before firing); it must be full while PURSUING and reach
    budget − n_shots only once every shot has left the rail (by t_kill).
    """
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id = _build_run_dir(tmp_path, cfg)
    animator = MissionAnimator(tmp_path, _CONFIG_PATH)
    budget = cfg.rocket_budget_per_sortie

    def rockets(t: float) -> int:
        return animator._a29_hud_fields("baseline", a29_id, t)[1]

    # Full all through PURSUING and right up to the first shot.
    assert rockets(_T_PURSUIT + 1.0) == budget
    assert rockets(_T_ENV) == budget
    assert rockets(_T_FIRE - 0.001) == budget
    # First rocket away at t_fire; all gone by t_kill; monotone non-increasing.
    assert rockets(_T_FIRE) == budget - 1
    assert rockets(_T_KILL) == budget - _N_SHOTS
    seq = [rockets(t) for t in range(int(_T_FIRE), int(_T_KILL) + 1, 2)]
    assert seq == sorted(seq, reverse=True), "rocket count must be monotone non-increasing"


def test_hud_suppressed_at_base_and_full_fuel_idle(tmp_path: Path) -> None:
    """Smoke 3 — AT_BASE A-29s get no HUD; on-station (no engagement) shows full fuel, no target."""
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id = _build_run_dir(tmp_path, cfg)
    from cuas_sim.viz.hud import should_render_hud

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)

    # Before the mission the engaged A-29 is AT_BASE → HUD suppressed (FR-12).
    state0, *_ = animator._a29_hud_fields("baseline", a29_id, 0.0)
    assert state0 == A29State.AT_BASE.value
    assert should_render_hud(state0) is False

    # Another A-29 that never appears in the log is AT_BASE the whole run.
    other = sorted(cfg.a29_schedules)[1]
    state_other, *_ = animator._a29_hud_fields("baseline", other, _T_KILL)
    assert should_render_hud(state_other) is False

    # On station just before the engagement: full rockets, no active target.
    state_cap, rockets_cap, _fuel, target_cap, _m = animator._a29_hud_fields(
        "baseline", a29_id, _T_PURSUIT - 10.0
    )
    assert state_cap == A29State.AT_CAP_PEA.value
    assert rockets_cap == cfg.rocket_budget_per_sortie
    assert target_cap is None


def test_info_event_does_not_shadow_rtb_fuel_limit(tmp_path: Path) -> None:
    """Smoke 5 — the RTB_FUEL_LIMIT_INFO diagnostic must not shadow the real state.

    ``records.log_rtb_fuel_limit`` logs a non-transition info event
    (``old == new == RTB_FUEL_LIMIT_INFO``) at the *same* t_s as the real
    ``→ RTB_FUEL_LIMIT`` transition. The parser must drop it so the HUD (and the
    FR-10 magenta marker) report ``RTB_FUEL_LIMIT`` on the bingo→FARP leg.
    """
    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id = _build_run_dir(tmp_path, cfg)

    t_bingo = _T_KILL + 1_000.0
    t_farp = t_bingo + 1_500.0
    # Append a bingo sequence with the interleaved INFO record (same t_s).
    extra = [
        json.dumps({"config": "baseline", "a29": a29_id, "t_s": t_bingo,
                    "old": A29State.AT_CAP_PEA.value, "new": A29State.RTB_FUEL_LIMIT.value}),
        json.dumps({"config": "baseline", "a29": a29_id, "t_s": t_bingo,
                    "old": "RTB_FUEL_LIMIT_INFO", "new": "RTB_FUEL_LIMIT_INFO",
                    "pos_xy": [0.0, 0.0], "t_pea_actual_s": 1.0}),
        json.dumps({"config": "baseline", "a29": a29_id, "t_s": t_farp,
                    "old": A29State.RTB_FUEL_LIMIT.value, "new": A29State.AT_FARP_POST_PEA.value}),
    ]
    with gzip.open(tmp_path / "events_baseline.jsonl.gz", "at", encoding="utf-8") as fh:
        for line in extra:
            fh.write(line + "\n")

    animator = MissionAnimator(tmp_path, _CONFIG_PATH)
    # Mid bingo→FARP leg: the derived state is the real RTB_FUEL_LIMIT, not INFO.
    state, *_ = animator._a29_hud_fields("baseline", a29_id, (t_bingo + t_farp) / 2.0)
    assert state == A29State.RTB_FUEL_LIMIT.value


def test_render_a29_hud_returns_annotation(tmp_path: Path) -> None:
    """Smoke 4 — render_a29_hud draws a matplotlib Annotation on the axes."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    from cuas_sim.viz.hud import render_a29_hud

    cfg = MissionConfig.from_yaml(_CONFIG_PATH)
    a29_id = _build_run_dir(tmp_path, cfg)
    animator = MissionAnimator(tmp_path, _CONFIG_PATH)
    t_mid = (_T_ENV + _T_KILL) / 2.0
    state, rockets, fuel_pct, target, marker = animator._a29_hud_fields(
        "baseline", a29_id, t_mid
    )

    fig, ax = plt.subplots()
    try:
        ann = render_a29_hud(
            ax=ax, a29_id=a29_id, state=state, rockets_remaining=rockets,
            fuel_pct=fuel_pct, target_uas_id=target, marker_lonlat=marker,
        )
        assert isinstance(ann, Annotation)
        # The compact box mentions the rocket count and target.
        text = ann.get_text()
        assert str(rockets) in text
        assert target in text
    finally:
        plt.close(fig)
