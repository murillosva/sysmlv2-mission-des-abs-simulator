"""Pure-function oracle tests for ``cuas_sim.viz.replay`` (Spec 019 §5.1).

Verifies: Spec 019

Group B (Red phase): all tests below must FAIL until C-1 lands the new
module ``src/cuas_sim/viz/replay.py``. Article III (Test-Oracle First).

Marker ``viz`` keeps these out of the default smoke selection — run with
``pytest tests/unit/test_viz_replay.py -v -m viz`` to target them.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

pytestmark = pytest.mark.viz


# ─── T1 ──────────────────────────────────────────────────────────────────────


def test_replay_pursuit_converges_to_envelope() -> None:
    """``replay_pursuit`` iterates ``pure_pursuit_step`` until separation ≤ envelope.

    UAS heading East at 50 m/s, A-29 at CAP 30 km south-east of UAS spawn.
    Last waypoint MUST satisfy distance ≤ envelope_max_m + 100 m and time
    MUST be within ±5 % of the analytical ``t_envelope_entry`` from
    ``collision_course_intercept`` (closed-form reference).
    """
    from cuas_sim.geometry.intercept import collision_course_intercept
    from cuas_sim.geometry.projection import XY
    from cuas_sim.viz.replay import replay_pursuit

    uas_spawn = (0.0, 0.0)
    # 30 km south-east of UAS spawn: SE direction → +x, -y in projected frame.
    offset = 30_000.0 / math.sqrt(2.0)
    cap_xy = (offset, -offset)
    uas_heading = 0.0   # East
    uas_speed = 50.0
    v_a29 = 110.0
    envelope_max = 14_000.0
    t_pursuit_start = 0.0

    # Analytical reference: collision-course intercept (closed-form,
    # bit-exact source for envelope-entry time on a fixed heading).
    geom = collision_course_intercept(
        a29_xy=XY(*cap_xy),
        a29_speed_mps=v_a29,
        uas_xy=XY(*uas_spawn),
        uas_heading_rad=uas_heading,
        uas_speed_mps=uas_speed,
        envelope_max_m=envelope_max,
    )
    assert geom.feasible, "Test fixture must be feasible — adjust geometry."
    t_env_expected = geom.t_envelope_entry

    waypoints = replay_pursuit(
        cap_xy=cap_xy,
        uas_spawn_xy=uas_spawn,
        uas_heading_rad=uas_heading,
        uas_speed_mps=uas_speed,
        t_pursuit_start_s=t_pursuit_start,
        t_envelope_entry_s=t_env_expected,
        v_a29_mps=v_a29,
        envelope_max_m=envelope_max,
        dt=2.0,
    )
    assert waypoints, "replay_pursuit returned no waypoints."

    last = waypoints[-1]
    # UAS position at last.t_s under constant-velocity projection.
    ux = uas_spawn[0] + uas_speed * math.cos(uas_heading) * (last.t_s - t_pursuit_start)
    uy = uas_spawn[1] + uas_speed * math.sin(uas_heading) * (last.t_s - t_pursuit_start)
    sep = math.hypot(ux - last.x_m, uy - last.y_m)
    assert sep <= envelope_max + 100.0, (
        f"Last separation {sep:.1f} m exceeds envelope+100m={envelope_max + 100.0:.1f}"
    )
    # ±5 % tolerance on analytical envelope entry time. Pure pursuit lags the
    # collision-course optimum, so a small positive bias is expected.
    rel_err = abs(last.t_s - t_env_expected) / max(1.0, t_env_expected)
    assert rel_err <= 0.05, (
        f"Envelope-entry time off by {rel_err * 100:.2f} % (>5%). "
        f"got={last.t_s:.2f}, expected={t_env_expected:.2f}"
    )


# ─── T2 ──────────────────────────────────────────────────────────────────────


def test_replay_engaging_respects_frozen_heading() -> None:
    """ENGAGING waypoints advance colinearly along ``env_entry_heading``.

    Displacement between consecutive waypoints MUST be colinear with
    ``(cos h, sin h)`` within 1e-6 rad AND magnitude == ``v * dt`` within
    1e-3 m. Replicates the Spec 015/016 frozen-heading semantics.
    """
    from cuas_sim.viz.replay import replay_engaging

    env_entry_xy = (1000.0, -500.0)
    heading = math.radians(37.0)   # arbitrary non-axial heading
    t0 = 100.0
    t1 = 130.0
    v = 144.0
    dt = 1.0

    waypoints = replay_engaging(
        env_entry_xy=env_entry_xy,
        env_entry_heading_rad=heading,
        t_envelope_entry_s=t0,
        t_end_s=t1,
        v_a29_mps=v,
        dt=dt,
    )
    assert len(waypoints) >= 2, "replay_engaging needs at least two waypoints."

    cos_h, sin_h = math.cos(heading), math.sin(heading)
    for a, b in pairwise(waypoints):
        dx = b.x_m - a.x_m
        dy = b.y_m - a.y_m
        mag = math.hypot(dx, dy)
        # Colinearity: cross product of displacement with heading unit must be
        # near zero (angular deviation ≤ 1e-6 rad when normalised).
        cross = dx * sin_h - dy * cos_h
        assert abs(cross) <= 1e-6 * max(1.0, mag), (
            f"Displacement not colinear with frozen heading: cross={cross:.3e}"
        )
        assert b.t_s > a.t_s, "Non-monotonic time in ENGAGING replay."
        # Magnitude == v * (t_b - t_a) within 1e-3 m.
        expected_mag = v * (b.t_s - a.t_s)
        assert abs(mag - expected_mag) <= 1e-3, (
            f"Step magnitude {mag:.6f} != v·dt={expected_mag:.6f}"
        )
        # Heading recorded on each waypoint must equal the frozen heading.
        assert abs(b.heading_rad - heading) <= 1e-9, (
            f"Heading drifted: {b.heading_rad} vs {heading}"
        )


# ─── T3 ──────────────────────────────────────────────────────────────────────


def test_replay_rtb_terminates_at_target() -> None:
    """``replay_rtb`` last waypoint == target position (xy and time)."""
    from cuas_sim.viz.replay import replay_rtb

    start = (12_345.0, -6_789.0)
    target = (-3_500.0, 8_000.0)
    t_start = 250.0
    t_end = 487.5

    waypoints = replay_rtb(
        start_xy=start, target_xy=target, t_start_s=t_start, t_end_s=t_end
    )
    assert waypoints, "replay_rtb returned no waypoints."

    last = waypoints[-1]
    assert abs(last.x_m - target[0]) < 1e-6
    assert abs(last.y_m - target[1]) < 1e-6
    assert abs(last.t_s - t_end) < 1e-9


# ─── T4 ──────────────────────────────────────────────────────────────────────


def test_build_a29_trajectory_phase_continuity() -> None:
    """Across the full trajectory, every consecutive pair satisfies
    spatial_gap ≤ v_a29 * dt + 10 m AND dt > 0 — i.e. the A-29 never
    teleports between PURSUING → ENGAGING → RTB transitions.
    """
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.geometry.coordinates import GeoPoint
    from cuas_sim.geometry.intercept import collision_course_intercept
    from cuas_sim.geometry.projection import XY, LocalProjection
    from cuas_sim.policies.weapon import make_weapon_policy
    from cuas_sim.viz.replay import build_a29_trajectory

    cfg = MissionConfig()  # defaults
    proj = LocalProjection(cfg.aoo_centroid_lat, cfg.aoo_centroid_lon)

    # A-29-1 patrols CAP-1 by default schedule.
    cap_xy = proj.project(cfg.caps["CAP-1"])

    # UAS spawned ~25 km north of CAP-1, heading south at 50 m/s.
    spawn_xy = XY(cap_xy.x, cap_xy.y + 25_000.0)
    spawn_geo = proj.unproject(spawn_xy)
    uas_heading = -math.pi / 2.0   # south
    uas_speed = 50.0
    t_spawn = 9_200.0
    t_pursuit = 9_300.0   # 100 s after spawn

    spawns_df = pd.DataFrame([{
        "uas_id": 1,
        "spawn_time_s": t_spawn,
        "lat": spawn_geo.lat,
        "lon": spawn_geo.lon,
        "heading_rad": uas_heading,
        "speed_mps": uas_speed,
        "archetype_name": "fixed-wing",
        "sector_index": 22,
    }])

    weapon = make_weapon_policy(cfg, "baseline")
    env_max = weapon.envelope_max_m

    # Use analytical envelope entry as anchor for the engagement record.
    uas_xy_at_ps = XY(
        spawn_xy.x + uas_speed * math.cos(uas_heading) * (t_pursuit - t_spawn),
        spawn_xy.y + uas_speed * math.sin(uas_heading) * (t_pursuit - t_spawn),
    )
    geom = collision_course_intercept(
        a29_xy=cap_xy, a29_speed_mps=cfg.a29_cruise_speed_mps,
        uas_xy=uas_xy_at_ps, uas_heading_rad=uas_heading, uas_speed_mps=uas_speed,
        envelope_max_m=env_max,
    )
    assert geom.feasible
    t_env = t_pursuit + geom.t_envelope_entry
    t_fire = t_env + 1.0
    t_kill = t_fire + 5.0   # short engagement, baseline single-shot

    engagements_df = pd.DataFrame([{
        "config_name": "baseline",
        "a29_id": "A-29-1",
        "uas_id": 1,
        "aor_name": "AoR-1",
        "sector_index": 22,
        "archetype_name": "fixed-wing",
        "t_pursuit_start_s": t_pursuit,
        "t_envelope_entry_s": t_env,
        "t_fire_s": t_fire,
        "t_kill_s": t_kill,
        "initial_distance_m": float(geom.initial_distance_m),
        "intercept_distance_m": 0.0,
        "firing_distance_m": 1_500.0,
        "is_neutralized": True,
        "is_neutralized_before_zida": True,
        "moe1_engagement_time_s": t_kill - t_pursuit,
        "moe2_mean_per_engagement": 1.0,
        "moe2_n_samples": 1,
        "moe3_p_hit_expected": 0.8,
        "mop1_heading_update_count": 5,
        "mop2_acquisition_range_m": cfg.baseline_max_acquisition_m,
        "mop3_envelope_max_m": env_max,
        "n_shots": 1,
        "rockets_remaining": cfg.rocket_budget_per_sortie - 1,
        "feasible": True,
        "note": "",
    }])

    arr = build_a29_trajectory(
        a29_id="A-29-1",
        config_name="baseline",
        engagements_df=engagements_df,
        events=[],
        spawns_df=spawns_df,
        mission_cfg=cfg,
        weapon_policy=weapon,
    )
    assert arr.shape[1] == 4, f"Expected (N, 4) array; got shape {arr.shape}"
    assert arr.shape[0] >= 4, "Trajectory should contain at least anchor + 3 phases."

    v_a29 = cfg.a29_cruise_speed_mps
    for i in range(arr.shape[0] - 1):
        t0, lon0, lat0, _ = arr[i]
        t1, lon1, lat1, _ = arr[i + 1]
        dt = t1 - t0
        assert dt > 0.0, f"Non-positive dt at i={i}: t0={t0}, t1={t1}"
        xy0 = proj.project(GeoPoint(lat=float(lat0), lon=float(lon0)))
        xy1 = proj.project(GeoPoint(lat=float(lat1), lon=float(lon1)))
        gap = math.hypot(xy1.x - xy0.x, xy1.y - xy0.y)
        bound = v_a29 * dt + 10.0
        assert gap <= bound, (
            f"Spatial jump too large at i={i}: gap={gap:.2f} m, bound={bound:.2f} m"
        )


# ─── Deployment + recovery legs (Phase 5) ──────────────────────────────────────


def test_build_a29_trajectory_deploys_and_recovers() -> None:
    """With an event log the trajectory flies SBPV→FARP→CAP…CAP→FARP→SBPV.

    Without the deployment/recovery legs the A-29 sat parked at the CAP for the
    whole replay; this checks the symbolic pre-PEA (TRANSIT_TO_FARP →
    REPOSITIONING_TO_CAP → HOLDING_ORBIT) and recovery (RTB_FUEL_LIMIT →
    AT_FARP_POST_PEA → RECOVERING_TO_BASE → MISSION_COMPLETE) transitions drive
    the marker through the base, the FARP, and the CAP at the logged times.
    """
    import numpy as np
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.policies.weapon import make_weapon_policy
    from cuas_sim.states import A29State
    from cuas_sim.viz.replay import build_a29_trajectory

    cfg = MissionConfig()
    a29_id = "A-29-1"
    schedule = cfg.a29_schedules[a29_id]
    base = cfg.main_base
    farp = cfg.farps[schedule.farp_name]
    cap = cfg.caps[f"CAP-{schedule.cap_index}"]

    t_farp_arr, t_cap_arr, t_pea = 3_180.0, 4_740.0, 9_180.0
    t_rtb, t_farp2, t_complete = 20_000.0, 21_900.0, 25_100.0
    events = [
        (0.0, A29State.TRANSIT_TO_FARP.value),
        (t_farp_arr, A29State.AT_FARP_PRE_PEA.value),
        (t_farp_arr, A29State.REPOSITIONING_TO_CAP.value),
        (t_cap_arr, A29State.HOLDING_ORBIT.value),
        (t_pea, A29State.AT_CAP_PEA.value),
        (t_rtb, A29State.RTB_FUEL_LIMIT.value),
        (t_farp2, A29State.AT_FARP_POST_PEA.value),
        (t_farp2, A29State.RECOVERING_TO_BASE.value),
        (t_complete, A29State.MISSION_COMPLETE.value),
    ]

    weapon = make_weapon_policy(cfg, "baseline")
    arr = build_a29_trajectory(
        a29_id=a29_id,
        config_name="baseline",
        engagements_df=pd.DataFrame(),   # this A-29 never engages
        events=events,
        spawns_df=pd.DataFrame(),
        mission_cfg=cfg,
        weapon_policy=weapon,
    )

    # Strictly monotonic in time (binary lookup precondition).
    assert np.all(np.diff(arr[:, 0]) > 0.0), "trajectory time not strictly increasing"

    def sample(t: float) -> tuple[float, float]:
        ts = arr[:, 0]
        if t <= ts[0]:
            return float(arr[0, 1]), float(arr[0, 2])
        if t >= ts[-1]:
            return float(arr[-1, 1]), float(arr[-1, 2])
        j = int(np.searchsorted(ts, t, side="left"))
        t0, lo0, la0, _ = arr[j - 1]
        t1, lo1, la1, _ = arr[j]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return float(lo0 + f * (lo1 - lo0)), float(la0 + f * (la1 - la0))

    # Starts parked at SBPV (the bug: it used to start at the CAP).
    assert arr[0, 0] == pytest.approx(0.0)
    assert sample(0.0) == pytest.approx((base.lon, base.lat), abs=1e-4)
    # SBPV→FARP transit arrives at the FARP at the reposition instant.
    assert sample(t_farp_arr) == pytest.approx((farp.lon, farp.lat), abs=1e-4)
    # Holds on the CAP through the whole PEA.
    assert sample(t_pea + 1_000.0) == pytest.approx((cap.lon, cap.lat), abs=1e-4)
    # Recovers to the FARP, then to SBPV — does NOT remain at the CAP.
    assert sample(t_farp2) == pytest.approx((farp.lon, farp.lat), abs=1e-4)
    assert arr[-1, 0] == pytest.approx(t_complete)
    assert sample(t_complete) == pytest.approx((base.lon, base.lat), abs=1e-4)
    # Mid-recovery is en route (neither parked at the CAP nor yet at the FARP).
    mid = sample((t_rtb + t_farp2) / 2.0)
    assert math.hypot(mid[0] - cap.lon, mid[1] - cap.lat) > 1e-3
    assert math.hypot(mid[0] - farp.lon, mid[1] - farp.lat) > 1e-3


# ─── T5 ──────────────────────────────────────────────────────────────────────


def test_missile_flyout_reaches_target_at_impact() -> None:
    """Missile event for a neutralized engagement satisfies:
    t_impact_s ≤ t_kill_s AND distance(linear_interp(shooter→target, t_kill),
    target_at_kill) < 100 m.

    Positions in :class:`~cuas_sim.viz.missile_flyout.MissileEvent` are
    ``(lon_deg, lat_deg)`` — the animator's plotting frame (Phase 3 unit
    reconciliation; see the module docstring). The 100 m oracle is therefore
    evaluated in projected meters via :class:`LocalProjection`.
    """
    import numpy as np
    import pandas as pd

    from cuas_sim.config import MissionConfig
    from cuas_sim.geometry.coordinates import GeoPoint
    from cuas_sim.geometry.projection import XY, LocalProjection
    from cuas_sim.viz.missile_flyout import missile_events_from_engagements

    cfg = MissionConfig()
    proj = LocalProjection(cfg.aoo_centroid_lat, cfg.aoo_centroid_lon)

    # UAS heading East at 50 m/s from a known spawn point.
    uas_speed = 50.0
    uas_heading = 0.0
    t_spawn = 1_000.0
    spawn_xy = XY(0.0, 0.0)
    spawn_geo = proj.unproject(spawn_xy)

    # A-29 shooter ~1.5 km west of the UAS spawn; stationary over the window so
    # the cache lookup is unambiguous. Its position does not drive the oracle
    # (the interp clamps to target_xy at t_kill ≥ t_impact) but a realistic
    # offset keeps the fixture honest.
    shooter_geo = proj.unproject(XY(-1_500.0, 0.0))
    a29_id = "A-29-1"
    cache: dict[str, dict[str, np.ndarray]] = {
        "baseline": {
            a29_id: np.array(
                [
                    [t_spawn, shooter_geo.lon, shooter_geo.lat, 0.0],
                    [t_spawn + 500.0, shooter_geo.lon, shooter_geo.lat, 0.0],
                ],
                dtype=np.float64,
            )
        }
    }

    v_munition = 800.0
    firing_distance = 1_500.0
    t_fire = 1_100.0
    tof = firing_distance / v_munition       # 1.875 s
    t_kill = t_fire + 2.0                     # munition arrives just before kill

    spawns_df = pd.DataFrame([{
        "uas_id": 7, "spawn_time_s": t_spawn,
        "lat": spawn_geo.lat, "lon": spawn_geo.lon,
        "heading_rad": uas_heading, "speed_mps": uas_speed,
        "archetype_name": "fixed-wing", "sector_index": 22,
    }])
    engagements_df = pd.DataFrame([{
        "config_name": "baseline", "a29_id": a29_id, "uas_id": 7,
        "t_fire_s": t_fire, "t_kill_s": t_kill,
        "firing_distance_m": firing_distance,
        "is_neutralized": True, "n_shots": 1, "feasible": True,
    }])

    events = missile_events_from_engagements(
        engagements_df=engagements_df,
        a29_trajectory_cache=cache,
        spawns_df=spawns_df,
        v_munition_mps=v_munition,
    )
    assert len(events) == 1, f"single-shot engagement → one event, got {len(events)}"
    ev = events[0]
    assert ev.shot_index == 1
    assert ev.is_neutralized is True
    assert ev.t_impact_s <= t_kill + 1e-9, (
        f"impact {ev.t_impact_s} must be ≤ kill {t_kill}"
    )
    # Neutralised: the killing missile impacts exactly on t_kill (even spacing
    # ends at the kill); its fire time sits one time-of-flight earlier.
    assert ev.t_impact_s == pytest.approx(t_kill, abs=1e-6)
    assert ev.t_fire_s == pytest.approx(t_kill - tof, abs=1e-6)

    # linear_interp(shooter → target, t_kill_s) with progress clamped to [0, 1].
    span = ev.t_impact_s - ev.t_fire_s
    frac = 1.0 if span <= 0.0 else (t_kill - ev.t_fire_s) / span
    frac = max(0.0, min(1.0, frac))
    interp_lon = ev.shooter_xy[0] + frac * (ev.target_xy[0] - ev.shooter_xy[0])
    interp_lat = ev.shooter_xy[1] + frac * (ev.target_xy[1] - ev.shooter_xy[1])

    # target_at_kill — UAS projected forward to t_kill by constant velocity.
    dt_kill = t_kill - t_spawn
    tgt_xy_kill = XY(
        spawn_xy.x + uas_speed * math.cos(uas_heading) * dt_kill,
        spawn_xy.y + uas_speed * math.sin(uas_heading) * dt_kill,
    )
    interp_xy = proj.project(GeoPoint(lat=interp_lat, lon=interp_lon))
    dist_m = math.hypot(interp_xy.x - tgt_xy_kill.x, interp_xy.y - tgt_xy_kill.y)
    assert dist_m < 100.0, (
        f"interp at t_kill is {dist_m:.1f} m from UAS@t_kill (>100 m)"
    )
