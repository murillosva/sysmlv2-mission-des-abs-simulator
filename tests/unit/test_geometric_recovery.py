from __future__ import annotations

# Note: Spec 018 parametrizes this test on uas_spawn_window_enabled in [True, False]
"""Geometric recovery phases 5/6 + zero-default refuel (FR-15, FR-16, FR-17).

Group B (Red phase): must FAIL until C-5 adds ``geometric_recovery_phases`` /
``farp_refuel_time_s`` and C-9 rewrites the post-PEA flow to use ``t_to_farp``.
"""

import pytest
import simpy

from cuas_sim.agents.a29 import A29
from cuas_sim.config import A29Schedule, MissionConfig
from cuas_sim.geometry.projection import LocalProjection
from cuas_sim.metrics.records import RunLogger
from cuas_sim.policies.tracking import make_tracking_policy
from cuas_sim.policies.weapon import make_weapon_policy
from cuas_sim.rng import RNGRegistry
from cuas_sim.states import A29State
from cuas_sim.world.aor import AoRMap

_A29_ID = "A-29-1"


@pytest.mark.parametrize("uas_spawn_window_enabled", [True, False])
def test_refuel_default_zero(uas_spawn_window_enabled: bool) -> None:
    # Spec 018 parametrization: phases 5/6 geometry must be bit-identical under
    # both modes; only the terminal state name changes (DEPARTING_CAP ⇄ RTB_FUEL_LIMIT).
    """Normal end-of-PEA recovery is geometric: CAP→FARP + FARP→SBPV at cruise
    speed, with no intermediate refuel delay (``farp_refuel_time_s = 0``)."""
    from cuas_sim.geometry.projection import t_to_farp

    cfg = MissionConfig(
        skip_pre_pea=False, t_pea_start_s=0.0,
        farp_refuel_time_s=0.0, geometric_recovery_phases=True,
        a29_endurance_s=1.0e9,  # huge: never bingo during this idle run
        uas_spawn_window_enabled=uas_spawn_window_enabled,
    )
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat, lon0_deg=cfg.aoo_centroid_lon)
    aor_map = AoRMap.from_config(cfg)
    log = RunLogger()
    sched = A29Schedule(
        farp_name=cfg.a29_schedules[_A29_ID].farp_name, cap_index=1,
        phase1_sbpv_to_farp_s=1.0, phase2_farp_to_cap_s=1.0,
        phase3_holding_s=1.0, phase4_pea_s=5.0,
        # legacy fixed durations — must be IGNORED under geometric recovery.
        phase5_cap_to_farp_s=99_999.0, phase6_farp_to_sbpv_s=99_999.0,
    )
    a29 = A29(
        a29_id=_A29_ID, schedule=sched, aor_name=aor_map.aor_of_a29(_A29_ID),
        cfg=cfg, aor_map=aor_map, projection=proj,
        tracking=make_tracking_policy(cfg, "modernized"),
        weapon=make_weapon_policy(cfg, "modernized"),
        rng=RNGRegistry.from_master_seed(1).per_config("modernized"),
        run_logger=log, config_name="modernized",
    )
    env = simpy.Environment()
    inbox: simpy.Store = simpy.Store(env)  # empty: PEA simply times out
    env.process(a29.run(env, inbox))
    env.run()

    evs = log.events["modernized"]
    # Spec 018 FR-7/FR-9: terminal name flips to RTB_FUEL_LIMIT under flag=True.
    terminal = (
        A29State.RTB_FUEL_LIMIT.value
        if uas_spawn_window_enabled
        else A29State.DEPARTING_CAP.value
    )
    t_depart = next(e.t_s for e in evs if e.new == terminal)
    t_complete = next(e.t_s for e in evs if e.new == A29State.MISSION_COMPLETE.value)

    v = cfg.a29_cruise_speed_mps
    cap_xy = proj.project(cfg.caps["CAP-1"])
    farp_xy = proj.project(cfg.farps[sched.farp_name])
    sbpv_xy = proj.project(cfg.main_base)
    expected = t_to_farp(cap_xy, farp_xy, v) + t_to_farp(farp_xy, sbpv_xy, v)

    # No refuel delay → recovery time equals pure geometry, far from the 199998 s
    # legacy fixed durations.
    assert abs((t_complete - t_depart) - expected) < 1e-3
