from __future__ import annotations

# Note: Spec 018 parametrizes this test on uas_spawn_window_enabled in [True, False]
"""End-to-end bingo-fuel leak + RTB_BINGO flow with real SimPy (T11, Article VIII).

Group B (Red phase): must FAIL until C-6..C-9 land the fuel clock, the
truncation, the ``bingo-fuel`` leak and the ``RTB_BINGO`` state.

A single A-29 with a short endurance is handed a UAS far downrange. Chasing it
drives the A-29 away from its (nearby) FARP; the fuel clock crosses the
distance-to-FARP mid-pursuit, so the engagement is aborted and — with no other
A-29 and no pending pool — the UAS leaks with ``reason="bingo-fuel"``.
"""

import math

import pytest
import simpy

from cuas_sim.agents.a29 import A29
from cuas_sim.config import A29Schedule, MissionConfig
from cuas_sim.geometry.coordinates import GeoPoint
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.metrics.records import RunLogger
from cuas_sim.policies.tracking import make_tracking_policy
from cuas_sim.policies.weapon import make_weapon_policy
from cuas_sim.rng import RNGRegistry
from cuas_sim.states import A29State
from cuas_sim.world.aor import AoRMap

_A29_ID = "A-29-1"


@pytest.mark.parametrize("uas_spawn_window_enabled", [True, False])
def test_bingo_fuel_leak_when_no_free_a29(uas_spawn_window_enabled: bool) -> None:
    # Spec 018 parametrization: under flag=True, the terminal state is
    # ``RTB_FUEL_LIMIT`` instead of ``RTB_BINGO`` (FR-7/FR-9).
    base = MissionConfig(uas_spawn_window_enabled=uas_spawn_window_enabled)
    proj = LocalProjection(lat0_deg=base.aoo_centroid_lat, lon0_deg=base.aoo_centroid_lon)

    # CAP-1 at the projection origin; designated FARP 5 km north (≈35 s away).
    cap_geo = proj.unproject(XY(0.0, 0.0))
    farp_geo = proj.unproject(XY(0.0, 5_000.0))
    caps = {**base.caps, "CAP-1": cap_geo}
    farps = {**base.farps, "F": farp_geo}
    sched = A29Schedule(
        farp_name="F", cap_index=1,
        phase1_sbpv_to_farp_s=0.0, phase2_farp_to_cap_s=0.0,
        phase3_holding_s=0.0, phase4_pea_s=300.0,
        phase5_cap_to_farp_s=0.0, phase6_farp_to_sbpv_s=0.0,
    )
    cfg = base.replaced(
        skip_pre_pea=True, t_pea_start_s=0.0,
        a29_endurance_s=80.0,                # short: margin burned during the chase
        caps=caps, farps=farps,
        a29_schedules={**base.a29_schedules, _A29_ID: sched},
        geometric_recovery_phases=True, farp_refuel_time_s=0.0,
    )

    aor_map = AoRMap.from_config(cfg)
    log = RunLogger()
    a29 = A29(
        a29_id=_A29_ID, schedule=sched, aor_name=aor_map.aor_of_a29(_A29_ID),
        cfg=cfg, aor_map=aor_map, projection=proj,
        tracking=make_tracking_policy(cfg, "modernized"),
        weapon=make_weapon_policy(cfg, "modernized"),
        rng=RNGRegistry.from_master_seed(1).per_config("modernized"),
        run_logger=log, config_name="modernized",
    )

    # UAS 60 km east of the CAP (away from the FARP), flying −x head-on.
    spawn_xy = XY(60_000.0, 0.0)
    pt = proj.unproject(spawn_xy)
    from cuas_sim.world.arrivals import UASBirth

    birth = UASBirth(
        uas_id=1, spawn_time_s=0.0, sector_index=39,
        point=GeoPoint(lat=pt.lat, lon=pt.lon, name="UAS-1"),
        heading_rad=math.pi, archetype=cfg.uas_mr, omega=0.0,
    )

    env = simpy.Environment()
    inbox: simpy.Store = simpy.Store(env)
    inbox.put(birth)
    env.process(a29.run(env, inbox))
    env.run()

    bingo_leaks = [lk for lk in log.leaks if lk["reason"] == "bingo-fuel"]
    assert len(bingo_leaks) == 1
    assert bingo_leaks[0]["uas_id"] == 1

    seen = {e.new for e in log.events["modernized"]}
    if uas_spawn_window_enabled:
        # Spec 018 FR-7: unified terminal state
        assert A29State.RTB_FUEL_LIMIT.value in seen
        assert A29State.RTB_BINGO.value not in seen
    else:
        # Legacy Spec 017 path
        assert A29State.RTB_BINGO.value in seen
    assert A29State.MISSION_COMPLETE.value in seen
