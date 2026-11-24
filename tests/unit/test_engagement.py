"""Unit tests for EngagementController — Spec 015 multi-shot re-engagement.

Supersedes the Spec 014 single-shot t_fire tests (multi-shot model).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import simpy

from cuas_sim.agents.a29 import A29
from cuas_sim.agents.uas import UAS
from cuas_sim.config import A29Schedule, MissionConfig, UASArchetypeSpec
from cuas_sim.engagement.controller import EngagementController
from cuas_sim.geometry.coordinates import GeoPoint
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.metrics.records import RunLogger
from cuas_sim.policies.tracking import make_tracking_policy
from cuas_sim.policies.weapon import _BaseWeaponPolicy, make_weapon_policy
from cuas_sim.rng import RNGRegistry
from cuas_sim.states import A29State
from cuas_sim.world.aor import AoRMap
from cuas_sim.world.arrivals import UASBirth

_MR = UASArchetypeSpec(
    name="multi-rotor", max_speed_mps=29.0, dimension_m=1.0, ceiling_m=6000.0,
)


def _controller(cfg: MissionConfig, proj: LocalProjection) -> EngagementController:
    return EngagementController(
        cfg=cfg,
        tracking_policy=make_tracking_policy(cfg, "modernized"),
        weapon_policy=make_weapon_policy(cfg, "modernized"),
        projection=proj,
        config_name="modernized",
    )


def _uas(spawn_xy: XY, proj: LocalProjection) -> UAS:
    return UAS(
        uas_id=1,
        spawn_time_s=0.0,
        spawn_point=proj.unproject(spawn_xy),
        spawn_xy=spawn_xy,
        archetype=_MR,
        heading_rad=math.pi,  # flying −x, toward an A-29 at the origin
        sector_index=39,
    )


def _run(cfg: MissionConfig, uas: UAS, proj: LocalProjection):
    return _controller(cfg, proj).run(
        a29_id="A-29-T", aor_name="AoR-1", uas=uas,
        a29_xy_at_notify=XY(0.0, 0.0), t_notify_s=0.0,
        rng_tracking=np.random.default_rng(0),
        rng_firing=np.random.default_rng(0),
        rng_p_hit=np.random.default_rng(0),
        rng_reengage=np.random.default_rng(0),
    )


def test_short_philosophy_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MissionConfig()
    # Force the SHORT philosophy (idx 0). Spec 017 §5.4 — draw returns
    # ``(distance, philosophy_idx)`` and short = single shot keys off idx 0.
    monkeypatch.setattr(_BaseWeaponPolicy, "draw_firing_distance",
                        lambda self, rng: (2_000.0, 0))
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat,
                           lon0_deg=cfg.aoo_centroid_lon)
    out = _run(cfg, _uas(XY(60_000.0, 0.0), proj), proj)
    assert out.feasible
    assert out.n_shots == 1  # short ⇒ exactly one shot, no re-engagement


def test_medium_long_reengages(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MissionConfig(reengagement_triangular_s=(10.0, 10.0, 10.0))
    # Force the LONG philosophy (idx 2). Spec 017 §5.4 — tuple return.
    monkeypatch.setattr(_BaseWeaponPolicy, "draw_firing_distance",
                        lambda self, rng: (14_000.0, 2))
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat,
                           lon0_deg=cfg.aoo_centroid_lon)
    out = _run(cfg, _uas(XY(60_000.0, 0.0), proj), proj)
    assert out.feasible
    assert out.n_shots >= 2  # long ⇒ re-engages while closing
    assert out.n_shots <= cfg.rocket_budget_per_sortie
    assert out.rockets_remaining == cfg.rocket_budget_per_sortie - out.n_shots


# ─── C-4: A-29 agent — budget/Winchester, Lever-2 return, pending pool ────────

_A29_ID = "A-29-1"


def _make_a29(
    cfg: MissionConfig,
    *,
    config_name: str = "modernized",
    seed: int = 42,
) -> tuple[A29, LocalProjection, RunLogger]:
    """Build a single, fully-wired A-29 for a focused SimPy drive."""
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat, lon0_deg=cfg.aoo_centroid_lon)
    aor_map = AoRMap.from_config(cfg)
    log = RunLogger()
    a29 = A29(
        a29_id=_A29_ID,
        schedule=cfg.a29_schedules[_A29_ID],
        aor_name=aor_map.aor_of_a29(_A29_ID),
        cfg=cfg,
        aor_map=aor_map,
        projection=proj,
        tracking=make_tracking_policy(cfg, config_name),
        weapon=make_weapon_policy(cfg, config_name),
        rng=RNGRegistry.from_master_seed(seed).per_config(config_name),
        run_logger=log,
        config_name=config_name,
    )
    return a29, proj, log


def _cap_xy_of(a29: A29, proj: LocalProjection) -> XY:
    cap_geo = a29.cfg.caps[f"CAP-{a29.schedule.cap_index}"]
    return proj.project(cap_geo)


def _birth_near_cap(
    a29: A29, proj: LocalProjection, *, uas_id: int, dx_m: float,
    archetype: UASArchetypeSpec, spawn_time_s: float = 0.0,
) -> UASBirth:
    """A UAS spawned ``dx_m`` east of the A-29's CAP, flying −x toward it."""
    cap = _cap_xy_of(a29, proj)
    spawn_xy = XY(cap.x + dx_m, cap.y)
    pt = proj.unproject(spawn_xy)
    return UASBirth(
        uas_id=uas_id, spawn_time_s=spawn_time_s, sector_index=39,
        point=GeoPoint(lat=pt.lat, lon=pt.lon, name=f"UAS-{uas_id}"),
        heading_rad=math.pi, archetype=archetype, omega=0.0,
    )


def _return_leg_seconds(log: RunLogger, config_name: str) -> float:
    """Observed RETURNING_TO_CAP → AT_CAP_PEA duration from the event log."""
    evs = log.events[config_name]
    t_ret = next(e.t_s for e in evs if e.new == A29State.RETURNING_TO_CAP.value)
    t_cap = next(e.t_s for e in evs
                 if e.t_s >= t_ret and e.new == A29State.AT_CAP_PEA.value)
    return t_cap - t_ret


def test_winchester_after_budget_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MissionConfig(
        skip_pre_pea=True, t_pea_start_s=0.0,
        rocket_budget_per_sortie=3, reengagement_triangular_s=(10.0, 10.0, 10.0),
    )
    # Force LONG philosophy + P_hit=0 so the planned sequence is spent in full
    # (capped at the 3-rocket budget) without an early Bernoulli stop.
    monkeypatch.setattr(_BaseWeaponPolicy, "draw_firing_distance",
                        lambda self, rng: (14_000.0, 2))
    monkeypatch.setattr(_BaseWeaponPolicy, "p_hit_at",
                        lambda self, d, arch: 0.0)

    a29, proj, log = _make_a29(cfg)
    env = simpy.Environment()
    inbox: simpy.Store = simpy.Store(env)
    b1 = _birth_near_cap(a29, proj, uas_id=1, dx_m=60_000.0, archetype=cfg.uas_mr)
    b2 = _birth_near_cap(a29, proj, uas_id=2, dx_m=60_000.0, archetype=cfg.uas_mr)
    inbox.put(b1)
    inbox.put(b2)
    env.process(a29.run(env, inbox))
    env.run()

    # Budget fully spent on the first engagement (3 rockets), none left.
    assert a29.rockets_left == 0
    eng_uas = [e.fields_["uas_id"] for e in log.engagements
               if e.fields_["config_name"] == "modernized"]
    assert eng_uas == [1]                       # only the 1st was engaged
    winch = [lk for lk in log.leaks if lk["reason"] == "winchester"]
    assert len(winch) == 1 and winch[0]["uas_id"] == 2  # 2nd leaks Winchester


def test_lever2_return_from_engagement_end() -> None:
    cfg = MissionConfig(skip_pre_pea=True, t_pea_start_s=0.0)
    a29, proj, log = _make_a29(cfg, seed=7)
    env = simpy.Environment()
    inbox: simpy.Store = simpy.Store(env)
    birth = _birth_near_cap(a29, proj, uas_id=1, dx_m=60_000.0,
                            archetype=cfg.uas_mr)
    inbox.put(birth)
    env.process(a29.run(env, inbox))
    env.run()

    # Recompute the expected engagement end point with an identically-seeded
    # standalone controller (single engagement ⇒ identical RNG draw order).
    rng = RNGRegistry.from_master_seed(7).per_config("modernized")
    cap_xy = _cap_xy_of(a29, proj)
    uas = UAS.from_birth(birth, proj)
    expected = EngagementController(
        cfg=cfg, tracking_policy=make_tracking_policy(cfg, "modernized"),
        weapon_policy=make_weapon_policy(cfg, "modernized"),
        projection=proj, config_name="modernized",
    ).run(
        a29_id=_A29_ID, aor_name=a29.aor_name, uas=uas,
        a29_xy_at_notify=cap_xy, t_notify_s=0.0,
        rng_tracking=rng.tracking_updates, rng_firing=rng.firing_phil,
        rng_p_hit=rng.p_hit_outcome, rng_reengage=rng.reengagement_interval,
        rockets_available=cfg.rocket_budget_per_sortie,
    )
    assert expected.feasible and expected.end_xy is not None
    exp_return = math.hypot(
        expected.end_xy.x - cap_xy.x, expected.end_xy.y - cap_xy.y
    ) / cfg.a29_cruise_speed_mps
    assert exp_return > 0.0
    assert abs(_return_leg_seconds(log, "modernized") - exp_return) < 1e-6


def test_pending_pool_claim_nearest_and_prune_evaded() -> None:
    from cuas_sim.agents.e99m_dispatcher import PendingPool

    cfg = MissionConfig(skip_pre_pea=True, t_pea_start_s=0.0)
    a29, proj, log = _make_a29(cfg)
    cap_xy = _cap_xy_of(a29, proj)

    # Slow multi-rotors still approaching the CAP at now_s (id 2 is nearer).
    far = _birth_near_cap(a29, proj, uas_id=1, dx_m=90_000.0, archetype=cfg.uas_mr)
    near = _birth_near_cap(a29, proj, uas_id=2, dx_m=50_000.0, archetype=cfg.uas_mr)
    # A fixed-wing that has already flown past its 80 nmi ZIDA depth by now_s.
    evaded = _birth_near_cap(a29, proj, uas_id=3, dx_m=0.0, archetype=cfg.uas_fw)

    pool = PendingPool(config_name="modernized", projection=proj, run_logger=log)
    for b in (far, near, evaded):
        pool.add(b, t_added_s=0.0)

    now_s = 1_300.0  # FW (130 m/s) → ~169 km > 80 nmi (148 km) from spawn
    first = pool.claim(cap_xy=cap_xy, now_s=now_s,
                       a29_speed_mps=cfg.a29_cruise_speed_mps,
                       envelope_max_m=a29.weapon.envelope_max_m)
    assert first is not None and first.uas_id == 2          # T8: nearest feasible
    pe = [lk for lk in log.leaks if lk["reason"] == "pending-evaded"]
    assert len(pe) == 1 and pe[0]["uas_id"] == 3            # T9: evaded pruned

    second = pool.claim(cap_xy=cap_xy, now_s=now_s + 1.0,
                        a29_speed_mps=cfg.a29_cruise_speed_mps,
                        envelope_max_m=a29.weapon.envelope_max_m)
    assert second is not None and second.uas_id == 1        # next nearest
    assert pool.claim(cap_xy=cap_xy, now_s=now_s + 2.0,
                      a29_speed_mps=cfg.a29_cruise_speed_mps,
                      envelope_max_m=a29.weapon.envelope_max_m) is None


def test_a29_consults_pending_pool_on_free() -> None:
    from cuas_sim.agents.e99m_dispatcher import PendingPool

    cfg = MissionConfig(skip_pre_pea=True, t_pea_start_s=0.0)
    a29, proj, log = _make_a29(cfg)
    pool = PendingPool(config_name="modernized", projection=proj, run_logger=log)
    claimable = _birth_near_cap(a29, proj, uas_id=10, dx_m=40_000.0,
                                archetype=cfg.uas_mr)
    pool.add(claimable, t_added_s=0.0)
    a29.pending_pool = pool

    env = simpy.Environment()
    inbox: simpy.Store = simpy.Store(env)  # empty → only the pool feeds work
    env.process(a29.run(env, inbox))
    env.run()

    eng_uas = [e.fields_["uas_id"] for e in log.engagements
               if e.fields_["config_name"] == "modernized"]
    assert 10 in eng_uas                  # claimed from the pool and engaged
    assert pool.claim(cap_xy=_cap_xy_of(a29, proj), now_s=0.0,
                      a29_speed_mps=cfg.a29_cruise_speed_mps,
                      envelope_max_m=a29.weapon.envelope_max_m) is None


# ─── C-5: E99M dispatcher — global nearest-free + pending pool ────────────────

def _make_fleet(
    cfg: MissionConfig, *, config_name: str = "modernized", seed: int = 42,
) -> tuple[dict[str, A29], dict[str, tuple[float, float]],
           LocalProjection, AoRMap, RunLogger]:
    """All six A-29s of one configuration (not started — for routing tests)."""
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat, lon0_deg=cfg.aoo_centroid_lon)
    aor_map = AoRMap.from_config(cfg)
    log = RunLogger()
    reg = RNGRegistry.from_master_seed(seed)
    agents: dict[str, A29] = {
        aid: A29(
            a29_id=aid, schedule=sched, aor_name=aor_map.aor_of_a29(aid),
            cfg=cfg, aor_map=aor_map, projection=proj,
            tracking=make_tracking_policy(cfg, config_name),
            weapon=make_weapon_policy(cfg, config_name),
            rng=reg.per_config(config_name), run_logger=log,
            config_name=config_name, state=A29State.AT_CAP_PEA,
        )
        for aid, sched in cfg.a29_schedules.items()
    }
    cap_xy = {
        aid: (lambda p: (p.x, p.y))(proj.project(cfg.caps[f"CAP-{s.cap_index}"]))
        for aid, s in cfg.a29_schedules.items()
    }
    return agents, cap_xy, proj, aor_map, log


def _dispatch_once(
    cfg: MissionConfig, agents: dict[str, A29],
    cap_xy: dict[str, tuple[float, float]], proj: LocalProjection,
    aor_map: AoRMap, birth: UASBirth, *, pool: object | None = None,
) -> list[str]:
    """Run the dispatcher for a single birth; return the routed A-29 id(s)."""
    from cuas_sim.agents.e99m_dispatcher import E99MDispatcher

    env = simpy.Environment()
    inbox: simpy.Store = simpy.Store(env)
    outboxes = {aid: simpy.Store(env) for aid in agents}
    disp = E99MDispatcher(
        config_name="modernized", aor_map=aor_map, a29_agents=agents,
        cap_xy_map=cap_xy, projection=proj, pending_pool=pool,
    )
    inbox.put(birth)
    env.process(disp.run(env, inbox, outboxes))
    env.run(until=1.0)
    return [aid for aid, s in outboxes.items() if s.items]


def _birth_at_cap(
    proj: LocalProjection, cfg: MissionConfig,
    cap_xy: tuple[float, float], *, uas_id: int, sector_index: int = 1,
) -> UASBirth:
    pt = proj.unproject(XY(cap_xy[0], cap_xy[1]))
    return UASBirth(
        uas_id=uas_id, spawn_time_s=0.0, sector_index=sector_index,
        point=GeoPoint(lat=pt.lat, lon=pt.lon, name=f"UAS-{uas_id}"),
        heading_rad=math.pi, archetype=cfg.uas_mr, omega=0.0,
    )


def test_dispatch_global_nearest_free_not_primary_restricted() -> None:
    cfg = MissionConfig()
    agents, cap_xy, proj, aor_map, _ = _make_fleet(cfg)
    # Sector 1 → primary A-29-1 (AoR-1); AoR-2 (A-29-2) is its only adjacent.
    birth = _birth_at_cap(proj, cfg, cap_xy["A-29-1"], uas_id=1, sector_index=1)
    # Busy out the primary + its adjacent: the old primary/adjacent rule would
    # drop here; the global rule must still route to a free, non-adjacent A-29.
    agents["A-29-1"].state = A29State.ENGAGING
    agents["A-29-2"].state = A29State.ENGAGING
    free = [aid for aid in agents if agents[aid].is_free]
    expected = min(free, key=lambda a: math.hypot(
        cap_xy[a][0] - cap_xy["A-29-1"][0], cap_xy[a][1] - cap_xy["A-29-1"][1]))
    routed = _dispatch_once(cfg, agents, cap_xy, proj, aor_map, birth)
    assert routed == [expected]
    assert routed[0] not in ("A-29-1", "A-29-2")  # primary busy didn't block


def test_dispatch_tiebreak_lowest_id() -> None:
    cfg = MissionConfig()
    agents, _, proj, aor_map, _ = _make_fleet(cfg)
    # Synthetic CAPs: A-29-2 and A-29-5 are exactly equidistant (origin); the
    # rest are far. A tie must resolve to the lower numeric id (Article IV).
    cap_xy = {"A-29-1": (1e6, 0.0), "A-29-2": (0.0, 0.0), "A-29-3": (1e6, 0.0),
              "A-29-4": (1e6, 0.0), "A-29-5": (0.0, 0.0), "A-29-6": (1e6, 0.0)}
    birth = _birth_at_cap(proj, cfg, (0.0, 0.0), uas_id=2, sector_index=20)
    routed = _dispatch_once(cfg, agents, cap_xy, proj, aor_map, birth)
    assert routed == ["A-29-2"]


def test_dispatcher_parks_in_pool_when_all_busy() -> None:
    from cuas_sim.agents.e99m_dispatcher import PendingPool

    cfg = MissionConfig()
    agents, cap_xy, proj, aor_map, log = _make_fleet(cfg)
    for a in agents.values():
        a.state = A29State.ENGAGING            # every A-29 busy
    pool = PendingPool(config_name="modernized", projection=proj, run_logger=log)
    birth = _birth_at_cap(proj, cfg, cap_xy["A-29-3"], uas_id=7, sector_index=9)
    routed = _dispatch_once(cfg, agents, cap_xy, proj, aor_map, birth, pool=pool)
    assert routed == []                        # not forwarded to any A-29
    assert len(pool) == 1                      # parked, not dropped


def test_a29_full_lifecycle_and_infeasible_engagement() -> None:
    # Exercises the pre-/post-PEA symbolic phases (skip_pre_pea=False), an
    # infeasible engagement (no pursuit/return leg, leak recorded), and the
    # defensive pool-None / same-state guards (AC-9 coverage of a29.py).
    cfg = MissionConfig(skip_pre_pea=False, t_pea_start_s=0.0,
                        a29_cruise_speed_mps=50.0)
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat, lon0_deg=cfg.aoo_centroid_lon)
    aor_map = AoRMap.from_config(cfg)
    log = RunLogger()
    sched = A29Schedule(
        farp_name=cfg.a29_schedules[_A29_ID].farp_name, cap_index=1,
        phase1_sbpv_to_farp_s=1.0, phase2_farp_to_cap_s=1.0,
        phase3_holding_s=1.0, phase4_pea_s=5.0,
        phase5_cap_to_farp_s=1.0, phase6_farp_to_sbpv_s=1.0,
    )
    a29 = A29(
        a29_id=_A29_ID, schedule=sched, aor_name=aor_map.aor_of_a29(_A29_ID),
        cfg=cfg, aor_map=aor_map, projection=proj,
        tracking=make_tracking_policy(cfg, "modernized"),
        weapon=make_weapon_policy(cfg, "modernized"),
        rng=RNGRegistry.from_master_seed(1).per_config("modernized"),
        run_logger=log, config_name="modernized",
    )
    # Fast fixed-wing flying directly away from the CAP: the slowed (50 m/s)
    # A-29 cannot catch a 130 m/s receding target → pure-pursuit loop (Spec
    # 016 FR-8) detects N=3 consecutive non-decreasing separations and returns
    # feasible=False. Spawn outside the modernized envelope_max (14 km) so the
    # loop actually iterates instead of jumping to ENGAGING on step 1.
    cap = a29._cap_xy()
    pt = proj.unproject(XY(cap.x + 16_000.0, cap.y))
    birth = UASBirth(
        uas_id=1, spawn_time_s=0.0, sector_index=1,
        point=GeoPoint(lat=pt.lat, lon=pt.lon, name="UAS-1"),
        heading_rad=0.0, archetype=cfg.uas_fw, omega=0.0,  # +x, receding
    )
    env = simpy.Environment()
    inbox: simpy.Store = simpy.Store(env)
    inbox.put(birth)
    env.process(a29.run(env, inbox))
    env.run()

    seen = {e.new for e in log.events["modernized"]}
    assert A29State.TRANSIT_TO_FARP.value in seen          # pre-PEA ran
    assert A29State.RECOVERING_TO_BASE.value in seen        # post-PEA ran
    assert A29State.MISSION_COMPLETE.value in seen
    assert any(not e.fields_["feasible"] for e in log.engagements)
    assert A29State.RETURNING_TO_CAP.value not in seen      # no return leg

    # Defensive guards: pool-None drain is a no-op generator; setting the
    # current state again is a no-op (no event logged).
    assert list(a29._drain_pending_pool(env, 0.0)) == []
    before = len(log.events["modernized"])
    a29._set_state(env, a29.state)
    assert len(log.events["modernized"]) == before
