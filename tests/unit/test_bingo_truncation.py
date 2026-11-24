from __future__ import annotations

"""Bingo-fuel truncation inside ``EngagementController.run`` (FR-8, FR-9, FR-12, TD-1).

Group B (Red phase): must FAIL until C-7 adds the ``fuel_exhaust_t`` / ``farp_xy``
inputs to ``run()`` and the ``bingo_aborted`` / ``bingo_xy`` / ``bingo_t_s`` fields
to ``EngagementOutcome``.
"""

import math

import numpy as np

from cuas_sim.config import MissionConfig, UASArchetypeSpec
from cuas_sim.engagement.controller import EngagementController
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.policies.tracking import make_tracking_policy
from cuas_sim.policies.weapon import make_weapon_policy

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


def _uas(spawn_xy: XY, proj: LocalProjection):
    from cuas_sim.agents.uas import UAS

    return UAS(
        uas_id=1,
        spawn_time_s=0.0,
        spawn_point=proj.unproject(spawn_xy),
        spawn_xy=spawn_xy,
        archetype=_MR,
        heading_rad=math.pi,  # flying −x, head-on toward an A-29 at the origin
        sector_index=39,
    )


def _run(cfg, proj, uas, *, fuel_exhaust_t, farp_xy):
    return _controller(cfg, proj).run(
        a29_id="A-29-T", aor_name="AoR-1", uas=uas,
        a29_xy_at_notify=XY(0.0, 0.0), t_notify_s=0.0,
        rng_tracking=np.random.default_rng(0),
        rng_firing=np.random.default_rng(0),
        rng_p_hit=np.random.default_rng(0),
        rng_reengage=np.random.default_rng(0),
        fuel_exhaust_t=fuel_exhaust_t,
        farp_xy=farp_xy,
    )


def test_bingo_crossing_during_pursuit() -> None:
    """The FARP sits at the A-29's start; chasing a UAS at +x grows the distance
    to the FARP while the fuel clock runs down. The crossing truncates the
    engagement before envelope entry (~400 s away)."""
    cfg = MissionConfig()
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat, lon0_deg=cfg.aoo_centroid_lon)
    uas = _uas(XY(60_000.0, 0.0), proj)

    out = _run(cfg, proj, uas, fuel_exhaust_t=200.0, farp_xy=XY(0.0, 0.0))

    assert out.bingo_aborted is True
    assert out.bingo_xy is not None
    # remaining(t)=200−t, t_to_farp≈t ⇒ crossing near t≈100 s, well before the
    # ~400 s envelope entry. The kill never happens (truncated).
    assert 0.0 < out.bingo_t_s < 400.0
    assert not out.is_neutralized


def test_no_bingo_when_fuel_abundant() -> None:
    """With effectively infinite endurance the engagement proceeds normally —
    no truncation, at least one shot."""
    cfg = MissionConfig()
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat, lon0_deg=cfg.aoo_centroid_lon)
    uas = _uas(XY(60_000.0, 0.0), proj)

    out = _run(cfg, proj, uas, fuel_exhaust_t=float("inf"), farp_xy=XY(0.0, 0.0))

    assert out.bingo_aborted is False
    assert out.feasible
    assert out.n_shots >= 1
