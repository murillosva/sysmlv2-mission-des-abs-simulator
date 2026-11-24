"""Known-answer tests for the MOE-1 <-> Eq. (3.5) reconciliation (Spec 025).

Group B (Red phase, Article III): these MUST FAIL until Group C lands the
``ZIDABoundary.crossing_time_s`` helper, the ``EngagementOutcome.t_ev_s`` field,
the controller ``t_ev`` computation and the ``_compute_moe1`` Eqs. 3.5/3.6 rewrite.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cuas_sim.config import MissionConfig, UASArchetypeSpec
from cuas_sim.engagement.controller import EngagementController
from cuas_sim.geometry.projection import XY, LocalProjection
from cuas_sim.metrics.aggregator import MetricsAggregator
from cuas_sim.policies.tracking import make_tracking_policy
from cuas_sim.policies.weapon import _BaseWeaponPolicy, make_weapon_policy
from cuas_sim.world.boundary import ZIDABoundary

_MR = UASArchetypeSpec(
    name="multi-rotor", max_speed_mps=29.0, dimension_m=1.0, ceiling_m=6000.0,
)


def _all_moe1(rows: list[dict]) -> float:
    return next(float(r["value"]) for r in rows
               if r["aor"] == "ALL" and r["metric"] == "MOE-1")


def _fixed_tpea(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    """Force ``_t_pea_for_aor`` to a constant so the formula is isolated."""
    monkeypatch.setattr(
        MetricsAggregator, "_t_pea_for_aor",
        lambda self, aor_name, config_name=None: value,
    )


def _moe1_for(df: pd.DataFrame) -> float:
    return _all_moe1(MetricsAggregator(cfg=MissionConfig())._compute_moe1(df, "baseline"))


# ─── T1 — neutralized single (orientation + complement) ─────────────────────


def test_neutralized_single(monkeypatch: pytest.MonkeyPatch) -> None:
    _fixed_tpea(monkeypatch, 3000.0)
    df = pd.DataFrame([{
        "aor_name": "AoR-1", "config_name": "baseline", "is_neutralized": True,
        "moe1_engagement_time_s": 300.0, "t_ev_s": float("nan"),
        "t_pursuit_start_s": 0.0,
    }])
    # 1 - 300/3000 = 0.90
    assert _moe1_for(df) == pytest.approx(0.90, abs=1e-9)


# ─── T2 — evaded engagement with penalty (1 + E/N) ──────────────────────────


def test_evaded_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    _fixed_tpea(monkeypatch, 3000.0)
    df = pd.DataFrame([
        {"aor_name": "AoR-1", "config_name": "baseline", "is_neutralized": True,
         "moe1_engagement_time_s": 300.0, "t_ev_s": float("nan"),
         "t_pursuit_start_s": 0.0},
        # Evaded: numerator MUST use t_ev (600), NOT moe1_engagement_time_s (999).
        {"aor_name": "AoR-1", "config_name": "baseline", "is_neutralized": False,
         "moe1_engagement_time_s": 999.0, "t_ev_s": 600.0,
         "t_pursuit_start_s": 0.0},
    ])
    # N=2, E=1, pen=1.5; q_A=300/3000=0.1; q_B=(600*1.5)/3000=0.3
    # MOE-1 = 1 - mean(0.1, 0.3) = 0.80
    assert _moe1_for(df) == pytest.approx(0.80, abs=1e-9)


# ─── T4 — orientation reversed (faster engagement ⇒ higher MOE-1) ───────────


def test_orientation_reversed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fixed_tpea(monkeypatch, 3000.0)

    def moe1(t_ef: float) -> float:
        df = pd.DataFrame([{
            "aor_name": "AoR-1", "config_name": "baseline", "is_neutralized": True,
            "moe1_engagement_time_s": t_ef, "t_ev_s": float("nan"),
            "t_pursuit_start_s": 0.0,
        }])
        return _moe1_for(df)

    fast, slow = moe1(300.0), moe1(600.0)
    assert fast == pytest.approx(0.90, abs=1e-9)   # complement applied
    assert fast > slow                              # reversed vs utilisation


# ─── T3 (helper) — 80 nmi crossing time ─────────────────────────────────────


def test_crossing_time_80nmi() -> None:
    proj = LocalProjection(lat0_deg=-5.0, lon0_deg=-68.0)
    spawn_xy = XY(0.0, 0.0)
    spawn_pt = proj.unproject(spawn_xy)
    zida = ZIDABoundary()  # 80 nmi
    t_cross = zida.crossing_time_s(
        spawn_point=spawn_pt, spawn_xy=spawn_xy, heading_rad=0.0,
        speed_mps=100.0, spawn_time_s=0.0, projection=proj,
    )
    assert t_cross > 0.0
    # At t_cross the UAS is ~exactly at the 80 nmi depth (consistent with is_inside).
    cur = XY(spawn_xy.x + 100.0 * math.cos(0.0) * t_cross,
             spawn_xy.y + 100.0 * math.sin(0.0) * t_cross)
    pen = zida.penetration_m(spawn_pt, proj.unproject(cur))
    assert pen == pytest.approx(zida.radius_m, abs=5.0)


# ─── T3 (controller) — t_ev_s logged: finite if evaded, NaN if neutralized ──


def test_t_ev_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MissionConfig()
    proj = LocalProjection(lat0_deg=cfg.aoo_centroid_lat, lon0_deg=cfg.aoo_centroid_lon)
    from cuas_sim.agents.uas import UAS

    def _outcome(force_hit: bool):
        monkeypatch.setattr(
            _BaseWeaponPolicy, "p_hit_at",
            lambda self, distance_m, archetype: (1.0 if force_hit else 0.0),
        )
        uas = UAS(
            uas_id=1, spawn_time_s=0.0, spawn_point=proj.unproject(XY(60_000.0, 0.0)),
            spawn_xy=XY(60_000.0, 0.0), archetype=_MR, heading_rad=math.pi,
            sector_index=39,
        )
        ctrl = EngagementController(
            cfg=cfg, tracking_policy=make_tracking_policy(cfg, "modernized"),
            weapon_policy=make_weapon_policy(cfg, "modernized"),
            projection=proj, config_name="modernized",
        )
        return ctrl.run(
            a29_id="A-29-T", aor_name="AoR-1", uas=uas,
            a29_xy_at_notify=XY(0.0, 0.0), t_notify_s=0.0,
            rng_tracking=np.random.default_rng(0), rng_firing=np.random.default_rng(0),
            rng_p_hit=np.random.default_rng(0), rng_reengage=np.random.default_rng(0),
        )

    miss = _outcome(force_hit=False)
    assert not miss.is_neutralized
    assert math.isfinite(miss.t_ev_s)
    assert miss.t_ev_s > 0.0

    kill = _outcome(force_hit=True)
    assert kill.is_neutralized
    assert math.isnan(kill.t_ev_s)
