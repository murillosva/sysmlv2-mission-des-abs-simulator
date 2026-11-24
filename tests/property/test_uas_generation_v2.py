"""Property-based oracles for Spec 030 — Miranda (2025) v2 UAS generation.

Spec: Spec 030 §5.2 (P1-P6).
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pyproj import Geod

from cuas_sim.config import MissionConfig
from cuas_sim.world.border_polygon import default_border_polygon, sample_sector

_GEOD = Geod(ellps="WGS84")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "zida_shield_default.yaml"
_SYSML_ROOT = _REPO_ROOT / "sysml"
_MU = 14.0
_SIGMA = (27 - 1) / 6


@lru_cache(maxsize=1)
def _cfg() -> MissionConfig:
    return MissionConfig.from_yaml_with_sysml(_CONFIG_PATH, sysml_root=_SYSML_ROOT)


@lru_cache(maxsize=1)
def _polygon():
    return default_border_polygon(_cfg())


class _ScriptedRNG:
    """Fake generator matching spawn_in_sector / sample_sector call patterns."""

    def __init__(self, uniforms: list[float], normals: list[float] | None = None):
        self._uniforms = list(uniforms)
        self._normals = list(normals or [0.0])
        self._ui = 0
        self._ni = 0

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        if self._ui < len(self._uniforms):
            v = self._uniforms[self._ui]
            self._ui += 1
            return v
        return self._uniforms[-1]

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        v = self._normals[min(self._ni, len(self._normals) - 1)]
        self._ni += 1
        return loc + v * scale


# ─── P1 — sector 1 has exactly zero mass ───────────────────────────────────


@given(u=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=500)
def test_sector1_zero_mass(u: float) -> None:
    pg = _polygon()
    # After a rejected first draw the fake returns 0.5 (accepted -> sector 15).
    rng = _ScriptedRNG([u, 0.5])
    k = sample_sector(pg, rng, mu=_MU, sigma=_SIGMA, rule="cdf")
    assert k >= 2


# ─── P2 — spawn point lies on the stored polyline ──────────────────────────


def _pt_seg_dist_deg(px: float, py: float, ax: float, ay: float,
                     bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


@given(idx=st.integers(min_value=1, max_value=27),
       omega=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=500)
def test_point_on_polyline(idx: int, omega: float) -> None:
    pg = _polygon()
    draw = pg.spawn_in_sector(idx, _ScriptedRNG([omega]), 0.0, interpolation="arc")
    arc = pg.segment(idx).arc
    px, py = draw.point.lon, draw.point.lat
    best = min(
        _pt_seg_dist_deg(px, py, a[1], a[0], b[1], b[0])
        for a, b in pairwise(arc)
    )
    assert best < 1e-8   # linear interp -> point is on a lat/lon subsegment


# ─── P3 — omega monotonic (points march forward along the arc) ─────────────


@given(idx=st.integers(min_value=1, max_value=27))
@settings(max_examples=50)
def test_omega_monotonic(idx: int) -> None:
    pg = _polygon()
    omegas = np.linspace(0.0, 1.0, 60)
    pts = [pg.spawn_in_sector(idx, _ScriptedRNG([float(w)]), 0.0,
                              interpolation="arc").point for w in omegas]
    total = 0.0
    for p, q in pairwise(pts):
        _, _, d = _GEOD.inv(p.lon, p.lat, q.lon, q.lat)
        total += d / 1000.0
    s_total = pg.segment(idx).arc[-1][2]
    # Ordered traversal: the chord sum approaches the arc length from below.
    assert 0.985 * s_total <= total <= s_total * 1.0005


# ─── P4 — heading rule invariance (coincide or differ by pi) ────────────────


@given(idx=st.integers(min_value=1, max_value=27))
@settings(max_examples=50)
def test_heading_rule_invariance(idx: int) -> None:
    pg = _polygon()
    lit = pg.spawn_in_sector(idx, _ScriptedRNG([0.5]), 0.0, heading_rule="literal")
    flip = pg.spawn_in_sector(idx, _ScriptedRNG([0.5]), 0.0, heading_rule="east_flip")
    d = abs((lit.heading_rad - flip.heading_rad + math.pi) % (2 * math.pi) - math.pi)
    assert d < 1e-6 or abs(d - math.pi) < 1e-6


# ─── P5 — stream neutrality across interp x heading combos ─────────────────


def test_stream_neutrality() -> None:
    from cuas_sim.simulation.run import run_paired

    base = _cfg().replaced(lambda_per_sec=18.0 / 3600.0, skip_pre_pea=True,
                           sector_draw_rule="pdf")
    combos = [("arc", "literal"), ("arc", "east_flip"),
              ("chord", "literal"), ("chord", "east_flip")]
    births = {}
    for interp, heading in combos:
        cfg = base.replaced(spawn_interpolation=interp, heading_inward_rule=heading)
        res = run_paired(cfg, seed=11, persist=False, max_arrivals=30)
        births[(interp, heading)] = res.log.uas_births

    sectors = {c: [b["sector_index"] for b in bs] for c, bs in births.items()}
    ref = sectors[combos[0]]
    for c in combos[1:]:
        assert sectors[c] == ref, f"sector sequence differs for {c}"
    # The two chord variants encode omega bijectively in lat/lon -> identical.
    cl = [(b["lat"], b["lon"]) for b in births[("chord", "literal")]]
    cf = [(b["lat"], b["lon"]) for b in births[("chord", "east_flip")]]
    assert cl == cf


# ─── P6 — determinism (same seed -> bit-identical) in all 8 flag combos ────


@pytest.mark.parametrize("interp", ["arc", "chord"])
@pytest.mark.parametrize("rule", ["cdf", "pdf"])
@pytest.mark.parametrize("heading", ["literal", "east_flip"])
def test_determinism_all_flag_combos(interp: str, rule: str, heading: str) -> None:
    import pandas as pd

    from cuas_sim.simulation.run import run_paired

    cfg = _cfg().replaced(lambda_per_sec=18.0 / 3600.0, skip_pre_pea=True,
                          spawn_interpolation=interp, sector_draw_rule=rule,
                          heading_inward_rule=heading)
    a = run_paired(cfg, seed=99, persist=False, max_arrivals=20)
    b = run_paired(cfg, seed=99, persist=False, max_arrivals=20)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)
