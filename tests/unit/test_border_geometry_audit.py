"""Known-answer guards for the geometric primitives of the border audit.

``scripts/border_geometry_audit.py`` produces the numbers the dissertation
publishes (fraction of chord spawns inside Brazilian territory, inward-heading
adherence). A silently wrong ray-cast or point-to-polyline distance would
corrupt every one of them without failing anything, so both primitives get a
known-answer test here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from border_geometry_audit import (
    _load_brazil_ring,
    _polyline_lonlat,
    distance_to_polyline_km,
    points_in_ring,
    sector_pmf,
)


def _square() -> np.ndarray:
    """Closed unit square ring, counter-clockwise."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])


def test_points_in_ring_unit_square() -> None:
    ring = _square()
    lons = np.array([0.5, 0.01, 0.99, -0.5, 1.5, 0.5, 0.5])
    lats = np.array([0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 1.5])
    expected = np.array([True, True, True, False, False, False, False])
    assert list(points_in_ring(lons, lats, ring)) == list(expected)


def test_points_in_ring_is_not_trivially_constant() -> None:
    # Guards against a degenerate implementation returning all-True/all-False.
    # The grid is offset so no sample lands exactly on an edge (see the boundary
    # test below — on-edge classification is a convention, not a correctness
    # property, which is why the audit keeps a 200 m tolerance).
    ring = _square()
    lons = np.linspace(-0.97, 2.03, 61)
    lats = np.full_like(lons, 0.5)
    inside = points_in_ring(lons, lats, ring)
    assert inside.any() and not inside.all()
    assert list(inside) == list((lons > 0.0) & (lons < 1.0))


def test_points_in_ring_boundary_convention_is_half_open() -> None:
    # Ray casting with `px < x_cross` makes the left/bottom edges count as
    # interior and the right/top edges as exterior. Documented, not relied on:
    # `_INSIDE_TOL_KM` in the audit discards everything within 200 m of the line.
    ring = _square()
    lons = np.array([0.0, 1.0, 0.5, 0.5])
    lats = np.array([0.5, 0.5, 0.0, 1.0])
    assert list(points_in_ring(lons, lats, ring)) == [True, False, True, False]


def test_points_in_ring_brazil_anchors() -> None:
    ring = _load_brazil_ring()
    # Manaus and Brasilia are interior; Bogota, Lima and a mid-Atlantic point
    # are exterior. Coordinates are (lon, lat).
    lons = np.array([-60.02, -47.93, -74.07, -77.04, -30.00])
    lats = np.array([-3.10, -15.78, 4.71, -12.05, -10.00])
    expected = [True, True, False, False, False]
    assert list(points_in_ring(lons, lats, ring)) == expected


def test_arc_spawns_lie_on_the_polyline_chord_ones_do_not() -> None:
    from cuas_sim.config import MissionConfig
    from cuas_sim.world.border_polygon import _spawn_coord, default_border_polygon

    cfg = MissionConfig.from_yaml_with_sysml(
        _ROOT / "configs" / "zida_shield_default.yaml", sysml_root=_ROOT / "sysml")
    polygon = default_border_polygon(cfg)
    poly = _polyline_lonlat(polygon)

    worst_arc = 0.0
    worst_chord = 0.0
    for idx in polygon.indices:
        seg = polygon.segment(idx)
        for omega in (0.1, 0.3, 0.5, 0.7, 0.9):
            la, lo = _spawn_coord(seg, omega, "arc")
            worst_arc = max(worst_arc, distance_to_polyline_km(lo, la, poly))
            la, lo = _spawn_coord(seg, omega, "chord")
            worst_chord = max(worst_chord, distance_to_polyline_km(lo, la, poly))

    assert worst_arc < 1e-3          # < 1 m: on the stored polyline by construction
    assert worst_chord > 15.0        # the chord shortcut leaves it by tens of km


def test_sector_pmf_matches_miranda_rule() -> None:
    from cuas_sim.config import MissionConfig
    from cuas_sim.world.border_polygon import default_border_polygon

    cfg = MissionConfig.from_yaml_with_sysml(
        _ROOT / "configs" / "zida_shield_default.yaml", sysml_root=_ROOT / "sysml")
    pmf = sector_pmf(default_border_polygon(cfg), cfg.sector_mu, cfg.sector_sigma)

    assert pmf[0] == 0.0                                  # sector 1: zero mass
    assert pmf.sum() == np.float64(1.0)
    mean = float(np.sum(np.arange(1, 28) * pmf))
    assert abs(mean - 14.5) < 1e-3                        # half-sector offset
