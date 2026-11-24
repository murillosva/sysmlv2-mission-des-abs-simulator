"""Unit tests for the geometry submodule."""

from __future__ import annotations

import math

import pytest

from cuas_sim.geometry.coordinates import GeoPoint, parse_dms
from cuas_sim.geometry.haversine import haversine_m, t_pea_seconds
from cuas_sim.geometry.intercept import collision_course_intercept
from cuas_sim.geometry.projection import XY, LocalProjection

# ─── parse_dms ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("dms", "expected"),
    [
        ("8°42'59.61\"S", -8.716558333333334),
        ("63°54'7.46\"O", -63.90207222222222),
        ("0°25'53.75\"N", 0.4315972222222222),
        ("1°7'11.98\"N", 1.119994444),
    ],
)
def test_parse_dms_known_values(dms: str, expected: float) -> None:
    assert parse_dms(dms) == pytest.approx(expected, abs=1e-6)


def test_parse_dms_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_dms("not a coordinate")


def test_geopoint_range_checks() -> None:
    with pytest.raises(ValueError):
        GeoPoint(lat=100, lon=0)
    with pytest.raises(ValueError):
        GeoPoint(lat=0, lon=200)


# ─── projection ────────────────────────────────────────────────────────────


def test_projection_roundtrip_at_centroid() -> None:
    proj = LocalProjection(lat0_deg=-5.0, lon0_deg=-68.0)
    pt = GeoPoint(lat=-5.0, lon=-68.0)
    xy = proj.project(pt)
    assert math.isclose(xy.x, 0.0, abs_tol=1e-6)
    assert math.isclose(xy.y, 0.0, abs_tol=1e-6)
    back = proj.unproject(xy)
    assert back.lat == pytest.approx(-5.0, abs=1e-9)
    assert back.lon == pytest.approx(-68.0, abs=1e-9)


def test_projection_roundtrip_near_aoo() -> None:
    proj = LocalProjection(lat0_deg=-5.0, lon0_deg=-68.0)
    for lat, lon in [(-9.0, -73.0), (1.0, -67.0), (-6.5, -68.5)]:
        pt = GeoPoint(lat=lat, lon=lon)
        back = proj.unproject(proj.project(pt))
        assert back.lat == pytest.approx(lat, abs=1e-9)
        assert back.lon == pytest.approx(lon, abs=1e-9)


# ─── haversine ─────────────────────────────────────────────────────────────


def test_haversine_zero_distance() -> None:
    p = GeoPoint(lat=-8.0, lon=-63.0)
    assert haversine_m(p, p) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_lat() -> None:
    """One degree of latitude ≈ 111_195 m (along a meridian on a sphere)."""
    p1 = GeoPoint(lat=0.0, lon=0.0)
    p2 = GeoPoint(lat=1.0, lon=0.0)
    d = haversine_m(p1, p2)
    assert d == pytest.approx(111_195.0, rel=1e-3)


def test_t_pea_seconds_positive_for_typical_geometry() -> None:
    farp = GeoPoint(lat=-7.6, lon=-72.8)   # SBCZ
    cap = GeoPoint(lat=-8.7, lon=-72.1)    # roughly CAP-2
    t = t_pea_seconds(farp, cap, endurance_s=5.2 * 3600, cruise_speed_mps=144.0)
    assert t > 0
    # The transit consumes 2*d/v; the remainder must be ≤ endurance.
    assert t < 5.2 * 3600


# ─── intercept ─────────────────────────────────────────────────────────────
# Note: see also Spec 016 §FR-5.
# ``collision_course_intercept`` is deprecated for new PURSUING code (Spec 016
# replaces it with ``pure_pursuit_step`` + discrete integration), but remains
# in-tree for animation replay, Spec 014 regression fixtures, and the golden
# anti-regression tests below — those keep this function exercised.


def test_head_on_intercept_collinear() -> None:
    """A-29 and UAS approach head-on along +x; intercept time = d / (v_a + v_u)."""
    a = XY(0.0, 0.0)
    u = XY(10_000.0, 0.0)   # 10 km east
    # UAS heading west = π
    geom = collision_course_intercept(
        a29_xy=a, a29_speed_mps=144.0,
        uas_xy=u, uas_heading_rad=math.pi, uas_speed_mps=29.0,
        envelope_max_m=2_000.0,
    )
    assert geom.feasible
    expected_t = 10_000.0 / (144.0 + 29.0)
    assert geom.t_intercept == pytest.approx(expected_t, rel=1e-6)


def test_running_away_uas_infeasible_for_slower_a29() -> None:
    """If UAS heading east at v > A-29 speed, no intercept."""
    a = XY(0.0, 0.0)
    u = XY(5_000.0, 0.0)
    geom = collision_course_intercept(
        a29_xy=a, a29_speed_mps=100.0,
        uas_xy=u, uas_heading_rad=0.0, uas_speed_mps=130.0,
        envelope_max_m=2_000.0,
    )
    assert not geom.feasible


def test_envelope_entry_before_intercept() -> None:
    """The envelope-entry time should be ≤ intercept time, ≥ 0."""
    geom = collision_course_intercept(
        a29_xy=XY(0.0, 0.0), a29_speed_mps=144.0,
        uas_xy=XY(15_000.0, 0.0), uas_heading_rad=math.pi, uas_speed_mps=29.0,
        envelope_max_m=2_000.0,
    )
    assert geom.feasible
    assert 0.0 <= geom.t_envelope_entry <= geom.t_intercept


# ─── time_to_separation — Spec 014 ─────────────────────────────────────────


def test_time_to_separation_head_on_2km() -> None:
    from cuas_sim.geometry.intercept import time_to_separation

    t = time_to_separation(
        XY(0.0, 0.0), 144.0, XY(100_000.0, 0.0), math.pi, 29.0,
        a29_heading_rad=0.0, target_sep_m=2_000.0,
    )
    assert t == pytest.approx(98_000.0 / 173.0, rel=1e-6)  # ≈566.4740 s


def test_time_to_separation_head_on_500m_after_2km() -> None:
    from cuas_sim.geometry.intercept import time_to_separation

    base = (XY(0.0, 0.0), 144.0, XY(100_000.0, 0.0), math.pi, 29.0)
    t_500 = time_to_separation(*base, a29_heading_rad=0.0, target_sep_m=500.0)
    t_2000 = time_to_separation(*base, a29_heading_rad=0.0, target_sep_m=2_000.0)
    assert t_500 == pytest.approx(99_500.0 / 173.0, rel=1e-6)  # ≈575.1445 s
    # Bug corrigido: o tiro modernizado (2 km) ocorre ANTES do baseline (0.5 km).
    # Sob a interpolação antiga ambos caíam em t_intercept (filosofia curta).
    assert t_2000 < t_500


def test_time_to_separation_equals_envelope_entry_at_env_max() -> None:
    from cuas_sim.geometry.intercept import time_to_separation

    args = (XY(0.0, 0.0), 144.0, XY(100_000.0, 0.0), math.pi, 29.0)
    geom = collision_course_intercept(*args, envelope_max_m=14_000.0)
    t = time_to_separation(
        *args, a29_heading_rad=geom.a29_heading_rad, target_sep_m=14_000.0,
    )
    assert t == pytest.approx(geom.t_envelope_entry, rel=1e-9)


def test_time_to_separation_monotonic_chain() -> None:
    from cuas_sim.geometry.intercept import time_to_separation

    args = (XY(0.0, 0.0), 144.0, XY(100_000.0, 0.0), math.pi, 29.0)
    geom = collision_course_intercept(*args, envelope_max_m=14_000.0)
    h = geom.a29_heading_rad
    t_long = time_to_separation(*args, a29_heading_rad=h, target_sep_m=14_000.0)
    t_med = time_to_separation(*args, a29_heading_rad=h, target_sep_m=8_000.0)
    t_short = time_to_separation(*args, a29_heading_rad=h, target_sep_m=2_000.0)
    assert t_long <= t_med <= t_short <= geom.t_intercept + 1e-6


# T8 — golden anti-regressão de collision_course_intercept (FR-2 / NFR-1).
# Valores capturados da implementação PRÉ-mudança; DEVEM permanecer idênticos.
_GOLDEN_INTERCEPT = [
    ("headon_env2k", XY(0.0, 0.0), 144.0, XY(100_000.0, 0.0), math.pi, 29.0,
     2_000.0, True, 578.0346820809249, 83236.99421965318,
     2.0528761372990317e-12, 2.466302581616198e-17, 566.4739884393064,
     100000.0),
    ("headon_env14k", XY(0.0, 0.0), 144.0, XY(100_000.0, 0.0), math.pi, 29.0,
     14_000.0, True, 578.0346820809249, 83236.99421965318,
     2.0528761372990317e-12, 2.466302581616198e-17, 497.1098265895954,
     100000.0),
    ("oblique_fw", XY(0.0, 0.0), 144.0, XY(60_000.0, 40_000.0), math.pi * 0.75,
     130.0, 14_000.0, True, 779.8087545026029, -11683.047583865395,
     111683.04758386541, 1.6750261940619897, 628.4127303033418,
     72111.02550927979),
    ("tail_chase", XY(0.0, 0.0), 144.0, XY(5_000.0, 0.0), 0.0, 29.0,
     2_000.0, True, 43.47826086956522, 6260.869565217391, 0.0, 0.0,
     26.08695652173913, 5000.0),
    ("d0_lt_envmax", XY(0.0, 0.0), 144.0, XY(8_000.0, 0.0), math.pi, 29.0,
     14_000.0, True, 46.24277456647399, 6658.959537572255,
     1.6423009098392252e-13, 2.4663025816161973e-17, 0.0, 8000.0),
    ("equal_speed", XY(0.0, 0.0), 130.0, XY(50_000.0, 10_000.0), math.pi,
     130.0, 2_000.0, True, 200.0, 24000.0, 10000.000000000004,
     0.39479111969976166, 192.15535459447264, 50990.19513592785),
]


@pytest.mark.parametrize("case", _GOLDEN_INTERCEPT, ids=lambda c: c[0])
def test_collision_course_intercept_golden(case: tuple) -> None:
    (_name, a, asp, u, uh, usp, env, feas, t_int, ix, iy, head, t_env,
     d0) = case
    g = collision_course_intercept(a, asp, u, uh, usp, env)
    assert g.feasible is feas
    assert g.t_intercept == pytest.approx(t_int, rel=1e-12, abs=1e-9)
    assert g.intercept_xy.x == pytest.approx(ix, rel=1e-12, abs=1e-6)
    assert g.intercept_xy.y == pytest.approx(iy, rel=1e-12, abs=1e-6)
    assert g.a29_heading_rad == pytest.approx(head, rel=1e-12, abs=1e-15)
    assert g.t_envelope_entry == pytest.approx(t_env, rel=1e-12, abs=1e-9)
    assert g.initial_distance_m == pytest.approx(d0, rel=1e-12, abs=1e-6)
