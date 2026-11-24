"""Tests for world submodule: border polygon, AoR map, ZIDA boundary."""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from pyproj import Geod

from cuas_sim.config import BorderSegment, MissionConfig
from cuas_sim.geometry.coordinates import GeoPoint
from cuas_sim.world.aor import AoRMap
from cuas_sim.world.border_polygon import (
    BorderPolygon,
    default_border_polygon,
    sample_sector,
    truncated_normal_sector_pmf,
)
from cuas_sim.world.boundary import NMI_TO_M, ZIDABoundary

_GEOD = Geod(ellps="WGS84")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "zida_shield_default.yaml"
_SYSML_ROOT = _REPO_ROOT / "sysml"


@lru_cache(maxsize=1)
def _yaml_polygon() -> BorderPolygon:
    """The production 27-sector polygon (with `arc` after Spec 030 C1.2)."""
    cfg = MissionConfig.from_yaml_with_sysml(_CONFIG_PATH, sysml_root=_SYSML_ROOT)
    return default_border_polygon(cfg)


class _ScriptedRNG:
    """Fake generator returning scripted ``uniform``/``normal`` draws.

    Matches the call pattern of ``spawn_in_sector`` (uniform for omega, then
    normal for the heading deviation) and of ``sample_sector`` rule="cdf"
    (uniform for u, repeated on resort). Unscripted uniforms return the last
    scripted value (so resort loops terminate on an accepted value).
    """

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

# ─── BorderPolygon ─────────────────────────────────────────────────────────


def test_default_polygon_has_27_segments() -> None:
    cfg = MissionConfig()
    pg = default_border_polygon(cfg)
    expected = cfg.sector_max - cfg.sector_min + 1
    assert len(pg) == expected
    assert pg.min_index == cfg.sector_min
    assert pg.max_index == cfg.sector_max


def test_polygon_segment_index_membership() -> None:
    cfg = MissionConfig()
    pg = default_border_polygon(cfg)
    for k in range(cfg.sector_min, cfg.sector_max + 1):
        assert k in pg
    assert (cfg.sector_min - 1) not in pg
    assert (cfg.sector_max + 1) not in pg


def test_polygon_segment_has_two_geopoints() -> None:
    cfg = MissionConfig()
    pg = default_border_polygon(cfg)
    seg = pg.segment(1)
    assert isinstance(seg.start, GeoPoint)
    assert isinstance(seg.end, GeoPoint)
    # Endpoints are distinct.
    assert seg.start != seg.end


def test_spawn_in_sector_inside_segment() -> None:
    cfg = MissionConfig()
    pg = default_border_polygon(cfg)
    rng = np.random.default_rng(42)
    draw = pg.spawn_in_sector(10, rng, heading_dev_std_deg=15.0)
    assert draw.sector_index == 10
    seg = pg.segment(10)
    # Drawn point should lie on the segment polyline (within numerical tol).
    a = (draw.point.lat - seg.start.lat) / (seg.end.lat - seg.start.lat) \
        if (seg.end.lat - seg.start.lat) != 0 else None
    if a is not None:
        assert 0.0 <= a <= 1.0
    assert 0.0 <= draw.omega <= 1.0


def test_spawn_heading_points_inward() -> None:
    """Spawned UAS must head eastward (into Brazil) from the western border."""
    cfg = MissionConfig()
    pg = default_border_polygon(cfg)
    rng = np.random.default_rng(0)
    # heading_dev_std_deg=0 eliminates randomness; tests baseline formula only
    draw = pg.spawn_in_sector(14, rng, heading_dev_std_deg=0.0)
    # math convention: cos > 0 means East component present = heading into Brazil
    assert math.cos(draw.heading_rad) > 0, (
        f"UAS heading has negative East component (heading_rad={draw.heading_rad:.3f}), "
        "meaning UAS would move westward (out of Brazil)."
    )


# ─── Truncated-normal PMF / sampler ───────────────────────────────────────


def test_pmf_sums_to_one() -> None:
    cfg = MissionConfig()
    indices = tuple(range(cfg.sector_min, cfg.sector_max + 1))
    pmf = truncated_normal_sector_pmf(indices, mu=cfg.sector_mu, sigma=cfg.sector_sigma)
    assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-12)


def test_pmf_mode_near_mu() -> None:
    cfg = MissionConfig()
    indices = tuple(range(cfg.sector_min, cfg.sector_max + 1))
    pmf = truncated_normal_sector_pmf(indices, mu=cfg.sector_mu, sigma=cfg.sector_sigma)
    mode_idx = indices[int(pmf.argmax())]
    assert abs(mode_idx - round(cfg.sector_mu)) <= 1


def test_sample_sector_returns_valid_index() -> None:
    cfg = MissionConfig()
    pg = default_border_polygon(cfg)
    rng = np.random.default_rng(1)
    for _ in range(200):
        s = sample_sector(pg, rng, mu=cfg.sector_mu, sigma=cfg.sector_sigma)
        assert cfg.sector_min <= s <= cfg.sector_max


# ─── AoRMap ────────────────────────────────────────────────────────────────


def test_aor_map_full_coverage_default_config() -> None:
    """Every sector in [sector_min, sector_max] is mapped to exactly one AoR by default."""
    cfg = MissionConfig()
    m = AoRMap.from_config(cfg)
    expected_count = cfg.sector_max - cfg.sector_min + 1
    for s in range(cfg.sector_min, cfg.sector_max + 1):
        assert s in m.sector_to_aor
    assert len(m.sector_to_aor) == expected_count


def test_aor_map_a29_inverse_consistent() -> None:
    cfg = MissionConfig()
    m = AoRMap.from_config(cfg)
    for aor, a29 in m.aor_to_a29.items():
        assert m.a29_to_aor[a29] == aor
    for sector in range(cfg.sector_min, cfg.sector_max + 1):
        aor = m.aor_of_sector(sector)
        a29 = m.a29_of_aor(aor)
        assert m.a29_of_sector(sector) == a29


def test_aor_map_missing_coverage_raises() -> None:
    cfg = MissionConfig(aor_to_sectors={"AoR-1": [28, 29]})  # incomplete
    with pytest.raises(ValueError, match="does not cover"):
        AoRMap.from_config(cfg)


# ─── ZIDA boundary ────────────────────────────────────────────────────────


def test_zida_inside_at_spawn() -> None:
    z = ZIDABoundary(radius_nmi=80.0)
    p = GeoPoint(lat=-5.0, lon=-68.0)
    assert z.is_inside(p, p) is True
    assert z.penetration_m(p, p) == pytest.approx(0.0, abs=1e-6)


def test_zida_outside_beyond_80_nmi() -> None:
    """A point ~100 nmi south of spawn is outside the 80 nmi ZIDA."""
    z = ZIDABoundary(radius_nmi=80.0)
    spawn = GeoPoint(lat=-5.0, lon=-68.0)
    # 100 nmi ≈ 1.667 ° latitude south
    far = GeoPoint(lat=-6.667, lon=-68.0)
    assert z.is_inside(spawn, far) is False
    assert z.penetration_m(spawn, far) > 80 * NMI_TO_M


def test_zida_radius_conversion() -> None:
    z = ZIDABoundary(radius_nmi=80.0)
    assert z.radius_m == pytest.approx(80.0 * NMI_TO_M, abs=1e-9)


# ─── Spec 030 — Miranda (2025) v2 generation ───────────────────────────────


def test_sector_cdf_known_values() -> None:
    from cuas_sim.world.border_polygon import sector_cdf

    n = 27
    mu = (1 + n) / 2
    sigma = (n - 1) / 6
    f = sector_cdf(tuple(range(1, n + 1)), mu, sigma)
    assert f[0] == pytest.approx(0.001349898, abs=1e-9)
    assert f[13] == pytest.approx(0.5, abs=1e-9)
    assert f[26] == pytest.approx(0.998650102, abs=1e-9)
    # Table A.4 of the thesis (6 decimals).
    assert round(float(f[1]), 6) == 0.002809
    assert round(float(f[12]), 6) == 0.408747
    assert round(float(f[14]), 6) == 0.591253
    assert round(float(f[25]), 6) == 0.997191


def test_production_yaml_stores_calibration_at_full_precision() -> None:
    # test_sector_cdf_known_values proves the FORMULA reproduces the published
    # table; this proves the shipped YAML actually USES the formula instead of
    # the truncated literal 4.3333 that FR-9a replaced.
    cfg = MissionConfig.from_yaml_with_sysml(_CONFIG_PATH, sysml_root=_SYSML_ROOT)
    n = cfg.sector_max - cfg.sector_min + 1
    assert cfg.sector_mu == (1 + n) / 2
    assert cfg.sector_sigma == (n - 1) / 6          # exact, not approx
    # The +-3 sigma envelope must land exactly on the extreme sector indices —
    # this is what makes sector 1 carry zero mass and sector N close the column.
    assert cfg.sector_mu - 3 * cfg.sector_sigma == float(cfg.sector_min)
    assert cfg.sector_mu + 3 * cfg.sector_sigma == float(cfg.sector_max)


def test_truncated_sigma_would_break_the_published_table() -> None:
    # Documents the failure mode the guard above prevents, so a future reader
    # sees why full precision is not pedantry.
    from cuas_sim.world.border_polygon import sector_cdf

    n, mu = 27, 14.0
    exact = sector_cdf(tuple(range(1, n + 1)), mu, (n - 1) / 6)
    truncated = sector_cdf(tuple(range(1, n + 1)), mu, 4.3333)

    # 16 of the 27 rows change in the 6th decimal the dissertation publishes.
    differing = sum(1 for a, b in zip(exact, truncated, strict=True)
                    if round(float(a), 6) != round(float(b), 6))
    assert differing == 16
    # And the +-3 sigma envelope no longer lands on the extreme sectors.
    assert mu - 3 * 4.3333 != 1.0
    assert mu + 3 * 4.3333 != 27.0


def test_sample_sector_cdf_mapping() -> None:
    from cuas_sim.world.border_polygon import sector_cdf

    pg = _yaml_polygon()
    mu, sigma = 14.0, (27 - 1) / 6
    f = sector_cdf(tuple(pg.indices), mu, sigma)   # exact boundaries (F(1)~=0.001350, F(27)~=0.998650)

    def pick(u_first: float) -> int:
        # After a rejected first draw the fake returns 0.5 (accepted -> sector 15).
        rng = _ScriptedRNG([u_first, 0.5])
        return sample_sector(pg, rng, mu=mu, sigma=sigma, rule="cdf")

    assert pick(float(f[0])) == 2                       # u == F(1): accepted -> sector 2
    assert pick(0.5) == 15                              # F(14)=0.5 exact; cell (k-1, k]
    assert pick(0.499999900) == 14
    assert pick(math.nextafter(float(f[-1]), 0.0)) == 27   # just below F(27) -> sector 27
    # Rejections (outside [F(1), F(27))) resort to the accepted 0.5 -> 15.
    assert pick(math.nextafter(float(f[0]), 0.0)) == 15    # u < F(1) rejected
    assert pick(float(f[-1])) == 15                        # u >= F(27) rejected


def test_arc_vs_chord_sector21() -> None:
    pg = _yaml_polygon()
    arc = pg.spawn_in_sector(21, _ScriptedRNG([0.5]), 0.0, interpolation="arc")
    chord = pg.spawn_in_sector(21, _ScriptedRNG([0.5]), 0.0, interpolation="chord")
    assert arc.point.lat == pytest.approx(-9.396379572, abs=1e-6)
    assert arc.point.lon == pytest.approx(-73.197075237, abs=1e-6)
    assert chord.point.lat == pytest.approx(-9.221400500, abs=1e-6)
    assert chord.point.lon == pytest.approx(-72.873852000, abs=1e-6)
    _, _, sep = _GEOD.inv(arc.point.lon, arc.point.lat, chord.point.lon, chord.point.lat)
    assert sep / 1000.0 == pytest.approx(40.442, abs=0.005)


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference (rad), in [0, pi]."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def test_heading_literal_vs_flip() -> None:
    pg = _yaml_polygon()
    for sec in (24, 27):
        lit = pg.spawn_in_sector(sec, _ScriptedRNG([0.5]), 0.0, heading_rule="literal")
        flip = pg.spawn_in_sector(sec, _ScriptedRNG([0.5]), 0.0, heading_rule="east_flip")
        assert _ang_diff(lit.heading_rad, flip.heading_rad) == pytest.approx(math.pi, abs=1e-6)
    # Sector 14 is the control: the two rules coincide (preserves Spec 007 T1).
    lit14 = pg.spawn_in_sector(14, _ScriptedRNG([0.5]), 0.0, heading_rule="literal")
    flip14 = pg.spawn_in_sector(14, _ScriptedRNG([0.5]), 0.0, heading_rule="east_flip")
    assert _ang_diff(lit14.heading_rad, flip14.heading_rad) == pytest.approx(0.0, abs=1e-6)
    # Exact literal math-heading values (non-normalised convention).
    lit24 = pg.spawn_in_sector(24, _ScriptedRNG([0.5]), 0.0, heading_rule="literal")
    lit27 = pg.spawn_in_sector(27, _ScriptedRNG([0.5]), 0.0, heading_rule="literal")
    assert _ang_diff(lit24.heading_rad, 2.194938) == pytest.approx(0.0, abs=1e-5)
    assert _ang_diff(lit27.heading_rad, 1.571875) == pytest.approx(0.0, abs=1e-5)


def test_arc_interp_error_bound() -> None:
    pg = _yaml_polygon()
    worst_m = 0.0
    for idx in pg.indices:
        arc = pg.segment(idx).arc
        for (lat0, lon0, _), (lat1, lon1, _) in pairwise(arc):
            mlat, mlon = (lat0 + lat1) / 2.0, (lon0 + lon1) / 2.0
            az, _, dist = _GEOD.inv(lon0, lat0, lon1, lat1)
            glon, glat, _ = _GEOD.fwd(lon0, lat0, az, dist / 2.0)
            _, _, err = _GEOD.inv(mlon, mlat, glon, glat)
            worst_m = max(worst_m, err)
    assert worst_m <= 3.25


def test_arc_empty_falls_back_to_chord() -> None:
    # The synthetic default polygon has arc=() -> interpolation="arc" must match chord.
    pg = default_border_polygon(MissionConfig())
    idx = pg.indices[len(pg.indices) // 2]
    arc = pg.spawn_in_sector(idx, _ScriptedRNG([0.37]), 0.0, interpolation="arc")
    chord = pg.spawn_in_sector(idx, _ScriptedRNG([0.37]), 0.0, interpolation="chord")
    assert arc.point.lat == chord.point.lat     # bit-identical
    assert arc.point.lon == chord.point.lon


def test_orientation_invariant_raises() -> None:
    # A south-to-north polygon (start.lat of sector 1 < end.lat of last) is rejected.
    segs = (
        BorderSegment(index=1, start=GeoPoint(lat=-9.0, lon=-73.0),
                      end=GeoPoint(lat=-8.0, lon=-72.5)),
        BorderSegment(index=2, start=GeoPoint(lat=-8.0, lon=-72.5),
                      end=GeoPoint(lat=1.0, lon=-67.0)),
    )
    with pytest.raises(ValueError, match=r"FR-16|orient"):
        BorderPolygon(segs)


def test_default_polygon_is_north_to_south() -> None:
    pg = default_border_polygon(MissionConfig())
    assert pg.segment(pg.min_index).start.lat > pg.segment(pg.max_index).end.lat


def test_paired_spawns_bit_identical(tmp_path) -> None:
    import pandas as pd

    from cuas_sim.simulation.persistence import RunDirectory, persist_run
    from cuas_sim.simulation.run import run_paired

    base = MissionConfig.from_yaml_with_sysml(_CONFIG_PATH, sysml_root=_SYSML_ROOT)
    base = base.replaced(lambda_per_sec=18.0 / 3600.0, skip_pre_pea=True,
                         sector_draw_rule="pdf")

    def spawns_for(interp: str, heading: str) -> pd.DataFrame:
        cfg = base.replaced(spawn_interpolation=interp, heading_inward_rule=heading)
        res = run_paired(cfg, seed=7, persist=False, max_arrivals=40)
        run_dir = RunDirectory.create(tmp_path, slug=f"{interp}_{heading}")
        persist_run(res, run_dir)
        return pd.read_parquet(run_dir.spawns_parquet).sort_values("uas_id").reset_index(drop=True)

    r0 = spawns_for("chord", "east_flip")
    r1 = spawns_for("chord", "literal")
    r2 = spawns_for("arc", "literal")

    # R0 <-> R1: only the heading rule changes -> sector + position identical.
    assert (r0["sector_index"] == r1["sector_index"]).all()
    assert np.array_equal(r0["lat"].to_numpy(), r1["lat"].to_numpy())
    assert np.array_equal(r0["lon"].to_numpy(), r1["lon"].to_numpy())
    # R1 <-> R2: only the interpolation changes -> sector + heading identical.
    assert (r1["sector_index"] == r2["sector_index"]).all()
    assert np.array_equal(r1["heading_rad"].to_numpy(), r2["heading_rad"].to_numpy())
