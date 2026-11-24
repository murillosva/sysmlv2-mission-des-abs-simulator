"""Geometric audit of the incursion-generating border polygon.

Three blocks, all measured on the simulator's own code (``_spawn_coord`` and
``BorderPolygon.spawn_in_sector``), not on a reimplementation:

A. **Chord versus arc.** Under ``spawn_interpolation="chord"``, the fraction of origins
   that would fall inside Brazilian territory, their maximum depth, the mean offset from
   the adopted border line, and the arc-to-chord separation per sector.

B. **Compression of omega.** The ratio arc length / chord length per sector: with omega
   uniform along the chord, the density along the real border is compressed by this
   factor.

C. **Inward heading.** Advances 25 km along the perpendicular heading of each rule
   (``literal`` = ``az - 90`` unconditionally; ``east_flip`` = the earlier heuristic) and
   tests whether the point lies inside Brazilian territory. The block runs both
   interpolations, and the two rules are comparable only within one interpolation (one
   row of the summary); the conclusion is the same in both: the literal rule points
   inward, and the heuristic points outward in sectors 24 and 27.

The measurement is deterministic: omega walks a uniform grid, with no random numbers (the
audit describes the geometry, not the draw). Point-in-polygon is a ray-casting test in
numpy over the vendored ring ``scripts/data/brazil_ne50m_mainland.json``; no extra
dependency. It runs no simulation and touches no measure.

Usage::

    python scripts/border_geometry_audit.py
    python scripts/border_geometry_audit.py --samples 201 \
        --out outputs/verification/border_geometry_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Geod

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from cuas_sim.config import MissionConfig  # noqa: E402
from cuas_sim.world.border_polygon import (  # noqa: E402
    BorderPolygon,
    _spawn_coord,
    default_border_polygon,
    sector_cdf,
)

_GEOD = Geod(ellps="WGS84")

#: Exterior ring of mainland Brazil (Natural Earth 50m), vendored.
_BRAZIL = _HERE / "data" / "brazil_ne50m_mainland.json"

#: Distance advanced along the heading for the inside-territory test.
_HEADING_PROBE_KM = 25.0

#: Tolerance for "strictly inside": below it the point lies on the line, not in the
#: interior (the Natural Earth ring is the adopted border).
_INSIDE_TOL_KM = 0.2


# ─── Point in polygon (ray casting, no shapely) ────────────────────────────


def _load_brazil_ring() -> np.ndarray:
    data = json.loads(_BRAZIL.read_text(encoding="utf-8"))
    return np.asarray(data["ring_lonlat"], dtype=float)


def points_in_ring(lons: np.ndarray, lats: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorized ray casting: ``True`` for points inside ``ring``.

    Half-open rule in latitude (``y_i <= y < y_j`` or ``y_j <= y < y_i``), which handles
    vertices without double counting. Pure: no random numbers, no I/O.
    """
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    px = lons[:, None]
    py = lats[:, None]
    straddles = ((y1 <= py) & (py < y2)) | ((y2 <= py) & (py < y1))
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    crossings = straddles & (px < x_cross)
    return (crossings.sum(axis=1) % 2).astype(bool)


# ─── Distance to the polyline ──────────────────────────────────────────────


def _polyline_lonlat(polygon: BorderPolygon) -> np.ndarray:
    """Join the arcs of the 27 sectors into a single ``(lon, lat)`` polyline."""
    pts: list[tuple[float, float]] = []
    for idx in polygon.indices:
        for lat, lon, _s in polygon.segment(idx).arc:
            if not pts or pts[-1] != (lon, lat):
                pts.append((lon, lat))
    return np.asarray(pts, dtype=float)


def distance_to_polyline_km(lon: float, lat: float, poly: np.ndarray) -> float:
    """Shortest geodesic distance from the point to the polyline, in km.

    Finds the foot of the perpendicular on the nearest sub-segment in plane coordinates
    (the local distortion does not matter for choosing the sub-segment) and measures the
    final distance with ``Geod.inv``.
    """
    ax, ay = poly[:-1, 0], poly[:-1, 1]
    bx, by = poly[1:, 0], poly[1:, 1]
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(den > 0, ((lon - ax) * dx + (lat - ay) * dy) / den, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    j = int(np.argmin((lon - cx) ** 2 + (lat - cy) ** 2))
    return float(_GEOD.inv(lon, lat, cx[j], cy[j])[2]) / 1000.0


# ─── Audit blocks ──────────────────────────────────────────────────────────


def sector_pmf(polygon: BorderPolygon, mu: float, sigma: float) -> np.ndarray:
    """Sector-selection pmf under the CDF rule.

    ``p(k) = F(k) - F(k-1)``, renormalized; ``p(1) = 0`` by construction, since
    ``mu - 3*sigma = 1`` and every ``u <= F(1)`` is rejected.
    """
    f = sector_cdf(polygon.indices, mu, sigma)
    p = np.zeros(len(polygon.indices))
    p[1:] = np.diff(f)
    return p / p.sum()


def audit_chord_vs_arc(
    polygon: BorderPolygon, ring: np.ndarray, samples: int, pmf: np.ndarray
) -> dict[str, Any]:
    """Blocks A and B: chord versus arc, and the compression of omega."""
    poly = _polyline_lonlat(polygon)
    omegas = np.linspace(0.0, 1.0, samples)

    per_sector: list[dict[str, Any]] = []
    all_chord_lon: list[float] = []
    all_chord_lat: list[float] = []
    all_offset: list[float] = []
    all_sep: list[float] = []

    for idx in polygon.indices:
        seg = polygon.segment(idx)
        _az, _, chord_m = _GEOD.inv(seg.start.lon, seg.start.lat, seg.end.lon, seg.end.lat)
        chord_km = chord_m / 1000.0
        arc_km = seg.arc[-1][2] if seg.arc else chord_km

        lons_c, lats_c, offs, seps = [], [], [], []
        for w in omegas:
            la_a, lo_a = _spawn_coord(seg, float(w), "arc")
            la_c, lo_c = _spawn_coord(seg, float(w), "chord")
            lons_c.append(lo_c)
            lats_c.append(la_c)
            offs.append(distance_to_polyline_km(lo_c, la_c, poly))
            seps.append(float(_GEOD.inv(lo_a, la_a, lo_c, la_c)[2]) / 1000.0)

        inside = points_in_ring(np.asarray(lons_c), np.asarray(lats_c), ring)
        offs_arr = np.asarray(offs)
        strictly_inside = inside & (offs_arr > _INSIDE_TOL_KM)

        per_sector.append({
            "sector": idx,
            "arc_km": round(arc_km, 4),
            "chord_km": round(chord_km, 4),
            "compression_ratio": round(arc_km / chord_km, 4),
            "chord_max_separation_from_arc_km": round(float(np.max(seps)), 4),
            "chord_max_offset_from_border_km": round(float(offs_arr.max()), 4),
            "chord_fraction_inside_brazil": round(float(strictly_inside.mean()), 6),
        })
        all_chord_lon += lons_c
        all_chord_lat += lats_c
        all_offset += offs
        all_sep += seps

    off = np.asarray(all_offset)
    sep = np.asarray(all_sep)
    inside_all = points_in_ring(np.asarray(all_chord_lon), np.asarray(all_chord_lat), ring)
    strictly_in = inside_all & (off > _INSIDE_TOL_KM)
    off_border = off > _INSIDE_TOL_KM

    worst_sep = max(per_sector, key=lambda r: r["chord_max_separation_from_arc_km"])
    worst_comp = max(per_sector, key=lambda r: r["compression_ratio"])
    frac_by_sector = np.asarray([r["chord_fraction_inside_brazil"] for r in per_sector])

    return {
        "sampling": {
            "omega_grid_points_per_sector": samples,
            "sectors": len(polygon),
            "total_points": int(off.size),
            "inside_tolerance_km": _INSIDE_TOL_KM,
            "note": "uniform grid in omega; two weightings reported — uniform over "
                    "sectors (pure geometry) and weighted by the sector selection "
                    "pmf (what the simulator actually draws)",
        },
        "chord": {
            "fraction_inside_brazil": round(float(strictly_in.mean()), 6),
            "fraction_inside_brazil_pmf_weighted": round(float(frac_by_sector @ pmf), 6),
            "fraction_off_border": round(float(off_border.mean()), 6),
            "fraction_outside_brazil_off_border": round(
                float((off_border & ~inside_all).mean()), 6),
            "max_depth_inside_brazil_km": round(
                float(off[strictly_in].max()) if strictly_in.any() else 0.0, 4),
            "mean_offset_from_border_km": round(float(off.mean()), 4),
            "max_offset_from_border_km": round(float(off.max()), 4),
            "max_separation_from_arc_km": round(float(sep.max()), 4),
            "max_separation_sector": worst_sep["sector"],
        },
        "arc": {
            "fraction_inside_brazil": 0.0,
            "max_offset_from_border_km": 0.0,
            "note": "zero by construction: omega walks the stored polyline itself",
        },
        "omega_compression": {
            "max_ratio": worst_comp["compression_ratio"],
            "max_ratio_sector": worst_comp["sector"],
            "note": "arc_km / chord_km — factor by which a uniform omega on the "
                    "chord compresses the induced density along the real border",
        },
        "per_sector": per_sector,
    }


def audit_heading_inwardness(
    polygon: BorderPolygon, ring: np.ndarray, samples: int
) -> dict[str, Any]:
    """Block C: ``literal`` versus ``east_flip`` heading, by inside-territory test."""

    class _ZeroRNG:
        """Deterministic stub: omega set by the caller, zero heading deviation."""

        def __init__(self, omega: float) -> None:
            self._omega = omega

        def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
            return self._omega

        def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
            return 0.0

    omegas = np.linspace(0.0, 1.0, samples + 2)[1:-1]     # excludes the vertices
    results: dict[str, Any] = {}

    # Both interpolations: the rules are comparable only within one of them.
    for interp in ("arc", "chord"):
        col: dict[str, Any] = {}
        for rule in ("literal", "east_flip"):
            per_sector = []
            hits = 0
            total = 0
            for idx in polygon.indices:
                lons, lats = [], []
                for w in omegas:
                    draw = polygon.spawn_in_sector(
                        idx, _ZeroRNG(float(w)), 0.0,
                        interpolation=interp, heading_rule=rule,
                    )
                    # math heading -> compass, then advance the probe distance.
                    compass = (90.0 - math.degrees(draw.heading_rad)) % 360.0
                    lon2, lat2, _ = _GEOD.fwd(
                        draw.point.lon, draw.point.lat, compass,
                        _HEADING_PROBE_KM * 1000.0)
                    lons.append(lon2)
                    lats.append(lat2)
                inside = points_in_ring(np.asarray(lons), np.asarray(lats), ring)
                per_sector.append({
                    "sector": idx,
                    "fraction_inward": round(float(inside.mean()), 6),
                })
                hits += int(inside.sum())
                total += int(inside.size)
            failing = [r for r in per_sector if r["fraction_inward"] < 0.5]
            col[rule] = {
                "fraction_inward": round(hits / total, 6),
                "sectors_pointing_outward": [r["sector"] for r in failing],
                "per_sector": per_sector,
            }
        results[interp] = col

    results["probe"] = {
        "distance_km": _HEADING_PROBE_KM,
        "positions_per_sector": int(omegas.size),
        "heading_deviation_deg": 0.0,
        "note": "perpendicular only (deviation forced to zero) — isolates the rule",
    }
    return results


# ─── Command line ──────────────────────────────────────────────────────────


def build_audit(config_path: Path, sysml_root: Path, samples: int) -> dict[str, Any]:
    cfg = MissionConfig.from_yaml_with_sysml(config_path, sysml_root=sysml_root)
    polygon = default_border_polygon(cfg)
    ring = _load_brazil_ring()
    pmf = sector_pmf(polygon, cfg.sector_mu, cfg.sector_sigma)
    return {
        "field_version": "1.0",
        "spec": "specs/030-uas-generation-miranda-v2/spec.md",
        "config": config_path.name,
        "polygon": {
            "sectors": len(polygon),
            "min_index": polygon.min_index,
            "max_index": polygon.max_index,
            "arc_vertices_stored": sum(len(polygon.segment(i).arc) for i in polygon.indices),
        },
        "territory_reference": json.loads(_BRAZIL.read_text(encoding="utf-8"))["meta"],
        "chord_vs_arc": audit_chord_vs_arc(polygon, ring, samples, pmf),
        "heading_inwardness": audit_heading_inwardness(polygon, ring, 99),
    }


def _print_summary(a: dict[str, Any]) -> None:
    c = a["chord_vs_arc"]["chord"]
    k = a["chord_vs_arc"]["omega_compression"]
    h = a["heading_inwardness"]
    s = a["chord_vs_arc"]["sampling"]
    print("=" * 72)
    print(f"BORDER GEOMETRY AUDIT — {a['polygon']['sectors']} sectors, "
          f"{a['polygon']['arc_vertices_stored']} arc vertices")
    print("=" * 72)
    print("\nA. CHORD vs ARC  (spawn_interpolation)")
    print(f"   points sampled ................. {s['total_points']} "
          f"({s['omega_grid_points_per_sector']} per sector, uniform in omega)")
    print(f"   chord inside Brazil ............ {100 * c['fraction_inside_brazil']:7.2f} %"
          f"  (uniform over sectors)")
    print(f"   chord inside Brazil ............ "
          f"{100 * c['fraction_inside_brazil_pmf_weighted']:7.2f} %  (weighted by sector pmf)")
    print(f"   chord outside Brazil, off line . "
          f"{100 * c['fraction_outside_brazil_off_border']:7.2f} %")
    print(f"   chord off the border line ...... {100 * c['fraction_off_border']:7.2f} %")
    print(f"   max depth inside Brazil ........ {c['max_depth_inside_brazil_km']:7.2f} km")
    print(f"   mean offset from border ........ {c['mean_offset_from_border_km']:7.2f} km")
    print(f"   max separation arc<->chord ..... {c['max_separation_from_arc_km']:7.2f} km"
          f"  (sector {c['max_separation_sector']})")
    print(f"   arc inside Brazil .............. {0.0:7.2f} %  (zero by construction)")
    print("\nB. OMEGA COMPRESSION  (uniform on chord != uniform on border)")
    print(f"   worst ratio .................... {k['max_ratio']:7.2f}x "
          f"(sector {k['max_ratio_sector']})")
    top = sorted(a["chord_vs_arc"]["per_sector"],
                 key=lambda r: -r["compression_ratio"])[:5]
    for r in top:
        print(f"     sector {r['sector']:2d}: chord {r['chord_km']:6.2f} km for "
              f"{r['arc_km']:6.2f} km of arc -> {r['compression_ratio']:.2f}x")
    print(f"\nC. INWARD HEADING  (heading_inward_rule; "
          f"{h['probe']['distance_km']:.0f} km probe, deviation 0)")
    print(f"   {'spawn on':<10} {'literal':>12} {'east_flip':>12}   sectors pointing outward")
    for interp in ("arc", "chord"):
        lit, efl = h[interp]["literal"], h[interp]["east_flip"]
        out = efl["sectors_pointing_outward"] or lit["sectors_pointing_outward"]
        print(f"   {interp:<10} {100 * lit['fraction_inward']:11.2f} % "
              f"{100 * efl['fraction_inward']:11.2f} %   {out if out else 'none'}")
    print("   (compare the two rules within a row, not across rows)")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path,
                   default=_ROOT / "configs" / "zida_shield_default.yaml")
    p.add_argument("--sysml", type=Path, default=_ROOT / "sysml")
    p.add_argument("--samples", type=int, default=201,
                   help="points of the omega grid per sector (default 201)")
    p.add_argument("--out", type=Path,
                   default=_ROOT / "outputs" / "verification" / "border_geometry_audit.json")
    args = p.parse_args(argv)

    audit = build_audit(args.config, args.sysml, args.samples)
    _print_summary(audit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
