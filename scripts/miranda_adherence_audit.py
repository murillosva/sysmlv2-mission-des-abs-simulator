"""Audit of the incursion generator against the method of Miranda (2025).

Runs a Monte Carlo batch on the simulator's own code (``sample_sector`` and
``BorderPolygon.spawn_in_sector``, built from the scenario file) and writes
``outputs/verification/miranda_adherence_audit.json`` with the measured values, the seeds
and the sample size. It is the single source of these numbers: the statistical test
``tests/statistical/test_miranda_v2_adherence.py`` calls the same function and only
asserts tolerance bands, so the test and the artifact cannot diverge.

Five quantities, all with the same sample size:

* occurrences of sector 1, which must be exactly 0 (``mu - 3*sigma = 1``: every
  ``u <= F(1)`` is rejected, so every accepted ``u`` selects ``k >= 2``);
* redraw rate, the mass outside the +-3 sigma envelope;
* maximum deviation between the empirical and the theoretical sector pmf;
* empirical spread of the heading and fraction within +-45 degrees (the Gaussian
  deviation is not truncated, unlike the sector draw);
* split between the two threat classes.

It also writes the per-sector table (theoretical F(k) and p(k), empirical frequency and
end coordinates of each segment), so that the table can be generated from the simulator
instead of transcribed.

Deterministic: explicit seeds, no timestamp, sorted keys; two runs write byte-identical
JSON. It runs no simulation and touches no measure.

Usage::

    python scripts/miranda_adherence_audit.py
    python scripts/miranda_adherence_audit.py --n-mc 20000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))

from cuas_sim.config import MissionConfig  # noqa: E402
from cuas_sim.world.border_polygon import (  # noqa: E402
    BorderPolygon,
    default_border_polygon,
    sample_sector,
    sector_cdf,
)

#: Seeds of the batch. Fixed: changing them invalidates the published artifact.
SEED_SECTOR = 2025
SEED_GEOMETRY = 4242
SEED_ARCHETYPE = 7

#: Size of the published batch.
N_MC = 200_000


class _CountingRNG:
    """Wrap a ``Generator`` and count the ``uniform`` draws (measures the redraw rate)."""

    def __init__(self, gen: np.random.Generator) -> None:
        self._g = gen
        self.n_uniform = 0

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        self.n_uniform += 1
        return float(self._g.uniform(low, high))

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return float(self._g.normal(loc, scale))


class _PlainRNG:
    """Scalar-float adapter over a ``Generator`` (no counting)."""

    def __init__(self, gen: np.random.Generator) -> None:
        self._g = gen

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self._g.uniform(low, high))

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return float(self._g.normal(loc, scale))


def _pt_seg_dist_deg(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Point-to-segment distance in the (lon, lat) plane, in degrees."""
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _perpendiculars(polygon: BorderPolygon) -> dict[int, float]:
    """Perpendicular heading of each sector, with the deviation forced to zero."""
    out: dict[int, float] = {}
    for idx in polygon.indices:
        draw = polygon.spawn_in_sector(
            idx, _PlainRNG(np.random.default_rng(0)), 0.0, heading_rule="literal")
        out[idx] = draw.heading_rad
    return out


def build_audit(
    config_path: Path,
    sysml_root: Path,
    n_mc: int = N_MC,
) -> dict[str, Any]:
    """Run the batch and return the artifact dictionary. Deterministic."""
    cfg = MissionConfig.from_yaml_with_sysml(config_path, sysml_root=sysml_root)
    polygon = default_border_polygon(cfg)
    n_sectors = len(polygon)

    sec_rng = _CountingRNG(np.random.default_rng(SEED_SECTOR))
    geo_rng = _PlainRNG(np.random.default_rng(SEED_GEOMETRY))
    perp = _perpendiculars(polygon)

    counts = np.zeros(n_sectors + 1, dtype=np.int64)     # indexed by sector 1..N
    deviations = np.empty(n_mc)
    max_off_deg = 0.0

    for i in range(n_mc):
        k = sample_sector(polygon, sec_rng, mu=cfg.sector_mu, sigma=cfg.sector_sigma,
                          rule=cfg.sector_draw_rule)
        counts[k] += 1
        draw = polygon.spawn_in_sector(
            k, geo_rng, cfg.heading_perpendicular_deviation_deg,
            interpolation=cfg.spawn_interpolation, heading_rule=cfg.heading_inward_rule)
        dev = (draw.heading_rad - perp[k] + math.pi) % (2 * math.pi) - math.pi
        deviations[i] = math.degrees(dev)
        arc = polygon.segment(k).arc
        if arc:
            off = min(_pt_seg_dist_deg(draw.point.lon, draw.point.lat,
                                       a[1], a[0], b[1], b[0])
                      for a, b in pairwise(arc))
            max_off_deg = max(max_off_deg, off)

    # ── theoretical reference ─────────────────────────────────────────────
    f = sector_cdf(polygon.indices, cfg.sector_mu, cfg.sector_sigma)
    p_theo = np.zeros(n_sectors)
    p_theo[1:] = np.diff(f)
    p_theo /= p_theo.sum()
    p_emp = counts[1:] / counts[1:].sum()
    rejected_mass = float(f[0] + (1.0 - f[-1]))

    resort_rate = (sec_rng.n_uniform - n_mc) / sec_rng.n_uniform
    arch_draws = np.random.default_rng(SEED_ARCHETYPE).random(n_mc) < cfg.uas_archetype_p_mr

    per_sector = []
    for i, idx in enumerate(polygon.indices):
        seg = polygon.segment(idx)
        per_sector.append({
            "sector": idx,
            "cdf": round(float(f[i]), 9),
            "p_theoretical": round(float(p_theo[i]), 9),
            "p_empirical": round(float(p_emp[i]), 9),
            "count": int(counts[idx]),
            "lat_final": seg.end.lat,
            "lon_final": seg.end.lon,
        })

    return {
        "field_version": "1.0",
        "spec": "specs/030-uas-generation-miranda-v2/spec.md §5.3",
        "config": config_path.name,
        "source": "cuas_sim.world.border_polygon (simulator code, not a reimplementation)",
        "parameters": {
            "n_sectors": n_sectors,
            "mu": cfg.sector_mu,
            "sigma": cfg.sector_sigma,
            "sigma_formula": "(N-1)/6",
            "heading_deviation_sigma_deg": cfg.heading_perpendicular_deviation_deg,
            "heading_deviation_truncated": False,
            "p_multirotor": cfg.uas_archetype_p_mr,
            "sector_draw_rule": cfg.sector_draw_rule,
            "spawn_interpolation": cfg.spawn_interpolation,
            "heading_inward_rule": cfg.heading_inward_rule,
        },
        "sampling": {
            "n_mc": n_mc,
            "seed_sector": SEED_SECTOR,
            "seed_geometry": SEED_GEOMETRY,
            "seed_archetype": SEED_ARCHETYPE,
            "note": "the same n_mc for every quantity — sector, heading and threat class",
        },
        "theoretical": {
            "cdf_first": round(float(f[0]), 9),
            "cdf_last": round(float(f[-1]), 9),
            "cdf_median_sector": round(float(f[n_sectors // 2]), 9),
            "rejected_mass": round(rejected_mass, 9),
            "pmf_mean": round(float(np.sum(np.arange(1, n_sectors + 1) * p_theo)), 6),
            "p_sector_1": float(p_theo[0]),
            "heading_within_45deg": 0.9973,
        },
        "empirical": {
            "sector_1_count": int(counts[1]),
            "resort_rate": round(resort_rate, 9),
            "pmf_max_deviation": round(float(np.max(np.abs(p_emp - p_theo))), 9),
            "heading_sigma_deg": round(float(deviations.std()), 6),
            "heading_within_45deg": round(float(np.mean(np.abs(deviations) <= 45.0)), 6),
            "fraction_multirotor": round(float(arch_draws.mean()), 6),
            "max_offset_from_polyline_deg": float(f"{max_off_deg:.3e}"),
        },
        "per_sector": per_sector,
    }


def _print_summary(a: dict[str, Any]) -> None:
    e, t, s = a["empirical"], a["theoretical"], a["sampling"]
    print("=" * 72)
    print(f"MIRANDA (2025) ADHERENCE AUDIT — {a['parameters']['n_sectors']} sectors, "
          f"N = {s['n_mc']}")
    print("=" * 72)
    print(f"   seeds .......................... sector {s['seed_sector']}, "
          f"geometry {s['seed_geometry']}, archetype {s['seed_archetype']}")
    print(f"   mu / sigma ..................... {a['parameters']['mu']} / "
          f"{a['parameters']['sigma']:.6f}  = (N-1)/6")
    print("\n   quantity                          measured      reference")
    print(f"   sector-1 occurrences .......... {e['sector_1_count']:12d}   "
          f"{0:12d}  (exact)")
    print(f"   resort rate ................... {100 * e['resort_rate']:11.3f} %  "
          f"{100 * t['rejected_mass']:11.3f} %")
    print(f"   pmf max deviation ............. {e['pmf_max_deviation']:12.5f}   "
          f"{'< 0.002':>12}")
    print(f"   heading sigma ................. {e['heading_sigma_deg']:11.3f} deg"
          f"{a['parameters']['heading_deviation_sigma_deg']:12.1f} deg")
    print(f"   heading within +-45 deg ....... {100 * e['heading_within_45deg']:11.3f} %  "
          f"{100 * t['heading_within_45deg']:11.2f} %")
    print(f"   multi-rotor share ............. {100 * e['fraction_multirotor']:11.2f} %  "
          f"{100 * a['parameters']['p_multirotor']:11.2f} %")
    print(f"   max offset from polyline ...... {e['max_offset_from_polyline_deg']:12.2e}   "
          f"{'< 1e-8':>12} deg")
    print(f"\n   theoretical pmf mean .......... {t['pmf_mean']:12.4f}   "
          f"(mu = {a['parameters']['mu']}; half-sector offset is inherent to the "
          f"(k-1, k] cell convention)")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path,
                   default=_ROOT / "configs" / "zida_shield_default.yaml")
    p.add_argument("--sysml", type=Path, default=_ROOT / "sysml")
    p.add_argument("--n-mc", type=int, default=N_MC)
    p.add_argument("--out", type=Path,
                   default=_ROOT / "outputs" / "verification" / "miranda_adherence_audit.json")
    args = p.parse_args(argv)

    audit = build_audit(args.config, args.sysml, args.n_mc)
    _print_summary(audit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
