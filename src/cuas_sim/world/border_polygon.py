"""Border polygon for stochastic UAS incursion generation.

Methodology from Miranda (2025); geometry rebuilt for this case study:

* the **source** generating polygon of Miranda (2025) has 92 sectors of ~100 km
  spanning the whole Brazilian dry border;
* this study rebuilds its own polygon, restricted to the BR-CO + BR-PE border
  clipped to the mission area of operations (1.80°N..12.50°S, 76°W..60°W), which
  yields **27 sectors of 100.000 km of border arc each, indexed 1..27
  north→south**. The numbering is this study's own and does **not** map onto the
  source's sectors.

Three chained draws per incursion (``sample_sector`` + ``spawn_in_sector``):

1. **Sector.** Inverse-CDF transform on ``F(k) = Φ(k; μ, σ)`` with rejection
   outside ``[F(1), F(N))`` — the ±3σ envelope. Since ``μ = (1+N)/2`` and
   ``σ = (N−1)/6`` place ``μ − 3σ = 1``, sector 1 carries **exactly zero mass**
   (a property of the source rule, not a defect). See :func:`sample_sector`.
2. **Origin.** ``ω ~ U(0, 1)`` walks the stored **border arc** of the sector
   (:attr:`BorderSegment.arc`, curvilinear abscissa), so a uniform ω maps to a
   uniform position in *border length*:

   .. math::

       \\omega &\\sim \\mathcal{U}(0, 1)\\\\
       P_t &= \\mathrm{arc}(S,\\ \\omega \\cdot s_{\\mathrm{total}})

   The straight chord ``P_t = P_i + ω·(P_f − P_i)`` is the **legacy** path
   (``interpolation="chord"``) and the silent fallback when a segment carries no
   arc; it displaces the origin off the border line by up to 20.4 km.
3. **Heading.** ``azimuth(P_i → P_f) − 90°`` — the **left** perpendicular of the
   traversal, unconditional. It points inland only under the north→south
   traversal invariant enforced by :meth:`BorderPolygon.__init__`. A Gaussian deviation ``N(0, σ=15°)`` is added, centred on the
   perpendicular and symmetric on both sides: ~68% within ±15° (1σ), ~95% within
   ±30° (2σ), ~99.7% within ±45° (3σ — a **90° total aperture**, i.e. 45° per
   side, not a ±90° spread). The deviation is **not truncated** (no hard cap, and
   unlike the sector draw), so ~0.3% of entries exceed ±45°.

.. note::

   **Assumption marker.** Miranda (2025) publishes the polygon as a figure only,
   not as a numeric table, so the geometry here is a reconstruction of the
   *method*, not a transcription of the source's coordinates. Production runs
   load the exact 27-sector polyline (309 arc vertices) from
   ``configs/zida_shield_default.yaml``. The chain built by
   :func:`default_border_polygon` when no YAML is supplied is a coarse synthetic
   **approximation** for the test path only (see ``docs/assumptions.md`` §4).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import cache
from itertools import pairwise

import numpy as np
from pyproj import Geod

from cuas_sim.config import BorderSegment, MissionConfig
from cuas_sim.geometry.coordinates import GeoPoint

logger = logging.getLogger(__name__)

#: Geodesic engine used for great-circle azimuth computation.
_GEOD: Geod = Geod(ellps="WGS84")


@dataclass(frozen=True, slots=True)
class _SpawnDraw:
    """A single Miranda-style spawn realisation."""

    sector_index: int
    point: GeoPoint
    heading_rad: float
    omega: float


class BorderPolygon:
    """The 27-sector incursion polygon for the case study.

    Wraps a tuple of :class:`BorderSegment` records and adds the stochastic
    spawn draws. Construction enforces the north→south traversal invariant
    and validates each stored arc polyline.
    """

    def __init__(self, segments: tuple[BorderSegment, ...]) -> None:
        if not segments:
            raise ValueError("BorderPolygon needs at least one segment")
        idx_to_seg: dict[int, BorderSegment] = {}
        for s in segments:
            if s.index in idx_to_seg:
                raise ValueError(f"Duplicate sector index {s.index}")
            idx_to_seg[s.index] = s
        self._idx_to_seg = idx_to_seg
        self.indices: tuple[int, ...] = tuple(sorted(idx_to_seg))
        self.min_index: int = self.indices[0]
        self.max_index: int = self.indices[-1]

        # Traversal must be net north→south (start of the
        # first sector strictly north of the end of the last). The az−90°
        # inward perpendicular is well-defined only under this
        # orientation, since the Brazilian interior is then always to the left.
        first, last = idx_to_seg[self.min_index], idx_to_seg[self.max_index]
        if not first.start.lat > last.end.lat:
            raise ValueError(
                "BorderPolygon must be oriented north->south: sector "
                f"{self.min_index} start lat {first.start.lat} is not north of "
                f"sector {self.max_index} end lat {last.end.lat}"
            )

        # Validate each stored arc polyline.
        for s in segments:
            if not s.arc:
                continue
            if len(s.arc) < 2:
                raise ValueError(f"Sector {s.index}: arc needs >= 2 vertices")
            s_km = [v[2] for v in s.arc]
            if s_km[0] != 0.0:
                raise ValueError(f"Sector {s.index}: arc s_km[0] must be 0, got {s_km[0]}")
            if any(b <= a for a, b in pairwise(s_km)):
                raise ValueError(f"Sector {s.index}: arc s_km must be strictly increasing")
            for vtx, ref, label in ((s.arc[0], s.start, "start"), (s.arc[-1], s.end, "end")):
                _, _, dist = _GEOD.inv(vtx[1], vtx[0], ref.lon, ref.lat)
                if dist > 1.0:
                    raise ValueError(
                        f"Sector {s.index}: arc {label} vertex is {dist:.2f} m "
                        f"from segment {label} (>1 m)"
                    )

    def __len__(self) -> int:
        return len(self._idx_to_seg)

    def __contains__(self, idx: int) -> bool:
        return idx in self._idx_to_seg

    def segment(self, idx: int) -> BorderSegment:
        try:
            return self._idx_to_seg[idx]
        except KeyError as exc:
            raise KeyError(f"Sector {idx} not in polygon") from exc

    def spawn_in_sector(
        self,
        sector_idx: int,
        rng_geom: np.random.Generator,
        heading_dev_std_deg: float,
        *,
        interpolation: str = "arc",
        heading_rule: str = "literal",
    ) -> _SpawnDraw:
        """Draw a spawn point and inward heading on the given sector.

        Parameters
        ----------
        sector_idx
            Sector index (must be present in the polygon).
        rng_geom
            World-shared geometry generator (CRN).
        heading_dev_std_deg
            σ of the Gaussian heading deviation around the perpendicular
            (the case study fixes this at 15°, Section 3.3.3.3).
        interpolation
            Curve ω runs along: ``"arc"`` interpolates on
            the stored border polyline (falls back to chord when the segment has
            no arc); ``"chord"`` interpolates on the straight vertex-to-vertex
            chord. ω itself is a single ``uniform`` draw either way.
        heading_rule
            Inward-perpendicular selection: ``"literal"``
            = ``az − 90°`` unconditional (Miranda 2025, correct under the N→S
            orientation); ``"east_flip"`` = legacy east-component
            heuristic. Draw order (uniform ω, then normal deviation) is
            identical for both, so the stream consumption is unchanged.
        """
        seg = self.segment(sector_idx)
        omega = float(rng_geom.uniform(0.0, 1.0))

        lat, lon = _spawn_coord(seg, omega, interpolation)
        point = GeoPoint(lat=lat, lon=lon, name=f"spawn-{sector_idx}")

        # Inward perpendicular heading.
        # pyproj.Geod azimuth is from north, clockwise. We convert to the
        # atan2 (math) convention (from +x = East, counter-clockwise) so that
        # the simulator's plane geometry uses a single angle convention.
        az_fwd, _, _ = _GEOD.inv(seg.start.lon, seg.start.lat, seg.end.lon, seg.end.lat)
        perp_primary = az_fwd - 90.0        # Miranda (2025): left perpendicular
        if heading_rule == "literal":
            inward_compass = perp_primary
        elif heading_rule == "east_flip":
            # Legacy east-component heuristic (points outward on E–W sectors,
            # e.g. 24 and 27 — kept only for the legacy configuration).
            inward_compass = (
                perp_primary if math.sin(math.radians(perp_primary)) > 0
                else az_fwd + 90.0
            )
        else:
            raise ValueError(
                f"heading_rule must be 'literal' or 'east_flip', got {heading_rule!r}"
            )
        deviation = float(rng_geom.normal(loc=0.0, scale=heading_dev_std_deg))
        inward_compass += deviation
        # Convert compass → math: math_h = 90 − compass_h, in degrees.
        math_heading_rad = math.radians(90.0 - inward_compass)
        return _SpawnDraw(sector_index=sector_idx, point=point,
                          heading_rad=math_heading_rad, omega=omega)


def _spawn_coord(seg: BorderSegment, omega: float, interpolation: str) -> tuple[float, float]:
    """Return ``(lat, lon)`` of the spawn point for coefficient ``omega``.

    ``"arc"`` walks ``omega × s_total`` km along the stored
    border polyline (binary search on the curvilinear abscissa + linear interp
    on the ``range/3``-length subsegment); it falls back to the chord when the
    segment carries no arc. ``"chord"`` is the straight vertex-to-vertex
    blend (legacy). Pure — no RNG, no I/O.
    """
    if interpolation not in ("arc", "chord"):
        raise ValueError(f"interpolation must be 'arc' or 'chord', got {interpolation!r}")
    if interpolation == "arc" and seg.arc:
        arc = seg.arc
        target = omega * arc[-1][2]           # km from the sector start
        lo, hi = 0, len(arc) - 1
        while hi - lo > 1:                     # binary search on the abscissa
            mid = (lo + hi) // 2
            if arc[mid][2] <= target:
                lo = mid
            else:
                hi = mid
        span = arc[lo + 1][2] - arc[lo][2]
        f = (target - arc[lo][2]) / span if span > 1e-12 else 0.0
        lat = arc[lo][0] + f * (arc[lo + 1][0] - arc[lo][0])
        lon = arc[lo][1] + f * (arc[lo + 1][1] - arc[lo][1])
        return lat, lon
    # Chord fallback (interpolation == "chord", or arc empty).
    lat = seg.start.lat + omega * (seg.end.lat - seg.start.lat)
    lon = seg.start.lon + omega * (seg.end.lon - seg.start.lon)
    return lat, lon


# ─── Default approximate polygon ───────────────────────────────────────────


def default_border_polygon(cfg: MissionConfig) -> BorderPolygon:
    """Build a geographically reasonable default polygon (sectors 1..27).

    Generates a chain of equal-length geodesic segments along a coarse
    approximation of the Colombia/Peru/Brazil dry border, ordered
    **north→south**: sector 1 near the northern
    (Colombia/Brazil) end and sector 27 near the southern (Peru/Brazil) end.
    **This is an approximation, not to scale** — the segments are ~50 km, half
    the 100-km border arc of the real sectors, and carry no ``arc`` polyline.
    Production runs load the exact 27-sector geometry from
    ``configs/zida_shield_default.yaml``.
    """
    if cfg.border_sectors is not None:
        return BorderPolygon(cfg.border_sectors)

    logger.warning(
        "Using DEFAULT approximated border polygon. "
        "Load configs/zida_shield_default.yaml for paper-faithful runs."
    )

    # Anchor points (decimal degrees), ordered north→south.
    start_lat, start_lon = 1.20, -67.50      # north (near sector 1)
    end_lat, end_lon = -9.65, -73.0          # south (near sector 27)

    n_segments = cfg.sector_max - cfg.sector_min + 1  # 27 sectors

    # Distribute the n_segments uniformly along the geodesic from start→end,
    # re-projecting each step. The Geod.fwd call uses the actual azimuth at each
    # point so the chain follows the great circle.
    az_initial, _, total_distance_m = _GEOD.inv(start_lon, start_lat, end_lon, end_lat)

    # Per-segment length = total / n_segments, which preserves the chain
    # endpoints. The straight anchor-to-anchor geodesic is ~1 346 km, so these
    # are ~50-km segments — HALF the 100-km border arc of the real sectors. That
    # is the sense in which this fallback is an approximation (docs/assumptions
    # §4): it is topologically valid (N→S) but not to scale, and carries
    # no `arc`, so ω falls back to the chord.
    seg_len_m = total_distance_m / n_segments

    segments = []
    cur_lat, cur_lon = start_lat, start_lon
    cur_az = az_initial
    for i in range(n_segments):
        nxt_lon, nxt_lat, _back_az = _GEOD.fwd(cur_lon, cur_lat, cur_az, seg_len_m)
        idx = cfg.sector_min + i
        segments.append(BorderSegment(
            index=idx,
            start=GeoPoint(lat=cur_lat, lon=cur_lon, name=f"P{idx-1}"),
            end=GeoPoint(lat=nxt_lat, lon=nxt_lon, name=f"P{idx}"),
        ))
        cur_lat, cur_lon = nxt_lat, nxt_lon
        # New forward azimuth: az from current to end.
        cur_az, _, _ = _GEOD.inv(cur_lon, cur_lat, end_lon, end_lat)
    return BorderPolygon(tuple(segments))


# ─── Sector selection ──────────────────────────────────────────────────────

#: Safety cap on the ±3σ resampling loop.
_MAX_RESORTEIOS: int = 10_000


@cache
def sector_cdf(indices: tuple[int, ...], mu: float, sigma: float) -> np.ndarray:
    """Cumulative selection probability ``F(k) = Φ(k; μ, σ)`` per sector.

    Evaluated at the integer index of each sector, with the
    exact normal CDF ``0.5·(1 + erf((k−μ)/(σ√2)))`` (no scipy). Pure and
    ``lru_cache``d per ``(indices, μ, σ)`` so the column is built once per
    polygon, not per arrival. Callers must not mutate the array.
    """
    root2 = math.sqrt(2.0)
    return np.array(
        [0.5 * (1.0 + math.erf((k - mu) / (sigma * root2))) for k in indices],
        dtype=float,
    )


def truncated_normal_sector_pmf(
    indices: tuple[int, ...], mu: float, sigma: float
) -> np.ndarray:
    """Truncated-normal PMF over a discrete set of sector indices (legacy).

    Computes ``φ((k - μ)/σ)`` at each ``k ∈ indices``, then normalises — the
    legacy ``rule="pdf"`` path. Preserved for the legacy
    configuration and as the oracle for the CDF rule.
    """
    arr = np.asarray(indices, dtype=float)
    # Standardise then evaluate φ (pdf of standard normal). We don't worry
    # about the normalisation constant since we renormalise immediately.
    z = (arr - mu) / sigma
    weights = np.exp(-0.5 * z * z)
    weights /= weights.sum()
    return weights


def sample_sector(
    polygon: BorderPolygon,
    rng_sector: np.random.Generator,
    mu: float,
    sigma: float,
    *,
    rule: str = "cdf",
) -> int:
    """Draw a sector index.

    ``rule="cdf"`` (Miranda 2025): draw ``u ~ U(0,1)``; if ``u`` falls outside
    ``[F(1), F(N))`` (the ±3σ envelope) resample; otherwise pick the sector of
    smallest ``F(k)`` that exceeds ``u`` via binary search. Since ``μ−3σ = 1``,
    every accepted ``u`` selects ``k ≥ 2`` (sector 1 has zero mass). ``rule="pdf"``
    is the legacy renormalised-normal draw.
    """
    if rule == "cdf":
        f = sector_cdf(polygon.indices, mu, sigma)
        lo, hi = float(f[0]), float(f[-1])
        for _ in range(_MAX_RESORTEIOS):
            u = float(rng_sector.uniform(0.0, 1.0))
            if u < lo or u >= hi:
                continue                      # outside ±3σ: resample
            pos = int(np.searchsorted(f, u, side="right"))
            return int(polygon.indices[pos])
        raise RuntimeError("sector resampling did not converge; check mu and sigma")
    if rule == "pdf":
        pmf = truncated_normal_sector_pmf(polygon.indices, mu, sigma)
        return int(rng_sector.choice(polygon.indices, p=pmf))
    raise ValueError(f"rule must be 'cdf' or 'pdf', got {rule!r}")
