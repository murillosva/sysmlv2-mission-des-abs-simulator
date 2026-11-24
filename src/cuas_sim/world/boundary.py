"""ZIDA boundary check for MOS-1.

A UAS is *neutralized before the ZIDA* (MOS-1 success) if the great-circle
distance from its position at the kill time to the spawn point of the
incursion is **at most** 80 nmi (Eq. 3.2). Otherwise it is a
**leak** (MOS-1 failure) and the engagement, if any, occurred too late.

The check uses Haversine on the raw lat/lon for consistency with Szvaticsek
(2025); the in-simulation kinematics on the projected plane only feed the
endpoint coordinates back via :meth:`LocalProjection.unproject`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cuas_sim.geometry.coordinates import GeoPoint
from cuas_sim.geometry.haversine import haversine_m
from cuas_sim.geometry.projection import XY

if TYPE_CHECKING:
    from cuas_sim.geometry.projection import LocalProjection

#: 1 nautical mile in meters (canonical).
NMI_TO_M: float = 1_852.0


@dataclass(frozen=True, slots=True)
class ZIDABoundary:
    """80 nautical-mile depth boundary measured from the entry point.

    Per the case study, the ZIDA is a depth defined from the entry point, not
    a fixed polygon — each incursion has its own ZIDA centered on its spawn.
    """

    radius_nmi: float = 80.0

    @property
    def radius_m(self) -> float:
        return self.radius_nmi * NMI_TO_M

    def is_inside(self, spawn_point: GeoPoint, current_point: GeoPoint) -> bool:
        """Return True iff ``current_point`` is within the ZIDA radius of spawn."""
        return haversine_m(spawn_point, current_point) <= self.radius_m

    def penetration_m(self, spawn_point: GeoPoint, current_point: GeoPoint) -> float:
        """Distance traveled from spawn (meters; positive)."""
        return haversine_m(spawn_point, current_point)

    def crossing_time_s(
        self,
        *,
        spawn_point: GeoPoint,
        spawn_xy: XY,
        heading_rad: float,
        speed_mps: float,
        spawn_time_s: float,
        projection: LocalProjection,
        tol_m: float = 1e-3,
        max_iter: int = 100,
    ) -> float:
        """Projected mission time the UAS crosses the ZIDA depth.

        The UAS flies a constant heading at constant ``speed_mps``
        (:math:`xy(t) = xy_0 + v\\cdot(\\cos h, \\sin h)\\cdot(t - t_0)`); the
        Haversine penetration from ``spawn_point`` grows monotonically, so a
        bisection on ``t`` converges to the instant ``penetration_m == radius_m``.
        This mirrors the same ``ZIDABoundary`` (Haversine) used by the MOS-1 ZIDA
        scoring, **not** the projection-plane Euclidean distance.
        Pure: no RNG, no I/O. Returns ``inf`` for a non-moving UAS (never crosses).
        """
        if speed_mps <= 0.0:
            return float("inf")
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        def _pen_at(t: float) -> float:
            dt = t - spawn_time_s
            cur = XY(spawn_xy.x + speed_mps * cos_h * dt,
                     spawn_xy.y + speed_mps * sin_h * dt)
            return self.penetration_m(spawn_point, projection.unproject(cur))

        lo = spawn_time_s
        hi = spawn_time_s + 2.0 * self.radius_m / speed_mps
        # Expand the upper bracket until the penetration provably exceeds the
        # radius (robust to the ~0.5% projection-vs-Haversine discrepancy).
        for _ in range(max_iter):
            if _pen_at(hi) >= self.radius_m:
                break
            hi += self.radius_m / speed_mps
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            pen = _pen_at(mid)
            if abs(pen - self.radius_m) <= tol_m:
                return mid
            if pen < self.radius_m:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
