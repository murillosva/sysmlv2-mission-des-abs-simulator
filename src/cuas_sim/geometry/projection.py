"""Local equirectangular projection for in-simulation arithmetic.

Per the architectural decisions of the parent paper, *all* in-simulation
kinematics happen on a flat 2D plane in meters. The projection is centered at
the Area of Operations (AOO) centroid (default ``(-5°, -68°)``) and uses the
standard equirectangular formula:

.. math::

    x &= R \\cdot \\cos(\\phi_0) \\cdot (\\lambda - \\lambda_0)\\\\
    y &= R \\cdot (\\phi - \\phi_0)

with :math:`R = 6\\,378\\,137\\,\\mathrm{m}` (WGS-84 equatorial radius) and angles
in radians. Inverse:

.. math::

    \\phi &= \\phi_0 + y / R\\\\
    \\lambda &= \\lambda_0 + x / (R \\cdot \\cos(\\phi_0))

This is intentionally simpler than a proper conformal projection (e.g. UTM /
Lambert) because the AOO is small enough that the residual error is
~0.5 % over the full extent (documented in ``docs/assumptions.md``). The
:math:`T_{\\mathrm{PEA},k}` scalar (nominal value) uses Haversine on
the raw lat/lon instead — see :mod:`cuas_sim.geometry.haversine`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from cuas_sim.geometry.coordinates import GeoPoint

#: WGS-84 equatorial radius in meters. Plain spherical earth is good enough at
#: the ~0.5% projection-error scale we already accept.
EARTH_RADIUS_M: Final[float] = 6_378_137.0


@dataclass(frozen=True, slots=True)
class XY:
    """Cartesian 2D point in projected meters."""

    x: float
    y: float

    def __sub__(self, other: XY) -> XY:
        return XY(self.x - other.x, self.y - other.y)

    def __add__(self, other: XY) -> XY:
        return XY(self.x + other.x, self.y + other.y)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True, slots=True)
class LocalProjection:
    """Local equirectangular projection.

    Parameters
    ----------
    lat0_deg
        Center latitude (decimal degrees).
    lon0_deg
        Center longitude (decimal degrees).
    radius_m
        Sphere radius. Defaults to WGS-84 equatorial radius.
    """

    lat0_deg: float
    lon0_deg: float
    radius_m: float = EARTH_RADIUS_M

    @property
    def lat0_rad(self) -> float:
        return math.radians(self.lat0_deg)

    @property
    def lon0_rad(self) -> float:
        return math.radians(self.lon0_deg)

    @property
    def cos_lat0(self) -> float:
        return math.cos(self.lat0_rad)

    def project(self, p: GeoPoint) -> XY:
        """Project a geographic point to local meters."""
        lat = math.radians(p.lat)
        lon = math.radians(p.lon)
        x = self.radius_m * self.cos_lat0 * (lon - self.lon0_rad)
        y = self.radius_m * (lat - self.lat0_rad)
        return XY(x=x, y=y)

    def unproject(self, xy: XY) -> GeoPoint:
        """Inverse of :meth:`project`.

        Clamps latitude to [-90, 90] and wraps longitude to [-180, 180] so
        that UAS travelling far from the AoO centroid (e.g. at high lambda)
        never produce invalid GeoPoint values.
        """
        lat_deg = math.degrees(self.lat0_rad + xy.y / self.radius_m)
        lon_deg = math.degrees(self.lon0_rad + xy.x / (self.radius_m * self.cos_lat0))
        lat_deg = max(-90.0, min(90.0, lat_deg))
        lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
        return GeoPoint(lat=lat_deg, lon=lon_deg)

    def project_many(self, points: list[GeoPoint]) -> list[XY]:
        """Project a sequence of geographic points."""
        return [self.project(p) for p in points]


def t_to_farp(pos: XY, farp_xy: XY, v_cruise: float) -> float:
    """Time for an A-29 at ``pos`` to reach its designated FARP at cruise speed.

    The shared geometric primitive of the bingo trigger and the geometric
    recovery phases — :math:`t = \\lVert pos - farp\\rVert_{xy} / v_{cruise}`,
    matching ``RecoverToMainBase`` ("transits ... at cruise speed") in the SysML
    ``OperationsPackage``. Pure, O(1): no RNG, no I/O. Returns ``0.0`` when
    ``pos == farp_xy``.
    """
    return (pos - farp_xy).norm() / v_cruise
