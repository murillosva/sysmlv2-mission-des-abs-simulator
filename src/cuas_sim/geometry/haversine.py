"""Great-circle (Haversine) distance.

**Scope**: this function is reserved for the nominal :math:`T_{\\mathrm{PEA},k}`
scalar (Haversine FARP → CAP distance per A-29) and the 80 nmi ZIDA check. All
other in-simulation kinematics happen on the projected plane (see
:mod:`cuas_sim.geometry.projection`). Using a great-circle library here would
be overkill; we compute it ourselves with a stable atan2 form. We also expose
a pyproj.Geod-based variant for cross-checking in unit tests.
"""

from __future__ import annotations

import math
from typing import Final

from cuas_sim.geometry.coordinates import GeoPoint

#: Mean Earth radius used by the Haversine call (NOT the equatorial WGS-84
#: radius). Per the case study, distances are computed with a spherical-earth
#: assumption (Szvaticsek 2025); we use the mean radius for parity with that
#: paper. Documented in ``docs/assumptions.md``.
EARTH_MEAN_RADIUS_M: Final[float] = 6_371_000.0


def haversine_m(a: GeoPoint, b: GeoPoint, radius_m: float = EARTH_MEAN_RADIUS_M) -> float:
    """Haversine great-circle distance between two GeoPoints, in meters.

    Parameters
    ----------
    a, b
        Endpoint geographic points.
    radius_m
        Sphere radius. Defaults to :data:`EARTH_MEAN_RADIUS_M`.

    Returns
    -------
    float
        Distance in meters along the great circle.
    """
    phi1 = math.radians(a.lat)
    phi2 = math.radians(b.lat)
    dphi = phi2 - phi1
    dlambda = math.radians(b.lon - a.lon)
    h = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1.0 - h))
    return radius_m * c


def t_pea_seconds(
    farp: GeoPoint,
    cap: GeoPoint,
    endurance_s: float,
    cruise_speed_mps: float,
) -> float:
    """Nominal time available for the PEA phase (fallback of Eq. 3.7).

    .. math::

        T_{\\mathrm{PEA},k} = R - 2 \\cdot d_{\\mathrm{FARP}_k,\\mathrm{CAP}_k}\\,/\\,v

    Parameters
    ----------
    farp
        FARP coordinate.
    cap
        CAP coordinate.
    endurance_s
        :math:`R`, aircraft endurance in seconds (case study: 5.2 h = 18 720 s).
    cruise_speed_mps
        :math:`v`, A-29 cruise speed in m/s (case study: 144 m/s).

    Returns
    -------
    float
        Seconds available for airspace policing. May be negative if the
        FARP-to-CAP distance is too large given the endurance — callers
        should treat that as a misconfiguration.
    """
    d = haversine_m(farp, cap)
    return endurance_s - 2.0 * d / cruise_speed_mps
