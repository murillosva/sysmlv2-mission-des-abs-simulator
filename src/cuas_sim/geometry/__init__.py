"""Geometry submodule.

Provides the local equirectangular projection used for in-simulation arithmetic,
DMS coordinate parsing, a Haversine wrapper restricted to the nominal
T_PEA,k scalar and the 80 nmi ZIDA check, and a closed-form collision-course intercept solver.
"""

from cuas_sim.geometry.coordinates import GeoPoint, parse_dms
from cuas_sim.geometry.haversine import haversine_m
from cuas_sim.geometry.intercept import InterceptGeometry, collision_course_intercept
from cuas_sim.geometry.projection import XY, LocalProjection

__all__ = [
    "XY",
    "GeoPoint",
    "InterceptGeometry",
    "LocalProjection",
    "collision_course_intercept",
    "haversine_m",
    "parse_dms",
]
