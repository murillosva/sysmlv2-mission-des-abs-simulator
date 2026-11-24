"""World submodule: border polygon, AoR mapping, Poisson UAS spawner.

Implements the stochastic incursion process of Miranda (2025) rebuilt for the
27-sector BR-CO/BR-PE strip of this case study, the AoR ↔ sector
mapping, and the 80 nmi ZIDA boundary.
"""

from cuas_sim.world.aor import AoRMap
from cuas_sim.world.arrivals import UASSpawner
from cuas_sim.world.border_polygon import BorderPolygon, default_border_polygon
from cuas_sim.world.boundary import ZIDABoundary

__all__ = [
    "AoRMap",
    "BorderPolygon",
    "UASSpawner",
    "ZIDABoundary",
    "default_border_polygon",
]
