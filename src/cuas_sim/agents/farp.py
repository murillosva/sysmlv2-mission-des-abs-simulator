"""FARP — Forward Arming and Refueling Point.

In the simulator the FARP is a passive geographic record (refueling time is
folded into the pre-PEA durations of Table 3.2). It is kept as a separate type so
that future iterations of the OPT loop can introduce FARP-side capacity
constraints (e.g. ground crew servicing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cuas_sim.geometry.coordinates import GeoPoint


@dataclass(slots=True)
class FARP:
    """Forward Arming and Refueling Point."""

    name: str
    location: GeoPoint
    #: A-29s nominally based here (per Table 3.2).
    a29s: list[str] = field(default_factory=list)
