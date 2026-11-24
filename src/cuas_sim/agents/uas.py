"""UAS process — straight-line constant-velocity flier.

A UAS is born at the border, flies a constant heading at constant speed, and
either gets intercepted (terminates) or runs the full PEA window unmolested
(terminates as a "leak"). Position is updated lazily: the UAS exposes
:meth:`xy_at` so the A-29 / engagement code can query the projected location
at any future time, avoiding fine-grained timestep loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cuas_sim.geometry.coordinates import GeoPoint
from cuas_sim.geometry.projection import XY, LocalProjection

if TYPE_CHECKING:
    from cuas_sim.config import UASArchetypeSpec
    from cuas_sim.world.arrivals import UASBirth


@dataclass(slots=True)
class UAS:
    """One UAS instance.

    Constructed once per :class:`~cuas_sim.world.arrivals.UASBirth` per
    configuration (so the paired baseline and modernized runs each have
    their own physical UAS clone; they share kinematics but live independent
    intercept lives).
    """

    uas_id: int
    spawn_time_s: float
    spawn_point: GeoPoint
    spawn_xy: XY
    archetype: UASArchetypeSpec
    heading_rad: float
    sector_index: int
    #: ``True`` once an A-29 has destroyed or aborted the engagement on this UAS.
    is_alive: bool = True
    #: Time at which the UAS was killed (NaN if still alive).
    kill_time_s: float = float("nan")
    #: Time at which the UAS exited the PEA window (NaN if killed).
    leak_time_s: float = float("nan")

    @classmethod
    def from_birth(cls, birth: UASBirth, projection: LocalProjection) -> UAS:
        spawn_xy = projection.project(birth.point)
        return cls(
            uas_id=birth.uas_id,
            spawn_time_s=birth.spawn_time_s,
            spawn_point=birth.point,
            spawn_xy=spawn_xy,
            archetype=birth.archetype,
            heading_rad=birth.heading_rad,
            sector_index=birth.sector_index,
        )

    # ─── Kinematics ──────────────────────────────────────────────────────

    @property
    def speed_mps(self) -> float:
        return self.archetype.max_speed_mps

    def xy_at(self, t_s: float) -> XY:
        """Projected position at simulation time ``t_s``."""
        dt = t_s - self.spawn_time_s
        if dt < 0:
            return self.spawn_xy
        v = self.speed_mps
        dx = v * math.cos(self.heading_rad) * dt
        dy = v * math.sin(self.heading_rad) * dt
        return XY(self.spawn_xy.x + dx, self.spawn_xy.y + dy)

    def point_at(self, t_s: float, projection: LocalProjection) -> GeoPoint:
        """Geographic position at ``t_s``."""
        return projection.unproject(self.xy_at(t_s))

    def mark_killed(self, t_s: float) -> None:
        self.is_alive = False
        self.kill_time_s = t_s

    def mark_leak(self, t_s: float) -> None:
        self.is_alive = False
        self.leak_time_s = t_s
