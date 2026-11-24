"""E-99M AEW&C dispatcher — nearest free A-29 among all six.

Per the case study (Section 3.3.3.1), the E-99M detects every incursion when it
crosses the border and assigns it to the **nearest free A-29** of the
configuration, whatever the area of interest of the incursion.

Routing rules (in order):
1. Candidates: every A-29 of the configuration that is not pursuing or
   engaging (state ∉ {PURSUING, ENGAGING}); an A-29 returning to its CAP point
   is a candidate and receives the UAS in its own queue.
2. Among the candidates, pick the nearest by CAP-to-UAS-spawn Euclidean
   distance, with a deterministic numeric-id tie-break.
3. If every A-29 is busy, the UAS is parked in the :class:`PendingPool` and
   claimed by the first A-29 that becomes free; it is recorded as evaded if it
   crosses the 80 nmi limit before that.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cuas_sim.agents.uas import UAS
from cuas_sim.geometry.intercept import collision_course_intercept
from cuas_sim.world.boundary import ZIDABoundary

if TYPE_CHECKING:
    import simpy

    from cuas_sim.agents.a29 import A29
    from cuas_sim.geometry.projection import XY, LocalProjection
    from cuas_sim.metrics.records import RunLogger
    from cuas_sim.world.aor import AoRMap
    from cuas_sim.world.arrivals import UASBirth

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingItem:
    """One parked UAS birth awaiting a free A-29."""

    birth: UASBirth
    t_added_s: float


@dataclass(slots=True)
class PendingPool:
    """UAS that arrived while every A-29 was busy.

    Births the dispatcher cannot allocate immediately are parked here instead
    of being dropped. A freed A-29 consults the pool (see
    :meth:`cuas_sim.agents.a29.A29._drain_pending_pool`): it prunes any
    pending UAS that has crossed its 80 nmi ZIDA depth without ever being
    engaged (leak ``reason="pending-evaded"``) and claims the nearest
    still-inside, intercept-feasible one. The claim/prune API is defined here
    so the pool lives inside the dispatcher module.
    """

    config_name: str
    projection: LocalProjection
    run_logger: RunLogger
    zida_nmi: float = 80.0
    _items: list[_PendingItem] = field(default_factory=list)

    def add(self, birth: UASBirth, *, t_added_s: float) -> None:
        self._items.append(_PendingItem(birth=birth, t_added_s=t_added_s))

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)

    def claim(
        self,
        *,
        cap_xy: XY,
        now_s: float,
        a29_speed_mps: float,
        envelope_max_m: float,
    ) -> UASBirth | None:
        """Prune evaded pendings, then return the nearest claimable one.

        Selection metric: Euclidean CAP→current-UAS-position distance with a
        deterministic ``uas_id`` tie-break. A pending UAS is
        *claimable* iff it is still within 80 nmi of its spawn (ZIDA depth,
        Haversine) **and** the intercept from the CAP is feasible. A pending
        UAS that has crossed 80 nmi without ever being engaged is a definitive
        leak (``reason="pending-evaded"``) and is removed from the pool.
        """
        zida = ZIDABoundary(radius_nmi=self.zida_nmi)
        survivors: list[_PendingItem] = []
        for item in self._items:
            uas = UAS.from_birth(item.birth, self.projection)
            cur_pt = self.projection.unproject(uas.xy_at(now_s))
            if zida.is_inside(item.birth.point, cur_pt):
                survivors.append(item)
            else:
                self.run_logger.log_uas_leak(
                    self.config_name, item.birth, reason="pending-evaded"
                )
        self._items = survivors

        best: _PendingItem | None = None
        best_key: tuple[float, int] = (math.inf, 0)
        for item in self._items:
            uas = UAS.from_birth(item.birth, self.projection)
            cur_xy = uas.xy_at(now_s)
            geom = collision_course_intercept(
                a29_xy=cap_xy,
                a29_speed_mps=a29_speed_mps,
                uas_xy=cur_xy,
                uas_heading_rad=uas.heading_rad,
                uas_speed_mps=uas.speed_mps,
                envelope_max_m=envelope_max_m,
            )
            if not geom.feasible:
                continue
            key = (math.hypot(cur_xy.x - cap_xy.x, cur_xy.y - cap_xy.y),
                   item.birth.uas_id)
            if key < best_key:
                best_key = key
                best = item
        if best is None:
            return None
        self._items.remove(best)
        return best.birth


@dataclass(slots=True)
class E99MDispatcher:
    """One dispatcher per configuration; reads a config-specific store."""

    config_name: str
    aor_map: AoRMap
    a29_agents: dict[str, A29]
    cap_xy_map: dict[str, tuple[float, float]]   # A-29 id → CAP (x_m, y_m)
    projection: LocalProjection
    #: Shared pool for births that arrive while every A-29 is
    #: busy. ``None`` falls back to the legacy drop behaviour.
    pending_pool: PendingPool | None = None

    def run(
        self,
        env: simpy.Environment,
        inbox: simpy.Store,
        outboxes: dict[str, simpy.Store],
    ) -> Generator[Any, Any, None]:
        """Pull births and forward to the globally nearest free A-29.

        Candidates are **all** A-29s of this
        configuration (no primary/adjacent restriction); the nearest free one
        by CAP→UAS-spawn distance is chosen, with a deterministic numeric-id
        tie-break. If every A-29 is busy the birth is parked in
        :attr:`pending_pool` (claimed later by the first A-29 that frees up)
        instead of being dropped.
        """
        while True:
            birth: UASBirth = yield inbox.get()

            _uas = self.projection.project(birth.point)
            uas_x, uas_y = _uas.x, _uas.y
            free = [aid for aid in self.a29_agents if self.a29_agents[aid].is_free]

            if not free:
                if self.pending_pool is not None:
                    self.pending_pool.add(birth, t_added_s=float(env.now))
                    logger.debug(
                        "[%s] All A-29 busy for UAS-%d — parked in pending pool",
                        self.config_name, birth.uas_id,
                    )
                else:
                    logger.info(
                        "[%s] All A-29 busy for UAS-%d — no pool, dropping",
                        self.config_name, birth.uas_id,
                    )
                continue

            # Nearest by CAP→spawn; tie-break by numeric A-29 id.
            chosen = min(
                free,
                key=lambda aid: (
                    math.hypot(
                        self.cap_xy_map[aid][0] - uas_x,
                        self.cap_xy_map[aid][1] - uas_y,
                    ),
                    _numeric_id(aid),
                ),
            )

            if chosen not in outboxes:
                logger.warning(
                    "[%s] Outbox for %s missing — dropping UAS-%d",
                    self.config_name, chosen, birth.uas_id,
                )
                continue

            yield outboxes[chosen].put(birth)


def _numeric_id(a29_id: str) -> int:
    """Numeric suffix of an ``A-29-<n>`` id (deterministic tie-break key)."""
    try:
        return int(a29_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0
